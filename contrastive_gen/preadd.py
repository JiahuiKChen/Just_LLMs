import os
os.environ['HF_HOME'] = '/datastor1/jiahuikchen/hf_cache'

### Re-implementation of https://github.com/jonnypei/acl23-preadd/blob/main/methods/method_preadd.py
### in a standalone file
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict


class ContrastiveGen:
    def __init__(self, model_string):
        """
        Initializes the ContrastiveGen class with a specified model.
        
        Args:
            model_string (str): The Hugging Face model string (e.g., 'facebook/opt-6.7b').
            preadd_strength (float): strength and direction of influence 
                (positive or negative) exterted by the prefix or the contrastive sentence 
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_string)
        self.model = AutoModelForCausalLM.from_pretrained(model_string, device_map="auto")

    def get_next_logprobs(self, input_ids):
        """
        Computes log probabilities for the next token given input IDs.
        """
        with torch.no_grad():
            # Returns raw logits (unnormalized prediction scores over vocab) for each token position
            outputs = self.model(input_ids)
            logits = outputs.logits[:, -1, :]  # Get logits for the last token
            # In preadd they DON'T normalize
        #     logprobs = torch.log_softmax(logits, dim=-1)  # Convert logits to log probabilities
        return {'logits': logits}

    def contrastive_generate(
        self, 
        prompt, 
        prefix, 
        preadd_strength=2.0,
        max_tokens=32, 
        temperature=1, # uses logprobs with no scaling, NOT equivalent to greedy
        top_k=0, 
        top_p=1,
        use_prefix=False
    ):
        """
        Generates controlled text using prefix-adaptive decoding (if use_prefix=True).
        Otherwise generates maximally different text using contrastive sentences
        
        Args:
            prompt (str): The input prompt for text generation.
            prefix (str): The control prefix or contrastive sentence to guide generation.
            strength (float): Strength of prefix adaptation.
            max_tokens (int): The maximum number of tokens to generate.
            temperature (float): Sampling temperature.
            top_k (int): Top-k sampling value.
            top_p (float): Top-p sampling value.
        
        Returns:
            str: The generated text (with the prompt removed).
        """
        # Tokenize prompt 
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        prompt_token_length = prompt_ids.size()[-1]

        if use_prefix:
            # Tokenize prefix prepended to prompt
            prefix_ids = self.tokenizer.encode(prefix + " " + prompt, return_tensors="pt").to(self.model.device)
        else:
            # Tokenize the contrastive sentence
            prefix_ids = self.tokenizer.encode(prefix, return_tensors="pt").to(self.model.device)

        for _ in range(max_tokens):
            # Get log probabilities for prompt and prefix, shape: (1, vocab size)
            prompt_output = self.get_next_logprobs(prompt_ids)
            prefix_output = self.get_next_logprobs(prefix_ids)

            # Apply top-k and top-p filtering to base log probabilities
            base_logprobs = self.top_k_top_p_filtering(prompt_output["logits"], top_k=top_k, top_p=top_p)
            # base_logprobs = torch.Tensor(prompt_output['logits']) # no filtering

            # Adjust log probabilities using prefix adaptation
            diff = prefix_output["logits"] - base_logprobs
            final_logprobs = (base_logprobs + diff * preadd_strength) / temperature

            # Sample the next token from adjusted probabilities
            next_token_ind = torch.multinomial(torch.softmax(final_logprobs, dim=-1), num_samples=1).item()
            next_token = torch.tensor([[next_token_ind]]).to(self.model.device)

            # Append the token to both prompt and prefix sequences
            prompt_ids = torch.cat([prompt_ids, next_token], dim=-1)
            prefix_ids = torch.cat([prefix_ids, next_token], dim=-1)

        # Decode generated tokens, removing input prompt tokens
        generations_only = prompt_ids[0][prompt_token_length:] 
        generated_text = self.tokenizer.decode(generations_only, skip_special_tokens=True)
        # full_generated_text = self.tokenizer.decode(prompt_ids[0], skip_special_tokens=True) 

        return generated_text

    @staticmethod
    def top_k_top_p_filtering(logits, top_k=0, top_p=1.0):
        """
        Filters logits using top-k and/or nucleus (top-p) filtering.
        Original implementation: https://github.com/jonnypei/acl23-preadd/blob/main/utils/utils.py#L100
        
        Args:
            logits (torch.Tensor): Logits to filter.
            top_k (int): Keep only top-k tokens with highest probability.
            top_p (float): Keep only tokens with cumulative probability >= top_p.
        
        Returns:
            torch.Tensor: Filtered logits.
        """
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        
        if top_k > 0:
            indices_to_remove = sorted_indices[top_k:]
            logits[indices_to_remove] = -float("Inf")

        if top_p < 1.0:
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            indices_to_remove = cumulative_probs > top_p
            if indices_to_remove.any():
                logits[sorted_indices[indices_to_remove]] = -float("Inf")

        return logits

    def generate(
        self,
        prompt,
        do_sample=False, # Defaults to greedy generation  
        temperature=1.0, 
        top_k=0, 
        top_p=1,
        max_tokens=32
    ):
        """
        Regular, non-contrastive generation via HuggingFace API

        Defaults to greedy generation
        """
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        prompt_token_length = input_ids.size()[-1]
        outputs = self.model.generate(
            input_ids,
            do_sample=do_sample,  
            top_k=top_k,        
            top_p=top_p,      
            temperature=temperature, 
            max_new_tokens=max_tokens
        ) 
        # Decode only the stuff after the prompt
        generated_text = self.tokenizer.decode(outputs[0][prompt_token_length:], skip_special_tokens=True)

        return generated_text


def test_sentiment_control():
    """
    Testing preadd with their sentiment control task, 
    success defined as negative prompt -> positive continuation using a positive-prefix

    Uses Stanford IMDB negative reviews and positive sentiment prompt from preadd paper
    """
    model_string = "facebook/opt-6.7b"  # Replace with your desired Hugging Face model
    # "EleutherAI/gpt-j-6B" # other model they experiment with

    # Load negative sentiment prompts from Stanford IMDB dataset: 
    # https://huggingface.co/datasets/stanfordnlp/imdb/viewer/plain_text/train?views%5B%5D=train&row=68
    # neg_reviews = json.load(open('prefix_tests/imdb_neg_reviews.json', 'r'))
    neg_reviews = json.load(open('prefix_tests/simple_neg_reviews.json', 'r'))
    # Their positive-sentiment-encouraging prefix
    prefix = "The following text exhibits a very positive sentiment and/or opinion."
    max_tokens = 64 

    # Preadd strength of 2 is what they use for their sentiment control task 
    preadd = ContrastiveGen(model_string)

    preadd_test_gens = defaultdict(dict) 
    for review_ind in neg_reviews:
        review = neg_reviews[review_ind]
        generated_text = preadd.contrastive_generate(
            review, 
            prefix, 
            max_tokens=max_tokens, 
            use_prefix=True
        )

        # Save generations and original reviews to results JSON 
        preadd_test_gens[review_ind] = {
            'gen_continuation': generated_text,
            'negative_review': review
        }
        json.dump(preadd_test_gens, open('prefix_tests/preadd_simple_test.json', 'w'), indent=4)


def contrastive_gen(
    base_sentence, 
    contrast_sentence, 
    preadd_strength=2.0,
    model_string="facebook/opt-6.7b",
    max_tokens=64,
    temperature=1, 
    top_k=0, 
    top_p=1,
):
    """
    Given 2 sentences (additive in relation to each other)
    generate text with contrastive generation.

    Outputs:
        Baseline completion (no contrastive generation) for each sentence 
        Contrastive generation encouraging (with preadd_strength) the contrast_sentence
        Contrastive generation discouraging (with preadd_strength) the contrast_sentence
    """
    preadd = ContrastiveGen(model_string)  

    # Baseline gen for base sentence
    generated_text = preadd.generate(
        base_sentence,
        do_sample=True,
        temperature=temperature, 
        top_k=top_k, 
        top_p=top_p,
        max_tokens=max_tokens
    ) 
    print(f"Regular generation for prompt: ``{base_sentence}``:\n``{generated_text}``")

    # Baseline gen for contrasting sentence
    generated_text = preadd.generate(
        contrast_sentence,
        do_sample=True,
        temperature=temperature, 
        top_k=top_k, 
        top_p=top_p,
        max_tokens=max_tokens
    ) 
    print(f"\n\nRegular generation for prompt: ``{contrast_sentence}``:\n``{generated_text}``")

    print(f"\n\nEncouraged sentence (strength {preadd_strength}): ``{contrast_sentence}``\nBase sentence: ``{base_sentence}``")
    generated_text = preadd.contrastive_generate(
        base_sentence, 
        contrast_sentence, 
        preadd_strength=preadd_strength,
        max_tokens=max_tokens, 
        use_prefix=True
    )
    print(f"Generation: ``{generated_text}``\n\n")

    print(f"Discouraged sentence (strength {-preadd_strength}): ``{contrast_sentence}``\nBase sentence: ``{base_sentence}``")
    generated_text = preadd.contrastive_generate(
        base_sentence, 
        contrast_sentence, 
        preadd_strength=-preadd_strength,
        max_tokens=max_tokens, 
        use_prefix=True
    )
    print(f"Generation: ``{generated_text}``")


if __name__ == "__main__":
    model_string = "facebook/opt-6.7b"  # Replace with your desired Hugging Face model
    # "EleutherAI/gpt-j-6B" # other model they experiment with

    # Test implementation by using prompt prefixes for sentiment control
    # test_sentiment_control() 

    # Contrastive generation with 'test' sentences
    contrastive_gen(
        'Betsy\'s picky, she eats chicken nuggets.', 
        'Betsy\'s picky, she just eats chicken nuggets.', 
        preadd_strength=2.0
    )

    # contrastive_gen(
    #     'Yvonne thinks about becoming an astronaut.', 
    #     'Yvonne just thinks about becoming an astronaut.', 
    #     preadd_strength=2.0
    # )
