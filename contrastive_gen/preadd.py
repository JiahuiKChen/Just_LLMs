import os
os.environ['HF_HOME'] = '/datastor1/jiahuikchen/hf_cache'

### Re-implementation of https://github.com/jonnypei/acl23-preadd/blob/main/methods/method_preadd.py
### in a standalone file
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict


class PreAdd:
    def __init__(self, model_string, preadd_strength=2.0):
        """
        Initializes the PreAdd class with a specified model.
        
        Args:
            model_string (str): The Hugging Face model string (e.g., 'facebook/opt-6.7b').
            preadd_strength (float): strength and direction of influence 
                (positive or negative) exterted by the prefix or the contrastive sentence 
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_string)
        self.model = AutoModelForCausalLM.from_pretrained(model_string, device_map="auto")
        self.preadd_strength = preadd_strength

    def get_next_logprobs(self, input_ids):
        """
        Computes log probabilities for the next token given input IDs.
        
        Args:
            input_ids (torch.Tensor): Tokenized input IDs.
        
        Returns:
            dict: A dictionary containing logits and cache ID.
        """
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[:, -1, :]  # Get logits for the last token
            # In preadd they DON'T normalize
        #     logprobs = torch.log_softmax(logits, dim=-1)  # Convert logits to log probabilities
        # return {"logits": logprobs}
        return {'logits': logits}

    def generate(
        self, 
        prompt, 
        prefix, 
        max_tokens=32, 
        temperature=1, # they don't normalize token logits at all
        top_k=0, 
        top_p=1,
        use_prefix=False
    ):
        """
        Generates controlled text using prefix-adaptive decoding.
        
        Args:
            prompt (str): The input prompt for text generation.
            prefix (str): The control prefix or contrastive sentence to guide generation.
            strength (float): Strength of prefix adaptation.
            max_tokens (int): The maximum number of tokens to generate.
            temperature (float): Sampling temperature.
            top_k (int): Top-k sampling value.
            top_p (float): Top-p sampling value.
            use_prefix (boolean): 
                Default False, uses preadd with alternate sentences.
                If True, uses preadd with prompt prefixes (as in original work).
        
        Returns:
            str: The generated text.
        """
        # Tokenize prompt 
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        prompt_token_length = prompt_ids.size()[-1]

        if prefix:
            # Tokenize prefix prepended to prompt
            prefix_ids = self.tokenizer.encode(prefix + " " + prompt, return_tensors="pt")
        else:
            # Tokenize the contrastive sentence
            prefix_ids = self.tokenizer.encode(prefix)

        for _ in range(max_tokens):
            # Get log probabilities for prompt and prefix, shape: (1, vocab size)
            prompt_output = self.get_next_logprobs(prompt_ids)
            prefix_output = self.get_next_logprobs(prefix_ids)

            # Apply top-k and top-p filtering to base log probabilities
            base_logprobs = self.top_k_top_p_filtering(prompt_output["logits"], top_k=top_k, top_p=top_p)
            # base_logprobs = torch.Tensor(prompt_output['logits']) # no filtering

            # Adjust log probabilities using prefix adaptation
            diff = prefix_output["logits"] - base_logprobs
            final_logprobs = (base_logprobs + diff * self.preadd_strength) / temperature

            # Sample the next token from adjusted probabilities
            next_token = torch.multinomial(torch.softmax(final_logprobs, dim=-1), num_samples=1).item()

            # Append the token to both prompt and prefix sequences
            prompt_ids = torch.cat([prompt_ids, torch.tensor([[next_token]])], dim=-1)
            prefix_ids = torch.cat([prefix_ids, torch.tensor([[next_token]])], dim=-1)

        # Decode generated tokens, removing input prompt tokens
        generations_only = prompt_ids[0][prompt_token_length:] 
        generated_text = self.tokenizer.decode(generations_only, skip_special_tokens=True)
        # full_generated_text = self.tokenizer.decode(prompt_ids[0], skip_special_tokens=True) 
        # breakpoint()

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
    preadd = PreAdd(model_string, 2.0)

    preadd_test_gens = defaultdict(dict) 
    for review_ind in neg_reviews:
        review = neg_reviews[review_ind]
        generated_text = preadd.generate(
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
    preadd = PreAdd(model_string, preadd_strength)    

    # contrasting_controls = [
    #     'How are you?',
    #     'I saw a dog'
    # ]

    print(f"Encouraged sentence: ``{contrast_sentence}``\nBase sentence: ``{base_sentence}``")
    generated_text = preadd.generate(
        base_sentence, 
        contrast_sentence, 
        max_tokens=max_tokens, 
        use_prefix=True
    )
    print(f"Generation: ``{generated_text}``\n\n\n")

    print(f"Encouraged sentence: ``{base_sentence}``\nBase sentence: ``{contrast_sentence}``")
    generated_text = preadd.generate(
        contrast_sentence, 
        base_sentence, 
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
        'How are you?', 
        'I saw a dog', 
        preadd_strength=5.0
    )