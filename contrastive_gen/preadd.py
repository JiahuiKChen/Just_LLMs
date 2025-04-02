import os
os.environ['HF_HOME'] = '/datastor1/jiahuikchen/hf_cache'

### Re-implementation of https://github.com/jonnypei/acl23-preadd/blob/main/methods/method_preadd.py
### in a standalone file
import csv
import json
import torch
from tqdm import tqdm
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
        greedy=True, # if true, takes argmax of token logprobs AFTER preadd combination
        temperature=1, # default uses logprobs with no scaling, NOT equivalent to greedy
        filter_base_probs=True, # filter base prompt logprobs with top_k and top_p
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
            temperature (float): Sampling temperature, denominator of preadd logprob distribution.
            filter_base_probs (bool): whether or not to filter prompt's logprobs with top_k, top_p.
            top_k (int): Top-k sampling value, only applied to prompt's logprobs BEFORE preadd combination.
            top_p (float): Top-p sampling value, only applied to prompt's logprobs BEFORE preadd combination.
            use_prefix (bool): if true appends prefix to prompt for contrastive logprobs, otherwise encodes only prefix.
        
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

            if filter_base_probs:
                # Apply top-k and top-p filtering to base log probabilities
                base_logprobs = self.top_k_top_p_filtering(prompt_output["logits"], top_k=top_k, top_p=top_p)
            else:
                # Don't apply top k or top p filtering to base generation logprobs
                base_logprobs = torch.Tensor(prompt_output['logits'])

            # Adjust log probabilities using preadd
            diff = prefix_output["logits"] - base_logprobs
            final_logprobs = (base_logprobs + diff * preadd_strength) / temperature

            # Take the highest probability token if greedy
            if greedy:
                next_token_ind = torch.argmax(torch.softmax(final_logprobs, dim=-1)).item()  
            else:
                # Sample the next token if not greedy 
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
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs['input_ids'].to(self.model.device)
        attention_mask = inputs['attention_mask'].to(self.model.device)
        prompt_token_length = input_ids.size()[-1]
        outputs = self.model.generate(
            input_ids,
            do_sample=do_sample,  
            top_k=top_k,        
            top_p=top_p,      
            temperature=temperature, 
            max_new_tokens=max_tokens,
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.eos_token_id
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


def run_contrastive_gen(
    base_sentence, 
    contrast_sentence, 
    greedy,
    preadd_strength=2.0,
    model_string="facebook/opt-6.7b",
    max_tokens=64,
    temperature=1, 
    top_k=0, 
    top_p=1,
    use_prefix=False
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
        do_sample=True if not greedy else False,
        temperature=temperature, 
        top_k=top_k, 
        top_p=top_p,
        max_tokens=max_tokens
    ) 
    print(f"Regular generation for prompt: ``{base_sentence}``:\n``{generated_text}``")

    # Baseline gen for contrasting sentence
    generated_text = preadd.generate(
        contrast_sentence,
        do_sample=True if not greedy else False,
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
        greedy=greedy,
        use_prefix=use_prefix,
    )
    print(f"Generation: ``{generated_text}``\n\n")

    print(f"Discouraged sentence (strength {-preadd_strength}): ``{contrast_sentence}``\nBase sentence: ``{base_sentence}``")
    generated_text = preadd.contrastive_generate(
        base_sentence, 
        contrast_sentence, 
        preadd_strength=-preadd_strength,
        max_tokens=max_tokens, 
        greedy=greedy,
        use_prefix=use_prefix
    )
    print(f"Generation: ``{generated_text}``")


def just_contrastive_gen(
    use_convo_prompt = False,
    greedy=True, 
    preadd_strength=2.0,
    models = ["allenai/OLMo-2-1124-7B", "meta-llama/Llama-3.1-8B", "facebook/opt-6.7b"]
):
    """
    For all 60 "just" sentences, generates baseline completions for each sentence
    and 2 preadd completions (encouraging and discouraging "just")
    """
    # Sentence prompts
    with open('/datastor1/jiahuikchen/Just_LLMs/just_sentences.csv', 'r') as f:
        data = list(csv.DictReader(f))
        w_just = [row['w_just'] for row in data]
        wo_just = [row['wo_just'] for row in data]

    # Extract model names 
    model_parts = [model.split('/')[-1] for model in models]

    for model_part, model_string in zip(model_parts, models):
        # Create a TSV file for the model 
        tsv_filename = f"{model_part}_{preadd_strength}.tsv"
        with open(tsv_filename, 'w', newline='') as file:
            writer = csv.writer(file, delimiter='\t')
            writer.writerow(["Just", "JustEncouragedPreadd", "JustNoPreadd", "NoJust", "JustDiscouragedPreadd", "NoJustNoPreadd"])  # Header row
        
        preadd = ContrastiveGen(model_string)  
        max_tokens=64
        temperature = 1.0 
        top_k=0
        top_p=1

        # Baseline and preadd generations for all just minimal pair sentences
        for sentence_i in tqdm(range(len(w_just))):
            just_sentence = w_just[sentence_i]
            no_just_sentence = wo_just[sentence_i]

            if use_convo_prompt:
                # # convo_greedy_gens convo_top1_gens
                # just_sentence = f"Dave says \"{just_sentence}\"\nThen Sally asks "
                # no_just_sentence = f"Dave says \"{no_just_sentence}\"\nThen Sally asks "
                # convo_1
                just_sentence = f"Dave and Sally are having a conversation. Dave says \"{just_sentence}\"\nThen Sally asks "
                no_just_sentence = f"Dave and Sally are having a conversation. Dave says \"{no_just_sentence}\"\nThen Sally asks "
                # # convo_2
                # just_sentence = f"Dave and Sally are two friends having a conversation. Dave says \"{just_sentence}\" Then Sally asks "
                # no_just_sentence = f"Dave and Sally are two friends having a conversation. Dave says \"{no_just_sentence}\"\nThen Sally asks "

            # Preadd generation encouraging just sentence
            just_preadd_enc = preadd.contrastive_generate(
                no_just_sentence, 
                just_sentence, 
                preadd_strength=preadd_strength,
                temperature=temperature, 
                top_k=top_k, 
                top_p=top_p,
                max_tokens=max_tokens,
                greedy=greedy,
                use_prefix=False,
                filter_base_probs=True if not greedy else False
            )

            # Baseline gen for just sentence
            just_baseline = preadd.generate(
                just_sentence,
                do_sample=True if not greedy else False,
                temperature=temperature, 
                top_k=top_k, 
                top_p=top_p,
                max_tokens=max_tokens
            ) 

            # Preadd generation for discouraging just sentence
            just_preadd_disc = preadd.contrastive_generate(
                no_just_sentence, 
                just_sentence, 
                preadd_strength=-preadd_strength,
                temperature=temperature, 
                top_k=top_k, 
                top_p=top_p,
                max_tokens=64,
                greedy=greedy,
                use_prefix=False,
                filter_base_probs=True if not greedy else False
            )            

            # Baseline gen for no just sentence
            no_just_baseline = preadd.generate(
                no_just_sentence,
                do_sample=True if not greedy else False,
                temperature=temperature, 
                top_k=top_k, 
                top_p=top_p,
                max_tokens=max_tokens
            ) 

            # Append generated_text to the created TSV file
            with open(tsv_filename, 'a', newline='') as file:
                writer = csv.writer(file, delimiter='\t')
                writer.writerow([just_sentence, just_preadd_enc, just_baseline, no_just_sentence, just_preadd_disc, no_just_baseline])


if __name__ == "__main__":
    # "EleutherAI/gpt-j-6B" # other model they experiment with

    # Test implementation by using prompt prefixes for sentiment control
    # test_sentiment_control() 

    # Contrastive generation with 'test' sentences
    # run_contrastive_gen(
    #     'I saw a cat', 
    #     'I saw a dog', 
    #     preadd_strength=2.0,
    #     greedy=False,
    #     use_prefix=False
    # )

    # # Betsy "just" test
    # run_contrastive_gen(
    #     'Betsy\'s picky, she eats chicken nuggets.', 
    #     'Betsy\'s picky, she just eats chicken nuggets.', 
    #     preadd_strength=2.0,
    #     greedy=True,
    #     use_prefix=False
    # )

    # # Yvonne "just" test
    # run_contrastive_gen(
    #     'Yvonne thinks about becoming an astronaut.', 
    #     'Yvonne just thinks about becoming an astronaut.', 
    #     preadd_strength=2.0,
    #     greedy=True,
    #     use_prefix=False
    # )

    # All "just" minimal pair generation runs
    # just_contrastive_gen(models=["allenai/OLMo-2-1124-7B"], greedy=True, use_convo_prompt=True) 
    # just_contrastive_gen(models=["meta-llama/Llama-3.1-8B"], greedy=True, use_convo_prompt=True) 
    just_contrastive_gen(models=["google/gemma-7b"], greedy=True, use_convo_prompt=True) 
    # just_contrastive_gen(models=["Qwen/Qwen2.5-7B"], greedy=True, use_convo_prompt=True)  
    # just_contrastive_gen(models=["facebook/opt-6.7b"], greedy=False, use_convo_prompt=True)  
