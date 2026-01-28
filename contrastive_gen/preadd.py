import os
os.environ['HF_HOME'] = '/datastor1/jiahuikchen/hf_cache'

### Re-implementation of https://github.com/jonnypei/acl23-preadd/blob/main/methods/method_preadd.py
### in a standalone file
import csv
import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict


class ContrastiveGen:
    def __init__(self, model_string):
        """
        Initializes the ContrastiveGen class with a specified model.

        Args:
            model_string (str): The Hugging Face model string (e.g., 'facebook/opt-6.7b').
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
        preadd_strength (float): Strength of prefix adaptation.
        max_tokens (int): The maximum number of tokens to generate.
        greedy (bool): If true, takes argmax of token logprobs AFTER preadd combination.
        temperature (float): Sampling temperature, denominator of preadd logprob distribution.
        filter_base_probs (bool): Whether to filter prompt's logprobs with top_k, top_p.
        top_k (int): Top-k sampling value, only applied to prompt's logprobs BEFORE preadd combination.
        top_p (float): Top-p sampling value, only applied to prompt's logprobs BEFORE preadd combination.
        use_prefix (bool): If true, appends prefix to prompt for contrastive logprobs, otherwise encodes only prefix.

        
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

        Args:
        prompt (str): The input prompt for text generation.
        do_sample (bool): Whether to use sampling or greedy generation.
        temperature (float): Sampling temperature.
        top_k (int): Top-k sampling value.
        top_p (float): Top-p sampling value.
        max_tokens (int): The maximum number of tokens to generate.
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

    Args:
    base_sentence (str): The base sentence to use.
    contrast_sentence (str): The contrastive sentence to encourage/discourage.
    greedy (bool): Whether to use greedy generation.
    preadd_strength (float): Strength of prefix adaptation.
    model_string (str): Model to use.
    max_tokens (int): The maximum number of tokens to generate.
    temperature (float): Sampling temperature.
    top_k (int): Top-k sampling value.
    top_p (float): Top-p sampling value.
    use_prefix (bool): If true, uses prefix-adaptive decoding.

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


def just_sentences_contrastive_gen(
    use_convo_prompt = False,
    greedy=True, 
    preadd_strength=2.0,
    models = ["allenai/OLMo-2-1124-7B", "meta-llama/Llama-3.1-8B", "facebook/opt-6.7b"]
):
    """
    For all 60 "just" sentences, generates baseline completions for each sentence
    and 2 preadd completions (encouraging and discouraging "just")

    Args:
    use_convo_prompt (bool): If true, applies conversational prompt wrapping.
    greedy (bool): Whether to use greedy generation.
    preadd_strength (float): Strength of prefix adaptation.
    models (list): List of model strings to use.
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

### Daily Dialogue generation
SYSTEM_INSTRUCTION = (
    "System: You are participating in a two-person conversation. "
    "Generate a natural, context-aware response as Speaker 1. "
    "Keep your response to 1-2 sentences and respond directly to what Speaker 2 just said."
)

def _build_convo_prompt(
        context_text: str, 
        last_utterance: str, 
        system: str = SYSTEM_INSTRUCTION
    ) -> str:
    """
    Build a conversation prompt with explicit Speaker 1 and Speaker 2 labels.
    The context alternates between speakers, last_utterance is Speaker 2's turn,
    and we prompt for Speaker 1's next response.
    """
    # Split context by __eou__ delimiter
    turns = [t.strip() for t in context_text.split("__eou__") if t.strip()]
    
    # Assign alternating speaker labels starting with Speaker 1
    labeled_turns = []
    for i, turn in enumerate(turns):
        speaker_label = "Speaker 1" if i % 2 == 0 else "Speaker 2"
        labeled_turns.append(f"{speaker_label}: {turn}")
    
    # Determine which speaker says the last_utterance
    # Since turns alternate and we want to prompt for Speaker 1's response,
    # last_utterance should be Speaker 2's line
    next_speaker_num = len(turns) % 2 + 1  # 1 if even turns, 2 if odd turns
    last_speaker_label = f"Speaker {next_speaker_num}"
    
    # Build the full conversation
    convo = "\n".join(labeled_turns)
    
    # Add the last utterance and prompt for Speaker 1's response
    # Arrange so Speaker 1 always responds next
    if next_speaker_num == 1:
        # If last_utterance would be Speaker 1, swap the logic
        # Actually, to ensure Speaker 1 always responds, treat last_utterance as Speaker 2
        prompt = (
            f"{system}\n\n"
            f"Conversation:\n{convo}\n"
            f"Speaker 2: {last_utterance}\n"
            f"Speaker 1:"
        )
    else:
        prompt = (
            f"{system}\n\n"
            f"Conversation:\n{convo}\n"
            f"Speaker 2: {last_utterance}\n"
            f"Speaker 1:"
        )
    
    return prompt

def just_dd_contrastive_gen(
    dd_tsv_path: str = "just_dd.txt",
    out_path: str = "just_dd",
    greedy: bool = True,
    preadd_strength: float = 2.0,
    models = ["allenai/OLMo-2-1124-7B"],
    max_tokens: int = 64,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
):
    """
    Generate dialogue completions with explicit speaker labels (Speaker 1, Speaker 2).
    The LLM completes Speaker 1's next response after Speaker 2's utterance.
    Outputs are written to files inside `out_path` (created if missing).
    """
    # Read TSV
    with open(dd_tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    # Ensure output directory exists
    out_dir = Path(out_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_parts = [m.split("/")[-1] for m in models]
    prefix = Path(dd_tsv_path).stem  # e.g., "just_dd" from "just_dd.txt"

    for model_part, model_string in zip(model_parts, models):
        out_file = out_dir / f"{model_part}_preAdd{preadd_strength}_{prefix}.tsv"

        # Write header
        with out_file.open("w", newline="", encoding="utf-8") as outf:
            writer = csv.writer(outf, delimiter="\t")
            writer.writerow([
                "id",
                "Context",
                "WWord",
                "WoWord",
                "WPrompt",
                "WPreaddEnc",
                "WBaseline",
                "WoPrompt",
                "WPreaddDisc",
                "WoBaseline",
            ])

        preadd = ContrastiveGen(model_string)

        for row in tqdm(rows, desc=f"Generating ({model_part})"):
            rid = row.get("id", "")
            ctx = row.get("context", "") or row.get("Context", "")
            w_sent = row.get("w_word", "") or row.get("wword", "")
            wo_sent = row.get("wo_word", "") or row.get("woword", "")

            if not ctx or not w_sent or not wo_sent:
                continue

            # Build prompts with explicit speaker labels
            w_prompt = _build_convo_prompt(ctx, w_sent, SYSTEM_INSTRUCTION)
            wo_prompt = _build_convo_prompt(ctx, wo_sent, SYSTEM_INSTRUCTION)

            # Baseline generations
            w_baseline = preadd.generate(
                w_prompt,
                do_sample=not greedy,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            wo_baseline = preadd.generate(
                wo_prompt,
                do_sample=not greedy,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_tokens=max_tokens,
            )

            # Contrastive: encourage 'just' (wo_prompt as base, w_prompt as contrast)
            w_preadd_enc = preadd.contrastive_generate(
                prompt=wo_prompt,
                prefix=w_prompt,
                preadd_strength=preadd_strength,
                max_tokens=max_tokens,
                greedy=greedy,
                temperature=temperature,
                filter_base_probs=True if not greedy else False,
                top_k=top_k,
                top_p=top_p,
                use_prefix=False,
            )

            # Contrastive: discourage 'just'
            w_preadd_disc = preadd.contrastive_generate(
                prompt=wo_prompt,
                prefix=w_prompt,
                preadd_strength=-preadd_strength,
                max_tokens=max_tokens,
                greedy=greedy,
                temperature=temperature,
                filter_base_probs=True if not greedy else False,
                top_k=top_k,
                top_p=top_p,
                use_prefix=False,
            )

            # Write results
            with out_file.open("a", newline="", encoding="utf-8") as outf:
                writer = csv.writer(outf, delimiter="\t")
                writer.writerow([
                    rid,
                    ctx,
                    w_sent,
                    wo_sent,
                    w_prompt,
                    w_preadd_enc,
                    w_baseline,
                    wo_prompt,
                    w_preadd_disc,
                    wo_baseline,
                ])


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

    ### All "just" minimal pair generation runs
    # just_sentences_contrastive_gen(models=["allenai/OLMo-2-1124-7B"], greedy=True, use_convo_prompt=True) 
    # just_sentences_contrastive_gen(models=["meta-llama/Llama-3.1-8B"], greedy=True, use_convo_prompt=True) 
    # just_sentences_contrastive_gen(models=["google/gemma-7b"], greedy=True, use_convo_prompt=True) 
    # just_sentences_contrastive_gen(models=["Qwen/Qwen2.5-7B"], greedy=True, use_convo_prompt=True)  
    # just_sentences_contrastive_gen(models=["facebook/opt-6.7b"], greedy=False, use_convo_prompt=True)  

    ### Daily Dialogue "just" generations
    # RUNNING: Olmo, Llama
    # Generate dialogue completions for the dataset, one file per model
    just_dd_contrastive_gen(
        dd_tsv_path="/datastor1/jiahuikchen/Just_LLMs/dd_data/filtered_just_dd.tsv",
        out_path="filtered_just_dd",
        greedy=True,
        preadd_strength=2.0,
        models=["meta-llama/Llama-3.1-8B"],  
        max_tokens=64,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
    )
