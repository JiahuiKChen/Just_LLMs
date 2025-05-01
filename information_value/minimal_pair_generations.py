import os
import re
import csv
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict


DEVICE = 'cuda'

def clean_and_parse_question(generations_list, prompt_token_length, tokenizer):
    """
    Given a list of generations and token length of prompt, will remove prompt from generation and parse out the first question in the completion.
    Will skip the generation if no '?' is in completion
    """
    # Remove prompt from generations
    clean_generations = []
    for generation_i in range(len(generations_list)):
        output = generations_list[generation_i]
        new_tokens = output[prompt_token_length:]
        decoded_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        clean_generations.append(decoded_text.strip())
    # Parse out questions (will be skipped if no '?' is contained)
    generated_questions = []
    for s in clean_generations:
        match = re.match(r'^(.*?\?)', s)
        if match:
            generated_questions.append(match.group(1))
    
    print(f"Parsed {len(generated_questions)} questions out of {len(generations_list)} generations")

    return generated_questions


def gen_questions(model_string="meta-llama/Llama-3.1-8B", num_questions=5):
    """
    With the given model, generates num_questions completions for each "just" minimal pair.
    Uses this conversation template to generate questions, then parses out the question:
        "Dave and Sally are having a conversation. Dave says \"{just_sentence}\"\nThen Sally asks "

    Outputs a JSON of format:
        'pair_id': {
            'just_questions': [...],
            'no_just_questions': [...],
            'excluded_question': question that SHOULD be excluded according to ling theory,
            'just_sentence': ...,
            'no_just_sentence': ...,
        }
    """
    # Init LLM
    tokenizer = AutoTokenizer.from_pretrained(model_string)
    model = AutoModelForCausalLM.from_pretrained(model_string, device_map="auto")

    # Read in minimal pairs and their excluded questions
    with open('../just_sentences.csv', 'r') as f:
        minimal_pairs = list(csv.DictReader(f))
        w_just_sentences = [row['w_just'] for row in minimal_pairs]
        wo_just_sentences = [row['wo_just'] for row in minimal_pairs]
        excluded_questions = [row['Question_bad_w_just'] for row in minimal_pairs] 

    # Conversation template
    conversation_template = 'Dave and Sally are having a conversation. Dave says "{sentence}"\nThen Sally asks '

    # Generations for all minimal pair sentences
    all_generations = defaultdict(dict)
    for i in tqdm(range(len(w_just_sentences))):
        pair_id = f"pair_{i}"
        just_sentence = w_just_sentences[i]
        no_just_sentence = wo_just_sentences[i]
        excluded_q = excluded_questions[i]
        all_generations[pair_id]['excluded_question'] = excluded_q
        all_generations[pair_id]['just_sentence'] = just_sentence
        all_generations[pair_id]['no_just_sentence'] = no_just_sentence 

        # Generate num_questions completions using beam search (gets most LIKELY generations) for "just" -- TODO: make inference strategy a param
        just_prompt = conversation_template.format(sentence=just_sentence)
        # Tokenize the prompt separately to better parse the response
        prompt_tokens = tokenizer.encode(just_prompt, add_special_tokens=False)
        prompt_token_length = len(prompt_tokens)

        just_inputs = tokenizer(just_prompt, return_tensors='pt', return_token_type_ids=False)
        outputs = model.generate(
            **just_inputs,
            max_length=50,
            do_sample=True,
            # num_beams=num_questions,   # x most likely completions
            num_return_sequences=num_questions, 
            top_p=1.0,  # most random sampling, lower top_p is more focused              
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_questions = clean_and_parse_question(
            generations_list=outputs,
            prompt_token_length=prompt_token_length,
            tokenizer=tokenizer 
        )
        all_generations[pair_id]['just_questions'] = generated_questions

        # Generate num_questions completions using beam search (gets most LIKELY generations) for "no just"
        no_just_prompt = conversation_template.format(sentence=no_just_sentence)
        # Tokenize the prompt separately to better parse the response
        prompt_tokens = tokenizer.encode(no_just_prompt, add_special_tokens=False)
        prompt_token_length = len(prompt_tokens)

        no_just_inputs = tokenizer(no_just_prompt, return_tensors='pt', return_token_type_ids=False)
        outputs = model.generate(
            **no_just_inputs,
            max_length=50,
            do_sample=True,
            num_beams=num_questions,   # x most likely completions
            num_return_sequences=num_questions, 
            # top_p=1.0,  # most random sampling, lower top_p is more focused              
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_questions = clean_and_parse_question(
            generations_list=outputs,
            prompt_token_length=prompt_token_length,
            tokenizer=tokenizer 
        )
        all_generations[pair_id]['no_just_questions'] = generated_questions

        with open(f"{model_string.split('/')[-1]}_{num_questions}generations.json", 'w') as f:
            json.dump(all_generations, f, indent=4)


### Generates 5 questions for each sentence in minimal pair
gen_questions() 