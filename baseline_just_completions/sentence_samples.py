import os
os.environ['HF_HOME'] = '/datastor1/jiahuikchen/hf_cache'
import csv
import json
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer


DEVICE = 'cuda:5'
### Sentence prompts
with open('just_sentences.csv', 'r') as f:
    data = list(csv.DictReader(f))
    w_just = [row['w_just'] for row in data]
    wo_just = [row['wo_just'] for row in data]


### Olmo 7B 
olmo = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-1124-7B").to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-7B")


def gen_100_samples(sentence_prompt):
    # Tokenize the prompt separately to better parse the response
    prompt_tokens = tokenizer.encode(sentence_prompt, add_special_tokens=False)
    prompt_token_length = len(prompt_tokens)

    just_inputs = tokenizer(sentence_prompt, return_tensors='pt', return_token_type_ids=False).to(DEVICE)
    # 100 is shape of first dim
    response = olmo.generate(
        **just_inputs, 
        max_new_tokens=50, 
        do_sample=True,
        top_k=100,
        num_return_sequences=100,
        temperature=0.7
    )
    # Decode outputs, removing input tokens
    generated_texts = []
    for generation_i in range(len(response)):
        output = response[generation_i]
        new_tokens = output[prompt_token_length:]
        decoded_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        generated_texts.append(decoded_text.strip())
    
    return generated_texts


### Generating 100 samples for each of the minimal pair sentence prompts
# Results are saved as sentence_id: { rank: sentence, ... }
w_just_generations = defaultdict(dict)
wo_just_generations = defaultdict(dict)
# For each sentence with/without just, get the top 100 sample responses
for i in tqdm(range(len(w_just))):
    # Sentence prompts WITH JUST
    just_sentence = w_just[i]
    w_just_generations[f'sentence_{i}']['sentence_prompt'] = just_sentence
    just_responses = gen_100_samples(sentence_prompt=just_sentence)
    for generation_i in range(len(just_responses)):
        w_just_generations[f'sentence_{i}'][f'generation_{generation_i}'] = just_responses[generation_i] 
    json.dump(w_just_generations, open('w_just_responses.json', 'w'), indent=4)

    # Sentence prompts WITHOUT JUST
    wo_just_sentence = wo_just[i]
    wo_just_generations[f'sentence_{i}']['sentence_prompt'] = wo_just_sentence 
    wo_just_responses = gen_100_samples(sentence_prompt=wo_just_sentence)
    for generation_i in range(len(wo_just_responses)):
        wo_just_generations[f'sentence_{i}'][f'generation_{generation_i}'] = wo_just_responses[generation_i] 
    json.dump(wo_just_generations, open('wo_just_responses.json', 'w'), indent=4)


# ### Parse into CSV for better comparisons for presentation
# # Each CSV file is for 1 sentence and contains rows of: generation_i, with just generation, without just generation
# w_just_dict = json.load(open('w_just_responses.json', 'r'))
# wo_just_dict = json.load(open('wo_just_responses.json', 'r'))
# fieldnames = ['generation_rank', 'just_response', 'wo_just_response']
# for sentence_id in w_just_dict:
#     just_generations = w_just_dict[sentence_id]
#     wo_just_generations = wo_just_dict[sentence_id]

#     with open(f'generation_csvs/sentence_{int(sentence_id.split('_')[-1])+1}.csv', 'w+', newline='') as file:
#         writer = csv.writer(file, delimiter='\t') 
#         writer.writerow(fieldnames)

#         for generation_id in just_generations:
#             if generation_id == 'sentence_prompt':
#                 continue
#             writer.writerow([
#                 generation_id,
#                 just_generations[generation_id],
#                 wo_just_generations[generation_id]
#             ])