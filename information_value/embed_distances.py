import json
import csv
import numpy as np
from scipy.spatial.distance import cosine, euclidean
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline


t2vec_model = TextToEmbeddingModelPipeline(
    encoder="text_sonar_basic_encoder",
    tokenizer="text_sonar_basic_encoder"
)
# sentences = ['My name is SONAR.', 'I can embed the sentences into vectorial space.']
# embeddings = t2vec_model.predict(sentences, source_lang="eng_Latn")
# print(embeddings.shape)

# Want: average of each sentence's generations + embedding (CSV)
# AVERAGEs overall: distance from no_just generations and excluded question 

# --- Load your JSON data ---
json_name = "Llama-3.1-8B_temp0.5_5generations.json"
with open(json_name, 'r') as f:
    data = json.load(f)

# --- CSV Output Setup ---
csv_fields = [
    'pair_id', 'excluded_question', 'just_sentence', 'no_just_sentence',
    'just_questions', 'no_just_questions',
    'avg_just_cosine', 'avg_just_euclidean',
    'avg_no_just_cosine', 'avg_no_just_euclidean'
]

rows = []

# --- For overall averages ---
all_just_cosines = []
all_just_euclideans = []
all_no_just_cosines = []
all_no_just_euclideans = []

for pair_id, pair in data.items():
    excluded_question = pair['excluded_question']
    just_questions = pair['just_questions']
    no_just_questions = pair['no_just_questions']

    # Prepare all sentences for embedding
    sentences = [excluded_question] + just_questions + no_just_questions
    embeddings = t2vec_model.predict(sentences, source_lang="eng_Latn")
    emb_excluded = embeddings[0]
    emb_just = embeddings[1:1+len(just_questions)]
    emb_no_just = embeddings[1+len(just_questions):]

    # Calculate distances for just_questions
    just_cosines = []
    just_euclideans = []
    for emb in emb_just:
        just_cosines.append(cosine(emb_excluded, emb))
        just_euclideans.append(euclidean(emb_excluded, emb))
    avg_just_cosine = float(np.mean(just_cosines)) if just_cosines else None
    avg_just_euclidean = float(np.mean(just_euclideans)) if just_euclideans else None
    all_just_cosines.extend(just_cosines)
    all_just_euclideans.extend(just_euclideans)

    # Calculate distances for no_just_questions
    no_just_cosines = []
    no_just_euclideans = []
    for emb in emb_no_just:
        no_just_cosines.append(cosine(emb_excluded, emb))
        no_just_euclideans.append(euclidean(emb_excluded, emb))
    avg_no_just_cosine = float(np.mean(no_just_cosines)) if no_just_cosines else None
    avg_no_just_euclidean = float(np.mean(no_just_euclideans)) if no_just_euclideans else None
    all_no_just_cosines.extend(no_just_cosines)
    all_no_just_euclideans.extend(no_just_euclideans)

    # Prepare CSV row
    row = [
        pair_id,
        excluded_question,
        pair['just_sentence'],
        pair['no_just_sentence'],
        json.dumps(just_questions),  # Store as JSON string for readability
        json.dumps(no_just_questions),
        avg_just_cosine,
        avg_just_euclidean,
        avg_no_just_cosine,
        avg_no_just_euclidean
    ]
    rows.append(row)

    # --- Write to CSV ---
    with open(f"{json_name.split('.json')[0]}.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(csv_fields)
        writer.writerows(rows)


# --- Print overall averages ---
overall_just_cosine = float(np.mean(all_just_cosines)) if all_just_cosines else None
overall_no_just_cosine = float(np.mean(all_no_just_cosines)) if all_no_just_cosines else None
overall_just_euclidean = float(np.mean(all_just_euclideans)) if all_just_euclideans else None
overall_no_just_euclidean = float(np.mean(all_no_just_euclideans)) if all_no_just_euclideans else None

print(f"Overall average cosine distance (just_questions): {overall_just_cosine:.4f}")
print(f"Overall average cosine distance (no_just_questions): {overall_no_just_cosine:.4f}")
print(f"Overall average euclidean distance (just_questions): {overall_just_euclidean:.4f}")
print(f"Overall average euclidean distance (no_just_questions): {overall_no_just_euclidean:.4f}")