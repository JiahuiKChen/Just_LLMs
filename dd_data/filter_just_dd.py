###
### just DD data: 
# Original rows (excluding header): 3232
# Filtered rows: 1182
#
### not DD data:
# Original rows (excluding header): 3543
# Filtered rows: 1307
"""
Filter dialogue TSV rows where the specified word appears in the last sentence
of the w_word column and the context column contains at most two '__eou__'
markers.

This script:
- Reads an input TSV file (e.g., 'just_dd.tsv') with at least 'w_word'
  and 'context' columns.
- Interprets '__eou__' as a sentence boundary marker in those columns.
- Keeps only rows where:
  1) All of w_word, wo_word, followup, and context are valid strings,
  2) The last non-empty sentence in the w_word field contains the specified
     word (case-insensitive), and
  3) The context field contains no more than two occurrences of '__eou__'.
- Writes all surviving rows (with all original columns preserved) to
  'filtered_<input filename>'.
- Prints the number of rows in the original file (excluding the header)
  and the number of rows in the filtered output.
"""
import argparse
import csv
import os


def is_valid_string(value) -> bool:
    """
    Return True iff the value is a non-empty string (not None, not just whitespace).
    """
    return isinstance(value, str) and value.strip() != ""


def last_sentence_contains_word(w_word_value: str, word: str) -> bool:
    """
    Return True iff the last sentence of w_word contains the specified word
    (case-insensitive), where sentences are separated by '__eou__'.
    """
    if not w_word_value:
        return False
    # Split by __eou__, take last non-empty segment
    parts = [p.strip() for p in w_word_value.split("__eou__") if p.strip()]
    if not parts:
        return False
    last_sent = parts[-1]
    # Tokenize naively on whitespace and punctuation
    # and check for exact word match (case-insensitive)
    tokens = []
    for tok in last_sent.replace(",", " ").replace(".", " ").replace("!", " ") \
                        .replace("?", " ").replace(";", " ").replace(":", " ") \
                        .split():
        t = tok.strip().lower()
        if t:
            tokens.append(t)
    return word.lower() in tokens

def last_sentence_is_question(w_word_value: str) -> bool:
    """
    Return True iff the last non-empty sentence of w_word is a question
    (i.e. ends with '?').
    """
    if not w_word_value:
        return False
    parts = [p.strip() for p in w_word_value.split("__eou__") if p.strip()]
    if not parts:
        return False
    return parts[-1].rstrip().endswith("?")

def context_max_two_eou(context_value: str) -> bool:
    """
    Return True iff context has at most 2 occurrences of '__eou__'.
    """
    if not context_value:
        return True
    return context_value.count("__eou__") <= 2

def main(input_path: str = "just_dd.tsv", word: str = "just"):
    # Construct output filename as "filtered_<input filename>"
    input_dir = os.path.dirname(input_path)
    input_filename = os.path.basename(input_path)
    output_filename = f"filtered_{input_filename}"
    output_path = os.path.join(input_dir, output_filename) if input_dir else output_filename
    original_rows = 0
    kept_rows = []

    # Read TSV
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames

        for row in reader:
            original_rows += 1

            w_word_val = row.get("w_word", "")
            wo_word_val = row.get("wo_word", "")
            followup_val = row.get("followup", "")
            context_val = row.get("context", "")

            # Check that all required fields are valid strings
            if not all(is_valid_string(v) for v in [w_word_val, wo_word_val, followup_val, context_val]):
                continue

            if (last_sentence_contains_word(w_word_val, word)
                    and context_max_two_eou(context_val)
                    and not last_sentence_is_question(w_word_val)):
                kept_rows.append(row)

    # Write filtered TSV, preserving all columns
    with open(output_path, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in kept_rows:
            writer.writerow(row)

    print(f"Original rows (excluding header): {original_rows}")
    print(f"Filtered rows: {len(kept_rows)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter dialogue TSV rows based on word occurrence.")
    parser.add_argument("--input", type=str, default="just_dd.tsv",
                        help="Path to the input TSV file (default: just_dd.tsv)")
    parser.add_argument("--word", type=str, default="just",
                        help="Word to filter for in the last sentence (default: just)")
    args = parser.parse_args()
    main(input_path=args.input, word=args.word)
