"""
Calculate probabilities of follow-up utterances given conversation context with/without discourse particles.

This script measures how removing a discourse particle (like "just") affects the probability
of the observed follow-up response in a conversation.

EXPERIMENT:
    Compare P(followup | context + w_word) vs P(followup | context + wo_word)
    and compare full-sequence likelihoods:
    P(context + w_word + followup) vs P(context + wo_word + followup)
    where:
        - w_word: sentence WITH discourse particle (e.g., "The train just got here")
        - wo_word: sentence WITHOUT discourse particle (e.g., "The train got here")
        - followup: ground-truth response (e.g., "How do you know?")

BASIC USAGE:
    # Default: Uses Meta-Llama-3-8B model WITH context
    python calculate_followup_probabilities.py

    # Run WITHOUT conversation context
    python calculate_followup_probabilities.py --no_context

    # Use a different model
    python calculate_followup_probabilities.py --model_name Qwen/Qwen2.5-7B

    # Custom data path and output
    python calculate_followup_probabilities.py --data_path ../dd_data/my_data.tsv --output_path custom_results.csv

IMPORTANT PARAMETERS:
    --model_name        HuggingFace model to use (default: meta-llama/Meta-Llama-3-8B)
    --include_context   Include conversation history (DEFAULT: True)
    --no_context        Exclude conversation history (overrides --include_context)
    --data_path         Path to input TSV file (default: ../dd_data/filtered_just_dd.tsv)
    --output_path       Where to save results (default: results/<model>_<context/noContext>_<datafile>_removal.csv)

OUTPUT:
    Tab-separated CSV file with columns:
        - Input: id, word, context, w_word, wo_word, followup
        - Results:
            log_prob_w_word, log_prob_wo_word
            log_prob_followup_with, log_prob_followup_without
            log_prob_full_with, log_prob_full_without

    The raw followup log prob difference is computed as:
        log_prob_followup_with - log_prob_followup_without
    (positive = particle increases followup probability)

    The raw full-sequence log prob difference is computed as:
        log_prob_full_with - log_prob_full_without
    (positive = particle increases full-sequence probability)

CONVERSATION FORMAT:
    Conversations are formatted with speaker labels:
        Speaker 1: Hello
        Speaker 2: Hi there
        Speaker 1: How are you?

    This is more natural for LLMs than the raw __eou__ markers in the data.
"""

import argparse
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


def load_model(model_name="meta-llama/Meta-Llama-3-8B"):
    """Load the language model and tokenizer."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    model.eval()

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def _capitalize_first_alpha(text: str) -> str:
    """Capitalize the first alphabetic character, preserving any leading punctuation."""
    if not text:
        return text
    for i, ch in enumerate(text):
        if ch.isalpha():
            return text[:i] + ch.upper() + text[i + 1:]
    return text


def calculate_log_probability(model, tokenizer, context_text, target_text, device):
    """
    Calculate the log probability of target_text given context_text.

    Args:
        model: The language model
        tokenizer: The tokenizer
        context_text: The conditioning text (e.g., "The train just got here.")
        target_text: The text to score (e.g., "How do you know?")
        device: The device to run on

    Returns:
        (total_log_prob, target_token_count)
    """
    # Combine context and target
    full_text = context_text + target_text

    # Tokenize the full text once to avoid boundary-tokenization issues.
    if tokenizer.is_fast:
        enc = tokenizer(
            full_text,
            add_special_tokens=False,
            return_offsets_mapping=True
        )
        full_tokens = enc["input_ids"]
        offsets = enc["offset_mapping"]
        context_char_len = len(context_text)
        context_token_count = 0
        straddles_boundary = False
        for (start, end) in offsets:
            # Some fast tokenizers may return (0, 0) for special tokens; ignore.
            if start == end == 0:
                continue
            if end <= context_char_len:
                context_token_count += 1
            else:
                if start < context_char_len < end:
                    straddles_boundary = True
                break
        if straddles_boundary:
            print("Warning: context/target boundary splits a token; scoring may be approximate.")
    else:
        # Fallback: tokenize separately (may be slightly imprecise at the boundary).
        context_tokens = tokenizer.encode(context_text, add_special_tokens=False)
        target_tokens = tokenizer.encode(target_text, add_special_tokens=False)
        full_tokens = context_tokens + target_tokens
        context_token_count = len(context_tokens)

    # Optionally prepend BOS if the tokenizer defines it.
    if tokenizer.bos_token_id is not None:
        full_tokens = [tokenizer.bos_token_id] + full_tokens
        context_len = context_token_count + 1
    else:
        context_len = context_token_count

    target_len = len(full_tokens) - context_len
    if target_len <= 0:
        return 0.0, 0

    # Convert to tensors
    input_ids = torch.tensor([full_tokens]).to(device)

    # Get logits
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    # Get log probabilities
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # Sum log probabilities for target tokens using masked-label scoring.
    total_log_prob = 0.0
    # logits[i] predicts token[i+1], so score positions where the NEXT token is in target.
    start_idx = max(context_len - 1, 0)
    end_idx = len(full_tokens) - 1
    for i in range(start_idx, end_idx):
        next_token_id = full_tokens[i + 1]
        token_log_prob = log_probs[0, i, next_token_id].item()
        total_log_prob += token_log_prob

    scored_target_len = target_len
    if tokenizer.bos_token_id is None and context_len == 0 and target_len > 0:
        # Without a BOS, the model cannot score the very first token.
        scored_target_len = target_len - 1
        print("Warning: no BOS token; first target token not scored.")

    return total_log_prob, scored_target_len


def format_conversation(context, utterance):
    """
    Format the conversation context and utterance with speaker labels.

    Converts from __eou__ markers to newline-separated speaker labels:
        "Hello__eou__Hi there__eou__How are you?"
        becomes:
        "Speaker 1: Hello\nSpeaker 2: Hi there\nSpeaker 1: How are you?\n"

    Args:
        context: Conversation history with __eou__ markers
        utterance: The utterance to add

    Returns:
        Formatted conversation string with speaker labels
    """
    if not context:
        # No context - just return the utterance with a speaker label
        return f"Speaker 1: {utterance}\n"

    # Parse the context into turns
    turns = context.split("__eou__")
    turns = [t.strip() for t in turns if t.strip()]

    # Add the current utterance
    turns.append(utterance)

    # Format with alternating speaker labels (assuming 2-person conversation)
    formatted_turns = []
    for i, turn in enumerate(turns):
        speaker = "Speaker 1" if i % 2 == 0 else "Speaker 2"
        formatted_turns.append(f"{speaker}: {turn}")

    return "\n".join(formatted_turns) + "\n"


def format_context_only(context):
    """
    Format just the conversation context without any utterance.

    Args:
        context: Conversation history with __eou__ markers

    Returns:
        Formatted context string with speaker labels, or empty string if no context
    """
    if not context:
        return ""

    # Parse the context into turns
    turns = context.split("__eou__")
    turns = [t.strip() for t in turns if t.strip()]

    if not turns:
        return ""

    # Format with alternating speaker labels
    formatted_turns = []
    for i, turn in enumerate(turns):
        speaker = "Speaker 1" if i % 2 == 0 else "Speaker 2"
        formatted_turns.append(f"{speaker}: {turn}")

    return "\n".join(formatted_turns) + "\n"


def process_dataset(data_path, model, tokenizer, include_context=True, output_path=None):
    """
    Process the dataset and calculate probabilities.

    Args:
        data_path: Path to the TSV file
        model: The language model
        tokenizer: The tokenizer
        include_context: Whether to include conversation context
        output_path: Path to save results (optional)

    Returns:
        DataFrame with results
    """
    # Load data
    df = pd.read_csv(data_path, sep='\t')
    print(f"Loaded {len(df)} examples from {data_path}")

    # Get device
    device = next(model.parameters()).device

    # Store results
    results = []

    # Process each row
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        # Validate required fields - skip row if any are invalid
        try:
            w_word = row['w_word']
            wo_word = row['wo_word']
            followup = row['followup']
            context = row['context'] if include_context else ""

            # Check if required fields are valid strings
            if pd.isna(w_word) or not isinstance(w_word, str) or w_word.strip() == "":
                print(f"Skipping row {idx}: invalid w_word")
                continue
            if pd.isna(wo_word) or not isinstance(wo_word, str) or wo_word.strip() == "":
                print(f"Skipping row {idx}: invalid wo_word")
                continue
            if pd.isna(followup) or not isinstance(followup, str) or followup.strip() == "":
                print(f"Skipping row {idx}: invalid or empty followup")
                continue
            # Capitalize first alphabetic character of the utterance variants.
            # This avoids lowercased sentence starts after particle removal.
            w_word = _capitalize_first_alpha(w_word)
            wo_word = _capitalize_first_alpha(wo_word)
            if include_context and (pd.isna(context) or not isinstance(context, str)):
                # If context is required but invalid, treat as empty string
                context = ""
        except (KeyError, TypeError) as e:
            print(f"Skipping row {idx}: error reading fields - {e}")
            continue

        # Format the prompts (context + utterance)
        prompt_with = format_conversation(context, w_word)
        prompt_without = format_conversation(context, wo_word)

        # Format context only (for sentence logprob calculation)
        context_only = format_context_only(context)

        # Determine speaker indices based on number of context turns
        if context:
            context_turns = [t.strip() for t in context.split("__eou__") if t.strip()]
            num_context_turns = len(context_turns)
        else:
            num_context_turns = 0

        # Utterance speaker (w_word/wo_word) comes after context
        utterance_speaker_idx = num_context_turns
        utterance_speaker = "Speaker 1" if utterance_speaker_idx % 2 == 0 else "Speaker 2"

        # Followup speaker comes after utterance
        followup_speaker_idx = num_context_turns + 1
        followup_speaker = "Speaker 1" if followup_speaker_idx % 2 == 0 else "Speaker 2"

        # Format utterances and followup with speaker labels
        w_word_with_label = f"{utterance_speaker}: {w_word}\n"
        wo_word_with_label = f"{utterance_speaker}: {wo_word}\n"
        followup_with_label = f"{followup_speaker}: {followup}"

        # Calculate logprobs of the sentences (w_word and wo_word given context)
        log_prob_w_word, _ = calculate_log_probability(
            model, tokenizer, context_only, w_word_with_label, device
        )
        log_prob_wo_word, _ = calculate_log_probability(
            model, tokenizer, context_only, wo_word_with_label, device
        )

        # Calculate logprobs of followup given context + sentence
        log_prob_followup_with, followup_tokens_with = calculate_log_probability(
            model, tokenizer, prompt_with, followup_with_label, device
        )
        log_prob_followup_without, followup_tokens_without = calculate_log_probability(
            model, tokenizer, prompt_without, followup_with_label, device
        )
        if followup_tokens_with != followup_tokens_without:
            print(
                f"Warning: followup token counts differ at row {idx} "
                f"({followup_tokens_with} vs {followup_tokens_without})"
            )

        # Calculate logprobs of full sequence (context + sentence + followup)
        full_with = prompt_with + followup_with_label
        full_without = prompt_without + followup_with_label
        log_prob_full_with, full_tokens_with = calculate_log_probability(
            model, tokenizer, "", full_with, device
        )
        log_prob_full_without, full_tokens_without = calculate_log_probability(
            model, tokenizer, "", full_without, device
        )
        # Token counts can differ between w_word and wo_word variants; no warning needed.

        # Store results
        results.append({
            'id': row['id'],
            'word': row['word'],
            'context': context if include_context else "[NO CONTEXT]",
            'w_word': w_word,
            'wo_word': wo_word,
            'followup': followup,
            # Raw sentence logprobs (w_word and wo_word given context)
            'log_prob_w_word': log_prob_w_word,
            'log_prob_wo_word': log_prob_wo_word,
            # Raw followup logprobs
            'log_prob_followup_with': log_prob_followup_with,
            'log_prob_followup_without': log_prob_followup_without,
            # Raw full-sequence logprobs (context + sentence + followup)
            'log_prob_full_with': log_prob_full_with,
            'log_prob_full_without': log_prob_full_without,
            'include_context': include_context
        })

    # Create results dataframe
    results_df = pd.DataFrame(results)

    # Raw followup log probability difference (WITH - WITHOUT)
    raw_diff = results_df['log_prob_followup_with'] - results_df['log_prob_followup_without']
    # Raw full-sequence log probability difference (WITH - WITHOUT)
    full_diff = results_df['log_prob_full_with'] - results_df['log_prob_full_without']

    # Count how many times particle increased/decreased probability (raw)
    increased = (raw_diff > 0).sum()
    decreased = (raw_diff < 0).sum()
    unchanged = (raw_diff == 0).sum()
    full_increased = (full_diff > 0).sum()
    full_decreased = (full_diff < 0).sum()
    full_unchanged = (full_diff == 0).sum()

    # Build summary statistics string
    summary_lines = [
        "=" * 80,
        "SUMMARY STATISTICS",
        "=" * 80,
        f"Total examples processed: {len(results_df)}",
        f"Include context: {include_context}",
        "",
        "--- RAW FOLLOWUP LOG PROBABILITIES ---",
        f"Average log probability WITH particle: {results_df['log_prob_followup_with'].mean():.4f}",
        f"Average log probability WITHOUT particle: {results_df['log_prob_followup_without'].mean():.4f}",
        f"Average log probability difference: {raw_diff.mean():.4f}",
        f"  (positive = particle increases followup probability)",
        "",
        f"Median log probability difference: {raw_diff.median():.4f}",
        f"Std dev of log probability difference: {raw_diff.std():.4f}",
        "",
        f"Particle INCREASED followup probability: {increased} ({increased/len(results_df)*100:.1f}%)",
        f"Particle DECREASED followup probability: {decreased} ({decreased/len(results_df)*100:.1f}%)",
        f"No change: {unchanged}",
        "",
        "--- FULL-SEQUENCE LOG PROBABILITIES ---",
        f"Average log probability WITH particle: {results_df['log_prob_full_with'].mean():.4f}",
        f"Average log probability WITHOUT particle: {results_df['log_prob_full_without'].mean():.4f}",
        f"Average log probability difference: {full_diff.mean():.4f}",
        f"  (positive = particle increases full-sequence probability)",
        "",
        f"Median log probability difference: {full_diff.median():.4f}",
        f"Std dev of log probability difference: {full_diff.std():.4f}",
        "",
        f"Particle INCREASED full-sequence probability: {full_increased} ({full_increased/len(results_df)*100:.1f}%)",
        f"Particle DECREASED full-sequence probability: {full_decreased} ({full_decreased/len(results_df)*100:.1f}%)",
        f"No change: {full_unchanged}",
        "=" * 80,
    ]
    summary = "\n".join(summary_lines)

    # Print summary to stdout
    print("\n" + summary + "\n")

    # Save results if output path specified
    if output_path:
        results_df.to_csv(output_path, sep='\t', index=False)
        print(f"Results saved to: {output_path}")

        # Save summary statistics to a text file with the same name
        summary_path = output_path.rsplit('.', 1)[0] + '.txt'
        with open(summary_path, 'w') as f:
            f.write(summary + "\n")
        print(f"Summary saved to: {summary_path}")

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Calculate followup probabilities with/without discourse particles"
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='../dd_data/filtered_just_dd.tsv',
        help='Path to the input TSV file'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='meta-llama/Meta-Llama-3-8B',
        help='HuggingFace model name (default: meta-llama/Meta-Llama-3-8B)'
    )
    parser.add_argument(
        '--include_context',
        action='store_true',
        default=True,
        help='Include conversation context in probability calculation'
    )
    parser.add_argument(
        '--no_context',
        action='store_true',
        help='Do NOT include conversation context (overrides --include_context)'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Path to save results CSV (default: results/<model>_<context/noContext>_<datafile>_removal.csv)'
    )

    args = parser.parse_args()

    # Determine whether to include context
    include_context = not args.no_context if args.no_context else args.include_context

    # Auto-generate output path if not specified
    if args.output_path is None:
        # Extract model name after "/" (e.g., "Qwen/Qwen2.5-1.5B" -> "Qwen2.5-1.5B")
        if '/' in args.model_name:
            model_name_short = args.model_name.split('/')[-1]
        else:
            model_name_short = args.model_name

        context_str = "context" if include_context else "noContext"
        # Extract base name from data file (e.g., "../dd_data/just_dd.tsv" -> "just_dd")
        data_basename = os.path.splitext(os.path.basename(args.data_path))[0]
        os.makedirs("results", exist_ok=True)
        args.output_path = os.path.join("results", f"{model_name_short}_{context_str}_{data_basename}_removal.csv")

    # Load model
    model, tokenizer = load_model(args.model_name)

    # Process dataset
    results_df = process_dataset(
        args.data_path,
        model,
        tokenizer,
        include_context=include_context,
        output_path=args.output_path
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
