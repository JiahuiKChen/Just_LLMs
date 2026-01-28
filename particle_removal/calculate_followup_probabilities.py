"""
Calculate probabilities of follow-up utterances given conversation context with/without discourse particles.

This script measures how removing a discourse particle (like "just") affects the probability
of the observed follow-up response in a conversation.

EXPERIMENT:
    Compare P(followup | context + w_word) vs P(followup | context + wo_word)
    where:
        - w_word: sentence WITH discourse particle (e.g., "The train just got here")
        - wo_word: sentence WITHOUT discourse particle (e.g., "The train got here")
        - followup: ground-truth response (e.g., "How do you know?")

BASIC USAGE:
    # Default: Uses Qwen2.5-1.5B model WITH context
    python calculate_followup_probabilities.py

    # Run WITHOUT conversation context
    python calculate_followup_probabilities.py --no_context

    # Use a different model
    python calculate_followup_probabilities.py --model_name Qwen/Qwen2.5-7B

    # Custom data path and output
    python calculate_followup_probabilities.py --data_path ../dd_data/my_data.tsv --output_path custom_results.csv

IMPORTANT PARAMETERS:
    --model_name        HuggingFace model to use (default: Qwen/Qwen2.5-1.5B)
    --include_context   Include conversation history (DEFAULT: True)
    --no_context        Exclude conversation history (overrides --include_context)
    --data_path         Path to input TSV file (default: ../dd_data/just_dd.tsv)
    --output_path       Where to save results (default: results/<model>_<context/noContext>_<datafile>_removal.csv)

OUTPUT:
    Tab-separated CSV file with columns:
        - Input: id, word, context, w_word, wo_word, followup
        - Results: log_prob_with_particle, log_prob_without_particle, log_prob_diff, prob_ratio

    log_prob_diff > 0: particle INCREASES followup probability
    log_prob_diff < 0: particle DECREASES followup probability

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
import numpy as np


def load_model(model_name="Qwen/Qwen2.5-1.5B"):
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
        Total log probability of the entire target sequence
    """
    # Combine context and target
    full_text = context_text + target_text

    # Tokenize
    context_tokens = tokenizer.encode(context_text, add_special_tokens=True)
    full_tokens = tokenizer.encode(full_text, add_special_tokens=True)

    # Convert to tensors
    input_ids = torch.tensor([full_tokens]).to(device)

    # Get logits
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    # Calculate log probabilities for the target tokens
    # We want P(target | context), so we look at tokens after the context
    context_len = len(context_tokens)
    target_len = len(full_tokens) - context_len

    if target_len <= 0:
        return 0.0

    # Get log probabilities
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # Sum log probabilities for target tokens
    total_log_prob = 0.0
    # logits[i] predicts token[i+1], so we iterate through positions that predict target tokens
    for i in range(context_len - 1, len(full_tokens) - 1):
        token_id = full_tokens[i + 1]
        token_log_prob = log_probs[0, i, token_id].item()
        total_log_prob += token_log_prob

    # Return total log probability of the entire sequence
    return total_log_prob


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
            if include_context and (pd.isna(context) or not isinstance(context, str)):
                # If context is required but invalid, treat as empty string
                context = ""
        except (KeyError, TypeError) as e:
            print(f"Skipping row {idx}: error reading fields - {e}")
            continue

        # Format the prompts
        prompt_with = format_conversation(context, w_word)
        prompt_without = format_conversation(context, wo_word)

        # Calculate probabilities
        log_prob_with = calculate_log_probability(
            model, tokenizer, prompt_with, followup, device
        )
        log_prob_without = calculate_log_probability(
            model, tokenizer, prompt_without, followup, device
        )

        # Store results
        results.append({
            'id': row['id'],
            'word': row['word'],
            'context': context if include_context else "[NO CONTEXT]",
            'w_word': w_word,
            'wo_word': wo_word,
            'followup': followup,
            'log_prob_with_particle': log_prob_with,
            'log_prob_without_particle': log_prob_without,
            'prob_with_particle': np.exp(log_prob_with),
            'prob_without_particle': np.exp(log_prob_without),
            'log_prob_diff': log_prob_with - log_prob_without,
            'prob_ratio': np.exp(log_prob_with - log_prob_without),
            'include_context': include_context
        })

    # Create results dataframe
    results_df = pd.DataFrame(results)

    # Count how many times particle increased/decreased probability
    increased = (results_df['log_prob_diff'] > 0).sum()
    decreased = (results_df['log_prob_diff'] < 0).sum()
    unchanged = (results_df['log_prob_diff'] == 0).sum()

    # Build summary statistics string
    summary_lines = [
        "=" * 80,
        "SUMMARY STATISTICS",
        "=" * 80,
        f"Total examples processed: {len(results_df)}",
        f"Include context: {include_context}",
        "",
        f"Average log probability WITH particle: {results_df['log_prob_with_particle'].mean():.4f}",
        f"Average log probability WITHOUT particle: {results_df['log_prob_without_particle'].mean():.4f}",
        f"Average log probability difference: {results_df['log_prob_diff'].mean():.4f}",
        f"  (positive = particle increases followup probability)",
        "",
        f"Median log probability difference: {results_df['log_prob_diff'].median():.4f}",
        f"Std dev of log probability difference: {results_df['log_prob_diff'].std():.4f}",
        "",
        f"Particle INCREASED followup probability: {increased} ({increased/len(results_df)*100:.1f}%)",
        f"Particle DECREASED followup probability: {decreased} ({decreased/len(results_df)*100:.1f}%)",
        f"No change: {unchanged}",
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
        default='../dd_data/just_dd.tsv',
        help='Path to the input TSV file'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='Qwen/Qwen2.5-1.5B',
        help='HuggingFace model name (default: Qwen/Qwen2.5-1.5B)'
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
