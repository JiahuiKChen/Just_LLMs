#!/usr/bin/env python3
"""Shared helpers for not-only discourse unit localization experiments."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generation.common import (  # noqa: E402
    build_prompt_pair,
    calculate_log_probability,
    load_model,
    normalize_response,
    sample_unique_responses_nucleus,
    validate_stimulus_row,
)

DEFAULT_MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
DEFAULT_DATA_PATH = "dd_data/filtered_not_dd.tsv"
DEFAULT_HUMAN_NEGATIVES = 5
DEFAULT_GENERATED_NEGATIVES = 3
DEFAULT_MASK_PERCENTS = (0.5, 1.0, 5.0)
DEFAULT_NUM_FOLDS = 5
DEFAULT_RANDOM_SEEDS = tuple(range(10))


def stable_int(*parts: object) -> int:
    """Return a deterministic integer hash for the provided parts."""
    joined = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch RNGs for deterministic runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_mask_percents(raw: object) -> List[float]:
    """Parse mask percentages from CLI input."""
    if raw is None:
        return list(DEFAULT_MASK_PERCENTS)
    if isinstance(raw, (list, tuple)):
        values = [float(value) for value in raw]
    else:
        values = [float(part.strip()) for part in str(raw).split(",") if part.strip()]
    if not values:
        raise ValueError("At least one mask percent is required.")
    return values


def parse_random_seeds(raw: object) -> List[int]:
    """Parse random seeds from CLI input."""
    if raw is None:
        return list(DEFAULT_RANDOM_SEEDS)
    if isinstance(raw, (list, tuple)):
        values = [int(value) for value in raw]
    else:
        values = [int(part.strip()) for part in str(raw).split(",") if part.strip()]
    if not values:
        raise ValueError("At least one random seed is required.")
    return values


def model_short_name(model_name: str) -> str:
    """Return the model basename used in output paths."""
    return model_name.split("/")[-1]


def default_results_dir(model_name: str, dataset_name: str = "not", row_limit: Optional[int] = None) -> Path:
    """Return the default output directory for a localization run."""
    base = REPO_ROOT / "localization" / "results" / model_short_name(model_name) / dataset_name
    if row_limit is not None:
        return base / f"rowlimit_{row_limit}"
    return base


def rename_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Replace pandas-generated Unnamed columns with stable labels."""
    renamed = {}
    unnamed_count = 0
    for column in df.columns:
        if isinstance(column, str) and column.startswith("Unnamed:"):
            renamed[column] = "source_file_index" if unnamed_count == 0 else f"source_file_index_{unnamed_count}"
            unnamed_count += 1
    return df.rename(columns=renamed)


def followup_token_length(text: str) -> int:
    """Return a rough whitespace-token count for matching distractors."""
    return len([part for part in str(text).strip().split() if part])


def final_punctuation(text: str) -> str:
    """Return the final punctuation mark if one exists."""
    stripped = str(text).rstrip()
    if not stripped:
        return ""
    return stripped[-1] if stripped[-1] in ".?!,;:" else ""


def json_dumps(value: object) -> str:
    """Serialize JSON using compact separators."""
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def json_loads(value: object) -> object:
    """Deserialize JSON values stored inside TSV fields."""
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    if value is None:
        return []
    if isinstance(value, (dict, tuple)):
        return value
    text = str(value).strip()
    if not text:
        return []
    return json.loads(text)


def load_source_rows(data_path: str, row_limit: Optional[int] = None, word_filter: str = "not") -> pd.DataFrame:
    """Load and validate source rows from the not-control dataset."""
    df = pd.read_csv(data_path, sep="\t")
    df = rename_unnamed_columns(df)
    if "source_row_index" not in df.columns:
        df.insert(0, "source_row_index", range(len(df)))

    if "word" in df.columns and word_filter:
        mask = df["word"].fillna("").astype(str).str.strip().str.lower() == word_filter
        df = df.loc[mask].copy()

    valid_rows = []
    for _, row in df.iterrows():
        try:
            context, w_word, wo_word = validate_stimulus_row(row, include_context=True)
        except ValueError:
            continue

        followup = row.get("followup")
        if pd.isna(followup) or not isinstance(followup, str) or not followup.strip():
            continue

        record = row.to_dict()
        record["context"] = context
        record["w_word"] = w_word
        record["wo_word"] = wo_word
        record["followup"] = str(followup).strip()
        valid_rows.append(record)

    pilot_df = pd.DataFrame(valid_rows)
    if row_limit is not None:
        pilot_df = pilot_df.iloc[:row_limit].copy()

    pilot_df.reset_index(drop=True, inplace=True)
    pilot_df.insert(0, "pilot_row_index", range(len(pilot_df)))
    return pilot_df


def attach_human_negative_followups(
    df: pd.DataFrame,
    num_negatives: int = DEFAULT_HUMAN_NEGATIVES,
    seed: int = 0,
) -> pd.DataFrame:
    """Attach deterministic human distractor followups to each row."""
    rows = []
    for row in df.itertuples(index=False):
        target_followup = str(row.followup).strip()
        target_norm = normalize_response(target_followup)
        target_len = followup_token_length(target_followup)
        target_punct = final_punctuation(target_followup)

        candidates = []
        for other in df.itertuples(index=False):
            if other.pilot_row_index == row.pilot_row_index:
                continue
            other_followup = str(other.followup).strip()
            other_norm = normalize_response(other_followup)
            if not other_followup or other_norm == target_norm:
                continue
            punct_match = int(final_punctuation(other_followup) == target_punct)
            length_diff = abs(followup_token_length(other_followup) - target_len)
            tie_break = stable_int(seed, row.pilot_row_index, other.pilot_row_index, other_norm)
            candidates.append(
                (
                    -punct_match,
                    length_diff,
                    tie_break,
                    int(other.pilot_row_index),
                    int(other.source_row_index),
                    other_followup,
                    other_norm,
                )
            )

        selected_followups = []
        selected_pilot_indices = []
        selected_source_indices = []
        seen_norms = set()
        for _, _, _, pilot_index, source_index, followup, followup_norm in sorted(candidates):
            if followup_norm in seen_norms:
                continue
            seen_norms.add(followup_norm)
            selected_followups.append(followup)
            selected_pilot_indices.append(pilot_index)
            selected_source_indices.append(source_index)
            if len(selected_followups) >= num_negatives:
                break

        record = row._asdict()
        record["human_negative_followups"] = json_dumps(selected_followups)
        record["human_negative_pilot_row_indices"] = json_dumps(selected_pilot_indices)
        record["human_negative_source_row_indices"] = json_dumps(selected_source_indices)
        record["human_negative_count"] = len(selected_followups)
        record["followup_token_length"] = target_len
        record["followup_final_punctuation"] = target_punct
        rows.append(record)

    return pd.DataFrame(rows)


def prepare_not_pilot_dataset(
    data_path: str,
    output_path: Optional[Path] = None,
    human_negatives: int = DEFAULT_HUMAN_NEGATIVES,
    seed: int = 0,
    row_limit: Optional[int] = None,
) -> pd.DataFrame:
    """Build the processed not-only pilot dataset and optionally save it."""
    pilot_df = load_source_rows(data_path=data_path, row_limit=row_limit, word_filter="not")
    pilot_df = attach_human_negative_followups(pilot_df, num_negatives=human_negatives, seed=seed)
    pilot_df["build_seed"] = seed
    pilot_df["human_negative_target"] = human_negatives
    pilot_df["row_limit"] = row_limit if row_limit is not None else ""
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pilot_df.to_csv(output_path, sep="\t", index=False)
    return pilot_df


def assign_folds(num_rows: int, num_folds: int, seed: int) -> np.ndarray:
    """Assign deterministic cross-validation folds to rows."""
    if num_rows <= 0:
        raise ValueError("Cannot assign folds to an empty dataset.")
    num_folds = max(1, min(num_folds, num_rows))
    indices = np.arange(num_rows)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(indices)
    fold_ids = np.empty(num_rows, dtype=int)
    for fold_id, subset in enumerate(np.array_split(shuffled, num_folds)):
        fold_ids[subset] = fold_id
    return fold_ids


def build_prompt_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Construct paired prompts for each row."""
    prompts_with = []
    prompts_without = []
    next_speakers = []
    for row in df.itertuples(index=False):
        prompt_pair = build_prompt_pair(row.context, row.w_word, row.wo_word)
        prompts_with.append(prompt_pair.prompt_p)
        prompts_without.append(prompt_pair.prompt_p_prime)
        next_speakers.append(prompt_pair.next_speaker)
    result = df.copy()
    result["prompt_with"] = prompts_with
    result["prompt_without"] = prompts_without
    result["next_speaker"] = next_speakers
    return result


def score_targets_for_prompt(
    model,
    tokenizer,
    prompt_text: str,
    targets: Sequence[str],
    device: torch.device,
    layer_mask: Optional[np.ndarray] = None,
) -> List[tuple[float, int]]:
    """Score multiple target strings under the same prompt."""
    scores = []
    for target_text in targets:
        score = calculate_log_probability_with_ablation(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            target_text=target_text,
            device=device,
            layer_mask=layer_mask,
        )
        scores.append(score)
    return scores


def mean_or_nan(values: Sequence[float]) -> float:
    """Return the mean of values or NaN if the sequence is empty."""
    if not values:
        return float("nan")
    return float(np.mean(values))


def compute_margin(gold_log_prob: float, negative_log_probs: Sequence[float]) -> float:
    """Return the gold-vs-negatives log-prob margin."""
    negative_mean = mean_or_nan(negative_log_probs)
    if math.isnan(negative_mean):
        return float("nan")
    return gold_log_prob - negative_mean


def compute_row_metrics(
    model,
    tokenizer,
    row: pd.Series,
    device: torch.device,
    layer_mask: Optional[np.ndarray] = None,
    generated_negatives_with: Optional[Sequence[str]] = None,
    generated_negatives_without: Optional[Sequence[str]] = None,
) -> dict:
    """Compute gold and margin metrics for a single row."""
    prompt_with = row["prompt_with"]
    prompt_without = row["prompt_without"]
    gold_followup = row["followup"]
    human_negative_followups = list(json_loads(row["human_negative_followups"]))

    gold_with, gold_with_len = calculate_log_probability_with_ablation(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt_with,
        target_text=gold_followup,
        device=device,
        layer_mask=layer_mask,
    )
    gold_without, gold_without_len = calculate_log_probability_with_ablation(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt_without,
        target_text=gold_followup,
        device=device,
        layer_mask=layer_mask,
    )

    human_scores_with = score_targets_for_prompt(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt_with,
        targets=human_negative_followups,
        device=device,
        layer_mask=layer_mask,
    )
    human_scores_without = score_targets_for_prompt(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt_without,
        targets=human_negative_followups,
        device=device,
        layer_mask=layer_mask,
    )

    human_negative_log_probs_with = [score for score, _ in human_scores_with]
    human_negative_log_probs_without = [score for score, _ in human_scores_without]
    margin_human_with = compute_margin(gold_with, human_negative_log_probs_with)
    margin_human_without = compute_margin(gold_without, human_negative_log_probs_without)

    row_metrics = {
        "gold_log_prob_with": gold_with,
        "gold_log_prob_without": gold_without,
        "gold_token_count_with": gold_with_len,
        "gold_token_count_without": gold_without_len,
        "gold_delta": gold_with - gold_without,
        "margin_human_with": margin_human_with,
        "margin_human_without": margin_human_without,
        "effect_delta": margin_human_with - margin_human_without,
        "human_negative_log_probs_with": json_dumps(human_negative_log_probs_with),
        "human_negative_log_probs_without": json_dumps(human_negative_log_probs_without),
    }

    if generated_negatives_with is not None or generated_negatives_without is not None:
        negatives_with = list(human_negative_followups)
        negatives_without = list(human_negative_followups)
        if generated_negatives_with:
            negatives_with.extend(generated_negatives_with)
        if generated_negatives_without:
            negatives_without.extend(generated_negatives_without)

        hybrid_scores_with = score_targets_for_prompt(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_with,
            targets=negatives_with,
            device=device,
            layer_mask=layer_mask,
        )
        hybrid_scores_without = score_targets_for_prompt(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_without,
            targets=negatives_without,
            device=device,
            layer_mask=layer_mask,
        )
        hybrid_negative_log_probs_with = [score for score, _ in hybrid_scores_with]
        hybrid_negative_log_probs_without = [score for score, _ in hybrid_scores_without]
        margin_hybrid_with = compute_margin(gold_with, hybrid_negative_log_probs_with)
        margin_hybrid_without = compute_margin(gold_without, hybrid_negative_log_probs_without)
        row_metrics.update(
            {
                "generated_negative_followups_with": json_dumps(list(generated_negatives_with or [])),
                "generated_negative_followups_without": json_dumps(list(generated_negatives_without or [])),
                "margin_hybrid_with": margin_hybrid_with,
                "margin_hybrid_without": margin_hybrid_without,
                "effect_delta_hybrid": margin_hybrid_with - margin_hybrid_without,
                "hybrid_negative_log_probs_with": json_dumps(hybrid_negative_log_probs_with),
                "hybrid_negative_log_probs_without": json_dumps(hybrid_negative_log_probs_without),
            }
        )

    return row_metrics


def get_decoder_layers(model) -> Sequence[torch.nn.Module]:
    """Return decoder block modules for supported causal LM architectures."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    raise ValueError(f"Unsupported model architecture for residual-layer access: {type(model)}")


def get_hidden_size(model) -> int:
    """Return the decoder hidden size from the model config."""
    for attr in ("hidden_size", "n_embd", "d_model"):
        if hasattr(model.config, attr):
            return int(getattr(model.config, attr))
    raise ValueError("Could not infer hidden size from model config.")


def get_device(model) -> torch.device:
    """Return a device suitable for input tensors."""
    return next(model.parameters()).device


def encode_prompt_target(
    tokenizer,
    prompt_text: str,
    target_text: str,
) -> tuple[List[int], int, int]:
    """Tokenize prompt+target once and return ids, prompt length, and target length."""
    full_text = prompt_text + target_text
    if tokenizer.is_fast:
        enc = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
        full_tokens = enc["input_ids"]
        offsets = enc["offset_mapping"]
        prompt_char_len = len(prompt_text)
        prompt_token_count = 0
        for start, end in offsets:
            if start == end == 0:
                continue
            if end <= prompt_char_len:
                prompt_token_count += 1
            else:
                break
    else:
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
        target_tokens = tokenizer.encode(target_text, add_special_tokens=False)
        full_tokens = prompt_tokens + target_tokens
        prompt_token_count = len(prompt_tokens)

    if tokenizer.bos_token_id is not None:
        full_tokens = [tokenizer.bos_token_id] + full_tokens
        prompt_len = prompt_token_count + 1
    else:
        prompt_len = prompt_token_count

    target_len = len(full_tokens) - prompt_len
    return full_tokens, prompt_len, target_len


@contextmanager
def residual_ablation_hooks(
    model,
    layer_mask: Optional[np.ndarray],
    start_position: int,
) -> Iterator[None]:
    """Register forward hooks that zero selected residual dimensions."""
    hooks = []
    if layer_mask is None:
        yield
        return

    layer_mask = np.asarray(layer_mask, dtype=bool)
    decoder_layers = get_decoder_layers(model)
    if layer_mask.shape[0] != len(decoder_layers):
        raise ValueError(
            f"Layer-mask shape {layer_mask.shape} does not match decoder layer count {len(decoder_layers)}."
        )

    def make_hook(unit_indices: np.ndarray):
        index_tensor = None if len(unit_indices) == 0 else torch.tensor(unit_indices, dtype=torch.long)

        def hook_fn(_module, _inputs, output):
            nonlocal index_tensor
            if index_tensor is None or start_position < 0:
                return output
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            if hidden.shape[1] <= start_position:
                return output
            if index_tensor.device != hidden.device:
                index_tensor = index_tensor.to(hidden.device)
            modified = hidden.clone()
            modified[:, start_position:, index_tensor] = 0.0
            if isinstance(output, tuple):
                return (modified,) + output[1:]
            if isinstance(output, list):
                return [modified, *output[1:]]
            return modified

        return hook_fn

    try:
        for layer, mask in zip(decoder_layers, layer_mask):
            unit_indices = np.flatnonzero(mask.astype(bool))
            if len(unit_indices) == 0:
                continue
            hooks.append(layer.register_forward_hook(make_hook(unit_indices)))
        yield
    finally:
        for hook in hooks:
            hook.remove()


def calculate_log_probability_with_ablation(
    model,
    tokenizer,
    prompt_text: str,
    target_text: str,
    device: torch.device,
    layer_mask: Optional[np.ndarray] = None,
) -> tuple[float, int]:
    """Calculate log P(target | prompt) with an optional residual-unit ablation."""
    if layer_mask is None:
        return calculate_log_probability(model, tokenizer, prompt_text, target_text, device)

    full_tokens, prompt_len, target_len = encode_prompt_target(tokenizer, prompt_text, target_text)
    if target_len <= 0:
        return 0.0, 0

    input_ids = torch.tensor([full_tokens], device=device)
    start_position = max(prompt_len - 1, 0)

    with torch.no_grad():
        with residual_ablation_hooks(model, layer_mask=layer_mask, start_position=start_position):
            logits = model(input_ids=input_ids).logits

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    total_log_prob = 0.0
    end_idx = len(full_tokens) - 1
    for position in range(start_position, end_idx):
        next_token_id = full_tokens[position + 1]
        total_log_prob += log_probs[0, position, next_token_id].item()

    scored_target_len = target_len
    if tokenizer.bos_token_id is None and prompt_len == 0 and target_len > 0:
        scored_target_len = target_len - 1
    return total_log_prob, scored_target_len


def extract_prompt_hidden_states(
    model,
    tokenizer,
    prompts: Sequence[str],
    batch_size: int = 4,
) -> np.ndarray:
    """Extract final-token hidden states from every decoder block for prompt-only inputs."""
    device = get_device(model)
    prompt_batches = []
    for start in tqdm(range(0, len(prompts), batch_size), desc="Prompt states"):
        batch_prompts = list(prompts[start : start + batch_size])
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(encoded["input_ids"], device=device)
        final_positions = attention_mask.sum(dim=1) - 1

        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True, use_cache=False)

        layer_vectors = []
        for hidden_state in outputs.hidden_states[1:]:
            positions = final_positions.to(hidden_state.device)
            gathered = hidden_state[torch.arange(hidden_state.shape[0], device=hidden_state.device), positions]
            layer_vectors.append(gathered.float().cpu())
        prompt_batches.append(torch.stack(layer_vectors, dim=1).numpy())

    return np.concatenate(prompt_batches, axis=0)


def compute_signed_correlations(delta_states: np.ndarray, effect_delta: np.ndarray) -> np.ndarray:
    """Compute per-unit Pearson correlations between prompt deltas and effect deltas."""
    if delta_states.ndim != 3:
        raise ValueError(f"Expected delta_states with shape [rows, layers, hidden], got {delta_states.shape}")
    flat_states = delta_states.reshape(delta_states.shape[0], -1).astype(np.float64)
    y = np.asarray(effect_delta, dtype=np.float64)
    y_centered = y - np.nanmean(y)
    y_var = np.sum(y_centered**2)
    if not np.isfinite(y_var) or y_var <= 0:
        return np.zeros(delta_states.shape[1:], dtype=np.float32)

    x_centered = flat_states - np.nanmean(flat_states, axis=0, keepdims=True)
    numerator = x_centered.T @ y_centered
    denominator = np.sqrt(np.sum(x_centered**2, axis=0) * y_var)
    correlations = np.zeros_like(numerator, dtype=np.float64)
    valid = denominator > 0
    correlations[valid] = numerator[valid] / denominator[valid]
    correlations = np.nan_to_num(correlations, nan=0.0, posinf=0.0, neginf=0.0)
    return correlations.reshape(delta_states.shape[1:]).astype(np.float32)


def build_mask_from_scores(signed_scores: np.ndarray, percent: float) -> np.ndarray:
    """Select the top percentage of units by absolute score."""
    if percent <= 0:
        raise ValueError("Mask percent must be positive.")
    abs_scores = np.abs(np.asarray(signed_scores, dtype=np.float64))
    total_units = abs_scores.size
    top_k = max(1, int(math.ceil(total_units * (percent / 100.0))))
    flat_scores = abs_scores.reshape(-1)
    top_indices = np.argpartition(flat_scores, -top_k)[-top_k:]
    mask = np.zeros(total_units, dtype=bool)
    mask[top_indices] = True
    return mask.reshape(abs_scores.shape)


def sample_random_mask_from_complement(mask: np.ndarray, seed: int) -> np.ndarray:
    """Sample a same-size random mask from units outside the learned mask when possible."""
    mask = np.asarray(mask, dtype=bool)
    total_units = mask.size
    num_selected = int(mask.sum())
    available = np.flatnonzero(~mask.reshape(-1))
    if len(available) < num_selected:
        available = np.arange(total_units)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(available, size=num_selected, replace=False)
    random_mask = np.zeros(total_units, dtype=bool)
    random_mask[chosen] = True
    return random_mask.reshape(mask.shape)


def mask_percent_slug(percent: float) -> str:
    """Convert a mask percent to a filesystem-friendly slug."""
    return str(percent).replace(".", "p")


def expected_random_pairwise_intersection(total_units: int, selected_units: int) -> float:
    """Expected pairwise overlap count for two random same-size masks."""
    return (selected_units * selected_units) / float(total_units)


def expected_random_pairwise_jaccard(total_units: int, selected_units: int) -> float:
    """Expected pairwise Jaccard overlap for two random same-size masks."""
    intersection = expected_random_pairwise_intersection(total_units, selected_units)
    union = (2 * selected_units) - intersection
    if union <= 0:
        return float("nan")
    return intersection / union


def pairwise_overlap_stats(masks: Sequence[np.ndarray], percent: float) -> pd.DataFrame:
    """Compute pairwise overlap diagnostics for a list of fold masks."""
    if not masks:
        raise ValueError("At least one mask is required.")
    flattened = [np.asarray(mask, dtype=bool).reshape(-1) for mask in masks]
    total_units = flattened[0].size
    selected_units = int(flattened[0].sum())
    rows = []
    for i, left in enumerate(flattened):
        for j, right in enumerate(flattened):
            intersection = int(np.logical_and(left, right).sum())
            union = int(np.logical_or(left, right).sum())
            rows.append(
                {
                    "mask_percent": percent,
                    "fold_i": i,
                    "fold_j": j,
                    "intersection_count": intersection,
                    "jaccard": intersection / union if union else float("nan"),
                    "selected_units": selected_units,
                    "total_units": total_units,
                    "expected_random_intersection": expected_random_pairwise_intersection(
                        total_units, selected_units
                    ),
                    "expected_random_jaccard": expected_random_pairwise_jaccard(total_units, selected_units),
                }
            )
    return pd.DataFrame(rows)


def plot_overlap_heatmap(overlap_df: pd.DataFrame, value_column: str, output_path: Path, title: str) -> None:
    """Plot a fold-overlap heatmap from pairwise overlap stats."""
    import matplotlib.pyplot as plt

    pivot = overlap_df.pivot(index="fold_i", columns="fold_j", values=value_column)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(pivot.values, cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Fold j")
    ax.set_ylabel("Fold i")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    for row_idx in range(pivot.shape[0]):
        for col_idx in range(pivot.shape[1]):
            ax.text(
                col_idx,
                row_idx,
                f"{pivot.iloc[row_idx, col_idx]:.3f}",
                ha="center",
                va="center",
                color="white" if pivot.iloc[row_idx, col_idx] < pivot.values.max() * 0.65 else "black",
                fontsize=8,
            )
    fig.colorbar(image, ax=ax, shrink=0.85)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_layer_counts(masks: Sequence[np.ndarray], percent: float, output_path: Path) -> None:
    """Plot the mean number of selected units per layer across folds."""
    import matplotlib.pyplot as plt

    mask_array = np.stack([np.asarray(mask, dtype=bool) for mask in masks], axis=0)
    mean_counts = mask_array.sum(axis=2).mean(axis=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.arange(len(mean_counts)), mean_counts, color="steelblue")
    ax.set_title(f"Mean Selected Units per Layer ({percent:g}%)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean selected units")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_before_after_histogram(
    df: pd.DataFrame,
    metric_column: str,
    baseline_condition: str,
    ablated_condition: str,
    output_path: Path,
    title: str,
) -> None:
    """Plot before/after histograms for one metric and one ablation condition."""
    import matplotlib.pyplot as plt

    baseline = df.loc[df["ablation_condition"] == baseline_condition, metric_column].dropna().to_numpy()
    ablated = df.loc[df["ablation_condition"] == ablated_condition, metric_column].dropna().to_numpy()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(baseline, bins=40, alpha=0.6, label=baseline_condition, color="gray", edgecolor="black")
    ax.hist(ablated, bins=40, alpha=0.6, label=ablated_condition, color="darkorange", edgecolor="black")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel(metric_column)
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generated_negative_cache_path(run_dir: Path) -> Path:
    """Return the default cache path for generated distractors."""
    return run_dir / "generated_negative_cache.tsv"


def build_generated_negative_cache(
    df: pd.DataFrame,
    model,
    tokenizer,
    output_path: Path,
    generated_negatives: int = DEFAULT_GENERATED_NEGATIVES,
    top_p: float = 0.9,
    temperature: float = 1.0,
    sample_batch_size: int = 16,
    max_sample_draws: int = 100,
    max_new_tokens: int = 64,
    base_seed: int = 20260506,
) -> pd.DataFrame:
    """Sample and cache generated distractor followups for auxiliary evaluation."""
    device = get_device(model)
    rows = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Generated distractors"):
        excluded = {normalize_response(str(row.followup).strip())}
        excluded.update(normalize_response(text) for text in json_loads(row.human_negative_followups))
        prompt_pair = build_prompt_pair(row.context, row.w_word, row.wo_word)

        prompt_specs = [
            ("with", prompt_pair.prompt_p),
            ("without", prompt_pair.prompt_p_prime),
        ]
        record = {
            "pilot_row_index": int(row.pilot_row_index),
            "source_row_index": int(row.source_row_index),
        }
        for label, prompt in prompt_specs:
            seed = stable_int(base_seed, row.pilot_row_index, label)
            samples = sample_unique_responses_nucleus(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                samples_per_condition=generated_negatives,
                top_p=top_p,
                temperature=temperature,
                sample_batch_size=sample_batch_size,
                max_sample_draws=max_sample_draws,
                max_new_tokens=max_new_tokens,
                seed=seed,
            )
            selected = []
            seen_norms = set(excluded)
            for sample in samples:
                response = sample["response"]
                response_norm = normalize_response(response)
                if not response or response_norm in seen_norms:
                    continue
                seen_norms.add(response_norm)
                selected.append(response)
                if len(selected) >= generated_negatives:
                    break
            record[f"generated_negative_followups_{label}"] = json_dumps(selected)
            record[f"generated_negative_count_{label}"] = len(selected)
        rows.append(record)

    cache_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_df.to_csv(output_path, sep="\t", index=False)
    return cache_df


def merge_generated_negative_cache(df: pd.DataFrame, cache_df: pd.DataFrame) -> pd.DataFrame:
    """Attach generated negative columns to the processed pilot dataset."""
    merged = df.merge(cache_df, on=["pilot_row_index", "source_row_index"], how="left")
    for column in (
        "generated_negative_followups_with",
        "generated_negative_followups_without",
    ):
        if column not in merged.columns:
            merged[column] = json_dumps([])
        else:
            merged[column] = merged[column].fillna(json_dumps([]))
    return merged


def summarize_metric_frame(df: pd.DataFrame, metric_columns: Sequence[str]) -> dict:
    """Summarize means and sign accuracies for a set of metric columns."""
    summary = {}
    for column in metric_columns:
        values = pd.to_numeric(df[column], errors="coerce")
        summary[f"{column}_mean"] = float(values.mean())
        summary[f"{column}_sign_accuracy"] = float((values > 0).mean())
    return summary
