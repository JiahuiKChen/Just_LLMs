#!/usr/bin/env python3
"""Shared helpers for discourse-particle localization experiments."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

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
    validate_stimulus_row,
)

DEFAULT_MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
DEFAULT_NOT_DATA_PATH = "dd_data/filtered_not_dd.tsv"
DEFAULT_MASK_PERCENTS = (0.5, 1.0, 5.0)
DEFAULT_RANDOM_SEEDS = tuple(range(10))
DEFAULT_PARTICLES = ("just", "only", "not")
DEFAULT_POOL_TOP_K = 500
DEFAULT_POOL_NUM_FOLDS = 2
DEFAULT_RANDOM_MASK_SEEDS = (0, 1, 2, 3)
PARTICLE_DATA_PATHS = {
    "just": "dd_data/filtered_just_dd.tsv",
    "only": "dd_data/filtered_only_dd.tsv",
    "not": DEFAULT_NOT_DATA_PATH,
}


@dataclass(frozen=True)
class EncodedSequenceBatch:
    """Batch of prompt+target token ids with prompt/target boundaries."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt_lens: np.ndarray
    target_lens: np.ndarray


@dataclass(frozen=True)
class PredictionPositionSpec:
    """One relative prediction position searched by activation patching."""

    label: str
    position_order: int
    target_offset: Optional[int]
    use_last: bool = False


def stable_int(*parts: object) -> int:
    """Return a deterministic integer hash for the provided parts."""
    joined = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


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


def default_results_root(model_name: str) -> Path:
    """Return the default root directory for a model's localization runs."""
    return REPO_ROOT / "localization" / "results" / model_short_name(model_name)


def default_results_dir(model_name: str, dataset_name: str = "not", row_limit: Optional[int] = None) -> Path:
    """Return the default output directory for a localization run."""
    base = default_results_root(model_name) / dataset_name
    if row_limit is not None:
        return base / f"rowlimit_{row_limit}"
    return base


def normalize_particle_name(particle: str) -> str:
    """Normalize a particle label used in file paths and filtering."""
    return str(particle).strip().lower()


def particle_data_path(particle: str) -> str:
    """Return the default DailyDialogue TSV path for a particle."""
    normalized = normalize_particle_name(particle)
    if normalized not in PARTICLE_DATA_PATHS:
        raise KeyError(f"Unsupported particle: {particle}")
    return PARTICLE_DATA_PATHS[normalized]


def rename_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Replace pandas-generated Unnamed columns with stable labels."""
    renamed = {}
    unnamed_count = 0
    for column in df.columns:
        if isinstance(column, str) and column.startswith("Unnamed:"):
            renamed[column] = "source_file_index" if unnamed_count == 0 else f"source_file_index_{unnamed_count}"
            unnamed_count += 1
    return df.rename(columns=renamed)


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


def load_source_rows(
    data_path: str,
    row_limit: Optional[int] = None,
    word_filter: Optional[str] = None,
) -> pd.DataFrame:
    """Load and validate source rows from a particle dataset."""
    df = pd.read_csv(data_path, sep="\t")
    df = rename_unnamed_columns(df)
    if "source_row_index" not in df.columns:
        df.insert(0, "source_row_index", range(len(df)))

    if "word" in df.columns and word_filter:
        normalized_filter = normalize_particle_name(word_filter)
        mask = df["word"].fillna("").astype(str).str.strip().str.lower() == normalized_filter
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


def load_particle_source_rows(
    particle: str,
    data_path: Optional[str] = None,
    row_limit: Optional[int] = None,
) -> pd.DataFrame:
    """Load validated DailyDialogue rows for one particle."""
    normalized = normalize_particle_name(particle)
    resolved_data_path = data_path or particle_data_path(normalized)
    return load_source_rows(
        data_path=resolved_data_path,
        row_limit=row_limit,
        word_filter=normalized,
    )


def find_generation_candidates_path(
    model_name: str,
    particle: str,
    results_dir: Optional[Path] = None,
) -> Path:
    """Locate the per-model candidate TSV for one particle."""
    root = Path(results_dir) if results_dir is not None else REPO_ROOT / "generation" / "results"
    pattern = f"{model_short_name(model_name)}_context_{normalize_particle_name(particle)}_generation_candidates.tsv"
    matches = sorted(
        path
        for path in root.rglob(pattern)
        if not path.name.endswith("_generation_candidates_weighted.tsv")
    )
    if not matches:
        raise FileNotFoundError(f"Could not find candidate TSV matching {pattern} under {root}")
    if len(matches) > 1:
        raise FileExistsError(
            f"Found multiple candidate TSVs for {model_name} / {particle}: "
            + ", ".join(str(path) for path in matches)
        )
    return matches[0]


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


def assign_group_folds(group_values: Sequence[object], num_folds: int, seed: int) -> np.ndarray:
    """Assign folds at the group level and broadcast them back to all rows."""
    series = pd.Series(list(group_values))
    if series.empty:
        raise ValueError("Cannot assign group folds to an empty sequence.")
    unique_values = pd.unique(series)
    fold_ids = assign_folds(len(unique_values), num_folds=num_folds, seed=seed)
    fold_map = {group_value: int(fold_id) for group_value, fold_id in zip(unique_values, fold_ids)}
    return series.map(fold_map).to_numpy(dtype=int, copy=False)


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


def get_decoder_layers(model) -> Sequence[torch.nn.Module]:
    """Return decoder block modules for supported causal LM architectures."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    raise ValueError(f"Unsupported model architecture for residual-layer access: {type(model)}")


def get_component_modules(model, component: str) -> Sequence[torch.nn.Module]:
    """Return per-layer modules for component-level activation patching."""
    normalized = normalize_particle_name(component)
    decoder_layers = list(get_decoder_layers(model))
    if normalized == "resid":
        return decoder_layers
    if normalized == "attn":
        attr = "self_attn"
    elif normalized == "mlp":
        attr = "mlp"
    else:
        raise ValueError(f"Unsupported patch component: {component}")

    modules = []
    missing_layers = []
    for layer_index, layer in enumerate(decoder_layers):
        if not hasattr(layer, attr):
            missing_layers.append(layer_index)
            continue
        modules.append(getattr(layer, attr))
    if missing_layers:
        raise ValueError(
            f"Component {normalized} is not available on decoder layers: "
            + ", ".join(str(value) for value in missing_layers)
        )
    return modules


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


def _last_token_index_before_char(tokenizer, text: str, char_end: int) -> int:
    """Return the token index whose token starts before char_end and is last in the prefix."""
    if char_end <= 0:
        raise ValueError("char_end must be positive.")
    if tokenizer.is_fast:
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        token_index = -1
        for idx, (start, end) in enumerate(enc["offset_mapping"]):
            if start == end == 0:
                continue
            if start < char_end:
                token_index = idx
            else:
                break
        if token_index < 0:
            raise ValueError("Could not resolve a token before the requested character boundary.")
        return int(token_index)

    prefix_tokens = tokenizer.encode(text[:char_end], add_special_tokens=False)
    if not prefix_tokens:
        raise ValueError("Could not resolve a token before the requested character boundary.")
    return len(prefix_tokens) - 1


def resolve_utterance_final_positions(
    tokenizer,
    prompts: Sequence[str],
    utterances: Sequence[str],
) -> np.ndarray:
    """Resolve the final token of the particle-bearing utterance inside each prompt."""
    if len(prompts) != len(utterances):
        raise ValueError("prompts and utterances must have identical lengths.")
    positions = np.full((len(prompts), 1), fill_value=-1, dtype=np.int64)
    bos_offset = 1 if tokenizer.bos_token_id is not None else 0
    for row_idx, (prompt, utterance) in enumerate(zip(prompts, utterances)):
        prompt_text = str(prompt)
        utterance_text = str(utterance).strip()
        if not utterance_text:
            raise ValueError(f"Empty utterance for row {row_idx}.")
        start_char = prompt_text.rfind(utterance_text)
        if start_char < 0:
            raw_utterance = str(utterance)
            start_char = prompt_text.rfind(raw_utterance)
            utterance_text = raw_utterance
        if start_char < 0:
            raise ValueError(
                f"Could not find utterance in prompt for row {row_idx}: {utterance_text!r}"
            )
        end_char = start_char + len(utterance_text)
        token_index = _last_token_index_before_char(tokenizer, prompt_text, end_char)
        positions[row_idx, 0] = int(token_index) + bos_offset
    return positions


def build_encoded_sequence_batch(
    tokenizer,
    prompts: Sequence[str],
    targets: Sequence[str],
    device: torch.device,
) -> EncodedSequenceBatch:
    """Tokenize prompt+target pairs into one padded batch."""
    if len(prompts) != len(targets):
        raise ValueError("prompts and targets must have identical lengths.")
    if not prompts:
        raise ValueError("At least one prompt/target pair is required.")

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id or eos_token_id for sequence batching.")

    encoded_rows = [
        encode_prompt_target(tokenizer, prompt_text=prompt, target_text=target)
        for prompt, target in zip(prompts, targets)
    ]
    full_tokens = [row[0] for row in encoded_rows]
    prompt_lens = np.asarray([row[1] for row in encoded_rows], dtype=np.int64)
    target_lens = np.asarray([row[2] for row in encoded_rows], dtype=np.int64)
    max_len = max(len(tokens) for tokens in full_tokens)

    input_ids = torch.full(
        (len(full_tokens), max_len),
        fill_value=pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for row_idx, tokens in enumerate(full_tokens):
        token_tensor = torch.tensor(tokens, dtype=torch.long, device=device)
        input_ids[row_idx, : len(tokens)] = token_tensor
        attention_mask[row_idx, : len(tokens)] = 1

    return EncodedSequenceBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_lens=prompt_lens,
        target_lens=target_lens,
    )


def build_prediction_position_specs(
    patch_scope: str,
    onset_tokens: int,
) -> list[PredictionPositionSpec]:
    """Return the ordered relative prediction positions for one patch scope."""
    if onset_tokens <= 0:
        raise ValueError("onset_tokens must be positive.")

    specs: list[PredictionPositionSpec] = []
    if patch_scope == "prompt_boundary":
        specs.append(PredictionPositionSpec(label="predict_0", position_order=0, target_offset=0))
    elif patch_scope == "utterance_final":
        specs.append(PredictionPositionSpec(label="utterance_final", position_order=0, target_offset=None))
    elif patch_scope == "response_onset":
        for offset in range(onset_tokens):
            specs.append(
                PredictionPositionSpec(
                    label=f"predict_{offset}",
                    position_order=len(specs),
                    target_offset=offset,
                )
            )
    elif patch_scope == "full_final":
        specs.append(PredictionPositionSpec(label="predict_last", position_order=0, target_offset=None, use_last=True))
    elif patch_scope == "hybrid":
        for offset in range(onset_tokens):
            specs.append(
                PredictionPositionSpec(
                    label=f"predict_{offset}",
                    position_order=len(specs),
                    target_offset=offset,
                )
            )
        specs.append(
            PredictionPositionSpec(
                label="predict_last",
                position_order=len(specs),
                target_offset=None,
                use_last=True,
            )
        )
    else:
        raise ValueError(f"Unsupported patch_scope: {patch_scope}")
    return specs


def resolve_prediction_positions(
    prompt_lens: Sequence[int],
    target_lens: Sequence[int],
    position_specs: Sequence[PredictionPositionSpec],
) -> np.ndarray:
    """Resolve ordered relative prediction positions to absolute token indices."""
    prompt_array = np.asarray(prompt_lens, dtype=np.int64)
    target_array = np.asarray(target_lens, dtype=np.int64)
    if prompt_array.shape != target_array.shape:
        raise ValueError("prompt_lens and target_lens must have identical shapes.")

    position_matrix = np.full((len(prompt_array), len(position_specs)), fill_value=-1, dtype=np.int64)
    for row_idx, (prompt_len, target_len) in enumerate(zip(prompt_array, target_array)):
        used_positions: set[int] = set()
        if int(target_len) <= 0:
            continue
        for column_idx, spec in enumerate(position_specs):
            if spec.use_last:
                absolute_position = int(prompt_len) + int(target_len) - 2
            else:
                if spec.target_offset is None:
                    raise ValueError(
                        f"Position {spec.label!r} is not resolvable from prompt/target lengths alone."
                    )
                if spec.target_offset >= int(target_len):
                    continue
                absolute_position = int(prompt_len) - 1 + spec.target_offset
            if absolute_position in used_positions:
                continue
            used_positions.add(absolute_position)
            position_matrix[row_idx, column_idx] = absolute_position
    return position_matrix


def score_target_log_probs_from_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    prompt_lens: Sequence[int],
    target_lens: Sequence[int],
    bos_token_id: Optional[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score each target continuation from model logits."""
    prompt_array = np.asarray(prompt_lens, dtype=np.int64)
    target_array = np.asarray(target_lens, dtype=np.int64)
    if logits.ndim != 3:
        raise ValueError(f"Expected logits with shape [batch, seq, vocab], got {tuple(logits.shape)}")
    if input_ids.ndim != 2:
        raise ValueError(f"Expected input_ids with shape [batch, seq], got {tuple(input_ids.shape)}")
    if logits.shape[:2] != input_ids.shape:
        raise ValueError(
            f"logits and input_ids must agree on [batch, seq], got {tuple(logits.shape[:2])} and {tuple(input_ids.shape)}"
        )
    if len(prompt_array) != logits.shape[0] or len(target_array) != logits.shape[0]:
        raise ValueError("prompt_lens and target_lens must align with the batch size.")

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    totals = np.zeros(logits.shape[0], dtype=np.float64)
    counts = np.zeros(logits.shape[0], dtype=np.int64)
    per_token = np.full(logits.shape[0], np.nan, dtype=np.float64)
    for row_idx, (prompt_len, target_len) in enumerate(zip(prompt_array, target_array)):
        if target_len <= 0:
            continue
        start_position = max(int(prompt_len) - 1, 0)
        end_position = int(prompt_len) + int(target_len) - 1
        total_log_prob = 0.0
        for position in range(start_position, end_position):
            next_token_id = int(input_ids[row_idx, position + 1].item())
            total_log_prob += float(log_probs[row_idx, position, next_token_id].item())
        scored_target_len = int(target_len)
        if bos_token_id is None and int(prompt_len) == 0 and int(target_len) > 0:
            scored_target_len = int(target_len) - 1
        totals[row_idx] = total_log_prob
        counts[row_idx] = scored_target_len
        if scored_target_len > 0:
            per_token[row_idx] = total_log_prob / float(scored_target_len)
    return totals, counts, per_token


def _extract_hidden_from_layer_output(output):
    """Return the tensor-like hidden state from a decoder-layer output."""
    return output[0] if isinstance(output, (tuple, list)) else output


def _replace_hidden_in_layer_output(output, hidden):
    """Rebuild a decoder-layer output after modifying the hidden state."""
    if isinstance(output, tuple):
        return (hidden,) + output[1:]
    if isinstance(output, list):
        return [hidden, *output[1:]]
    return hidden


@contextmanager
def residual_capture_hooks(
    model,
    position_matrix: np.ndarray,
) -> Iterator[list[Optional[torch.Tensor]]]:
    """Capture decoder-layer residual vectors at example-specific token positions."""
    hooks = []
    position_matrix = np.asarray(position_matrix, dtype=np.int64)
    if position_matrix.ndim != 2:
        raise ValueError(f"Expected position_matrix with shape [batch, sites], got {position_matrix.shape}")

    decoder_layers = get_decoder_layers(model)
    batch_size, site_count = position_matrix.shape
    captured_layers: list[Optional[torch.Tensor]] = [None] * len(decoder_layers)
    valid_batch, valid_site = np.nonzero(position_matrix >= 0)
    valid_positions = position_matrix[valid_batch, valid_site]
    batch_index_tensor = None if len(valid_batch) == 0 else torch.tensor(valid_batch, dtype=torch.long)
    site_index_tensor = None if len(valid_site) == 0 else torch.tensor(valid_site, dtype=torch.long)
    position_tensor = None if len(valid_positions) == 0 else torch.tensor(valid_positions, dtype=torch.long)

    def make_hook(layer_index: int):
        nonlocal batch_index_tensor, site_index_tensor, position_tensor

        def hook_fn(_module, _inputs, output):
            nonlocal batch_index_tensor, site_index_tensor, position_tensor
            hidden = _extract_hidden_from_layer_output(output)
            layer_capture = hidden.new_zeros((batch_size, site_count, hidden.shape[-1]))
            if batch_index_tensor is not None:
                if batch_index_tensor.device != hidden.device:
                    batch_index_tensor = batch_index_tensor.to(hidden.device)
                    site_index_tensor = site_index_tensor.to(hidden.device)
                    position_tensor = position_tensor.to(hidden.device)
                gathered = hidden[batch_index_tensor, position_tensor]
                layer_capture[batch_index_tensor, site_index_tensor] = gathered
            captured_layers[layer_index] = layer_capture.detach().cpu()
            return output

        return hook_fn

    try:
        for layer_index, layer in enumerate(decoder_layers):
            hooks.append(layer.register_forward_hook(make_hook(layer_index)))
        yield captured_layers
    finally:
        for hook in hooks:
            hook.remove()


@contextmanager
def component_capture_hooks(
    model,
    component: str,
    position_matrix: np.ndarray,
) -> Iterator[list[Optional[torch.Tensor]]]:
    """Capture per-layer component vectors at example-specific token positions."""
    hooks = []
    position_matrix = np.asarray(position_matrix, dtype=np.int64)
    if position_matrix.ndim != 2:
        raise ValueError(f"Expected position_matrix with shape [batch, sites], got {position_matrix.shape}")

    component_modules = get_component_modules(model, component)
    batch_size, site_count = position_matrix.shape
    captured_layers: list[Optional[torch.Tensor]] = [None] * len(component_modules)
    valid_batch, valid_site = np.nonzero(position_matrix >= 0)
    valid_positions = position_matrix[valid_batch, valid_site]
    batch_index_tensor = None if len(valid_batch) == 0 else torch.tensor(valid_batch, dtype=torch.long)
    site_index_tensor = None if len(valid_site) == 0 else torch.tensor(valid_site, dtype=torch.long)
    position_tensor = None if len(valid_positions) == 0 else torch.tensor(valid_positions, dtype=torch.long)

    def make_hook(layer_index: int):
        nonlocal batch_index_tensor, site_index_tensor, position_tensor

        def hook_fn(_module, _inputs, output):
            nonlocal batch_index_tensor, site_index_tensor, position_tensor
            hidden = _extract_hidden_from_layer_output(output)
            layer_capture = hidden.new_zeros((batch_size, site_count, hidden.shape[-1]))
            if batch_index_tensor is not None:
                if batch_index_tensor.device != hidden.device:
                    batch_index_tensor = batch_index_tensor.to(hidden.device)
                    site_index_tensor = site_index_tensor.to(hidden.device)
                    position_tensor = position_tensor.to(hidden.device)
                gathered = hidden[batch_index_tensor, position_tensor]
                layer_capture[batch_index_tensor, site_index_tensor] = gathered
            captured_layers[layer_index] = layer_capture.detach().cpu()
            return output

        return hook_fn

    try:
        for layer_index, module in enumerate(component_modules):
            hooks.append(module.register_forward_hook(make_hook(layer_index)))
        yield captured_layers
    finally:
        for hook in hooks:
            hook.remove()


def build_residual_patch_plan(
    layer_position_pairs: Sequence[tuple[int, int]],
    target_positions: np.ndarray,
    source_cache: torch.Tensor,
) -> dict[int, list[dict[str, torch.Tensor]]]:
    """Map layer/position site selections to per-layer patch operations."""
    target_positions = np.asarray(target_positions, dtype=np.int64)
    if target_positions.ndim != 2:
        raise ValueError(f"Expected target_positions with shape [batch, sites], got {target_positions.shape}")
    if source_cache.ndim != 4:
        raise ValueError(
            "Expected source_cache with shape [layers, batch, sites, hidden], got "
            f"{tuple(source_cache.shape)}"
        )
    if source_cache.shape[1:3] != target_positions.shape:
        raise ValueError(
            "source_cache and target_positions must agree on [batch, sites], got "
            f"{tuple(source_cache.shape[1:3])} and {tuple(target_positions.shape)}"
        )

    plan: dict[int, list[dict[str, torch.Tensor]]] = {}
    for layer_index, position_index in layer_position_pairs:
        if not (0 <= layer_index < source_cache.shape[0]):
            raise ValueError(f"Layer index {layer_index} is out of range for source_cache.")
        if not (0 <= position_index < target_positions.shape[1]):
            raise ValueError(f"Position index {position_index} is out of range for target_positions.")
        valid_batch = np.flatnonzero(target_positions[:, position_index] >= 0)
        if len(valid_batch) == 0:
            continue
        plan.setdefault(layer_index, []).append(
            {
                "batch_indices": torch.tensor(valid_batch, dtype=torch.long),
                "target_positions": torch.tensor(target_positions[valid_batch, position_index], dtype=torch.long),
                "source_vectors": source_cache[layer_index, valid_batch, position_index].clone(),
            }
        )
    return plan


@contextmanager
def residual_patch_hooks(
    model,
    patch_plan: Optional[dict[int, list[dict[str, torch.Tensor]]]],
) -> Iterator[None]:
    """Replace decoder-layer residual vectors at example-specific token positions."""
    hooks = []
    if not patch_plan:
        yield
        return

    decoder_layers = get_decoder_layers(model)

    def make_hook(specs: list[dict[str, torch.Tensor]]):
        cached_specs = [
            {
                "batch_indices": spec["batch_indices"],
                "target_positions": spec["target_positions"],
                "source_vectors": spec["source_vectors"],
            }
            for spec in specs
        ]

        def hook_fn(_module, _inputs, output):
            hidden = _extract_hidden_from_layer_output(output)
            modified = None
            for spec in cached_specs:
                if spec["batch_indices"].numel() == 0:
                    continue
                if spec["batch_indices"].device != hidden.device:
                    spec["batch_indices"] = spec["batch_indices"].to(hidden.device)
                    spec["target_positions"] = spec["target_positions"].to(hidden.device)
                    spec["source_vectors"] = spec["source_vectors"].to(hidden.device, dtype=hidden.dtype)
                elif spec["source_vectors"].dtype != hidden.dtype:
                    spec["source_vectors"] = spec["source_vectors"].to(dtype=hidden.dtype)
                if modified is None:
                    modified = hidden.clone()
                modified[spec["batch_indices"], spec["target_positions"]] = spec["source_vectors"]
            if modified is None:
                return output
            return _replace_hidden_in_layer_output(output, modified)

        return hook_fn

    try:
        for layer_index, specs in patch_plan.items():
            if not (0 <= layer_index < len(decoder_layers)):
                raise ValueError(f"Layer index {layer_index} is out of range for decoder layers.")
            hooks.append(decoder_layers[layer_index].register_forward_hook(make_hook(specs)))
        yield
    finally:
        for hook in hooks:
            hook.remove()


@contextmanager
def component_patch_hooks(
    model,
    component: str,
    patch_plan: Optional[dict[int, list[dict[str, torch.Tensor]]]],
) -> Iterator[None]:
    """Replace component vectors at example-specific token positions."""
    hooks = []
    if not patch_plan:
        yield
        return

    component_modules = get_component_modules(model, component)

    def make_hook(specs: list[dict[str, torch.Tensor]]):
        cached_specs = [
            {
                "batch_indices": spec["batch_indices"],
                "target_positions": spec["target_positions"],
                "source_vectors": spec["source_vectors"],
            }
            for spec in specs
        ]

        def hook_fn(_module, _inputs, output):
            hidden = _extract_hidden_from_layer_output(output)
            modified = None
            for spec in cached_specs:
                if spec["batch_indices"].numel() == 0:
                    continue
                if spec["batch_indices"].device != hidden.device:
                    spec["batch_indices"] = spec["batch_indices"].to(hidden.device)
                    spec["target_positions"] = spec["target_positions"].to(hidden.device)
                    spec["source_vectors"] = spec["source_vectors"].to(hidden.device, dtype=hidden.dtype)
                elif spec["source_vectors"].dtype != hidden.dtype:
                    spec["source_vectors"] = spec["source_vectors"].to(dtype=hidden.dtype)
                if modified is None:
                    modified = hidden.clone()
                modified[spec["batch_indices"], spec["target_positions"]] = spec["source_vectors"]
            if modified is None:
                return output
            return _replace_hidden_in_layer_output(output, modified)

        return hook_fn

    try:
        for layer_index, specs in patch_plan.items():
            if not (0 <= layer_index < len(component_modules)):
                raise ValueError(f"Layer index {layer_index} is out of range for {component} modules.")
            hooks.append(component_modules[layer_index].register_forward_hook(make_hook(specs)))
        yield
    finally:
        for hook in hooks:
            hook.remove()


def run_sequence_model(
    model,
    batch: EncodedSequenceBatch,
    capture_positions: Optional[np.ndarray] = None,
    patch_plan: Optional[dict[int, list[dict[str, torch.Tensor]]]] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run one batched forward pass with optional residual capture and patching."""
    capture_context = (
        residual_capture_hooks(model, capture_positions)
        if capture_positions is not None
        else nullcontext([])
    )
    with torch.no_grad():
        with capture_context as captured_layers:
            with residual_patch_hooks(model, patch_plan=patch_plan):
                outputs = model(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    use_cache=False,
                )

    capture_tensor = None
    if capture_positions is not None:
        if any(layer_capture is None for layer_capture in captured_layers):
            raise ValueError("Residual capture hooks did not record every decoder layer.")
        capture_tensor = torch.stack([layer_capture for layer_capture in captured_layers if layer_capture is not None], dim=0)
    return outputs.logits, capture_tensor


def run_component_sequence_model(
    model,
    batch: EncodedSequenceBatch,
    component: str,
    capture_positions: Optional[np.ndarray] = None,
    patch_plan: Optional[dict[int, list[dict[str, torch.Tensor]]]] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run a batched forward pass with optional component capture and patching."""
    capture_context = (
        component_capture_hooks(model, component=component, position_matrix=capture_positions)
        if capture_positions is not None
        else nullcontext([])
    )
    with torch.no_grad():
        with capture_context as captured_layers:
            with component_patch_hooks(model, component=component, patch_plan=patch_plan):
                outputs = model(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    use_cache=False,
                )

    capture_tensor = None
    if capture_positions is not None:
        if any(layer_capture is None for layer_capture in captured_layers):
            raise ValueError(f"{component} capture hooks did not record every layer.")
        capture_tensor = torch.stack(
            [layer_capture for layer_capture in captured_layers if layer_capture is not None],
            dim=0,
        )
    return outputs.logits, capture_tensor


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
            hidden = _extract_hidden_from_layer_output(output)
            if hidden.shape[1] <= start_position:
                return output
            if index_tensor.device != hidden.device:
                index_tensor = index_tensor.to(hidden.device)
            modified = hidden.clone()
            modified[:, start_position:, index_tensor] = 0.0
            return _replace_hidden_in_layer_output(output, modified)

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

    input_ids = torch.tensor([full_tokens], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    start_position = max(prompt_len - 1, 0)

    with torch.no_grad():
        with residual_ablation_hooks(model, layer_mask=layer_mask, start_position=start_position):
            logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits

    totals, counts, _ = score_target_log_probs_from_logits(
        logits=logits,
        input_ids=input_ids,
        prompt_lens=[prompt_len],
        target_lens=[target_len],
        bos_token_id=tokenizer.bos_token_id,
    )
    return float(totals[0]), int(counts[0])


def score_target_pair(
    model,
    tokenizer,
    prompt_with: str,
    prompt_without: str,
    target_text: str,
    device: torch.device,
    layer_mask: Optional[np.ndarray] = None,
) -> dict:
    """Score one target under the paired particle-present and particle-removed prompts."""
    log_prob_with, token_count_with = calculate_log_probability_with_ablation(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt_with,
        target_text=target_text,
        device=device,
        layer_mask=layer_mask,
    )
    log_prob_without, token_count_without = calculate_log_probability_with_ablation(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt_without,
        target_text=target_text,
        device=device,
        layer_mask=layer_mask,
    )
    return {
        "log_prob_with": log_prob_with,
        "log_prob_without": log_prob_without,
        "token_count_with": token_count_with,
        "token_count_without": token_count_without,
        "context_advantage": log_prob_with - log_prob_without,
    }


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


def extract_response_onset_hidden_states(
    model,
    tokenizer,
    prompts: Sequence[str],
    targets: Sequence[str],
    batch_size: int = 4,
    onset_tokens: int = 3,
    desc: str = "Onset states",
) -> np.ndarray:
    """Extract mean hidden states over the first few target-token positions."""
    if len(prompts) != len(targets):
        raise ValueError("prompts and targets must have identical lengths.")
    if onset_tokens <= 0:
        raise ValueError("onset_tokens must be positive.")

    device = get_device(model)
    sequence_batches = []
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id or eos_token_id for onset extraction.")

    for start in tqdm(range(0, len(prompts), batch_size), desc=desc):
        batch_prompts = list(prompts[start : start + batch_size])
        batch_targets = list(targets[start : start + batch_size])
        encoded_rows = [
            encode_prompt_target(tokenizer, prompt_text=prompt, target_text=target)
            for prompt, target in zip(batch_prompts, batch_targets)
        ]
        full_tokens = [row[0] for row in encoded_rows]
        prompt_lens = [row[1] for row in encoded_rows]
        target_lens = [row[2] for row in encoded_rows]
        max_len = max(len(tokens) for tokens in full_tokens)

        input_ids = torch.full(
            (len(full_tokens), max_len),
            fill_value=pad_token_id,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros_like(input_ids)
        for row_idx, tokens in enumerate(full_tokens):
            token_tensor = torch.tensor(tokens, dtype=torch.long, device=device)
            input_ids[row_idx, : len(tokens)] = token_tensor
            attention_mask[row_idx, : len(tokens)] = 1

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)

        layer_vectors = []
        for hidden_state in outputs.hidden_states[1:]:
            sample_vectors = []
            for row_idx, (prompt_len, target_len) in enumerate(zip(prompt_lens, target_lens)):
                if target_len <= 0:
                    raise ValueError("All onset-localization targets must contain at least one token.")
                start_pos = prompt_len
                end_pos = min(prompt_len + onset_tokens, prompt_len + target_len)
                pooled = hidden_state[row_idx, start_pos:end_pos].mean(dim=0)
                sample_vectors.append(pooled.float().cpu())
            layer_vectors.append(torch.stack(sample_vectors, dim=0))
        sequence_batches.append(torch.stack(layer_vectors, dim=1).numpy())

    return np.concatenate(sequence_batches, axis=0)


def extract_full_sequence_hidden_states(
    model,
    tokenizer,
    prompts: Sequence[str],
    targets: Sequence[str],
    batch_size: int = 4,
    desc: str = "Sequence states",
) -> np.ndarray:
    """Extract final-token hidden states from full prompt+target sequences."""
    if len(prompts) != len(targets):
        raise ValueError("prompts and targets must have identical lengths (number of examples, NOT string length).")
    device = get_device(model)
    sequence_batches = []
    for start in tqdm(range(0, len(prompts), batch_size), desc=desc):
        batch_prompts = list(prompts[start : start + batch_size])
        batch_targets = list(targets[start : start + batch_size])
        batch_texts = [
            f"{prompt}{target}"
            for prompt, target in zip(batch_prompts, batch_targets)
        ]
        encoded = tokenizer(
            batch_texts,
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
            gathered = hidden_state[
                torch.arange(hidden_state.shape[0], device=hidden_state.device),
                positions,
            ]
            layer_vectors.append(gathered.float().cpu())
        sequence_batches.append(torch.stack(layer_vectors, dim=1).numpy())

    return np.concatenate(sequence_batches, axis=0)


def compute_welch_t_scores(
    states_with: np.ndarray,
    states_without: np.ndarray,
    use_abs: bool = True,
) -> np.ndarray:
    """Compute per-unit Welch t-statistics between two activation sets."""
    with_array = np.asarray(states_with, dtype=np.float64)
    without_array = np.asarray(states_without, dtype=np.float64)
    if with_array.ndim != 3 or without_array.ndim != 3:
        raise ValueError(
            "Expected activation arrays with shape [rows, layers, hidden], got "
            f"{with_array.shape} and {without_array.shape}"
        )
    if with_array.shape[1:] != without_array.shape[1:]:
        raise ValueError(
            "Activation arrays must agree on [layers, hidden], got "
            f"{with_array.shape[1:]} and {without_array.shape[1:]}"
        )
    if with_array.shape[0] <= 0 or without_array.shape[0] <= 0:
        raise ValueError("Welch t-statistics require at least one observation per condition.")

    if use_abs:
        with_array = np.abs(with_array)
        without_array = np.abs(without_array)

    n_with = with_array.shape[0]
    n_without = without_array.shape[0]
    mean_with = with_array.mean(axis=0)
    mean_without = without_array.mean(axis=0)
    var_with = with_array.var(axis=0, ddof=1) if n_with > 1 else np.zeros_like(mean_with)
    var_without = without_array.var(axis=0, ddof=1) if n_without > 1 else np.zeros_like(mean_without)
    denominator = np.sqrt((var_with / max(n_with, 1)) + (var_without / max(n_without, 1)))
    scores = np.zeros_like(mean_with, dtype=np.float64)
    valid = denominator > 0
    scores[valid] = (mean_with[valid] - mean_without[valid]) / denominator[valid]
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    return scores.astype(np.float32)


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


def build_mask_from_directional_scores(
    directional_scores: np.ndarray,
    percent: float,
    require_positive: bool = True,
) -> np.ndarray:
    """Select a fixed-size mask from directional scores, preferring positive units."""
    if percent <= 0:
        raise ValueError("Mask percent must be positive.")
    scores = np.asarray(directional_scores, dtype=np.float64)
    total_units = scores.size
    top_k = max(1, int(math.ceil(total_units * (percent / 100.0))))
    flat_scores = scores.reshape(-1)
    selected: list[int] = []

    if require_positive:
        positive_indices = np.flatnonzero(flat_scores > 0)
        if len(positive_indices) > 0:
            positive_scores = flat_scores[positive_indices]
            take = min(top_k, len(positive_indices))
            chosen_positive = positive_indices[np.argpartition(positive_scores, -take)[-take:]]
            selected.extend(int(index) for index in chosen_positive.tolist())

    if len(selected) < top_k:
        remaining_needed = top_k - len(selected)
        if selected:
            selected_mask = np.zeros(total_units, dtype=bool)
            selected_mask[selected] = True
            candidate_indices = np.flatnonzero(~selected_mask)
        else:
            candidate_indices = np.arange(total_units)
        candidate_scores = flat_scores[candidate_indices]
        extra_indices = candidate_indices[np.argpartition(candidate_scores, -remaining_needed)[-remaining_needed:]]
        selected.extend(int(index) for index in extra_indices.tolist())

    mask = np.zeros(total_units, dtype=bool)
    mask[np.asarray(selected[:top_k], dtype=int)] = True
    return mask.reshape(scores.shape)


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


def summarize_particle_ablation_rows(row_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize per-target ablation effects for generated and gold targets."""
    required = {
        "particle",
        "target_type",
        "mask_percent",
        "ablation_condition",
        "evaluation_id",
        "log_prob_with",
        "log_prob_without",
        "context_advantage",
        "log_prob_with_change_from_none",
        "log_prob_without_change_from_none",
        "context_advantage_change_from_none",
    }
    missing = sorted(required - set(row_df.columns))
    if missing:
        raise KeyError("Missing columns for particle-ablation summary: " + ", ".join(missing))

    rows = []
    grouped = row_df.groupby(
        ["particle", "target_type", "mask_percent", "ablation_condition"],
        dropna=False,
        sort=False,
    )
    for (particle, target_type, mask_percent, condition), condition_df in grouped:
        values_with = pd.to_numeric(condition_df["log_prob_with_change_from_none"], errors="coerce")
        values_without = pd.to_numeric(condition_df["log_prob_without_change_from_none"], errors="coerce")
        values_adv = pd.to_numeric(condition_df["context_advantage_change_from_none"], errors="coerce")
        rows.append(
            {
                "particle": particle,
                "target_type": target_type,
                "mask_percent": mask_percent,
                "ablation_condition": condition,
                "num_rows": int(condition_df["evaluation_id"].nunique()),
                "log_prob_with_mean": float(pd.to_numeric(condition_df["log_prob_with"], errors="coerce").mean()),
                "log_prob_without_mean": float(pd.to_numeric(condition_df["log_prob_without"], errors="coerce").mean()),
                "context_advantage_mean": float(pd.to_numeric(condition_df["context_advantage"], errors="coerce").mean()),
                "log_prob_with_change_from_none_mean": float(values_with.mean()),
                "log_prob_without_change_from_none_mean": float(values_without.mean()),
                "context_advantage_change_from_none_mean": float(values_adv.mean()),
                "prompt_with_decrease_fraction": float((values_with < 0).mean()),
            }
        )

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        return summary_df

    random_mean_rows = []
    random_df = row_df.loc[row_df["ablation_condition"] == "random"].copy()
    if not random_df.empty:
        random_target_means = (
            random_df.groupby(
                ["particle", "target_type", "mask_percent", "evaluation_id"],
                dropna=False,
                sort=False,
            )[[
                "log_prob_with",
                "log_prob_without",
                "context_advantage",
                "log_prob_with_change_from_none",
                "log_prob_without_change_from_none",
                "context_advantage_change_from_none",
            ]]
            .mean()
            .reset_index()
        )
        for (particle, target_type, mask_percent), condition_df in random_target_means.groupby(
            ["particle", "target_type", "mask_percent"],
            dropna=False,
            sort=False,
        ):
            values_with = pd.to_numeric(condition_df["log_prob_with_change_from_none"], errors="coerce")
            values_without = pd.to_numeric(condition_df["log_prob_without_change_from_none"], errors="coerce")
            values_adv = pd.to_numeric(condition_df["context_advantage_change_from_none"], errors="coerce")
            random_mean_rows.append(
                {
                    "particle": particle,
                    "target_type": target_type,
                    "mask_percent": mask_percent,
                    "ablation_condition": "random_mean",
                    "num_rows": int(condition_df["evaluation_id"].nunique()),
                    "log_prob_with_mean": float(pd.to_numeric(condition_df["log_prob_with"], errors="coerce").mean()),
                    "log_prob_without_mean": float(pd.to_numeric(condition_df["log_prob_without"], errors="coerce").mean()),
                    "context_advantage_mean": float(pd.to_numeric(condition_df["context_advantage"], errors="coerce").mean()),
                    "log_prob_with_change_from_none_mean": float(values_with.mean()),
                    "log_prob_without_change_from_none_mean": float(values_without.mean()),
                    "context_advantage_change_from_none_mean": float(values_adv.mean()),
                    "prompt_with_decrease_fraction": float((values_with < 0).mean()),
                }
            )
        summary_df = pd.concat([summary_df, pd.DataFrame(random_mean_rows)], ignore_index=True, sort=False)

    random_lookup = summary_df.loc[summary_df["ablation_condition"] == "random_mean"].set_index(
        ["particle", "target_type", "mask_percent"]
    )
    for metric in (
        "log_prob_with_change_from_none_mean",
        "log_prob_without_change_from_none_mean",
        "context_advantage_change_from_none_mean",
    ):
        comparison_column = metric.replace("_change_from_none_mean", "_localized_minus_random_mean")
        summary_df[comparison_column] = np.nan
        localized_mask = summary_df["ablation_condition"] == "localized"
        for idx in summary_df.loc[localized_mask].index:
            row = summary_df.loc[idx]
            key = (row["particle"], row["target_type"], row["mask_percent"])
            if key not in random_lookup.index:
                continue
            summary_df.at[idx, comparison_column] = row[metric] - random_lookup.loc[key, metric]
    return summary_df
