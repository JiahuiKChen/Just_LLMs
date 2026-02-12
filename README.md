# Just_LLMs

Exploration of LLMs' understanding and usage of "just" and other discourse particles.

**particle_removal**
Scripts for measuring how removing a discourse particle in the preceding sentence changes the probability of a followup.

Usage (examples):
1. Compute followup probabilities (raw + per-token log probs):
```bash
python particle_removal/calculate_followup_probabilities.py
```
1. Plot histogram of full-sequence (unconditional) log-prob differences per-token:
```bash
python particle_removal/plot_probabilities.py --results_path results/Meta-Llama-3-8B_noContext_filtered_just_dd_removal.csv
```
1. Plot histogram of conditional (followup) log-prob differences per-token:
```bash
python particle_removal/plot_probabilities.py --results_path results/Meta-Llama-3-8B_noContext_filtered_just_dd_removal.csv --conditional
```


Notes:
1. `calculate_followup_probabilities.py` tokenizes the full string once (fast tokenizers) to avoid boundary-tokenization issues when scoring `P(followup | context + w_word/wo_word)`. It warns if a token straddles the context/target boundary.
1. The script computes an unconditional/full-sequence comparison: `log P(context + w_word + followup)` vs `log P(context + wo_word + followup)` (stored as `log_prob_full_with` and `log_prob_full_without`). `plot_probabilities.py` defaults to plotting the per-token version of this; use `--conditional` to plot the per-token conditional followup difference instead.
1. Per-token normalization is included in the outputs (e.g., `log_prob_full_with_per_token`). Plotting/analysis scripts default to per-token differences.
1. Plotting/analysis scripts abbreviate the model name in auto-generated output filenames to keep them short.
