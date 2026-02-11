# Just_LLMs

Exploration of LLMs' understanding and usage of "just" and other discourse particles.

**particle_removal**
Scripts for measuring how removing a discourse particle in the preceding sentence changes the probability of a followup.

Usage (examples):
1. Compute followup probabilities (raw log probs only):
```bash
python particle_removal/calculate_followup_probabilities.py
```
1. Plot histogram of full-sequence (unconditional) log-prob differences:
```bash
python particle_removal/plot_probabilities.py --results_path results/Meta-Llama-3-8B_noContext_filtered_just_dd_removal.csv
```
1. Plot histogram of conditional (followup) log-prob differences:
```bash
python particle_removal/plot_probabilities.py --results_path results/Meta-Llama-3-8B_noContext_filtered_just_dd_removal.csv --conditional
```
1. Find worst decreases (raw diff):
```bash
python particle_removal/find_worst_decreases.py --results_path results/Meta-Llama-3-8B_noContext_filtered_just_dd_removal.csv
```
1. Stratified plot by `log_prob_wo_word` (raw diff):
```bash
python particle_removal/plot_stratified_probabilities.py --results_path results/Meta-Llama-3-8B_noContext_filtered_just_dd_removal.csv
```

Notes:
1. `calculate_followup_probabilities.py` now tokenizes the full string once (fast tokenizers) to avoid boundary-tokenization issues when scoring `P(followup | context + w_word/wo_word)`. It warns if a token straddles the context/target boundary.
1. The script also computes an unconditional/full-sequence comparison: `log P(context + w_word + followup)` vs `log P(context + wo_word + followup)` (stored as `log_prob_full_with` and `log_prob_full_without`). `plot_probabilities.py` defaults to plotting this; use `--conditional` to plot the conditional followup difference instead.
1. Per-token normalization has been removed; all analysis uses raw log-probability differences by default.
1. Plotting/analysis scripts abbreviate the model name in auto-generated output filenames to keep them short.
