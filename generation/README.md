# Generation Experiment

This experiment samples plausible next-turn responses from two prompt variants:

- `S`: `context + w_word`, the originally spoken utterance with the particle.
- `S'`: `context + wo_word`, the utterance with the particle removed.

The `followup` column from the stimulus files is copied into outputs as metadata
only. It is not included in the generation prompt.

By default, responses are sampled with nucleus sampling at `top_p=0.9`, following
the DailyDialogue sampling choice the user requested from *Information Value:
Measuring Utterance Predictability as Distance from Plausible Alternatives*.

Run the Meta-Llama-3-8B batch across all usable GPUs:

```bash
bash generation/run_llama3_generation_batch.sh
```

The main outputs are written under `generation/results/llama3/`:

- `*_generation_candidates.tsv`: one row per sampled response.
- `*_condition_summary.tsv`: aggregate own-context advantage by dataset and condition.
- `*_item_condition_summary.tsv`: item-level summaries for responses sampled from `S` and `S'`.
- `*_paired_item_summary.tsv`: item-level pairing and overlap between the two sampled sets.
- `*_summary.txt`: compact text summary.

The main evaluation field is `own_context_log_prob_advantage`:

- For responses sampled from `S`: `log p(response | S) - log p(response | S')`.
- For responses sampled from `S'`: `log p(response | S') - log p(response | S)`.

Positive values mean the response is more likely under the context it was sampled
from. `not.tsv` is labeled as `meaning_control`; other words are labeled as
`discourse_particle`.

## EMNLP figure outputs

Create the paper and appendix context-delta visualizations:

```bash
/datastor1/jiahuikchen/miniconda3/envs/just_llms/bin/python generation/plot_generation_context_deltas.py
```

The script reads the eight model result directories from
`run_all_models_generation_batch.sh`, excludes the bad Qwen thinking run, and
writes PDF/SVG/PNG figures plus audit TSVs to `generation/results/figures/`.
