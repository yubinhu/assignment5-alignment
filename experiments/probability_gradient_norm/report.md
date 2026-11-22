# Probability versus gradient norm: frozen OLMo pilot

On this frozen OLMo checkpoint, gpt-6-astra responses had lower likelihood per token than greedy OLMo responses (mean NLL difference +1.254 nats/token). Their matched-prefix gradient norms were 1.929 times greedy norms (paired geometric mean; 95% prompt-bootstrap CI [1.758, 2.112]). This is a measured association in this pilot, not a universal likelihood-to-gradient rule. Excluding short prefixes preserves the directions of these two contrasts (norm ratio 1.923).

Analysis uses 32 complete prompt clusters for paired comparisons.

The measured quantity is the full-parameter Euclidean norm of the response log-probability gradient. It has no reward/advantage factor. No parameter updates are performed. Matched responses use a common prefix length within each prompt. The mean log-probability gradient norm is exactly the summed log-probability gradient norm divided by scored token count (it is not the average of separate token-gradient norms).

| Response source | Gradient norm / greedy (geometric mean, 95% CI) | Mean NLL difference vs greedy (nats/token, 95% CI) |
|---|---:|---:|
| OLMo T=1 (A) | 1.575 [1.437, 1.720] | +0.945 [+0.770, +1.158] |
| OLMo T=1 (B) | 1.472 [1.360, 1.598] | +0.803 [+0.614, +1.001] |
| OLMo T=1.5 | 2.868 [2.691, 3.060] | +8.756 [+8.252, +9.222] |
| gpt-6-astra | 1.929 [1.758, 2.112] | +1.254 [+1.068, +1.454] |

A positive NLL difference means lower model likelihood per token. A gradient ratio above one means a larger gradient norm. Intervals resample whole prompts, preserving paired conditions; there are 2,000 draws.

| Measurement | Pooled Spearman (95% CI) | Within-prompt centered Pearson (95% CI) |
|---|---:|---:|
| full | 0.566 [0.482, 0.646] | 0.178 [0.083, 0.293] |
| matched_complete_clusters | 0.856 [0.810, 0.901] | 0.805 [0.776, 0.836] |

Correlations compare mean NLL with log mean-gradient norm; centered Pearson first subtracts each prompt's mean from both quantities. Correlation intervals use 1,000 prompt-bootstrap draws. Pooled association can arise from differences between response sources and does not imply the same association within each source.

| Source | Within-source matched Spearman (descriptive) | Prompts |
|---|---:|---:|
| OLMo greedy | 0.773 | 32 |
| OLMo T=1 (A) | 0.655 | 32 |
| OLMo T=1 (B) | 0.777 | 32 |
| OLMo T=1.5 | -0.113 | 32 |
| gpt-6-astra | 0.832 | 32 |

## Short-prefix sensitivity

Excluding 10 prompts with matched prefixes shorter than 32 tokens leaves 22 complete clusters. The same paired prompt-bootstrap analysis gives:

| Source | Gradient norm / greedy (geometric mean, 95% CI) | Mean NLL difference vs greedy (nats/token) |
|---|---:|---:|
| OLMo T=1 (A) | 1.477 [1.319, 1.633] | +0.912 |
| OLMo T=1 (B) | 1.449 [1.342, 1.579] | +0.759 |
| OLMo T=1.5 | 2.761 [2.579, 2.973] | +8.825 |
| gpt-6-astra | 1.923 [1.726, 2.150] | +1.124 |

## Post-hoc formatting check

A final audit found that Astra responses lacked the leading space usually present in OLMo responses. As a post-hoc sensitivity check, one leading space was added to Astra responses and their matched-prefix measurements were recomputed against the same greedy responses. The original observations above are unchanged.

Across 32 paired prompts, spaced-Astra/greedy norm ratio is **1.919** (95% CI [1.751, 2.110]); mean NLL difference is **+1.224** nats/token (95% CI [+1.038, +1.424]). Intervals again resample prompts (2,000 draws). This checks one formatting difference, not all teacher/style confounds. Raw measurements are in `whitespace_sensitivity.json`.

## Response lengths and compute

Lengths count scored response tokens. Matched prefixes are common within each question; they need not be the same length across different questions.

| Source | Full tokens: min / median / max | Matched tokens: min / median / max | Truncated responses / total |
|---|---:|---:|---:|
| OLMo greedy | 35 / 65.5 / 214 | 9 / 39.5 / 88 | 0 / 32 |
| OLMo T=1 (A) | 9 / 87.5 / 256 | 9 / 39.5 / 88 | 3 / 32 |
| OLMo T=1 (B) | 23 / 80.5 / 256 | 9 / 39.5 / 88 | 2 / 32 |
| OLMo T=1.5 | 19 / 256 / 256 | 9 / 39.5 / 88 | 28 / 32 |
| gpt-6-astra | 45 / 74 / 140 | 9 / 39.5 / 88 | 0 / 32 |

The scoring run reports **174.67 GPU-function seconds** on 1 × NVIDIA A100-SXM4-80GB: 0.0485 GPU-hours of function execution. This figure excludes provisioning and image-build time.

The separate post-hoc formatting check reports **9.93 GPU-function seconds**.

Overall accounting, including failed startup, provisioning, and the post-hoc check, is tracked separately in [compute_budget.json](compute_budget.json). The primary run's function time alone is not the total budget accounting.

## Scope and limitations

- This is a small descriptive pilot on the base checkpoint, not an RL-trained policy. It measures gradient norms, not variance of reward-weighted gradient estimators.
- A different model's response is not assumed to have low OLMo likelihood: the NLL contrasts test that premise.
- No correctness evaluation was performed. Greedy base-model responses can copy the question. The T=1.5 condition is a stress test; inspected outputs were mainly incoherent and 28/32 responses reached the cap. It should not be read as ordinary high-quality reasoning.
- Greedy responses are deterministic conditional on the prompt/model. The two T=1 draws are distinct samples, but all response conditions for one question remain one statistical cluster.
- OLMo generation uses BF16 and scoring uses FP32. Both use the same checkpoint, but floating-point differences introduce a small numerical mismatch between the sampling and scoring policies.
- The gpt-6-astra responses are generated through one Codex subagent batch; its sampling temperature is not exposed. Prompting/style differences confound a general teacher-versus-student claim.
- Matched response lengths remove a basic token-count confound, but responses still visit different prefixes. Tokens within a response are dependent, and their parameter gradients can reinforce or cancel.
- Generation/scoring caps and truncation may change the response population. Full-response results have an additional length confound; prefix results are not complete-answer results.
- These intervals describe resampling the selected prompts, not uncertainty over model checkpoints or teacher seeds. The exploratory comparisons have no multiple-testing correction.

See `summary.json` for all statistics and run metadata, and `per_response.csv` for individual response measurements. Text and token IDs are preserved in `results.json`. Plots: `matched_likelihood_vs_gradient.png` and `paired_gradient_ratios.png`.

## Reproduce

Run from the repository root. The first command starts a new GPU experiment using the saved prompts and Astra responses; the second repeats the formatting check. The last command only analyzes existing results locally.

```bash
.venv/bin/modal run experiments/probability_gradient_norm/run_modal.py
.venv/bin/modal run experiments/probability_gradient_norm/run_modal.py --whitespace
.venv/bin/python experiments/probability_gradient_norm/analyze.py
```
