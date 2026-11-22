"""Analyze the frozen-model likelihood/score-gradient pilot; never trains a model.

Usage: .venv/bin/python experiments/probability_gradient_norm/analyze.py [results.json]
Uncertainty resamples whole prompts, retaining all responses within a prompt.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr


CONDITIONS = ("greedy", "sample_t1_a", "sample_t1_b", "sample_t15", "astra")
LABELS = {
    "greedy": "OLMo greedy",
    "sample_t1_a": "OLMo T=1 (A)",
    "sample_t1_b": "OLMo T=1 (B)",
    "sample_t15": "OLMo T=1.5",
    "astra": "gpt-6-astra",
}
COLORS = dict(zip(CONDITIONS, ("#333333", "#0072B2", "#56B4E9", "#E69F00", "#CC79A7")))


def interval(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return [float(x) for x in np.quantile(a, [0.025, 0.975])] if len(a) else [None, None]


def valid_metric(record, mode):
    m = record.get(mode, {})
    try:
        return (
            float(m["n_tokens"]) > 0
            and np.isfinite(float(m["mean_logprob"]))
            and np.isfinite(float(m["grad_norm_mean"]))
            and float(m["grad_norm_mean"]) > 0
            and np.isfinite(float(m["grad_norm_sum"]))
            and float(m["grad_norm_sum"]) > 0
        )
    except (KeyError, TypeError, ValueError):
        return False


def correlations(rows, mode):
    """Mean NLL versus log score-gradient norm, pooled and prompt-centered."""
    if len(rows) < 3:
        return np.array([np.nan, np.nan])
    x = np.array([-float(r[mode]["mean_logprob"]) for r in rows])
    y = np.log([float(r[mode]["grad_norm_mean"]) for r in rows])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho = float(spearmanr(x, y).statistic)
        groups = defaultdict(list)
        for j, r in enumerate(rows):
            groups[r["id"]].append(j)
        xc, yc = x.copy(), y.copy()
        for indices in groups.values():
            xc[indices] -= x[indices].mean()
            yc[indices] -= y[indices].mean()
        centered_r = float(pearsonr(xc, yc).statistic)
    return np.array([rho, centered_r])


def correlation_summary(rows, mode, rng):
    valid = [r for r in rows if valid_metric(r, mode)]
    grouped = defaultdict(list)
    for r in valid:
        grouped[r["id"]].append(r)
    ids = list(grouped)
    point = correlations(valid, mode)
    bootstrap = []
    if len(ids) >= 2:
        for _ in range(1000):
            sampled = rng.choice(len(ids), size=len(ids), replace=True)
            bootstrap.append(correlations([r for j in sampled for r in grouped[ids[j]]], mode))
    boot = np.asarray(bootstrap).reshape(-1, 2)
    keys = ("pooled_spearman", "within_prompt_centered_pearson")
    return {
        "n_responses": len(valid),
        "n_prompts": len(ids),
        **{
            key: {
                "estimate": float(point[i]) if np.isfinite(point[i]) else None,
                "cluster_bootstrap_95_ci": interval(boot[:, i]),
            }
            for i, key in enumerate(keys)
        },
    }


def describe(rows, mode):
    rows = [r for r in rows if valid_metric(r, mode)]
    if not rows:
        return {"n": 0}
    norms = np.array([r[mode]["grad_norm_mean"] for r in rows], dtype=float)
    lengths = np.array([r[mode]["n_tokens"] for r in rows], dtype=int)
    return {
        "n": len(rows),
        "mean_n_tokens": float(lengths.mean()),
        "min_n_tokens": int(lengths.min()),
        "median_n_tokens": float(np.median(lengths)),
        "max_n_tokens": int(lengths.max()),
        "mean_nll_per_token": float(np.mean([-r[mode]["mean_logprob"] for r in rows])),
        "median_grad_norm_mean": float(np.median(norms)),
        "geometric_mean_grad_norm_mean": float(np.exp(np.log(norms).mean())),
        "median_grad_norm_sum": float(np.median([r[mode]["grad_norm_sum"] for r in rows])),
    }


def paired_summary(complete, rng):
    paired = {}
    for c in CONDITIONS[1:]:
        if not complete:
            paired[c] = {"n_prompts": 0}
            continue
        log_ratio = np.array([
            np.log(g[c]["matched"]["grad_norm_mean"] / g["greedy"]["matched"]["grad_norm_mean"])
            for g in complete.values()
        ])
        delta_nll = np.array([
            g["greedy"]["matched"]["mean_logprob"] - g[c]["matched"]["mean_logprob"]
            for g in complete.values()
        ])
        draw = rng.integers(0, len(log_ratio), size=(2000, len(log_ratio)))
        boot_log = log_ratio[draw].mean(axis=1)
        paired[c] = {
            "n_prompts": len(log_ratio),
            "mean_log_grad_norm_ratio": float(log_ratio.mean()),
            "log_ratio_95_ci": interval(boot_log),
            "geometric_mean_grad_norm_ratio": float(np.exp(log_ratio.mean())),
            "ratio_95_ci": interval(np.exp(boot_log)),
            "mean_delta_nll_per_token": float(delta_nll.mean()),
            "delta_nll_95_ci": interval(delta_nll[draw].mean(axis=1)),
            "fraction_larger_norm_than_greedy": float(np.mean(log_ratio > 0)),
        }
    return paired


def analyze(data):
    records = data.get("records", [])
    if not records:
        raise ValueError("No records found; finish the scoring run before analysis.")
    grouped = defaultdict(dict)
    duplicates = set()
    unknown = []
    for r in records:
        if r.get("condition") not in CONDITIONS or "id" not in r:
            unknown.append(r.get("id"))
            continue
        if r["condition"] in grouped[r["id"]]:
            duplicates.add(r["id"])
        grouped[r["id"]][r["condition"]] = r
    complete = {
        i: g for i, g in grouped.items()
        if i not in duplicates
        and all(c in g and valid_metric(g[c], "matched") for c in CONDITIONS)
        and len({int(g[c]["matched"]["n_tokens"]) for c in CONDITIONS}) == 1
    }
    paired_rows = [g[c] for g in complete.values() for c in CONDITIONS]
    clean = [r for i, g in grouped.items() if i not in duplicates for r in g.values()]
    rng = np.random.default_rng(336)
    summaries = {}
    for c in CONDITIONS:
        rows = [r for r in clean if r["condition"] == c]
        summaries[c] = {
            "full": describe(rows, "full"),
            "matched_complete_clusters": describe([g[c] for g in complete.values()], "matched"),
            "truncated_count": sum(bool(r.get("truncated", False)) for r in rows),
        }
    paired = paired_summary(complete, rng)
    min32 = {i: g for i, g in complete.items() if g["greedy"]["matched"]["n_tokens"] >= 32}
    per_condition_correlations = {}
    for c in CONDITIONS:
        rows = [g[c] for g in complete.values()]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho = float(spearmanr(
                [-r["matched"]["mean_logprob"] for r in rows],
                [r["matched"]["grad_norm_mean"] for r in rows],
            ).statistic) if len(rows) >= 3 else np.nan
        per_condition_correlations[c] = {"n_prompts": len(rows), "spearman": rho if np.isfinite(rho) else None}
    summary = {
        "metadata": data.get("metadata", {}),
        "seed": 336,
        "bootstrap_unit": "prompt (all conditions retained together)",
        "paired_bootstrap_draws": 2000,
        "correlation_bootstrap_draws": 1000,
        "n_complete_prompt_clusters": len(complete),
        "excluded_incomplete_or_invalid_prompt_ids": sorted(set(grouped) - set(complete)),
        "duplicate_prompt_ids": sorted(duplicates),
        "unrecognized_record_ids": unknown,
        "conditions": summaries,
        "paired_matched_vs_greedy": paired,
        "matched_min32_sensitivity": {
            "n_complete_prompt_clusters": len(min32),
            "excluded_short_prefix_prompt_ids": sorted(set(complete) - set(min32)),
            "paired_matched_vs_greedy": paired_summary(min32, rng),
        },
        "matched_per_condition_spearman_descriptive": per_condition_correlations,
        "correlations": {
            "full": correlation_summary(clean, "full", rng),
            "matched_complete_clusters": correlation_summary(paired_rows, "matched", rng),
        },
    }
    return summary, paired_rows


def whitespace_summary(data):
    """Optional post-hoc check, with whole-question paired bootstrap."""
    rows = [r for r in data.get("records", [])
            if valid_metric(r, "astra_leading_space") and valid_metric(r, "greedy")
            and r["astra_leading_space"]["n_tokens"] == r["greedy"]["n_tokens"]]
    result = {"n_prompts": len(rows), "description": data.get("description"),
              "gpu_function_seconds": data.get("gpu_function_seconds"),
              "post_hoc": True}
    if not rows:
        return result
    log_ratio = np.array([np.log(r["astra_leading_space"]["grad_norm_mean"]
                                / r["greedy"]["grad_norm_mean"]) for r in rows])
    delta_nll = np.array([r["greedy"]["mean_logprob"]
                         - r["astra_leading_space"]["mean_logprob"] for r in rows])
    rng = np.random.default_rng(336)
    draw = rng.integers(0, len(rows), size=(2000, len(rows)))
    result.update({
        "geometric_mean_grad_norm_ratio": float(np.exp(log_ratio.mean())),
        "ratio_95_ci": interval(np.exp(log_ratio[draw].mean(axis=1))),
        "mean_delta_nll_per_token": float(delta_nll.mean()),
        "delta_nll_95_ci": interval(delta_nll[draw].mean(axis=1)),
        "bootstrap_draws": 2000,
        "bootstrap_unit": "prompt",
    })
    return result


def plots(summary, rows, destination):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8.5, 5.3), layout="constrained")
    for c in CONDITIONS:
        selected = [r for r in rows if r["condition"] == c]
        ax.scatter(
            [-r["matched"]["mean_logprob"] for r in selected],
            [r["matched"]["grad_norm_mean"] for r in selected],
            color=COLORS[c], label=LABELS[c], s=30, alpha=.75,
        )
    ax.set(xlabel="Mean negative log probability (nats/token; larger = less likely)",
           ylabel="Norm of mean log-probability gradient (log scale)",
           title="Likelihood and gradient magnitude at matched response lengths")
    ax.set_yscale("log")
    ax.grid(alpha=.18)
    ax.legend(frameon=False)
    fig.savefig(destination / "matched_likelihood_vs_gradient.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.5), layout="constrained")
    paired = summary["paired_matched_vs_greedy"]
    for y, c in enumerate(CONDITIONS[1:]):
        r = paired[c]
        if not r.get("n_prompts"):
            continue
        v = r["mean_log_grad_norm_ratio"]
        lo, hi = r["log_ratio_95_ci"]
        ax.errorbar(v, y, xerr=[[v-lo], [hi-v]], fmt="o", color=COLORS[c], capsize=4)
    ax.axvline(0, color="#555555", linestyle="--", linewidth=1)
    ax.set_yticks(range(4), [LABELS[c] for c in CONDITIONS[1:]])
    ax.set(xlabel="Mean log(norm / greedy norm); 0 = equal magnitude",
           title="Paired gradient-norm comparisons: 95% prompt-bootstrap intervals")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.18)
    fig.savefig(destination / "paired_gradient_ratios.png", dpi=180)
    plt.close(fig)


def report(s):
    astra = s["paired_matched_vs_greedy"].get("astra", {})
    sens = s["matched_min32_sensitivity"]
    takeaway = "The input did not contain a complete paired Astra comparison."
    if astra.get("n_prompts"):
        ratio = astra["geometric_mean_grad_norm_ratio"]
        nll = astra["mean_delta_nll_per_token"]
        direction = "lower" if nll > 0 else "higher" if nll < 0 else "equal"
        lo, hi = astra["ratio_95_ci"]
        takeaway = (
            f"On this frozen OLMo checkpoint, gpt-6-astra responses had {direction} likelihood per token "
            f"than greedy OLMo responses (mean NLL difference {nll:+.3f} nats/token). "
            f"Their matched-prefix gradient norms were {ratio:.3f} times greedy norms "
            f"(paired geometric mean; 95% prompt-bootstrap CI [{lo:.3f}, {hi:.3f}]). "
            "This is a measured association in this pilot, not a universal likelihood-to-gradient rule."
        )
        sr = sens["paired_matched_vs_greedy"].get("astra", {})
        if sr.get("n_prompts"):
            same_direction = ((ratio > 1) == (sr["geometric_mean_grad_norm_ratio"] > 1)
                              and (nll > 0) == (sr["mean_delta_nll_per_token"] > 0))
            takeaway += (
                f" Excluding short prefixes {'preserves' if same_direction else 'changes'} the directions "
                f"of these two contrasts (norm ratio {sr['geometric_mean_grad_norm_ratio']:.3f})."
            )
    lines = [
        "# Probability versus gradient norm: frozen OLMo pilot", "",
        takeaway, "",
        f"Analysis uses {s['n_complete_prompt_clusters']} complete prompt clusters for paired comparisons.", "",
        "The measured quantity is the full-parameter Euclidean norm of the response log-probability gradient. "
        "It has no reward/advantage factor. No parameter updates are performed. "
        "Matched responses use a common prefix length within each prompt. The mean log-probability "
        "gradient norm is exactly the summed log-probability gradient norm divided by scored token count "
        "(it is not the average of separate token-gradient norms).", "",
        "| Response source | Gradient norm / greedy (geometric mean, 95% CI) | Mean NLL difference vs greedy (nats/token, 95% CI) |",
        "|---|---:|---:|",
    ]
    for c, r in s["paired_matched_vs_greedy"].items():
        if not r.get("n_prompts"):
            continue
        lo, hi = r["ratio_95_ci"]
        dl, dh = r["delta_nll_95_ci"]
        lines.append(f"| {LABELS[c]} | {r['geometric_mean_grad_norm_ratio']:.3f} [{lo:.3f}, {hi:.3f}] "
                     f"| {r['mean_delta_nll_per_token']:+.3f} [{dl:+.3f}, {dh:+.3f}] |")
    lines += ["", "A positive NLL difference means lower model likelihood per token. "
              "A gradient ratio above one means a larger gradient norm. "
              "Intervals resample whole prompts, preserving paired conditions; there are 2,000 draws.", "",
              "| Measurement | Pooled Spearman (95% CI) | Within-prompt centered Pearson (95% CI) |",
              "|---|---:|---:|"]
    def fmt(r):
        p = r["estimate"]
        lo, hi = r["cluster_bootstrap_95_ci"]
        return "undefined" if p is None or lo is None else f"{p:.3f} [{lo:.3f}, {hi:.3f}]"
    for mode, r in s["correlations"].items():
        lines.append(f"| {mode} | {fmt(r['pooled_spearman'])} | {fmt(r['within_prompt_centered_pearson'])} |")
    lines += ["", "Correlations compare mean NLL with log mean-gradient norm; centered Pearson first "
              "subtracts each prompt's mean from both quantities. Correlation intervals use 1,000 prompt-bootstrap draws. "
              "Pooled association can arise from differences between response sources and does not imply the same "
              "association within each source.", "",
              "| Source | Within-source matched Spearman (descriptive) | Prompts |",
              "|---|---:|---:|"]
    for c, r in s["matched_per_condition_spearman_descriptive"].items():
        rho = "undefined" if r["spearman"] is None else f"{r['spearman']:.3f}"
        lines.append(f"| {LABELS[c]} | {rho} | {r['n_prompts']} |")
    lines += ["", "## Short-prefix sensitivity", "",
              f"Excluding {len(sens['excluded_short_prefix_prompt_ids'])} prompts with matched prefixes shorter than "
              f"32 tokens leaves {sens['n_complete_prompt_clusters']} complete clusters. "
              "The same paired prompt-bootstrap analysis gives:", "",
              "| Source | Gradient norm / greedy (geometric mean, 95% CI) | Mean NLL difference vs greedy (nats/token) |",
              "|---|---:|---:|"]
    for c, r in sens["paired_matched_vs_greedy"].items():
        if not r.get("n_prompts"):
            continue
        lo, hi = r["ratio_95_ci"]
        lines.append(f"| {LABELS[c]} | {r['geometric_mean_grad_norm_ratio']:.3f} [{lo:.3f}, {hi:.3f}] "
                     f"| {r['mean_delta_nll_per_token']:+.3f} |")
    whitespace = s.get("whitespace_sensitivity")
    if whitespace and whitespace.get("n_prompts"):
        lo, hi = whitespace["ratio_95_ci"]
        dl, dh = whitespace["delta_nll_95_ci"]
        lines += ["", "## Post-hoc formatting check", "",
                  "A final audit found that Astra responses lacked the leading space usually present in OLMo "
                  "responses. As a post-hoc sensitivity check, one leading space was added to Astra responses "
                  "and their matched-prefix measurements were recomputed against the same greedy responses. "
                  "The original observations above are unchanged.", "",
                  f"Across {whitespace['n_prompts']} paired prompts, spaced-Astra/greedy norm ratio is "
                  f"**{whitespace['geometric_mean_grad_norm_ratio']:.3f}** (95% CI [{lo:.3f}, {hi:.3f}]); "
                  f"mean NLL difference is **{whitespace['mean_delta_nll_per_token']:+.3f}** nats/token "
                  f"(95% CI [{dl:+.3f}, {dh:+.3f}]). Intervals again resample prompts (2,000 draws). "
                  "This checks one formatting difference, not all teacher/style confounds. "
                  "Raw measurements are in `whitespace_sensitivity.json`."]
    lines += ["", "## Response lengths and compute", "",
              "Lengths count scored response tokens. Matched prefixes are common within each question; "
              "they need not be the same length across different questions.", "",
              "| Source | Full tokens: min / median / max | Matched tokens: min / median / max | Truncated responses / total |",
              "|---|---:|---:|---:|"]
    def lengths(r):
        return (f"{r['min_n_tokens']} / {r['median_n_tokens']:g} / {r['max_n_tokens']}"
                if r.get("n") else "unavailable")
    for c, r in s["conditions"].items():
        lines.append(f"| {LABELS[c]} | {lengths(r['full'])} | {lengths(r['matched_complete_clusters'])} "
                     f"| {r['truncated_count']} / {r['full']['n']} |")
    metadata = s["metadata"]
    if "gpu_function_seconds" in metadata:
        seconds = float(metadata["gpu_function_seconds"])
        count = int(metadata.get("gpu_count", 1))
        lines += ["", f"The scoring run reports **{seconds:.2f} GPU-function seconds** on "
                  f"{count} × {metadata.get('gpu', 'GPU')}: {seconds * count / 3600:.4f} GPU-hours "
                  "of function execution. This figure excludes provisioning and image-build time."]
    if whitespace and whitespace.get("gpu_function_seconds") is not None:
        lines += ["", f"The separate post-hoc formatting check reports "
                  f"**{whitespace['gpu_function_seconds']:.2f} GPU-function seconds**."]
    lines += ["", "Overall accounting, including failed startup, provisioning, and the post-hoc check, "
              "is tracked separately in [compute_budget.json](compute_budget.json). "
              "The primary run's function time alone is not the total budget accounting."]
    lines += ["",
              "## Scope and limitations", "",
              "- This is a small descriptive pilot on the base checkpoint, not an RL-trained policy. "
              "It measures gradient norms, not variance of reward-weighted gradient estimators.",
              "- A different model's response is not assumed to have low OLMo likelihood: the NLL contrasts test that premise.",
              "- No correctness evaluation was performed. Greedy base-model responses can copy the question. "
              "The T=1.5 condition is a stress test; inspected outputs were mainly incoherent "
              f"and {s['conditions']['sample_t15']['truncated_count']}/{s['conditions']['sample_t15']['full']['n']} "
              "responses reached the cap. It should not be read as ordinary high-quality reasoning.",
              "- Greedy responses are deterministic conditional on the prompt/model. The two T=1 draws are distinct samples, "
              "but all response conditions for one question remain one statistical cluster.",
              "- OLMo generation uses BF16 and scoring uses FP32. Both use the same checkpoint, but floating-point "
              "differences introduce a small numerical mismatch between the sampling and scoring policies.",
              "- The gpt-6-astra responses are generated through one Codex subagent batch; its sampling temperature "
              "is not exposed. Prompting/style differences confound a general teacher-versus-student claim.",
              "- Matched response lengths remove a basic token-count confound, but responses still visit different prefixes. "
              "Tokens within a response are dependent, and their parameter gradients can reinforce or cancel.",
              "- Generation/scoring caps and truncation may change the response population. Full-response results "
              "have an additional length confound; prefix results are not complete-answer results.",
              "- These intervals describe resampling the selected prompts, not uncertainty over model checkpoints or teacher seeds. "
              "The exploratory comparisons have no multiple-testing correction.", "",
              "See `summary.json` for all statistics and run metadata, and `per_response.csv` for individual "
              "response measurements. Text and token IDs are preserved in `results.json`. "
              "Plots: `matched_likelihood_vs_gradient.png` and `paired_gradient_ratios.png`.", "",
              "## Reproduce", "",
              "Run from the repository root. The first command starts a new GPU experiment using the saved "
              "prompts and Astra responses; the second repeats the formatting check. The last command "
              "only analyzes existing results locally.", "",
              "```bash",
              ".venv/bin/modal run experiments/probability_gradient_norm/run_modal.py",
              ".venv/bin/modal run experiments/probability_gradient_norm/run_modal.py --whitespace",
              ".venv/bin/python experiments/probability_gradient_norm/analyze.py",
              "```", ""]
    return "\n".join(lines)


def write_csv(records, destination):
    fields = ["id", "condition", "n_tokens", "matched_n_tokens", "truncated"]
    fields += [f"{mode}_{metric}" for mode in ("full", "matched")
               for metric in ("mean_nll", "grad_norm_sum", "grad_norm_mean")]
    with (destination / "per_response.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {"id": record.get("id"), "condition": record.get("condition"),
                   "n_tokens": record.get("full", {}).get("n_tokens", record.get("n_tokens")),
                   "matched_n_tokens": record.get("matched", {}).get("n_tokens"),
                   "truncated": record.get("truncated", False)}
            for mode in ("full", "matched"):
                metrics = record.get(mode, {})
                logprob = metrics.get("mean_logprob")
                row[f"{mode}_mean_nll"] = -float(logprob) if logprob is not None else None
                for metric in ("grad_norm_sum", "grad_norm_mean"):
                    row[f"{mode}_{metric}"] = metrics.get(metric)
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="?", type=Path, default=Path(__file__).with_name("results.json"))
    args = parser.parse_args()
    data = json.loads(args.results.read_text())
    s, rows = analyze(data)
    destination = args.results.parent
    whitespace_path = destination / "whitespace_sensitivity.json"
    if whitespace_path.exists():
        s["whitespace_sensitivity"] = whitespace_summary(json.loads(whitespace_path.read_text()))
    (destination / "summary.json").write_text(json.dumps(s, indent=2, allow_nan=False) + "\n")
    plots(s, rows, destination)
    (destination / "report.md").write_text(report(s))
    write_csv(data["records"], destination)
    print(f"Analyzed {s['n_complete_prompt_clusters']} complete prompt clusters; wrote summary, report and two plots to {destination}")


if __name__ == "__main__":
    main()
