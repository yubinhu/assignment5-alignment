"""Plot exported ablations offline: uv run python scripts/plot_grpo_ablations.py.

Refresh inputs first with scripts/export_wandb.py:
  --include-run k7qkb5jd
  --name-prefix lr- --output-dir experiments/grpo_learning_rate/wandb_export
  --name-prefix prompt- --output-dir experiments/grpo_prompt_ablation/wandb_export
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from plot_grpo_metrics import METRICS, REPO_ROOT


def load_runs(problem):
    root = REPO_ROOT / "experiments" / problem / "wandb_export"
    runs = []
    for item in json.loads((root / "manifest.json").read_text()):
        if item["state"] != "finished":
            continue
        metadata = json.loads((root / item["id"] / "metadata.json").read_text())
        history = json.loads((root / item["id"] / "history.json").read_text())
        final = [row for row in history if row.get("eval/avg_rewards") is not None]
        if not final or final[-1]["_step"] != 200 or metadata["config"]["steps"] != 200:
            raise ValueError(f"Incomplete 200-step run: {item['id']}")
        runs.append({**metadata, "history": history})
    return runs


def curves(runs, metric):
    points = []
    for run in runs:
        rows = [row for row in run["history"] if row.get(metric) is not None]
        values = {row["_step"]: float(row[metric]) for row in rows}
        if not values or len(values) != len(rows):
            raise ValueError(f"Missing or repeated steps: {run['id']}, {metric}")
        points.append(values)
    steps = sorted(points[0])
    if any(set(p) != set(steps) for p in points):
        raise ValueError(f"Mismatched logged steps for {metric}")
    array = np.array([[p[s] for s in steps] for p in points])
    # W&B may serialize NaN as a string. Preserve unavailable values as gaps,
    # rather than replacing them or averaging a changing subset of seeds.
    array[~np.isfinite(array)] = np.nan
    return steps, array


def mean_ci(array):
    return array.mean(axis=0), 1.96 * array.std(axis=0, ddof=1) / np.sqrt(len(array))


def check_groups(groups, varying_key):
    ignored = {varying_key, "seed", "run_name", "training_device", "inference_gpu"}
    reference = None
    for label, runs in groups:
        seeds = [r["config"]["seed"] for r in runs]
        if len(seeds) < 2 or len(seeds) != len(set(seeds)):
            raise ValueError(f"Need at least two distinct seeds for {label}")
        for run in runs:
            config = {k: v for k, v in run["config"].items() if k not in ignored}
            if reference is None:
                reference = config
            if config != reference:
                raise ValueError(f"Unexpected configuration difference: {run['id']}")


def metric_figure(groups, metric, title, ylabel):
    fig, axes = plt.subplots(1, len(groups), figsize=(13, 4.5), sharex=True, sharey=True)
    has_missing = False
    for ax, (label, runs) in zip(axes, groups):
        steps, array = curves(runs, metric)
        missing = int(np.isnan(array).sum())
        has_missing |= missing > 0
        if missing:
            print(f"{label}: {missing}/{array.size} nonfinite {metric} values shown as gaps")
        mean, half = mean_ci(array)
        ax.fill_between(steps, mean - half, mean + half, color="#607D9C",
                        alpha=0.22, linewidth=0, label="Approx. 95% CI")
        for values in array:
            ax.plot(steps, values, color="#D55E00", alpha=0.25, linewidth=0.9)
        ax.plot(steps, mean, color="#202938", linewidth=2, label="Mean")
        panel_title = f"{label} (n={len(runs)})"
        if missing:
            panel_title += f"\n{missing}/{array.size} values nonfinite"
        ax.set(title=panel_title, xlabel="Training step", xlim=(0, 200))
        ax.grid(axis="y", color="#dfe3e8", linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(ylabel)
    fig.suptitle(title, fontsize=15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.04), ncol=2, frameon=False)
    footnote = "Pointwise interval: mean ± 1.96 × sample SD / √n."
    if has_missing:
        footnote += " Nonfinite values are gaps; mean/CI require all seeds to be finite."
    fig.text(0.5, 0.015, footnote, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.14, 1, 0.95))
    return fig


def save_figure(fig, root, name, pdf):
    fig.savefig(root / f"{name}.png", dpi=180)
    pdf.savefig(fig)
    plt.close(fig)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    baseline = load_runs("grpo_experiments_standard_on_policy")
    lr_runs = load_runs("grpo_learning_rate")
    prompt_runs = load_runs("grpo_prompt_ablation")
    rates = [5e-6, 1e-5, 2e-5]
    lr_groups = [(f"LR {rate:g}", [r for r in baseline + lr_runs
                                 if r["config"]["learning_rate"] == rate]) for rate in rates]
    prompt_groups = [(label, [r for r in baseline + prompt_runs
                             if Path(r["config"]["prompt_path"]).name == filename])
                     for label, filename in [
                         ("Zero-shot", "r1_zero.prompt"),
                         ("Question-only", "question_only.prompt"),
                         ("Three-shot", "r1_zero_three_shot_gsm8k.prompt")]]
    check_groups(lr_groups, "learning_rate")
    check_groups(prompt_groups, "prompt_path")

    root = REPO_ROOT / "experiments/grpo_learning_rate"
    with PdfPages(root / "all_metrics.pdf") as pdf:
        finals = [curves(runs, "eval/avg_rewards")[1][:, -1] for _, runs in lr_groups]
        if any(not np.isfinite(values).all() for values in finals):
            raise ValueError("Nonfinite final validation rewards; cannot summarize as normal outcomes.")
        means, halves = zip(*(mean_ci(values) for values in finals))
        means, halves = np.array(means), np.array(halves)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.fill_between(rates, means - halves, means + halves, color="#607D9C",
                        alpha=0.22, label="Approx. 95% CI")
        ax.plot(rates, means, "o-", color="#202938", linewidth=2, label="Mean")
        for rate, values in zip(rates, finals):
            ax.scatter([rate] * len(values), values, color="#D55E00", alpha=0.25, zorder=3)
        ax.set(xscale="log", xlabel="Learning rate", ylabel="Final validation reward",
               title="Learning-rate ablation · step 200")
        ax.set_xticks(rates, [f"{rate:g}\n(n={len(values)})" for rate, values in zip(rates, finals)])
        ax.minorticks_off()
        ax.grid(axis="y", color="#dfe3e8")
        ax.legend(frameon=False)
        fig.tight_layout()
        save_figure(fig, root, "final_validation_reward", pdf)
        fig = metric_figure(lr_groups, "eval/avg_rewards", "Validation reward by learning rate", "Reward")
        save_figure(fig, root, "validation_total_reward", pdf)

    root = REPO_ROOT / "experiments/grpo_prompt_ablation"
    with PdfPages(root / "all_metrics.pdf") as pdf:
        for metric, title, ylabel, filename in METRICS:
            fig = metric_figure(prompt_groups, metric, title + " · prompt ablation", ylabel)
            save_figure(fig, root, filename, pdf)
    for label, runs in lr_groups + prompt_groups:
        final = curves(runs, "eval/avg_rewards")[1][:, -1]
        print(f"{label}: n={len(runs)}, final rewards={final.tolist()}, mean={final.mean():.6f}")
    print("Saved 2 learning-rate PNGs and 8 prompt-ablation PNGs, plus one PDF per problem.")


if __name__ == "__main__":
    main()
