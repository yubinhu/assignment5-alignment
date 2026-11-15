"""Regenerate GRPO figures from locally exported W&B histories.

Run from the repository root:
    uv run python scripts/plot_grpo_metrics.py

No W&B credentials or network access are needed once dependencies are installed.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS = [
    ("train/loss", "Training loss", "Loss", "training_loss"),
    ("train/grad_norm", "Gradient norm", "Gradient norm", "gradient_norm"),
    ("train/avg_token_entropy", "Token entropy", "Entropy (nats)", "token_entropy"),
    ("train/avg_rewards", "Training total reward", "Reward", "training_total_reward"),
    ("train/avg_format_rewards", "Training format reward", "Reward", "training_format_reward"),
    ("eval/avg_rewards", "Validation total reward", "Reward", "validation_total_reward"),
    ("eval/avg_format_rewards", "Validation format reward", "Reward", "validation_format_reward"),
    ("eval/avg_resp_len", "Validation response length", "Tokens", "validation_response_length"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=REPO_ROOT / "experiments/grpo_experiments_standard_on_policy/wandb_export")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "experiments/grpo_experiments_standard_on_policy")
    args = parser.parse_args()
    manifest = json.loads((args.input_dir / "manifest.json").read_text())
    runs = sorted((r for r in manifest if r["state"] == "finished"), key=lambda r: r["seed"])
    if len(runs) < 2:
        raise ValueError("At least two finished runs are needed for a confidence interval.")
    seeds = [r["seed"] for r in runs]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Multiple finished runs for one seed; select one per seed in the manifest.")
    histories = {
        r["seed"]: json.loads((args.input_dir / r["id"] / "history.json").read_text())
        for r in runs
    }
    # Validate all metrics before writing any figures. Missing evaluation rows
    # are normal; compare only steps where the metric was actually logged.
    prepared = {}
    for metric, *_ in METRICS:
        values = []
        for seed in seeds:
            points = {}
            for row in histories[seed]:
                if row.get(metric) is None:
                    continue
                step = row["_step"]
                if step in points:
                    raise ValueError(f"Duplicate step {step}: seed {seed}, {metric}")
                points[step] = float(row[metric])
            if not points or not np.isfinite(list(points.values())).all():
                raise ValueError(f"Missing or nonfinite values: seed {seed}, {metric}")
            values.append(points)
        steps = sorted(values[0])
        if any(set(points) != set(steps) for points in values):
            raise ValueError(f"Unaligned logged steps for {metric}; no interpolation is performed.")
        array = np.array([[points[s] for s in steps] for points in values])
        half_width = 1.96 * array.std(axis=0, ddof=1) / np.sqrt(len(runs))
        prepared[metric] = (steps, array, array.mean(axis=0), half_width)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "pdf.fonttype": 42,
    })
    excluded = [r for r in manifest if r["state"] != "finished"]
    subtitle = f"{len(runs)} completed seeds ({', '.join(map(str, seeds))})"
    if excluded:
        subtitle += "; incomplete runs excluded"
    last_step = max(max(steps) for steps, _, _, _ in prepared.values())

    def draw(ax, metric, title, ylabel):
        steps, array, mean, half_width = prepared[metric]
        ax.fill_between(steps, mean - half_width, mean + half_width,
                        color="#607D9C", alpha=0.22, linewidth=0,
                        label="Approx. 95% CI", zorder=1)
        for run_values in array:
            ax.plot(steps, run_values, color="#D55E00",
                    alpha=0.25, linewidth=0.9, zorder=2)
        ax.plot(steps, mean, color="#202938", linewidth=2,
                label=f"Mean (n={len(runs)})", zorder=3)
        ax.set(title=title, xlabel="Training step", ylabel=ylabel, xlim=(0, last_step))
        ax.grid(axis="y", color="#dfe3e8", linewidth=0.6)
        ax.set_axisbelow(True)
        if ylabel == "Reward":
            # Keep the full normal-approximation interval visible, even when
            # it extends beyond the range of observed rewards.
            ax.set_ylim(min(0, float((mean - half_width).min())),
                        max(1, float((mean + half_width).max())))

    with PdfPages(out / "all_metrics.pdf") as pdf:
        for metric, title, ylabel, filename in METRICS:
            fig, ax = plt.subplots(figsize=(8, 5))
            draw(ax, metric, title, ylabel)
            ax.legend(frameon=False, loc="best")
            fig.text(0.5, 0.015, subtitle, ha="center", fontsize=9, color="#566170")
            fig.tight_layout(rect=(0, 0.05, 1, 1))
            fig.savefig(out / f"{filename}.png", dpi=180)
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Generated {len(METRICS)} PNG plots and all_metrics.pdf in {out}")



if __name__ == "__main__":
    main()
