"""Export GRPO run data using the saved W&B login (no credentials in code).

From the repository root:
    uv run python scripts/export_wandb.py
    uv run python scripts/plot_grpo_metrics.py

Exports full, unsampled history, metadata, and rollout tables. All matching
run states are recorded; the plotting script selects finished runs only.
"""

import argparse
import csv
import json
from pathlib import Path

import wandb


REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project", default="harvey-hobby-org/grpo_experiments_standard_on_policy",
        help="W&B entity/project path",
    )
    parser.add_argument("--name-prefix", default="standard-grpo")
    parser.add_argument(
        "--output-dir", type=Path,
        help="Defaults to experiments/<W&B project name>/wandb_export",
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = REPO_ROOT / "experiments" / args.project.rstrip("/").split("/")[-1] / "wandb_export"
    api = wandb.Api(timeout=60)
    runs = [r for r in api.runs(args.project) if r.name.startswith(args.name_prefix)]
    if not runs:
        raise SystemExit("No matching runs; existing export left unchanged.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for run in runs:
        dest = args.output_dir / run.id
        dest.mkdir(exist_ok=True)
        # Unlike history(), scan_history() retrieves every logged row.
        rows = list(run.scan_history())
        (dest / "history.json").write_text(json.dumps(rows, indent=2))
        scalar_keys = sorted({
            key for row in rows for key, value in row.items()
            if value is None or isinstance(value, (str, int, float, bool))
        })
        with (dest / "history.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=scalar_keys, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    key: value for key, value in row.items()
                    if value is None or isinstance(value, (str, int, float, bool))
                })
        metadata = {
            "id": run.id, "name": run.name, "state": run.state,
            "config": run.config, "summary": dict(run.summary), "url": run.url,
        }
        # W&B wraps nested summary dictionaries in mapping objects.
        (dest / "metadata.json").write_text(json.dumps(metadata, indent=2, default=dict))
        tables = []
        for file in run.files():
            if file.name.startswith("media/table/") and file.name.endswith(".json"):
                file.download(root=str(dest), exist_ok=True)
                tables.append(file.name)
        item = {
            "id": run.id, "name": run.name, "seed": run.config.get("seed"),
            "state": run.state, "rows": len(rows),
            "metric_counts": {
                key: sum(row.get(key) is not None for row in rows)
                for key in scalar_keys if key.startswith(("train/", "eval/"))
            },
            "tables": tables,
        }
        manifest.append(item)
        print(f"{run.id}: {run.name} ({run.state}), {len(rows)} rows, {len(tables)} tables", flush=True)

    # Publish the refreshed manifest only after all downloads succeed.
    temporary = args.output_dir / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(args.output_dir / "manifest.json")
    print(f"Exported {len(manifest)} runs to {args.output_dir}")


if __name__ == "__main__":
    main()
