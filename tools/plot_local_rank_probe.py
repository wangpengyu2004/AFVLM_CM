"""Plot recorded rank-response curves and paired gains; never trains or loads a model."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

METRICS = {"cls": "accuracy", "caption": "CIDEr", "vqa": "VQA_accuracy"}
LABELS = {"cls": "Classification accuracy (%)", "caption": "Caption CIDEr",
          "vqa": "VQA answer accuracy (%)"}


def plot_results(run_root: Path, output: Path, epoch: int | None = None) -> list[str]:
    import matplotlib.pyplot as plt

    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "completed" or any(job["status"] != "completed"
                                                for job in manifest["jobs"]):
        raise ValueError(
            "Complete the selected sweep before plotting; partial/failed rows are not means"
        )
    epoch = int(manifest["local_epochs"]) if epoch is None else epoch
    if not 1 <= epoch <= int(manifest["local_epochs"]):
        raise ValueError("Selected epoch is outside the recorded training budget")
    path = run_root / "metrics_long.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    scores = {}
    for row in rows:
        if (int(row["epoch"]) == epoch and row["primary"] == "True"
                and row["metric"] == METRICS[row["task"]] and row["job_status"] == "completed"):
            key = row["task"], row["client"], int(row["seed"]), int(row["lora_rank"])
            if key in scores:
                raise ValueError(f"Duplicate primary score: {key}")
            scores[key] = float(row["value"])
    groups = defaultdict(lambda: {"clients": set(), "seeds": set(), "ranks": set()})
    for job in manifest["jobs"]:
        group = groups[job["task"]]
        group["clients"].add(job["client"])
        group["seeds"].add(job["seed"])
        group["ranks"].add(job["lora_rank"])
    for task, group in groups.items():
        if len(group["ranks"]) < 2:
            raise ValueError(f"Rank curves require at least two ranks for {task}")
        for client in group["clients"]:
            for seed in group["seeds"]:
                for rank in sorted(group["ranks"]):
                    if (task, client, seed, rank) not in scores:
                        raise ValueError(f"Incomplete paired matrix for {task}/{client}/seed{seed}")
    output.mkdir(parents=True, exist_ok=True)
    tasks = [task for task in METRICS if task in groups]
    if not tasks:
        raise ValueError("No primary rank results")
    figure, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 4), squeeze=False)
    for axis, task in zip(axes[0], tasks, strict=True):
        clients, seeds = sorted(groups[task]["clients"]), sorted(groups[task]["seeds"])
        ranks = sorted(groups[task]["ranks"])
        for client in clients:
            axis.plot(ranks, [statistics.fmean(scores[task, client, seed, rank]
                       for seed in seeds) for rank in ranks], "o--", linewidth=1,
                       alpha=0.65, label=f"{client} (seed mean)")
        # Average clients WITHIN each seed; seed SD is not a client-population CI.
        per_seed = [[statistics.fmean(scores[task, client, seed, rank] for client in clients)
                     for seed in seeds] for rank in ranks]
        means = [statistics.fmean(values) for values in per_seed]
        axis.plot(ranks, means, "o-", color="black", linewidth=2.2,
                  label=f"equal-client mean (clients={len(clients)}, seeds={len(seeds)})")
        if len(seeds) > 1:
            deviations = [statistics.stdev(values) for values in per_seed]
            axis.fill_between(ranks, [m - s for m, s in zip(means, deviations, strict=True)],
                              [m + s for m, s in zip(means, deviations, strict=True)],
                              color="black", alpha=0.12, label="across-seed SD (not CI)")
        axis.set_xscale("log", base=2)
        axis.set_xticks(ranks, [str(rank) for rank in ranks])
        axis.set_xlabel("LoRA rank")
        axis.set_ylabel(LABELS[task])
        axis.set_title(f"{task}: fixed epoch {epoch}")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    figure.tight_layout()
    paths = []
    for extension in ("png", "svg"):
        path = output / f"rank_validation_epoch{epoch}.{extension}"
        figure.savefig(path, dpi=180)
        paths.append(str(path))
    plt.close(figure)
    figure, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 4), squeeze=False)
    for axis, task in zip(axes[0], tasks, strict=True):
        clients, seeds = sorted(groups[task]["clients"]), sorted(groups[task]["seeds"])
        ranks = sorted(groups[task]["ranks"])
        pairs = list(zip(ranks[:-1], ranks[1:], strict=True))
        width = 0.8 / len(pairs)
        for index, (low, high) in enumerate(pairs):
            gains = [statistics.fmean(scores[task, client, seed, high]
                     - scores[task, client, seed, low] for seed in seeds) for client in clients]
            axis.bar([i + (index - (len(pairs) - 1) / 2) * width for i in range(len(clients))],
                     gains, width=width, label=f"rank {low} to {high}")
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(list(range(len(clients))), clients)
        axis.set_ylabel(f"Gain in {LABELS[task]}")
        axis.set_title(f"{task}: paired gains, epoch {epoch}")
        axis.legend()
    figure.tight_layout()
    for extension in ("png", "svg"):
        path = output / f"rank_gains_epoch{epoch}.{extension}"
        figure.savefig(path, dpi=180)
        paths.append(str(path))
    plt.close(figure)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--epoch", type=int, help="Defaults to final fixed-budget epoch (5)")
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    root = args.run_directory.resolve()
    paths = plot_results(root, args.output_directory or root / "figures", args.epoch)
    print(json.dumps({"figures": paths}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
