#!/usr/bin/env python3
"""Plot the complete MDL curves emitted by mdl_stream_comparison."""

import argparse
import csv
from collections import defaultdict

import matplotlib.pyplot as plt


def read_rows(path):
    rows = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for key in ("epsilon", "model_bits", "residual_bits", "total_bits"):
                row[key] = float(row[key])
            row["representatives"] = int(row["representatives"])
            rows[row["stream"]].append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("output_path")
    args = parser.parse_args()
    rows = read_rows(args.csv_path)
    order = ("structured", "clustered", "random")
    colors = {"structured": "#2166ac", "clustered": "#b2182b", "random": "#4d4d4d"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for stream in order:
        points = sorted(rows[stream], key=lambda row: row["epsilon"])
        eps = [row["epsilon"] for row in points]
        total = [row["total_bits"] for row in points]
        model = [row["model_bits"] for row in points]
        residual = [row["residual_bits"] for row in points]
        best = min(points, key=lambda row: row["total_bits"])
        label = f"{stream} (ε*={best['epsilon']:g})"
        axes[0].plot(eps, total, marker="o", color=colors[stream], label=label)
        axes[0].scatter([best["epsilon"]], [best["total_bits"]], color=colors[stream], s=55, zorder=4)
        axes[1].plot(eps, model, marker="o", linestyle="--", color=colors[stream], alpha=0.85,
                     label=f"{stream}: model")
        axes[1].plot(eps, residual, marker="s", color=colors[stream], alpha=0.85,
                     label=f"{stream}: residual")

    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xlabel("epsilon")
        axis.grid(True, alpha=0.25)
    axes[0].set_title("Total description length")
    axes[0].set_ylabel("bits")
    axes[0].legend(fontsize=8)
    axes[1].set_title("MDL components")
    axes[1].set_ylabel("bits")
    axes[1].legend(fontsize=7, ncol=2)
    fig.suptitle("FUTCache MDL epsilon selection: identical codec and grid")
    fig.savefig(args.output_path, dpi=180)


if __name__ == "__main__":
    main()
