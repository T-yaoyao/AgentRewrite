#!/usr/bin/env python3
"""Test different y-axis scale options for ablation study figure."""

from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def generate_ln_version():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times New Roman PS MT",
                "Nimbus Roman",
                "Liberation Serif",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "font.size": 11,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 11,
            "legend.fontsize": 11.5,
        }
    )

    methods = [
        "PASR",
        "without\nRule Group",
        "without\nRule Example",
        "without\nRule Knowledge",
    ]

    raw_data = {
        "Mean": [3.12, 6.14, 11.62, 21.15],
        "Median": [0.95, 0.94, 1.08, 1.74],
        "75th": [2.42, 7.81, 11.26, 19.79],
        "95th": [11.12, 26.58, 42.43, 84.01],
    }

    styles = {
        "Mean": {"color": "#7ebce6", "edgecolor": "#111111", "hatch": "///"},
        "Median": {"color": "#185e78", "edgecolor": "#111111", "hatch": "\\\\\\"},
        "75th": {"color": "#944005", "edgecolor": "#111111", "hatch": "xxx"},
        "95th": {"color": "#f79e38", "edgecolor": "#111111", "hatch": "..."},
    }

    # Version 1: Natural log ln(s) exactly as in reference image!
    # ln(s) values
    ln_data = {k: [np.log(v) for v in vals] for k, vals in raw_data.items()}

    x = np.arange(len(methods))
    total_group_width = 0.68
    n_bars = len(raw_data)
    bar_width = total_group_width / n_bars
    offsets = np.arange(n_bars) * bar_width - (total_group_width / 2.0) + (bar_width / 2.0)

    fig, ax = plt.subplots(figsize=(9.2, 4.3))

    for i, (metric, values) in enumerate(ln_data.items()):
        pos = x + offsets[i]
        st = styles[metric]
        raw_vals = raw_data[metric]
        
        # We plot ln values, but display raw seconds on top of bars
        bars = ax.bar(
            pos,
            values,
            width=bar_width,
            label=metric,
            color=st["color"],
            edgecolor=st["edgecolor"],
            hatch=st["hatch"],
            linewidth=1.0,
            zorder=3,
        )

        for rect, raw_v, ln_v in zip(bars, raw_vals, values):
            h = rect.get_height()
            # If h < 0 (like ln(0.95) = -0.05), place text just above 0 or h
            y_text = max(h, 0) + 0.12
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                y_text,
                f"{raw_v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.0,
                color="#222222",
                fontfamily="serif",
            )

    ax.set_ylabel("Execution Time (ln(s))", color="#111111", labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(-0.3, 5.2)
    ax.set_yticks(np.arange(0, 6, 1))

    # Grid and full box border like reference image
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.7, color="#d9d9d9", alpha=0.8)
    ax.axhline(0, color="#666666", linewidth=0.8, zorder=2)
    for spine in ax.spines.values():
        spine.set_color("#111111")
        spine.set_linewidth(1.0)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=4,
        frameon=True,
        edgecolor="#222222",
        fancybox=False,
        columnspacing=2.2,
        handlelength=1.8,
        handleheight=1.0,
    )

    fig.subplots_adjust(top=0.83, bottom=0.18, left=0.09, right=0.97)

    out_dir = Path("/root/AgentRewrite/figures")
    fig.savefig(out_dir / "ablation_rule_ln.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def generate_log_scale_version():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times New Roman PS MT",
                "Nimbus Roman",
                "Liberation Serif",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "font.size": 11,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 11,
            "legend.fontsize": 11.5,
        }
    )

    methods = [
        "PASR",
        "without\nRule Group",
        "without\nRule Example",
        "without\nRule Knowledge",
    ]

    raw_data = {
        "Mean": [3.12, 6.14, 11.62, 21.15],
        "Median": [0.95, 0.94, 1.08, 1.74],
        "75th": [2.42, 7.81, 11.26, 19.79],
        "95th": [11.12, 26.58, 42.43, 84.01],
    }

    styles = {
        "Mean": {"color": "#7ebce6", "edgecolor": "#111111", "hatch": "///"},
        "Median": {"color": "#185e78", "edgecolor": "#111111", "hatch": "\\\\\\"},
        "75th": {"color": "#944005", "edgecolor": "#111111", "hatch": "xxx"},
        "95th": {"color": "#f79e38", "edgecolor": "#111111", "hatch": "..."},
    }

    x = np.arange(len(methods))
    total_group_width = 0.68
    n_bars = len(raw_data)
    bar_width = total_group_width / n_bars
    offsets = np.arange(n_bars) * bar_width - (total_group_width / 2.0) + (bar_width / 2.0)

    fig, ax = plt.subplots(figsize=(9.2, 4.3))
    ax.set_yscale("log")

    for i, (metric, values) in enumerate(raw_data.items()):
        pos = x + offsets[i]
        st = styles[metric]
        
        bars = ax.bar(
            pos,
            values,
            width=bar_width,
            label=metric,
            color=st["color"],
            edgecolor=st["edgecolor"],
            hatch=st["hatch"],
            linewidth=1.0,
            zorder=3,
        )

        for rect, raw_v in zip(bars, values):
            h = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                h * 1.15,
                f"{raw_v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.0,
                color="#222222",
                fontfamily="serif",
            )

    ax.set_ylabel("Execution Time (s, log scale)", color="#111111", labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0.5, 160)
    ax.set_yticks([0.5, 1, 2, 5, 10, 20, 50, 100])
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())

    # Grid and full box border like reference image
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.7, color="#d9d9d9", alpha=0.8, which="both")
    for spine in ax.spines.values():
        spine.set_color("#111111")
        spine.set_linewidth(1.0)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=4,
        frameon=True,
        edgecolor="#222222",
        fancybox=False,
        columnspacing=2.2,
        handlelength=1.8,
        handleheight=1.0,
    )

    fig.subplots_adjust(top=0.83, bottom=0.18, left=0.09, right=0.97)

    out_dir = Path("/root/AgentRewrite/figures")
    fig.savefig(out_dir / "ablation_rule_log.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    generate_ln_version()
    generate_log_scale_version()
