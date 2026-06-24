"""Generate the publication-quality figure set for the paper.

Reads the experiment log + reference benchmark and writes a consistent set of
figures to results/figures/. Run from the project root:

    python -m genai_experiments.make_figures
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from src.algorithms.greedy import greedy_tour
from src.algorithms.simulated_annealing import simulated_annealing
from src.utils.distance import distance_matrix
from src.utils.parser import parse_tsp

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {"gpt": "#4C72B0", "claude": "#DD8452"}
INSTANCE_ORDER = ["eil51", "berlin52", "ch130", "d198", "pr439", "pr1002"]
ALGO_LABELS = {
    "greedy": "Greedy",
    "simulated_annealing": "Sim. Annealing",
    "genetic_algorithm": "Genetic Algo.",
}


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


def fig_quality_by_algorithm_model(df):
    """Mean gap +/- std by algorithm, grouped by model (with error bars).

    Error bars are asymmetric: clipped at 0 below the mean since gap-to-optimum
    cannot be negative, but the full std is shown above the mean.
    """
    algos = list(ALGO_LABELS)
    models = ["gpt", "claude"]
    x = np.arange(len(algos))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    for k, model in enumerate(models):
        sub = df[df.model == model]
        means = np.array([sub[sub.algorithm == a].gap_pct.mean() for a in algos])
        stds = np.array([sub[sub.algorithm == a].gap_pct.std() for a in algos])
        lower_err = np.minimum(stds, means)  # clip at 0
        ax.bar(x + (k - 0.5) * width, means, width,
               yerr=[lower_err, stds], capsize=4,
               label=model, color=COLORS[model])
    ax.set_xticks(x)
    ax.set_xticklabels([ALGO_LABELS[a] for a in algos])
    ax.set_ylabel("gap-to-optimal (%)  (mean, error bars = std clipped at 0)")
    ax.set_title("Solution quality by algorithm and model")
    ax.set_ylim(bottom=0)
    ax.legend(title="model")
    _save(fig, "fig_quality_by_algorithm_model.png")


def fig_strategy_model_heatmap(df):
    heat = df.pivot_table(index="strategy", columns="model", values="gap_pct", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(heat.values, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(heat.columns)))
    ax.set_xticklabels(heat.columns)
    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels(heat.index)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, f"{heat.values[i, j]:.1f}", ha="center", va="center")
    ax.set_title("Mean gap-to-optimal (%) by strategy × model")
    fig.colorbar(im, label="gap %")
    _save(fig, "fig_strategy_model_heatmap.png")


def fig_gap_by_instance(df):
    """How gap scales with instance size, per model (greedy excluded -> deterministic)."""
    sub = df[df.algorithm != "greedy"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for model in ["gpt", "claude"]:
        means = [sub[(sub.model == model) & (sub.instance == inst)].gap_pct.mean()
                 for inst in INSTANCE_ORDER]
        ax.plot(INSTANCE_ORDER, means, "o-", label=model, color=COLORS[model], linewidth=2)
    # Annotate each x-tick with the city count to make the size ladder explicit.
    city_counts = {"eil51": 51, "berlin52": 52, "ch130": 130, "d198": 198,
                   "pr439": 439, "pr1002": 1002}
    labels = [f"{inst}\n(n={city_counts[inst]})" for inst in INSTANCE_ORDER]
    ax.set_xticks(range(len(INSTANCE_ORDER)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("mean gap-to-optimal (%)")
    ax.set_xlabel("instance (increasing size →)")
    ax.set_title("Solution quality vs. instance size (SA + GA)")
    ax.set_ylim(bottom=0)
    ax.legend(title="model")
    _save(fig, "fig_gap_by_instance.png")


def fig_reliability(df):
    """Ran-rate by model x algorithm (the timeout / failure story)."""
    rate = df.pivot_table(index="algorithm", columns="model", values="ran", aggfunc="mean")
    rate = rate.reindex(list(ALGO_LABELS))
    x = np.arange(len(rate))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k, model in enumerate(["gpt", "claude"]):
        ax.bar(x + (k - 0.5) * width, rate[model] * 100, width, label=model, color=COLORS[model])
    ax.set_xticks(x)
    ax.set_xticklabels([ALGO_LABELS[a] for a in rate.index])
    ax.set_ylabel("ran successfully (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Reliability: generated code that executed and returned a valid tour")
    ax.legend(title="model")
    _save(fig, "fig_reliability.png")


def fig_llm_vs_reference(df, ref):
    ref_gap = ref.groupby("algorithm").gap_pct.mean()
    algos = list(ALGO_LABELS)
    comp = pd.DataFrame({
        "reference": [ref_gap[a] for a in algos],
        "LLM mean": [df[df.algorithm == a].gap_pct.mean() for a in algos],
        "LLM best": [df[df.algorithm == a].gap_pct.min() for a in algos],
    }, index=[ALGO_LABELS[a] for a in algos])
    fig, ax = plt.subplots(figsize=(8, 5))
    comp.plot(kind="bar", ax=ax, color=["#7f7f7f", "#4C72B0", "#55A868"])
    ax.set_ylabel("gap-to-optimal (%)")
    ax.set_title("LLM-generated vs. hand-coded reference (mean across 6 instances)")
    ax.set_xlabel("")
    plt.xticks(rotation=0)
    # Annotate every bar with its value so 0 % / sub-1 % bars are not invisible.
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=2, fontsize=9)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.08)  # headroom for the labels
    _save(fig, "fig_llm_vs_reference.png")


def fig_strategy_variance(df):
    """Mean +/- std gap by strategy per model -- shows which strategies are stable."""
    strategies = ["zero_shot", "few_shot", "chain_of_thought", "iterative_refinement"]
    x = np.arange(len(strategies))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    for k, model in enumerate(["gpt", "claude"]):
        sub = df[df.model == model]
        means = np.array([sub[sub.strategy == s].gap_pct.mean() for s in strategies])
        stds = np.array([sub[sub.strategy == s].gap_pct.std() for s in strategies])
        lower_err = np.minimum(stds, means)  # gap cannot go below 0
        ax.bar(x + (k - 0.5) * width, means, width,
               yerr=[lower_err, stds], capsize=4,
               label=model, color=COLORS[model])
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in strategies])
    ax.set_ylabel("gap-to-optimal (%)  (mean, error bars = std clipped at 0)")
    ax.set_title("Prompting strategy: quality and run-to-run variance")
    ax.set_ylim(bottom=0)
    ax.legend(title="model")
    _save(fig, "fig_strategy_variance.png")


# ----------------------------- explanatory figures -----------------------------

def fig_tour_comparison():
    """Greedy vs. SA reference tours on eil51 — makes the gap visual."""
    coords = parse_tsp(ROOT / "data" / "tsplib" / "eil51.tsp")
    pts = np.array(coords)
    D = distance_matrix(coords)
    opt = 426
    g_tour, g_len = greedy_tour(D)
    sa_tour, sa_len = simulated_annealing(D)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, tour, length, name in [
        (axes[0], g_tour, g_len, "Greedy nearest-neighbor"),
        (axes[1], sa_tour, sa_len, "Simulated Annealing"),
    ]:
        loop = tour + [tour[0]]
        ax.plot(pts[loop, 0], pts[loop, 1], "-", color="#4C72B0", linewidth=1.2, zorder=1)
        ax.scatter(pts[:, 0], pts[:, 1], color="#DD8452", s=22, zorder=2)
        gap = (length - opt) / opt * 100
        ax.set_title(f"{name}\nlength {length:.0f}  ({gap:.1f}% over optimum)")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    fig.suptitle("Reference tours on eil51 (optimum = 426)", fontsize=13)
    _save(fig, "fig_tour_comparison.png")


def fig_pipeline():
    """Flow diagram of the GenAI experiment harness (methodology, §5)."""
    steps = [
        "Prompt template\n(strategy × algorithm)",
        "LLM\n(GPT-5.5 / Claude Opus 4.7)",
        "Code extraction\n(parse fenced block)",
        "Sandboxed executor\n(subprocess, 60–300 s timeout)",
        "Benchmark on\n6 TSPLIB instances (51–1002 cities)",
        "experiment_log.csv\n(+ saved code & prompts)",
    ]
    fig, ax = plt.subplots(figsize=(6.5, 9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(steps) * 2)
    ax.axis("off")
    y_positions = [(len(steps) - i) * 2 - 1 for i in range(len(steps))]
    for i, (text, y) in enumerate(zip(steps, y_positions)):
        box = FancyBboxPatch((2.5, y - 0.55), 5, 1.1, boxstyle="round,pad=0.1",
                             linewidth=1.4, edgecolor="#333333", facecolor="#EAF0F7")
        ax.add_patch(box)
        ax.text(5, y, text, ha="center", va="center", fontsize=10)
        if i < len(steps) - 1:
            ax.annotate("", xy=(5, y_positions[i + 1] + 0.6), xytext=(5, y - 0.6),
                        arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.5))
    # Feedback loop: executor (step 3) back to LLM (step 1) for iterative refinement.
    ax.annotate("", xy=(7.5, y_positions[1]), xytext=(7.5, y_positions[3]),
                arrowprops=dict(arrowstyle="-|>", color="#C44E52", lw=1.4,
                                connectionstyle="arc3,rad=-0.5"))
    ax.text(9.0, (y_positions[1] + y_positions[3]) / 2, "iterative\nrefinement\n(on failure)",
            ha="center", va="center", fontsize=8.5, color="#C44E52")
    ax.set_title("GenAI experiment pipeline", fontsize=13)
    _save(fig, "fig_pipeline.png")


def fig_two_opt():
    """Before/after illustration of a 2-opt move removing a crossing."""
    pts = {0: (0, 0), 1: (2, 2), 2: (2, 0), 3: (0, 2)}
    before = [0, 1, 2, 3]   # edges 0-1 and 2-3 cross
    after = [0, 2, 1, 3]    # after reversing the middle segment: perimeter

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    for ax, tour, title, hi_color in [
        (axes[0], before, "Before: edges cross", "#C44E52"),
        (axes[1], after, "After 2-opt: segment reversed", "#55A868"),
    ]:
        loop = tour + [tour[0]]
        xs = [pts[c][0] for c in loop]
        ys = [pts[c][1] for c in loop]
        ax.plot(xs, ys, "-", color="#4C72B0", linewidth=1.6, zorder=1)
        # highlight the two edges that change (the closing edge + first edge here)
        ax.plot([pts[tour[0]][0], pts[tour[1]][0]], [pts[tour[0]][1], pts[tour[1]][1]],
                color=hi_color, linewidth=2.6, zorder=2)
        ax.plot([pts[tour[2]][0], pts[tour[3]][0]], [pts[tour[2]][1], pts[tour[3]][1]],
                color=hi_color, linewidth=2.6, zorder=2)
        for c, (x, y) in pts.items():
            ax.scatter([x], [y], color="#333333", s=40, zorder=3)
            ax.text(x, y + 0.12, str(c), ha="center", fontsize=11)
        ax.set_title(title)
        ax.set_xlim(-0.6, 2.6)
        ax.set_ylim(-0.6, 2.6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    fig.suptitle("The 2-opt move (Simulated Annealing neighborhood)", fontsize=13)
    _save(fig, "fig_two_opt.png")


def main():
    df = pd.read_csv(ROOT / "genai_experiments" / "experiment_log.csv")
    ref = pd.read_csv(ROOT / "results" / "benchmark_results.csv")
    print(f"Generating figures from {len(df)} rows -> {FIG_DIR.relative_to(ROOT)}")
    fig_quality_by_algorithm_model(df)
    fig_strategy_model_heatmap(df)
    fig_gap_by_instance(df)
    fig_reliability(df)
    fig_llm_vs_reference(df, ref)
    fig_strategy_variance(df)
    # explanatory figures
    fig_tour_comparison()
    fig_pipeline()
    fig_two_opt()
    print("done.")


if __name__ == "__main__":
    main()
