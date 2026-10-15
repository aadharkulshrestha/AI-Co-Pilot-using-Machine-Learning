import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------
# Model comparison data
# -----------------------------------------
models = [
    "Our AI Co-Pilot",
    "Random Forest",
    "Linear SVM",
    "Logistic Regression",
    "Gradient Boosting"
]

accuracy = [94.20, 96.5, 80.5, 84.5, 96.5]

# Precision and recall for our previous model were not verified.
precision = [np.nan, 96.5, 81.0, 84.5, 96.5]
recall = [np.nan, 96.5, 80.5, 84.5, 96.5]

f1_score = [0.939, 0.965, 0.810, 0.845, 0.965]

# -----------------------------------------
# Styling
# -----------------------------------------
colors = [
    "#2878B5",  # Our AI Co-Pilot
    "#59A14F",
    "#F28E2B",
    "#B07AA1",
    "#76B7B2"
]

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

fig.suptitle(
    "AI Co-Pilot vs Existing Machine Learning Models",
    fontsize=19,
    fontweight="bold",
    color="#163A63"
)

def draw_chart(ax, values, title, ylabel, percent=False):
    bars = ax.bar(
        models,
        values,
        color=colors,
        edgecolor="white",
        linewidth=1
    )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=9)

    # Keep missing values as gaps, not zero scores.
    valid = [v for v in values if not np.isnan(v)]

    if percent:
        ax.set_ylim(0, 110)
        labels = [
            f"{v:.1f}%" if not np.isnan(v) else "N/V"
            for v in values
        ]
    else:
        ax.set_ylim(0, 1.08)
        labels = [
            f"{v:.3f}" if not np.isnan(v) else "N/V"
            for v in values
        ]

    for bar, label, value in zip(bars, labels, values):
        if not np.isnan(value):
            ax.annotate(
                label,
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                fontweight="bold"
            )
        else:
            ax.annotate(
                "N/V",
                (bar.get_x() + bar.get_width() / 2, 0.03),
                ha="center",
                fontsize=9,
                color="dimgray"
            )

    ax.tick_params(axis="x", labelrotation=0)

# Accuracy
draw_chart(
    axes[0, 0], accuracy,
    "Accuracy Comparison",
    "Accuracy (%)",
    percent=True
)

# Precision
draw_chart(
    axes[0, 1], precision,
    "Precision Comparison",
    "Precision (%)",
    percent=True
)

# Recall
draw_chart(
    axes[1, 0], recall,
    "Recall Comparison",
    "Recall (%)",
    percent=True
)

# F1 score
draw_chart(
    axes[1, 1], f1_score,
    "F1-Score Comparison",
    "Macro F1-Score",
    percent=False
)

fig.text(
    0.5, 0.015,
    "N/V = Not verified. Existing-model values are approximate supplied benchmarks.",
    ha="center",
    fontsize=10,
    color="dimgray"
)

plt.tight_layout(rect=[0, 0.05, 1, 0.94])

# Resolve output directories
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "ai_copilot_model_comparison.png")
json_path = os.path.join(script_dir, "model_comparison_results.json")

# 1. Store plot image
plt.savefig(image_path, dpi=300, bbox_inches="tight")
if os.path.abspath(os.getcwd()) != script_dir:
    plt.savefig("ai_copilot_model_comparison.png", dpi=300, bbox_inches="tight")

# 2. Store results into JSON file
results_data = {
    "comparison_title": "AI Co-Pilot vs Existing Machine Learning Models",
    "benchmark_results": [
        {
            "model": m.replace("\n", " "),
            "accuracy_percent": None if np.isnan(acc) else acc,
            "precision_percent": None if np.isnan(prec) else prec,
            "recall_percent": None if np.isnan(rec) else rec,
            "macro_f1": None if np.isnan(f1) else f1
        }
        for m, acc, prec, rec, f1 in zip(models, accuracy, precision, recall, f1_score)
    ]
}

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results_data, f, indent=2)

if os.path.abspath(os.getcwd()) != script_dir:
    with open("model_comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

# 3. Print summary table and saved file paths
print("\n" + "=" * 68)
print("              AI CO-PILOT MODEL COMPARISON BENCHMARK")
print("=" * 68)
print(f"{'Model':<22} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<10}")
print("-" * 68)
for m, acc, prec, rec, f1 in zip(models, accuracy, precision, recall, f1_score):
    clean_name = m.replace('\n', ' ')
    acc_str = f"{acc:.2f}%" if not np.isnan(acc) else "N/V"
    prec_str = f"{prec:.2f}%" if not np.isnan(prec) else "N/V"
    rec_str = f"{rec:.2f}%" if not np.isnan(rec) else "N/V"
    f1_str = f"{f1:.3f}" if not np.isnan(f1) else "N/V"
    print(f"{clean_name:<22} {acc_str:<12} {prec_str:<12} {rec_str:<12} {f1_str:<10}")
print("=" * 68)
print(f"[✓] Plot image saved to:    {image_path}")
print(f"[✓] Benchmark JSON saved to: {json_path}")
print("=" * 68 + "\n")

if "--show" in sys.argv:
    plt.show()
else:
    plt.close()
