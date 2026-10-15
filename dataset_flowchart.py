
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# -----------------------------------------
# Canvas
# -----------------------------------------
fig, ax = plt.subplots(figsize=(16, 12))
ax.set_xlim(0, 16)
ax.set_ylim(0, 15)
ax.axis("off")

fig.suptitle(
    "AI Co-Pilot: Combined Aviation Dataset Pipeline",
    fontsize=20,
    fontweight="bold",
    color="#163A63",
    y=0.97
)

# -----------------------------------------
# Colors
# -----------------------------------------
BLUE = "#DCEEFF"
GREEN = "#DDF5E3"
ORANGE = "#FFF0D9"
PURPLE = "#EDE3FF"
RED = "#FFE2E2"
DARK = "#263746"

def box(x, y, w, h, text, color, fontsize=10,
        edgecolor="#526579", linewidth=1.5):
    """Draw a rounded box with centered text."""
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.08,rounding_size=0.18",
        facecolor=color,
        edgecolor=edgecolor,
        linewidth=linewidth
    )
    ax.add_patch(patch)

    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center",
        fontsize=fontsize,
        color=DARK,
        wrap=True
    )

def arrow(x1, y1, x2, y2):
    """Draw a directional connector."""
    patch = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.6,
        color="#465665",
        connectionstyle="arc3,rad=0"
    )
    ax.add_patch(patch)

def section(y, h, title, color):
    """Draw a light background for a pipeline stage."""
    patch = FancyBboxPatch(
        (0.35, y), 15.3, h,
        boxstyle="round,pad=0.08,rounding_size=0.2",
        facecolor=color,
        edgecolor="#A0AAB4",
        linewidth=1.0,
        alpha=0.38,
        zorder=0
    )
    ax.add_patch(patch)

    ax.text(
        8, y + h - 0.35, title,
        ha="center", va="center",
        fontsize=13,
        fontweight="bold",
        color="#163A63"
    )

# -----------------------------------------
# Stage 1: Data sources
# -----------------------------------------
section(11.8, 2.3, "1. RAW AVIATION DATA SOURCES", BLUE)

box(
    0.8, 12.15, 4.4, 1.0,
    "NASA ASRS Incident Database\n6,978 cleaned reports",
    BLUE, 11, "#3478B8"
)

box(
    5.8, 12.15, 4.4, 1.0,
    "OpenSky Squawk 7700\n832 cleaned emergency trajectories",
    GREEN, 11, "#2E9860"
)

box(
    10.8, 12.15, 4.2, 1.0,
    "Normal Flight Envelope Generator\n1,716 baseline sequences",
    PURPLE, 10, "#8054C7"
)

# -----------------------------------------
# Stage 2: Preprocessing
# -----------------------------------------
section(8.6, 2.7, "2. INGESTION & CONTEXT MIXING (dataset_generator.py)", ORANGE)

box(
    0.8, 9.25, 4.4, 1.0,
    "map_asrs_record()\nPhase, severity, hazard and SOP",
    RED, 10, "#C64D4D"
)

box(
    5.8, 9.25, 4.4, 1.0,
    "map_opensky_record()\nCall-sign, aircraft type and transponder state",
    GREEN, 10, "#2E9860"
)

box(
    10.8, 9.05, 4.2, 1.4,
    "12-Parameter Feature Synthesizer\nAltitude, speed, vertical rate, pitch,\nroll, heading, N1, G, distance,\nwind, energy and flight phase",
    PURPLE, 9, "#8054C7"
)

# Source-to-mapping connections
arrow(3.0, 12.15, 3.0, 10.25)
arrow(8.0, 12.15, 8.0, 10.25)

# Mapping and normal-flight inputs to synthesizer
arrow(5.2, 9.65, 10.8, 9.65)
arrow(10.2, 9.45, 10.8, 9.45)
arrow(12.9, 12.15, 12.9, 10.45)

# -----------------------------------------
# Stage 3: Combined dataset
# -----------------------------------------
section(6.3, 1.7, "3. UNIFIED COMBINED DATASET", GREEN)

box(
    3.0, 6.55, 10.0, 0.75,
    "9,526 sequences  |  15 timesteps per sequence  |  12 features per timestep",
    GREEN, 12, "#20804A"
)

arrow(12.9, 9.05, 8.0, 7.3)

# -----------------------------------------
# Stage 4: Train/test split
# -----------------------------------------
section(3.5, 2.0, "4. TRAIN / TEST PARTITION (80:20)", RED)

box(
    1.1, 3.9, 6.1, 0.9,
    "Training Set (80%)\n7,620 sequential samples",
    BLUE, 12, "#3478B8"
)

box(
    8.8, 3.9, 6.1, 0.9,
    "Testing Set (20%)\n1,906 sequential samples",
    ORANGE, 12, "#C18A00"
)

arrow(8.0, 6.55, 4.15, 4.8)
arrow(8.0, 6.55, 11.85, 4.8)

# -----------------------------------------
# Stage 5: Dataset composition
# -----------------------------------------
section(0.4, 2.5, "5. DATASET COMPOSITION", PURPLE)

composition = [
    ("NASA ASRS", 6978, "#3478B8"),
    ("OpenSky", 832, "#2E9860"),
    ("Normal Flight", 1716, "#8054C7")
]

total = sum(item[1] for item in composition)
left = 1.0
bar_y = 1.35
bar_width = 14.0
bar_height = 0.45

for name, count, color in composition:
    width = bar_width * count / total

    ax.barh(
        bar_y,
        width,
        left=left,
        height=bar_height,
        color=color,
        edgecolor="white"
    )

    if width > 1.0:
        ax.text(
            left + width / 2,
            bar_y,
            f"{name}\n{count:,}",
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            fontweight="bold"
        )

    left += width

ax.text(
    8, 0.75,
    "Total: 9,526 sequences",
    ha="center",
    fontsize=12,
    fontweight="bold",
    color="#163A63"
)

# -----------------------------------------
# Export
# -----------------------------------------
plt.savefig(
    "combined_dataset_flowchart.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.savefig(
    "combined_dataset_flowchart.pdf",
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

print("Flowchart saved successfully:")
print("- combined_dataset_flowchart.png")
print("- combined_dataset_flowchart.pdf")
