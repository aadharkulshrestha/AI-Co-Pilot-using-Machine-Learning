import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# -------------------------------------------------------------
# Canvas Setup - Publication Grade Geometry
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(16, 17.5), dpi=300)
ax.set_xlim(0, 16)
ax.set_ylim(0, 18.0)
ax.axis("off")

# Title Header
fig.suptitle(
    "AI Co-Pilot: Aviation Dataset Pipeline & Feature Engineering Architecture",
    fontsize=18,
    fontweight="bold",
    color="#0B2545",
    y=0.985
)

# -------------------------------------------------------------
# Premium Academic Color Palette
# -------------------------------------------------------------
BLUE_BG   = "#EEF5FB"
BLUE_BD   = "#2A75D3"
GREEN_BG  = "#EBF7F0"
GREEN_BD  = "#27AE60"
ORANGE_BG = "#FEF7EE"
ORANGE_BD = "#E67E22"
PURPLE_BG = "#F5F0FB"
PURPLE_BD = "#8E44AD"
RED_BG    = "#FDEDEC"
RED_BD    = "#E74C3C"

def draw_section_card(x, y, w, h, stage_label, stage_title, bgcolor, border_color, title_x=None, title_align="left"):
    """Draw a light grouping container with clean stage banner."""
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.08,rounding_size=0.22",
        facecolor=bgcolor,
        edgecolor=border_color,
        linewidth=1.3,
        alpha=0.55,
        zorder=1
    )
    ax.add_patch(patch)

    tx = (x + 0.35) if title_x is None else title_x
    ax.text(
        tx, y + h - 0.30,
        f"{stage_label}: {stage_title}",
        ha=title_align, va="center",
        fontsize=11.5,
        fontweight="bold",
        color="#0B2545",
        zorder=2
    )

def draw_card(x, y, w, h, title, body_text, facecolor="#FFFFFF", edgecolor="#2C3E50", title_color="#0B2545", fontsize=9.2, linewidth=1.4):
    """Draw a clean publication card with bold title and body text."""
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.06,rounding_size=0.18",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=3
    )
    ax.add_patch(patch)

    ax.text(
        x + w / 2, y + h - 0.26, title,
        ha="center", va="top",
        fontsize=fontsize + 0.6,
        fontweight="bold",
        color=title_color,
        zorder=4
    )
    ax.text(
        x + w / 2, y + 0.16, body_text,
        ha="center", va="bottom",
        fontsize=fontsize - 0.3,
        color="#34495E",
        wrap=True,
        linespacing=1.25,
        zorder=4
    )

def draw_arrow(x1, y1, x2, y2, color="#2C3E50", lw=1.6):
    """Draw a clean directional connector arrow."""
    patch = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=lw,
        color=color,
        connectionstyle="arc3,rad=0",
        zorder=2
    )
    ax.add_patch(patch)

# -------------------------------------------------------------
# STAGE 1: RAW AVIATION DATA INGESTION
# -------------------------------------------------------------
# Container: Y in [14.6, 17.3]
draw_section_card(0.5, 14.6, 15.0, 2.7, "STAGE 1", "RAW MULTI-MODAL AVIATION DATA INGESTION", BLUE_BG, BLUE_BD, title_x=8.0, title_align="center")

draw_card(
    0.9, 14.85, 4.4, 1.75,
    "NASA ASRS Database",
    "6,978 cleaned incident records\nMulti-system operational narratives\nExpert pilot human factor contexts",
    facecolor="#FFFFFF", edgecolor="#1F618D", title_color="#1B4F72"
)

draw_card(
    5.8, 14.85, 4.4, 1.75,
    "OpenSky Squawk 7700",
    "832 emergency trajectory segments\nHigh-density ADS-B telemetry\nDynamic 30-min gap boundaries",
    facecolor="#FFFFFF", edgecolor="#229954", title_color="#196F3D"
)

draw_card(
    10.7, 14.85, 4.4, 1.75,
    "Normal Flight Generator",
    "1,716 nominal baseline trajectories\nClimb, cruise, and descent envelopes\nZero-hazard operational profiles",
    facecolor="#FFFFFF", edgecolor="#884EA0", title_color="#6C3483"
)

# Inter-stage connector arrows: from bottom of Stage 1 cards (14.85) to top of Stage 2 mappers (13.10)
draw_arrow(3.1, 14.85, 3.1, 13.10, color="#1F618D")
draw_arrow(8.0, 14.85, 8.0, 13.10, color="#229954")
draw_arrow(12.9, 14.85, 12.9, 13.10, color="#884EA0")

# -------------------------------------------------------------
# STAGE 2: INGESTION, PARSING & 12-D FEATURE SYNTHESIS
# -------------------------------------------------------------
# Container: Y in [8.3, 14.0]
# Arrows enter at x = 3.1, 8.0, 12.9.
# Title placed at x = 3.65 (left aligned): "STAGE 2: DATA ADAPTER PIPELINE" (27 chars, spans x in [3.65, 7.20]).
# Arrow 1 is at 3.1 (< 3.65), Arrow 2 is at 8.0 (> 7.20). Zero collision!
draw_section_card(0.5, 8.3, 15.0, 5.7, "STAGE 2", "DATA ADAPTER PIPELINE", ORANGE_BG, ORANGE_BD, title_x=3.65, title_align="left")

# Row 1: 3 Domain Mappers (Y in [11.65, 13.10])
draw_card(
    0.9, 11.65, 4.4, 1.45,
    "map_asrs_record()",
    "Parses flight phase, severity level,\nsystem hazard flags & target SOP action",
    facecolor="#FFFFFF", edgecolor="#C0392B", title_color="#922B21"
)

draw_card(
    5.8, 11.65, 4.4, 1.45,
    "map_opensky_record()",
    "Extracts callsign, aircraft type, ADS-B\nspatial track coordinates & kinematic rates",
    facecolor="#FFFFFF", edgecolor="#229954", title_color="#196F3D"
)

draw_card(
    10.7, 11.65, 4.4, 1.45,
    "map_normal_envelope()",
    "Simulates standard operating envelopes\nand baseline aerodynamic limits",
    facecolor="#FFFFFF", edgecolor="#884EA0", title_color="#6C3483"
)

# Arrows from Mappers converging down into Feature Synthesizer (from 11.65 to 10.50)
draw_arrow(3.1, 11.65, 4.5, 10.50, color="#C0392B")
draw_arrow(8.0, 11.65, 8.0, 10.50, color="#229954")
draw_arrow(12.9, 11.65, 11.5, 10.50, color="#884EA0")

# Row 2: Unified 12-Parameter Feature Synthesizer Box (Y in [8.55, 10.50])
draw_card(
    0.9, 8.55, 14.2, 1.95,
    "12-Parameter Aerodynamic Feature Synthesizer (Z-Score Standardized on Train Set)",
    "Kinematic Variables: Altitude (h), Airspeed (Va), Vertical Rate (h_dot), Pitch Angle (θ), Roll Angle (ϕ), Heading (ψ)\nControl & Energy: Throttle (N1), Load Factor (G), Runway Distance (d_rw), Wind (Vw), Specific Energy Index (E), Flight Phase Code (κ)\nEnforces Zero Lookahead Leakage via Pre-Fitted Train Scaler Parameters",
    facecolor="#FFFFFF", edgecolor="#D35400", title_color="#A04000", fontsize=9.4
)

# Arrow from Feature Synthesizer down to Stage 3 Unified Dataset (from 8.55 to 7.20)
draw_arrow(8.0, 8.55, 8.0, 7.20, color="#D35400")

# -------------------------------------------------------------
# STAGE 3: UNIFIED HIGH-DIMENSIONAL SEQUENCE DATASET
# -------------------------------------------------------------
# Container: Y in [5.4, 7.7]
# Title at x = 0.9 (left aligned): "STAGE 3: UNIFIED SEQUENCE TENSOR" (spans x in [0.9, 4.6]).
# Arrow arrives at x = 8.0. Clear gap of 3.4 units!
draw_section_card(0.5, 5.4, 15.0, 2.3, "STAGE 3", "UNIFIED SEQUENCE TENSOR", GREEN_BG, GREEN_BD, title_x=0.9, title_align="left")

# Card: Y in [5.65, 7.20]
draw_card(
    1.0, 5.65, 14.0, 1.45,
    "Leakage-Free Standardized Sequential Tensor: X ∈ ℝ^(9,526 × 15 × 12)",
    "Total Ingested Sequences: N = 9,526  |  Temporal Horizon: T = 15 Timesteps  |  Feature Channels: F = 12\nTemporal Stride: S = 5 timesteps  |  Uniform Temporal Discretization Across Heterogeneous Trajectories",
    facecolor="#FFFFFF", edgecolor="#1E8449", title_color="#145A32", fontsize=10.0
)

# Vertical arrows from Stage 3 down to Stage 4 partitions
# Left arrow: at x = 4.3, drops from 5.65 down to 4.45 into Training card
# Right arrow: at x = 11.7, drops from 5.65 down to 4.45 into Testing card
draw_arrow(4.3, 5.65, 4.3, 4.45, color="#1F618D")
draw_arrow(11.7, 5.65, 11.7, 4.45, color="#D35400")

# -------------------------------------------------------------
# STAGE 4: LEAKAGE-CONTROLLED SEGMENT-LEVEL SPLIT (80 : 20)
# -------------------------------------------------------------
# Container: Y in [2.9, 4.9]
# Left arrow at x = 4.3, Right arrow at x = 11.7.
# Title placed at x = 5.8 (left aligned): "STAGE 4: TRAIN / TEST SPLIT" (spans x in [5.8, 8.8]).
# Left arrow is at 4.3 (< 5.8), Right arrow is at 11.7 (> 8.8). Zero collision!
draw_section_card(0.5, 2.9, 15.0, 2.0, "STAGE 4", "TRAIN / TEST SPLIT (80 : 20)", RED_BG, RED_BD, title_x=5.8, title_align="left")

# Cards: Y in [3.10, 4.45]
draw_card(
    0.9, 3.10, 6.8, 1.35,
    "Training Partition (80%)",
    "7,620 Sequential Samples (Fit split only)\nUsed exclusively for model optimization and scaling statistics",
    facecolor="#FFFFFF", edgecolor="#2980B9", title_color="#1B4F72", fontsize=9.3
)

draw_card(
    8.3, 3.10, 6.8, 1.35,
    "Held-Out Testing Partition (20%)",
    "1,906 Sequential Samples (Strictly unseen)\nIsolates entire flight trajectories to prevent temporal autocorrelation leakage",
    facecolor="#FFFFFF", edgecolor="#D35400", title_color="#A04000", fontsize=9.3
)

# -------------------------------------------------------------
# STAGE 5: DATASET COMPOSITION BREAKDOWN
# -------------------------------------------------------------
# Container: Y in [0.4, 2.5]
draw_section_card(0.5, 0.4, 15.0, 2.1, "STAGE 5", "MULTI-SOURCE CORPUS COMPOSITION & SEVERITY DISTRIBUTION", PURPLE_BG, PURPLE_BD, title_x=8.0, title_align="center")

composition = [
    ("NASA ASRS Reports", 6978, "#2980B9", "73.2%"),
    ("Normal Envelopes", 1716, "#8E44AD", "18.0%"),
    ("OpenSky Squawk 7700", 832, "#27AE60", "8.7%")
]

total_count = sum(item[1] for item in composition)
bar_left = 1.0
bar_y = 1.55
bar_width = 14.0
bar_height = 0.50

# Draw Stacked Progress Bar
for name, count, color, pct in composition:
    w_seg = bar_width * (count / total_count)
    ax.barh(
        bar_y, w_seg, left=bar_left, height=bar_height,
        color=color, edgecolor="#FFFFFF", linewidth=1.5, zorder=3
    )
    # Centered bold percentage inside the bar
    ax.text(
        bar_left + w_seg / 2, bar_y,
        pct,
        ha="center", va="center",
        fontsize=10.5, fontweight="bold",
        color="#FFFFFF", zorder=4
    )
    bar_left += w_seg

# Clean 3-Column Legend Badges below the stacked bar
badge_centers = [3.3, 8.0, 12.7]
for i, (name, count, color, pct) in enumerate(composition):
    cx = badge_centers[i]
    rect = FancyBboxPatch(
        (cx - 2.1, 1.00), 0.35, 0.24,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=color, edgecolor="#FFFFFF", linewidth=1.0, zorder=3
    )
    ax.add_patch(rect)
    ax.text(
        cx - 1.6, 1.12,
        f"{name}: {count:,} ({pct})",
        ha="left", va="center",
        fontsize=9.2, fontweight="bold",
        color="#1C2833", zorder=4
    )

# Summary Caption at bottom
ax.text(
    8.0, 0.65,
    f"Total Ingested Sequences: {total_count:,} Continuous Windows  |  12 Standardized Aerospace Parameters  |  Zero Data Leakage",
    ha="center", va="center",
    fontsize=10.0, fontweight="bold", color="#0B2545"
)

# -------------------------------------------------------------
# Save to Filesystem
# -------------------------------------------------------------
root_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(root_dir, "backend", "saved_models")

out_root_png = os.path.join(root_dir, "combined_dataset_flowchart.png")
out_root_pdf = os.path.join(root_dir, "combined_dataset_flowchart.pdf")
out_backend_png = os.path.join(backend_dir, "combined_dataset_flowchart.png")
out_backend_pdf = os.path.join(backend_dir, "combined_dataset_flowchart.pdf")

plt.savefig(out_root_png, dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig(out_root_pdf, dpi=300, bbox_inches="tight", facecolor="white")

if os.path.exists(backend_dir):
    plt.savefig(out_backend_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(out_backend_pdf, dpi=300, bbox_inches="tight", facecolor="white")

print("Fixed dataset flowchart saved successfully to:")
print(" -", out_root_png)
print(" -", out_backend_png)
plt.close()
