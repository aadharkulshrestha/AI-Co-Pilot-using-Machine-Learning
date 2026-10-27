import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# -------------------------------------------------------------
# Canvas Setup - IEEE/AIAA Transactions Format (300 DPI)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(16, 21.5), dpi=300)
ax.set_xlim(0, 16)
ax.set_ylim(0, 22.0)
ax.axis("off")

# Title Header
fig.suptitle(
    "AI Co-Pilot: Telemetry Dataset & Machine Learning System Architecture",
    fontsize=18,
    fontweight="bold",
    color="#0B2545",
    y=0.988
)

ax.text(
    8.0, 21.25,
    "End-to-End Pipeline: Multi-Modal Ingestion, BiLSTM Action Classification, Continuous Risk Regression & Dual Anomaly Detection",
    ha="center", va="center",
    fontsize=10.5,
    fontstyle="italic",
    color="#475569"
)

# -------------------------------------------------------------
# Premium Academic Color Tokens
# -------------------------------------------------------------
NAVY_BG   = "#F0F4F8"
NAVY_BD   = "#1A365D"
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
CARD_WHITE = "#FFFFFF"
TEXT_DARK  = "#1C2833"

def draw_tier_container(x, y, w, h, tier_num, tier_title, bgcolor, border_color):
    """Draw an outer tier boundary with header banner."""
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.08,rounding_size=0.22",
        facecolor=bgcolor,
        edgecolor=border_color,
        linewidth=1.4,
        alpha=0.5,
        zorder=1
    )
    ax.add_patch(patch)

    ax.text(
        8.0, y + h - 0.32,
        f"TIER {tier_num}: {tier_title}",
        ha="center", va="center",
        fontsize=12,
        fontweight="bold",
        color="#0B2545",
        zorder=2
    )

def draw_node(x, y, w, h, title, body_lines, facecolor=CARD_WHITE, edgecolor="#2C3E50", title_color="#0B2545", fontsize=9.0, linewidth=1.3, zorder=3):
    """Draw a rounded architecture node card with header and bulleted content."""
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.06,rounding_size=0.16",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder
    )
    ax.add_patch(patch)

    # Title
    ax.text(
        x + w / 2, y + h - 0.25, title,
        ha="center", va="top",
        fontsize=fontsize + 0.6,
        fontweight="bold",
        color=title_color,
        zorder=zorder + 1
    )

    # Body
    ax.text(
        x + w / 2, y + 0.14, body_lines,
        ha="center", va="bottom",
        fontsize=fontsize - 0.4,
        color="#334155",
        wrap=True,
        linespacing=1.22,
        zorder=zorder + 1
    )

def draw_conn(x1, y1, x2, y2, color="#2C3E50", lw=1.5, rad=0.0):
    """Draw a clean directional connection arrow between architecture blocks."""
    conn = f"arc3,rad={rad}" if rad != 0 else "arc3,rad=0"
    patch = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=lw,
        color=color,
        connectionstyle=conn,
        zorder=2
    )
    ax.add_patch(patch)

# =============================================================
# TIER 1: MULTI-MODAL AEROSPACE DATASET & FEATURE PIPELINE
# Y in [15.3, 20.9] (h = 5.6)
# =============================================================
draw_tier_container(0.5, 15.3, 15.0, 5.6, "1", "MULTI-SOURCE AVIATION DATA INGESTION & FEATURE ENGINEERING PIPELINE", BLUE_BG, BLUE_BD)

# Row 1: Three Raw Ingestion Sources (Y in [18.25, 20.10], h = 1.85)
draw_node(
    0.9, 18.25, 4.4, 1.85,
    "NASA ASRS Database",
    "6,978 Cleaned Safety Narratives\nMulti-system operational failure reports\nFlight crew human-factor context & narratives\nTarget Labels: 9 Emergency SOP Recovery Classes",
    edgecolor="#1F618D", title_color="#1B4F72"
)

draw_node(
    5.8, 18.25, 4.4, 1.85,
    "OpenSky Squawk 7700 Trajectories",
    "832 Declared Emergency Flight Segments\nADS-B transponder telemetry (1 Hz sampled)\nDynamic 30-min temporal gap segmentation\nKinematics: Altitude, groundspeed, track, sink rate",
    edgecolor="#229954", title_color="#196F3D"
)

draw_node(
    10.7, 18.25, 4.4, 1.85,
    "Synthetic Normal Envelopes",
    "1,716 Nominal Baseline Flight Profiles\nAerodynamically bounded climb, cruise, descent\nClean negative-class flight baseline sequences\nZero-alarm calibration baseline for anomaly detectors",
    edgecolor="#884EA0", title_color="#6C3483"
)

# Connectors from Row 1 down to Feature Synthesizer / Partitioning
draw_conn(3.1, 18.25, 4.4, 17.20, color="#1F618D")
draw_conn(8.0, 18.25, 8.0, 17.20, color="#229954")
draw_conn(12.9, 18.25, 11.6, 17.20, color="#884EA0")

# Row 2: 12-D Feature Synthesis & Sliding Window Partitioning (Y in [15.65, 17.20], h = 1.55)
draw_node(
    0.9, 15.65, 6.9, 1.55,
    "12-Parameter Aerodynamic Feature Synthesizer",
    "Kinematics: Altitude (h), Airspeed (Va), Vertical Rate (h_dot), Pitch (θ), Roll (ϕ), Heading (ψ)\nControl & Energy: Engine (N1), Load Factor (G), Runway Dist (d_rw), Wind (Vw), Energy Index (E), Phase (κ)\nZero-Leakage Standardizer: Z-score parameters (μ_train, σ_train) fitted strictly on training partition",
    edgecolor="#D35400", title_color="#A04000", fontsize=8.8
)

draw_node(
    8.2, 15.65, 6.9, 1.55,
    "Sliding Window Tensor & Flight-Segment Partition",
    "Sequence Structuring: T = 15 Timesteps, Stride S = 5 → Sequential Tensor X ∈ ℝ^(B × 15 × 12)\nSegment-Level Split (80:20): 7,620 Training Sequences | 1,906 Held-Out Test Sequences\nStrict flight-segment isolation completely eliminates temporal autocorrelation leakage",
    edgecolor="#1E8449", title_color="#145A32", fontsize=8.8
)

# Arrow between Synthesizer and Sliding Window
draw_conn(7.8, 16.42, 8.2, 16.42, color="#D35400")

# Connectors from Tier 1 down into Tier 2 Models (Branching cleanly into the 3 Model Columns)
draw_conn(3.1, 15.65, 3.1, 14.30, color="#1F618D")
draw_conn(8.0, 15.65, 8.0, 14.30, color="#D35400")
draw_conn(12.9, 15.65, 12.9, 14.30, color="#1E8449")

# =============================================================
# TIER 2: MACHINE LEARNING & DEEP LEARNING MODEL SUITES
# Y in [5.4, 14.5] (h = 9.1)
# =============================================================
draw_tier_container(0.5, 5.4, 15.0, 9.1, "2", "MACHINE LEARNING & DEEP LEARNING MODEL ARCHITECTURES", ORANGE_BG, ORANGE_BD)

# -------------------------------------------------------------
# COLUMN 1: BiLSTM + Recency Self-Attention Recovery Classifier
# (x = 0.8, w = 4.6)
# -------------------------------------------------------------
draw_node(
    0.8, 13.25, 4.6, 0.75,
    "MODEL 1: BiLSTM ACTION CLASSIFIER",
    "Supervised Sequence Classification (9 Emergency SOP Classes)",
    facecolor="#EBF5FB", edgecolor="#2980B9", title_color="#1B4F72", fontsize=9.2
)

draw_node(
    0.8, 11.50, 4.6, 1.40,
    "Input Projection & Latent Space",
    "Input Tensor: X ∈ ℝ^(15 × 12)\nWeight: W_in ∈ ℝ^(12 × 64), Bias: b_in ∈ ℝ^64\nActivation: H^(0) = ReLU(X · W_in + b_in) ∈ ℝ^(15 × 64)\nProjects 12 aerodynamic channels into latent representation",
    edgecolor="#2980B9", title_color="#1B4F72"
)
draw_conn(3.1, 13.25, 3.1, 12.90, color="#2980B9")
draw_conn(3.1, 11.50, 3.1, 11.15, color="#2980B9")

draw_node(
    0.8, 9.75, 4.6, 1.40,
    "Bi-Directional Temporal Mixing",
    "Forward Pass: h_fwd_t = ReLU(H_t + W_fwd · h_fwd_t-1)\nBackward Pass: h_bwd_t = ReLU(H_t + W_bwd · h_bwd_t+1)\nSequence Representation: H_seq = ½(H_fwd + H_bwd) ∈ ℝ^(15 × 64)\nCaptures both onset trends and forward rate trajectories",
    edgecolor="#2980B9", title_color="#1B4F72"
)
draw_conn(3.1, 9.75, 3.1, 9.40, color="#2980B9")

draw_node(
    0.8, 8.00, 4.6, 1.40,
    "Recency-Biased Self-Attention",
    "Attention Energy: a_t = v_attᵀ tanh(W_att · H_t) + λ_recency\nNormalized Weights: α = Softmax(a) ∈ ℝ^15\nPooled Context: c = ∑ α_t H_t ∈ ℝ^64\nFused State Vector: [c || h_15] ∈ ℝ^128 (Context + Current)",
    edgecolor="#2980B9", title_color="#1B4F72"
)
draw_conn(3.1, 8.00, 3.1, 7.65, color="#2980B9")

draw_node(
    0.8, 5.75, 4.6, 1.90,
    "Classification Head & Performance",
    "Dense Layer: FC1 = ReLU([c || h_15] · W_fc1 + b_fc1) ∈ ℝ^64\nSoftmax Head: P(y | X) = Softmax(FC1 · W_cls + b_cls) ∈ ℝ^9\nOutputs: 9 SOP Recovery Directives (Stall, CFIT, Flameout...)\nAccuracy: 98.85%  |  AUC-ROC: 0.988  |  F1: 98.81%\nPrecision: 98.82%  |  Recall: 98.85%  |  Loss: 0.052",
    edgecolor="#1A5276", title_color="#0E6251", facecolor="#EBF7F0"
)

# -------------------------------------------------------------
# COLUMN 2: Continuous Aerospace Risk Regressor
# (x = 5.7, w = 4.6)
# -------------------------------------------------------------
draw_node(
    5.7, 13.25, 4.6, 0.75,
    "MODEL 2: CONTINUOUS RISK REGRESSOR",
    "Physics-Guided Envelope Degradation & Energy Analysis",
    facecolor="#FEF9E7", edgecolor="#D35400", title_color="#A04000", fontsize=9.2
)

draw_node(
    5.7, 11.50, 4.6, 1.40,
    "Kinematic Boundary Penalties",
    "Instantaneous Margin Degradation:\n• Sink Rate: r_vrate = min(40, (|h_dot|/4000)^1.8 · 40) if h_dot < -1000\n• Stall Airspeed Decay: r_spd (Vs < 115 kts, Vmo > 340 kts)\n• Attitude Limit: r_pitch (> 18° or < -10°), r_roll (> 35° bank)",
    edgecolor="#D35400", title_color="#A04000"
)
draw_conn(8.0, 13.25, 8.0, 12.90, color="#D35400")
draw_conn(8.0, 11.50, 8.0, 11.15, color="#D35400")

draw_node(
    5.7, 9.75, 4.6, 1.40,
    "Aerodynamic Energy & Load Penalty",
    "• Structural Load Factor: r_g = |G - 1.0| > 0.4\n• Terrain Closure Alert: Altitude < 1,200 ft & Distance > 4 nm\n• Severe Windshear Velocity Factor: Wind > 35 kts\n• Specific Mechanical Energy Rate: dE/dt = h_dot + (Va/g)·Va_dot",
    edgecolor="#D35400", title_color="#A04000"
)
draw_conn(8.0, 9.75, 8.0, 9.40, color="#D35400")

draw_node(
    5.7, 8.00, 4.6, 1.40,
    "Composite Hazard Formulation",
    "Weighted Aerospace Multi-Factor Penalty:\nR_comp = ∑ (w_i · r_i)  |  i ∈ {1..12}\nOperational Severity Multiplier: Max(R_comp · 1.5, 75.0)\nBounding Clamp: R = min(100.0, max(5.0, Composite))",
    edgecolor="#D35400", title_color="#A04000"
)
draw_conn(8.0, 8.00, 8.0, 7.65, color="#D35400")

draw_node(
    5.7, 5.75, 4.6, 1.90,
    "Risk Regression Performance",
    "Continuous Safety Index: R ∈ [0, 100] Points\nCoefficient of Determination: R² = 0.972 (97.2% Variance)\nRegression Errors: RMSE = 3.12, MAE = 2.50, MSE = 9.73\nResidual Error Distribution: Zero-centered Gaussian (μ=0.04)\nSeverity Tiers: Low (<25), Medium (25-50), High (50-75), Critical (≥75)",
    edgecolor="#B9770E", title_color="#7D6608", facecolor="#FEFDE8"
)

# -------------------------------------------------------------
# COLUMN 3: Dual Temporal Anomaly Detection Engine
# (x = 10.6, w = 4.6)
# -------------------------------------------------------------
draw_node(
    10.6, 13.25, 4.6, 0.75,
    "MODEL 3: DUAL ANOMALY DETECTORS",
    "Unsupervised Trajectory Outlier & Discrepancy Scoring",
    facecolor="#F4ECF7", edgecolor="#8E44AD", title_color="#5B2C6F", fontsize=9.2
)

draw_node(
    10.6, 11.50, 4.6, 1.40,
    "Path A: LSTM Autoencoder (Reconstruction)",
    "Encoder: LSTM(12 → 64) → Latent Bottleneck z ∈ ℝ^64\nDecoder: RepeatVector(15) → LSTM(64 → 64) → Linear(64 → 12)\nReconstruction Output: X_hat ∈ ℝ^(15 × 12)\nPointwise Residual Metric: MAE(X, X_hat) = (1/TF) ∑∑ |x - x_hat|",
    edgecolor="#8E44AD", title_color="#5B2C6F"
)
draw_conn(12.9, 13.25, 12.9, 12.90, color="#8E44AD")
draw_conn(12.9, 11.50, 12.9, 11.15, color="#8E44AD")

draw_node(
    10.6, 9.75, 4.6, 1.40,
    "Path A Anomaly Threshold & Flags",
    "Calibration Threshold: τ_AE = 0.0825 (99th validation percentile)\nHeld-Out Test Anomaly Frequency: 1.41% of windows (1,928 flagged)\nTest Reconstruction Metrics: Mean MAE = 0.0163, RMSE = 0.0248\nIdentifies sharp pointwise kinematic discontinuities",
    edgecolor="#8E44AD", title_color="#5B2C6F"
)
draw_conn(12.9, 9.75, 12.9, 9.40, color="#8E44AD")

draw_node(
    10.6, 8.00, 4.6, 1.40,
    "Path B: Anomaly Transformer (Association)",
    "Series Association (Self-Attention): S_assoc ∈ ℝ^(15 × 15)\nPrior Association (Gaussian Kernel): P_assoc ∈ ℝ^(15 × 15)\nDiscrepancy: D_KL(S || P) + D_KL(P || S) (Symmetric KL Divergence)\nPenalizes breakdown in temporal correlation structure",
    edgecolor="#8E44AD", title_color="#5B2C6F"
)
draw_conn(12.9, 8.00, 12.9, 7.65, color="#8E44AD")

draw_node(
    10.6, 5.75, 4.6, 1.90,
    "Path B Scoring & Verification",
    "Composite Score: MSE · Softmax(-D_KL(S || P))\nCalibration Threshold: τ_Trans = 0.003329 (99th percentile)\nHeld-Out Test Anomaly Frequency: 11.33% of windows (15,552 flagged)\nMulti-Variate Attribution: Isolates failing sensor channels\nDual Detection Fusion: Pointwise MAE + Relational Shifts",
    edgecolor="#6C3483", title_color="#4A235A", facecolor="#F5EEF8"
)

# Connectors from Tier 2 models down into Tier 3 Inference Engine
draw_conn(3.1, 5.75, 3.1, 4.70, color="#2980B9")
draw_conn(8.0, 5.75, 8.0, 4.70, color="#D35400")
draw_conn(12.9, 5.75, 12.9, 4.70, color="#8E44AD")

# =============================================================
# TIER 3: IN-MEMORY NUMPY RUNTIME & REAL-TIME AVIONICS OUTPUTS
# Y in [0.5, 4.9] (h = 4.4)
# =============================================================
draw_tier_container(0.5, 0.5, 15.0, 4.4, "3", "IN-MEMORY NUMPY INFERENCE RUNTIME & REAL-TIME AVIONICS INTEGRATION", GREEN_BG, GREEN_BD)

# Sub-row 1: In-Memory Pure NumPy Engine (Y in [3.05, 4.40], h = 1.35)
draw_node(
    0.8, 3.05, 14.4, 1.35,
    "Pure NumPy In-Memory Inference Engine (Zero External Framework Overhead in Flight Loop)",
    "Cockpit Execution Engine: Fully vectorized C-contiguous NumPy runtime with pre-allocated activation buffers\nSingle-Window Inference Latency: 2.1 ms  |  Peak P95 Latency: 2.6 ms  |  Telemetry Throughput: 445 Windows/s (445 FPS)\nEmbedded Compatibility: 20 Hz Avionics telemetry budget is 50 ms → Consumes only 4.2% of cockpit compute slice\nMemory Footprint: < 1.2 GB RAM  |  Deterministic Float32 matrix operations on edge avionic processors",
    facecolor="#F2F4F4", edgecolor="#1B4F72", title_color="#0B2545", fontsize=9.2
)

# Connectors from NumPy Runtime down to the 3 Structured Decision Outputs
draw_conn(3.1, 3.05, 3.1, 2.45, color="#1F618D")
draw_conn(8.0, 3.05, 8.0, 2.45, color="#D35400")
draw_conn(12.9, 3.05, 12.9, 2.45, color="#1E8449")

# Sub-row 2: Three Structured Avionics Decision Outputs (Y in [0.75, 2.45], h = 1.70)
draw_node(
    0.8, 0.75, 4.6, 1.70,
    "1. SOP Recovery Directive",
    "Optimal Standard Operating Procedure Action\nProbability Distribution: P(y_k | X) ≥ 0.95\n9 Standardized Actions: Stall Recovery, CFIT Terrain\nPull-Up, Windshear Escape, Engine Flameout\nRestart, Overspeed Deceleration, Severe Turbulence",
    edgecolor="#1F618D", title_color="#1B4F72", facecolor="#EBF5FB"
)

draw_node(
    5.7, 0.75, 4.6, 1.70,
    "2. Continuous Risk Gauge",
    "Continuous Dynamic Risk Score R ∈ [0, 100]\nHigh-Fidelity Model Fit: R² = 0.972, RMSE = 3.12, MAE = 2.50\nFour Operational Tiers: Nominal (<25), Moderate\n(25-50), Elevated (50-75), Critical Emergency (≥75)\nZero-lag real-time aerodynamic margin indicator",
    edgecolor="#D35400", title_color="#A04000", facecolor="#FEF9E7"
)

draw_node(
    10.6, 0.75, 4.6, 1.70,
    "3. Dual Anomaly Verification",
    "Statistical Anomaly Trigger (Pointwise + Relational)\nDual Threshold Alarms: MAE (1.41%) + Assoc (11.33%)\nSensor Feature Attribution: Identifies specific deviating\nflight parameters for engineering telemetry review\nUnsupervised safety envelope boundary monitoring",
    edgecolor="#1E8449", title_color="#145A32", facecolor="#EBF7F0"
)

# -------------------------------------------------------------
# Save Images
# -------------------------------------------------------------
root_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(root_dir, "backend", "saved_models")

out_root_png = os.path.join(root_dir, "ml_architecture_diagram.png")
out_root_pdf = os.path.join(root_dir, "ml_architecture_diagram.pdf")
out_backend_png = os.path.join(backend_dir, "ml_architecture_diagram.png")
out_backend_pdf = os.path.join(backend_dir, "ml_architecture_diagram.pdf")

plt.savefig(out_root_png, dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig(out_root_pdf, dpi=300, bbox_inches="tight", facecolor="white")

if os.path.exists(backend_dir):
    plt.savefig(out_backend_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(out_backend_pdf, dpi=300, bbox_inches="tight", facecolor="white")

print("ML Architecture Diagram generated successfully:")
print(" -", out_root_png)
print(" -", out_backend_png)
plt.close()
