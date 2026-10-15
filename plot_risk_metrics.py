
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

# -----------------------------------------
# 1. Project metrics
# -----------------------------------------
np.random.seed(42)

N = 1906
TARGET_R2 = 0.972
TARGET_RMSE = 3.12

# Synthetic illustrative data, NOT actual model predictions.
# Construct data to match both reported metrics mathematically.
target_variance = TARGET_RMSE**2 / (1 - TARGET_R2)

y_true = np.random.normal(50, 1, N)
y_true = (y_true - y_true.mean()) / y_true.std()
y_true = y_true * np.sqrt(target_variance) + 50

# Generate errors independent of the true scores
errors = np.random.normal(0, 1, N)

# Remove error mean and its linear correlation with y_true
x = y_true - y_true.mean()
errors = errors - errors.mean()
errors = errors - (np.dot(errors, x) / np.dot(x, x)) * x
errors = errors - errors.mean()

# Scale errors to achieve the target RMSE
errors = errors / np.sqrt(np.mean(errors**2)) * TARGET_RMSE

y_pred = y_true + errors

# Calculate metrics
r2 = r2_score(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print(f"R² Score: {r2:.3f}")
print(f"RMSE: {rmse:.2f} points")
print(f"Test samples: {N}")

# -----------------------------------------
# 2. Create two-panel visualization
# -----------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

fig.suptitle(
    "AI Co-Pilot: Risk Score Regression Results",
    fontsize=19,
    fontweight="bold",
    color="#12366b"
)

# LEFT: True vs Predicted scores
ax = axes[0]

ax.scatter(
    y_true, y_pred,
    alpha=0.55,
    s=20,
    label="Test samples"
)

limits = [
    min(y_true.min(), y_pred.min()),
    max(y_true.max(), y_pred.max())
]

ax.plot(
    limits, limits,
    "r--",
    linewidth=2,
    label="Perfect prediction"
)

ax.set_title(
    f"True vs Predicted Risk Scores (R² = {r2:.3f})",
    fontsize=13,
    fontweight="bold"
)
ax.set_xlabel("True Risk Score", fontsize=12)
ax.set_ylabel("Predicted Risk Score", fontsize=12)
ax.grid(True, linestyle="--", alpha=0.35)
ax.legend()

ax.text(
    0.05, 0.95,
    f"R² Score = {r2:.3f}\n"
    "Higher is better; 1.0 is perfect",
    transform=ax.transAxes,
    va="top",
    fontsize=11,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
)

# RIGHT: Prediction error distribution
ax = axes[1]
prediction_errors = y_pred - y_true

ax.hist(
    prediction_errors,
    bins=35,
    edgecolor="black",
    alpha=0.75
)

ax.axvline(
    0,
    color="red",
    linestyle="--",
    linewidth=2,
    label="Zero error"
)

ax.set_title(
    f"Prediction Error Distribution (RMSE = {rmse:.2f})",
    fontsize=13,
    fontweight="bold"
)
ax.set_xlabel("Prediction Error (Predicted - True)", fontsize=12)
ax.set_ylabel("Number of Samples", fontsize=12)
ax.grid(True, linestyle="--", alpha=0.3)
ax.legend()

ax.text(
    0.05, 0.95,
    f"RMSE = {rmse:.2f} points\n"
    "Lower is better; 0 is perfect",
    transform=ax.transAxes,
    va="top",
    fontsize=11,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
)

fig.text(
    0.5, 0.015,
    "Illustrative synthetic data — not actual model evaluation predictions",
    ha="center",
    fontsize=10,
    color="dimgray"
)

plt.tight_layout(rect=[0, 0.045, 1, 0.92])

# Save high-resolution figure
plt.savefig(
    "risk_score_regression_results.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
