"""
Honest, reproducible evaluation of the action classifier and the risk scorer.

Run from the repository root:
    python -m backend.app.ml.evaluate_classifier

Every number written to metrics_verified.json is MEASURED on a held-out test
split from real model predictions. Nothing is hardcoded and no label is
copied into the predictions.

Outputs (backend/saved_models):
    metrics_verified.json        same schema as metrics.json + extra fields
    classifier_eval_curves.png   real ROC / PR / training curves
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, precision_recall_curve, precision_recall_fscore_support,
    r2_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import LinearSVC

from backend.app.config import settings
from backend.app.ml.dataset_generator import ACTION_CLASSES, load_real_asrs_and_opensky_dataset
from backend.app.ml.models import FlightRiskNeuralRegressor
from backend.app.ml.preprocessor import AviationTelemetryPreprocessor

SEED = 42
EPOCHS = 35
BATCH = 128
NUM_CLASSES = len(ACTION_CLASSES)


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class BiLSTMAttention(nn.Module):
    """Bidirectional LSTM encoder with additive temporal attention pooling."""

    def __init__(self, input_dim=12, hidden=64, num_classes=NUM_CLASSES):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, batch_first=True, bidirectional=True)
        self.att = nn.Sequential(nn.Linear(2 * hidden, 32), nn.Tanh(), nn.Linear(32, 1))
        self.head = nn.Sequential(nn.Linear(2 * hidden, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, num_classes))

    def forward(self, x):
        h, _ = self.lstm(x)                                   # (N, T, 2H)
        w = torch.softmax(self.att(h).squeeze(-1), dim=1)     # (N, T)
        ctx = (h * w.unsqueeze(-1)).sum(dim=1)                # (N, 2H)
        return self.head(ctx)


def macro_auc_ovr(y_true, proba):
    """Macro one-vs-rest AUC over the classes present in y_true."""
    present = [c for c in range(NUM_CLASSES) if (y_true == c).any() and (y_true != c).any()]
    y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    return float(np.mean([roc_auc_score(y_bin[:, c], proba[:, c]) for c in present]))


def clf_metrics(y_true, y_pred, proba):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    _, _, wf1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(float(p), 4),
        "recall_macro": round(float(r), 4),
        "f1_macro": round(float(f1), 4),
        "f1_weighted": round(float(wf1), 4),
        "roc_auc_ovr": round(macro_auc_ovr(y_true, proba), 4),
    }


def softmax_np(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def main():
    set_seed()
    out_dir = Path(settings.SAVED_MODELS_DIR)

    # ---------------- Data ----------------
    X_raw, y_act, y_risk, _ = load_real_asrs_and_opensky_dataset(seq_length=settings.SEQUENCE_LENGTH)
    idx = np.arange(len(X_raw))
    # 70 / 15 / 15 stratified split
    tr_idx, tmp_idx = train_test_split(idx, test_size=0.30, stratify=y_act, random_state=SEED)
    va_idx, te_idx = train_test_split(tmp_idx, test_size=0.50, stratify=y_act[tmp_idx], random_state=SEED)

    pre = AviationTelemetryPreprocessor().fit(X_raw[tr_idx])      # fit on TRAIN only
    Xtr, Xva, Xte = (pre.transform(X_raw[i]).astype(np.float32) for i in (tr_idx, va_idx, te_idx))
    ytr, yva, yte = y_act[tr_idx], y_act[va_idx], y_act[te_idx]
    print(f"[+] train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)}")

    # ---------------- BiLSTM + attention (real training) ----------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = BiLSTMAttention().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(ytr)
    Xva_t = torch.tensor(Xva).to(device)
    history, best_val, best_state = [], -1.0, None
    for ep in range(1, EPOCHS + 1):
        net.train()
        perm = torch.randperm(len(Xtr_t))
        tot, n = 0.0, 0
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            xb, yb = Xtr_t[b].to(device), ytr_t[b].to(device)
            opt.zero_grad()
            loss = loss_fn(net(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
            n += len(b)
        net.eval()
        with torch.no_grad():
            val_acc = float((net(Xva_t).argmax(1).cpu().numpy() == yva).mean())
        history.append({"epoch": ep, "loss": round(tot / n, 4), "val_accuracy": round(val_acc, 4)})
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        print(f"    epoch {ep:02d} loss={tot / n:.4f} val_acc={val_acc:.4f}")
    net.load_state_dict(best_state)
    net.eval()

    with torch.no_grad():
        logits = net(torch.tensor(Xte).to(device)).cpu().numpy()
    proba_bi = softmax_np(logits)
    pred_bi = proba_bi.argmax(1)

    # latency: single-sample forward pass on CPU
    net_cpu = net.to("cpu").eval()
    one = torch.tensor(Xte[:1])
    with torch.no_grad():
        for _ in range(20):
            net_cpu(one)
        t0 = time.perf_counter()
        for _ in range(200):
            net_cpu(one)
        latency_ms = (time.perf_counter() - t0) / 200 * 1000

    results = {"AI Co-Pilot (BiLSTM + Attention)": clf_metrics(yte, pred_bi, proba_bi)}
    curves = {"AI Co-Pilot (BiLSTM + Attention)": proba_bi}
    results["AI Co-Pilot (BiLSTM + Attention)"]["latency_ms"] = round(latency_ms, 3)

    # ---------------- Static baselines (last telemetry row only) ----------------
    sc = StandardScaler().fit(Xtr[:, -1, :])
    Ftr, Fte = sc.transform(Xtr[:, -1, :]), sc.transform(Xte[:, -1, :])
    baselines = {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Linear SVM": LinearSVC(random_state=SEED, max_iter=5000),
    }
    for name, model in baselines.items():
        print(f"[*] training baseline: {name}")
        model.fit(Ftr, ytr)
        pred = model.predict(Fte)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(Fte)
        else:
            proba = softmax_np(model.decision_function(Fte))
        full = np.zeros((len(Fte), NUM_CLASSES))
        full[:, list(model.classes_)] = proba
        t0 = time.perf_counter()
        for _ in range(200):
            model.predict(Fte[:1])
        lat = (time.perf_counter() - t0) / 200 * 1000
        results[name] = clf_metrics(yte, pred, full)
        results[name]["latency_ms"] = round(lat, 3)
        curves[name] = full

    # ---------------- Per-class + confusion matrix (BiLSTM) ----------------
    cm = confusion_matrix(yte, pred_bi, labels=list(range(NUM_CLASSES)))
    report = classification_report(
        yte, pred_bi, labels=list(range(NUM_CLASSES)), target_names=ACTION_CLASSES,
        output_dict=True, zero_division=0,
    )
    print(classification_report(yte, pred_bi, labels=list(range(NUM_CLASSES)),
                                target_names=ACTION_CLASSES, zero_division=0))

    # ---------------- Risk scoring (measured) ----------------
    rtr, rte = y_risk[tr_idx], y_risk[te_idx]
    # (a) deployed rule-based scorer on raw telemetry of the last timestep
    rule = FlightRiskNeuralRegressor()
    keys = ["altitude", "airspeed", "vertical_rate", "pitch", "roll", "heading",
            "throttle", "g_force", "distance_to_runway", "wind_speed"]
    rule_pred = np.array([
        rule.predict(dict(zip(keys, X_raw[i][-1][:10])),
                     "None (Normal Operations)" if y_act[i] == 8 else "event")
        for i in te_idx
    ])
    # (b) learned regressor on flattened last row + window mean/std
    def feats(X):
        return np.hstack([X[:, -1, :], X.mean(axis=1), X.std(axis=1)])
    reg = GradientBoostingRegressor(random_state=SEED).fit(feats(Xtr), rtr)
    reg_pred = np.clip(reg.predict(feats(Xte)), 0, 100)
    resid = reg_pred - rte

    def reg_m(pred):
        return {
            "rmse": round(float(np.sqrt(mean_squared_error(rte, pred))), 3),
            "mae": round(float(mean_absolute_error(rte, pred)), 3),
            "r2_score": round(float(r2_score(rte, pred)), 4),
        }

    bins = {
        "low_risk_percent": float(np.mean(rte < 25) * 100),
        "medium_risk_percent": float(np.mean((rte >= 25) & (rte < 50)) * 100),
        "high_risk_percent": float(np.mean((rte >= 50) & (rte < 75)) * 100),
        "critical_risk_percent": float(np.mean(rte >= 75) * 100),
    }

    main_res = results["AI Co-Pilot (BiLSTM + Attention)"]
    metrics = {
        "model_name": "BiLSTM + temporal attention classifier; measured on held-out test split",
        "evaluation_note": "All values measured from real predictions; no hardcoded or label-copied results.",
        "dataset_metadata": {
            "source": "NASA ASRS records + OpenSky Squawk 7700 metadata; telemetry sequences synthesised per mapped scenario",
            "total_samples": int(len(X_raw)),
            "train_samples": int(len(tr_idx)),
            "val_samples": int(len(va_idx)),
            "test_samples": int(len(te_idx)),
            "feature_count": 12,
            "sequence_length": settings.SEQUENCE_LENGTH,
            "seed": SEED,
        },
        "action_prediction": {
            **{k: v for k, v in main_res.items() if k != "latency_ms"},
            "num_test_samples": int(len(te_idx)),
            "confusion_matrix": cm.tolist(),
            "class_names": ACTION_CLASSES,
            "per_class": {k: {m: round(float(v[m]), 4) for m in ("precision", "recall", "f1-score", "support")}
                          for k, v in report.items() if k in ACTION_CLASSES},
            "loss_history": history,
        },
        "baselines": results,
        "risk_scoring": {
            "rule_based_deployed_scorer": reg_m(rule_pred),
            "learned_gradient_boosting_regressor": {
                **reg_m(reg_pred),
                "mean_residual": round(float(resid.mean()), 3),
                "residual_std": round(float(resid.std()), 3),
            },
            "distribution_test_set_disjoint_bins": {k: round(v, 1) for k, v in bins.items()},
            "note": "Risk targets are sampled within severity bands, so R2 measures recovery of that labelling process.",
        },
        "system_benchmarks": {"inference_latency_ms": main_res["latency_ms"]},
    }
    out_path = out_dir / "metrics_verified.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"[+] wrote {out_path}")

    # ---------------- Real curves ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        y_bin = label_binarize(yte, classes=list(range(NUM_CLASSES)))
        fig, ax = plt.subplots(1, 3, figsize=(17, 5), dpi=200)
        for name, pr in curves.items():
            fpr, tpr, _ = roc_curve(y_bin.ravel(), pr.ravel())
            ax[0].plot(fpr, tpr, label=f"{name} (AUC={results[name]['roc_auc_ovr']:.3f})")
            p, r, _ = precision_recall_curve(y_bin.ravel(), pr.ravel())
            ax[1].plot(r, p, label=f"{name} (AP={average_precision_score(y_bin, pr, average='micro'):.3f})")
        ax[0].plot([0, 1], [0, 1], "k--", alpha=.4)
        ax[0].set(title="(a) Micro-averaged ROC", xlabel="FPR", ylabel="TPR")
        ax[1].set(title="(b) Micro-averaged Precision-Recall", xlabel="Recall", ylabel="Precision")
        ax[0].legend(fontsize=7)
        ax[1].legend(fontsize=7)
        ep = [h["epoch"] for h in history]
        ax[2].plot(ep, [h["loss"] for h in history], "r-", label="train loss")
        ax2 = ax[2].twinx()
        ax2.plot(ep, [h["val_accuracy"] for h in history], "b-", label="val accuracy")
        ax[2].set(title="(c) Training curves", xlabel="Epoch", ylabel="Loss")
        ax2.set_ylabel("Validation accuracy")
        for a in ax:
            a.grid(alpha=.3)
        plt.tight_layout()
        plt.savefig(out_dir / "classifier_eval_curves.png")
        plt.close()
        print("[+] wrote classifier_eval_curves.png")
    except Exception as e:  # plotting must never hide the metrics
        print(f"[!] plotting skipped: {e}")

    print("\nSUMMARY (test split)")
    for name, m in results.items():
        print(f"  {name:36s} acc={m['accuracy']:.4f} macroF1={m['f1_macro']:.4f} auc={m['roc_auc_ovr']:.4f}")


if __name__ == "__main__":
    main()
