import os
import json
import time
import psutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

# Import the model architecture
from train_lstm_autoencoder import LSTMAutoencoder, get_project_root, apply_scaler

def calculate_errors(model, data_loader, device):
    model.eval()
    all_mae = []
    all_rmse = []
    
    start_time = time.time()
    with torch.no_grad():
        for batch in data_loader:
            x_batch = batch[0].to(device)
            out = model(x_batch)
            
            # Errors per window: shape (batch, seq, features) -> mean across seq and features
            mae = torch.mean(torch.abs(out - x_batch), dim=(1, 2))
            mse = torch.mean(torch.pow(out - x_batch, 2), dim=(1, 2))
            rmse = torch.sqrt(mse)
            
            all_mae.extend(mae.cpu().numpy())
            all_rmse.extend(rmse.cpu().numpy())
            
    total_time = time.time() - start_time
    return np.array(all_mae), np.array(all_rmse), total_time

def evaluate_model():
    root_dir = get_project_root()
    data_dir = root_dir / "backend" / "dataset" / "processed_sequences"
    models_dir = root_dir / "backend" / "saved_models"
    reports_dir = root_dir / "backend" / "reports"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")
    
    # 1. Load Data
    X_train_raw = np.load(data_dir / "X_train.npy")
    X_val_raw = np.load(data_dir / "X_val.npy")
    X_test_raw = np.load(data_dir / "X_test.npy")
    
    with open(models_dir / "lstm_ae_scaler.json", "r") as f:
        scaler_params = json.load(f)
    mean = np.array(scaler_params["mean"])
    std = np.array(scaler_params["std"])
    
    X_train = apply_scaler(X_train_raw, mean, std)
    X_val = apply_scaler(X_val_raw, mean, std)
    X_test = apply_scaler(X_test_raw, mean, std)
    
    n_features = X_train.shape[2]
    
    # 2. Load Model
    model = LSTMAutoencoder(n_features=n_features, hidden_dim=64).to(device)
    model.load_state_dict(torch.load(models_dir / "lstm_autoencoder.pth", map_location=device))
    
    # Create DataLoaders
    batch_size = 512
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32)), batch_size=batch_size)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32)), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32)), batch_size=batch_size)
    
    # 3. Calculate Errors
    print("Calculating reconstruction errors for Train...")
    train_mae, train_rmse, train_time = calculate_errors(model, train_loader, device)
    
    print("Calculating reconstruction errors for Val...")
    val_mae, val_rmse, val_time = calculate_errors(model, val_loader, device)
    
    print("Calculating reconstruction errors for Test...")
    test_mae, test_rmse, test_time = calculate_errors(model, test_loader, device)
    
    # 4. Threshold Selection (using Validation Data Only)
    # Selection rule: 99th percentile of Validation MAE
    # This assumes that up to 1% of the validation segments might be extreme anomalies.
    threshold_mae = float(np.percentile(val_mae, 99))
    print(f"Selected Anomaly Threshold (99th percentile of Val MAE): {threshold_mae:.4f}")
    
    # 5. Evaluate held-out Test set
    test_anomalies_flagged = int(np.sum(test_mae > threshold_mae))
    test_anomaly_rate = test_anomalies_flagged / len(test_mae)
    print(f"Test windows flagged as anomalies: {test_anomalies_flagged} ({test_anomaly_rate*100:.2f}%)")
    
    # 6. Performance Benchmarks
    # Measure latency on a batch of size 1 (simulating real-time inference)
    latency_loader = DataLoader(TensorDataset(torch.tensor(X_test[:1000], dtype=torch.float32)), batch_size=1)
    latencies = []
    model.eval()
    with torch.no_grad():
        for batch in latency_loader:
            x_b = batch[0].to(device)
            start = time.perf_counter()
            _ = model(x_b)
            latencies.append((time.perf_counter() - start) * 1000) # ms
            
    latencies = np.array(latencies)
    avg_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))
    throughput = 1000.0 / avg_latency if avg_latency > 0 else 0
    
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / (1024 * 1024)
    
    # 7. Compile Report
    def get_distribution_stats(errors):
        return {
            "mean": float(np.mean(errors)),
            "std": float(np.std(errors)),
            "min": float(np.min(errors)),
            "max": float(np.max(errors)),
            "p50": float(np.percentile(errors, 50)),
            "p95": float(np.percentile(errors, 95)),
            "p99": float(np.percentile(errors, 99))
        }

    results = {
        "hardware_and_environment": {
            "device": str(device),
            "memory_usage_mb": float(memory_mb)
        },
        "performance_benchmarks": {
            "batch_size_tested": 1,
            "average_inference_latency_ms": avg_latency,
            "p95_inference_latency_ms": p95_latency,
            "throughput_fps": throughput
        },
        "threshold_selection": {
            "rule": "99th percentile of Validation MAE",
            "threshold_value": threshold_mae
        },
        "reconstruction_error_mae": {
            "train": get_distribution_stats(train_mae),
            "val": get_distribution_stats(val_mae),
            "test": get_distribution_stats(test_mae)
        },
        "test_evaluation": {
            "total_windows": len(test_mae),
            "flagged_windows": test_anomalies_flagged,
            "flagged_percentage": float(test_anomaly_rate * 100)
        },
        "limitations_and_disclaimers": [
            "Supervised metrics (Accuracy, Precision, Recall, F1) are NOT reported because independent, validated event labels are missing for these windows.",
            "Flagged anomalies represent statistical outliers in trajectory reconstruction, NOT confirmed safety emergencies. A human-in-the-loop or verified label database is required for true emergency confirmation.",
            "The training dataset contains emergency squawk 7700 flights, meaning the Autoencoder was exposed to anomalous data during training. The threshold essentially flags 'the most extreme 1% of behaviors' within an already-biased emergency population, rather than identifying rare anomalies within normal flights."
        ]
    }
    
    with open(reports_dir / "experiment_e1_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    md_report = f"""# Experiment E1: LSTM Autoencoder Anomaly Detection

## 1. Hardware & Performance
- **Inference Device**: {device}
- **Average Latency (Batch=1)**: {avg_latency:.2f} ms
- **95th Percentile Latency (P95)**: {p95_latency:.2f} ms
- **Throughput**: {throughput:.2f} windows/sec
- **Process Memory**: {memory_mb:.1f} MB

## 2. Threshold Selection
- **Rule**: 99th percentile of Validation Mean Absolute Error (MAE)
- **Selected Threshold**: {threshold_mae:.4f}

## 3. Reconstruction Error (MAE) Distributions
| Split | Mean | P50 (Median) | P95 | P99 | Max |
|---|---|---|---|---|---|
| Train | {results['reconstruction_error_mae']['train']['mean']:.4f} | {results['reconstruction_error_mae']['train']['p50']:.4f} | {results['reconstruction_error_mae']['train']['p95']:.4f} | {results['reconstruction_error_mae']['train']['p99']:.4f} | {results['reconstruction_error_mae']['train']['max']:.4f} |
| Validation | {results['reconstruction_error_mae']['val']['mean']:.4f} | {results['reconstruction_error_mae']['val']['p50']:.4f} | {results['reconstruction_error_mae']['val']['p95']:.4f} | {results['reconstruction_error_mae']['val']['p99']:.4f} | {results['reconstruction_error_mae']['val']['max']:.4f} |
| Held-out Test | {results['reconstruction_error_mae']['test']['mean']:.4f} | {results['reconstruction_error_mae']['test']['p50']:.4f} | {results['reconstruction_error_mae']['test']['p95']:.4f} | {results['reconstruction_error_mae']['test']['p99']:.4f} | {results['reconstruction_error_mae']['test']['max']:.4f} |

## 4. Test Set Evaluation
- **Total Test Windows**: {len(test_mae):,}
- **Flagged as Anomalous (MAE > {threshold_mae:.4f})**: {test_anomalies_flagged:,} ({test_anomaly_rate*100:.2f}%)

## 5. Limitations (Research Context)
1. **No Supervised Metrics Claimed:** We do not claim Precision, Recall, or F1 scores because we lack independent ground-truth labels for these specific sequence windows.
2. **Flagged $\\neq$ Confirmed Emergency:** High reconstruction error signifies a statistical outlier in the trajectory, not a confirmed aviation emergency.
3. **Training Distribution Bias:** Because the OpenSky dataset used consists of Squawk 7700 emergencies, the autoencoder learned to reconstruct *anomalous* flights. The flagged test windows represent the most extreme 1% of an already abnormal population. Future work must incorporate millions of normal flights to establish a true baseline.
"""
    
    with open(reports_dir / "experiment_e1_results.md", "w") as f:
        f.write(md_report)
        
    print(f"Evaluation complete. Reports saved to {reports_dir}.")

if __name__ == "__main__":
    evaluate_model()
