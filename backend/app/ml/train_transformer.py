import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from backend.app.ml.dataset_generator import generate_synthetic_dataset
from backend.app.ml.preprocessor import AviationTelemetryPreprocessor
from backend.app.ml.transformer_model import FlightActionTransformer
from backend.app.config import settings

def train_transformer_model():
    print("[*] Generating synthetic sequence dataset for Transformer Training (DEMO MODE)...")
    # Generate smaller demo set for fast execution
    X_raw, y_raw = generate_synthetic_dataset(num_samples=800, seq_length=settings.SEQUENCE_LENGTH)
    
    print("[*] Preprocessing data...")
    preprocessor = AviationTelemetryPreprocessor()
    X_norm = preprocessor.fit_transform(X_raw)
    
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y_raw, test_size=0.2, random_state=42)
    
    # Convert to PyTorch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    print("[*] Initializing Transformer Model...")
    model = FlightActionTransformer(
        input_dim=X_norm.shape[2],
        d_model=64,
        num_heads=4,
        num_layers=2,
        num_classes=9,
        dropout=0.1
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    epochs = 15
    print(f"[*] Training Transformer for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            logits, _ = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}")
            
    print("[*] Evaluating Model...")
    model.eval()
    with torch.no_grad():
        logits, _ = model(X_test_t)
        probs = torch.softmax(logits, dim=1).numpy()
        preds = np.argmax(probs, axis=1)
        y_test_np = y_test_t.numpy()
        
        acc = accuracy_score(y_test_np, preds)
        prec = precision_score(y_test_np, preds, average='weighted', zero_division=0)
        rec = recall_score(y_test_np, preds, average='weighted', zero_division=0)
        f1 = f1_score(y_test_np, preds, average='weighted', zero_division=0)
        try:
            auc = roc_auc_score(y_test_np, probs, multi_class='ovr')
        except ValueError:
            auc = 0.5
            
    print(f"    Accuracy: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")
    
    # Save Model and Metrics
    save_dir = Path("backend/saved_models/transformer")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), save_dir / "transformer_weights.pth")
    preprocessor.save(save_dir / "preprocessor.json")
    
    metrics = {
        "model_name": "AI Co-Pilot Transformer Sequence Classifier",
        "dataset": "SYNTHETIC (DEMO MODE)",
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc
    }
    
    with open("backend/saved_models/transformer_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("[+] Transformer training complete. Model saved.")

if __name__ == "__main__":
    train_transformer_model()
