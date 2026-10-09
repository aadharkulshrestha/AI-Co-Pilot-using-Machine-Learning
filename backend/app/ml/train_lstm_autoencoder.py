import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import pickle

class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, hidden_dim=64):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n_features)
        
    def forward(self, x):
        _, (h_n, _) = self.encoder(x)
        h_n = h_n.transpose(0, 1)
        h_n = h_n.repeat(1, x.size(1), 1)
        decoded, _ = self.decoder(h_n)
        return self.fc(decoded)

def get_project_root():
    return Path(__file__).resolve().parent.parent.parent.parent

def fit_scaler(X_train):
    print("Fitting normalization parameters on training data ONLY...")
    N, seq_len, n_features = X_train.shape
    X_train_flat = X_train.reshape(-1, n_features)
    
    mean = np.mean(X_train_flat, axis=0)
    std = np.std(X_train_flat, axis=0)
    std[std == 0] = 1.0 # Prevent division by zero
    
    return mean, std

def apply_scaler(X, mean, std):
    N, seq_len, n_features = X.shape
    X_flat = X.reshape(-1, n_features)
    X_scaled_flat = (X_flat - mean) / std
    return X_scaled_flat.reshape(N, seq_len, n_features)

def train_model():
    # Setup paths
    root_dir = get_project_root()
    data_dir = root_dir / "backend" / "dataset" / "processed_sequences"
    models_dir = root_dir / "backend" / "saved_models"
    reports_dir = root_dir / "backend" / "reports"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    # Configuration
    batch_size = 512
    epochs = 1
    hidden_dim = 64
    patience = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Data
    print("Loading training and validation datasets...")
    X_train_raw = np.load(data_dir / "X_train.npy")
    X_val_raw = np.load(data_dir / "X_val.npy")
    
    n_features = X_train_raw.shape[2]
    
    # 2. Fit and Apply Scaler
    mean, std = fit_scaler(X_train_raw)
    scaler_params = {"mean": mean.tolist(), "std": std.tolist()}
    with open(models_dir / "lstm_ae_scaler.json", "w") as f:
        json.dump(scaler_params, f)
        
    X_train_scaled = apply_scaler(X_train_raw, mean, std)
    X_val_scaled = apply_scaler(X_val_raw, mean, std)
    
    # Create DataLoaders
    train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 3. Initialize Model
    model = LSTMAutoencoder(n_features=n_features, hidden_dim=hidden_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 4. Training Loop with Early Stopping
    print("Starting training...")
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x_batch = batch[0].to(device)
            optimizer.zero_grad()
            out = model(x_batch)
            loss = criterion(out, x_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)
            
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x_batch = batch[0].to(device)
                out = model(x_batch)
                loss = criterion(out, x_batch)
                val_loss += loss.item() * x_batch.size(0)
                
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), models_dir / "lstm_autoencoder.pth")
            print("  --> Model saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
                
    # 5. Save Learning Curves
    training_history = {
        "train_loss": train_losses,
        "val_loss": val_losses
    }
    with open(reports_dir / "experiment_e1_training_history.json", "w") as f:
        json.dump(training_history, f, indent=2)
        
    print("Training phase completed.")

if __name__ == "__main__":
    train_model()
