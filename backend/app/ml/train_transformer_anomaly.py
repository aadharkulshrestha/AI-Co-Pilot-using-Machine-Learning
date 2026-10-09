import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import math
import time

def get_project_root():
    return Path(__file__).resolve().parent.parent.parent.parent

class AnomalyAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_keys = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.sigma_proj = nn.Linear(d_model, n_heads)

    def forward(self, x):
        B, L, _ = x.shape
        Q = self.q_proj(x).view(B, L, self.n_heads, self.d_keys).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.n_heads, self.d_keys).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.n_heads, self.d_keys).transpose(1, 2)

        # Series Association
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_keys)
        series_assoc = F.softmax(scores, dim=-1) # (B, heads, L, L)
        
        out = torch.matmul(series_assoc, V).transpose(1, 2).reshape(B, L, self.d_model)

        # Prior Association (Gaussian distribution of attention)
        sigma = F.softplus(self.sigma_proj(x)) + 1e-4 # (B, L, heads)
        sigma = sigma.transpose(1, 2).unsqueeze(-1) # (B, heads, L, 1)
        
        idx = torch.arange(L, device=x.device, dtype=torch.float32)
        dist = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs() # (L, L)
        dist = dist.unsqueeze(0).unsqueeze(0) # (1, 1, L, L)
        
        prior_assoc = torch.exp(- (dist ** 2) / (2 * (sigma ** 2))) 
        prior_assoc = prior_assoc / (prior_assoc.sum(dim=-1, keepdim=True) + 1e-8)

        return out, series_assoc, prior_assoc

class AnomalyTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256, dropout=0.1):
        super().__init__()
        self.attention = AnomalyAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        attn_out, series_assoc, prior_assoc = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x, series_assoc, prior_assoc

class AnomalyTransformer(nn.Module):
    def __init__(self, n_features, d_model=64, n_heads=4, e_layers=2):
        super().__init__()
        self.embed = nn.Linear(n_features, d_model)
        self.layers = nn.ModuleList([AnomalyTransformerBlock(d_model, n_heads) for _ in range(e_layers)])
        self.reconstruct = nn.Linear(d_model, n_features)
        
    def forward(self, x):
        x = self.embed(x)
        series_assocs = []
        prior_assocs = []
        for layer in self.layers:
            x, s_assoc, p_assoc = layer(x)
            series_assocs.append(s_assoc)
            prior_assocs.append(p_assoc)
            
        out = self.reconstruct(x)
        return out, series_assocs, prior_assocs

def kl_loss(p, q):
    # p, q shape: (B, heads, L, L)
    res = p * (torch.log(p + 1e-8) - torch.log(q + 1e-8))
    return torch.mean(torch.sum(res, dim=-1))

def apply_scaler(X, mean, std):
    N, seq_len, n_features = X.shape
    X_flat = X.reshape(-1, n_features)
    X_scaled_flat = (X_flat - mean) / std
    return X_scaled_flat.reshape(N, seq_len, n_features)

def train_model():
    root_dir = get_project_root()
    data_dir = root_dir / "backend" / "dataset" / "processed_sequences"
    models_dir = root_dir / "backend" / "saved_models"
    reports_dir = root_dir / "backend" / "reports"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    X_train_raw = np.load(data_dir / "X_train.npy")
    X_val_raw = np.load(data_dir / "X_val.npy")
    
    with open(models_dir / "lstm_ae_scaler.json", "r") as f:
        scaler = json.load(f)
    mean = np.array(scaler['mean'])
    std = np.array(scaler['std'])
    
    # SUBSAMPLING FOR SPEED (User has a 10-minute constraint)
    # Using 25,000 samples for training instead of 590,000
    X_train = apply_scaler(X_train_raw[:25000], mean, std)
    X_val = apply_scaler(X_val_raw[:5000], mean, std)
    
    n_features = X_train.shape[2]
    batch_size = 512
    epochs = 3  # Proof of convergence without massive time cost
    patience = 2
    k_tradeoff = 3.0
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    
    model = AnomalyTransformer(n_features).to(device)
    
    print("Starting Anomaly Transformer Min-Max Training...")
    
    # Isolate Prior parameters (sigma_proj) vs Series parameters (everything else)
    prior_params = []
    series_params = []
    for name, param in model.named_parameters():
        if "sigma_proj" in name:
            prior_params.append(param)
        else:
            series_params.append(param)
            
    optimizer_series = optim.Adam(series_params, lr=1e-3)
    optimizer_prior = optim.Adam(prior_params, lr=1e-3)
    mse_criterion = nn.MSELoss()
    
    start_time = time.time()
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            x = batch[0].to(device)
            
            # --- Phase 1: Update Series (Optimize all non-prior weights) ---
            optimizer_series.zero_grad()
            out, series_assocs, prior_assocs = model(x)
            loss_mse = mse_criterion(out, x)
            
            loss_kl_series = 0
            for s_assoc, p_assoc in zip(series_assocs, prior_assocs):
                loss_kl_series += kl_loss(s_assoc, p_assoc.detach())
            loss_kl_series /= len(series_assocs)
            
            loss_phase1 = loss_mse - k_tradeoff * loss_kl_series
            loss_phase1.backward()
            optimizer_series.step()
            
            # --- Phase 2: Update Prior (Optimize sigma_proj weights) ---
            # Do a second forward pass with updated Series weights to avoid inplace graph errors
            optimizer_prior.zero_grad()
            out, series_assocs, prior_assocs = model(x)
            loss_mse = mse_criterion(out, x)
            
            loss_kl_prior = 0
            for s_assoc, p_assoc in zip(series_assocs, prior_assocs):
                loss_kl_prior += kl_loss(p_assoc, s_assoc.detach())
            loss_kl_prior /= len(series_assocs)
            
            loss_phase2 = loss_mse + k_tradeoff * loss_kl_prior
            loss_phase2.backward()
            optimizer_prior.step()
            
            total_loss += loss_mse.item() * x.size(0)
            
        train_loss = total_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                out, series_assocs, prior_assocs = model(x)
                loss_mse = mse_criterion(out, x)
                total_val_loss += loss_mse.item() * x.size(0)
                
        val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{epochs} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}")
        
        # Early Stopping based on Validation MSE
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), models_dir / "transformer_anomaly.pth")
            print("  --> Model saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
                
    training_duration = time.time() - start_time
    
    torch.save(model.state_dict(), models_dir / "transformer_anomaly.pth")
    
    history = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "duration_sec": training_duration
    }
    with open(reports_dir / "experiment_e2_training_history.json", "w") as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    train_model()
