import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return x

class FlightActionTransformer(nn.Module):
    def __init__(self, input_dim=12, d_model=64, num_heads=4, num_layers=2, num_classes=9, dropout=0.1):
        super(FlightActionTransformer, self).__init__()
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads, 
            dim_feedforward=d_model * 4, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Attention Pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
            nn.Softmax(dim=1)
        )
        
        # Classification head
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model // 2, num_classes)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, input_dim)
        """
        # 1. Project input and add positional encoding
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        
        # 2. Transformer
        x = self.transformer_encoder(x)
        
        # 3. Attention Pooling over sequence length
        # attn_weights shape: (batch_size, seq_len, 1)
        attn_weights = self.attention_pool(x)
        
        # pool shape: (batch_size, d_model)
        pooled = torch.sum(x * attn_weights, dim=1)
        
        # 4. Classification
        out = self.fc1(pooled)
        out = self.relu(out)
        out = self.dropout(out)
        logits = self.fc2(out)
        
        return logits, attn_weights.squeeze(-1)
