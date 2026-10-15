
import pytest
import torch
import numpy as np

from backend.app.ml.transformer_model import FlightActionTransformer
from backend.app.ml.train_transformer import train_transformer_model
from backend.app.ml.dataset_generator import generate_synthetic_dataset

def test_transformer_forward_pass():
    model = FlightActionTransformer(
        input_dim=12,
        d_model=64,
        num_heads=4,
        num_layers=2,
        num_classes=9
    )
    
    # Batch size 2, Sequence Length 15, Feature dim 12
    dummy_input = torch.randn(2, 15, 12)
    
    logits, attn_weights = model(dummy_input)
    
    assert logits.shape == (2, 9)
    assert attn_weights.shape == (2, 15)
    
    # Attention weights should sum to 1 over seq_len
    sums = attn_weights.sum(dim=1)
    assert torch.allclose(sums, torch.ones(2))

def test_sequence_generation():
    X, y = generate_synthetic_dataset(num_samples=10, seq_length=30)
    assert X.shape == (10, 30, 12)
    assert y.shape == (10,)
    assert np.min(y) >= 0 and np.max(y) < 9
