import pytest
import torch
from models import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    FeedForward,
    TransformerLayer,
    SpatialCNN,
    TemporalCNN,
    PositionalEncoding_PE,
    LearnableSpatialPositionalEncoding_LPE,
    F_CTrans,
    F_CTrans_modified
)

@pytest.fixture
def dummy_eeg_input():
    """
    Creates a dummy EEG input batch.
    Shape: (batch_size=4, num_channels=3, time_points=320)
    """
    return torch.randn(4, 3, 320)

def test_scaled_dot_product_attention():
    """
    Verifies that the attention module computes the correct shapes for output and weights.
    """    
    
    batch_size, nhead, seq_len, d_k = 4, 8, 16, 16
    q = torch.randn(batch_size, nhead, seq_len, d_k)
    k = torch.randn(batch_size, nhead, seq_len, d_k)
    v = torch.randn(batch_size, nhead, seq_len, d_k)
    
    attention = ScaledDotProductAttention(dropout=0.1)
    output, attn_weights = attention(q, k, v)
    
    assert output.shape == (batch_size, nhead, seq_len, d_k)
    assert attn_weights.shape == (batch_size, nhead, seq_len, seq_len)

def test_multi_head_attention():
    """
    Verifies Multi-Head Attention with projection and dimension reconstruction.
    """

    batch_size, nhead, seq_len, d_model = 4, 8,  16, 64
    x = torch.randn(batch_size, seq_len, d_model)
    
    mha = MultiHeadAttention(d_model=d_model, nhead=nhead, dropout=0.1)
    output, attn_weights = mha(x)
    
    assert output.shape == (batch_size, seq_len, d_model)
    assert attn_weights.shape == (batch_size, nhead, seq_len, seq_len)

def test_spatial_cnn(dummy_eeg_input):
    """
    Verifies that Spatial CNN correctly processes channel representations.
    """
    d_model = 64
    spatial_cnn = SpatialCNN(output_channels=d_model)
    output = spatial_cnn(dummy_eeg_input)
    
    # Expected output shape: (batch_size, num_channels, out_channels)
    # In this case: (4, 3, 64)
    assert output.shape == (4, 3, d_model)

def test_temporal_cnn(dummy_eeg_input):
    """
    Verifies that Temporal CNN reduces the time axis and collapses the spatial dimension.
    """    
    
    d_model = 64
    num_channels = dummy_eeg_input.shape[1]
    
    temporal_cnn = TemporalCNN(output_channels=d_model, num_channels=num_channels)
    output = temporal_cnn(dummy_eeg_input)
    
    # Expected output shape: (batch_size, seq_len_t, out_channels)
    assert output.shape[0] == dummy_eeg_input.shape[0]
    assert output.shape[2] == d_model

def test_f_ctrans(dummy_eeg_input):
    """
    Verifies the full forward pass of the standard F_CTrans model.
    """

    num_channels = 3
    num_classes = 2  
    d_model = 64
    
    model = F_CTrans(
        num_channels=num_channels,
        d_model=d_model,
        nhead=4,
        d_ff=128,
        dropout=0.1,
        num_layers=1,
        num_classes=num_classes
    )
    
    logits, (spatial_weights, temporal_weights) = model(dummy_eeg_input)

    # Output logits must be (batch_size, num_classes) -> (4, 2)
    assert logits.shape == (4, num_classes)
    assert len(spatial_weights) == 1
    assert len(temporal_weights) == 1

def test_f_ctrans_modified(dummy_eeg_input):
    """
    Verifies the full forward pass of the  F_CTrans_mofified  model.
    """

    num_channels = 3
    num_classes = 2
    d_model = 64
    
    model = F_CTrans_modified(
        num_channels=num_channels,
        d_model=d_model,
        nhead=4,
        d_ff=128,
        dropout=0.1,
        num_layers=1,
        num_classes=num_classes
    )
    
    logits, (spatial_weights, temporal_weights) = model(dummy_eeg_input)
    
    # Final output must still be (4, 2) 
    assert logits.shape == (4, num_classes)
    assert len(spatial_weights) == 1
    assert len(temporal_weights) == 1