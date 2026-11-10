import numpy as np
import mne
import pytest
import torch
from datapreprocessing import EEGpreprocessing, EEGDataset

@pytest.fixture
def dummy_raw():
    """
    Fixture that creates a synthetic MNE Raw object containing 3 EEG channels 
    and dummy 'T1'/'T2' annotations for testing.
    """
    
    sfreq = 250.0  # Initial sampling frequency in Hz
    n_channels = 3
    duration = 10.0  # Signal duration in seconds
    n_samples = int(sfreq * duration)
    
    # Generate random data (in Volts)
    data = np.random.randn(n_channels, n_samples) * 1e-6 
    ch_names = ['Fz', 'Cz', 'Pz']
    ch_types = ['eeg'] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    
    raw = mne.io.RawArray(data, info, verbose=False)
    
    # Add simulated annotations corresponding to motor imagery cues (T1: Left, T2: Right)    
    onset = [2.0, 6.0]  # in seconds
    duration_ann = [1.0, 1.0]
    description = ['T1', 'T2']
    annotations = mne.Annotations(onset, duration_ann, description)
    raw.set_annotations(annotations)
    
    return raw

def test_bandpassfilter(dummy_raw):
    """
    Tests whether resampling and band-pass filtering execute correctly 
    and update the underlying MNE Raw object properties.
    """

    target_sfreq = 160.0
    preprocessor = EEGpreprocessing(dummy_raw)
    
    raw_filtered = preprocessor.bandpassfilter(l_freq=1.0, h_freq=40.0, target_sfreq=target_sfreq)
    
    # Integrity assertions
    assert preprocessor.raw_filtered is not None
    assert isinstance(raw_filtered, mne.io.Raw)
    # Ensure sampling frequency was updated to the target frequency
    assert raw_filtered.info['sfreq'] == target_sfreq

def test_epoching(dummy_raw):
    """
    Tests epoch extraction based on 'T1' and 'Right' motor imagery annotations.
    """

    preprocessor = EEGpreprocessing(dummy_raw)

    # Filter the raw data prior to epoching
    preprocessor.bandpassfilter(l_freq=8.0, h_freq=30.0, target_sfreq=160.0)
    
    # Extract epochs around the event onset
    epochs = preprocessor.epoching(tmin=0.0, tmax=2.0)
    
    assert preprocessor.epochs is not None
    assert isinstance(epochs, mne.Epochs)

    # Verify that both target event classes are captured
    assert 'Left' in epochs.event_id
    assert 'Right' in epochs.event_id
    # Check that the extracted epoch count matches the annotated events (2)
    assert len(epochs) == 2

def test_eeg_dataset():
    """
    Tests PyTorch Dataset conversion, output dimensions, tensor data types, and indexing.
    """
    # Create dummy array data: 10 samples, 3 channels, 320 time steps (2 sec * 160 Hz)
    dummy_data = np.random.randn(10, 3, 320).astype(np.float32)
    dummy_labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    
    dataset = EEGDataset(dummy_data, dummy_labels)
    
    # Validate dataset length
    assert len(dataset) == 10
    
    # Retrieve the first sample using __getitem__
    sample_x, sample_y = dataset[0]
    
    # Check that outputs are PyTorch Tensors
    assert isinstance(sample_x, torch.Tensor)
    assert isinstance(sample_y, torch.Tensor)
    
    # Check dimensions and data types
    assert sample_x.shape == (3, 320)
    assert sample_x.dtype == torch.float32
    assert sample_y.dtype == torch.long