API Reference
=============



Data Preprocessing & Dataset
----------------------------

This module handles the conversion of raw continuous EEG recordings into tensor-ready datasets
for binary Motor Imagery classification. It provides ``EEGpreprocessing`` for resampling,
band-pass filtering, and extracting left- vs. right-hand motor imagery epochs (``T1``/ ``T2``
annotations) via MNE-Python, alongside ``EEGDataset`` to wrap processed NumPy arrays into a
standard PyTorch ``Dataset`` interface.

.. automodule:: eeg_bci_transformer_pipeline.datapreprocessing
   :show-inheritance:

Models 
-----------------------------

This module implements the dual-stream CNN-Transformer architectures used for binary Motor
Imagery classification. Both models process raw/preprocessed EEG epochs through parallel
**spatial** (``SpatialCNN`` with learnable electrode positional embeddings) and **temporal**
(``TemporalCNN`` with sinusoidal positional encoding) branches, followed by Multi-Head
Self-Attention encoder blocks:

* ``F_CTrans``: Reproduces the baseline dual-stream fusion architecture proposed by *Xie et al.
  (2022)*, combining average-pooled spatial and temporal representations into a single linear
  classification head.
* ``F_CTrans_modified``: Extends the baseline design by concatenating **Mean + Max pooling** on
  both branches, applying pre-fusion ``LayerNorm`` to align feature scales, and replacing the
  linear head with a multi-layer perceptron (MLP) classifier.

.. automodule:: eeg_bci_transformer_pipeline.models
   :show-inheritance:

Training & Early Stopping
-------------------------

This module manages the PyTorch training and evaluation lifecycle for the binary Motor Imagery
classifiers. It defines ``EarlyStopper`` to monitor validation loss progression and halt training
when improvements plateau, alongside ``EEGTrainer``, which handles device selection
(CUDA/MPS/CPU), mini-batch optimization, learning rate scheduling, and best-checkpoint restoration
across training epochs.



.. automodule:: eeg_bci_transformer_pipeline.train
   :show-inheritance: