Project Overview & Quickstart
=============================

Scientific Context
------------------

This repository implements an end-to-end deep learning pipeline for **Electroencephalography (EEG)
Motor Imagery (MI) classification** (e.g., Left vs. Right hand motor imagery on the PhysioNet
EEGMMIDB dataset).

The pipeline is structured into three modular stages:

1. **Signal Preprocessing** (``datapreprocessing``):
   Resampling, band-pass filtering in the sensorimotor rhythms (:math:`\mu` and :math:`\beta` bands),
   epoch extraction around cue annotations (``T1``, ``T2``), and channel-wise normalization wrapped
   in a PyTorch ``Dataset``.

2. **Hybrid CNN-Transformer Architectures** (``models``):

   * ``F_CTrans`` & ``F_CTrans_modified``: Dual-branch spatial and temporal convolutional-transformer
     networks with learnable positional encodings.

3. **Training & Evaluation** (``train``):
   Modular PyTorch training loop (``EEGTrainer``) supporting learning rate
   scheduling, model checkpointing, and optional early stopping (``EarlyStopping``).


