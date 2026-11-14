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

2. **Hybrid CNN-Transformer Architectures** (``models``, ``modelseegconfrmer``):

   * ``F_CTrans`` & ``F_CTrans_modified``: Dual-branch spatial and temporal convolutional-transformer
     networks with learnable positional encodings.
   * ``EEGConformer`` & ``EEGTransformerConv1D``: Compact convolutional-transformer baselines for
     spatio-temporal feature extraction.

3. **Training & Evaluation** (``train``):
   Modular PyTorch training loop (``EEGTrainer``) supporting gradient clipping, learning rate
   scheduling, model checkpointing, and optional early stopping (``EarlyStopping``).

Quickstart Example
------------------

The snippet below illustrates how to wrap preprocessed EEG epochs into a ``DataLoader``, instantiate
one of the Transformer models, and run a training step:

.. code-block:: python

   import torch
   from torch.utils.data import DataLoader
   from eeg_bci_transformer_pipeline.datapreprocessing import EEGDataset
   from eeg_bci_transformer_pipeline.models import F_CTrans_modified
   from eeg_bci_transformer_pipeline.train import EEGTrainer

   # 1. Create Dataset & DataLoader (data shape: [N_epochs, N_channels, Time_steps])
   dataset = EEGDataset(data, labels)
   loader = DataLoader(dataset, batch_size=32, shuffle=True)

   # 2. Initialize Model, Optimizer & Criterion
   model = F_CTrans_modified(num_channels=64, time_steps=641, num_classes=2)
   optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
   criterion = torch.nn.CrossEntropyLoss()

   # 3. Initialize Trainer
   trainer = EEGTrainer(
       model=model,
       criterion=criterion,
       optimizer=optimizer,
       device="cpu",
   )