# EEG BCI Transformer Pipeline

**EEG BCI Transformer Pipeline** is an end-to-end deep learning framework written in Python, built on [PyTorch](https://pytorch.org/) and [MNE-Python](https://mne.tools/), specifically designed for **binary Motor Imagery (MI) EEG classification (left hand vs. right hand)** in Brain-Computer Interface (BCI) applications.

The entire workflow, from EEG preprocessing and epoch extraction to model training and evaluation, is tailored for binary MI decoding. Within this shared binary pipeline, the project reproduces the **F_CTrans** architecture proposed by *Jin Xie et al. ("A Transformer-Based Approach Combining Deep Learning Network and Spatial-Temporal Information for Raw EEG Classification", IEEE TNSRE, 2022)* alongside a **modified architectural variant** (`F_CTrans_modified`).

---

## Dataset

The project uses the **[PhysioNet EEG Motor Movement/Imagery Dataset](https://doi.org/10.13026/C28G6P)** (Schalk, 2009):
- 109 subjects (Motor Imagery runs: `4, 8, and 12`)
- 64-channel EEG recordings
- Motor Imagery tasks (imagined left- vs. right-hand movement)
- Binary classification: Left Hand (`T1`) vs. Right Hand (`T2`)
- Sampling frequency: `160 Hz` (uniformly resampled)
- Data duration: Cue-locked epochs extracted from `tmin` to `tmax` (`4s` windows)

> **Dataset Citation:** Schalk, G. (2009). *EEG Motor Movement/Imagery Dataset* (version 1.0.0). PhysioNet. RRID:SCR_007345. [https://doi.org/10.13026/C28G6P](https://doi.org/10.13026/C28G6P)

---

## Project Structure

```text
eeg-bci-transformer-pipeline/
├── .github/                                # GitHub Actions CI workflows
├── docs/                                   # Sphinx documentation source files
├── notebook/                               # Final notebooks with definitive results & evaluations
│   └── train_val_F_CTrans/                 # Saved model weights and training checkpoints
├── roadtrip/                               # Exploratory notebooks documenting design iterations
│   └── saved_models/                       # Saved model weights and training checkpoints
├── src/
│   └── eeg_bci_transformer_pipeline/
│       ├── __init__.py
│       ├── datapreprocessing.py            # MNE preprocessing pipeline & PyTorch EEGDataset
│       ├── models.py                       # F_CTrans baseline & F_CTrans_modified architectures
│       └── train.py                        # EEGTrainer loop, validation & EarlyStopper
├── tests/                                  # Unit tests
├── config.yaml                             # Experiment & model hyperparameters configuration
├── noxfile.py                              # Automated linting (Ruff, Pylint), tests & docs sessions
├── pyproject.toml                          # Package metadata & build configuration
├── requirements.txt                        # Python dependencies
└── README.md