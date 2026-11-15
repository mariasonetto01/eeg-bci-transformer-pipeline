EEG BCI Transformer Pipeline documentation
==========================================

**EEG BCI Transformer Pipeline** is an end-to-end deep learning framework written in Python,
built on `PyTorch <https://pytorch.org/>`_ and `MNE-Python <https://mne.tools/>`_, specifically
designed for **binary Motor Imagery (MI) EEG classification (left hand vs. right hand)** in
Brain-Computer Interface (BCI) applications.

The entire workflow, from EEG preprocessing and epoch extraction to model training and
evaluation, is built around binary MI decoding. Within this shared binary pipeline, the project
reproduces the **F_CTrans** architecture proposed by *Jin Xie et al. ("A Transformer-Based
Approach Combining Deep Learning Network and Spatial-Temporal Information for Raw EEG
Classification", IEEE TNSRE, 2022)* alongside a **modified architectural variant** of the same
network.

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   overview
   api

.. toctree::
   :maxdepth: 1
   :caption: External Links

   Source code repository <https://github.com/mariasonetto01/eeg-bci-transformer-pipeline>