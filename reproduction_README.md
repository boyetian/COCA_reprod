
# COCA Reproduction Plan

This document outlines the plan to reproduce the experiments from the paper "When Small Guides Large: Cross-Model Co-Learning for Test-Time Adaptation (COCA)".

## 1. Datasets to Prepare

Based on the paper's abstract, the primary dataset for this reproduction is:

*   **ImageNet-C**: This dataset is used for evaluating the model's performance on corrupted images. It consists of 15 types of algorithmically generated corruptions applied to the ImageNet validation set.

Other potential datasets that might be useful for pre-training or as baselines, based on related works:

*   **ImageNet**: The original dataset for pre-training the models.
*   **CIFAR-10-C / CIFAR-100-C**: Corrupted versions of the CIFAR datasets, which are smaller and can be used for faster prototyping and debugging.
*   **ImageNet-R**: A dataset of renditions of ImageNet classes, used to test for robustness against stylistic changes.
*   **ImageNet-Sketch**: A dataset of sketch-like images corresponding to ImageNet classes.
*   **SVHN**: The Street View House Numbers dataset, which can be used for domain adaptation experiments (e.g., SVHN to MNIST).
*   **MNIST**: The Modified National Institute of Standards and Technology database of handwritten digits.

## 2. Models to Train

The paper mentions the following model architectures:

*   **ResNets**: A family of residual network architectures, commonly used as a baseline.
*   **Vision Transformers (ViTs)**: Transformer-based models for computer vision, specifically `ViT-Base`.
*   **Mobile-ViTs**: Lightweight Vision Transformers designed for mobile devices.

The core idea is to use a smaller model (e.g., Mobile-ViT) to guide a larger model (e.g., ViT-Base) during test-time adaptation.

## 3. Relevant Works and Papers

The following papers and research areas are relevant to this work and recommended for further reading:

*   **Test-Time Adaptation (TTA)**: This is the core field of the paper.
    *   "Tent: Fully Test-time Adaptation by Entropy Minimization"
    *   "Test-Time Adaptation via Conjugate Pseudo-Labels"
    *   "Test-Time Model Adaptation with Only Forward Passes"
*   **Domain Adaptation**: A broader field that deals with adapting models to new domains.
*   **Contrastive Learning**: The CMA repository suggests that contrastive learning can be used for model adaptation.
*   **Diffusion Models for Adaptation**: The DDA repository suggests using diffusion models to adapt the input data.

## 4. Possible Public Repositories

While the official repository for COCA is not yet public, the following repositories can be helpful for reproducing the work:

*   **[locuslab/tta_conjugate](https://github.com/locuslab/tta_conjugate)**: Implements "Test-Time Adaptation via Conjugate Pseudo-Labels" and provides a good starting point for TTA experiments.
*   **[brdav/cma](https://github.com/brdav/cma)**: Implements "Contrastive Model Adaptation" and uses a similar setup for domain adaptation in semantic segmentation.
*   **[mr-eggplant/FOA](https://github.com/mr-eggplant/FOA)**: Implements "Test-Time Model Adaptation with Only Forward Passes," which is a related TTA method.
*   **[shiyegao/DDA](https://github.com/shiyegao/DDA)**: Implements "Diffusion-Driven Test-Time Adaptation," which adapts the input data instead of the model.
 

Settings
  - SGD with momentum of 0.9
  - Batch_size=64
  - Learning_rate for R and V: 0.00025 and 0.001
  - Trainable params: only BN layers are updated
  - ImageNet-C: exclusively select severity 5