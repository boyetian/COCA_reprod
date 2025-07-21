
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

## 5. Proposed Method (COCA)

The proposed method, **COCA (Cross-Model Co-Learning for Test-Time Adaptation)**, consists of two main strategies:

1.  **Co-adaptation**: This strategy adaptively integrates complementary knowledge from other models during the TTA process. This helps to reduce the biases of individual models. In the paper's example, a smaller model (Mobile-ViT) guides a larger model (ViT-Base).
2.  **Self-adaptation**: This strategy enhances each model's unique strengths through unsupervised learning, allowing for diverse adaptation to the target domain.

The key idea is that even a much smaller model can provide valuable, confident knowledge to a larger model in an unsupervised, online setting.

## 6. Recommended Codebase, Filetree, and Construction

Based on the analysis of related repositories, here is a recommended file structure for the project:

```
COCA_reproduction/
├── reproduction_README.md
├── requirements.txt
├── configs/
│   ├── vit_base_mobilvit.yaml
│   └── resnet50_resnet18.yaml
├── data/
│   ├── download_data.sh
│   └── imagenet_c.py
├── models/
│   ├── coca.py
│   ├── vit.py
│   └── resnet.py
├── scripts/
│   └── train_coca.sh
├── main.py
└── utils/
    ├── augmentations.py
    └── metrics.py
```

### Explanation of the File Structure:

*   `reproduction_README.md`: The file you are currently reading.
*   `requirements.txt`: A file listing the Python dependencies for the project.
*   `configs/`: A directory to store configuration files for different experiments (e.g., model pairs, hyperparameters).
*   `data/`: A directory for data loading and preparation scripts.
    *   `download_data.sh`: A script to download and prepare the ImageNet-C dataset.
    *   `imagenet_c.py`: A PyTorch Dataset class for ImageNet-C.
*   `models/`: A directory for the model implementations.
    *   `coca.py`: The core implementation of the COCA method, including the co-adaptation and self-adaptation strategies.
    *   `vit.py`: Implementation of the Vision Transformer models (ViT-Base and Mobile-ViT).
    *   `resnet.py`: Implementation of the ResNet models.
*   `scripts/`: A directory for shell scripts to run experiments.
*   `main.py`: The main script to run the training and evaluation of the models.
*   `utils/`: A directory for utility functions.
    *   `augmentations.py`: Data augmentation functions.
    *   `metrics.py`: Functions to compute evaluation metrics (e.g., accuracy, ECE).

This structure is modular and allows for easy extension to new models, datasets, and experiments. 