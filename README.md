# COCA: When Small Guides Large: Cross-Model Co-Learning for Test-Time Adaptation

This repository contains a PyTorch implementation for the paper "When Small Guides Large: Cross-Model Co-Learning for Test-Time Adaptation (COCA)". It aims to reproduce the key experiments and results presented in the paper.

## Introduction

COCA is a novel approach for Test-Time Adaptation (TTA) that leverages a cross-model co-learning strategy. The core idea is that a smaller, more agile model can guide a larger, more powerful model during adaptation to out-of-distribution data. This repository provides the necessary code to run TTA experiments with COCA on the ImageNet-C dataset.

## Models and Datasets

### Models

The paper and this implementation utilize the following architectures:

*   **Anchor Model (Large):** Vision Transformer (`ViT-Base`)
*   **Auxiliary Model (Small):** ResNet-50

The principle of COCA is to use the auxiliary model to help the anchor model adapt at test time.

### Dataset

The primary dataset for evaluating TTA performance is:

*   **ImageNet-C**: This dataset consists of the ImageNet validation set corrupted by 15 different types of noise at 5 severity levels. This implementation focuses on severity level 5, as in the paper's settings. You can download it by running the `data/download_data.sh` script.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/chenqiang98/COCA_reprod.git
    cd COCA_reprod
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download the data:**
    The `ImageNet-C` dataset is used for this reproduction.
    ```bash
    bash data/download_data.sh
    ```

## Usage

To run a test-time adaptation experiment with COCA, use the `main.py` script. You can specify the corruption type and other parameters.

```bash
python main.py --corruption <corruption_type>
```

For example, to run adaptation on `gaussian_noise` at severity 5:

```bash
python main.py --corruption gaussian_noise --severity 5
```

### Configuration

The key hyperparameters for reproduction are set as defaults in `main.py`:

*   **Optimizer**: SGD with momentum 0.9
*   **Batch Size**: 64
*   **Learning Rate (ViT-Base)**: 0.00025
*   **Learning Rate (ResNet-50)**: 0.001
*   **Trainable Parameters**: Only Batch Normalization (BN) layers are updated during adaptation.
