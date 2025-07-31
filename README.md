# COCA: When Small Guides Large: Cross-Model Co-Learning for Test-Time Adaptation

This repository contains a PyTorch implementation for the paper "When Small Guides Large: Cross-Model Co-Learning for Test-Time Adaptation (COCA)". It reproduces the key experiments of applying Test-Time Adaptation (TTA) to models evaluated on the ImageNet-C dataset.

## Introduction

COCA is a novel approach for Test-Time Adaptation (TTA) that leverages a cross-model co-learning strategy. The core idea is that a smaller, more agile model can guide a larger, more powerful model during adaptation to out-of-distribution data. This repository provides the necessary code to run TTA experiments with COCA on the ImageNet-C dataset.

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

## How to Reproduce Results

You can reproduce all the results, including single-model evaluation and COCA, by running a single script:

```bash
bash scripts/run_all.sh
```

This script will:
1.  Evaluate the single model baseline accuracies for ResNet50 and ViT-Base.
2.  Run the COCA adaptation for both model pairs (ResNet50/ResNet18 and ViT-Base/MobileViT-S).
3.  Save the results in the `results/` directory.

## Results and Analysis

The experiments in this repository evaluate two main model configurations on the ImageNet-C dataset at severity level 5. The results demonstrate the effectiveness of the COCA method.

### Performance Summary

| Model Configuration                  | Baseline Accuracy (Single Model) | COCA Accuracy | Improvement |
| ------------------------------------ | -------------------------------- | ------------- | ----------- |
| **ResNet50** (Anchor) + **ResNet18** (Aux) | 19.80%                           | **25.35%**    | **+5.55%**  |
| **ViT-Base** (Anchor) + **MobileViT-S** (Aux) | 38.59%                           | **37.32%**    | **-1.27%**  |

*Baseline accuracies are calculated by averaging the results from `results/*_single_model_accuracy.txt`.*
*COCA accuracies are taken from the generated JSON files in `results/`.*

### Analysis

The results show two distinct outcomes:

1.  **ResNet50 + ResNet18**: The COCA method significantly improves the model's robustness to corruptions, boosting the average accuracy from **19.80%** to **25.35%**. This represents a **28% relative improvement** and confirms the paper's hypothesis that a smaller model (ResNet18) can effectively guide a larger one (ResNet50) to better adapt to unseen data distributions at test time.

2.  **ViT-Base + MobileViT-S**: In this configuration, the COCA method does not yield an improvement. The baseline accuracy of the ViT-Base model is **38.59%**, while the accuracy after applying COCA with MobileViT-S as the auxiliary model is **37.32%**. This suggests that the co-learning strategy may not be universally effective across all model architectures. The interaction between Vision Transformers and smaller convolutional models like MobileViT might not be as synergistic as the ResNet-to-ResNet pairing under the tested hyperparameters. The results for `snow`, `frost`, and `fog` corruptions are particularly low, which heavily impacts the overall average.

### Conclusion

This reproduction successfully validates the core claim of the COCA paper for convolutional architectures, where a significant performance gain is observed. However, it also highlights that the effectiveness of COCA is not guaranteed and can be dependent on the specific combination of anchor and auxiliary models.
