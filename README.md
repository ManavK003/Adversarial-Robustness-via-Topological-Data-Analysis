# Adversarial-Robustness-via-Topological-Data-Analysis

## Project Overview

This project applies **Topological Data Analysis (TDA)** to improve adversarial robustness of deep neural networks. We use persistent homology to characterize the manifold structure of feature spaces and design topology-preserving loss functions that force adversarial examples to remain on the natural data manifold.

### Key Contributions

1. **Topology-Preserving Training**: Novel loss function based on Wasserstein distance between persistence diagrams
2. **Adversarial Robustness**: 57.7% improvement against FGSM attacks (37.2% → 94.9%)
3. **Theoretical Framework**: Application of TDA to adversarial machine learning
4. **Comprehensive Evaluation**: Comparison across multiple attack strengths and datasets

---

## Key Results

### MNIST Classification

| Model | Clean Accuracy | FGSM (ε=0.3) | PGD (ε=0.3) |
|-------|----------------|--------------|-------------|
| **Standard** | 99.08% | 37.20% | 0.00% |
| **Robust (Ours)** | 98.39% | **94.92%** | **8.90%** |
| **Improvement** | -0.69% | **+57.72%** 🎯 | **+8.90%** 🎯 |

**Highlights:**
-  **94.92% accuracy** against FGSM attacks (vs 37.20% baseline)
-  **Minimal clean accuracy loss** (only 0.69%)
-  **155% relative improvement** on FGSM robustness
-  **Graceful degradation** under increasing attack strength
