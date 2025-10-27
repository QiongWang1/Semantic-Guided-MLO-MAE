# Evaluation of Baseline Methods from the MLO-MAE Paper on DermaMNIST

## 1. Introduction 
To evaluate the performance of different baseline methods described in the [MLO-MAE paper](https://arxiv.org/abs/2402.18128) on the [DermaMNIST](https://medmnist.com/) dataset, we reproduced 6 representative models under the same training and evaluation pipeline on the SCC cluster.
The goal is to compare the effectiveness of random-masking and learnable-masking strategies on medical image datasets.

## 2. Experimental Setup
- Name: DermaMNIST (7-class skin lesion classification)
- Input Size: 3×32×32 (resized from 28×28)
- Splits: Train = 7007, Validation = 1003, Test = 2005
- Evaluation Metrics: Accuracy, Precision, Recall, and F1-Score

Environment:

GPU: NVIDIA A10

PyTorch 2.3.1 + CUDA 11.8

Cluster: BU SCC (Slurm managed)

Training Epochs: 200

Batch Size: 128

Optimizer: AdamW, LR = 1.5e-4

Pretraining: None (all trained from scratch)