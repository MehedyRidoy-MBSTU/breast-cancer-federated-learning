# Federated Breast Cancer Classification Using DenseNet-121

## Overview

This project presents a privacy-preserving Federated Learning (FL) framework for breast cancer histopathological image classification.

The main objective is to develop a collaborative deep learning system where multiple decentralized clients can train a shared model without transferring raw medical images to a central server.

The project implements a Federated Averaging (FedAvg) algorithm using the Flower Federated Learning framework with a DenseNet-121 deep learning architecture.

The system simulates multiple healthcare institutions, where each client maintains its own local dataset and contributes only model updates during federated training.

---

# Research Objectives

The main objectives of this research project are:

- Develop a federated deep learning pipeline for breast cancer classification.
- Simulate decentralized healthcare environments using multiple clients.
- Train a global DenseNet-121 model without sharing raw medical images.
- Evaluate federated model performance using clinical classification metrics.
- Improve interpretability using Explainable AI techniques.

---

# Project Features

- Federated Learning using Flower Framework
- FedAvg aggregation algorithm
- DenseNet-121 based image classification
- Multi-client decentralized training
- Non-IID client data partitioning
- Independent test evaluation
- Threshold optimization analysis
- Explainable AI preparation using Grad-CAM

---

# Technologies Used

## Deep Learning

- PyTorch
- TorchVision
- DenseNet-121
- Transfer Learning
- AdamW Optimizer


## Federated Learning

- Flower Framework
- Federated Averaging (FedAvg)
- Client-Server Architecture


## Data Processing and Evaluation

- Python
- NumPy
- Pandas
- Scikit-learn


## Explainable AI

- Grad-CAM
- Model prediction visualization


---

# Dataset

## BreaKHis Breast Cancer Histopathological Dataset

This project uses the BreaKHis dataset for breast cancer histopathological image classification.

The dataset contains microscopic breast tissue images categorized into:

- Benign
- Malignant


The dataset was partitioned into multiple federated clients to simulate decentralized medical institutions.

Each client contains a unique subset of specimens, ensuring that no specimen appears in multiple clients.

Raw medical images remain locally stored and are not exchanged between clients and server.

---

# Federated Learning Architecture

                 Local Data

    Client 1  ─────┐
    Client 2  ─────┤
    Client 3  ─────┤
    Client 4  ─────┤
    Client 5  ─────┘

              Flower Server

                   |
                   |
          FedAvg Aggregation

                   |
                   |
         Global DenseNet-121 Model



---

# Federated Learning Configuration

| Parameter | Value |
|---|---|
| Framework | Flower |
| Aggregation Method | FedAvg |
| Model | DenseNet-121 |
| Number of Clients | 5 |
| Communication Rounds | 5 |
| Local Epochs | 1 |
| Batch Size | 16 |
| Optimizer | AdamW |
| Learning Rate | 1e-4 |
| Weight Decay | 1e-4 |
| Image Size | 224 × 224 |


---

# Client Data Distribution

The training dataset was divided into five independent federated clients.

Example distribution:

| Client | Images | Specimens |
|---|---:|---:|
| Client 1 | 1257 | 12 |
| Client 2 | 1154 | 11 |
| Client 3 | 929 | 11 |
| Client 4 | 1048 | 11 |
| Client 5 | 1103 | 11 |

Total:

- Training Images: 5491
- Unique Specimens: 56


---

# Model Architecture

DenseNet-121 was selected as the main classification architecture because of its effective feature reuse mechanism and strong performance in medical image analysis.

Dense connectivity enables better gradient propagation and allows the model to learn detailed histopathological patterns.

The final classification layer was modified for binary classification:
Output Classes:

0 → Benign
1 → Malignant

---

# Federated Training Process

The training process follows the FedAvg workflow:

1. Server initializes the global DenseNet-121 model.
2. Global parameters are distributed to clients.
3. Each client performs local training using its private dataset.
4. Clients return updated model parameters.
5. Server aggregates updates using FedAvg.
6. Updated global model is redistributed.
7. Process repeats for multiple communication rounds.


---

# Final Federated Model Performance

The final FedAvg DenseNet-121 model was evaluated on an independent test dataset.

Test Dataset:
Samples: 1346


## Evaluation Results

| Metric | Score |
|---|---:|
| Accuracy | 73.99% |
| Precision | 92.98% |
| Sensitivity (Recall) | 70.27% |
| Specificity | 84.73% |
| F1-score | 80.05% |
| ROC-AUC | 87.58% |


---

# Confusion Matrix

| | Predicted Benign | Predicted Malignant |
|-|-:|-:|
| Actual Benign | 294 | 53 |
| Actual Malignant | 297 | 702 |


---

# Threshold Analysis

The default classification threshold was:
Threshold = 0.50


Performance:

| Metric | Value |
|---|---:|
| F1-score | 80.05% |
| Sensitivity | 70.27% |
| Specificity | 84.73% |


Threshold optimization was performed to analyze sensitivity-specificity trade-offs.

Best F1 threshold:
Threshold = 0.02

F1-score = 89.47%

Sensitivity = 97.80%

Specificity = 40.06%



This demonstrates the importance of decision threshold selection in medical classification systems.


---

# Explainable AI (XAI)

Explainability is an important requirement for medical AI applications.

This project integrates Grad-CAM based visualization to analyze which image regions contribute to model predictions.

The purpose is to improve transparency and understand whether the model focuses on meaningful histopathological structures.

Planned visualization categories:

- True Positive cases
- True Negative cases
- False Positive cases
- False Negative cases


---

# Repository Structure

```text
breast-cancer-federated-learning/

├── src/
│
├─── models/
│   └── densenet121.py
│
├─── federated/
│   ├── server/
│   │   └── fedavg_server.py
│   │
│   ├── clients/
│   │   └── run_client.py
│   │
│   └── datasets/
│
├─── training/
│
├─── evaluation/
│   ├── evaluation_fedavg_densenet121.py
│   └── threshold_analysis_fedavg_densenet121.py
│
├─── xai/
│
├─── processed/
│   └── Dataset split files
│
├─── results/
│   └── Evaluation outputs
│
├─── checkpoints/
│   └── Saved model weights
│
├── README.md
│
└── requirements.txt
```


---
# Installation

## Create and Activate Environment

```bash
conda create -n flower_env python=3.11

conda activate flower_env
```

## Install Dependencies

```bash
pip install -r requirements.txt
```


# Running Federated Training

## Start Server

Open a terminal and run:

```bash
python -m src.federated.server.fedavg_server
```


## Start Clients

Open five separate terminals and run the following commands.


### Client 1

```bash
python -m src.federated.clients.run_client client_1
```


### Client 2

```bash
python -m src.federated.clients.run_client client_2
```


### Client 3

```bash
python -m src.federated.clients.run_client client_3
```


### Client 4

```bash
python -m src.federated.clients.run_client client_4
```


### Client 5

```bash
python -m src.federated.clients.run_client client_5
```


After all clients complete local training, the Flower server aggregates the client updates using the FedAvg algorithm and saves the final global DenseNet-121 model.



# Model Evaluation

After completing federated training, evaluate the final global model:

```bash
python -m src.evaluation.evaluation_fedavg_densenet121
```


The evaluation generates:

- Accuracy
- Precision
- Sensitivity
- Specificity
- F1-score
- ROC-AUC
- Confusion Matrix values


Evaluation results are saved:

```
results/
└── fedavg_densenet121_test_results.csv
```


Prediction outputs are saved:

```
results/
└── fedavg_densenet121_predictions.csv
```



# Threshold Optimization Analysis

To analyze the effect of classification threshold selection:

```bash
python -m src.evaluation.threshold_analysis_fedavg_densenet121
```


The analysis evaluates different probability thresholds and identifies optimal operating points based on:

- F1-score
- Sensitivity
- Specificity


Output:

```
results/
└── fedavg_densenet121_threshold_analysis.csv
```



# Federated Model Checkpoints

During training, the global model checkpoints are saved:

```
checkpoints/

├── fedavg_densenet121_round_1.pth
├── fedavg_densenet121_round_2.pth
├── fedavg_densenet121_round_3.pth
├── fedavg_densenet121_round_4.pth
├── fedavg_densenet121_round_5.pth
└── fedavg_densenet121_final.pth
```



# Hardware Environment

Experiments were performed on:

- Device: Apple MacBook with Apple Silicon M5
- Operating System: macOS
- Python Version: 3.11
- Deep Learning Framework: PyTorch
- Acceleration: Apple Metal Performance Shaders (MPS)



# Reproducibility

To reproduce the experiments:

1. Install required dependencies.

2. Prepare the BreaKHis dataset.

3. Generate federated client partitions.

4. Start the Flower server.

5. Launch all federated clients.

6. Evaluate the final global model.


The raw medical dataset is not included in this repository due to size and licensing limitations.

Users should download the original BreaKHis dataset from the official source and prepare the required directory structure.



# Limitations

Current limitations:

- Limited number of communication rounds due to computational resources.
- Simulated federated environment using local machines.
- No differential privacy mechanism implemented.
- No secure aggregation protocol implemented.
- Further validation on external medical datasets is required.



# Future Work

Future research directions include:

- Increasing communication rounds.
- Experimenting with additional federated algorithms:
  - FedProx
  - FedAdam
  - Scaffold
- Implementing privacy-preserving mechanisms.
- Adding complete DenseNet-121 Grad-CAM visualization.
- Performing multi-center clinical validation.
- Deploying the model as a medical decision-support prototype.



# Citation

If this project contributes to your research, please cite:

```
Federated Breast Cancer Classification Using DenseNet-121

A privacy-preserving deep learning framework using
Flower Federated Learning and FedAvg aggregation.
```


# Author

**Mehedy Ridoy**

GitHub:

https://github.com/MehedyRidoy-MBSTU


Research Project:

**Federated Learning Based Breast Cancer Classification Using Deep Neural Networks**