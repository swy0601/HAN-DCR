# Developer-Code Relationship Extraction Based on Baseline M2

## Overview
This module implements developer-code relationship extraction using the second baseline model (M2). It provides a complete solution for extracting relationships between developers and code artifacts, employing advanced graph neural network architectures and sophisticated data processing pipelines.

## Project Structure

### Main Training Scripts
- **`HanTrain.py`**: Implements HAN (Hierarchical Attention Network) based training with multi-layer attention mechanisms for developer-code relationship modeling, including contrastive learning and feature fusion mechanisms
- **`GatTrain.py`**: Implements GAT (Graph Attention Network) based training, using graph attention convolution layers for relationship learning and feature extraction
- **`GtnTrain.py`**: Implements GTN (Graph Transformation Network) based training, using graph transformation networks for multi-relation graph learning and clustering analysis

### Core Modules
- **`utils.py`**: Provides essential utilities including data loading, preprocessing, random seed management, and early stopping mechanisms, supporting heterogeneous graph data processing
- **`Test.py`**: Model evaluation script with comprehensive metrics calculation (Accuracy, Precision, Recall, F1, AUC, MCC), supporting performance evaluation of multiple models

### Model Definitions
- **`Models/`**: Deep learning model definition directory containing various graph neural network architectures
  - `gat/`: Graph attention network related components
  - `gtn/`: Graph transformation network model definitions
  - `han/`: Hierarchical attention network related implementations
  - `loss/`: Loss function definitions, including contrastive learning related losses

### Data Processing
- **`Preprocess/`**: Advanced preprocessing pipeline directory containing data transformation and feature extraction components
- **`DataCleanAndGraphBuild.py`**: Data cleaning and graph building script that extracts developer-code relationships from Git repositories and builds heterogeneous graphs

## Key Features

- **Graph Neural Network Learning**: Focuses on relationship learning capabilities with graph structure data
- **Advanced Architecture**: Uses sophisticated attention mechanisms and graph neural networks
- **Contrastive Learning**: Supports contrastive learning with positive and negative sample pairs
- **Feature Fusion**: Adaptive fusion of feature representations from different sources
- **Robust Evaluation**: Multiple evaluation metrics and comprehensive testing framework

## Components

### Training Framework
- **`HanTrain.py`**: Implements two-stage training process, first stage for contrastive learning, second stage for fine-tuning with real labels, supporting multi-random seed experiments
- **`GatTrain.py`**: Graph attention network based training framework with node-type subgraph processing and feature fusion mechanisms
- **`GtnTrain.py`**: Graph transformation network training implementation combining k-means clustering and silhouette coefficient evaluation

### Model Architecture
- **HAN Implementation**: Hierarchical attention networks for capturing multi-level relationships, including `HANLayer` and `HAN` model classes
- **GAT Implementation**: Graph attention mechanisms using `GATConv` for weighted neighbor aggregation
- **GTN Implementation**: Graph transformation networks for adaptive graph structure learning

### Data Processing Pipeline
- **Positive/Negative Sample Pair Construction**: Builds positive and negative sample pairs based on contribution similarity with similarity weight calculation
- **Feature Extraction**: Combines DMon model and graph neural networks for feature extraction
- **Mask Management**: Supports dynamic partitioning of training, validation, and test sets

## Usage
The system supports multiple training modes and evaluation strategies:
- Execute `HanTrain.py` for hierarchical attention based training
- Run `GatTrain.py` for graph attention based training
- Run `GtnTrain.py` for graph transformation based training
- Use `Test.py` for comprehensive model evaluation

## Technical Approach
Baseline M2 leverages advanced graph neural networks to model complex developer-code relationships. The system combines contrastive learning, graph structure feature fusion, and sophisticated attention mechanisms to identify and extract meaningful associations between developers and their code contributions.