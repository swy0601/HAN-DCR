# Developer-Code Relationship Extraction Based on Baseline M1

## Overview
This module implements developer-code relationship extraction using the first baseline model (M1). It provides a complete solution for extracting relationships between developers and code artifacts, including multiple deep learning model implementations and preprocessing workflows.

## Project Structure

### Main Training Scripts
- **`HanTrain.py`**: Implements HAN (Hierarchical Attention Network) based training workflow, using graph neural networks to process developer-code relationship data
- **`GatTrain.py`**: Implements GAT (Graph Attention Network) based training workflow, incorporating GATConv layers for relationship learning
- **`GtnTrain.py`**: Implements GTN (Graph Transformer Network) based training workflow, using GTN models for multi-relation graph learning
- **`Preprocess/train.py`**: Contains complete VST model (integrating structure, texture, and image features), implementing BertEmbedding layer and multi-modal feature fusion

### Core Modules
- **`utils.py`**: Provides data loading, preprocessing, random seed setting, and logging utility functions
- **`Test.py`**: Model evaluation script, calculating accuracy, precision, recall, F1 score, AUC, and other metrics

### Model Definitions
- **`Models/`**: Deep learning model definition directory
  - `model1.py`: Main model definition, containing unified model interface based on different backbones
  - `han/`: Hierarchical attention network related implementations
  - `gat/`: Graph attention network related components
  - `gtn/`: Graph transformer network model definitions
  - `loss/`: Loss function definitions, including LogLoss, SupConLoss, etc.
  - `kmeans.py`: K-means clustering algorithm implementation

### Preprocessing Module
- **`Preprocess/`**: Data preprocessing directory
  - `train.py`: Contains `BertConfig`, `BertEmbedding` and other components, implementing multi-modal feature extraction and fusion

## Features
- **Multi-model Support**: Supports various graph neural network models including HAN, GAT, GTN
- **Multi-modal Fusion**: Combines structure, texture, image and other features for relationship extraction
- **Complete Pipeline**: Includes complete workflow for training, validation, and testing
- **Performance Evaluation**: Provides multiple evaluation metrics (Accuracy, Precision, Recall, F1, AUC, MCC)

## Usage
Choose the corresponding training script according to specific requirements:
- Use HAN model: `python HanTrain.py`
- Use GAT model: `python GatTrain.py`
- Use GTN model: `python GtnTrain.py`
- Model evaluation: `python Test.py`

## Model Details
Baseline M1 model implements developer-code relationship extraction using multiple deep learning architectures. Through graph neural networks and multi-modal feature fusion techniques, it effectively identifies the association relationships between developers and code.