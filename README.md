# A GNN Routing Module Is All You Need for LSTM Rainfall-Runoff Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyTorch%20Geometric-2.0+-3C2179.svg)](https://pytorch-geometric.readthedocs.io/)

Official implementation of the paper: **"A GNN Routing Module Is All You Need for LSTM Rainfall-Runoff Models"**

**Authors:** Hamidreza Mosaffa, Florian Pappenberger, Christel Prudhomme, Matthew Chantry, Christoph Rüdiger, Hannah Cloke

**Affiliations:**
- University of Reading, UK
- European Centre for Medium-Range Weather Forecasts (ECMWF)

## 📋 Overview

This repository contains the implementation of a novel LSTM-Graph Neural Network (GNN) framework for rainfall-runoff modeling that explicitly integrates runoff generation and spatial flow routing. The framework combines:

- **LSTM networks** for local temporal runoff generation at each subbasin
- **Graph Neural Networks** for spatial flow routing across the river network topology

## 🏗️ Architecture

```
Input: [Precipitation, Temperature, Soil Moisture] + Static Catchment Attributes
    ↓
LSTM Encoder (per subbasin)
    ↓
Node Embeddings [temporal + static features]
    ↓
GNN Module (GAT/GCN/GraphSAGE/ChebNet)
    ↓
Output: Daily Discharge Predictions
```

## 📦 Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric 2.0+
- CUDA 11.0+ (optional, for GPU acceleration)

```

## 📧 Contact

- **Hamidreza Mosaffa** - h.mosaffa@reading.ac.uk

