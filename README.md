# SGSMA 2026 - Synchrophasor Anomaly Detection (IEEE 39-Bus System)

## Overview

This project is a machine learning solution for **anomaly detection, event classification, and localization** using high-frequency synchrophasor (PMU) data from the IEEE 39-Bus power system.

The system was built for the **SGSMA 2026 Synchrophasor Anomaly Detection Competition**, where the goal is to analyze real-time grid behavior and identify disturbances such as faults, outages, generator changes, load changes, and data quality issues.

The pipeline combines **time-series processing, multi-PMU feature fusion, engineered electrical features, and gradient boosting models (LightGBM)** to build a robust grid monitoring framework.

---

## Problem Statement

Given synchronized PMU data from multiple buses, the model must:

1. **Detect anomalies** (Normal vs abnormal grid behavior)
2. **Classify events** into one of the following:
   - Normal (0)
   - Fault (1)
   - Line Outage (2)
   - Generator Change (3)
   - Load Change (4)
   - Missing Data (5)
   - Missing + Physical (6)
   - Bad Data (7)
   - Unknown (8)

3. **Localize events** to identify the originating bus in the system.

---

## Dataset Description

### PMU Sources
Data is collected from 8 buses in the IEEE 39-bus system:

- Bus 2  
- Bus 5  
- Bus 6  
- Bus 10  
- Bus 19  
- Bus 22  
- Bus 29  
- Bus 39  

### Sampling Rate
- 30 frames per second (30 FPS)

### Features per PMU
Each PMU stream includes:

- `TIMESTAMP`
- Voltage (3-phase magnitude & angle)
- Current (3-phase magnitude & angle)
- Frequency
- ROCOF (Rate of Change of Frequency)
- `DATA_PRESENT`

---

## Methodology

### 1. Data Preprocessing

- All PMU streams are aligned using `TIMESTAMP`
- Missing timestamps and inconsistencies handled during merging
- Data from all buses is fused into a single multivariate time-series structure
- Label encoding applied for multi-class classification
- Careful splitting to avoid **temporal leakage**

---

### 2. Feature Engineering

To improve model performance, domain-informed features were engineered:

#### Temporal Features
- First-order differences:
  - ΔVoltage (dV/dt)
  - ΔCurrent (dI/dt)
  - ΔFrequency (dF/dt)
- ROCOF used as a dynamic system instability indicator

#### Electrical Features
- Phase imbalance indicators
- Magnitude-angle relationships
- Derived power-related approximations

#### Spatial Features
- Cross-bus feature concatenation
- Implicit learning of grid-wide interactions across PMUs

---

### 3. Windowing Strategy

Since PMU data is sequential:

- A sliding window approach is applied
- Each sample represents a short time sequence (~1 second windows)
- Each window is labeled based on the dominant event within that interval

This converts raw streaming data into supervised learning samples.

---

### 4. Model Architecture

The primary model used is:

### ✅ LightGBM Classifier

Reasons for selection:
- Strong performance on structured/tabular data
- Handles high-dimensional inputs efficiently
- Robust to noise and feature redundancy
- Fast training and inference

#### Training Setup:
- Multi-class classification
- Class weighting used to handle imbalance
- Early stopping based on validation performance
- Label encoding to ensure contiguous class indices (0–8)

---

## Evaluation Metrics

The model is evaluated using:

- **Macro F1-score** (primary metric)
- **Classification accuracy**
- **Confusion matrix analysis**
- **Localization accuracy (Top-1 bus prediction)**

---

## Results

### Performance Summary

- **Anomaly Detection F1-score:** ~0.97  
- **Event Classification Macro F1-score:** ~0.60  
- **Localization Accuracy:** ~0.90  

### Observations

- The model performs extremely well in **detecting disturbances**
- Localization is strong due to multi-PMU feature fusion
- Classification is harder due to:
  - Similarity between event types
  - Severe class imbalance
  - Overlapping grid behaviors

---

## Strengths of the Approach

- End-to-end PMU data fusion pipeline
- Strong performance on detection and localization tasks
- Efficient and scalable LightGBM-based architecture
- Domain-informed feature engineering
- Proper handling of time-series structure via windowing

---

## Limitations

- Moderate performance on fine-grained event classification
- Limited explicit modeling of grid topology
- No graph-based learning of bus connectivity
- Class imbalance impacts minority event categories

---

## Future Improvements

- Replace or complement LightGBM with:
  - LSTM / GRU for temporal learning
  - Temporal Convolutional Networks (TCN)
- Introduce Graph Neural Networks (GNNs) for grid structure modeling
- Improve class imbalance handling using:
  - Focal loss
  - Advanced sampling strategies
- Explore ensemble methods for better generalization

---

## Project Structure

```

.
├── data/
│   ├── bus_2.csv
│   ├── bus_5.csv
│   ├── ...
│
├── notebooks/
│   └── SGSMA_pipeline.ipynb
│
├── models/
│   └── lightgbm_model.pkl
│
├── README.md
└── requirements.txt

```

---

## Requirements

```

numpy
pandas
scikit-learn
lightgbm

## GitHub Publishing Notes

This repository is configured to keep the code and model artifacts while excluding large raw datasets and generated prediction outputs.

Ignored by `.gitignore`:
- `Competition_Testing Data Set 1/`
- `Competition_Testing Data Set 2/`
- `Training Dataset/`
- `predictions/`
- `predictions_output_test1/`
- `predictions_output_test2/`
- `F_Bamidele_SGSMA2026/`

When pushing to GitHub, commit the project source files, notebook, and README only; local dataset directories and output folders are intentionally excluded to keep the repository clean and lightweight.

matplotlib
seaborn

```

---

## Author

Machine Learning Engineer: Faith Bamidele  
Project Partner: Research collaborator (power systems domain research support)

---

## Notes

This project is designed for real-time power system monitoring scenarios using synchrophasor data and demonstrates how machine learning can be applied to grid stability analysis, anomaly detection, and event localization.
