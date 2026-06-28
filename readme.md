# Tsunami Prediction: Spatial Engineering & Cost-Asymmetric Optimization

This repository contains the data and ML pipelines for my capstone research on predicting earthquake-induced tsunamis from seismic telemetry. 

The core challenge of this dataset isn't just the class imbalance—it's the **cost asymmetry**. In disaster warning systems, a False Negative (missing a tsunami) has catastrophic consequences, whereas a False Positive (a false alarm) is just an inconvenience. Standard classifiers default to a 0.5 decision threshold and optimize for raw accuracy, which fails completely in this domain.

This project fixes that by explicitly engineering the spatial geometry of the data and tuning the operational thresholds to penalize missed tsunamis.

## Key Implementation Details

1. **Fixing Map Distortion (3D Projection):** Standard lat/lon coordinates break distance-based algorithms (like KNN) at the anti-meridian. I projected the geographic data onto a 3D Cartesian manifold ($x, y, z$) to preserve actual physical distances.
2. **Cyclical Time:** Handled the December-to-January boundary jump by using a continuous circular temporal encoding for the `Month` feature.
3. **Leak-Free Scaling:** Built a robust normalization pipeline that explicitly fits the `RobustScaler` only on training splits to prevent data leakage.
4. **Threshold Calibration:** Instead of accepting a Random Forest's default 0.5 threshold, I decoupled the structural training from the deployment calibration, using an $O(N \log N)$ sweep on the training probabilities to find the safest operational boundary.

## Results & Benchmarks

All models were evaluated using the **$F_2$-score**, which weights recall twice as heavily as precision (penalizing False Negatives 4x harder than False Positives).

### 1. Spatial Engineering Validation (KNN Ablation)
I isolated the feature engineering steps to prove the math actually improves the baseline density estimation:

| Configuration | $F_2$ Score | ROC-AUC | Note |
| :--- | :---: | :---: | :--- |
| Raw Baseline | 0.6401 | 0.7729 | Standard lat/lon |
| 3D Spatial Only | 0.6734 | 0.8143 | Fixed anti-meridian mapping |
| Cyclical Temporal | 0.7021 | 0.7996 | Fixed calendar boundaries |
| **Unified Champion** | **0.7288** | **0.8384** | **Combined pipeline** |

### 2. The Random Forest Threshold Optimization
At the default 0.5 threshold, the Random Forest mapped the underlying physics perfectly (AUC: 0.9497) but squashed probabilities and missed 7 tsunamis. By shifting the threshold to a safely calibrated $\tau^* = 0.3236$, the test $F_2$-score spiked, capturing 96.6% of actual events.

| Threshold | Generalization $F_2$ | True Positives (Caught) | False Negatives (Missed) | False Positives (Alarms) |
| :---: | :---: | :---: | :---: | :---: |
| Default (0.5000) | 0.8700 | 52 / 59 | 7 | 11 |
| **Calibrated (0.3236)** | **0.9223** | **57 / 59** | **2** | **16** |

## Running the Code

Install the dependencies:
```bash
pip install -r requirements.txt
```

Run the full 5-fold cross-validation grid search:
```bash
python main.py
```

Execute the isolated spatial feature ablation:
```bash
python knn_feature_ablation.py
```

Run the leak-free threshold coordinate sweep:
```bash
python optimizer_rf.py
```