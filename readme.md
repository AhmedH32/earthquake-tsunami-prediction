# Earthquake-Induced Tsunami Prediction: Spatial Feature Engineering and Cost-Asymmetric Optimization

[![Paper](https://img.shields.io/badge/Paper-Download_PDF-B31B1B?style=for-the-badge&logo=adobeacrobatreader&logoColor=white)](./Tsunami_Prediction_Spatial_Calibration_Paper.pdf)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](./LICENSE)

**Author:** Ahmed Hassan  
**Affiliation:** Arab Academy for Science, Technology and Maritime Transport (AASTMT)  
**Paper Artifact:** [`Tsunami_Prediction_Spatial_Calibration_Paper.pdf`](./Tsunami_Prediction_Spatial_Calibration_Paper.pdf)

---

## Abstract

Geophysical disaster warning systems operate under severe cost asymmetry: while a False Positive (false alarm) induces evacuation logistics and minor economic friction, a False Negative (undetected tsunami) results in catastrophic loss of life. Standard machine learning classifiers default to symmetric 0.5 decision boundaries, systematically optimizing for raw accuracy while penalizing the rare, high-consequence class.

This work introduces a geometric and decision-theoretic framework for seismic tsunami prediction:
1. **Geometric Manifold Projection:** Resolves anti-meridian boundary distortions inherent in planar latitude/longitude coordinates by projecting spherical coordinates onto a continuous 3D Cartesian manifold.
2. **Cyclical Temporal Modeling:** Eliminates calendar discontinuity boundaries via orthogonal trigonometric basis functions.
3. **Decoupled Cost-Asymmetric Calibration:** Optimizes the operational decision threshold via an $\mathcal{O}(N \log N)$ coordinate sweep over validation probability manifolds under an $F_2$-score objective (weighting recall twice as heavily as precision).

Empirical validation confirms that geometric feature projections elevate baseline KNN density estimation from an $F_2$-score of 0.6401 to 0.7288 (+13.8%), while decision boundary calibration elevates Random Forest generalization $F_2$-score to 0.9223, achieving 96.6% recall (57/59 tsunamis detected).

---

## Mathematical Formulation

### 1. 3D Cartesian Manifold Projection
Planar coordinate representations $\phi \in [-90^\circ, 90^\circ]$ (latitude) and $\lambda \in [-180^\circ, 180^\circ]$ (longitude) introduce an artificial boundary discontinuity along the $180^\circ$ meridian. To preserve geodesic neighborhood topology in Euclidean distance-based estimators, coordinates are mapped to the unit 2-sphere $\mathbb{S}^2 \subset \mathbb{R}^3$:

$$\begin{aligned} x &= \cos(\phi) \cdot \cos(\lambda) \\ y &= \cos(\phi) \cdot \sin(\lambda) \\ z &= \sin(\phi) \end{aligned}$$

### 2. Cyclical Temporal Encodings
To model seasonal geophysical patterns without introducing boundary penalties between December ($m=12$) and January ($m=1$), temporal indices are projected onto an orthogonal circular basis:

$$\mathbf{t}_{\text{month}} = \left[ \sin\left(\frac{2\pi m}{12}\right), \; \cos\left(\frac{2\pi m}{12}\right) \right]^T$$

### 3. Cost-Asymmetric Optimization ($F_2$-Score)
Given the safety-critical priority of minimizing False Negatives, models are optimized with respect to the parameterized $F_\beta$ metric with $\beta = 2$:

$$F_2 = (1 + 2^2) \frac{\text{Precision} \cdot \text{Recall}}{2^2 \cdot \text{Precision} + \text{Recall}} = \frac{5 \cdot \text{TP}}{5 \cdot \text{TP} + 4 \cdot \text{FN} + \text{FP}}$$

The operational decision threshold $\tau^* \in [0, 1]$ is identified via an empirical coordinate sweep:

$$\tau^* = \arg\max_{\tau} F_2 \left( \hat{y}_\tau, y \right), \quad \hat{y}_\tau = \mathbb{I}(\hat{P}(Y=1 \mid X) \ge \tau)$$

---

## Empirical Benchmarks & Results

### 1. Spatial & Temporal Feature Ablation (K-Nearest Neighbors)
To isolate and validate the impact of geometric transformations on local density estimation, feature pipelines were evaluated under identical 5-fold cross-validation splits:

| Configuration | Features Included | $F_2$ Score | ROC-AUC | Key Mechanism |
| :--- | :--- | :---: | :---: | :--- |
| **Raw Baseline** | Raw Latitude, Longitude | 0.6401 | 0.7729 | Standard planar coordinate mapping |
| **3D Spatial Only** | Cartesian $x, y, z$ | 0.6734 | 0.8143 | Eliminates anti-meridian fragmentation |
| **Cyclical Temporal** | $\sin(2\pi m/12), \cos(2\pi m/12)$ | 0.7021 | 0.7996 | Preserves December-January continuity |
| **Unified Champion** | **3D Spatial + Cyclical Temporal** | **0.7288** | **0.8384** | **Orthogonal geometric representation** |

### 2. Random Forest Operational Calibration Sweep
While the uncalibrated Random Forest achieves strong structural discrimination (ROC-AUC: 0.9497), the canonical $\tau = 0.5$ threshold squashes probability densities and allows 7 tsunamis to pass undetected. Calibrating the decision boundary to $\tau^* = 0.3236$ recovers 5 additional events, suppressing missed detections by 71.4%:

| Decision Threshold | Test $F_2$ | True Positives (Caught) | False Negatives (Missed) | False Positives (Alarms) | Recall |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Default ($\tau = 0.5000$) | 0.8700 | 52 / 59 | 7 | **11** | 88.1% |
| **Calibrated ($\tau^* = 0.3236$)** | **0.9223** | **57 / 59** | **2** | 16 | **96.6%** |

---

## Repository Structure

```
├── data/                       # Seismic telemetry datasets
├── src/                        # Data processing & feature pipeline modules
├── results/                    # Validation metrics and evaluation plots
├── knn_feature_ablation.py     # Isolated spatial & temporal ablation benchmark
├── optimizer_rf.py             # O(N log N) threshold calibration sweep
├── main.py                     # 5-fold cross-validation pipeline runner
├── requirements.txt            # Environment dependency manifest
├── Tsunami_Prediction_Spatial_Calibration_Paper.pdf  # Formal research manuscript
└── README.md                   # Project documentation
```

---

## Reproduction & Setup

### 1. Environment Setup
```bash
python -m venv .venv
source .venv/Scripts/activate  # On Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Execute Feature Ablations
```bash
python knn_feature_ablation.py
```

### 3. Run Decision Threshold Sweep
```bash
python optimizer_rf.py
```

---

## Citation

If you use this work, the feature engineering pipeline, or the threshold calibration methodology, please cite:

```bibtex
@article{hassan2026tsunami,
  title={Earthquake-Induced Tsunami Prediction: Spatial Feature Engineering and Cost-Asymmetric Optimization},
  author={Hassan, Ahmed},
  journal={Department of Computer Engineering, AASTMT},
  year={2026},
  url={[https://github.com/AhmedH32/earthquake-tsunami-prediction](https://github.com/AhmedH32/earthquake-tsunami-prediction)}
}
```