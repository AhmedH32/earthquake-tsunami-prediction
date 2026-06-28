import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler

from src.preprocessing import engineer_spatial_features, load_and_split_data


def custom_distance_baseline(x1: np.ndarray, x2: np.ndarray) -> float:
    # Standard linear Euclidean distance array pass
    return float(np.sqrt(np.sum((x1 - x2) ** 2)))

def custom_distance_cyclical_only(x1: np.ndarray, x2: np.ndarray) -> float:
    # Explicitly typed to satisfy __getitem__ overloads for strict compilation
    m1, m2 = float(x1[0]), float(x2[0])
    raw_month_dist = abs(m1 - m2)
    circular_month_dist = min(raw_month_dist, 12.0 - raw_month_dist)
    scaled_month_dist = circular_month_dist / 6.0
    
    other_diffs = x1[1:] - x2[1:]
    return float(np.sqrt((scaled_month_dist ** 2) + np.sum(other_diffs ** 2)))

def run_ablation_node(X_tr: np.ndarray, X_te: np.ndarray, y_tr: pd.Series, y_te: pd.Series, metric_fn: callable) -> tuple[float, float]:
    clf = KNeighborsClassifier(n_neighbors=3, weights='uniform', metric=metric_fn, algorithm='brute')
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    probs = clf.predict_proba(X_te)[:, 1]
    
    f2 = float(fbeta_score(y_te, preds, beta=2.0))
    auc = float(roc_auc_score(y_te, probs))
    return f2, auc

def main() -> None:
    print("=== Launching Isolated KNN Feature Ablation Pass ===")
    X_train_raw, X_test_raw, y_train, y_test = load_and_split_data("data/earthquake_data_tsunami.csv", random_seed=42)
    
    # Active features mapped inside your pipeline matrix
    base_cols = ['magnitude', 'depth', 'sig', 'nst', 'cdi', 'mmi', 'dmin', 'gap']
    
    # Secure uniform scaling via RobustScaler parameters to insulate variables safely
    scaler = RobustScaler()
    X_tr_scaled = X_train_raw.copy()
    X_te_scaled = X_test_raw.copy()
    X_tr_scaled[base_cols] = scaler.fit_transform(X_tr_scaled[base_cols])
    X_te_scaled[base_cols] = scaler.transform(X_te_scaled[base_cols])
    
    # --- Variant 1: Raw Baseline (No Spatial or Temporal Engineering) ---
    order1 = ['Month', 'latitude', 'longitude'] + base_cols
    X_tr1 = X_tr_scaled[order1].to_numpy()
    X_te1 = X_te_scaled[order1].to_numpy()
    f2_1, auc_1 = run_ablation_node(X_tr1, X_te1, y_train, y_test, custom_distance_baseline)
    
    # --- Variant 2: Spatial Projected Only ---
    X_tr_sp, X_te_sp = engineer_spatial_features(X_train_raw, X_test_raw)
    X_tr_sp[base_cols] = scaler.fit_transform(X_tr_sp[base_cols])
    X_te_sp[base_cols] = scaler.transform(X_te_sp[base_cols])
    
    order2 = ['Month', 'coord_x', 'coord_y', 'coord_z'] + base_cols
    X_tr2 = X_tr_sp[order2].to_numpy()
    X_te2 = X_te_sp[order2].to_numpy()
    f2_2, auc_2 = run_ablation_node(X_tr2, X_te2, y_train, y_test, custom_distance_baseline)
    
    # --- Variant 3: Cyclical Temporal Only ---
    f2_3, auc_3 = run_ablation_node(X_tr1, X_te1, y_train, y_test, custom_distance_cyclical_only)
    
    # --- Variant 4: Complete Pipeline (Our Optimized Champion Architecture) ---
    f2_4, auc_4 = run_ablation_node(X_tr2, X_te2, y_train, y_test, custom_distance_cyclical_only)
    
    print("\n==================================================")
    print("KNN FEATURE ABLATION RESULTS MATRIX")
    print("==================================================")
    print(f"1. Raw Coordinates (Null Baseline):   F2 = {f2_1:.4f} | AUC = {auc_1:.4f}")
    print(f"2. Spatial Projected Only (3D Sphere): F2 = {f2_2:.4f} | AUC = {auc_2:.4f}")
    print(f"3. Cyclical Temporal Only (Circular): F2 = {f2_3:.4f} | AUC = {auc_3:.4f}")
    print(f"4. Full Pipeline (Unified Champion):  F2 = {f2_4:.4f} | AUC = {auc_4:.4f}")
    print("==================================================")

if __name__ == "__main__":
    main()