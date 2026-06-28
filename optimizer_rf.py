import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, fbeta_score, roc_auc_score

from src.evaluate import find_optimal_f2_threshold
from src.preprocessing import fit_transform_pipeline


def main():
    print("=== Loading Data Engineering Pipeline ===")
    X_train, X_test, y_train, y_test = fit_transform_pipeline("data/earthquake_data_tsunami.csv", random_seed=42)
    
    print("=== Training Single Champion Random Forest Instance ===")
    champion_rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        min_samples_leaf=4,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    champion_rf.fit(X_train, y_train)
    
    print("=== Step 1: Optimizing Threshold Strictly on TRAINING Data ===")
    # Extract training probabilities to learn the hyperparameter safely
    y_train_probs = champion_rf.predict_proba(X_train)[:, 1]
    opt_threshold, train_f2 = find_optimal_f2_threshold(y_train.to_numpy(), y_train_probs)
    
    print(f"Discovered Optimal Training Threshold (tau*): {opt_threshold:.4f}")
    print(f"Safe Training F2-Score at this checkpoint:     {train_f2:.4f}")
    
    print("\n=== Step 2: Evaluating Frozen Threshold on HELD-OUT TEST Data ===")
    # Extract test probabilities 
    y_test_probs = champion_rf.predict_proba(X_test)[:, 1]
    
    # Apply the FROZEN threshold to the test probabilities (NO PEEKING!)
    y_pred_test_optimized = (y_test_probs >= opt_threshold).astype(int)
    
    # Calculate the honest, leak-free metrics
    honest_test_f2 = fbeta_score(y_test, y_pred_test_optimized, beta=2.0)
    auc_score = roc_auc_score(y_test, y_test_probs)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test_optimized).ravel()
    
    print("==================================================")
    print(f"HONEST TEST PERFORMANCE (Leak-Free):")
    print(f"Mathematically Optimal Threshold (tau*): {opt_threshold:.4f}")
    print(f"Honest Test F2-Score:                    {honest_test_f2:.4f}")
    print(f"Test ROC-AUC (Invariant):                {auc_score:.4f}")
    print("==================================================")
    
    print(f"\nLeak-Free Optimized Confusion Matrix Layout:")
    print(f"True Positives  (TP): {tp}  <-- Caught Tsunamis")
    print(f"False Negatives (FN): {fn}  <-- Missed Tsunamis")
    print(f"False Positives (FP): {fp}  <-- False Alarms")
    print(f"True Negatives  (TN): {tn}  <-- Correctly Ignored Normal Tremors")
    
    # Structure the record to update our summary file safely
    # 1. Structure the new optimized record map (Statically Type-Safe)
    optimized_record = {
        "Model": "RandomForest (Optimized)",
        "F2": float(honest_test_f2),
        "AUC": float(auc_score),
        "TP": int(tp),
        "FN": int(fn),
        "FP": int(fp),
        "TN": int(tn),
        "Best_Params": f"tau_optimized={opt_threshold:.4f}"
    }
    
    summary_path = "results/final_model_comparison_summary.csv"
    if os.path.exists(summary_path):
        summary_df = pd.read_csv(summary_path)
        summary_df = summary_df[summary_df["Model"] != "RandomForest (Optimized)"]
        summary_df = pd.concat([summary_df, pd.DataFrame([optimized_record])], ignore_index=True)
        summary_df.to_csv(summary_path, index=False)
        print(f"\n[SUCCESS] Updated {summary_path} with completely leak-free metrics.")

if __name__ == "__main__":
    main()