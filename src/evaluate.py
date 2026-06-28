import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    fbeta_score,
    make_scorer,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def calculate_f2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the F2-score.
    An F2-score weights recall 2x higher than precision (4x penalty on FN).
    """
    return float(fbeta_score(y_true, y_pred, beta=2.0))

# Create the stateful scorer object for GridSearchCV
# response_methods='predict' tells GridSearch to pass binary 0/1 predictions to our function
f2_scorer = make_scorer(calculate_f2_score, response_method='predict')

def generate_performance_report(model_name: str, y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray):
    """
    Calculates and prints the core evaluation metrics for the final test set comparison table.
    """
    f2 = calculate_f2_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_probs)
    
    # Extract raw structural confusion elements
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    print(f"=== Final Test Evaluation: {model_name} ===")
    print(f"F2-Score: {f2:.4f}")
    print(f"ROC-AUC:  {auc:.4f}")
    print(f"Confusion Matrix -> TP: {tp}, FN: {fn}, FP: {fp}, TN: {tn}\n")
    
    return {"F2": f2, "AUC": auc, "TP": tp, "FN": fn, "FP": fp, "TN": tn}

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, filename: str = "confusion_matrix.png"):
    """
    Generates an optimized, row-normalized confusion matrix heatmap
    and saves the output figure directly to disk for the LaTeX report.
    """
    #Compute the raw integer matrix
    cm = confusion_matrix(y_true, y_pred)
    
    #Compute the vectorized row-normalized rates (float64)
    cm_normalized = cm / cm.sum(axis=1, keepdims=True)
    
    #Construct a parallel 2D string array for custom visual text annotations
    #We map a string token containing the raw count and the percentage into each cell
    labels = np.empty((2, 2), dtype=object)
    
    #Extract the flat components cleanly
    tn, fp, fn, tp = cm.ravel()
    tn_r, fp_r, fn_r, tp_r = cm_normalized.ravel()
    
    labels[0, 0] = f"True Normal\n{tn}\n({tn_r*100:.1f}%)"
    labels[0, 1] = f"False Alarm\n{fp}\n({fp_r*100:.1f}%)"
    labels[1, 0] = f"Missed Tsunami\n{fn}\n({fn_r*100:.1f}%)"
    labels[1, 1] = f"True Tsunami\n{tp}\n({tp_r*100:.1f}%)"
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    sns.heatmap(
        data=cm_normalized, 
        annot=labels, 
        fmt="", 
        cmap="Blues", 
        cbar=True, 
        ax=ax,
        xticklabels=["Predicted Normal", "Predicted Tsunami"],
        yticklabels=["Actual Normal", "Actual Tsunami"]
    )
    
    ax.set_title("Normalized Seismic Confusion Matrix Matrix Layout")
    plt.tight_layout()
    
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_combined_roc_curves(roc_data: dict, filename: str = "results/combined_roc_curve.png"):
    """
    Plots the ROC curves for all evaluated models onto a single 
    matplotlib canvas for direct academic comparison.
    
    roc_data: Dict structured as {"ModelName": (y_true, y_probs)}
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    
    for model_name, (y_true, y_probs) in roc_data.items():
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        auc_score = roc_auc_score(y_true, y_probs)
        
        ax.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.4f})", lw=2)
        
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Guessing (AUC = 0.5000)")
    
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title("Receiver Operating Characteristic (ROC) Manifold Comparison")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300)
    plt.close()




def find_optimal_f2_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    
    denominator = (4 * precision) + recall
    f2_scores = np.where(denominator == 0, 0.0, (5 * precision * recall) / denominator)
    
    best_idx = np.argmax(f2_scores)
    
  
    if best_idx >= len(thresholds):
        best_idx = len(thresholds) - 1
        
    optimal_threshold = float(thresholds[best_idx])
    best_f2 = float(f2_scores[best_idx])
    
    return optimal_threshold, best_f2