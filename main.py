import os

import pandas as pd
from sklearn.model_selection import GridSearchCV

from src.evaluate import (
    f2_scorer,
    generate_performance_report,
    plot_combined_roc_curves,
    plot_confusion_matrix,
)
from src.models import get_model_ablation_registry
from src.preprocessing import fit_transform_pipeline


def run_experimental_pipeline(data_path: str, random_seed: int = 42):
    """
    Orchestrates the complete ML experimental lifecycle: data engineering,
    parallelized hyperparameter ablation passes, champion model extraction,
    and single-canvas ROC manifold evaluation.
    """
    print("=== Step 1: Executing Data Engineering Pipeline ===")
    X_train, X_test, y_train, y_test = fit_transform_pipeline(data_path, random_seed)
    print(f"Data successfully split. Training shape: {X_train.shape}, Test shape: {X_test.shape}\n")
    
    print("=== Step 2: Retrieving Model Ablation Registry ===")
    model_registry = get_model_ablation_registry(random_seed)
    
    final_comparison_records = []
    
    roc_data_collection = {}
    
    print("=== Step 3: Launching Parallel Hyperparameter Ablation Studies ===")
    for model_name, config in model_registry.items():
        print(f"\n--- Running 5-Fold Cross-Validation for: {model_name} ---")
        
        # Initialize execution worker
        # scoring=f2_scorer penalizes False Negatives 4x harder than False Positives
        # refit=True automatically trains the chosen configuration on 100% of the training matrix
        grid_search = GridSearchCV(
            estimator=config["estimator"],
            param_grid=config["param_grid"],
            scoring=f2_scorer,
            cv=5,
            refit=True,
            return_train_score=False
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Optimal Hyperparameters Discovered for {model_name}:")
        print(grid_search.best_params_)
        print(f"Best Validation F2-Score: {grid_search.best_score_:.4f}")
        
        cv_results_df = pd.DataFrame(grid_search.cv_results_)
        ablation_log = cv_results_df[['params', 'mean_test_score', 'std_test_score']]
        
        print("\nComplete Ablation Matrix Results:")
        print(ablation_log.to_string())
        
        os.makedirs("results", exist_ok=True)
        ablation_log.to_csv(f"results/{model_name}_ablation_study.csv", index=False)
        
        # Test Set Generalization Assessment
        champion_model = grid_search.best_estimator_
        y_pred = champion_model.predict(X_test)
        
        if hasattr(champion_model, "predict_proba"):
            y_probs = champion_model.predict_proba(X_test)[:, 1]
        else:
            y_probs = champion_model.predict(X_test)
            
        roc_data_collection[model_name] = (y_test, y_probs)
        
        metrics = generate_performance_report(model_name, y_test, y_pred, y_probs)
        plot_confusion_matrix(y_test, y_pred, filename=f"results/{model_name}_confusion_matrix.png")
        
        metrics["Model"] = model_name
        metrics["Best_Params"] = str(grid_search.best_params_)
        final_comparison_records.append(metrics)
        
    print("\n=== Step 3.5: Generating Combined ROC Manifold Graph ===")
    plot_combined_roc_curves(roc_data_collection, filename="results/combined_roc_curve.png")
        
    print("\n=== Step 4: Compiling Final Test-Set Summary Comparison Matrix ===")
    summary_df = pd.DataFrame(final_comparison_records)
    summary_df = summary_df[["Model", "F2", "AUC", "TP", "FN", "FP", "TN", "Best_Params"]]
    print(summary_df.to_string(index=False))
    summary_df.to_csv("results/final_model_comparison_summary.csv", index=False)

if __name__ == "__main__":
    DATASET_PATH = "data/earthquake_data_tsunami.csv"
    run_experimental_pipeline(DATASET_PATH, random_seed=42)