from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing import custom_seismic_distance


def get_model_ablation_registry(random_seed: int = 42) -> dict:
    """
    Constructs and returns a unified registry pairing choosing classifiers
    with their explicit hyperparameter ablation search spaces.
    """
    
    #Classifier Blueprint Initializations
    knn = KNeighborsClassifier(metric=custom_seismic_distance, algorithm='brute')
    
    dt = DecisionTreeClassifier(class_weight='balanced', random_state=random_seed)
    
    rf = RandomForestClassifier(class_weight='balanced', random_state=random_seed, n_jobs=-1)
    
    #Define Hyperparameter Search Spaces (At least 3 configurations per model)
    registry = {
        "KNN": {
            "estimator": knn,
            "param_grid": {
                "n_neighbors": [3, 5, 7, 11],
                "weights": ["uniform", "distance"]
            }
        },
        "DecisionTree": {
            "estimator": dt,
            "param_grid": {
                "criterion": ["gini", "entropy"],
                "max_depth": [4, 6, 8, 12, None],
                "min_samples_split": [5, 10, 20]
            }
        },
        "RandomForest": {
            "estimator": rf,
            "param_grid": {
                "n_estimators": [50, 100, 200],
                "max_depth": [6, 10, 14],
                "min_samples_leaf": [2, 4]
            }
        }
    }
    
    return registry