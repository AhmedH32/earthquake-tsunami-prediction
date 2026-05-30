import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


def load_and_split_data(file_path: str, random_seed: int = 42):
    """
    Loads, cleans, and partitions the tsunami dataset using stratified sampling.
    """
    df = pd.read_csv(file_path)
    
    # Clean the 18 anomalous hardware reporting errors
    mask = (df['nst'] == 0) & (df['gap'] == 0)
    df_clean = df[~mask].copy().reset_index(drop=True)
    
    # Isolate targets (Year is dropped implicitly here by omitting it from features later)
    X = df_clean.drop(columns=['tsunami', 'Year'])
    y = df_clean['tsunami']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=random_seed, 
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def engineer_spatial_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Transforms geographic Latitude/Longitude degrees into 3D Cartesian coordinates (x, y, z).
    """
    X_train_out = X_train.copy()
    X_test_out = X_test.copy()
    
    for df in [X_train_out, X_test_out]:
        lat_rad = df['latitude'] * (np.pi / 180.0)
        lon_rad = df['longitude'] * (np.pi / 180.0)
        
        df['coord_x'] = np.cos(lat_rad) * np.cos(lon_rad)
        df['coord_y'] = np.cos(lat_rad) * np.sin(lon_rad)
        df['coord_z'] = np.sin(lat_rad)
        
        df.drop(columns=['latitude', 'longitude'], inplace=True)
        
    return X_train_out, X_test_out

def scale_continuous_features(X_train: pd.DataFrame, X_test: pd.DataFrame, columns_to_scale: list):
    """
    Fits a RobustScaler strictly to the training set parameters to prevent leakage.
    """
    X_train_out = X_train.copy()
    X_test_out = X_test.copy()
    
    scaler = RobustScaler()
    scaler.fit(X_train_out[columns_to_scale])
    
    X_train_out[columns_to_scale] = scaler.transform(X_train_out[columns_to_scale])
    X_test_out[columns_to_scale] = scaler.transform(X_test_out[columns_to_scale])
    
    return X_train_out, X_test_out

def custom_seismic_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Optimized hybrid metric for KNN tracking month at index 0.
    """
    m1 = x1[0]
    m2 = x2[0]
    
    raw_month_dist = abs(m1 - m2)
    circular_month_dist = min(raw_month_dist, 12.0 - raw_month_dist)
    scaled_month_dist = circular_month_dist / 6.0
    
    other_diffs = x1[1:] - x2[1:]
    total_other_dist_squared = np.sum(other_diffs ** 2)
    
    total_distance = np.sqrt((scaled_month_dist ** 2) + total_other_dist_squared)
    return float(total_distance)

def fit_transform_pipeline(file_path: str, random_seed: int = 42):
    """
    The master pipeline engine. Executes loading, cleaning, spatial mapping, 
    continuous feature scaling, and explicit structural column reordering.
    """
    X_train, X_test, y_train, y_test = load_and_split_data(file_path, random_seed)
    X_train, X_test = engineer_spatial_features(X_train, X_test)
    
    
    columns_to_scale = ['magnitude', 'depth', 'sig', 'nst', 'cdi', 'mmi', 'dmin', 'gap']
    X_train, X_test = scale_continuous_features(X_train, X_test, columns_to_scale)
    
    
    ordered_features = [
        'Month',  
        'coord_x', 'coord_y', 'coord_z', 
        'magnitude', 'depth', 'sig', 'nst', 'cdi', 'mmi', 'dmin', 'gap'
    ]
    
    X_train_final = X_train[ordered_features]
    X_test_final = X_test[ordered_features]
    
    return X_train_final, X_test_final, y_train, y_test