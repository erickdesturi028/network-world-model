import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

def create_sequences(df: pd.DataFrame, seq_length: int = 10, pred_horizon: int = 1):
    """
    Normalizes feature vectors and builds sequence sliding windows (X, y) for PyTorch.
    X: Shape [num_samples, seq_length, num_features]
    y: Target labels or future states S_{t+pred_horizon}
    """
    # Separate features and target label
    feature_cols = [col for col in df.columns if col != 'is_attack']
    features = df[feature_cols].values
    labels = df['is_attack'].values if 'is_attack' in df.columns else np.zeros(len(df))

    # 1. Normalize numerical features
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)

    X_seq = []
    y_seq = []

    # 2. Build sliding windows (X = S_t-seq_length...S_t, y = S_{t+pred_horizon})
    for i in range(len(scaled_features) - seq_length - pred_horizon + 1):
        window = scaled_features[i : i + seq_length]
        target = labels[i + seq_length + pred_horizon - 1]
        
        X_seq.append(window)
        y_seq.append(target)

    # Convert to PyTorch Tensors
    X_tensor = torch.tensor(np.array(X_seq), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y_seq), dtype=torch.float32)

    return X_tensor, y_tensor, scaler

if __name__ == "__main__":
    from flow_parser import parse_flow_csv
    from state_builder import build_state_vectors

    file_path = "data/raw/sample_flow.csv"
    print(" Parsing and constructing state vectors...")
    raw_df = parse_flow_csv(file_path)
    state_df = build_state_vectors(raw_df, window_size=100)

    print(" Building 3D PyTorch sliding window sequences...")
    # Using 10 historical state vectors to forecast the state 1 step ahead
    X, y, scaler = create_sequences(state_df, seq_length=10, pred_horizon=1)

    print("\n Sequence Generation Complete!")
    print(f"X (Input Tensor Shape):  {X.shape} -> [Sequences, Sequence Length, Features]")
    print(f"y (Target Tensor Shape): {y.shape} -> [Sequences]")