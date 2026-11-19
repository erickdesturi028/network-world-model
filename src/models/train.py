import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ingestion.flow_parser import parse_flow_csv
from ingestion.state_builder import build_state_vectors
from ingestion.sequence_builder import create_sequences
from models.lstm_world_model import LSTMWorldModel

def train_world_model():
    file_path = "data/raw/sample_flow.csv"
    
    print("⏳ Loading raw data and generating state sequences...")
    raw_df = parse_flow_csv(file_path)
    state_df = build_state_vectors(raw_df, window_size=100)
    X, y, scaler = create_sequences(state_df, seq_length=10, pred_horizon=1)
    
    # Stratified Time-Block Split (Preserves class ratio across temporal blocks)
    skf = StratifiedKFold(n_splits=5, shuffle=False)
    train_idx, val_idx = next(skf.split(X, y))
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    print(f"📊 Stratified Sequence Split Complete:")
    print(f"   • Train Set: {len(X_train)} samples | Attack Ratio: {y_train.mean():.2%}")
    print(f"   • Test Set:  {len(X_val)} samples | Attack Ratio: {y_val.mean():.2%}")
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    model = LSTMWorldModel(input_dim=7, hidden_dim=64)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 5
    print("\n🚀 Retraining World Model with Stratified Time-Blocks...")
    model.train()
    
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X).squeeze(-1)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch}/{epochs}] - Loss: {avg_loss:.4f}")
        
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/lstm_world_model.pth")
    print("\n✅ Model retrained and saved to 'models/lstm_world_model.pth'")

if __name__ == "__main__":
    train_world_model()