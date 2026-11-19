import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ingestion.flow_parser import parse_flow_csv
from ingestion.state_builder import build_state_vectors
from ingestion.sequence_builder import create_sequences
from models.lstm_world_model import LSTMWorldModel

def evaluate_system():
    file_path = "data/raw/sample_flow.csv"
    model_path = "models/lstm_world_model.pth"
    
    print("⏳ Loading sequence dataset...")
    raw_df = parse_flow_csv(file_path)
    state_df = build_state_vectors(raw_df, window_size=100)
    X, y, _ = create_sequences(state_df, seq_length=10, pred_horizon=1)
    
    # Match the stratified evaluation block
    skf = StratifiedKFold(n_splits=5, shuffle=False)
    _, val_idx = next(skf.split(X, y))
    
    X_test = X[val_idx]
    y_test = y[val_idx]
    
    model = LSTMWorldModel(input_dim=7, hidden_dim=64)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    print(f"⏳ Evaluating model on Stratified Holdout Set ({len(X_test)} samples)...")
    with torch.no_grad():
        predictions = model(X_test).squeeze(-1).numpy()
    
    y_true = y_test.numpy()
    y_pred_binary = (predictions > 0.5).astype(int)
    
    auc_score = roc_auc_score(y_true, predictions)
    cm = confusion_matrix(y_true, y_pred_binary)
    
    print("\n" + "="*50)
    print("📊 AI WORLD MODEL EVALUATION REPORT")
    print("="*50)
    print(f"\nROC-AUC Score: {auc_score:.4f}")
    print("\nConfusion Matrix:")
    print(f"  True Negatives (Normal Secure States):   {cm[0][0]}")
    print(f"  False Positives (False Alarms):         {cm[0][1]}")
    print(f"  False Negatives (Missed Threats):       {cm[1][0]}")
    print(f"  True Positives (Forecasted Threats):    {cm[1][1]}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred_binary, target_names=["Benign State", "Attack Imminent"]))
    print("="*50)

if __name__ == "__main__":
    evaluate_system()