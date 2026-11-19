import pandas as pd
import numpy as np

def build_state_vectors(df: pd.DataFrame, window_size: int = 100) -> pd.DataFrame:
    """
    Groups individual parsed flow rows into sequential fixed-size time steps (S_t).
    Computes statistical aggregates for each window to build state vectors.
    """
    # Create window IDs (grouping every `window_size` flows into a single state S_t)
    df['window_id'] = np.arange(len(df)) // window_size
    
    # Aggregation rules for constructing state features
    agg_funcs = {
        'protocol': 'mean',
        'flow iat mean': 'mean',
        'flow iat std': 'mean',
        'syn flag cnt': 'sum',
        'ack flag cnt': 'sum',
        'fin flag cnt': 'sum'
    }
    
    # Filter available columns dynamically
    available_aggs = {k: v for k, v in agg_funcs.items() if k in df.columns}
    
    # Aggregate data per window to form state vector S_t
    state_df = df.groupby('window_id').agg(available_aggs)
    
    # Derive additional macro traffic dynamics
    if 'syn flag cnt' in state_df.columns and 'ack flag cnt' in state_df.columns:
        state_df['syn_ack_ratio'] = (state_df['syn flag cnt'] + 1) / (state_df['ack flag cnt'] + 1)
        
    # Preserve label mapping (if any row in window is an attack, mark state as attack)
    if 'label' in df.columns:
        label_df = df.groupby('window_id')['label'].apply(
            lambda x: 1 if any(str(l).lower() != 'benign' for l in x) else 0
        )
        state_df['is_attack'] = label_df
        
    return state_df

if __name__ == "__main__":
    from flow_parser import parse_flow_csv
    
    file_path = "data/raw/sample_flow.csv"
    print(" Step 1: Parsing raw CSV data...")
    raw_df = parse_flow_csv(file_path)
    
    print(" Step 2: Aggregating into sequential State Vectors (S_t)...")
    state_vectors = build_state_vectors(raw_df, window_size=100)
    
    print(" Success! State Vector Matrix Shape:", state_vectors.shape)
    print("\nFirst 3 State Vectors (S_0, S_1, S_2):")
    print(state_vectors.head(3))