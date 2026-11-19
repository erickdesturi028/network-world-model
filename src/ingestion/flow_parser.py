import pandas as pd
import numpy as np

def parse_flow_csv(file_path: str) -> pd.DataFrame:
    """
    Ingests raw flow CSV logs and extracts macro-level metrics.
    Standardizes feature names and cleans invalid values.
    """
    # Load raw CSV file
    df = pd.read_csv(file_path)
    
    # Clean whitespace and convert column names to lowercase
    df.columns = df.columns.str.strip().str.lower()
    
    # Define primary macro-level features we care about
    feature_cols = [
        'src ip', 'dst ip', 'src port', 'dst port', 'protocol',
        'tot pkts', 'tot size', 'flow dur', 'flow iat mean',
        'flow iat std', 'syn flag cnt', 'ack flag cnt', 'fin flag cnt', 'label'
    ]
    
    # Filter only available features in the uploaded dataset
    available_cols = [col for col in feature_cols if col in df.columns]
    parsed_df = df[available_cols].copy()
    
    # Replace NaN or infinite values with 0
    parsed_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    parsed_df.fillna(0, inplace=True)
    
    return parsed_df

if __name__ == "__main__":
    file_path = "data/raw/sample_flow.csv"
    try:
        print(" Reading and parsing data/raw/sample_flow.csv...")
        parsed_data = parse_flow_csv(file_path)
        print(" Success! Processed Data Shape:", parsed_data.shape)
        print("\nFirst 3 rows of parsed features:")
        print(parsed_data.head(3))
    except FileNotFoundError:
        print(f" Could not find file at {file_path}. Make sure sample_flow.csv is inside data/raw/")