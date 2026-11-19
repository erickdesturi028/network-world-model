import torch
import torch.nn as nn

class LSTMWorldModel(nn.Module):
    def __init__(self, input_dim: int = 7, hidden_dim: int = 64, num_layers: int = 2, output_dim: int = 1):
        super(LSTMWorldModel, self).__init__()
        
        # Recurrent layer to capture temporal transition dynamics P(S_{t+1} | S_t)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        
        # Dense classification head for future state threat probability
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, output_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: [batch_size, seq_length, input_dim]
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Extract representation from the final sequence timestep
        last_timestep_out = lstm_out[:, -1, :]
        
        # Forecast threat probability
        prediction = self.fc(last_timestep_out)
        return prediction

if __name__ == "__main__":
    # Test instantiation with mock sequence tensor [Batch=32, Seq_Len=10, Features=7]
    sample_input = torch.randn(32, 10, 7)
    model = LSTMWorldModel(input_dim=7, hidden_dim=64)
    output = model(sample_input)
    
    print("LSTM World Model Architecture Instantiated Successfully!")
    print(f"Input Shape:  {sample_input.shape}")
    print(f"Output Shape: {output.shape} -> [Batch_Size, Threat_Probability]")