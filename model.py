import torch
import torch.nn as nn
import torch.nn.functional as F

# 15x15 board = 225 squares
# 27 tile types: A-Z (26) + Blank (1)
BOARD_FEATURES = 225 * 27

# Rack represents the count of each tile you hold right now (0-7 for most tiles, 0-2 for blanks)
RACK_FEATURES = 27

TOTAL_INPUT_FEATURES = BOARD_FEATURES + RACK_FEATURES

class ClippedReLU(nn.Module):
    """
    Standard activation in NNUE architectures.
    Bounds output strictly between 0 and 1 (or 0 and 127 for int8 quantization).
    We'll use 1.0 for float training.
    """
    def __init__(self, max_val=1.0):
        super(ClippedReLU, self).__init__()
        self.max_val = max_val

    def forward(self, x):
        return torch.clamp(x, 0.0, self.max_val)


class ScrabbleNNUE(nn.Module):
    """
    Efficiently Updatable Neural Network for Scrabble block evaluations.
    Takes a sparsely encoded board and a densely encoded rack as inputs.
    """
    def __init__(self, hidden_dim=256, l2_dim=32, l3_dim=32):
        super(ScrabbleNNUE, self).__init__()
        
        # 1. Feature Transformer Layer (The Accumulator)
        # This layer acts as the updatable cache during move evaluation.
        # Shape: (Batch, TOTAL_INPUT_FEATURES) -> (Batch, hidden_dim)
        self.feature_transformer = nn.Linear(TOTAL_INPUT_FEATURES, hidden_dim)
        
        # Intermediate layers
        self.layer2 = nn.Linear(hidden_dim, l2_dim)
        self.layer3 = nn.Linear(l2_dim, l3_dim)
        
        # Output evaluate -> scalar (win prob or equity/point spread)
        self.output_layer = nn.Linear(l3_dim, 1)
        
        self.activation = ClippedReLU(max_val=1.0)
        
    def forward(self, board_sparse_tensor, rack_dense_tensor):
        """
        board_sparse_tensor: shape (B, 6075) - usually one-hot.
        rack_dense_tensor: shape (B, 27) - counts of tiles on rack.
        """
        # Combine features inside the network wrapper for simple processing
        x = torch.cat([board_sparse_tensor, rack_dense_tensor], dim=1)
        
        # Accumulator phase
        acc = self.feature_transformer(x)
        acc = self.activation(acc)
        
        # Deep network phase
        out = self.layer2(acc)
        out = self.activation(out)
        
        out = self.layer3(out)
        out = self.activation(out)
        
        # Output score
        score = self.output_layer(out)
        
        return score

if __name__ == "__main__":
    # Small test runner to print the param count and test a dummy pass
    model = ScrabbleNNUE()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Initialized Scrabble NNUE with {num_params:,} train parameters.")
    
    # Dummy tensors (Batch_size=2)
    dummy_board = torch.zeros(2, BOARD_FEATURES)
    # Put an 'A' (index 0) at square H8 (index 112)
    dummy_board[0, 112 * 27 + 0] = 1.0 
    
    dummy_rack = torch.zeros(2, RACK_FEATURES)
    dummy_rack[0, 0] = 2.0 # Client has 2 'A's on rack
    
    out_score = model(dummy_board, dummy_rack)
    print(f"Sample Model Output Shape: {out_score.shape}")
    print(f"Sample Outputs: {out_score.detach().flatten().tolist()}")
