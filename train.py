import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import ScrabbleNNUE
from parser import ScrabbleDataset

def train_engine(epochs=5, batch_size=64, learning_rate=1e-3):
    # 1. Initialize Dataset & DataLoader
    print("Loading dataset...")
    dataset = ScrabbleDataset(data_dir="dataset/gcg_files")
    
    # DataLoader handles batching, shuffling, and multi-processing (workers)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # 2. Initialize Model, Loss Function, and Optimizer
    print("Initializing NNUE Model...")
    model = ScrabbleNNUE()
    
    # Mean Squared Error is common for predicting scalar values like point differentials mapping
    criterion = nn.MSELoss()
    
    # Adam optimizer is typical for fast learning
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Check for GPU (Apple Silicon uses 'mps', Nvidia uses 'cuda')
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")
    model.to(device)
    
    print("\nStarting Training Loop...")
    for epoch in range(epochs):
        model.train() # Set to train mode
        total_loss = 0.0
        
        for batch_idx, (board_tensor, rack_tensor, equity_label) in enumerate(dataloader):
            # Move data to target device memory (GPU/CPU)
            board_tensor = board_tensor.to(device)
            rack_tensor = rack_tensor.to(device)
            equity_label = equity_label.to(device)
            
            # Zero out gradients from previous step
            optimizer.zero_grad()
            
            # Forward pass: Feed the data into the model
            predictions = model(board_tensor, rack_tensor)
            
            # Calculate loss (Prediction vs Exact label)
            loss = criterion(predictions, equity_label)
            
            # Backward pass: Compute gradients
            loss.backward()
            
            # Step the weights!
            optimizer.step()
            
            total_loss += loss.item()
            
            # Print intermediate progress
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx+1}/{len(dataloader)}] | Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"=== Epoch {epoch+1} Completed | Average Loss: {avg_loss:.4f} ===\n")
        
        # Save checkpoints!
        torch.save(model.state_dict(), f"scrabble_nnue_epoch_{epoch+1}.pt")
        print(f"Model saved to scrabble_nnue_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    import os
    if not os.path.exists("dataset/gcg_files"):
        print("Dataset directory not found!")
    else:
        # Warning: For real training on 48,000 files, batch_size should be large (2048+)
        train_engine(epochs=5, batch_size=64)
