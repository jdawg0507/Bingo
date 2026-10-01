import torch
from torch.utils.data import Dataset
import os
import random

from model import BOARD_FEATURES, RACK_FEATURES

class ScrabbleDataset(Dataset):
    """
    Parses .gcg files and converts them into PyTorch tensors for NNUE training.
    """
    def __init__(self, data_dir="dataset/gcg_files"):
        self.data_dir = data_dir
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".gcg")]

    def __len__(self):
        # We simulate returning a single state per file for easy batching,
        # but randomly sampling which move from the game we simulate up to.
        return len(self.files)

    def _parse_coords(self, loc):
        # 8G -> row 8, col G (Across)
        # F6 -> col F, row 6 (Down)
        if not loc or loc == "-" or loc == "--":
            return None, None, None
            
        if loc[0].isdigit():
            # Across: Number then Letter
            num = ""
            for char in loc:
                if char.isdigit(): num += char
                else: break
            row = int(num) - 1
            col_char = loc[len(num)]
            col = ord(col_char) - ord('a') if col_char.islower() else ord(col_char) - ord('A')
            return row, col, "across"
        elif loc[0].isalpha():
            # Down: Letter then Number
            col_char = loc[0]
            col = ord(col_char) - ord('a') if col_char.islower() else ord(col_char) - ord('A')
            row = int(loc[1:]) - 1
            return row, col, "down"
        return None, None, None

    def __getitem__(self, idx):
        file_path = self.files[idx]
        
        board_state = [['' for _ in range(15)] for _ in range(15)]
        moves = []
        
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith(">"):
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        moves.append(parts)
        
        if not moves:
            return torch.zeros(BOARD_FEATURES), torch.zeros(RACK_FEATURES), torch.zeros(1)
            
        # Pick a random valid move to predict
        target_move_idx = random.randint(0, len(moves) - 1)
        
        # 1. Replay the game up to target turn
        for i in range(target_move_idx):
            parts = moves[i]
            loc, word = parts[2], parts[3]
            
            # Extract points (to detect challenged words which lose points)
            pts_earned = int(parts[4].replace('+', '')) if '+' in parts[4] or '-' in parts[4] else 0
            if pts_earned < 0:
                continue # Ignore challenged plays that remove points
                
            row, col, direction = self._parse_coords(loc)
            if row is None or col is None: 
                continue
            
            r, c = row, col
            for char in word:
                # Placed letter (uppercase/lowercase)
                if char.isalpha():
                    board_state[r][c] = char
                # A period '.' means skipping writing because tile is already there
                
                if direction == "across":
                    c += 1
                else:
                    r += 1

        # 2. Extract features at that exact moment!
        target_parts = moves[target_move_idx]
        rack = target_parts[1]
        pts = float(target_parts[5]) # Using cumulative score or earned score as prediction heuristic
        
        board_tensor = torch.zeros(BOARD_FEATURES)
        rack_tensor = torch.zeros(RACK_FEATURES)
        
        # Map 15x15 board state to 6,075 dimension sparse tensor
        for r in range(15):
            for c in range(15):
                tile = board_state[r][c]
                if tile:
                    is_blank = tile.islower()
                    letter_idx = 26 if is_blank else (ord(tile.upper()) - ord('A'))
                    square_idx = r * 15 + c
                    board_tensor[square_idx * 27 + letter_idx] = 1.0
                    
        # Map 7 tiles on rack to 27 dimension vector
        # (e.g., ?ABEETT = 1 blank, 1 A, 1 B, 2 E, 2 T)
        for char in rack:
            if char == '?':
                rack_tensor[26] += 1.0
            elif char.isalpha():
                rack_tensor[ord(char.upper()) - ord('A')] += 1.0
                
        equity_label = torch.tensor([pts])
        
        return board_tensor, rack_tensor, equity_label

if __name__ == "__main__":
    # Test our skeleton!
    dataset = ScrabbleDataset()
    print(f"Detected {len(dataset)} games downloaded so far.")
    
    if len(dataset) > 0:
        b, r, e = dataset[0]
        print(f"Label shape: {e.shape}")
        print(f"Board tensor sum (empty right now): {b.sum()}")
