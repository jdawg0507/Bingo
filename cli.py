import argparse
import sys
from engine import ScrabbleEngine

def main():
    parser = argparse.ArgumentParser(description="BingoDAWG Headless CLI")
    parser.add_argument("--rack", type=str, required=True, help="Current tiles on rack (max 7), use ? for blanks")
    parser.add_argument("--board", type=str, help="225-character string representing the board (row by row). Use . for empty squares.")
    
    args = parser.parse_args()
    
    # Initialize 15x15 board
    board = [['' for _ in range(15)] for _ in range(15)]
    if args.board:
        if len(args.board) != 225:
            print("Error: Board string must be exactly 225 characters.", file=sys.stderr)
            sys.exit(1)
        for i in range(15):
            for j in range(15):
                char = args.board[i*15 + j]
                board[i][j] = '' if char in ['.', '-', ' '] else char.upper()
                
    rack = list(args.rack.upper())
    
    # Infer remaining bag composition
    bag = (
        ['E']*12 + ['A']*9 + ['I']*9 + ['O']*8 + ['N']*6 + ['R']*6 + ['T']*6 + ['L']*4 + ['S']*4 + ['U']*4 +
        ['D']*4 + ['G']*3 +
        ['B']*2 + ['C']*2 + ['M']*2 + ['P']*2 + ['F']*2 + ['H']*2 + ['V']*2 + ['W']*2 + ['Y']*2 +
        ['K']*1 + ['J']*1 + ['X']*1 + ['Q']*1 + ['Z']*1 +
        ['?']*2
    )
    for r in rack:
        if r in bag: bag.remove(r)
    for row in board:
        for val in row:
            if val and val in bag: bag.remove(val)
            
    # Initialize Engine and evaluate
    engine = ScrabbleEngine()
    best_move, best_eq = engine.get_best_move(board, rack, bag)
    
    if best_move:
        row, col, d, word, pts = best_move
        
        # Format coordinate (e.g., 8H or H8)
        r_str = str(row + 1)
        c_str = chr(col + ord('A'))
        loc = f"{r_str}{c_str}" if d == "across" else f"{c_str}{r_str}"
        
        # Standard Output exactly what standard benchmarking protocols expect
        print(f"MOVE: {loc} {word} {pts}")
    else:
        # Pass
        print("MOVE: - - 0")

if __name__ == '__main__':
    main()
