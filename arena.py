import time
import requests
import random
import os
import subprocess
from engine import ScrabbleEngine
from parser import ScrabbleDataset

# Number of games to play in the Arena
NUM_GAMES = 10 

engine = ScrabbleEngine()
dataset_parser = ScrabbleDataset()

bag_base = (
    ['E']*12 + ['A']*9 + ['I']*9 + ['O']*8 + ['N']*6 + ['R']*6 + ['T']*6 + ['L']*4 + ['S']*4 + ['U']*4 +
    ['D']*4 + ['G']*3 +
    ['B']*2 + ['C']*2 + ['M']*2 + ['P']*2 + ['F']*2 + ['H']*2 + ['V']*2 + ['W']*2 + ['Y']*2 +
    ['K']*1 + ['J']*1 + ['X']*1 + ['Q']*1 + ['Z']*1 +
    ['?']*2
)

def format_coord(r, c, d):
    row_str = str(r + 1)
    col_str = chr(c + ord('A'))
    return f"{row_str}{col_str}" if d == "across" else f"{col_str}{row_str}"

def apply_text_move_to_board(board, loc, word):
    row, col, d = dataset_parser._parse_coords(loc)
    if row is not None and col is not None:
        r, c = row, col
        for char in word:
            if char.isalpha():
                board[r][c] = char.upper()
            if d == "across": c += 1
            else: r += 1

def get_dawg_move(board, rack, bag):
    """Native Python execution, fastest for BingoDAWG."""
    d_move, eq = engine.get_best_move(board, rack, bag)
    if d_move:
        r, c, d, w, p = d_move
        return format_coord(r, c, d), w, p
    return None, None, 0

def get_macondo_move(gcg_history, rack):
    """Queries Macondo running locally via HTTP wrapper."""
    r_str = "".join(rack)
    # The Macondo wrapper chokes on '#player1' metadata, so we filter it out for the payload
    clean_history = [h for h in gcg_history if not h.startswith("#")]
    try:
        res = requests.post("http://localhost:5004/get_move", json={"gcg_history": clean_history, "rack": r_str}).json()
        raw_move = res.get("move_raw", "")
        if raw_move and "1:" in raw_move:
            parts = raw_move.strip().split()
            loc, word, pts = parts[1], parts[2], int(parts[-2])
            return loc, word, pts
    except Exception as e:
        print("MACONDO ERROR:", e)
    return None, None, 0

def get_quackle_move(gcg_history, rack, board):
    """
    Subprocess execution of Quackle's CLI.
    Note: You may need to pipe the gcg_history into a temp.gcg file and run:
    ./quackle -c temp.gcg --generate-move
    """
    # TODO: Replace with the actual Quackle headless command line call on your machine
    # Example snippet:
    # with open("temp.gcg", "w") as f: f.write("\n".join(gcg_history))
    # output = subprocess.check_output(["./quackle", "-c", "temp.gcg", "--top-move"])
    pass
    return None, None, 0

def run_arena(player1="DAWG", player2="MACONDO"):
    p1_wins, p2_wins, p1_score, p2_score = 0, 0, 0, 0
    highest_game = {"score": 0, "log": ""}

    print(f"--- {player1} VS {player2} ARENA ({NUM_GAMES} GAMES) ---")
    
    for game_idx in range(NUM_GAMES):
        board = [['' for _ in range(15)] for _ in range(15)]
        bag = bag_base[:]
        random.shuffle(bag)
        p1_rack, p2_rack, gcg_history = [], [], ["#player1 " + player1, "#player2 " + player2]
        
        def replenish(rack):
            while len(rack) < 7 and bag: rack.append(bag.pop())

        score_1, score_2 = 0, 0
        p1_turn = (game_idx % 2 == 0)

        passes = 0
        while passes < 2:
            current_rack = p1_rack if p1_turn else p2_rack
            replenish(current_rack)
            if not current_rack: break
            
            p_name = player1 if p1_turn else player2
            loc, word, pts = None, None, 0

            # Get move depending on player identity
            if p_name == "DAWG":
                loc, word, pts = get_dawg_move(board, current_rack, bag)
            elif p_name == "MACONDO":
                loc, word, pts = get_macondo_move(gcg_history, current_rack)
            elif p_name == "QUACKLE":
                loc, word, pts = get_quackle_move(gcg_history, current_rack, board)
                
            if word and loc:
                passes = 0
                r_str = "".join(current_rack)
                gcg_history.append(f">{p_name} {r_str} {loc} {word} +{pts}")
                
                # Apply move
                apply_text_move_to_board(board, loc, word)
                if p1_turn: score_1 += pts
                else: score_2 += pts
                
                print(f"[Turn {passes+1}] {p_name.ljust(7)} played {word.ljust(7)} at {loc.ljust(3)} for {str(pts).rjust(2)} pts")
                
                # Deplete rack
                for char in word:
                    char_u = char.upper()
                    if char_u in current_rack: current_rack.remove(char_u)
                    elif '?' in current_rack: current_rack.remove('?')
            else:
                passes += 1
                gcg_history.append(f">{p_name} {''.join(current_rack)} - - +0")
                print(f"[Turn {passes+1}] {p_name} passed.")
                
            p1_turn = not p1_turn

        # Endgame penalties
        p1_penalty = sum(engine.letter_values.get(c, 0) for c in p1_rack)
        p2_penalty = sum(engine.letter_values.get(c, 0) for c in p2_rack)
        if len(p1_rack) == 0 and len(bag) == 0:
            score_1 += p2_penalty
            score_2 -= p2_penalty
        elif len(p2_rack) == 0 and len(bag) == 0:
            score_2 += p1_penalty
            score_1 -= p1_penalty
        else:
            score_1 -= p1_penalty
            score_2 -= p2_penalty

        if score_1 > score_2: p1_wins += 1
        elif score_2 > score_1: p2_wins += 1
            
        print(f"Game {game_idx+1}/{NUM_GAMES} Complete! {player1}: {score_1} - {player2}: {score_2}")
        
        if score_1 + score_2 > highest_game["score"]:
            highest_game["score"] = score_1 + score_2
            highest_game["log"] = "\n".join(gcg_history)

        p1_score += score_1
        p2_score += score_2

    report = f"""# {player1} vs {player2} Arena Insights
Date: {time.strftime('%Y-%m-%d %H:%M:%S')} | Total Games: {NUM_GAMES}

### 🏆 Match Results
- **{player1} Wins**: {p1_wins} ({round(p1_wins/NUM_GAMES*100, 1)}%)
- **{player2} Wins**: {p2_wins} ({round(p2_wins/NUM_GAMES*100, 1)}%)

### 📊 Performance
- **{player1} Avg**: {round(p1_score / max(1, NUM_GAMES), 1)} pts
- **{player2} Avg**: {round(p2_score / max(1, NUM_GAMES), 1)} pts
"""
    with open("insights_report.md", "w") as f: f.write(report)
    with open("highest_scoring_arena_game.gcg", "w") as f: f.write(highest_game["log"])
    print("Arena matches complete! Checked insights_report.md for stats.")

if __name__ == "__main__":
    # You can change match-ups here! 
    run_arena(player1="DAWG", player2="MACONDO")
