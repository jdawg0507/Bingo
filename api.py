from flask import Flask, request, jsonify
from flask_cors import CORS
from engine import ScrabbleEngine
import random
import os
import parser

app = Flask(__name__)
CORS(app)

print("Initializing Scrabble Engine and NNUE Backend...")
engine = ScrabbleEngine()

# Exact standard English Scrabble tile distribution (100 tiles total)
bag_base = (
    ['E']*12 + ['A']*9 + ['I']*9 + ['O']*8 + ['N']*6 + ['R']*6 + ['T']*6 + ['L']*4 + ['S']*4 + ['U']*4 +
    ['D']*4 + ['G']*3 +
    ['B']*2 + ['C']*2 + ['M']*2 + ['P']*2 + ['F']*2 + ['H']*2 + ['V']*2 + ['W']*2 + ['Y']*2 +
    ['K']*1 + ['J']*1 + ['X']*1 + ['Q']*1 + ['Z']*1 +
    ['?']*2
)

import requests
# Global memory since ui is stateless
sim_gcg_history = []

@app.route("/simulate_move", methods=["POST"])
def simulate_move():
    global sim_gcg_history
    data = request.json or {}
    
    board = data.get("board")
    if not board:
        board = [['' for _ in range(15)] for _ in range(15)]
        
    bag = data.get("bag")
    if bag is None:
        bag = bag_base[:]
        random.shuffle(bag)
        sim_gcg_history = []
        
    dawg_rack = data.get("dawg_rack")
    if dawg_rack is None: dawg_rack = []
    
    quackle_rack = data.get("quackle_rack")
    if quackle_rack is None: quackle_rack = []

    def replenish(rack):
        while len(rack) < 7 and bag:
            rack.append(bag.pop())

    def extract_played(board_state, r, c, d, word):
        used = []
        for char in word:
            if board_state[r][c] == '': used.append(char)
            if d == "across": c += 1
            else: r += 1
        return used

    def format_coord(r, c, d):
        row_str = str(r + 1)
        col_str = chr(c + ord('A'))
        if d == "across": return f"{row_str}{col_str}"
        else: return f"{col_str}{row_str}"

    # Fill racks manually
    replenish(dawg_rack)
    dawg_move, equity = engine.get_best_move(board, dawg_rack, bag)
    
    if dawg_move:
        r, c, d, w, p = dawg_move
        used_tiles = extract_played(board, r, c, d, w)
        for x in used_tiles:
            if x in dawg_rack: dawg_rack.remove(x)
            elif '?' in dawg_rack: dawg_rack.remove('?')
        board = engine._apply_move_to_board(board, dawg_move)
        sim_gcg_history.append(f">DAWG {''.join(dawg_rack)} {format_coord(r, c, d)} {w} +{p}")
    else:
        sim_gcg_history.append(f">DAWG {''.join(dawg_rack)} - - +0")
        
    replenish(quackle_rack)
    
    opponent = data.get("opponent", "macondo")
    quackle_move = None
    q_equity = 0.0

    if opponent == "quackle":
        # Original mock logic (BingoDAWG playing against itself)
        quackle_move, q_equity = engine.get_best_move(board, quackle_rack, bag)
        if quackle_move:
            used_q = extract_played(board, quackle_move[0], quackle_move[1], quackle_move[2], quackle_move[3])
            for x in used_q:
                if x in quackle_rack: quackle_rack.remove(x)
                elif '?' in quackle_rack: quackle_rack.remove('?')
            board = engine._apply_move_to_board(board, quackle_move)
    else:
        # Send to Macondo Wrapper on 5004
        r_str = "".join(quackle_rack)
        try:
            res = requests.post("http://localhost:5004/get_move", json={
                "gcg_history": sim_gcg_history,
                "rack": r_str
            }).json()
            raw_move = res.get("move_raw", "")
            if raw_move and "1:" in raw_move:
                parts = raw_move.strip().split()
                loc = parts[1]
                word = parts[2]
                pts = int(parts[-2])
                
                # Reconstruct row/col from standard '8D', 'H8'
                col = ord(loc[-1].upper()) - ord('A') if loc[-1].isalpha() else ord(loc[0].upper()) - ord('A')
                row = int(''.join(filter(str.isdigit, loc))) - 1
                direction = "across" if loc[0].isdigit() else "down"
                
                quackle_move = (row, col, direction, word, pts)
                
                used_q = extract_played(board, row, col, direction, word)
                for x in used_q:
                    char_u = x.upper()
                    if char_u in quackle_rack: quackle_rack.remove(char_u)
                    elif '?' in quackle_rack: quackle_rack.remove('?')
                    
                sim_gcg_history.append(f">MACONDO {r_str} {loc} {word} +{pts}")
                
                for i, char in enumerate(word):
                    if char.isalpha():
                        board[row if direction == "across" else row+i][col+i if direction == "across" else col] = char.upper()
            else:
                sim_gcg_history.append(f">MACONDO {r_str} - - +0")
        except Exception as e:
            print("Macondo Move Error:", e)
            sim_gcg_history.append(f">MACONDO {r_str} - - +0")

    response = {
        "bag": bag,
        "dawg_rack": dawg_rack,
        "quackle_rack": quackle_rack,
        "dawg_move": None,
        "quackle_move": None
    }
    
    if dawg_move:
        r, c, d, w, p = dawg_move
        response["dawg_move"] = {"row": r, "col": c, "direction": d, "word": w, "points": p, "equity": equity}
        
    if quackle_move:
        qr, qc, qd, qw, qp = quackle_move
        response["quackle_move"] = {"row": qr, "col": qc, "direction": qd, "word": qw, "points": qp, "equity": q_equity}

    return jsonify(response)

@app.route("/batch_simulate", methods=["POST"])
def batch_simulate():
    data = request.json or {}
    num_games = data.get("games", 10)
    
    highest_scoring_move = {"points": 0, "word": "", "player": ""}
    highest_scoring_game = {"total": 0, "dawg": 0, "quackle": 0}
    hindsight_errors = 0
    dawg_wins = 0
    best_gcg_buffer = ""

    def format_coord(r, c, d):
        row_str = str(r + 1)
        col_str = chr(c + ord('A'))
        if d == "across": return f"{row_str}{col_str}"
        else: return f"{col_str}{row_str}"

    for g in range(num_games):
        board = [['' for _ in range(15)] for _ in range(15)]
        bag = bag_base[:]
        random.shuffle(bag)
        dawg_rack = []
        quackle_rack = []
        gcg_lines = ["# BingoDAWG Headless Batch Simulation\n", "#Player1 DAWG\n", "#Player2 QUACKLE\n"]
        
        def replenish(r):
            while len(r) < 7 and bag: r.append(bag.pop())

        def ext(b, rw, c, d, w):
            u = []
            for ch in w:
                if b[rw][c] == '': u.append(ch)
                if d == "across": c += 1
                else: rw += 1
            return u

        dawg_score = 0
        quack_score = 0
        avg_equity = 0
        
        while True:
            replenish(dawg_rack)
            d_move, eq = engine.get_best_move(board, dawg_rack, bag)
            if d_move:
                r, c, d, w, p = d_move
                loc = format_coord(r, c, d)
                r_str = "".join(dawg_rack)
                gcg_lines.append(f">DAWG {r_str} {loc} {w} +{p}\n")
                gcg_lines.append(f"# [NNUE EVAL]: Projected Equity {eq:.2f}\n")
                
                for x in ext(board, r, c, d, w):
                    if x in dawg_rack: dawg_rack.remove(x)
                    elif '?' in dawg_rack: dawg_rack.remove('?')
                board = engine._apply_move_to_board(board, d_move)
                dawg_score += p
                avg_equity += eq
                if p > highest_scoring_move["points"]:
                    highest_scoring_move = {"points": p, "word": w, "player": "DAWG"}
                    
            replenish(quackle_rack)
            q_move, qeq = engine.get_best_move(board, quackle_rack, bag)
            if q_move:
                qr, qc, qd, qw, qp = q_move
                qloc = format_coord(qr, qc, qd)
                qr_str = "".join(quackle_rack)
                gcg_lines.append(f">QUACKLE {qr_str} {qloc} {qw} +{qp}\n")
                gcg_lines.append(f"# [NNUE EVAL of QUACKLE]: Projected Equity {qeq:.2f}\n")
                
                for x in ext(board, qr, qc, qd, qw):
                    if x in quackle_rack: quackle_rack.remove(x)
                    elif '?' in quackle_rack: quackle_rack.remove('?')
                board = engine._apply_move_to_board(board, q_move)
                quack_score += qp
                if qp > highest_scoring_move["points"]:
                    highest_scoring_move = {"points": qp, "word": qw, "player": "QUACKLE"}
                    
            if not d_move and not q_move:
                break

        dawg_penalty = sum(engine.letter_values.get(c, 0) for c in dawg_rack)
        quack_penalty = sum(engine.letter_values.get(c, 0) for c in quackle_rack)
        if len(dawg_rack) == 0 and len(bag) == 0:
            dawg_score += quack_penalty; quack_score -= quack_penalty
        elif len(quackle_rack) == 0 and len(bag) == 0:
            quack_score += dawg_penalty; dawg_score -= dawg_penalty
        else:
            dawg_score -= dawg_penalty; quack_score -= quack_penalty

        if dawg_score > quack_score: dawg_wins += 1
        elif avg_equity > 0 and dawg_score < quack_score:
            hindsight_errors += 1

        if (dawg_score + quack_score) > highest_scoring_game["total"]:
            highest_scoring_game = {"total": dawg_score + quack_score, "dawg": dawg_score, "quackle": quack_score}
            best_gcg_buffer = "".join(gcg_lines)
            
    # Write the best game out to disk
    with open("highest_scoring_game.gcg", "w") as f:
        f.write(best_gcg_buffer)

    return jsonify({
        "games_played": num_games,
        "dawg_wins": dawg_wins,
        "highest_game": highest_scoring_game,
        "highest_move": highest_scoring_move,
        "hindsight_overestimations": hindsight_errors
    })

@app.route("/analyze_nigel", methods=["POST"])
def analyze_nigel_endpoint():
    data_dir = "dataset/gcg_files"
    if not os.path.exists(data_dir):
        return jsonify({"total_moves": 0, "match_rate": 0, "avg_deviation": 0})
        
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".gcg")]
    
    nigel_files = []
    for f_path in files:
        with open(f_path, "r", encoding="utf-8", errors="ignore") as f:
            header_lines = []
            for _ in range(5):
                try: header_lines.append(next(f))
                except StopIteration: break
            header = "".join(header_lines)
            if "Nigel_Richards" in header:
                nigel_files.append(f_path)
    
    total_moves = 0
    matches = 0
    cumulative_deviation = 0.0
    
    # Try 50 games for fast UI benchmarking
    nigel_files = nigel_files[:50]
    eval_dataset = parser.ScrabbleDataset()
    
    for f_path in nigel_files:
        board_state = [['' for _ in range(15)] for _ in range(15)]
        nigel_id = ""
        with open(f_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            for l in lines:
                if "#player1" in l and "Nigel_Richards" in l: nigel_id = ">" + l.split()[1]
                elif "#player2" in l and "Nigel_Richards" in l: nigel_id = ">" + l.split()[1]
        
        for line in lines:
            if line.startswith(nigel_id):
                parts = line.strip().split()
                if len(parts) >= 6:
                    rack = parts[1].replace('-', '').replace('+', '')
                    loc = parts[2]
                    word = parts[3]
                    try:
                        n_pts = float(parts[4].replace('+', '').replace('-', '')) 
                    except:
                        n_pts = 0
                    
                    row, col, d = eval_dataset._parse_coords(loc)
                    if row is not None and col is not None:
                        best_move, best_eq = engine.get_best_move(board_state, list(rack), [])
                        if best_move:
                            engine_word = best_move[3]
                            w_clean = word.replace('.', '').upper()
                            ew_clean = engine_word.replace('.', '').upper()
                            
                            if w_clean == ew_clean or abs(best_eq - n_pts) <= 1.0:
                                # High alignment match if same word or EV identically scores
                                matches += 1
                            else:
                                if best_eq > n_pts:
                                    cumulative_deviation += (best_eq - n_pts)
                            
                            total_moves += 1

            if line.startswith(">"):
                parts = line.strip().split()
                if len(parts) >= 5:
                    loc = parts[2]
                    word = parts[3]
                    row, col, d = eval_dataset._parse_coords(loc)
                    if row is not None and col is not None:
                        r, c = row, col
                        for char in word:
                            if char.isalpha():
                                board_state[r][c] = char.upper()
                            if d == "across": c += 1
                            else: r += 1

    match_rate = round((matches / max(1, total_moves)) * 100, 2)
    avg_dev = round(cumulative_deviation / max(1, total_moves), 2)
    
    return jsonify({
        "total_moves": total_moves,
        "match_rate": match_rate,
        "avg_deviation": avg_dev
    })

if __name__ == "__main__":
    app.run(port=5002, debug=False, use_reloader=False)
