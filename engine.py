import torch
import torch.nn.functional as F
import random
import copy
from model import ScrabbleNNUE

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_word = True

class ScrabbleEngine:
    def __init__(self, model_path=None, dictionary_path="sowpods.txt"):
        self.device = torch.device("cpu")
        self.model = ScrabbleNNUE().to(self.device)
        self.model.eval()

        self.trie = Trie()
        self.lexicon = set()
        print(f"Loading '{dictionary_path}' Lexicon into Prefix Trie...")
        try:
            with open(dictionary_path, 'r') as f:
                for line in f:
                    w = line.strip().upper()
                    if w.isalpha():
                        self.lexicon.add(w)
                        self.trie.insert(w)
        except Exception as e:
            print("Dict load error:", e)
            for w in ["HELLO", "WORLD", "JAKE", "QUACK"]: self.trie.insert(w)
        
        self.letter_values = {
            'A': 1, 'B': 3, 'C': 3, 'D': 2, 'E': 1, 'F': 4, 'G': 2, 'H': 4, 'I': 1, 'J': 8, 'K': 5, 'L': 1, 'M': 3,
            'N': 1, 'O': 1, 'P': 3, 'Q': 10, 'R': 1, 'S': 1, 'T': 1, 'U': 1, 'V': 4, 'W': 4, 'X': 8, 'Y': 4, 'Z': 10, '?': 0
        }

    def get_best_move(self, board_state, rack, bag_base):
        legal_moves = self.generate_legal_moves(board_state, rack)
        if not legal_moves:
            return None, 0.0
            
        # Score standard points first
        legal_moves.sort(key=lambda x: x[4], reverse=True)
        top_candidates = legal_moves[:20]
        
        best_move = None
        best_equity = -9999.0
        
        for move in top_candidates:
            eq = self.evaluate_move_monte_carlo(board_state, move, rack, bag_base, iterations=1)
            if eq > best_equity:
                best_equity = eq
                best_move = move
                
        return best_move, round(best_equity, 2)

    def _to_tensors(self, board, rack):
        # Neural Network expects flat 6075 tensor for mathematical speed!
        b = torch.zeros((1, 6075), dtype=torch.float32)
        r = torch.zeros((1, 27), dtype=torch.float32)
        return b, r

    def _calculate_score(self, board, start_r, start_c, direction, word):
        tw = [(0,0), (0,7), (0,14), (7,0), (7,14), (14,0), (14,7), (14,14)]
        dw = [(1,1), (2,2), (3,3), (4,4), (10,10), (11,11), (12,12), (13,13), (1,13), (2,12), (3,11), (4,10), (10,4), (11,3), (12,2), (13,1), (7,7)]
        tl = [(1,5), (1,9), (5,1), (5,5), (5,9), (5,13), (9,1), (9,5), (9,9), (9,13), (13,5), (13,9)]
        dl = [(0,3), (0,11), (2,6), (2,8), (3,0), (3,7), (3,14), (6,2), (6,6), (6,8), (6,12), (7,3), (7,11), (8,2), (8,6), (8,8), (8,12), (11,0), (11,7), (11,14), (12,6), (12,8), (14,3), (14,11)]
        
        score = 0
        word_mult = 1
        tiles_placed = 0
        
        for i, char in enumerate(word):
            r = start_r if direction == "across" else start_r + i
            c = start_c + i if direction == "across" else start_c
            
            letter_val = self.letter_values.get(char, 0)
            
            if board[r][c] == '':
                tiles_placed += 1
                pos = (r, c)
                if pos in tw: word_mult *= 3
                elif pos in dw: word_mult *= 2
                
                if pos in tl: score += (letter_val * 3)
                elif pos in dl: score += (letter_val * 2)
                else: score += letter_val
            else:
                score += letter_val
                
        score *= word_mult
        if tiles_placed == 7: score += 50
        return score

    def generate_legal_moves(self, board, rack):
        moves = []
        rack_chars = [c.upper() for c in rack]
        
        is_empty = all(board[r][c] == '' for r in range(15) for c in range(15))
        if is_empty:
            words = self._anagrams_from_rack(rack_chars, max_len=7)
            for w in set(words):
                if len(w) > 1:
                    start_c = 7 - (len(w)//2)
                    pts = self._calculate_score(board, 7, start_c, "across", w)
                    moves.append((7, start_c, "across", w, pts))
            return moves
            
        for r in range(15):
            for c in range(15):
                if board[r][c] != '':
                    anchor_letter = board[r][c].upper()[0]
                    words = self._anagrams_from_rack(rack_chars + [anchor_letter], max_len=len(rack_chars)+1)
                    
                    for w in words:
                        if len(w) <= 1: continue
                        if anchor_letter in w:
                            idx = w.find(anchor_letter)
                            
                            start_c = c - idx
                            if 0 <= start_c and start_c + len(w) <= 15:
                                if self._fits_horizontally(board, r, start_c, w):
                                    pts = self._calculate_score(board, r, start_c, "across", w)
                                    moves.append((r, start_c, "across", w, pts))
                                    
                            start_r = r - idx
                            if 0 <= start_r and start_r + len(w) <= 15:
                                if self._fits_vertically(board, start_r, c, w):
                                    pts = self._calculate_score(board, start_r, c, "down", w)
                                    moves.append((start_r, c, "down", w, pts))
                                    
        return list(set(moves))

    def _anagrams_from_rack(self, pool, max_len):
        valid = []
        def dfs(node, path, available):
            if node.is_word:
                valid.append("".join(path))
            if len(path) == max_len:
                return
            
            seen = set()
            for idx, c in enumerate(available):
                if c in seen: continue
                seen.add(c)
                
                if c == '?':
                    for alpha in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                        if alpha in node.children:
                            dfs(node.children[alpha], path + [alpha], available[:idx] + available[idx+1:])
                else:
                    if c in node.children:
                        dfs(node.children[c], path + [c], available[:idx] + available[idx+1:])
        
        dfs(self.trie.root, [], pool)
        return valid

    def _fits_horizontally(self, board, r, start_c, word):
        if start_c > 0 and board[r][start_c-1] != '': return False
        if start_c + len(word) < 15 and board[r][start_c+len(word)] != '': return False
        
        touches_anchors = False
        tiles_placed = 0
        for i, char in enumerate(word):
            c = start_c + i
            target = board[r][c].upper()
            if target != '':
                if target[0] != char:
                    return False
                touches_anchors = True
            else:
                if r > 0 and board[r-1][c] != '': return False
                if r < 14 and board[r+1][c] != '': return False
                tiles_placed += 1
        return touches_anchors and tiles_placed > 0

    def _fits_vertically(self, board, start_r, c, word):
        if start_r > 0 and board[start_r-1][c] != '': return False
        if start_r + len(word) < 15 and board[start_r+len(word)][c] != '': return False
        
        touches_anchors = False
        tiles_placed = 0
        for i, char in enumerate(word):
            _r = start_r + i
            target = board[_r][c].upper()
            if target != '':
                if target[0] != char:
                    return False
                touches_anchors = True
            else:
                if c > 0 and board[_r][c-1] != '': return False
                if c < 14 and board[_r][c+1] != '': return False
                tiles_placed += 1
        return touches_anchors and tiles_placed > 0

    def evaluate_move_monte_carlo(self, board_state, move, our_rack, bag, iterations=1):
        snapshot_board = self._apply_move_to_board(board_state, move)
        total_equity = 0.0
        for _ in range(iterations):
            b_tensor, r_tensor = self._to_tensors(snapshot_board, [])
            with torch.no_grad():
                nn_opp_equity = self.model(b_tensor, r_tensor).item()
            total_equity += (move[4] - nn_opp_equity)
        return total_equity / iterations

    def _apply_move_to_board(self, board_state, move):
        b = copy.deepcopy(board_state)
        r, c, d, w, _ = move
        for char in w:
            b[r][c] = char
            if d == "across": c += 1
            else: r += 1
        return b
