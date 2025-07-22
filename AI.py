# AI.py

import time
from GameLogic import *
import random
import threading
from collections import namedtuple

# --- Versioning ---
# v2.1 (Bugfix & Enhancement)
# - Implemented "Try TT Move First" optimization for significant search speedup.
# - LMR logic is now more robust, using is_move_tactical to avoid reducing special moves.

# --- Global Zobrist Hashing ---
def initialize_zobrist_table():
    """Creates a table of random numbers for each piece/square combination."""
    random.seed(42)
    table = {
        (r, c, piece_type, color): random.getrandbits(64)
        for r in range(ROWS) for c in range(COLS)
        for piece_type in [Pawn, Knight, Bishop, Rook, Queen, King, None]
        for color in ['white', 'black', None]
    }
    table['turn'] = random.getrandbits(64)
    return table

ZOBRIST_TABLE = initialize_zobrist_table()

def board_hash(board, turn):
    """Calculates the Zobrist hash for a given board state and turn."""
    hash_val = 0
    for r in range(ROWS):
        for c in range(COLS):
            piece = board[r][c]
            key = (r, c, type(piece) if piece else None, piece.color if piece else None)
            hash_val ^= ZOBRIST_TABLE.get(key, 0)
    if turn == 'black':
        hash_val ^= ZOBRIST_TABLE['turn']
    return hash_val

# --- Transposition Table Setup ---
TTEntry = namedtuple('TTEntry', ['score', 'depth', 'flag', 'best_move'])
TT_FLAG_EXACT = 0
TT_FLAG_LOWERBOUND = 1
TT_FLAG_UPPERBOUND = 2

class SearchCancelledException(Exception):
    """Exception raised when the AI search is cancelled by the UI."""
    pass

class ChessBot:
    search_depth = 3
    
    # --- Search Constants ---
    MATE_SCORE = 1000000
    DRAW_SCORE = 0
    QSEARCH_CHECKS_MAX_DEPTH = 4
    CAPTURE_SCORE_BONUS = 10000
    PROMOTION_SCORE_BONUS = 9000
    CHECK_SCORE_BONUS = 5000
    HASH_MOVE_SCORE_BONUS = 100000

    def __init__(self, board, color, app, cancellation_event, bot_name="AI Bot"):
        self.board = board
        self.color = color
        self.opponent_color = 'black' if color == 'white' else 'white'
        self.app = app
        self.cancellation_event = cancellation_event
        self.bot_name = bot_name
        self.tt = {}
        self.nodes_searched = 0
        self.MAX_PLY_KILLERS = 30
        self.killer_moves = [[None, None] for _ in range(self.MAX_PLY_KILLERS)]
        
        self.history_heuristic_table = [[[0 for _ in range(ROWS * COLS)] for _ in range(ROWS * COLS)] for _ in range(2)]

        # --- LMR Constants (Defined as INSTANCE attributes for robustness) ---
        self.LMR_DEPTH_THRESHOLD = 3
        self.LMR_MOVE_COUNT_THRESHOLD = 4
        self.LMR_REDUCTION = 1
    
    def evaluate_board(self, board, current_turn):
        """
        Evaluates the board based on Material and Piece-Square Tables (PSTs).
        This provides a strong positional understanding.
        """
        perspective_multiplier = 1 if current_turn == self.color else -1
        score_relative_to_ai = 0
        
        our_king_pos = find_king_pos(board, self.color)
        enemy_king_pos = find_king_pos(board, self.opponent_color)
        if not enemy_king_pos: return self.MATE_SCORE * perspective_multiplier
        if not our_king_pos: return -self.MATE_SCORE * perspective_multiplier

        total_material = 0
        for r in range(ROWS):
            for c in range(COLS):
                p = board[r][c]
                if p and not isinstance(p, King):
                    total_material += PIECE_VALUES.get(type(p), 0)
        is_endgame = total_material < ENDGAME_MATERIAL_THRESHOLD

        for r_eval in range(ROWS):
            for c_eval in range(COLS):
                piece_eval = board[r_eval][c_eval]
                if not piece_eval:
                    continue
                
                value = PIECE_VALUES.get(type(piece_eval), 0)

                pst = None
                if isinstance(piece_eval, King):
                    pst = PIECE_SQUARE_TABLES['king_endgame'] if is_endgame else PIECE_SQUARE_TABLES['king_midgame']
                else:
                    pst = PIECE_SQUARE_TABLES.get(type(piece_eval))
                
                if pst:
                    pst_score = pst[r_eval][c_eval] if piece_eval.color == 'white' else pst[ROWS - 1 - r_eval][c_eval]
                    value += pst_score

                if piece_eval.color == self.color:
                    score_relative_to_ai += value
                else:
                    score_relative_to_ai -= value
            
        return int(score_relative_to_ai * perspective_multiplier)

    def is_move_a_check(self, board, move, moving_player_color):
        sim_board = self.simulate_move(board, move[0], move[1])
        opponent_color = 'black' if moving_player_color == 'white' else 'white'
        return is_in_check(sim_board, opponent_color)

    def evaluate_move(self, board, move, moving_player_color):
        start, end = move
        piece = board[start[0]][start[1]]
        target = board[end[0]][end[1]]
        score = 0
        if target:
            score = self.CAPTURE_SCORE_BONUS + (PIECE_VALUES.get(type(target), 0) - PIECE_VALUES.get(type(piece), 0))
        if isinstance(piece, Pawn) and (end[0] == 0 or end[0] == ROWS - 1):
            score += self.PROMOTION_SCORE_BONUS
        if self.is_move_a_check(board, move, moving_player_color):
            score += self.CHECK_SCORE_BONUS
        return score

    def order_moves(self, board, moves, moving_player_color, ply_searched=0, hash_move=None):
        move_scores = {}
        color_index = 0 if moving_player_color == 'white' else 1

        for move in moves:
            # Base score from captures, promotions, checks
            score = self.evaluate_move(board, move, moving_player_color)
            
            # Add history and killer move scores for non-captures
            if board[move[1][0]][move[1][1]] is None:
                # --- HISTORY HEURISTIC USAGE ---
                from_sq = move[0][0] * COLS + move[0][1]
                to_sq = move[1][0] * COLS + move[1][1]
                score += self.history_heuristic_table[color_index][from_sq][to_sq]

                # Killer moves
                if ply_searched < self.MAX_PLY_KILLERS:
                    killers = self.killer_moves[ply_searched]
                    if move == killers[0]:
                        score += 50000
                    elif move == killers[1]:
                        score += 40000
            
            move_scores[move] = score

        # Always prioritize the PV-move from the transposition table
        if hash_move and hash_move in move_scores:
            move_scores[hash_move] += self.HASH_MOVE_SCORE_BONUS

        return sorted(moves, key=lambda m: move_scores.get(m, 0), reverse=True)

    def qsearch(self, board, alpha, beta, turn_multiplier, depth):
        if self.cancellation_event.is_set(): raise SearchCancelledException()
        self.nodes_searched += 1
        current_turn = self.color if turn_multiplier == 1 else self.opponent_color
        stand_pat = self.evaluate_board(board, current_turn)
        if stand_pat >= beta: return beta
        alpha = max(alpha, stand_pat)
        
        all_moves = generate_pseudo_legal_moves(board, current_turn)
        tactical_moves = [m for m in all_moves if is_move_tactical(board, m, is_qsearch_check=(depth > 0))]
        
        for move in self.order_moves(board, tactical_moves, current_turn):
            piece = board[move[0][0]][move[0][1]]
            target = board[move[1][0]][move[1][1]]
            is_promo = isinstance(piece, Pawn) and (move[1][0] == 0 or move[1][0] == ROWS - 1)
            next_depth = depth if (target or is_promo) else depth - 1
            child_board = self.simulate_move(board, move[0], move[1])
            score = -self.qsearch(child_board, -beta, -alpha, -turn_multiplier, next_depth)
            if score >= beta: return beta
            alpha = max(alpha, score)
        return alpha

    def negamax(self, board, depth, alpha, beta, turn_multiplier, ply):
        # --- 1. Initial Checks & Setup ---
        if self.cancellation_event.is_set(): raise SearchCancelledException()
        self.nodes_searched += 1
        current_turn = self.color if turn_multiplier == 1 else self.opponent_color
        original_alpha = alpha
        hash_val = board_hash(board, current_turn)

        # Check for draw by repetition
        if ply > 0 and self.app.position_counts.get(hash_val, 0) >= 2:
            return self.DRAW_SCORE

        # --- 2. Transposition Table Lookup ---
        hash_move = None
        tt_entry = self.tt.get(hash_val)
        if tt_entry and tt_entry.depth >= depth:
            if tt_entry.flag == TT_FLAG_EXACT: return tt_entry.score
            elif tt_entry.flag == TT_FLAG_LOWERBOUND: alpha = max(alpha, tt_entry.score)
            elif tt_entry.flag == TT_FLAG_UPPERBOUND: beta = min(beta, tt_entry.score)
            if alpha >= beta: return tt_entry.score
            hash_move = tt_entry.best_move

        # Check Extension: Increase depth if in check
        is_in_check_now = is_in_check(board, current_turn)
        if is_in_check_now: depth += 1
        
        # Base Case: Call Quiescence Search at leaf nodes
        if depth <= 0:
            return self.qsearch(board, alpha, beta, turn_multiplier, self.QSEARCH_CHECKS_MAX_DEPTH)

        # --- 3. "Try TT Move First" Optimization ---
        if hash_move:
            child_board_hash = self.simulate_move(board, hash_move[0], hash_move[1])
            if not is_in_check(child_board_hash, current_turn):
                child_turn = self.opponent_color if turn_multiplier == 1 else self.color
                child_hash = board_hash(child_board_hash, child_turn)
                self.app.position_counts[child_hash] = self.app.position_counts.get(child_hash, 0) + 1
                score = -self.negamax(child_board_hash, depth - 1, -beta, -alpha, -turn_multiplier, ply + 1)
                self.app.position_counts[child_hash] -= 1
                
                # If TT move causes a cutoff, we are done with this node
                if score >= beta:
                    self.tt[hash_val] = TTEntry(beta, depth, TT_FLAG_LOWERBOUND, hash_move)
                    return beta
                if score > alpha:
                    alpha = score

        # --- 4. Main Search Loop ---
        moves = generate_pseudo_legal_moves(board, current_turn)
        ordered_moves = self.order_moves(board, moves, current_turn, ply, hash_move=hash_move)
        legal_moves_found = 0
        best_move_for_node = hash_move  # Start with hash_move as a candidate
        
        for i, move in enumerate(ordered_moves):
            # Skip the hash_move if we already searched it above
            if move == hash_move:
                continue

            child_board = self.simulate_move(board, move[0], move[1])
            if is_in_check(child_board, current_turn):
                continue
            
            legal_moves_found += 1

            # --- 4a. Late Move Reductions (LMR) ---
            reduction = 0
            if (depth >= self.LMR_DEPTH_THRESHOLD and
                    i >= self.LMR_MOVE_COUNT_THRESHOLD and
                    not is_in_check_now and
                    not is_move_tactical(board, move)):
                reduction = self.LMR_REDUCTION
            
            # Recursive call with potential reduction
            child_turn = self.opponent_color if turn_multiplier == 1 else self.color
            child_hash = board_hash(child_board, child_turn)
            self.app.position_counts[child_hash] = self.app.position_counts.get(child_hash, 0) + 1
            
            score = -self.negamax(child_board, depth - 1 - reduction, -beta, -alpha, -turn_multiplier, ply + 1)

            # Re-search if a reduced move was surprisingly good
            if reduction > 0 and score > alpha:
                score = -self.negamax(child_board, depth - 1, -beta, -alpha, -turn_multiplier, ply + 1)
            
            self.app.position_counts[child_hash] -= 1
            
            # --- 4b. Update Alpha/Beta and Check for Cutoff ---
            if score > alpha:
                alpha = score
                best_move_for_node = move
            
            if alpha >= beta:
                is_capture = board[move[1][0]][move[1][1]] is not None
                if not is_capture:
                    color_index = 0 if current_turn == 'white' else 1
                    from_sq = move[0][0] * COLS + move[0][1]
                    to_sq = move[1][0] * COLS + move[1][1]
                    self.history_heuristic_table[color_index][from_sq][to_sq] += depth * depth
                
                    if ply < self.MAX_PLY_KILLERS and self.killer_moves[ply][0] != move:
                        self.killer_moves[ply][1] = self.killer_moves[ply][0]
                        self.killer_moves[ply][0] = move
                self.tt[hash_val] = TTEntry(beta, depth, TT_FLAG_LOWERBOUND, move)
                return beta
        
        # --- 5. Finalization ---
        # Account for hash move if it was the only legal one
        if hash_move and legal_moves_found == 0:
             legal_moves_found = 1
        
        # Check for checkmate or stalemate
        if legal_moves_found == 0:
            return -self.MATE_SCORE + ply if is_in_check_now else self.DRAW_SCORE
            
        # Store final result in Transposition Table
        flag = TT_FLAG_UPPERBOUND if alpha <= original_alpha else TT_FLAG_EXACT
        self.tt[hash_val] = TTEntry(alpha, depth, flag, best_move_for_node)
        return alpha

    def make_move(self):
        try:
            best_move_found = None
            self.killer_moves = [[None, None] for _ in range(self.MAX_PLY_KILLERS)]
            self.tt.clear()
            
            # --- HISTORY HEURISTIC DECAY ---
            # "Age" the history scores by halving them before a new search begins.
            for i in range(2):
                for j in range(ROWS * COLS):
                    for k in range(ROWS * COLS):
                        self.history_heuristic_table[i][j][k] //= 2

            for current_depth in range(1, self.search_depth + 1):
                iter_start_time = time.time()
                if self.cancellation_event.is_set(): raise SearchCancelledException()
                self.nodes_searched = 0
                
                root_moves = generate_pseudo_legal_moves(self.board, self.color)
                if not root_moves: return False
                
                root_hash = board_hash(self.board, self.color)
                hash_move = self.tt.get(root_hash, None)
                if hash_move: hash_move = hash_move.best_move

                ordered_root_moves = self.order_moves(self.board, root_moves, self.color, 0, hash_move=hash_move)
                best_score = -float('inf')
                alpha, beta = -float('inf'), float('inf')

                for i, move in enumerate(ordered_root_moves):
                    if self.cancellation_event.is_set(): raise SearchCancelledException()
                    child_board = self.simulate_move(self.board, move[0], move[1])
                    if is_in_check(child_board, self.color):
                        continue

                    child_hash = board_hash(child_board, self.opponent_color)
                    self.app.position_counts[child_hash] = self.app.position_counts.get(child_hash, 0) + 1
                    score = -self.negamax(child_board, current_depth - 1, -beta, -alpha, -1, 1)
                    self.app.position_counts[child_hash] -= 1

                    if score > best_score:
                        best_score = score
                        best_move_found = move
                    alpha = max(alpha, best_score)

                if not self.cancellation_event.is_set():
                    iter_duration = time.time() - iter_start_time
                    knps = (self.nodes_searched / iter_duration / 1000) if iter_duration > 0 else 0
                    eval_for_ui = best_score * (1 if self.color == 'white' else -1)
                    
                    print(f"{self.bot_name}: Depth {current_depth}, Eval={eval_for_ui/100:.2f}, Nodes={self.nodes_searched}, KNPS={knps:.1f}")
                    
                    if self.app: self.app.master.after(0, self.app.draw_eval_bar, eval_for_ui)

            if best_move_found:
                piece = self.board[best_move_found[0][0]][best_move_found[0][1]]
                self.board = piece.move(self.board, best_move_found[0], best_move_found[1])
                check_evaporation(self.board)
                return True
            return False
            
        except SearchCancelledException:
            print(f"AI ({self.color}): Search cancelled.")
            return False

    def get_adjacent_squares(self, pos):
        r, c = pos; squares = []
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0: continue
                if 0 <= r + dr < ROWS and 0 <= c + dc < COLS:
                    squares.append((r + dr, c + dc))
        return squares

    def simulate_move(self, board, start, end):
        new_board = copy_board(board)
        piece = new_board[start[0]][start[1]]
        if piece:
            new_board = piece.move(new_board, start, end)
            check_evaporation(new_board)
        return new_board

# -----------------------------------------------------------------------------
# Piece-Square Tables (PSTs) and Material Values
# -----------------------------------------------------------------------------

PIECE_VALUES = {
    Pawn: 100, Knight: 700, Bishop: 600,
    Rook: 500, Queen: 900, King: 100000
}
ENDGAME_MATERIAL_THRESHOLD = (PIECE_VALUES[Rook] * 2 + PIECE_VALUES[Knight] * 2 + PIECE_VALUES[Bishop] * 2 + PIECE_VALUES[Queen])

pawn_pst = [
    [  0,   0,   0,   0,   0,   0,   0,   0],
    [100, 100, 100, 100, 100, 100, 100, 100],
    [ 20,  20,  30,  40,  40,  30,  20,  20],
    [ 10,  10,  20,  30,  30,  20,  10,  10],
    [  5,   5,  10,  25,  25,  10,   5,   5],
    [  0,   0,   0,  10,  10,   0,   0,   0],
    [  0,   0,   0,   0,   0,   0,   0,   0],
    [  0,   0,   0,   0,   0,   0,   0,   0]
]

knight_pst = [
    [-50, -40, -30, -30, -30, -30, -40, -50],
    [-40, -20,   0,   5,   5,   0, -20, -40],
    [-30,   5,  10,  15,  15,  10,   5, -30],
    [-30,   5,  20,  30,  30,  20,   5, -30],
    [-30,   5,  20,  30,  30,  20,   5, -30],
    [-30,   0,  10,  15,  15,  10,   0, -30],
    [-40, -20,   0,   0,   0,   0, -20, -40],
    [-50, -40, -30, -30, -30, -30, -40, -50]
]

bishop_pst = [
    [-20, -10, -10, -10, -10, -10, -10, -20],
    [-10,   5,   0,   0,   0,   0,   5, -10],
    [-10,  10,  10,  10,  10,  10,  10, -10],
    [-10,   0,  10,  10,  10,  10,   0, -10],
    [-10,   5,   5,  10,  10,   5,   5, -10],
    [-10,   0,   5,  10,  10,   5,   0, -10],
    [-10,   0,   0,   0,   0,   0,   0, -10],
    [-20, -10, -10, -10, -10, -10, -10, -20]
]

rook_pst = [
    [  0,   5,   5,  10,  10,   5,   5,   0],
    [ -5,   0,   0,   5,   5,   0,   0,  -5],
    [ -5,   0,   0,   5,   5,   0,   0,  -5],
    [ -5,   0,   0,   5,   5,   0,   0,  -5],
    [ -5,   0,   0,   5,   5,   0,   0,  -5],
    [ -5,   0,   0,   5,   5,   0,   0,  -5],
    [  5,  10,  10,  20,  20,  10,  10,   5],
    [  0,   0,   0,  10,  10,   0,   0,   0]
]

queen_pst = [
    [-20, -10, -10,  -5,  -5, -10, -10, -20],
    [-10,   0,   5,   0,   0,   0,   5, -10],
    [-10,   5,   5,   5,   5,   5,   5, -10],
    [ -5,   0,   5,   5,   5,   5,   0,  -5],
    [  0,   0,   5,   5,   5,   5,   0,  -5],
    [-10,   0,   5,   5,   5,   5,   0, -10],
    [-10,   0,   0,   0,   0,   0,   0, -10],
    [-20, -10, -10,  -5,  -5, -10, -10, -20]
]

king_midgame_pst = [
    [ 20,  30,  10,   0,   0,  10,  30,  20],
    [ 20,  20,   0,   0,   0,   0,  20,  20],
    [-10, -20, -20, -20, -20, -20, -20, -10],
    [-20, -30, -30, -40, -40, -30, -30, -20],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30]
]

king_endgame_pst = [
    [-50, -30, -30, -30, -30, -30, -30, -50],
    [-30, -30,   0,   0,   0,   0, -30, -30],
    [-30, -10,  20,  30,  30,  20, -10, -30],
    [-30, -10,  30,  40,  40,  30, -10, -30],
    [-30, -10,  30,  40,  40,  30, -10, -30],
    [-30, -10,  20,  30,  30,  20, -10, -30],
    [-30, -20, -10,   0,   0, -10, -20, -30],
    [-50, -40, -30, -20, -20, -30, -40, -50]
]

PIECE_SQUARE_TABLES = {
    Pawn: pawn_pst, 
    Knight: knight_pst, 
    Bishop: bishop_pst, 
    Rook: rook_pst, 
    Queen: queen_pst,
    'king_midgame': king_midgame_pst, 
    'king_endgame': king_endgame_pst
}