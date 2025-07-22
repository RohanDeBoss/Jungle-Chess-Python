# OpponentAI.py

import time
from GameLogic import *
import random
import threading
from collections import namedtuple

# --- Versioning ---
# v3.0.1 - Working baseline with refactor support and bugfixes.

# --- Global Zobrist Hashing ---
def initialize_zobrist_table():
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
    hash_val = 0
    for r in range(ROWS):
        for c in range(COLS):
            piece = board.grid[r][c]
            key = (r, c, type(piece) if piece else None, piece.color if piece else None)
            hash_val ^= ZOBRIST_TABLE.get(key, 0)
    if turn == 'black':
        hash_val ^= ZOBRIST_TABLE['turn']
    return hash_val
TTEntry = namedtuple('TTEntry', ['score', 'depth', 'flag', 'best_move'])
TT_FLAG_EXACT, TT_FLAG_LOWERBOUND, TT_FLAG_UPPERBOUND = 0, 1, 2
class SearchCancelledException(Exception): pass

class OpponentAI:
    search_depth = 3
    
    PIECE_VALUES = {
        Pawn: 100, Knight: 700, Bishop: 600,
        Rook: 500, Queen: 900, King: 100000
    }
    MATE_SCORE = 1000000
    DRAW_SCORE = 0
    QSEARCH_CHECKS_MAX_DEPTH = 4
    CAPTURE_SCORE_BONUS = 10000 # Note: This Opponent Bot uses a simpler ordering
    PROMOTION_SCORE_BONUS = 9000
    CHECK_SCORE_BONUS = 5000
    HASH_MOVE_SCORE_BONUS = 100000

    def __init__(self, board, color, app, cancellation_event, bot_name="Opponent AI"):
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
    
    def evaluate_board(self, board, current_turn):
        perspective_multiplier = 1 if current_turn == self.color else -1
        score_relative_to_ai = 0
        our_king_pos = board.find_king_pos(self.color)
        enemy_king_pos = board.find_king_pos(self.opponent_color)
        if not enemy_king_pos: return self.MATE_SCORE * perspective_multiplier
        if not our_king_pos: return -self.MATE_SCORE * perspective_multiplier
        for r_eval in range(ROWS):
            for c_eval in range(COLS):
                piece_eval = board.grid[r_eval][c_eval]
                if not piece_eval: continue
                is_our_piece = (piece_eval.color == self.color)
                value = self.PIECE_VALUES.get(type(piece_eval), 0)
                if isinstance(piece_eval, Knight):
                    value += len(piece_eval.get_valid_moves(board, (r_eval, c_eval))) * 5
                elif isinstance(piece_eval, Queen) and is_our_piece:
                    atomic_threats = 0
                    for move_q in piece_eval.get_valid_moves(board, (r_eval, c_eval)):
                        target_piece = board.grid[move_q[0]][move_q[1]]
                        if target_piece and target_piece.color == self.opponent_color:
                            if enemy_king_pos and max(abs(move_q[0] - enemy_king_pos[0]), abs(move_q[1] - enemy_king_pos[1])) == 1:
                                atomic_threats += 1
                    value += atomic_threats * 15
                if is_our_piece:
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
        piece = board.grid[start[0]][start[1]]
        target = board.grid[end[0]][end[1]]
        score = 0
        if target:
            score = self.CAPTURE_SCORE_BONUS + (self.PIECE_VALUES.get(type(target), 0) - self.PIECE_VALUES.get(type(piece), 0))
        if isinstance(piece, Pawn) and (end[0] == 0 or end[0] == ROWS - 1):
            score += self.PROMOTION_SCORE_BONUS
        if self.is_move_a_check(board, move, moving_player_color):
            score += self.CHECK_SCORE_BONUS
        return score

    def order_moves(self, board, moves, moving_player_color, ply_searched=0, hash_move=None):
        move_scores = {move: self.evaluate_move(board, move, moving_player_color) for move in moves}
        if hash_move and hash_move in move_scores:
            move_scores[hash_move] += self.HASH_MOVE_SCORE_BONUS
        if ply_searched < self.MAX_PLY_KILLERS:
            killers = self.killer_moves[ply_searched]
            if killers[0] and killers[0] in move_scores:
                move_scores[killers[0]] += 50000
            if killers[1] and killers[1] in move_scores:
                move_scores[killers[1]] += 40000
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
            child_board = self.simulate_move(board, move[0], move[1])
            score = -self.qsearch(child_board, -beta, -alpha, -turn_multiplier, depth - 1)
            if score >= beta: return beta
            alpha = max(alpha, score)
        return alpha

    def negamax(self, board, depth, alpha, beta, turn_multiplier, ply):
        if self.cancellation_event.is_set(): raise SearchCancelledException()
        self.nodes_searched += 1
        current_turn = self.color if turn_multiplier == 1 else self.opponent_color
        original_alpha = alpha
        hash_val = board_hash(board, current_turn)
        if ply > 0 and self.app.position_counts.get(hash_val, 0) >= 2: return self.DRAW_SCORE
        hash_move = None
        tt_entry = self.tt.get(hash_val)
        if tt_entry and tt_entry.depth >= depth:
            if tt_entry.flag == TT_FLAG_EXACT: return tt_entry.score
            elif tt_entry.flag == TT_FLAG_LOWERBOUND: alpha = max(alpha, tt_entry.score)
            elif tt_entry.flag == TT_FLAG_UPPERBOUND: beta = min(beta, tt_entry.score)
            if alpha >= beta: return tt_entry.score
            hash_move = tt_entry.best_move
        is_in_check_now = is_in_check(board, current_turn)
        if is_in_check_now: depth += 1
        if depth <= 0:
            return self.qsearch(board, alpha, beta, turn_multiplier, self.QSEARCH_CHECKS_MAX_DEPTH)
        moves = generate_pseudo_legal_moves(board, current_turn)
        ordered_moves = self.order_moves(board, moves, current_turn, ply, hash_move=hash_move)
        legal_moves_found = 0
        best_move_for_node = None
        for move in ordered_moves:
            child_board = self.simulate_move(board, move[0], move[1])
            if is_in_check(child_board, current_turn): continue
            legal_moves_found += 1
            child_turn = self.opponent_color if turn_multiplier == 1 else self.color
            child_hash = board_hash(child_board, child_turn)
            self.app.position_counts[child_hash] = self.app.position_counts.get(child_hash, 0) + 1
            score = -self.negamax(child_board, depth - 1, -beta, -alpha, -turn_multiplier, ply + 1)
            self.app.position_counts[child_hash] -= 1
            if score > alpha:
                alpha = score
                best_move_for_node = move
            if alpha >= beta:
                is_capture = board.grid[move[1][0]][move[1][1]] is not None
                if not is_capture and ply < self.MAX_PLY_KILLERS and self.killer_moves[ply][0] != move:
                    self.killer_moves[ply][1] = self.killer_moves[ply][0]
                    self.killer_moves[ply][0] = move
                self.tt[hash_val] = TTEntry(beta, depth, TT_FLAG_LOWERBOUND, move)
                return beta
        if legal_moves_found == 0:
            return -self.MATE_SCORE + ply if is_in_check_now else self.DRAW_SCORE
        flag = TT_FLAG_UPPERBOUND if alpha <= original_alpha else TT_FLAG_EXACT
        self.tt[hash_val] = TTEntry(alpha, depth, flag, best_move_for_node)
        return alpha

    def make_move(self):
        try:
            best_move_found = None
            self.killer_moves = [[None, None] for _ in range(self.MAX_PLY_KILLERS)]
            self.tt.clear()
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
                    if is_in_check(child_board, self.color): continue
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
                # --- THE FIX IS HERE ---
                self.board.make_move(best_move_found[0], best_move_found[1])
                return True
            return False
        except SearchCancelledException:
            print(f"AI ({self.color}): Search cancelled.")
            return False
    
    def simulate_move(self, board, start, end):
        new_board = board.clone()
        new_board.make_move(start, end)
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