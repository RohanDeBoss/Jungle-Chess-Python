# OpponentAI.py (v6.0b - UnboundLocalError Fix)

import time
from GameLogic import *
import random
from collections import namedtuple

# Zobrist hashing table for transposition table keys
ZOBRIST_TABLE = None

def initialize_zobrist_table():
    global ZOBRIST_TABLE
    if ZOBRIST_TABLE is not None:
        return

    random.seed(42) # Use a fixed seed for deterministic hashes
    table = {}
    piece_types = [Pawn, Knight, Bishop, Rook, Queen, King, None]
    colors = ['white', 'black', None]
    
    for r in range(ROWS):
        for c in range(COLS):
            for piece_type in piece_types:
                for piece_color in colors:
                    key = (r, c, piece_type, piece_color)
                    table[key] = random.getrandbits(64)
                    
    table['turn'] = random.getrandbits(64)
    ZOBRIST_TABLE = table

# Initialize the table when the module is loaded
initialize_zobrist_table()

def board_hash(board, turn):
    h = 0
    for r in range(ROWS):
        for c in range(COLS):
            piece = board.grid[r][c]
            if piece:
                key = (r, c, type(piece), piece.color)
            else:
                key = (r, c, None, None)
            
            h ^= ZOBRIST_TABLE.get(key, 0)

    if turn == 'black':
        h ^= ZOBRIST_TABLE['turn']
    return h

# --- Transposition Table Setup ---
TTEntry = namedtuple('TTEntry', ['score', 'depth', 'flag', 'best_move'])
TT_FLAG_EXACT = 0
TT_FLAG_LOWERBOUND = 1
TT_FLAG_UPPERBOUND = 2

class SearchCancelledException(Exception):
    """Custom exception to cleanly exit the search."""
    pass

class OpponentAI:
    # --- Search Constants ---
    search_depth = 3
    MATE_SCORE = 1000000
    DRAW_SCORE = 0
    DRAW_PENALTY = -10
    MAX_Q_SEARCH_DEPTH = 8
    
    # --- Move Ordering Bonuses ---
    BONUS_PV_MOVE = 2_000_000
    BONUS_GOOD_CAPTURE = 1_500_000
    BONUS_PROMOTION = 1_200_000
    BONUS_CHECKING_MOVE = 1_000_000
    BONUS_KILLER_1 = 900_000
    BONUS_KILLER_2 = 850_000
    BONUS_LOSING_CAPTURE = 300_000
    
    # --- Late Move Reduction (LMR) Constants ---
    LMR_DEPTH_THRESHOLD = 3
    LMR_MOVE_COUNT_THRESHOLD = 4
    LMR_REDUCTION = 1

    def __init__(self, board, color, position_counts, comm_queue, cancellation_event, bot_name="AI Bot"):
        self.board = board
        self.color = color
        self.opponent_color = 'black' if color == 'white' else 'white'
        self.position_counts = position_counts
        self.comm_queue = comm_queue
        self.cancellation_event = cancellation_event
        self.bot_name = bot_name
        self.tt = {}
        self.nodes_searched = 0
        self.killer_moves = [[None, None] for _ in range(30)]
        self.history_heuristic_table = [[[0 for _ in range(ROWS*COLS)] for _ in range(ROWS*COLS)] for _ in range(2)]

    def _report_log(self, message):
        self.comm_queue.put(('log', message))

    def _report_eval(self, score, depth):
        eval_for_ui = score if self.color == 'white' else -score
        self.comm_queue.put(('eval', eval_for_ui, depth))

    def _report_move(self, move):
        self.comm_queue.put(('move', move))

    def make_move(self):
        try:
            best_move_overall = None
            root_moves = get_all_legal_moves(self.board, self.color)
            if not root_moves:
                self._report_move(None)
                return
            
            best_move_overall = root_moves[0]
            root_hash = board_hash(self.board, self.color)

            for current_depth in range(1, self.search_depth + 1):
                iter_start_time = time.time()
                
                best_score_this_iter, best_move_this_iter = self._search_at_depth(
                    current_depth, root_moves, root_hash, best_move_overall
                )
                
                if not self.cancellation_event.is_set():
                    best_move_overall = best_move_this_iter
                    
                    iter_duration = time.time() - iter_start_time
                    knps = (self.nodes_searched / iter_duration / 1000) if iter_duration > 0 else 0
                    eval_for_ui = best_score_this_iter if self.color == 'white' else -best_score_this_iter

                    log_msg = (f"  > {self.bot_name} (Depth {current_depth}): "
                               f"Eval={eval_for_ui/100:+.2f}, Nodes={self.nodes_searched}, KNPS={knps:.1f}")
                    self._report_log(log_msg)
                    self._report_eval(best_score_this_iter, None)
                else:
                    raise SearchCancelledException()

            self._report_move(best_move_overall)

        except SearchCancelledException:
            self._report_log(f"AI ({self.color}): Search cancelled.")
            self._report_move(None)

    def ponder_indefinitely(self):
        pass

    def _search_at_depth(self, depth, root_moves, root_hash, pv_move):
        self.nodes_searched = 0
        best_score_this_iter = -float('inf')
        best_move_this_iter = None
        alpha, beta = -float('inf'), float('inf')
        
        ordered_root_moves = self.order_moves(self.board, root_moves, 0, pv_move)

        for move in ordered_root_moves:
            if self.cancellation_event.is_set():
                raise SearchCancelledException()

            child_board = self.board.clone()
            child_board.make_move(move[0], move[1])
            child_hash = board_hash(child_board, self.opponent_color)
            
            self.position_counts[child_hash] = self.position_counts.get(child_hash, 0) + 1
            
            search_path = {root_hash}
            score = -self.negamax(child_board, depth - 1, -beta, -alpha, self.opponent_color, 1, search_path)
            
            self.position_counts[child_hash] -= 1

            if score > best_score_this_iter:
                best_score_this_iter = score
                best_move_this_iter = move
            
            alpha = max(alpha, best_score_this_iter)
        
        return best_score_this_iter, best_move_this_iter

    def negamax(self, board, depth, alpha, beta, turn, ply, search_path):
        self.nodes_searched += 1
        if self.cancellation_event.is_set():
            raise SearchCancelledException()
            
        hash_val = board_hash(board, turn)
        if hash_val in search_path:
            return self.DRAW_PENALTY

        original_alpha = alpha
        tt_entry = self.tt.get(hash_val)
        if tt_entry and tt_entry.depth >= depth:
            if tt_entry.flag == TT_FLAG_EXACT:
                return tt_entry.score
            elif tt_entry.flag == TT_FLAG_LOWERBOUND:
                alpha = max(alpha, tt_entry.score)
            elif tt_entry.flag == TT_FLAG_UPPERBOUND:
                beta = min(beta, tt_entry.score)
            if alpha >= beta:
                return tt_entry.score

        if depth <= 0:
            return self.qsearch(board, alpha, beta, turn, ply)
            
        if ply > 0 and self.position_counts.get(hash_val, 0) >= 2:
            return self.DRAW_PENALTY

        opponent_turn = 'black' if turn == 'white' else 'white'
        is_check_now = is_in_check(board, turn)
        if is_check_now:
            depth += 1

        legal_moves = get_all_legal_moves(board, turn)
        if not legal_moves:
            return -self.MATE_SCORE + ply if is_check_now else self.DRAW_SCORE

        hash_move = tt_entry.best_move if tt_entry else None
        ordered_moves = self.order_moves(board, legal_moves, ply, hash_move)
        best_move_for_node = None
        new_search_path = search_path | {hash_val}

        for i, move in enumerate(ordered_moves):
            ### --- BUG FIX --- ###
            # Define is_capture at the top of the loop to prevent UnboundLocalError
            is_capture = board.grid[move[1][0]][move[1][1]] is not None
            ### --- END BUG FIX --- ###

            child_board = board.clone()
            child_board.make_move(move[0], move[1])
            child_hash = board_hash(child_board, opponent_turn)
            
            score = 0
            if child_hash in new_search_path or self.position_counts.get(child_hash, 0) >= 2:
                score = self.DRAW_PENALTY
            else:
                self.position_counts[child_hash] = self.position_counts.get(child_hash, 0) + 1
                
                reduction = 0
                if depth >= self.LMR_DEPTH_THRESHOLD and i >= self.LMR_MOVE_COUNT_THRESHOLD and not is_check_now and not is_capture:
                    reduction = self.LMR_REDUCTION
                    
                score = -self.negamax(child_board, depth - 1 - reduction, -beta, -alpha, opponent_turn, ply + 1, new_search_path)
                
                if reduction > 0 and score > alpha:
                    score = -self.negamax(child_board, depth - 1, -beta, -alpha, opponent_turn, ply + 1, new_search_path)
                    
                self.position_counts[child_hash] -= 1

            if score > alpha:
                alpha = score
                best_move_for_node = move
                
            if alpha >= beta:
                if not is_capture:
                    if ply < len(self.killer_moves) and self.killer_moves[ply][0] != move:
                        self.killer_moves[ply][1] = self.killer_moves[ply][0]
                        self.killer_moves[ply][0] = move
                        
                    color_index = 0 if turn == 'white' else 1
                    from_sq, to_sq = move[0][0]*COLS+move[0][1], move[1][0]*COLS+move[1][1]
                    self.history_heuristic_table[color_index][from_sq][to_sq] += depth * depth
                    
                self.tt[hash_val] = TTEntry(beta, depth, TT_FLAG_LOWERBOUND, move)
                return beta
                
        flag = TT_FLAG_EXACT if alpha > original_alpha else TT_FLAG_UPPERBOUND
        self.tt[hash_val] = TTEntry(alpha, depth, flag, best_move_for_node)
        return alpha

    def qsearch(self, board, alpha, beta, turn, ply):
        self.nodes_searched += 1
        if self.cancellation_event.is_set():
            raise SearchCancelledException()
        if ply >= self.MAX_Q_SEARCH_DEPTH:
            return self.evaluate_board(board, turn)

        stand_pat = self.evaluate_board(board, turn)
        if stand_pat >= beta:
            return beta
        alpha = max(alpha, stand_pat)

        legal_moves = get_all_legal_moves(board, turn)
        
        tactical_moves = []
        for m in legal_moves:
            is_capture = board.grid[m[1][0]][m[1][1]] is not None
            is_promotion = isinstance(board.grid[m[0][0]][m[0][1]], Pawn) and (m[1][0] in [0, ROWS - 1])
            if is_capture or is_promotion or self._is_checking_move(board, m, turn):
                tactical_moves.append(m)
        
        ordered_tactical_moves = self.order_moves(board, tactical_moves, ply, in_qsearch=True)
        for move in ordered_tactical_moves:
            child_board = board.clone()
            child_board.make_move(move[0], move[1])
            score = -self.qsearch(child_board, -beta, -alpha, 'black' if turn == 'white' else 'white', ply + 1)
            
            if score >= beta:
                return beta
            alpha = max(alpha, score)
            
        return alpha

    def _is_checking_move(self, board, move, turn):
        sim_board = board.clone()
        sim_board.make_move(move[0], move[1])
        opponent_color = 'black' if turn == 'white' else 'white'
        return is_in_check(sim_board, opponent_color)

    def order_moves(self, board, moves, ply, hash_move=None, in_qsearch=False):
        if not moves:
            return []
            
        move_scores = {}
        moving_color = board.grid[moves[0][0][0]][moves[0][0][1]].color
        color_index = 0 if moving_color == 'white' else 1
        killers = self.killer_moves[ply] if ply < len(self.killer_moves) else [None, None]
        
        for move in moves:
            score = 0
            start_pos, end_pos = move
            moving_piece = board.grid[start_pos[0]][start_pos[1]]
            target_piece = board.grid[end_pos[0]][end_pos[1]]
            is_capture = target_piece is not None
            is_promotion = isinstance(moving_piece, Pawn) and (end_pos[0] == 0 or end_pos[0] == ROWS - 1)

            if move == hash_move:
                score = self.BONUS_PV_MOVE
            elif is_capture:
                mvv_lva_score = PIECE_VALUES.get(type(target_piece), 0) - PIECE_VALUES.get(type(moving_piece), 0)
                if mvv_lva_score >= 0:
                    score = self.BONUS_GOOD_CAPTURE + mvv_lva_score
                else:
                    score = self.BONUS_LOSING_CAPTURE + mvv_lva_score
            elif is_promotion:
                score = self.BONUS_PROMOTION
            elif in_qsearch and self._is_checking_move(board, move, moving_color):
                score = self.BONUS_CHECKING_MOVE
            elif move == killers[0]:
                score = self.BONUS_KILLER_1
            elif move == killers[1]:
                score = self.BONUS_KILLER_2
            else:
                from_sq = start_pos[0] * COLS + start_pos[1]
                to_sq = end_pos[0] * COLS + end_pos[1]
                score = self.history_heuristic_table[color_index][from_sq][to_sq]
                
            move_scores[move] = score
            
        return sorted(moves, key=lambda m: move_scores.get(m, 0), reverse=True)
        
    def is_endgame(self, board):
        material = 0
        for r in board.grid:
            for p in r:
                if p and not isinstance(p, King):
                    material += PIECE_VALUES.get(type(p), 0)
        return material < ENDGAME_MATERIAL_THRESHOLD
    
    def evaluate_board(self, board, turn):
        score = 0
        in_endgame = self.is_endgame(board)
        
        for r in range(ROWS):
            for c in range(COLS):
                piece = board.grid[r][c]
                if not piece:
                    continue
                
                value = PIECE_VALUES.get(type(piece), 0)
                
                if isinstance(piece, King):
                    pst = PIECE_SQUARE_TABLES['king_endgame'] if in_endgame else PIECE_SQUARE_TABLES['king_midgame']
                else:
                    pst = PIECE_SQUARE_TABLES.get(type(piece))

                if pst:
                    if piece.color == 'white':
                        pst_score = pst[r][c]
                    else:
                        pst_score = pst[ROWS - 1 - r][c]
                    value += pst_score
                
                if piece.color == turn:
                    score += value
                else:
                    score -= value
                    
        return score

# -----------------------------------------------------------------------------
# Piece-Square Tables (PSTs) and Material Values for this Variant
# -----------------------------------------------------------------------------

# Values reflect the destructive power of the pieces
PIECE_VALUES = {
    Pawn: 100, Knight: 750, Bishop: 700,
    Rook: 650, Queen: 1000, King: 30000
}
ENDGAME_MATERIAL_THRESHOLD = (PIECE_VALUES[Rook] * 2 + PIECE_VALUES[Bishop] + PIECE_VALUES[Knight])

pawn_pst = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [100, 100, 100, 100, 100, 100, 100, 100],
    [50, 50, 50, 50, 50, 50, 50, 50],
    [30, 30, 40, 50, 50, 40, 30, 30],
    [20, 20, 30, 40, 40, 30, 20, 20],
    [10, 10, 20, 30, 30, 20, 10, 10],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
]

knight_pst = [
    [-50, -40, -30, -30, -30, -30, -40, -50],
    [-40, -20,   5,  10,  10,   5, -20, -40],
    [-30,   5,  20,  25,  25,  20,   5, -30],
    [-30,  10,  25,  35,  35,  25,  10, -30],
    [-30,  10,  25,  35,  35,  25,  10, -30],
    [-30,  10,  20,  25,  25,  20,  10, -30],
    [-40, -20,   5,  10,  10,   5, -20, -40],
    [-50, -40, -30, -30, -30, -30, -40, -50]
]

bishop_pst = [
    [-20, -10, -10, -10, -10, -10, -10, -20],
    [-10,   0,   0,   0,   0,   0,   0, -10],
    [-10,   0,   5,  10,  10,   5,   0, -10],
    [-10,   5,   5,  10,  10,   5,   5, -10],
    [-10,   0,  10,  10,  10,  10,   0, -10],
    [-10,  10,  10,  10,  10,  10,  10, -10],
    [-10,   5,   0,   0,   0,   0,   5, -10],
    [-20, -10, -10, -10, -10, -10, -10, -20]
]

rook_pst = [
    [  0,   0,   0,  10,  10,   0,   0,   0],
    [  5,  10,  10,  20,  20,  10,  10,   5],
    [ -5,   0,   0,   5,   5,   0,   0,  -5],
    [ -5,   0,   0,   5,   5,   0,   0,  -5],
    [ -5,   0,   0,   5,   5,   0,   0,  -5],
    [ -5,   0,   0,   5,   5,   0,   0,  -5],
    [ -5,   0,   0,   5,   5,   0,   0,  -5],
    [  0,   5,   5,  10,  10,   5,   5,   0]
]

queen_pst = [
    [-20, -10, -10,  -5,  -5, -10, -10, -20],
    [-10,   0,   0,   0,   0,   0,   0, -10],
    [-10,   0,   5,   5,   5,   5,   0, -10],
    [ -5,   0,   5,   5,   5,   5,   0,  -5],
    [  0,   0,   5,   5,   5,   5,   0,  -5],
    [-10,   5,   5,   5,   5,   5,   5, -10],
    [-10,   0,   5,   0,   0,   0,   5, -10],
    [-20, -10, -10,  -5,  -5, -10, -10, -20]
]

king_midgame_pst = [
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-20, -30, -30, -40, -40, -30, -30, -20],
    [-10, -20, -20, -20, -20, -20, -20, -10],
    [ 20,  20,   0,   0,   0,   0,  20,  20],
    [ 20,  30,  10,   0,   0,  10,  30,  20]
]

king_endgame_pst = [
    [-50, -40, -30, -20, -20, -30, -40, -50],
    [-30, -20, -10,   0,   0, -10, -20, -30],
    [-30, -10,  20,  30,  30,  20, -10, -30],
    [-30, -10,  30,  40,  40,  30, -10, -30],
    [-30, -10,  30,  40,  40,  30, -10, -30],
    [-30, -10,  20,  30,  30,  20, -10, -30],
    [-30, -30,   0,   0,   0,   0, -30, -30],
    [-50, -30, -30, -30, -30, -30, -30, -50]
]

PIECE_SQUARE_TABLES = {
    Pawn: pawn_pst, Knight: knight_pst, Bishop: bishop_pst, Rook: rook_pst, 
    Queen: queen_pst, 'king_midgame': king_midgame_pst, 'king_endgame': king_endgame_pst
}