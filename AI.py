# AI.py (v6.0 - Advanced Search Engine)
# - Implemented advanced pruning: Null Move Pruning (NMP), Late Move Reduction (LMR), and Delta Pruning.
# - Overhauled the core search algorithm (negamax) and move ordering for significant performance gains.
# - Fixed critical bugs related to recursion, variable scope, and transposition table logic.

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

class ChessBot:
    # --- Search Constants ---
    search_depth = 3
    MATE_SCORE = 1000000
    DRAW_SCORE = 0
    DRAW_PENALTY = -10
    MAX_Q_SEARCH_DEPTH = 8
    
    # --- Pruning Constants ---
    DELTA_PRUNING_MARGIN = 200 # For quiescence search
    NMP_DEPTH_THRESHOLD = 3
    NMP_REDUCTION = 3
    LMR_DEPTH_THRESHOLD = 3
    LMR_MOVE_COUNT_THRESHOLD = 4
    LMR_REDUCTION = 1
    
    # --- Move Ordering Bonuses ---
    BONUS_PV_MOVE = 2_000_000
    BONUS_GOOD_CAPTURE = 1_500_000
    BONUS_PROMOTION = 1_200_000
    BONUS_CHECKING_MOVE = 1_000_000
    BONUS_KILLER_1 = 900_000
    BONUS_KILLER_2 = 850_000
    BONUS_LOSING_CAPTURE = 300_000
    
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
        self.killer_moves = [[None, None] for _ in range(50)] # Increased max ply for safety
        self.history_heuristic_table = [[[0 for _ in range(ROWS*COLS)] for _ in range(ROWS*COLS)] for _ in range(2)]

    # --- Communication and Formatting Methods ---
    def _report_log(self, message):
        self.comm_queue.put(('log', message))

    def _report_eval(self, score, depth):
        eval_for_ui = score if self.color == 'white' else -score
        self.comm_queue.put(('eval', eval_for_ui, depth))

    def _report_move(self, move):
        self.comm_queue.put(('move', move))

    def _format_move(self, move):
        if not move:
            return "None"
        (r1, c1), (r2, c2) = move
        files = "abcdefgh"
        ranks = "87654321"
        return f"{files[c1]}{ranks[r1]}-{files[c2]}{ranks[r2]}"

    # --- Main Search Entry Points ---
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
        try:
            best_move_overall = None
            root_moves = get_all_legal_moves(self.board, self.color)
            if not root_moves:
                return
            
            best_move_overall = root_moves[0]
            root_hash = board_hash(self.board, self.color)
            
            for current_depth in range(1, 100):
                if self.cancellation_event.is_set():
                    raise SearchCancelledException()

                iter_start_time = time.time()

                best_score_this_iter, best_move_this_iter = self._search_at_depth(
                    current_depth, root_moves, root_hash, best_move_overall
                )
                
                if not self.cancellation_event.is_set():
                    best_move_overall = best_move_this_iter
                    
                    iter_duration = time.time() - iter_start_time
                    knps = (self.nodes_searched / iter_duration / 1000) if iter_duration > 0 else 0
                    move_str = self._format_move(best_move_this_iter)

                    log_msg = (f"  > Analysis (D{current_depth}): {move_str}, "
                               f"Time={iter_duration:.2f}s, Nodes={self.nodes_searched}, KNPS={knps:.1f}")
                    self._report_log(log_msg)

                    self._report_eval(best_score_this_iter, current_depth)
                else:
                    raise SearchCancelledException()

        except SearchCancelledException:
            pass
        finally:
            self._report_log(f"{self.bot_name} ({self.color}): Pondering stopped.")

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
            
            ### --- FIX: Use the new parameter name --- ###
            # The initial call to negamax must now include the check status.
            child_is_in_check_flag = is_in_check(child_board, self.opponent_color)
            score = -self.negamax(child_board, depth - 1, -beta, -alpha, self.opponent_color, 1, search_path, child_is_in_check_flag)
            ### --- END FIX --- ###

            self.position_counts[child_hash] -= 1

            if score > best_score_this_iter:
                best_score_this_iter = score
                best_move_this_iter = move
            
            alpha = max(alpha, best_score_this_iter)
        
        return best_score_this_iter, best_move_this_iter

    def negamax(self, board, depth, alpha, beta, turn, ply, search_path, is_in_check_flag):
        self.nodes_searched += 1
        if self.cancellation_event.is_set():
            raise SearchCancelledException()

        # --- Step 1: Repetition and Max Depth Checks ---
        hash_val = board_hash(board, turn)
        if ply > 0 and (hash_val in search_path or self.position_counts.get(hash_val, 0) >= 2):
            return self.DRAW_PENALTY
        
        if depth <= 0:
            return self.qsearch(board, alpha, beta, turn, ply)

        # --- Step 2: Transposition Table Probe ---
        original_alpha = alpha
        tt_entry = self.tt.get(hash_val)
        if ply > 0 and tt_entry and tt_entry.depth >= depth:
            if tt_entry.flag == TT_FLAG_EXACT: return tt_entry.score
            elif tt_entry.flag == TT_FLAG_LOWERBOUND: alpha = max(alpha, tt_entry.score)
            elif tt_entry.flag == TT_FLAG_UPPERBOUND: beta = min(beta, tt_entry.score)
            if alpha >= beta: return tt_entry.score

        # --- Step 3: Null Move Pruning (NMP) ---
        opponent_turn = 'black' if turn == 'white' else 'white'
        if not is_in_check_flag and depth >= self.NMP_DEPTH_THRESHOLD and not self.is_endgame(board) and ply > 0:
            # Pass a null move to the opponent and see if it's still good for us.
            # Note: We pass 'False' for the new 'is_in_check_flag'.
            score = -self.negamax(board, depth - 1 - self.NMP_REDUCTION, -beta, -beta + 1, opponent_turn, ply + 1, new_search_path, False)
            if score >= beta:
                # If giving a free move is still too good, this position is a win. Prune.
                return beta

        # --- Step 4: Main Search Logic ---
        if is_in_check_flag:
            depth += 1 # Check extension

        legal_moves = get_all_legal_moves(board, turn)
        if not legal_moves:
            return -self.MATE_SCORE + ply if is_in_check_flag else self.DRAW_SCORE

        # --- Step 5: Move Ordering and Iteration ---
        hash_move = tt_entry.best_move if tt_entry else None
        ordered_moves = self.order_moves(board, legal_moves, ply, hash_move)
        best_move_for_node = None
        new_search_path = search_path | {hash_val}

        for i, move in enumerate(ordered_moves):
            is_capture = board.grid[move[1][0]][move[1][1]] is not None
            
            child_board = board.clone()
            child_board.make_move(move[0], move[1])
            
            # --- Step 6: Late Move Reduction (LMR) ---
            reduction = 0
            if (depth >= self.LMR_DEPTH_THRESHOLD and i >= self.LMR_MOVE_COUNT_THRESHOLD and 
                not is_in_check_flag and not is_capture):
                reduction = self.LMR_REDUCTION

            # --- Step 7: Recursive Search Call (PVS framework) ---
            child_is_in_check_flag = is_in_check(child_board, opponent_turn)
            score = 0
            
            # Principal Variation Search (PVS) Optimization
            if i == 0: # First move is the PV move, do a full search
                score = -self.negamax(child_board, depth - 1 - reduction, -beta, -alpha, opponent_turn, ply + 1, new_search_path, child_is_in_check_flag)
            else: # For other moves, try a quick "zero-window" search
                score = -self.negamax(child_board, depth - 1 - reduction, -alpha - 1, -alpha, opponent_turn, ply + 1, new_search_path, child_is_in_check_flag)
                # If the zero-window search failed high, it means this move is better than the PV. We must re-search with a full window.
                if score > alpha and score < beta:
                    score = -self.negamax(child_board, depth - 1 - reduction, -beta, -alpha, opponent_turn, ply + 1, new_search_path, child_is_in_check_flag)

            if reduction > 0 and score > alpha:
                score = -self.negamax(child_board, depth - 1, -beta, -alpha, opponent_turn, ply + 1, new_search_path, child_is_in_check_flag)
            
            if score > alpha:
                alpha = score
                best_move_for_node = move
                
            if alpha >= beta:
                if not is_capture:
                    if ply < len(self.killer_moves) and self.killer_moves[ply][0] != move:
                        self.killer_moves[ply][1] = self.killer_moves[ply][0]
                        self.killer_moves[ply][0] = move
                        
                    color_index = 0 if turn == 'white' else 1
                    from_sq = move[0][0]*COLS + move[0][1]
                    to_sq = move[1][0]*COLS + move[1][1]
                    self.history_heuristic_table[color_index][from_sq][to_sq] += depth * depth
                break # Beta-cutoff

        # --- Step 8: Transposition Table Store ---
        flag = TT_FLAG_EXACT if alpha > original_alpha else TT_FLAG_UPPERBOUND
        if alpha >= beta:
            flag = TT_FLAG_LOWERBOUND
        
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
        
        biggest_gain = PIECE_VALUES[Queen] + self.DELTA_PRUNING_MARGIN
        if stand_pat + biggest_gain < alpha:
            return alpha

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
            opponent_turn = 'black' if turn == 'white' else 'white'
            score = -self.qsearch(child_board, -beta, -alpha, opponent_turn, ply + 1)
            
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
        
        scores = {}
        killers = self.killer_moves[ply] if ply < len(self.killer_moves) else [None, None]
        moving_color = board.grid[moves[0][0][0]][moves[0][0][1]].color
        color_index = 0 if moving_color == 'white' else 1

        # Score every move based on its type to establish a sorting priority
        for move in moves:
            score = 0
            start_pos, end_pos = move
            moving_piece = board.grid[start_pos[0]][start_pos[1]]
            target_piece = board.grid[end_pos[0]][end_pos[1]]
            is_capture = target_piece is not None

            if is_capture:
                # MVV-LVA: Most Valuable Victim - Least Valuable Aggressor.
                # Capturing a high-value piece with a low-value one is best.
                # We give this a very high base score to ensure captures are checked first.
                mvv_lva_score = PIECE_VALUES.get(type(target_piece), 0) - PIECE_VALUES.get(type(moving_piece), 0)
                score = self.BONUS_GOOD_CAPTURE + mvv_lva_score
            else: # Is a Quiet Move
                if move == killers[0]:
                    score = self.BONUS_KILLER_1
                elif move == killers[1]:
                    score = self.BONUS_KILLER_2
                else:
                    # History Heuristic for other quiet moves
                    from_sq = start_pos[0] * COLS + start_pos[1]
                    to_sq = end_pos[0] * COLS + end_pos[1]
                    score = self.history_heuristic_table[color_index][from_sq][to_sq]
            
            scores[move] = score

        # Sort all moves based on their calculated scores.
        moves.sort(key=lambda m: scores.get(m, 0), reverse=True)

        # The hash_move (Principal Variation move from the TT) is the best predictor.
        # If it exists, we manually move it to the very front of the list to be
        # searched first, as it's the most likely to cause a quick beta-cutoff.
        if hash_move in moves:
            moves.remove(hash_move)
            moves.insert(0, hash_move)
            
        return moves
        
    def is_endgame(self, board):
        material = 0
        for r in board.grid:
            for p in r:
                if p and not isinstance(p, King):
                    material += PIECE_VALUES.get(type(p), 0)
        return material < ENDGAME_MATERIAL_THRESHOLD
    
    def evaluate_board(self, board, turn_to_move):
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
                    pst_score = pst[r][c] if piece.color == 'white' else pst[ROWS - 1 - r][c]
                    value += pst_score
                
                if piece.color == turn_to_move:
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