# OPAI.py (v7.5 - LMR + NMP!)

import time
from GameLogic import *
import random
from collections import namedtuple

# (Zobrist/TT setup is unchanged)
ZOBRIST_TABLE = None
def initialize_zobrist_table():
    global ZOBRIST_TABLE
    if ZOBRIST_TABLE is not None: return
    random.seed(42)
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
initialize_zobrist_table()

def board_hash(board, turn):
    h = 0
    for r in range(ROWS):
        for c in range(COLS):
            piece = board.grid[r][c]
            key = (r, c, type(piece) if piece else None, piece.color if piece else None)
            h ^= ZOBRIST_TABLE.get(key, 0)
    if turn == 'black': h ^= ZOBRIST_TABLE['turn']
    return h

TTEntry = namedtuple('TTEntry', ['score', 'depth', 'flag', 'best_move'])
TT_FLAG_EXACT, TT_FLAG_LOWERBOUND, TT_FLAG_UPPERBOUND = 0, 1, 2

class SearchCancelledException(Exception): pass

class OpponentAI:
    search_depth = 3
    MATE_SCORE, DRAW_SCORE, DRAW_PENALTY = 1000000, 0, -10
    MAX_Q_SEARCH_DEPTH = 8
    DELTA_PRUNING_MARGIN = 200
    LMR_DEPTH_THRESHOLD, LMR_MOVE_COUNT_THRESHOLD, LMR_REDUCTION = 3, 4, 2
    NMP_MIN_DEPTH, NMP_BASE_REDUCTION, NMP_DEPTH_DIVISOR = 3, 2, 6
    BONUS_PV_MOVE, BONUS_GOOD_CAPTURE, BONUS_PROMOTION = 2_000_000, 1_500_000, 1_200_000
    BONUS_CHECKING_MOVE, BONUS_KILLER_1, BONUS_KILLER_2 = 1_000_000, 900_000, 850_000
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
        self.killer_moves = [[None, None] for _ in range(50)]
        self.history_heuristic_table = [[[0 for _ in range(ROWS*COLS)] for _ in range(ROWS*COLS)] for _ in range(2)]

    def _report_log(self, message): self.comm_queue.put(('log', message))
    def _report_eval(self, score, depth): self.comm_queue.put(('eval', score if self.color == 'white' else -score, depth))
    def _report_move(self, move): self.comm_queue.put(('move', move))
    def _format_move(self, move):
        if not move: return "None"
        (r1, c1), (r2, c2) = move
        return f"{'abcdefgh'[c1]}{'87654321'[r1]}-{'abcdefgh'[c2]}{'87654321'[r2]}"

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
                best_score_this_iter, best_move_this_iter = self._search_at_depth(current_depth, root_moves, root_hash, best_move_overall)
                
                if not self.cancellation_event.is_set():
                    best_move_overall = best_move_this_iter
                    iter_duration = time.time() - iter_start_time
                    knps = (self.nodes_searched / iter_duration / 1000) if iter_duration > 0 else 0
                    eval_for_ui = best_score_this_iter if self.color == 'white' else -best_score_this_iter
                    log_msg = f"  > {self.bot_name} (Depth {current_depth}): Eval={eval_for_ui/100:+.2f}, Nodes={self.nodes_searched}, KNPS={knps:.1f}"
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
            if not root_moves: return
            best_move_overall = root_moves[0]
            root_hash = board_hash(self.board, self.color)
            for current_depth in range(1, 100):
                if self.cancellation_event.is_set(): raise SearchCancelledException()
                iter_start_time = time.time()
                best_score_this_iter, best_move_this_iter = self._search_at_depth(current_depth, root_moves, root_hash, best_move_overall)
                if not self.cancellation_event.is_set():
                    best_move_overall = best_move_this_iter
                    iter_duration = time.time() - iter_start_time
                    knps = (self.nodes_searched / iter_duration / 1000) if iter_duration > 0 else 0
                    move_str = self._format_move(best_move_this_iter)
                    log_msg = f"  > Analysis (D{current_depth}): {move_str}, Time={iter_duration:.2f}s, Nodes={self.nodes_searched}, KNPS={knps:.1f}"
                    self._report_log(log_msg)
                    self._report_eval(best_score_this_iter, current_depth)
                else:
                    raise SearchCancelledException()
        except SearchCancelledException: pass
        finally: self._report_log(f"{self.bot_name} ({self.color}): Pondering stopped.")

    def _search_at_depth(self, depth, root_moves, root_hash, pv_move):
        self.nodes_searched = 0
        best_score_this_iter, best_move_this_iter = -float('inf'), None
        alpha, beta = -float('inf'), float('inf')
        ordered_root_moves = self.order_moves(self.board, root_moves, 0, pv_move)
        
        for i, move in enumerate(ordered_root_moves):
            if self.cancellation_event.is_set(): raise SearchCancelledException()
            
            child_board = self.board.clone()
            child_board.make_move(move[0], move[1])
            
            search_path = {root_hash}
            child_hash = board_hash(child_board, self.opponent_color)
            self.position_counts[child_hash] = self.position_counts.get(child_hash, 0) + 1
            
            score = -self.negamax(child_board, depth - 1, -beta, -alpha, self.opponent_color, 1, search_path)
            
            self.position_counts[child_hash] -= 1

            if score > best_score_this_iter:
                best_score_this_iter = score
                best_move_this_iter = move
            
            alpha = max(alpha, best_score_this_iter)
            
        return best_score_this_iter, best_move_this_iter

    def negamax(self, board, depth, alpha, beta, turn, ply, search_path):
        self.nodes_searched += 1
        if self.cancellation_event.is_set(): raise SearchCancelledException()
        
        hash_val = board_hash(board, turn)
        if ply > 0 and hash_val in search_path:
            return self.DRAW_PENALTY

        original_alpha = alpha
        tt_entry = self.tt.get(hash_val)

        if ply > 0 and tt_entry and tt_entry.depth >= depth:
            if tt_entry.flag == TT_FLAG_EXACT: return tt_entry.score
            elif tt_entry.flag == TT_FLAG_LOWERBOUND: alpha = max(alpha, tt_entry.score)
            elif tt_entry.flag == TT_FLAG_UPPERBOUND: beta = min(beta, tt_entry.score)
            if alpha >= beta: return tt_entry.score

        if depth <= 0:
            return self.qsearch(board, alpha, beta, turn, ply)

        if ply > 0 and self.position_counts.get(hash_val, 0) >= 2:
            return self.DRAW_PENALTY

        opponent_turn = 'black' if turn == 'white' else 'white'
        is_in_check_flag = is_in_check(board, turn)
        
        if is_in_check_flag:
            depth += 1
        
        # --- [CRITICAL FIX] Removed the self-sabotaging `not is_pv_node` condition ---
        if (depth >= self.NMP_MIN_DEPTH and ply > 0 and not is_in_check_flag and
            beta < self.MATE_SCORE - 200 and 
            any(not isinstance(p, (Pawn, King)) for r in board.grid for p in r if p and p.color == turn)):
            
            nmp_reduction = self.NMP_BASE_REDUCTION + (depth // self.NMP_DEPTH_DIVISOR)
            nmp_search_path = search_path | {hash_val}
            score = -self.negamax(board, depth - 1 - nmp_reduction, -beta, -beta + 1, opponent_turn, ply + 1, nmp_search_path)
            
            if score >= beta:
                self.tt[hash_val] = TTEntry(beta, depth, TT_FLAG_LOWERBOUND, None)
                return beta 

        new_search_path = search_path | {hash_val}
        legal_moves = get_all_legal_moves(board, turn)
        if not legal_moves:
            return -self.MATE_SCORE + ply if is_in_check_flag else self.DRAW_SCORE

        hash_move = tt_entry.best_move if tt_entry else None
        ordered_moves = self.order_moves(board, legal_moves, ply, hash_move)
        best_move_for_node = None
        
        for i, move in enumerate(ordered_moves):
            is_capture = board.grid[move[1][0]][move[1][1]] is not None
            child_board = board.clone()
            child_board.make_move(move[0], move[1])
            
            child_hash = board_hash(child_board, opponent_turn)
            self.position_counts[child_hash] = self.position_counts.get(child_hash, 0) + 1
            
            reduction = 0
            if (depth >= self.LMR_DEPTH_THRESHOLD and i >= self.LMR_MOVE_COUNT_THRESHOLD and 
                not is_in_check_flag and not is_capture):
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
        if self.cancellation_event.is_set(): raise SearchCancelledException()
        if ply >= self.MAX_Q_SEARCH_DEPTH: return self.evaluate_board(board, turn)
        
        stand_pat = self.evaluate_board(board, turn)
        if stand_pat >= beta: return beta
        
        biggest_gain = PIECE_VALUES[Queen] + self.DELTA_PRUNING_MARGIN
        if stand_pat + biggest_gain < alpha: return alpha
        
        alpha = max(alpha, stand_pat)

        legal_moves = get_all_legal_moves(board, turn)
        tactical_moves = [m for m in legal_moves if board.grid[m[1][0]][m[1][1]] is not None or 
                          (isinstance(board.grid[m[0][0]][m[0][1]], Pawn) and (m[1][0] in [0, ROWS - 1])) or
                          self._is_checking_move(board, m, turn)]
                          
        ordered_tactical_moves = self.order_moves(board, tactical_moves, ply, in_qsearch=True)
        for move in ordered_tactical_moves:
            child_board = board.clone()
            child_board.make_move(move[0], move[1])
            score = -self.qsearch(child_board, -beta, -alpha, 'black' if turn == 'white' else 'white', ply + 1)
            if score >= beta: return beta
            alpha = max(alpha, score)
        return alpha

    def _is_checking_move(self, board, move, turn):
        sim_board = board.clone()
        sim_board.make_move(move[0], move[1])
        return is_in_check(sim_board, 'black' if turn == 'white' else 'white')

    def order_moves(self, board, moves, ply, hash_move=None, in_qsearch=False):
        if not moves:
            return []

        scores = {}
        killers = self.killer_moves[ply] if ply < len(self.killer_moves) else [None, None]
        
        moving_color = board.grid[moves[0][0][0]][moves[0][0][1]].color
        color_index = 0 if moving_color == 'white' else 1
        
        for move in moves:
            score = 0
            start_pos, end_pos = move
            moving_piece = board.grid[start_pos[0]][start_pos[1]]
            target_piece = board.grid[end_pos[0]][end_pos[1]]
            
            if move == hash_move:
                score = self.BONUS_PV_MOVE
            elif target_piece is not None:
                mvv_lva_score = PIECE_VALUES.get(type(target_piece), 0) - PIECE_VALUES.get(type(moving_piece), 0)
                score = self.BONUS_GOOD_CAPTURE + mvv_lva_score
            else:
                if move == killers[0]: score = self.BONUS_KILLER_1
                elif move == killers[1]: score = self.BONUS_KILLER_2
                else: score = self.history_heuristic_table[color_index][start_pos[0]*COLS+start_pos[1]][end_pos[0]*COLS+end_pos[1]]
            scores[move] = score
            
        moves.sort(key=lambda m: scores.get(m, 0), reverse=True)
        return moves
        
    def is_endgame(self, board):
        return sum(PIECE_VALUES.get(type(p), 0) for row in board.grid for p in row if p and not isinstance(p, King)) < ENDGAME_MATERIAL_THRESHOLD
    
    def evaluate_board(self, board, turn_to_move):
        score = 0
        in_endgame = self.is_endgame(board)
        for r in range(ROWS):
            for c in range(COLS):
                piece = board.grid[r][c]
                if not piece: continue
                value = PIECE_VALUES.get(type(piece), 0)
                
                pst = PIECE_SQUARE_TABLES.get(type(piece))
                if isinstance(piece, King):
                    pst = PIECE_SQUARE_TABLES['king_endgame'] if in_endgame else PIECE_SQUARE_TABLES['king_midgame']

                if pst: 
                    value += pst[r][c] if piece.color == 'white' else pst[ROWS - 1 - r][c]
                
                if piece.color == turn_to_move:
                    score += value
                else:
                    score -= value
        return score

# -----------------------------------------------------------------------------
# Piece-Square Tables (PSTs) and Material Values for this Variant
# -----------------------------------------------------------------------------

PIECE_VALUES = {
    Pawn: 100, Knight: 800, Bishop: 700,
    Rook: 650, Queen: 900, King: 30000
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