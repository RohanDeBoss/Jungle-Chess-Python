# v83.0 (Refactored Evaluation: Combined White/Black Logic Loop)

import time
from GameLogic import generate_legal_moves_generator
from GameLogic import *
import random
from collections import namedtuple

# --- CONSTANT PIECE VALUES (User Specified) ---
PIECE_VALUES = {
    Pawn: 100, Knight: 800, Bishop: 650, Rook: 600, Queen: 850, King: 20000
}

INITIAL_PHASE_MATERIAL = (PIECE_VALUES[Rook] * 4 + PIECE_VALUES[Knight] * 4 +
                          PIECE_VALUES[Bishop] * 4 + PIECE_VALUES[Queen] * 2)

# Zobrist/TT setup
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
    for piece in board.white_pieces:
        if piece.pos:
            key = (piece.pos[0], piece.pos[1], type(piece), piece.color)
            h ^= ZOBRIST_TABLE.get(key, 0)
    for piece in board.black_pieces:
        if piece.pos:
            key = (piece.pos[0], piece.pos[1], type(piece), piece.color)
            h ^= ZOBRIST_TABLE.get(key, 0)
    if turn == 'black': h ^= ZOBRIST_TABLE['turn']
    return h

TTEntry = namedtuple('TTEntry', ['score', 'depth', 'flag', 'best_move'])
TT_FLAG_EXACT, TT_FLAG_LOWERBOUND, TT_FLAG_UPPERBOUND = 0, 1, 2

class SearchCancelledException(Exception): pass

class ChessBot:
    search_depth = 3
    MATE_SCORE, DRAW_SCORE = 1000000, 0
    MAX_Q_SEARCH_DEPTH = 8
    LMR_DEPTH_THRESHOLD, LMR_MOVE_COUNT_THRESHOLD, LMR_REDUCTION = 3, 4, 1
    NMP_MIN_DEPTH, NMP_BASE_REDUCTION, NMP_DEPTH_DIVISOR = 3, 2, 6
    Q_SEARCH_SAFETY_MARGIN = 850

    BONUS_PV_MOVE = 10_000_000
    BONUS_CAPTURE = 8_000_000
    BONUS_KILLER_1 = 4_000_000
    BONUS_KILLER_2 = 3_000_000
    BONUS_QN_TACTIC = 3_500_000
    
    def __init__(self, board, color, position_counts, comm_queue, cancellation_event, bot_name=None, ply_count=0, game_mode="bot", max_moves=200):
        self.board = board
        self.color = color
        self.opponent_color = 'black' if color == 'white' else 'white'
        self.position_counts = position_counts
        self.comm_queue = comm_queue
        self.cancellation_event = cancellation_event
        self.ply_count = ply_count
        self.game_mode = game_mode
        self.max_moves = max_moves
        
        if bot_name is None:
            if self.__class__.__name__ == "OpponentAI":
                self.bot_name = "OP Bot"
            else:
                self.bot_name = "AI Bot"
        else:
            self.bot_name = bot_name

        self._initialize_search_state()

    def _initialize_search_state(self):
        self.tt = {}
        self.nodes_searched = 0
        self.killer_moves = [[None, None] for _ in range(50)]
        self.history_heuristic_table = [[[0 for _ in range(ROWS*COLS)] for _ in range(ROWS*COLS)] for _ in range(2)]

    def _report_log(self, message): self.comm_queue.put(('log', message))
    def _report_eval(self, score, depth): self.comm_queue.put(('eval', score if self.color == 'white' else -score, depth))
    def _report_move(self, move): self.comm_queue.put(('move', move))
    def _format_move(self, move):
        return format_move(move)

    def make_move(self):
        try:
            root_moves = get_all_legal_moves(self.board, self.color)
            if not root_moves:
                self._report_move(None)
                return
            
            best_move_overall = root_moves[0]
            for current_depth in range(1, self.search_depth + 1):
                iter_start_time = time.time()
                root_hash = board_hash(self.board, self.color)
                best_score_this_iter, best_move_this_iter = self._search_at_depth(current_depth, root_moves, root_hash, best_move_overall)
                
                if not self.cancellation_event.is_set():
                    best_move_overall = best_move_this_iter
                    iter_duration = time.time() - iter_start_time
                    knps = (self.nodes_searched / iter_duration / 1000) if iter_duration > 0 else 0
                    eval_for_ui = best_score_this_iter if self.color == 'white' else -best_score_this_iter
                    move_str = self._format_move(best_move_this_iter)
                    
                    log_msg = f"  > {self.bot_name} (D{current_depth}): {move_str}, Eval={eval_for_ui/100:+.2f}, Nodes={self.nodes_searched}, KNPS={knps:.1f}, Time={iter_duration:.2f}s"
                    
                    self._report_log(log_msg)
                    self._report_eval(best_score_this_iter, None)
                else:
                    raise SearchCancelledException()
            
            self._report_move(best_move_overall)
        except SearchCancelledException:
            self._report_log(f"{self.bot_name} ({self.color}): Search cancelled.")
            self._report_move(None)

    def ponder_indefinitely(self):
        try:
            if is_insufficient_material(self.board):
                self._report_log(f"{self.bot_name} ({self.color}): Position is a draw by insufficient material.")
                return
                
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
                    eval_for_ui = best_score_this_iter if self.color == 'white' else -best_score_this_iter
                    move_str = self._format_move(best_move_this_iter)
                    
                    log_msg = f"  > {self.bot_name} (D{current_depth}): {move_str}, Eval={eval_for_ui/100:+.2f}, Nodes={self.nodes_searched}, KNPS={knps:.1f}, Time={iter_duration:.2f}s"
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
        
        all_moves_draw = True
        for i, move in enumerate(ordered_root_moves):
            if self.cancellation_event.is_set(): raise SearchCancelledException()
            
            child_board = self.board.clone()
            child_board.make_move(move[0], move[1])

            search_path = {root_hash}
            child_hash = board_hash(child_board, self.opponent_color)
            self.position_counts[child_hash] = self.position_counts.get(child_hash, 0) + 1
            
            score = -self.negamax(child_board, depth - 1, -beta, -alpha, self.opponent_color, 1, search_path)
            
            self.position_counts[child_hash] -= 1
            if score != self.DRAW_SCORE: all_moves_draw = False

            if score > best_score_this_iter:
                best_score_this_iter = score
                best_move_this_iter = move
            alpha = max(alpha, best_score_this_iter)
        
        if all_moves_draw: best_score_this_iter = self.DRAW_SCORE
        return best_score_this_iter, best_move_this_iter

    def order_moves(self, board, moves, ply, hash_move):
        if not moves: return []
        scores = {}
        killers = self.killer_moves[ply] if ply < len(self.killer_moves) else [None, None]
        color_index = 0 if (self.color if ply % 2 == 0 else self.opponent_color) == 'white' else 1
        
        for move in moves:
            score = 0
            if move == hash_move:
                score = self.BONUS_PV_MOVE
            else:
                moving_piece = board.grid[move[0][0]][move[0][1]]
                target_piece = board.grid[move[1][0]][move[1][1]]
                
                if target_piece is not None:
                    swing = calculate_material_swing(board, move, PIECE_VALUES)
                    score = self.BONUS_CAPTURE + swing
                else:
                    if move in killers: score = self.BONUS_KILLER_1 if move == killers[0] else self.BONUS_KILLER_2
                    elif isinstance(moving_piece, (Queen, Knight)): score = self.BONUS_QN_TACTIC
                    else:
                        from_idx, to_idx = move[0][0]*COLS+move[0][1], move[1][0]*COLS+move[1][1]
                        score = self.history_heuristic_table[color_index][from_idx][to_idx]
            scores[move] = score
        moves.sort(key=lambda m: scores.get(m, 0), reverse=True)
        return moves

    def negamax(self, board, depth, alpha, beta, turn, ply, search_path):
        self.nodes_searched += 1
        if self.cancellation_event.is_set(): raise SearchCancelledException()
        
        hash_val = board_hash(board, turn)
        
        if ply > 0:
            if hash_val in search_path: return self.DRAW_SCORE
            if self.position_counts.get(hash_val, 0) >= 2: return self.DRAW_SCORE

        if is_insufficient_material(board): return self.DRAW_SCORE
        if self.position_counts.get(hash_val, 0) >= 3: return self.DRAW_SCORE
        if self.ply_count + ply >= self.max_moves: return self.DRAW_SCORE

        original_alpha = alpha
        tt_entry = self.tt.get(hash_val)
        if ply > 0 and tt_entry and tt_entry.depth >= depth:
            if tt_entry.flag == TT_FLAG_EXACT: return tt_entry.score
            elif tt_entry.flag == TT_FLAG_LOWERBOUND: alpha = max(alpha, tt_entry.score)
            elif tt_entry.flag == TT_FLAG_UPPERBOUND: beta = min(beta, tt_entry.score)
            if alpha >= beta: return tt_entry.score

        if depth <= 0: return self.qsearch(board, alpha, beta, turn, ply)

        opponent_turn = 'black' if turn == 'white' else 'white'
        is_in_check_flag = is_in_check(board, turn)
        if is_in_check_flag: depth += 1
        
        if (depth >= self.NMP_MIN_DEPTH and ply > 0 and not is_in_check_flag and
            beta < self.MATE_SCORE - 200 and 
            any(not isinstance(p, (Pawn, King)) for p in (board.white_pieces if turn == 'white' else board.black_pieces))):
            
            nmp_reduction = self.NMP_BASE_REDUCTION + (depth // self.NMP_DEPTH_DIVISOR)
            score = -self.negamax(board, depth - 1 - nmp_reduction, -beta, -beta + 1, opponent_turn, ply + 1, search_path | {hash_val})
            if score >= beta:
                self.tt[hash_val] = TTEntry(beta, depth, TT_FLAG_LOWERBOUND, None)
                return beta 

        legal_moves_list = get_all_legal_moves(board, turn)
        if not legal_moves_list:
            if is_in_check_flag: return -self.MATE_SCORE + ply
            return self.DRAW_SCORE

        hash_move = tt_entry.best_move if tt_entry else None
        ordered_moves = self.order_moves(board, legal_moves_list, ply, hash_move)
        
        tactical_moves_set = set(generate_all_tactical_moves(board, turn))
        best_move_for_node = None
        
        for i, move in enumerate(ordered_moves):
            child_board = board.clone()
            child_board.make_move(move[0], move[1])

            is_tactical = move in tactical_moves_set
            reduction = 0
            if (depth >= self.LMR_DEPTH_THRESHOLD and i >= self.LMR_MOVE_COUNT_THRESHOLD and 
                not is_in_check_flag and not is_tactical):
                reduction = self.LMR_REDUCTION

            score = -self.negamax(child_board, depth - 1 - reduction, -beta, -alpha, opponent_turn, ply + 1, search_path | {hash_val})
            if reduction > 0 and score > alpha:
                score = -self.negamax(child_board, depth - 1, -beta, -alpha, opponent_turn, ply + 1, search_path | {hash_val})
            
            if score > alpha:
                alpha, best_move_for_node = score, move
            if alpha >= beta:
                if not is_tactical:
                    if ply < len(self.killer_moves) and self.killer_moves[ply][0] != move:
                        self.killer_moves[ply][1], self.killer_moves[ply][0] = self.killer_moves[ply][0], move
                    moving_piece = board.grid[move[0][0]][move[0][1]]
                    if moving_piece:
                        color_index = 0 if moving_piece.color == 'white' else 1
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

        game_status, _ = get_game_state(board, turn, self.position_counts, self.ply_count + ply, self.max_moves)
        if game_status != "ongoing":
            if game_status == "checkmate": return -self.MATE_SCORE + ply
            return self.DRAW_SCORE

        if ply >= self.MAX_Q_SEARCH_DEPTH:
            score = self.evaluate_board(board, turn)
            return score
        
        stand_pat = self.evaluate_board(board, turn)
        is_in_check_flag = is_in_check(board, turn)
        if not is_in_check_flag:
            if stand_pat >= beta: return beta
            alpha = max(alpha, stand_pat)

        promising_moves = get_all_legal_moves(board, turn) if is_in_check_flag else list(generate_all_tactical_moves(board, turn))
        
        scored_moves = []
        for move in promising_moves:
            swing = calculate_material_swing(board, move, PIECE_VALUES)
            scored_moves.append((swing, move))
        scored_moves.sort(key=lambda item: item[0], reverse=True)

        for swing, move in scored_moves:
            if not is_in_check_flag and stand_pat + swing + self.Q_SEARCH_SAFETY_MARGIN < alpha: continue
            sim_board = board.clone()
            sim_board.make_move(move[0], move[1])
            if is_in_check(sim_board, turn): continue
            search_score = -self.qsearch(sim_board, -beta, -alpha, ('black' if turn == 'white' else 'white'), ply + 1)
            if search_score >= beta: return beta
            alpha = max(alpha, search_score)
            
        if is_in_check_flag and alpha < -self.MATE_SCORE + 100: return -self.MATE_SCORE + ply
        return alpha

    # --- SIMPLIFIED EVALUATION: SINGLE LOOP FOR BOTH SIDES ---
    def evaluate_board(self, board, turn_to_move):
        if is_insufficient_material(board): return self.DRAW_SCORE
        
        grid = board.grid
        
        # Heuristic Weights
        PAWN_PHALANX_BONUS = 5
        ROOK_ALIGNMENT_BONUS = 15
        PIECE_DOMINANCE_FACTOR = 40
        
        scores = {'white': {'mg': 0, 'eg': 0, 'pawn': 0, 'piece': 0, 'last': None, 'phase': 0}, 
                  'black': {'mg': 0, 'eg': 0, 'pawn': 0, 'piece': 0, 'last': None, 'phase': 0}}

        # 1. Main Loop: Material, PST, Piece Counts, Basic Heuristics
        for color, pieces, enemy_king_pos in [('white', board.white_pieces, board.black_king_pos), 
                                              ('black', board.black_pieces, board.white_king_pos)]:
            stats = scores[color]
            is_white = (color == 'white')
            
            for piece in pieces:
                ptype = type(piece); r, c = piece.pos
                r_idx = r if is_white else ROWS - 1 - r
                
                if ptype is Pawn: stats['pawn'] += 1
                elif ptype is not King: 
                    stats['piece'] += 1; stats['last'] = ptype
                    stats['phase'] += PIECE_VALUES[ptype]

                val = PIECE_VALUES[ptype]
                stats['mg'] += val; stats['eg'] += val
                
                if ptype is King:
                    stats['mg'] += PIECE_SQUARE_TABLES['king_midgame'][r_idx][c]
                    stats['eg'] += PIECE_SQUARE_TABLES['king_endgame'][r_idx][c]
                elif PIECE_SQUARE_TABLES.get(ptype):
                    pst = PIECE_SQUARE_TABLES[ptype][r_idx][c]
                    stats['mg'] += pst; stats['eg'] += pst

                if ptype is Pawn:
                    if (c > 0 and isinstance(grid[r][c-1], Pawn) and grid[r][c-1].color == color) or \
                       (c < COLS-1 and isinstance(grid[r][c+1], Pawn) and grid[r][c+1].color == color):
                        stats['mg'] += PAWN_PHALANX_BONUS
                elif ptype is Rook:
                    if enemy_king_pos and (r == enemy_king_pos[0] or c == enemy_king_pos[1]):
                        stats['mg'] += ROOK_ALIGNMENT_BONUS

        # 2. Relative Heuristics
        w, b = scores['white'], scores['black']
        
        # Piece Dominance
        if w['piece'] > b['piece']: w['eg'] += PIECE_DOMINANCE_FACTOR // (b['piece'] + 1)
        elif b['piece'] > w['piece']: b['eg'] += PIECE_DOMINANCE_FACTOR // (w['piece'] + 1)

        # Pawn Scarcity & Lone Wolf
        for stats in [w, b]:
            if stats['pawn'] < 4:
                penalty = int(-250 * (4 - stats['pawn'])**2 / 16)
                stats['mg'] += penalty; stats['eg'] += penalty
            if stats['piece'] == 1 and stats['last'] in (Rook, Bishop):
                stats['eg'] += min(0, -200 + (stats['pawn'] * 50))

        # King Tropism
        white_king, black_king = board.white_king_pos, board.black_king_pos
        phase = min(256, ((w['phase'] + b['phase']) * 256) // INITIAL_PHASE_MATERIAL) if INITIAL_PHASE_MATERIAL > 0 else 0
        inv_phase = 256 - phase
        
        if white_king and black_king:
            dist = abs(white_king[0] - black_king[0]) + abs(white_king[1] - black_king[1])
            penalty = (dist * dist * inv_phase * 50) // (196 * 256)
            if w['eg'] > b['eg']: w['eg'] -= penalty
            elif b['eg'] > w['eg']: b['eg'] -= penalty

        mg_score = w['mg'] - b['mg']
        eg_score = w['eg'] - b['eg']
        final_score = (mg_score * phase + eg_score * inv_phase) >> 8
        
        return final_score if turn_to_move == 'white' else -final_score

# --- Piece-Square Tables (PSTs) ---
pawn_pst = [
    [  0,   0,   0,   0,   0,   0,   0,   0],
    [ 90,  90,  90,  90,  90,  90,  90,  90],
    [ 50,  50,  50,  50,  50,  50,  50,  50],
    [ 30,  30,  40,  50,  50,  40,  30,  30],
    [ 20,  20,  30,  40,  40,  30,  20,  20],
    [ 10,  10,  20,  30,  30,  20,  10,  10],
    [  0,   0,   0,  -5, -10,   0,   0,   0],
    [  0,   0,   0,   0,   0,   0,   0,   0]
]
knight_pst = [
    [-60, -40, -30, -30, -30, -30, -40, -60],
    [-40, -20,   5,  10,  10,   5, -20, -40],
    [-30,   5,  20,  25,  25,  20,   5, -30],
    [-30,  10,  25,  35,  35,  25,  10, -30],
    [-30,  10,  25,  35,  35,  25,  10, -30],
    [-30,  10,  20,  25,  25,  30,  10, -30],
    [-40, -20,   5,  10,  10,   5, -20, -40],
    [-60, -50, -30, -30, -30, -30, -50, -60]
]
bishop_pst = [
    [-20, -10, -10, -10, -10, -10, -10, -20],
    [-10,   0,   0,   0,   0,   0,   0, -10],
    [-10,   0,   5,  10,  10,   5,   0, -10],
    [-10,   5,   5,  10,  10,   5,   5, -10],
    [-10,   5,  15,  10,  10,  15,   5, -10],
    [-10,  10,  10,  10,  10,  10,  10, -10],
    [-10,   5,   0,   0,   0,   0,   5, -10],
    [-20, -10, -10, -15, -15, -10, -10, -20]
]
rook_pst = [
    [ 10,  10,  10,  10,  10,  10,  10,  10],
    [ 10,  10,  10,  20,  20,  10,  10,  10],
    [  5,   0,   0,   5,   5,   0,   0,   5],
    [  5,   0,   0,   5,   5,   0,   0,   5],
    [  5,   0,   0,   5,   5,   0,   0,   5],
    [  5,   0,   0,   5,   5,   0,   0,   5],
    [  0,   0,   0,   0,   0,   0,   0,   0],
    [  0,   0,   0,   5,   5,   5,   5,   0]
]
queen_pst = [
    [-30, -20, -10, -10, -10, -10, -20, -30],
    [-20,   0,   0,   0,   0,   0,   0, -20],
    [-10,   0,   5,  10,  10,   5,   0, -10],
    [ -5,  10,  20,  30,  30,  20,  10,  -5],
    [ -5,   5,  20,  30,  30,  20,   5,  -5],
    [-10,   5,  15,  15,  15,  15,   5, -10],
    [-20, -10,   0,   5,   5,   0, -10, -20],
    [-30, -20, -20, -10, -20, -20, -20, -30]
]
king_midgame_pst = [
    [-60, -50, -50, -50, -50, -50, -50, -60],
    [-40, -50, -50, -60, -60, -50, -50, -40],
    [-40, -50, -50, -60, -60, -50, -50, -40],
    [-40, -50, -50, -60, -60, -50, -50, -40],
    [-30, -40, -40, -40, -40, -40, -40, -30],
    [-10, -10, -20, -20, -20, -20, -10, -10],
    [ -5,   0,   5,   5,   5,   5,   0,  -5],
    [-20,  10,  20,  30,  30,  20,  10, -20]
]

# Endgame King becomes a mediocre active piece, but still not desperate to rush the center.
king_endgame_pst = [
    [-40, -30, -30, -30, -30, -30, -30, -40], 
    [-30, -10,   0,   0,   0,   0, -10, -30],
    [-30,   0,  10,  20,  20,  10,   0, -30],
    [-30,   5,  20,  20,  20,  20,   5, -30],
    [-30,   5,  15,  20,  20,  15,   5, -30], 
    [-30,   0,  10,  10,  10,  10,   0, -30],
    [-30,   0,   5,   5,   5,   5,   0, -30],
    [-40, -30, -10, -10, -10, -10, -30, -40]
]

PIECE_SQUARE_TABLES = {
    Pawn: pawn_pst, Knight: knight_pst, Bishop: bishop_pst, Rook: rook_pst, 
    Queen: queen_pst, 'king_midgame': king_midgame_pst, 'king_endgame': king_endgame_pst
}