# v78.3 (Optimized Tapered Map & Inlined Game State Check)

import time
from GameLogic import generate_legal_moves_generator
from GameLogic import *
import random
from collections import namedtuple

# --- Tapered Piece Values from current baseline ---
PIECE_VALUES_MG = {
    Pawn: 100, Knight: 850, Bishop: 700, Rook: 600, Queen: 900, King: 20000
}
PIECE_VALUES_EG = {
    Pawn: 100, Knight: 800, Bishop: 650, Rook: 700, Queen: 700, King: 20000
}
INITIAL_PHASE_MATERIAL = (PIECE_VALUES_MG[Rook] * 4 + PIECE_VALUES_MG[Knight] * 4 + PIECE_VALUES_MG[Bishop] * 4 + PIECE_VALUES_MG[Queen] * 2)


# Zobrist/TT setup is unchanged
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
    # Optimized in v78.2 - Kept here
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

    # --- OPTIMIZATION 1: No List Concatenation ---
    def _calculate_tapered_map(self, board):
        phase_material_score = 0
        # Iterate lists separately to avoid creating a temporary list with +
        for p in board.white_pieces:
            if not isinstance(p, (Pawn, King)):
                phase_material_score += PIECE_VALUES_MG.get(type(p), 0)
        for p in board.black_pieces:
            if not isinstance(p, (Pawn, King)):
                phase_material_score += PIECE_VALUES_MG.get(type(p), 0)

        phase = min(256, (phase_material_score * 256) // INITIAL_PHASE_MATERIAL) if INITIAL_PHASE_MATERIAL > 0 else 0
        
        tapered_map = {
            ptype: (vals_mg * phase + PIECE_VALUES_EG[ptype] * (256 - phase)) >> 8
            for ptype, vals_mg in PIECE_VALUES_MG.items()
        }
        return tapered_map, phase

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
        
        tapered_vals_by_type, _ = self._calculate_tapered_map(self.board)
        ordered_root_moves = self.order_moves(self.board, root_moves, 0, pv_move, tapered_vals_by_type)
        
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

    def order_moves(self, board, moves, ply, hash_move, tapered_vals_by_type):
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
                    swing = calculate_material_swing(board, move, tapered_vals_by_type)
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
        if ply > 0 and hash_val in search_path: return self.DRAW_SCORE
        
        # --- OPTIMIZATION 2: INLINED Game State Checks ---
        # Replaces `get_game_state` to avoid redundant `has_legal_moves` generation.
        
        # 1. Check for simple draws first (fastest)
        if is_insufficient_material(board):
             return self.DRAW_SCORE
             
        # 2. Check for Repetition (using existing hash)
        if self.position_counts.get(hash_val, 0) >= 3:
             return self.DRAW_SCORE

        # 3. Check for Move Limit
        if self.ply_count + ply >= self.max_moves:
            return self.DRAW_SCORE

        # 4. Transposition Table Lookup
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

        # 5. Generate Moves & Check for Mate/Stalemate
        legal_moves_with_boards = list(generate_legal_moves_generator(board, turn, yield_boards=True))
        
        if not legal_moves_with_boards:
            # If no moves, we are in Mate or Stalemate.
            if is_in_check_flag: return -self.MATE_SCORE + ply
            return self.DRAW_SCORE

        legal_moves_list = [move for move, _ in legal_moves_with_boards]
        move_to_board_map = dict(legal_moves_with_boards)

        hash_move = tt_entry.best_move if tt_entry else None
        tapered_vals_by_type, _ = self._calculate_tapered_map(board)
        ordered_moves = self.order_moves(board, legal_moves_list, ply, hash_move, tapered_vals_by_type)
        
        tactical_moves_set = set(generate_all_tactical_moves(board, turn))
        best_move_for_node = None
        
        for i, move in enumerate(ordered_moves):
            child_board = move_to_board_map.get(move)
            if not child_board: continue

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

        # Note: We keep get_game_state here because qsearch doesn't generate all legal moves so we need it to reliably detect game-ending conditions if necessary.
        game_status, _ = get_game_state(board, turn, self.position_counts, self.ply_count + ply, self.max_moves)
        if game_status != "ongoing":
            if game_status == "checkmate": return -self.MATE_SCORE + ply
            return self.DRAW_SCORE

        if ply >= self.MAX_Q_SEARCH_DEPTH:
            score, _ = self.evaluate_board(board, turn)
            return score
        
        stand_pat, tapered_vals_by_type = self.evaluate_board(board, turn)
        if not tapered_vals_by_type: return self.DRAW_SCORE
        
        is_in_check_flag = is_in_check(board, turn)
        if not is_in_check_flag:
            if stand_pat >= beta: return beta
            alpha = max(alpha, stand_pat)

        promising_moves = get_all_legal_moves(board, turn) if is_in_check_flag else list(generate_all_tactical_moves(board, turn))
        
        scored_moves = []
        for move in promising_moves:
            swing = calculate_material_swing(board, move, tapered_vals_by_type)
            scored_moves.append((swing, move))
        scored_moves.sort(key=lambda item: item[0], reverse=True)

        for swing, move in scored_moves:
            if not is_in_check_flag and stand_pat + swing + self.Q_SEARCH_SAFETY_MARGIN < alpha:
                continue
            sim_board = board.clone()
            sim_board.make_move(move[0], move[1])
            if is_in_check(sim_board, turn): continue
            
            search_score = -self.qsearch(sim_board, -beta, -alpha, ('black' if turn == 'white' else 'white'), ply + 1)
            if search_score >= beta: return beta
            alpha = max(alpha, search_score)
            
        if is_in_check_flag and alpha < -self.MATE_SCORE + 100:
             return -self.MATE_SCORE + ply

        return alpha

    def evaluate_board(self, board, turn_to_move):
        if is_insufficient_material(board):
            return self.DRAW_SCORE, {}
        
        tapered_vals_by_type, phase = self._calculate_tapered_map(board)

        white_pieces, black_pieces, grid = board.white_pieces, board.black_pieces, board.grid
        QUEEN_AOE_POTENTIAL_BONUS, ROOK_PIERCING_POTENTIAL_BONUS = 25, 20
        white_pst_mg, white_pst_eg, black_pst_mg, black_pst_eg = 0, 0, 0, 0

        for piece in white_pieces:
            piece_type = type(piece)
            r, c = piece.pos
            mg_val, eg_val = PIECE_VALUES_MG.get(piece_type, 0), PIECE_VALUES_EG.get(piece_type, 0)
            white_pst_mg += mg_val
            white_pst_eg += eg_val
            if piece_type is King:
                white_pst_mg += PIECE_SQUARE_TABLES['king_midgame'][r][c]
                white_pst_eg += PIECE_SQUARE_TABLES['king_endgame'][r][c]
            elif PIECE_SQUARE_TABLES.get(piece_type):
                pst_val = PIECE_SQUARE_TABLES[piece_type][r][c]
                white_pst_mg += pst_val; white_pst_eg += pst_val
            if piece_type is Queen:
                for adj_r, adj_c in ADJACENT_SQUARES_MAP.get(piece.pos, set()):
                    target = grid[adj_r][adj_c]
                    if target and target.color == 'black': white_pst_mg += QUEEN_AOE_POTENTIAL_BONUS
            elif piece_type is Rook:
                for dr, dc in DIRECTIONS['rook']:
                    for i in range(1, 8):
                        nr, nc = r + dr*i, c + dc*i
                        if not (0 <= nr < ROWS and 0 <= nc < COLS): break
                        target = grid[nr][nc]
                        if target and target.color == 'black': white_pst_mg += ROOK_PIERCING_POTENTIAL_BONUS
            elif piece_type is Knight:
                knight_tapered_value = tapered_vals_by_type[Knight]
                for threatened_pos in KNIGHT_ATTACKS_FROM.get(piece.pos, set()):
                    target = grid[threatened_pos[0]][threatened_pos[1]]
                    if target and target.color == 'black':
                        target_val = tapered_vals_by_type[type(target)]
                        bonus = (target_val - knight_tapered_value) if type(target) is Knight else target_val
                        white_pst_mg += bonus // 4

        for piece in black_pieces:
            piece_type = type(piece)
            r, c = piece.pos
            r_flipped = ROWS - 1 - r
            mg_val, eg_val = PIECE_VALUES_MG.get(piece_type, 0), PIECE_VALUES_EG.get(piece_type, 0)
            black_pst_mg += mg_val
            black_pst_eg += eg_val
            if piece_type is King:
                black_pst_mg += PIECE_SQUARE_TABLES['king_midgame'][r_flipped][c]
                black_pst_eg += PIECE_SQUARE_TABLES['king_endgame'][r_flipped][c]
            elif PIECE_SQUARE_TABLES.get(piece_type):
                pst_val = PIECE_SQUARE_TABLES[piece_type][r_flipped][c]
                black_pst_mg += pst_val; black_pst_eg += pst_val
            if piece_type is Queen:
                for adj_r, adj_c in ADJACENT_SQUARES_MAP.get(piece.pos, set()):
                    target = grid[adj_r][adj_c]
                    if target and target.color == 'white': black_pst_mg += QUEEN_AOE_POTENTIAL_BONUS
            elif piece_type is Rook:
                for dr, dc in DIRECTIONS['rook']:
                    for i in range(1, 8):
                        nr, nc = r + dr*i, c + dc*i
                        if not (0 <= nr < ROWS and 0 <= nc < COLS): break
                        target = grid[nr][nc]
                        if target and target.color == 'white': black_pst_mg += ROOK_PIERCING_POTENTIAL_BONUS
            elif piece_type is Knight:
                knight_tapered_value = tapered_vals_by_type[Knight]
                for threatened_pos in KNIGHT_ATTACKS_FROM.get(piece.pos, set()):
                    target = grid[threatened_pos[0]][threatened_pos[1]]
                    if target and target.color == 'white':
                        target_val = tapered_vals_by_type[type(target)]
                        bonus = (target_val - knight_tapered_value) if type(target) is Knight else target_val
                        black_pst_mg += bonus // 4

        mg_score = white_pst_mg - black_pst_mg
        eg_score = white_pst_eg - black_pst_eg
        final_score = (mg_score * phase + eg_score * (256 - phase)) >> 8
        score_for_player = final_score if turn_to_move == 'white' else -final_score
        return score_for_player, tapered_vals_by_type

# --- Piece-Square Tables (PSTs) ---
pawn_pst = [
    [  0,   0,   0,   0,   0,   0,   0,   0],
    [ 90,  90,  90,  90,  90,  90,  90,  90],
    [ 50,  50,  50,  50,  50,  50,  50,  50],
    [ 30,  30,  40,  50,  50,  40,  30,  30],
    [ 20,  20,  30,  40,  40,  30,  20,  20],
    [ 10,  10,  20,  30,  30,  20,  10,  10],
    [  0,   0,   0,   0,   0,   0,   0,   0],
    [  0,   0,   0,   0,   0,   0,   0,   0]
]
knight_pst = [
    [-50, -40, -30, -30, -30, -30, -40, -50],
    [-40, -20,   5,  10,  10,   5, -20, -40],
    [-30,   5,  20,  25,  25,  20,   5, -30],
    [-30,  10,  25,  35,  35,  25,  10, -30],
    [-30,  10,  25,  35,  35,  25,  10, -30],
    [-30,  10,  20,  25,  25,  20,  10, -30],
    [-40, -20,   5,  10,  10,   5, -20, -40],
    [-50, -45, -30, -30, -30, -30, -45, -50]
]
bishop_pst = [
    [-20, -10, -10, -10, -10, -10, -10, -20],
    [-10,   0,   0,   0,   0,   0,   0, -10],
    [-10,   0,   5,  10,  10,   5,   0, -10],
    [-10,   5,   5,  10,  10,   5,   5, -10],
    [-10,   0,  15,  10,  10,  15,   0, -10],
    [-10,  10,  10,  10,  10,  10,  10, -10],
    [-10,   5,   0,   0,   0,   0,   5, -10],
    [-20, -10, -10, -15, -15, -10, -10, -20]
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
    [ -5,   0,  10,  15,  15,  10,   0,  -5],
    [  0,   0,  10,  15,  15,  10,   0,  -5],
    [-10,   5,  15,   5,   5,  15,   5, -10],
    [-10,   0,   5,   0,   0,   0,   5, -10],
    [-20, -10, -10,  -5, -15, -10, -10, -20]
]
king_midgame_pst = [
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-20, -30, -30, -40, -40, -30, -30, -20],
    [-10, -20, -20, -20, -20, -20, -20, -10],
    [ 20,  20,   0,   0,   0,   0,  20,  20],
    [ 20,  30,  10,   0,   5,  10,  30,  20]
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