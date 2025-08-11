# OPAI.py (v51 - Final Qsearch fixes)

import time
from GameLogic import *
import random
from collections import namedtuple
from GameLogic import _generate_legal_moves

# --- Tapered Piece Values for Jungle Chess ---
PIECE_VALUES_MG = {
    Pawn: 100, Knight: 800, Bishop: 700, Rook: 600, Queen: 900, King: 20000
}
PIECE_VALUES_EG = {
    Pawn: 100, Knight: 800, Bishop: 650, Rook: 800, Queen: 700, King: 20000
}
INITIAL_PHASE_MATERIAL = (PIECE_VALUES_MG[Rook] * 4 + PIECE_VALUES_MG[Knight] * 4 +
                          PIECE_VALUES_MG[Bishop] * 4 + PIECE_VALUES_MG[Queen] * 2)


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
    for piece in board.white_pieces + board.black_pieces:
        key = (piece.pos[0], piece.pos[1], type(piece), piece.color)
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
    LMR_DEPTH_THRESHOLD, LMR_MOVE_COUNT_THRESHOLD, LMR_REDUCTION = 3, 4, 1
    NMP_MIN_DEPTH, NMP_BASE_REDUCTION, NMP_DEPTH_DIVISOR = 3, 2, 6
    Q_SEARCH_SAFETY_MARGIN = 400
    
    # Move ordering bonuses
    BONUS_PV_MOVE = 10_000_000
    BONUS_CAPTURE = 8_000_000
    BONUS_QN_TACTIC = 5_000_000 # High priority for Q/N non-captures
    BONUS_KILLER_1 = 4_000_000
    BONUS_KILLER_2 = 3_500_000
    
    def __init__(self, board, color, position_counts, comm_queue, cancellation_event, bot_name="OP Bot"):
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

    def _get_piece_value(self, piece):
        return PIECE_VALUES_MG.get(type(piece), 0)

    def get_attackers(self, board, square, color):
        attackers = []
        piece_list = board.white_pieces if color == 'white' else board.black_pieces
        for p in piece_list:
            if square in p.get_valid_moves(board, p.pos):
                attackers.append(p)
        return attackers

    def see(self, board, move, turn):
        initial_gain = calculate_material_swing(board, move)
        sim_board = board.clone()
        sim_board.make_move(move[0], move[1])
        opponent_color = 'black' if turn == 'white' else 'white'
        attackers = self.get_attackers(sim_board, move[1], opponent_color)
        if not attackers:
            return initial_gain
        best_attacker = min(attackers, key=lambda p: self._get_piece_value(p))
        recapture_move = (best_attacker.pos, move[1])
        return initial_gain - self.see(sim_board, recapture_move, opponent_color)

    def order_moves(self, board, moves, ply, hash_move=None):
        if not moves: return []
        scores = {}
        killers = self.killer_moves[ply] if ply < len(self.killer_moves) else [None, None]
        moving_color = self.color if ply % 2 == 0 else self.opponent_color
        color_index = 0 if moving_color == 'white' else 1
        
        for move in moves:
            score = 0
            if move == hash_move:
                score = self.BONUS_PV_MOVE
            else:
                moving_piece = board.grid[move[0][0]][move[0][1]]
                target_piece = board.grid[move[1][0]][move[1][1]]
                
                if target_piece is not None:
                    # Fast MVV-LVA heuristic for captures
                    score = self.BONUS_CAPTURE + (self._get_piece_value(target_piece) * 10 - self._get_piece_value(moving_piece))
                else:
                    # Quiet moves, prioritized
                    if move in killers:
                        score = self.BONUS_KILLER_1 if move == killers[0] else self.BONUS_KILLER_2
                    # Prioritize non-capturing moves by pieces with high tactical potential
                    elif isinstance(moving_piece, (Queen, Knight)):
                        score = self.BONUS_QN_TACTIC
                    else:
                        from_idx = move[0][0] * COLS + move[0][1]
                        to_idx = move[1][0] * COLS + move[1][1]
                        score = self.history_heuristic_table[color_index][from_idx][to_idx]

            scores[move] = score
        moves.sort(key=lambda m: scores.get(m, 0), reverse=True)
        return moves

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
            self._report_log(f"OP ({self.color}): Search cancelled.")
            self._report_move(None)

    def ponder_indefinitely(self):
        try:
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
        move_to_board_map = {m: b for m, b in _generate_legal_moves(self.board, self.color, yield_boards=True)}

        for i, move in enumerate(ordered_root_moves):
            if self.cancellation_event.is_set(): raise SearchCancelledException()
            
            child_board = move_to_board_map.get(move)
            if not child_board: continue

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
        
        # Check Extension is the only extension needed here.
        if is_in_check_flag:
            depth += 1
        
        if (depth >= self.NMP_MIN_DEPTH and ply > 0 and not is_in_check_flag and
            beta < self.MATE_SCORE - 200 and 
            any(not isinstance(p, (Pawn, King)) for p in (board.white_pieces if turn == 'white' else board.black_pieces))):
            
            nmp_reduction = self.NMP_BASE_REDUCTION + (depth // self.NMP_DEPTH_DIVISOR)
            score = -self.negamax(board, depth - 1 - nmp_reduction, -beta, -beta + 1, opponent_turn, ply + 1, search_path | {hash_val})
            
            if score >= beta:
                self.tt[hash_val] = TTEntry(beta, depth, TT_FLAG_LOWERBOUND, None)
                return beta 

        new_search_path = search_path | {hash_val}
        
        legal_moves_with_boards = list(_generate_legal_moves(board, turn, yield_boards=True))
        
        if not legal_moves_with_boards:
            return -self.MATE_SCORE + ply if is_in_check_flag else self.DRAW_SCORE

        legal_moves_list = [move for move, board in legal_moves_with_boards]
        move_to_board_map = dict(legal_moves_with_boards)

        hash_move = tt_entry.best_move if tt_entry else None
        ordered_moves = self.order_moves(board, legal_moves_list, ply, hash_move)
        
        best_move_for_node = None
        
        for i, move in enumerate(ordered_moves):
            child_board = move_to_board_map.get(move)
            if not child_board: continue

            child_hash = board_hash(child_board, opponent_turn)
            self.position_counts[child_hash] = self.position_counts.get(child_hash, 0) + 1
            
            is_capture = board.grid[move[1][0]][move[1][1]] is not None
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
        if ply >= self.MAX_Q_SEARCH_DEPTH: return self.evaluate_board(board, turn)
        
        is_in_check_flag = is_in_check(board, turn)
        
        # If we are in check, the stand-pat evaluation is not reliable.
        if not is_in_check_flag:
            stand_pat = self.evaluate_board(board, turn)
            if stand_pat >= beta: return beta
            alpha = max(alpha, stand_pat)

        opponent_color = 'black' if turn == 'white' else 'white'
        promising_moves = []

        # --- THE DEFINITIVE TWO-TIERED ARCHITECTURE (Your Superior Design) ---
        if is_in_check_flag:
            # TIER 1 (IN CHECK): Accuracy is paramount. This logic is correct and remains.
            promising_moves = get_all_legal_moves(board, turn)
            if not promising_moves:
                return -self.MATE_SCORE + ply # It's checkmate.
        else:
            # TIER 2 (NOT IN CHECK): Performance is key. We only look for "violent" moves.
            for piece in list(board.white_pieces if turn == 'white' else board.black_pieces):
                for end_pos in piece.get_valid_moves(board, piece.pos):
                    move = (piece.pos, end_pos)

                    # --- THE CRITICAL BUG FIX: Use the VARIANT-AWARE check ---
                    material_swing = calculate_material_swing(board, move)
                    is_capture = material_swing != 0
                    
                    is_promotion = isinstance(piece, Pawn) and (end_pos[0] == 0 or end_pos[0] == ROWS - 1)
                    
                    if is_capture:
                        # Delta Pruning, now correctly using the variant-aware material_swing
                        if stand_pat + material_swing + self.Q_SEARCH_SAFETY_MARGIN < alpha:
                            continue
                        promising_moves.append(move)
                    elif is_promotion:
                        promising_moves.append(move)
        
        # --- Move Ordering & Final Search ---
        # This is now fully variant-aware because the capture list is correct.
        move_scores = {move: self.see(board, move, turn) for move in promising_moves}
        promising_moves.sort(key=lambda m: move_scores.get(m, 0), reverse=True)

        for move in promising_moves:
            # When in check, we must search all escapes, even if they look like material losses.
            # When not in check, we can safely prune moves with a negative SEE score.
            if not is_in_check_flag and move_scores.get(move, 0) < 0:
                 continue
                 
            sim_board = board.clone()
            sim_board.make_move(move[0], move[1])
            
            # Legality is guaranteed by our generation methods.
            search_score = -self.qsearch(sim_board, -beta, -alpha, opponent_color, ply + 1)
            if search_score >= beta:
                return beta
            alpha = max(alpha, search_score)
            
        return alpha
        
    def evaluate_board(self, board, turn_to_move):
        # --- Heuristic Bonuses (Tuned King Mobility) ---
        ROOK_SEMI_OPEN_FILE_BONUS = 30 # Highest value: file with enemy targets
        ROOK_OPEN_FILE_BONUS = 15      # Medium value: file for mobility
        
        # --- THE CHANGE ---
        # New, more conservative bonus for king mobility as requested.
        # It rewards having escape squares but does not over-incentivize an exposed king.
        KING_MOBILITY_BONUS = [0, 30, 45] # Bonus for 0, 1, or 2+ legal moves
        # --- END OF CHANGE ---

        white_material_mg, white_material_eg = 0, 0
        black_material_mg, black_material_eg = 0, 0
        white_pst_mg, white_pst_eg = 0, 0
        black_pst_mg, black_pst_eg = 0, 0
        phase_material_score = 0

        # --- Base Evaluation: Material and Piece-Square Tables ---
        for piece in board.white_pieces:
            piece_type = type(piece)
            r, c = piece.pos
            mg_val = PIECE_VALUES_MG.get(piece_type, 0)
            white_material_mg += mg_val
            white_material_eg += PIECE_VALUES_EG.get(piece_type, 0)
            if not isinstance(piece, (Pawn, King)):
                phase_material_score += mg_val
            if piece_type == King:
                white_pst_mg += PIECE_SQUARE_TABLES['king_midgame'][r][c]
                white_pst_eg += PIECE_SQUARE_TABLES['king_endgame'][r][c]
            else:
                pst_table = PIECE_SQUARE_TABLES.get(piece_type)
                if pst_table:
                    pst = pst_table[r][c]
                    white_pst_mg += pst
                    white_pst_eg += pst

        for piece in board.black_pieces:
            piece_type = type(piece)
            r_flipped = ROWS - 1 - piece.pos[0]
            c = piece.pos[1]
            mg_val = PIECE_VALUES_MG.get(piece_type, 0)
            black_material_mg += mg_val
            black_material_eg += PIECE_VALUES_EG.get(piece_type, 0)
            if not isinstance(piece, (Pawn, King)):
                phase_material_score += mg_val
            if piece_type == King:
                black_pst_mg += PIECE_SQUARE_TABLES['king_midgame'][r_flipped][c]
                black_pst_eg += PIECE_SQUARE_TABLES['king_endgame'][r_flipped][c]
            else:
                pst_table = PIECE_SQUARE_TABLES.get(piece_type)
                if pst_table:
                    pst = pst_table[r_flipped][c]
                    black_pst_mg += pst
                    black_pst_eg += pst

        # --- Fast, Variant-Aware Heuristics ---

        # 1. Rook Piercing Potential (Prioritizing Semi-Open Files)
        white_pawn_files = {p.pos[1] for p in board.white_pieces if isinstance(p, Pawn)}
        black_pawn_files = {p.pos[1] for p in board.black_pieces if isinstance(p, Pawn)}

        for piece in board.white_pieces:
            if isinstance(piece, Rook):
                c = piece.pos[1]
                is_friendly_pawn_on_file = c in white_pawn_files
                is_enemy_pawn_on_file = c in black_pawn_files
                
                if not is_friendly_pawn_on_file and is_enemy_pawn_on_file:
                    white_pst_mg += ROOK_SEMI_OPEN_FILE_BONUS
                elif not is_friendly_pawn_on_file and not is_enemy_pawn_on_file:
                    white_pst_mg += ROOK_OPEN_FILE_BONUS
        
        for piece in board.black_pieces:
            if isinstance(piece, Rook):
                c = piece.pos[1]
                is_friendly_pawn_on_file = c in black_pawn_files
                is_enemy_pawn_on_file = c in white_pawn_files

                if not is_friendly_pawn_on_file and is_enemy_pawn_on_file:
                    black_pst_mg += ROOK_SEMI_OPEN_FILE_BONUS
                elif not is_friendly_pawn_on_file and not is_enemy_pawn_on_file:
                    black_pst_mg += ROOK_OPEN_FILE_BONUS
        
        # 2. King Mobility (Safety and Escape Routes)
        if board.white_king_pos:
            white_king = board.grid[board.white_king_pos[0]][board.white_king_pos[1]]
            if white_king:
                num_moves = len(white_king.get_valid_moves(board, white_king.pos))
                # --- THE CHANGE ---
                # Cap the mobility index at 2 (for 2+ moves) as requested.
                mobility_index = min(num_moves, 2) 
                white_pst_mg += KING_MOBILITY_BONUS[mobility_index]
                # --- END OF CHANGE ---

        if board.black_king_pos:
            black_king = board.grid[board.black_king_pos[0]][board.black_king_pos[1]]
            if black_king:
                num_moves = len(black_king.get_valid_moves(board, black_king.pos))
                # --- THE CHANGE ---
                mobility_index = min(num_moves, 2)
                black_pst_mg += KING_MOBILITY_BONUS[mobility_index]
                # --- END OF CHANGE ---
        
        # --- Final Score Calculation ---
        if INITIAL_PHASE_MATERIAL == 0:
            phase = 0
        else:
            phase = (phase_material_score * 256) // INITIAL_PHASE_MATERIAL
        phase = min(phase, 256)

        mg_score = (white_material_mg + white_pst_mg) - (black_material_mg + black_pst_mg)
        eg_score = (white_material_eg + white_pst_eg) - (black_material_eg + black_pst_eg)
        final_score = (mg_score * phase + eg_score * (256 - phase)) >> 8
        
        return final_score if turn_to_move == 'white' else -final_score


# Piece-Square Tables (PSTs) remain unchanged
pawn_pst = [
    [ 0,  0 , 0,  0,  0,  0,  0,  0],
    [90, 90, 90, 90, 90, 90, 90, 90],
    [50, 50, 50, 50, 50, 50, 50, 50],
    [30, 30, 40, 50, 50, 40, 30, 30],
    [20, 20, 30, 40, 40, 30, 20, 20],
    [10, 10, 20, 30, 30, 20, 10, 10],
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0]
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