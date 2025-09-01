# v73.99 - Latest v74 minus the eval changes

import time
from GameLogic import generate_legal_moves_generator
from GameLogic import *
import random
from collections import namedtuple

# --- Tapered Piece Values for Jungle Chess ---
PIECE_VALUES_MG = {
    Pawn: 100, Knight: 850, Bishop: 700, Rook: 600, Queen: 900, King: 20000
}
PIECE_VALUES_EG = {
    Pawn: 100, Knight: 800, Bishop: 650, Rook: 750, Queen: 700, King: 20000
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
    Q_SEARCH_SAFETY_MARGIN = 850 # Safe margin for promotions (Queen val - Pawn val + buffer)
    
    BONUS_PV_MOVE = 10_000_000
    BONUS_CAPTURE = 8_000_000
    BONUS_KILLER_1 = 4_000_000
    BONUS_KILLER_2 = 3_500_000
    
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
        if not move: return "None"
        (r1, c1), (r2, c2) = move
        return f"{'abcdefgh'[c1]}{'87654321'[r1]}-{'abcdefgh'[c2]}{'87654321'[r2]}"

    def _get_piece_value(self, piece, board_context):
        if INITIAL_PHASE_MATERIAL == 0:
            phase = 0
        else:
            phase_material_score = sum(PIECE_VALUES_MG.get(type(p), 0) for p in board_context.white_pieces + board_context.black_pieces if not isinstance(p, (Pawn, King)))
            phase = (phase_material_score * 256) // INITIAL_PHASE_MATERIAL
        phase = min(phase, 256)
        mg_val = PIECE_VALUES_MG.get(type(piece), 0)
        eg_val = PIECE_VALUES_EG.get(type(piece), 0)
        return (mg_val * phase + eg_val * (256 - phase)) >> 8

    def order_moves(self, board, moves, ply, hash_move=None, tapered_vals_by_type=None):
        if not moves: return []
        
        # If the pre-computed map isn't provided (e.g., from the root search), calculate it once.
        if tapered_vals_by_type is None:
            all_pieces = board.white_pieces + board.black_pieces
            phase_material_score = sum(PIECE_VALUES_MG.get(type(p), 0) for p in all_pieces if not isinstance(p, (Pawn, King)))
            phase = min(256, (phase_material_score * 256) // INITIAL_PHASE_MATERIAL) if INITIAL_PHASE_MATERIAL > 0 else 0
            tapered_vals_by_type = {
                ptype: (vals_mg * phase + PIECE_VALUES_EG[ptype] * (256 - phase)) >> 8
                for ptype, vals_mg in PIECE_VALUES_MG.items()
            }

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
                    # Use the fast, pre-computed map
                    moving_val = tapered_vals_by_type.get(type(moving_piece), 0)
                    target_val = tapered_vals_by_type.get(type(target_piece), 0)
                    score = self.BONUS_CAPTURE + (target_val * 10 - moving_val)
                else:
                    if move in killers:
                        score = self.BONUS_KILLER_1 if move == killers[0] else self.BONUS_KILLER_2
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
            self._report_log(f"{self.bot_name} ({self.color}): Search cancelled.")
            self._report_move(None)

    def ponder_indefinitely(self):
        try:
            if is_insufficient_material(self.board):
                self._report_log(f"{self.bot_name} ({self.color}): Position is a draw by insufficient material.")
                
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
            
            # AI FIX: Manually create the board for the next state
            child_board = self.board.clone()
            child_board.make_move(move[0], move[1])

            search_path = {root_hash}
            child_hash = board_hash(child_board, self.opponent_color)
            self.position_counts[child_hash] = self.position_counts.get(child_hash, 0) + 1
            
            score = -self.negamax(child_board, depth - 1, -beta, -alpha, self.opponent_color, 1, search_path)
            
            self.position_counts[child_hash] -= 1

            if score != self.DRAW_SCORE:
                all_moves_draw = False

            if score > best_score_this_iter:
                best_score_this_iter = score
                best_move_this_iter = move
            
            alpha = max(alpha, best_score_this_iter)
        
        if all_moves_draw:
            best_score_this_iter = self.DRAW_SCORE

        return best_score_this_iter, best_move_this_iter


    def negamax(self, board, depth, alpha, beta, turn, ply, search_path):
        self.nodes_searched += 1
        if self.cancellation_event.is_set(): raise SearchCancelledException()
        
        hash_val = board_hash(board, turn)
        if ply > 0 and hash_val in search_path:
            return self.DRAW_SCORE
        
        game_status, _ = get_game_state(board, turn, self.position_counts, self.ply_count + ply, self.max_moves)
        if game_status != "ongoing":
            if game_status == "checkmate": return -self.MATE_SCORE + ply
            return self.DRAW_SCORE

        original_alpha = alpha
        tt_entry = self.tt.get(hash_val)

        if ply > 0 and tt_entry and tt_entry.depth >= depth:
            if tt_entry.flag == TT_FLAG_EXACT: return tt_entry.score
            elif tt_entry.flag == TT_FLAG_LOWERBOUND: alpha = max(alpha, tt_entry.score)
            elif tt_entry.flag == TT_FLAG_UPPERBOUND: beta = min(beta, tt_entry.score)
            if alpha >= beta: return tt_entry.score

        if depth <= 0:
            return self.qsearch(board, alpha, beta, turn, ply)

        opponent_turn = 'black' if turn == 'white' else 'white'
        is_in_check_flag = is_in_check(board, turn)
        
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

        # --- PERFORMANCE FIX: Pre-calculate tapered values ONCE per node ---
        # This is used for both move ordering and the LMR check, eliminating all redundancy.
        all_pieces = board.white_pieces + board.black_pieces
        phase_material_score = sum(PIECE_VALUES_MG.get(type(p), 0) for p in all_pieces if not isinstance(p, (Pawn, King)))
        phase = min(256, (phase_material_score * 256) // INITIAL_PHASE_MATERIAL) if INITIAL_PHASE_MATERIAL > 0 else 0
        tapered_vals_by_type = {
            ptype: (vals_mg * phase + PIECE_VALUES_EG[ptype] * (256 - phase)) >> 8
            for ptype, vals_mg in PIECE_VALUES_MG.items()
        }

        legal_moves_with_boards = list(generate_legal_moves_generator(board, turn, yield_boards=True))
        if not legal_moves_with_boards:
            return -self.MATE_SCORE + ply if is_in_check_flag else self.DRAW_SCORE

        legal_moves_list = [move for move, _ in legal_moves_with_boards]
        move_to_board_map = dict(legal_moves_with_boards)

        hash_move = tt_entry.best_move if tt_entry else None
        ordered_moves = self.order_moves(board, legal_moves_list, ply, hash_move, tapered_vals_by_type)
        
        best_move_for_node = None
        
        for i, move in enumerate(ordered_moves):
            child_board = move_to_board_map.get(move)
            if not child_board: continue

            # --- THE REAL OPTIMIZATION ---
            # Use the fast, clone-free swing calculator. This is much faster
            # than the old cloning method and safer than the flawed is_in_check method.
            # It's a simple, robust heuristic for LMR.
            is_capture_like_move = calculate_material_swing(board, move, tapered_vals_by_type) != 0

            reduction = 0
            if (depth >= self.LMR_DEPTH_THRESHOLD and i >= self.LMR_MOVE_COUNT_THRESHOLD and 
                not is_in_check_flag and not is_capture_like_move):
                reduction = self.LMR_REDUCTION

            score = -self.negamax(child_board, depth - 1 - reduction, -beta, -alpha, opponent_turn, ply + 1, search_path | {hash_val})

            if reduction > 0 and score > alpha:
                score = -self.negamax(child_board, depth - 1, -beta, -alpha, opponent_turn, ply + 1, search_path | {hash_val})
            
            if score > alpha:
                alpha = score
                best_move_for_node = move
                
            if alpha >= beta:
                if not is_capture_like_move:
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

        game_status, _ = get_game_state(board, turn, self.position_counts, self.ply_count + ply, self.max_moves)
        if game_status != "ongoing":
            if game_status == "checkmate": return -self.MATE_SCORE + ply
            return self.DRAW_SCORE

        if ply >= self.MAX_Q_SEARCH_DEPTH:
            score, _ = self.evaluate_board(board, turn)
            return score
        
        stand_pat, tapered_vals_by_type = self.evaluate_board(board, turn)
        
        is_in_check_flag = is_in_check(board, turn)
        
        if not is_in_check_flag:
            if stand_pat >= beta: return beta
            alpha = max(alpha, stand_pat)

        opponent_color = 'black' if turn == 'white' else 'white'
        
        if is_in_check_flag:
            promising_moves = get_all_legal_moves(board, turn)
        else:
            promising_moves = list(generate_all_tactical_moves(board, turn))

        if not tapered_vals_by_type:
            return self.DRAW_SCORE

        # Use our fast, authoritative swing calculator for move ordering.
        move_scores = {move: calculate_material_swing(board, move, tapered_vals_by_type) for move in promising_moves}
        promising_moves.sort(key=lambda m: move_scores.get(m, 0), reverse=True)

        for move in promising_moves:
            swing = move_scores.get(move, 0)
            
            # --- THE SAFE PRUNING FIX (DELTA PRUNING) ---
            # Prune only if the move has no realistic chance of raising alpha.
            if not is_in_check_flag and stand_pat + swing + self.Q_SEARCH_SAFETY_MARGIN < alpha:
                continue
                
            sim_board = board.clone()
            sim_board.make_move(move[0], move[1])
            
            if is_in_check(sim_board, turn):
                continue
            
            search_score = -self.qsearch(sim_board, -beta, -alpha, opponent_color, ply + 1)
            if search_score >= beta:
                return beta
            alpha = max(alpha, search_score)
            
        # If in check and no legal moves were found, it's checkmate.
        if is_in_check_flag and alpha < self.MATE_SCORE - 100 and not has_legal_moves(board, turn):
            return -self.MATE_SCORE + ply

        return alpha

    def evaluate_board(self, board, turn_to_move):
        if is_insufficient_material(board):
            # OPTIMIZATION: Return an empty dict as there's no need for tapered values here.
            return self.DRAW_SCORE, {}
        
        white_pieces = board.white_pieces
        black_pieces = board.black_pieces
        grid = board.grid

        QUEEN_AOE_POTENTIAL_BONUS = 25
        ROOK_PIERCING_POTENTIAL_BONUS = 20

        white_pst_mg, white_pst_eg = 0, 0
        black_pst_mg, black_pst_eg = 0, 0
        
        phase_material_score = sum(PIECE_VALUES_MG.get(type(p), 0) for p in white_pieces + black_pieces if not isinstance(p, (Pawn, King)))
        if INITIAL_PHASE_MATERIAL == 0:
            phase = 0
        else:
            phase = min(256, (phase_material_score * 256) // INITIAL_PHASE_MATERIAL)

        # --- This pre-computation is now done once and shared with qsearch ---
        tapered_vals_by_type = {
            ptype: (vals_mg * phase + PIECE_VALUES_EG[ptype] * (256 - phase)) >> 8
            for ptype, vals_mg in PIECE_VALUES_MG.items()
        }

        # --- Material and PST Calculation (White) ---
        for piece in white_pieces:
            piece_type = type(piece)
            r, c = piece.pos
            
            mg_val = PIECE_VALUES_MG.get(piece_type, 0)
            eg_val = PIECE_VALUES_EG.get(piece_type, 0)
            
            white_pst_mg += mg_val
            white_pst_eg += eg_val

            if piece_type is King:
                white_pst_mg += PIECE_SQUARE_TABLES['king_midgame'][r][c]
                white_pst_eg += PIECE_SQUARE_TABLES['king_endgame'][r][c]
            elif PIECE_SQUARE_TABLES.get(piece_type):
                pst_val = PIECE_SQUARE_TABLES[piece_type][r][c]
                white_pst_mg += pst_val
                white_pst_eg += pst_val

            if piece_type is Queen:
                for adj_r, adj_c in ADJACENT_SQUARES_MAP.get(piece.pos, set()):
                    target = grid[adj_r][adj_c]
                    if target and target.color == 'black':
                        white_pst_mg += QUEEN_AOE_POTENTIAL_BONUS
            
            elif piece_type is Rook:
                for dr, dc in DIRECTIONS['rook']:
                    for i in range(1, 8):
                        nr, nc = r + dr*i, c + dc*i
                        if not (0 <= nr < ROWS and 0 <= nc < COLS): break
                        target = grid[nr][nc]
                        if target and target.color == 'black':
                            white_pst_mg += ROOK_PIERCING_POTENTIAL_BONUS

            elif piece_type is Knight:
                knight_tapered_value = tapered_vals_by_type[Knight]
                for threatened_pos in KNIGHT_ATTACKS_FROM.get(piece.pos, set()):
                    target = grid[threatened_pos[0]][threatened_pos[1]]
                    if target and target.color == 'black':
                        target_val = tapered_vals_by_type[type(target)]
                        bonus = (target_val - knight_tapered_value) if type(target) is Knight else target_val
                        white_pst_mg += bonus // 4

        # --- Material and PST Calculation (Black) ---
        for piece in black_pieces:
            piece_type = type(piece)
            r, c = piece.pos
            r_flipped = ROWS - 1 - r
            
            mg_val = PIECE_VALUES_MG.get(piece_type, 0)
            eg_val = PIECE_VALUES_EG.get(piece_type, 0)

            black_pst_mg += mg_val
            black_pst_eg += eg_val

            if piece_type is King:
                black_pst_mg += PIECE_SQUARE_TABLES['king_midgame'][r_flipped][c]
                black_pst_eg += PIECE_SQUARE_TABLES['king_endgame'][r_flipped][c]
            elif PIECE_SQUARE_TABLES.get(piece_type):
                pst_val = PIECE_SQUARE_TABLES[piece_type][r_flipped][c]
                black_pst_mg += pst_val
                black_pst_eg += pst_val

            if piece_type is Queen:
                for adj_r, adj_c in ADJACENT_SQUARES_MAP.get(piece.pos, set()):
                    target = grid[adj_r][adj_c]
                    if target and target.color == 'white':
                        black_pst_mg += QUEEN_AOE_POTENTIAL_BONUS

            elif piece_type is Rook:
                for dr, dc in DIRECTIONS['rook']:
                    for i in range(1, 8):
                        nr, nc = r + dr*i, c + dc*i
                        if not (0 <= nr < ROWS and 0 <= nc < COLS): break
                        target = grid[nr][nc]
                        if target and target.color == 'white':
                            black_pst_mg += ROOK_PIERCING_POTENTIAL_BONUS

            elif piece_type is Knight:
                knight_tapered_value = tapered_vals_by_type[Knight]
                for threatened_pos in KNIGHT_ATTACKS_FROM.get(piece.pos, set()):
                    target = grid[threatened_pos[0]][threatened_pos[1]]
                    if target and target.color == 'white':
                        target_val = tapered_vals_by_type[type(target)]
                        bonus = (target_val - knight_tapered_value) if type(target) is Knight else target_val
                        black_pst_mg += bonus // 4

        # --- Final Score Calculation ---
        mg_score = white_pst_mg - black_pst_mg
        eg_score = white_pst_eg - black_pst_eg
        final_score = (mg_score * phase + eg_score * (256 - phase)) >> 8
        
        score_for_player = final_score if turn_to_move == 'white' else -final_score
        
        # --- RETURN BOTH the score and the pre-computed map ---
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