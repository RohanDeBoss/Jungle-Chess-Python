# AI.py (v99 - Tuned material values @depth2 & 200pos)

import time
import random
from collections import namedtuple
from GameLogic import *
from TablebaseManager import TablebaseManager

# --- EVALUATION CONSTANTS ---

MG_PIECE_VALUES = {
    Pawn: 100,
    Knight: 900,
    Bishop: 600,
    Rook: 500,
    Queen: 950,
    King: 20000
}

EG_PIECE_VALUES = {
    Pawn: 100,
    Knight: 900,
    Bishop: 550,
    Rook: 800,
    Queen: 1000,
    King: 20000
}

# Move-ordering tactical swing should track the same base scale as MG values.
ORDERING_VALUES = MG_PIECE_VALUES

INITIAL_PHASE_MATERIAL = (MG_PIECE_VALUES[Rook] * 4 + MG_PIECE_VALUES[Knight] * 4 +
                          MG_PIECE_VALUES[Bishop] * 4 + MG_PIECE_VALUES[Queen] * 2)

# --- ZOBRIST HASHING SETUP (Optimized Array Structure) ---
PIECE_TYPE_IDX = {Pawn: 0, Knight: 1, Bishop: 2, Rook: 3, Queen: 4, King: 5}

ZOBRIST_ARRAY = None
ZOBRIST_TURN = None

def initialize_zobrist_table():
    global ZOBRIST_ARRAY, ZOBRIST_TURN
    if ZOBRIST_ARRAY is not None: return
    random.seed(42)
    # Structure: [color (0=white, 1=black)][piece_type][row][col]
    ZOBRIST_ARRAY = [[[[random.getrandbits(64) for _ in range(8)] for _ in range(8)] for _ in range(6)] for _ in range(2)]
    ZOBRIST_TURN = random.getrandbits(64)

initialize_zobrist_table()

def board_hash(board, turn):
    h = 0
    arr = ZOBRIST_ARRAY
    idx = PIECE_TYPE_IDX
    
    for piece in board.white_pieces:
        if piece.pos:
            h ^= arr[0][idx[type(piece)]][piece.pos[0]][piece.pos[1]]
    for piece in board.black_pieces:
        if piece.pos:
            h ^= arr[1][idx[type(piece)]][piece.pos[0]][piece.pos[1]]
            
    if turn == 'black': 
        h ^= ZOBRIST_TURN
    return h

# --- SEARCH STRUCTURES ---
TTEntry = namedtuple('TTEntry', ['score', 'depth', 'flag', 'best_move'])
TT_FLAG_EXACT, TT_FLAG_LOWERBOUND, TT_FLAG_UPPERBOUND = 0, 1, 2

class SearchCancelledException(Exception): pass

class ChessBot:
    search_depth = 6
    MATE_SCORE = 1000000
    DRAW_SCORE = 0
    
    # --- Q-SEARCH SETTINGS ---
    MAX_Q_SEARCH_DEPTH = 10       # Balance of speed vs accuracy for that depth
    Q_MARGIN_MAX = 950            # Start margin (for sacrifices) (Ply 0-4)
    Q_MARGIN_MIN = 250            # End margin (for speed) (Ply 10)
    
    # --- HEURISTICS ---
    LMR_DEPTH_THRESHOLD = 3
    LMR_MOVE_COUNT_THRESHOLD = 4
    LMR_REDUCTION = 1
    NMP_MIN_DEPTH = 3
    NMP_BASE_REDUCTION = 2
    NMP_DEPTH_DIVISOR = 6
    USE_NULL_MOVE_PRUNING = True
    USE_PROBCUT = True
    PROBCUT_MIN_DEPTH = 6
    PROBCUT_REDUCTION = 2
    PROBCUT_MARGIN = 200

    BONUS_PV_MOVE = 10_000_000
    BONUS_CAPTURE = 8_000_000
    BONUS_KILLER_1 = 4_000_000
    BONUS_KILLER_2 = 3_000_000
    BAD_TACTIC_PENALTY = -2_000_000
    OPENING_TOTAL_PIECE_THRESHOLD = 23
    OPENING_BONUS_MAX_PLY = 1
    OPENING_KNIGHT_DEVELOP_BONUS = 100
    OPENING_KNIGHT_CENTER_WEIGHT = 22
    OPENING_PAWN_CENTER_WEIGHT = 10
    OPENING_CENTER_PAWN_BONUS = 28
    OPENING_CENTRAL_FILES = (COLS // 2 - 1, COLS // 2)
    ASP_WINDOW_INIT = 150 
    ASP_MAX_RETRIES = 3 
    
    # Evaluation constants (class-level to avoid reallocation in hot eval loop)
    EVAL_PAWN_PHALANX_BONUS = 5
    EVAL_ROOK_ALIGNMENT_BONUS = 15
    EVAL_PIECE_DOMINANCE_FACTOR = 40
    EVAL_PAIR_BONUS = 20
    EVAL_DOUBLE_ROOK_PENALTY = 15
    EVAL_ROOK_PAWN_SCALING = 5
    KNIGHT_ACTIVITY_BONUS = 12

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
        
        self.tb_manager = TablebaseManager()
        
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
        self.used_heuristic_eval = False
        self.tb_hits = 0
        self.killer_moves = [[None, None] for _ in range(max(200, self.max_moves))]
        self.history_heuristic_table = [[[0 for _ in range(64)] for _ in range(64)] for _ in range(2)]
        # NEW: Counter-Move Heuristic Table [color][prev_from_sq][prev_to_sq]
        self.counter_moves = [[[None for _ in range(64)] for _ in range(64)] for _ in range(2)]
    
    def _store_tt(self, hash_val, score, depth, flag, move):
        existing = self.tt.get(hash_val)
        # Only overwrite if the new search went deeper, or if it's an exact PV match
        if not existing or depth >= existing.depth or flag == TT_FLAG_EXACT:
            # FIX: If the new move is None (e.g. from ProbCut), keep the old best_move!
            best_move = move if move is not None else (existing.best_move if existing else None)
            self.tt[hash_val] = TTEntry(score, depth, flag, best_move)

    def _report_log(self, message): self.comm_queue.put(('log', message))
    def _report_eval(self, score, depth): self.comm_queue.put(('eval', score if self.color == 'white' else -score, depth))
    def _report_move(self, move): self.comm_queue.put(('move', move))
    def _format_move(self, board_before, move):
        if not move: return "None"
        child = board_before.clone()
        child.make_move(move[0], move[1])
        return format_move_san(board_before, child, move)

    def _ordering_tactical_swing(self, board, move, moving_piece, target_piece):
        return fast_approximate_material_swing(board, move, moving_piece, target_piece, ORDERING_VALUES)

    def _is_opening_position(self, board):
        return (len(board.white_pieces) + len(board.black_pieces)) >= self.OPENING_TOTAL_PIECE_THRESHOLD

    def _opening_development_bonus(self, move, moving_piece):
        if moving_piece is None:
            return 0

        (fr, fc), (tr, tc) = move
        from_center = abs(fr - 3.5) + abs(fc - 3.5)
        to_center = abs(tr - 3.5) + abs(tc - 3.5)
        center_delta = from_center - to_center
        bonus = 0

        if isinstance(moving_piece, Knight):
            bonus += int(center_delta * self.OPENING_KNIGHT_CENTER_WEIGHT)
            if (moving_piece.color == 'white' and fr == ROWS - 1) or (moving_piece.color == 'black' and fr == 0):
                bonus += self.OPENING_KNIGHT_DEVELOP_BONUS
            return bonus

        if isinstance(moving_piece, Pawn):
            bonus += int(center_delta * self.OPENING_PAWN_CENTER_WEIGHT)
            if tc in self.OPENING_CENTRAL_FILES:
                bonus += self.OPENING_CENTER_PAWN_BONUS
            return bonus

        return 0

    def _get_root_tb_eval_relative(self):
        root_abs = self.tb_manager.probe(self.board, self.color)
        if root_abs is None:
            return None
        self.tb_hits += 1
        return root_abs if self.color == 'white' else -root_abs

    def _get_best_tablebase_move_with_eval(self):
        root_abs = self.tb_manager.probe(self.board, self.color)
        if root_abs is None and not is_insufficient_material(self.board):
            return None, None

        best_move = None
        best_score = -float('inf')
        
        for move in get_all_legal_moves(self.board, self.color):
            sim = self.board.clone()
            sim.make_move(move[0], move[1])
            
            if not sim.find_king_pos(self.opponent_color):
                return move, self.MATE_SCORE - 1
            if not has_legal_moves(sim, self.opponent_color):
                return move, self.MATE_SCORE - 1
                 
            if len(sim.white_pieces) + len(sim.black_pieces) <= 2:
                score = 0
            else:
                score_abs = self.tb_manager.probe(sim, self.opponent_color)
                if score_abs is None:
                    return None, None
                self.tb_hits += 1
                score = score_abs if self.color == 'white' else -score_abs
                if score > self.MATE_SCORE - 1000: score -= 1
                elif score < -self.MATE_SCORE + 1000: score += 1
            
            if score > best_score or (score == best_score and score == 0 and random.random() > 0.5):
                best_score = score
                best_move = move
                
        return best_move, best_score

    def _run_depth_iteration(self, depth, root_moves, root_hash, pv_move, prev_iter_score=None, alpha_floor=None):
        iter_nodes = 0
        iter_tb_hits = 0
        any_heuristic_eval = False
        use_aspiration = (alpha_floor is None and prev_iter_score is not None and depth >= 2)

        if use_aspiration:
            window = self.ASP_WINDOW_INIT
            alpha_bound = prev_iter_score - window
            beta_bound = prev_iter_score + window
            retries = 0
            while True:
                best_score, best_move = self._search_at_depth(
                    depth, root_moves, root_hash, pv_move,
                    aspiration_window=(alpha_bound, beta_bound)
                )
                iter_nodes += self.nodes_searched
                iter_tb_hits += self.tb_hits
                if self.used_heuristic_eval:
                    any_heuristic_eval = True

                if self.cancellation_event.is_set():
                    raise SearchCancelledException()

                if best_score <= alpha_bound:       
                    alpha_bound -= window
                    window *= 2
                    retries += 1
                elif best_score >= beta_bound:      
                    beta_bound += window
                    window *= 2
                    retries += 1
                else:
                    break

                if retries >= self.ASP_MAX_RETRIES:
                    best_score, best_move = self._search_at_depth(
                        depth, root_moves, root_hash, pv_move, alpha_floor=alpha_floor
                    )
                    iter_nodes += self.nodes_searched
                    iter_tb_hits += self.tb_hits
                    if self.used_heuristic_eval:
                        any_heuristic_eval = True
                    break
        else:
            best_score, best_move = self._search_at_depth(
                depth, root_moves, root_hash, pv_move, alpha_floor=alpha_floor
            )
            iter_nodes = self.nodes_searched
            iter_tb_hits = self.tb_hits
            if self.used_heuristic_eval:
                any_heuristic_eval = True

        self.nodes_searched = iter_nodes
        self.tb_hits = iter_tb_hits
        self.used_heuristic_eval = any_heuristic_eval
        return best_score, best_move

    def _age_history_table(self):
        for color_idx in range(2):
            for from_sq in range(ROWS*COLS):
                for to_sq in range(ROWS*COLS):
                    self.history_heuristic_table[color_idx][from_sq][to_sq] //= 2

    def _get_pv_data(self, max_depth, root_move):
        if not root_move: return [], []
        
        pv_san = []
        pv_raw = []
        current_board = self.board.clone()
        current_turn = self.color
        current_ply = self.ply_count

        seen_hashes = set()
        move = root_move

        for i in range(max_depth):
            if not move: break
            
            san = self._format_move(current_board, move)
            move_num = (current_ply // 2) + 1
            if current_turn == 'white':
                pv_san.append(f"{move_num}. {san}")
            else:
                if i == 0:
                    pv_san.append(f"{move_num}... {san}")
                else:
                    pv_san.append(f"{san}")
            
            pv_raw.append(move)
            
            current_board.make_move(move[0], move[1])
            current_turn = 'black' if current_turn == 'white' else 'white'
            current_ply += 1
            
            h = board_hash(current_board, current_turn)
            if h in seen_hashes: break
            seen_hashes.add(h)
            
            tt_entry = self.tt.get(h)
            if not tt_entry or not tt_entry.best_move: break
            move = tt_entry.best_move
            if not isinstance(move, tuple): break

        return pv_san, pv_raw
    
    def _report_root_tb_solution(self, tb_move, tb_eval, perfect_play=False, emit_move=False):
        if not tb_move: return False
        root_tb_eval = self._get_root_tb_eval_relative()
        display_eval = root_tb_eval if root_tb_eval is not None else tb_eval
        if tb_eval > self.MATE_SCORE - 1000: display_eval = tb_eval

        eval_for_ui = display_eval if self.color == 'white' else -display_eval
        suffix = " (Perfect Play)" if perfect_play else ""
        self._report_log(f"  > {self.bot_name} (TB): {self._format_move(self.board, tb_move)}, Eval={eval_for_ui/100:+.2f}, TBhits={self.tb_hits}{suffix}")
        self._report_eval(display_eval, "TB")
        
        # --- PV BROADCAST ---
        move_num = (self.ply_count // 2) + 1
        prefix = f"{move_num}. " if self.color == 'white' else f"{move_num}... "
        pv_str = prefix + self._format_move(self.board, tb_move)
        self.comm_queue.put(('pv', display_eval, "TB", [pv_str], [tb_move]))
        # --------------------

        if emit_move: self._report_move(tb_move)
        return True

    def make_move(self):
        try:
            self._age_history_table()
            if len(self.board.white_pieces) + len(self.board.black_pieces) <= 4: 
                tb_move, tb_eval = self._get_best_tablebase_move_with_eval()
                if self._report_root_tb_solution(tb_move, tb_eval, emit_move=True): return

            root_moves = get_all_legal_moves(self.board, self.color)
            if not root_moves:
                self._report_move(None)
                return
            
            best_move_overall = root_moves[0]
            prev_iter_score = None
            total_nodes = 0
            root_hash = board_hash(self.board, self.color)
            for current_depth in range(1, self.search_depth + 1):
                iter_start_time = time.time()
                best_score_this_iter, best_move_this_iter = self._run_depth_iteration(
                    current_depth, root_moves, root_hash, best_move_overall, prev_iter_score=prev_iter_score
                )
                
                if not self.cancellation_event.is_set():
                    best_move_overall = best_move_this_iter
                    prev_iter_score = best_score_this_iter
                    total_nodes += self.nodes_searched
                    iter_duration = time.time() - iter_start_time
                    knps = (self.nodes_searched / iter_duration / 1000) if iter_duration > 0 else 0
                    eval_for_ui = best_score_this_iter if self.color == 'white' else -best_score_this_iter
                    move_str = self._format_move(self.board, best_move_this_iter)
                    depth_label = "TB" if not self.used_heuristic_eval else current_depth
                    
                    self._report_log(f"  > {self.bot_name} (D{depth_label}): {move_str}, Eval={eval_for_ui/100:+.2f}, NodesTotal={total_nodes}, KNPS={knps:.1f}, TBhits={self.tb_hits}, Time={iter_duration:.2f}s")
                    self._report_eval(best_score_this_iter, depth_label)

                    # --- PV BROADCAST ---
                    ui_eval = best_score_this_iter if self.color == 'white' else -best_score_this_iter
                    pv_str, pv_raw = self._get_pv_data(current_depth, best_move_this_iter)
                    self.comm_queue.put(('pv', ui_eval, depth_label, pv_str, pv_raw))
                    # --------------------

                    if best_score_this_iter > self.MATE_SCORE - 2000: break
                else:
                    raise SearchCancelledException()
            
            self._report_move(best_move_overall)
        except SearchCancelledException:
            self._report_move(None)

    def ponder_indefinitely(self):
        try:
            self._age_history_table()
            if is_insufficient_material(self.board): return
            if len(self.board.white_pieces) + len(self.board.black_pieces) <= 4:
                tb_move, tb_eval = self._get_best_tablebase_move_with_eval()
                if self._report_root_tb_solution(tb_move, tb_eval, perfect_play=True):
                    while not self.cancellation_event.is_set(): time.sleep(0.1)
                    return

            root_moves = get_all_legal_moves(self.board, self.color)
            if not root_moves: return
            
            best_move_overall = root_moves[0]
            root_hash = board_hash(self.board, self.color)
            tb_alpha_floor = None
            prev_iter_score = None
            total_nodes = 0
            for current_depth in range(1, 100):
                if self.cancellation_event.is_set(): raise SearchCancelledException()
                iter_start_time = time.time()
                best_score_this_iter, best_move_this_iter = self._run_depth_iteration(
                    current_depth, root_moves, root_hash, best_move_overall,
                    prev_iter_score=prev_iter_score, alpha_floor=tb_alpha_floor
                )
                
                if not self.cancellation_event.is_set():
                    best_move_overall = best_move_this_iter
                    prev_iter_score = best_score_this_iter
                    total_nodes += self.nodes_searched
                    iter_duration = time.time() - iter_start_time
                    knps = (self.nodes_searched / iter_duration / 1000) if iter_duration > 0 else 0
                    eval_for_ui = best_score_this_iter if self.color == 'white' else -best_score_this_iter
                    depth_label = "TB" if not self.used_heuristic_eval else current_depth
                    
                    self._report_log(f"  > {self.bot_name} (D{depth_label}): {self._format_move(self.board, best_move_this_iter)}, Eval={eval_for_ui/100:+.2f}, NodesTotal={total_nodes}, KNPS={knps:.1f}, TBhits={self.tb_hits}, Time={iter_duration:.2f}s")
                    self._report_eval(best_score_this_iter, depth_label)

                    # --- PV BROADCAST ---
                    ui_eval = best_score_this_iter if self.color == 'white' else -best_score_this_iter
                    pv_str, pv_raw = self._get_pv_data(current_depth, best_move_this_iter)
                    self.comm_queue.put(('pv', ui_eval, depth_label, pv_str, pv_raw))
                    # --------------------

                    if depth_label == "TB":
                        while not self.cancellation_event.is_set(): time.sleep(0.1)
                        return

                    # FIX: Only set the floor if we actually hit a raw Tablebase win, NOT a heuristic/search mate.
                    if best_score_this_iter > self.MATE_SCORE - 1000 and not self.used_heuristic_eval: 
                        tb_alpha_floor = best_score_this_iter
                    else: 
                        tb_alpha_floor = None
                else:
                    raise SearchCancelledException()
        except SearchCancelledException: pass

    def _search_at_depth(self, depth, root_moves, root_hash, pv_move, alpha_floor=None, aspiration_window=None):
        self.nodes_searched = 0
        self.used_heuristic_eval = False
        self.tb_hits = 0
        if alpha_floor is not None:
            best_score_this_iter = alpha_floor
            best_move_this_iter = pv_move if pv_move in root_moves else (root_moves[0] if root_moves else None)
        else:
            best_score_this_iter, best_move_this_iter = -float('inf'), None
        if alpha_floor is not None:
            alpha = alpha_floor
            beta = float('inf')
        elif aspiration_window is not None:
            alpha, beta = aspiration_window
        else:
            alpha = -float('inf')
            beta = float('inf')
        
        ordered_root_moves = self.order_moves(self.board, root_moves, 0, pv_move, self.color)
        
        all_moves_draw = True
        for move in ordered_root_moves:
            if self.cancellation_event.is_set(): raise SearchCancelledException()
            
            child_board = self.board.clone()
            child_board.make_move(move[0], move[1])

            search_path = {root_hash}
            child_hash = board_hash(child_board, self.opponent_color)
            self.position_counts[child_hash] = self.position_counts.get(child_hash, 0) + 1

            try:
                if alpha_floor is not None:
                    # PASS child_hash AND prev_move=move
                    probe_score = -self.negamax(
                        child_board, depth - 1, -(alpha_floor + 1), -alpha_floor,
                        self.opponent_color, 1, search_path, current_hash=child_hash, prev_move=move
                    )
                    if probe_score <= alpha_floor:
                        continue
                    score = -self.negamax(
                        child_board, depth - 1, -beta, -alpha,
                        self.opponent_color, 1, search_path, current_hash=child_hash, prev_move=move
                    )
                else:
                    # PASS child_hash AND prev_move=move
                    score = -self.negamax(child_board, depth - 1, -beta, -alpha, self.opponent_color, 1, search_path, current_hash=child_hash, prev_move=move)
            finally:
                self.position_counts[child_hash] -= 1

            if score != self.DRAW_SCORE: all_moves_draw = False

            if score > best_score_this_iter:
                best_score_this_iter = score
                best_move_this_iter = move
            alpha = max(alpha, best_score_this_iter)
        
        if alpha_floor is None and all_moves_draw:
            best_score_this_iter = self.DRAW_SCORE
        return best_score_this_iter, best_move_this_iter

    def negamax(self, board, depth, alpha, beta, turn, ply, search_path, current_hash=None, prev_move=None):
        self.nodes_searched += 1
        if self.cancellation_event.is_set(): raise SearchCancelledException()

        # --- 1. TABLEBASE / MATE LOOKUPS ---
        if len(board.white_pieces) + len(board.black_pieces) <= 4:
            tb_score_absolute = self.tb_manager.probe(board, turn)
            if tb_score_absolute is not None:
                self.tb_hits += 1
                tb_score = tb_score_absolute if turn == 'white' else -tb_score_absolute
                if tb_score > self.MATE_SCORE - 1000: return tb_score - ply
                elif tb_score < -self.MATE_SCORE + 1000: return tb_score + ply
                return tb_score

        hash_val = current_hash if current_hash is not None else board_hash(board, turn)
        
        if ply > 0:
            if hash_val in search_path or self.position_counts.get(hash_val, 0) >= 3: return self.DRAW_SCORE
        if is_insufficient_material(board) or self.ply_count + ply >= self.max_moves: return self.DRAW_SCORE

        # --- 2. TRANSPOSITION TABLE LOOKUP ---
        original_alpha = alpha
        tt_entry = self.tt.get(hash_val)
        if ply > 0 and tt_entry and tt_entry.depth >= depth:
            tt_score = tt_entry.score
            if tt_score > self.MATE_SCORE - 1000: tt_score -= ply
            elif tt_score < -self.MATE_SCORE + 1000: tt_score += ply

            self.used_heuristic_eval = True

            if tt_entry.flag == TT_FLAG_EXACT: return tt_score
            elif tt_entry.flag == TT_FLAG_LOWERBOUND: alpha = max(alpha, tt_score)
            elif tt_entry.flag == TT_FLAG_UPPERBOUND: beta = min(beta, tt_score)
            if alpha >= beta: return tt_score

        if depth <= 0: return self.qsearch(board, alpha, beta, turn, ply)

        opponent_turn = 'black' if turn == 'white' else 'white'
        is_in_check_flag = is_in_check(board, turn)
        if is_in_check_flag: depth += 1

        path_added = False
        if hash_val not in search_path:
            search_path.add(hash_val); path_added = True

        try:
            # --- 3. PROBCUT (Probability Cut) ---
            if (self.USE_PROBCUT and depth >= self.PROBCUT_MIN_DEPTH and not is_in_check_flag and
                abs(beta) < self.MATE_SCORE - 1000):
                
                shallow_beta = beta + self.PROBCUT_MARGIN
                prob_score = self.negamax(board, depth - self.PROBCUT_REDUCTION, shallow_beta - 1, shallow_beta, turn, ply, search_path, current_hash, prev_move)
                
                if prob_score >= shallow_beta:
                    score_to_store = beta
                    if beta > self.MATE_SCORE - 1000: score_to_store = beta + ply
                    elif beta < -self.MATE_SCORE + 1000: score_to_store = beta - ply
                    self._store_tt(hash_val, score_to_store, depth, TT_FLAG_LOWERBOUND, None)
                    return beta
            
            # --- 4. NULL MOVE PRUNING ---
            if (self.USE_NULL_MOVE_PRUNING and depth >= self.NMP_MIN_DEPTH and ply > 0 and not is_in_check_flag and abs(beta) < self.MATE_SCORE - 1000):
                if (board.piece_counts['white'][Knight] + board.piece_counts['white'][Bishop] + board.piece_counts['white'][Rook] + board.piece_counts['white'][Queen] > 0) and \
                   (board.piece_counts['black'][Knight] + board.piece_counts['black'][Bishop] + board.piece_counts['black'][Rook] + board.piece_counts['black'][Queen] > 0):
                    self.used_heuristic_eval = True
                    static_eval = self.evaluate_board(board, turn)
                    if static_eval >= beta:
                        reduction = self.NMP_BASE_REDUCTION + (depth // self.NMP_DEPTH_DIVISOR)
                        score = -self.negamax(board, depth - 1 - reduction, -beta, -beta + 1, opponent_turn, ply + 1, search_path, current_hash=hash_val ^ ZOBRIST_TURN)
                        if score >= beta: return beta

            # --- 5. MAIN MOVE SEARCH ---
            pseudo_moves = get_all_pseudo_legal_moves(board, turn)
            hash_move = tt_entry.best_move if tt_entry else None
            c_move = None
            if prev_move:
                c_move = self.counter_moves[0 if turn == 'white' else 1][prev_move[0][0]*8+prev_move[0][1]][prev_move[1][0]*8+prev_move[1][1]]

            ordered_entries = self.order_moves(board, pseudo_moves, ply, hash_move, turn, return_meta=True, counter_move=c_move)
            best_move_for_node = None
            legal_moves_count = 0
            quiet_moves_tried = []

            for move, meta in ordered_entries:
                is_good_tactic, moving_piece = meta
                child_board = board.clone()
                child_board.make_move(move[0], move[1])

                if not child_board.find_king_pos(opponent_turn): return self.MATE_SCORE - ply
                if is_in_check(child_board, turn): continue

                legal_moves_count += 1
                if not is_good_tactic: quiet_moves_tried.append(move)

                reduction = 0
                if depth >= 3 and legal_moves_count > 1 and not is_in_check_flag and not is_good_tactic:
                    reduction = 1
                    f_sq, t_sq = move[0][0]*8+move[0][1], move[1][0]*8+move[1][1]
                    if self.history_heuristic_table[0 if turn == 'white' else 1][f_sq][t_sq] < 0:
                        reduction += 1
                    if depth >= 8: reduction += 1

                child_hash = board_hash(child_board, opponent_turn)
                search_depth = max(0, depth - 1 - reduction)

                if legal_moves_count == 1:
                    score = -self.negamax(child_board, search_depth, -beta, -alpha, opponent_turn, ply + 1, search_path, current_hash=child_hash, prev_move=move)
                else:
                    score = -self.negamax(child_board, search_depth, -(alpha + 1), -alpha, opponent_turn, ply + 1, search_path, current_hash=child_hash, prev_move=move)
                    if score > alpha and score < beta:
                        score = -self.negamax(child_board, depth - 1, -beta, -alpha, opponent_turn, ply + 1, search_path, current_hash=child_hash, prev_move=move)

                if score > alpha:
                    alpha, best_move_for_node = score, move
                
                if alpha >= beta:
                    if not is_good_tactic:
                        if ply < len(self.killer_moves) and self.killer_moves[ply][0] != move:
                            self.killer_moves[ply][1], self.killer_moves[ply][0] = self.killer_moves[ply][0], move
                        if prev_move:
                            self.counter_moves[0 if turn == 'white' else 1][prev_move[0][0]*8+prev_move[0][1]][prev_move[1][0]*8+prev_move[1][1]] = move
                        if moving_piece:
                            c_idx = 0 if turn == 'white' else 1
                            bonus = depth * depth
                            f_idx, t_idx = move[0][0]*8+move[0][1], move[1][0]*8+move[1][1]
                            self.history_heuristic_table[c_idx][f_idx][t_idx] = min(2_000_000, self.history_heuristic_table[c_idx][f_idx][t_idx] + bonus)
                            for f_move in quiet_moves_tried:
                                if f_move != move:
                                    ff, ft = f_move[0][0]*8+f_move[0][1], f_move[1][0]*8+f_move[1][1]
                                    self.history_heuristic_table[c_idx][ff][ft] = max(-2_000_000, self.history_heuristic_table[c_idx][ff][ft] - bonus)

                    score_to_store = beta
                    if beta > self.MATE_SCORE - 1000: score_to_store = beta + ply
                    elif beta < -self.MATE_SCORE + 1000: score_to_store = beta - ply
                    self._store_tt(hash_val, score_to_store, depth, TT_FLAG_LOWERBOUND, move)
                    return beta

            if legal_moves_count == 0: return -self.MATE_SCORE + ply
            
            score_to_store = alpha
            if alpha > self.MATE_SCORE - 1000: score_to_store = alpha + ply
            elif alpha < -self.MATE_SCORE + 1000: score_to_store = alpha - ply
            flag = TT_FLAG_EXACT if alpha > original_alpha else TT_FLAG_UPPERBOUND
            self._store_tt(hash_val, score_to_store, depth, flag, best_move_for_node)
            return alpha
        finally:
            if path_added: search_path.remove(hash_val)

    def qsearch(self, board, alpha, beta, turn, ply):
        self.nodes_searched += 1
        if self.cancellation_event.is_set(): raise SearchCancelledException()

        # 1. Tablebase / Material / Depth Termination
        if len(board.white_pieces) + len(board.black_pieces) <= 4:
            tb_score_absolute = self.tb_manager.probe(board, turn)
            if tb_score_absolute is not None:
                self.tb_hits += 1
                tb_score = tb_score_absolute if turn == 'white' else -tb_score_absolute
                if tb_score > self.MATE_SCORE - 1000: return tb_score - ply
                elif tb_score < -self.MATE_SCORE + 1000: return tb_score + ply
                return tb_score

        if is_insufficient_material(board): return self.DRAW_SCORE
        
        # Termination at your requested Depth 10
        if ply >= self.MAX_Q_SEARCH_DEPTH:
            self.used_heuristic_eval = True
            return self.evaluate_board(board, turn)

        # 2. Stand Pat Evaluation
        self.used_heuristic_eval = True
        stand_pat = self.evaluate_board(board, turn)
        is_in_check_flag = is_in_check(board, turn)

        if not is_in_check_flag:
            if stand_pat >= beta: return beta
            alpha = max(alpha, stand_pat)

        # 3. Dynamic Margin Logic restored
        if ply <= 4:
            current_margin = self.Q_MARGIN_MAX
        else:
            # 700 point total drop over 6 steps = ~117 per ply
            current_margin = max(self.Q_MARGIN_MIN, self.Q_MARGIN_MAX - (ply - 4) * 117)

        # 4. Move Generation
        if is_in_check_flag:
            promising_moves = list(get_all_pseudo_legal_moves(board, turn))
        else:
            promising_moves = list(generate_all_tactical_moves(board, turn))

        # 5. Move Scoring and Pruning
        scored_moves = []
        for move in promising_moves:
            moving_piece = board.grid[move[0][0]][move[0][1]]
            target_piece = board.grid[move[1][0]][move[1][1]]
            swing = self._ordering_tactical_swing(board, move, moving_piece, target_piece)
            
            # Skip moves with negative tactical value if not in check
            if not is_in_check_flag and swing < 0:
                continue
            
            # Use dynamic current_margin instead of the old constant
            if not is_in_check_flag and (stand_pat + swing + current_margin < alpha):
                continue

            scored_moves.append((swing, move))

        # 6. Sort moves for efficiency
        scored_moves.sort(key=lambda item: item[0], reverse=True)

        # 7. Recursive Search Loop
        legal_moves_count = 0
        opponent_turn = 'black' if turn == 'white' else 'white'
        
        for swing, move in scored_moves:
            sim_board = board.clone()
            sim_board.make_move(move[0], move[1])

            # Immediate win detection
            if not sim_board.find_king_pos(opponent_turn):
                return self.MATE_SCORE - ply
            
            # Filter out moves that leave own king in check
            if is_in_check(sim_board, turn):
                continue

            legal_moves_count += 1
            
            # Recursive call
            search_score = -self.qsearch(sim_board, -beta, -alpha, opponent_turn, ply + 1)
            
            if search_score >= beta: 
                return beta
            alpha = max(alpha, search_score)

        # 8. Checkmate Detection
        if is_in_check_flag and legal_moves_count == 0:
            return -self.MATE_SCORE + ply

        return alpha

    def order_moves(self, board, moves, ply, hash_move, turn, return_meta=False, counter_move=None):
        if not moves: return [] if return_meta else []
        scored_moves = []
        move_meta = {}
        killers = self.killer_moves[ply] if ply < len(self.killer_moves) else [None, None]
        c_idx = 0 if turn == 'white' else 1
        is_opening = (ply <= self.OPENING_BONUS_MAX_PLY and self._is_opening_position(board))

        for move in moves:
            moving_piece = board.grid[move[0][0]][move[0][1]]
            target_piece = board.grid[move[1][0]][move[1][1]]

            swing = self._ordering_tactical_swing(board, move, moving_piece, target_piece)
            is_capture_or_promo = target_piece is not None or (isinstance(moving_piece, Pawn) and (move[1][0] == 0 or move[1][0] == ROWS - 1))
            # is_tactical removed — was computed but never used
            is_good_tactic = (swing > 0) or (swing == 0 and is_capture_or_promo)

            move_meta[move] = (is_good_tactic, moving_piece)

            if move == hash_move:
                score = self.BONUS_PV_MOVE
            elif (swing != 0) or is_capture_or_promo:
                score = self.BONUS_CAPTURE + swing if swing >= 0 else self.BAD_TACTIC_PENALTY + swing
            elif move in killers:
                score = 4_000_000 if move == killers[0] else 3_000_000
            elif move == counter_move:
                score = 2_000_000
            else:
                score = self.history_heuristic_table[c_idx][move[0][0]*8+move[0][1]][move[1][0]*8+move[1][1]]

            if is_opening: score += self._opening_development_bonus(move, moving_piece)
            scored_moves.append((score, move))

        scored_moves.sort(key=lambda item: item[0], reverse=True)
        return [(move, move_meta[move]) for _, move in scored_moves] if return_meta else [move for _, move in scored_moves]

    def evaluate_board(self, board, turn_to_move):
        if is_insufficient_material(board):
            return self.DRAW_SCORE

        scores_mg = [0, 0]; scores_eg = [0, 0]
        piece_counts = [0, 0]; pawn_counts = [0, 0]; last_piece_type = [None, None]
        rook_counts = [0, 0]; bishop_counts = [0, 0]; knight_counts = [0, 0]; queen_counts = [0, 0]

        king_pos = [board.white_king_pos, board.black_king_pos]
        piece_lists = [board.white_pieces, board.black_pieces]
        grid = board.grid
        phase_material_score = 0

        # Constants pulled from class level to avoid reallocation
        PAWN_PHALANX_BONUS    = self.EVAL_PAWN_PHALANX_BONUS
        ROOK_ALIGNMENT_BONUS  = self.EVAL_ROOK_ALIGNMENT_BONUS
        PIECE_DOMINANCE_FACTOR= self.EVAL_PIECE_DOMINANCE_FACTOR
        PAIR_BONUS            = self.EVAL_PAIR_BONUS
        DOUBLE_ROOK_PENALTY   = self.EVAL_DOUBLE_ROOK_PENALTY
        ROOK_PAWN_SCALING     = self.EVAL_ROOK_PAWN_SCALING
        KNIGHT_ACTIVITY_BONUS = self.KNIGHT_ACTIVITY_BONUS

        # 1. Main Loop
        for color_idx in (0, 1):
            pieces = piece_lists[color_idx]
            is_white = (color_idx == 0)
            my_color_name = 'white' if is_white else 'black'
            enemy_king = king_pos[1 - color_idx]

            for piece in pieces:
                ptype = type(piece); r, c = piece.pos

                if ptype is Pawn:
                    pawn_counts[color_idx] += 1
                elif ptype is not King:
                    piece_counts[color_idx] += 1
                    last_piece_type[color_idx] = ptype
                    phase_material_score += MG_PIECE_VALUES.get(ptype, 0)
                    if ptype is Rook:   rook_counts[color_idx] += 1
                    elif ptype is Bishop: bishop_counts[color_idx] += 1
                    elif ptype is Knight: knight_counts[color_idx] += 1
                    elif ptype is Queen:  queen_counts[color_idx] += 1

                val_mg = MG_PIECE_VALUES[ptype]
                val_eg = EG_PIECE_VALUES[ptype]
                r_pst = r if is_white else 7 - r

                if ptype is King:
                    scores_mg[color_idx] += PIECE_SQUARE_TABLES['king_midgame'][r_pst][c]
                    scores_eg[color_idx] += PIECE_SQUARE_TABLES['king_endgame'][r_pst][c]
                else:
                    scores_mg[color_idx] += val_mg
                    scores_eg[color_idx] += val_eg
                    pst_val = PIECE_SQUARE_TABLES[ptype][r_pst][c]
                    scores_mg[color_idx] += pst_val
                    scores_eg[color_idx] += pst_val

                # Variant Heuristics
                if ptype is Pawn:
                    if (c > 0 and isinstance(grid[r][c-1], Pawn) and grid[r][c-1].color == my_color_name) or \
                    (c < COLS-1 and isinstance(grid[r][c+1], Pawn) and grid[r][c+1].color == my_color_name):
                        scores_mg[color_idx] += PAWN_PHALANX_BONUS

                elif ptype is Rook:
                    if enemy_king and (r == enemy_king[0] or c == enemy_king[1]):
                        scores_mg[color_idx] += ROOK_ALIGNMENT_BONUS

                elif ptype is Knight:
                    for ar, ac in KNIGHT_ATTACKS_FROM[(r, c)]:
                        threatened = grid[ar][ac]
                        if threatened and threatened.color != my_color_name and type(threatened) is not King:
                            scores_mg[color_idx] += KNIGHT_ACTIVITY_BONUS
                            scores_eg[color_idx] += KNIGHT_ACTIVITY_BONUS

        # 2. Global Calculations
        phase = min(256, (phase_material_score * 256) // INITIAL_PHASE_MATERIAL) if INITIAL_PHASE_MATERIAL > 0 else 0
        inv_phase = 256 - phase

        total_pawns_on_board = pawn_counts[0] + pawn_counts[1]

        if piece_counts[0] > piece_counts[1]: scores_eg[0] += PIECE_DOMINANCE_FACTOR // (piece_counts[1] + 1)
        elif piece_counts[1] > piece_counts[0]: scores_eg[1] += PIECE_DOMINANCE_FACTOR // (piece_counts[0] + 1)

        LONE_ROOK_PENALTIES   = [550, 200, 150, 80, 40]
        LONE_BISHOP_PENALTIES = [650, 250, 170, 100, 50]

        for i in (0, 1):
            if pawn_counts[i] < 4:
                penalty = int(-250 * (4 - pawn_counts[i])**2 / 16)
                scores_mg[i] += penalty; scores_eg[i] += penalty

            if piece_counts[i] == 1 and pawn_counts[i] <= 4:
                penalty = 0
                if last_piece_type[i] is Rook:   penalty = LONE_ROOK_PENALTIES[pawn_counts[i]]
                elif last_piece_type[i] is Bishop: penalty = LONE_BISHOP_PENALTIES[pawn_counts[i]]
                if penalty > 0:
                    if i == 0 and scores_eg[0] > scores_eg[1]:
                        scores_eg[0] = max(scores_eg[1], scores_eg[0] - penalty)
                    elif i == 1 and scores_eg[1] > scores_eg[0]:
                        scores_eg[1] = max(scores_eg[0], scores_eg[1] - penalty)

            if bishop_counts[i] >= 2: scores_mg[i] += PAIR_BONUS; scores_eg[i] += PAIR_BONUS
            if knight_counts[i] >= 2: scores_mg[i] += PAIR_BONUS; scores_eg[i] += PAIR_BONUS
            if rook_counts[i] >= 2:   scores_mg[i] -= DOUBLE_ROOK_PENALTY; scores_eg[i] -= DOUBLE_ROOK_PENALTY

            if rook_counts[i] > 0:
                bonus = rook_counts[i] * total_pawns_on_board * ROOK_PAWN_SCALING
                scores_mg[i] += bonus; scores_eg[i] += bonus

        if king_pos[0] and king_pos[1]:
            dist = abs(king_pos[0][0] - king_pos[1][0]) + abs(king_pos[0][1] - king_pos[1][1])
            tropism_penalty = (dist * dist * inv_phase * 50) // 50176
            if scores_eg[0] > scores_eg[1]: scores_eg[0] -= tropism_penalty
            elif scores_eg[1] > scores_eg[0]: scores_eg[1] -= tropism_penalty

        mg_score = scores_mg[0] - scores_mg[1]
        eg_score = scores_eg[0] - scores_eg[1]
        final_score = (mg_score * phase + eg_score * inv_phase) >> 8

        can_force_mate = [True, True]
        for i in (0, 1):
            if (pawn_counts[i] == 0 and knight_counts[i] == 0 and queen_counts[i] == 0 and
                    (rook_counts[i] + bishop_counts[i]) < 2):
                can_force_mate[i] = False

        if final_score > 0 and not can_force_mate[0]: final_score //= 8
        elif final_score < 0 and not can_force_mate[1]: final_score //= 8

        return final_score if turn_to_move == 'white' else -final_score
    

# --- Piece-Square Tables (PSTs) ---
pawn_pst = [
    [  0,   0,   0,   0,   0,   0,   0,   0],
    [ 90,  90,  90,  90,  90,  90,  90,  90],
    [ 50,  50,  50,  50,  55,  50,  50,  50],
    [ 25,  30,  30,  45,  50,  30,  30,  25],
    [ 15,  15,  20,  30,  35,  20,  15,  15],
    [ 10,  10,  20,  25,  30,  20,  10,  10],
    [  0,   0,   0,  -5, -10,   0,   0,   0],
    [  0,   0,   0,   0,   0,   0,   0,   0]
]
knight_pst = [
    [-60, -40, -30, -30, -30, -30, -40, -60],
    [-40, -20,   5,  10,  10,   5, -20, -40],
    [-30,   5,  20,  25,  25,  20,   5, -30],
    [-30,  10,  25,  35,  35,  25,  10, -30],
    [-20,  15,  25,  35,  35,  25,  15, -20],
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
    [-10,  10,  10,   5,   5,  10,  10, -10],
    [-10,   5,   0,   0,   0,   0,   5, -10],
    [-20, -10, -10, -15, -15, -10, -10, -20]
]
rook_pst = [
    [ 10,  10,  10,  15,  15,  10,  10,  10],
    [ 15,  15,  15,  20,  20,  15,  15,  15],
    [  5,   0,   0,   5,   5,   0,   0,   5],
    [  5,   0,   0,   5,   5,   0,   0,   5],
    [  5,   0,   0,   5,   5,   0,   0,   5],
    [  5,   0,   0,   5,   5,   0,   0,   5],
    [  0,   0,   0,   5,   5,   0,   0,   0],
    [  0,   0,   0,  10,  10,   5,   5,   0]
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
    [-20,  10,  10,  10,  20,  10,  10, -20]
]
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