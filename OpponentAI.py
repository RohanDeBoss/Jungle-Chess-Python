# OPAI.py (v98.94 - (Intentionally old) TB repetition fix, Shadowing Fix, Repetition Fix)

import time
import random
from collections import namedtuple
from GameLogic import *
from TablebaseManager import TablebaseManager

# --- TIME CONSTANTS ---
TIME_BUFFER_SEC = 0.50
TIME_BUFFER_PCT = 0.05
MIN_MOVE_TIME   = 0.03

# --- EVALUATION CONSTANTS (Baseline) ---
MG_PIECE_VALUES = {
    Pawn: 100, Knight: 900, Bishop: 650, Rook: 550, Queen: 850, King: 20000
}

EG_PIECE_VALUES = {
    Pawn: 130, Knight: 800, Bishop: 550, Rook: 600, Queen: 850, King: 20000
}

ORDERING_VALUES = MG_PIECE_VALUES

INITIAL_PHASE_MATERIAL = (
    MG_PIECE_VALUES[Rook] * 4 + MG_PIECE_VALUES[Knight] * 4 +
    MG_PIECE_VALUES[Bishop] * 4 + MG_PIECE_VALUES[Queen] * 2
)

# --- ZOBRIST HASHING ---
PIECE_TYPE_IDX = {Pawn: 0, Knight: 1, Bishop: 2, Rook: 3, Queen: 4, King: 5}
ZOBRIST_ARRAY = None
ZOBRIST_TURN = None

def initialize_zobrist_table():
    global ZOBRIST_ARRAY, ZOBRIST_TURN
    if ZOBRIST_ARRAY is not None:
        return
    random.seed(42)
    ZOBRIST_ARRAY = [[[[random.getrandbits(64) for _ in range(8)] for _ in range(8)]
                      for _ in range(6)] for _ in range(2)]
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

class OpponentAI:
    search_depth = 6
    MATE_SCORE = 1000000
    DRAW_SCORE = 0

    MAX_Q_SEARCH_DEPTH = 10
    Q_MARGIN_MAX = 950
    Q_MARGIN_MIN = 250

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

    EVAL_PAWN_PHALANX_BONUS = 5
    EVAL_ROOK_ALIGNMENT_BONUS = 15
    EVAL_PIECE_DOMINANCE_FACTOR = 40
    EVAL_PAIR_BONUS = 20
    EVAL_DOUBLE_ROOK_PENALTY = 15
    EVAL_ROOK_PAWN_SCALING = 5
    KNIGHT_ACTIVITY_BONUS = 12

    def __init__(self, board, color, position_counts, comm_queue, cancellation_event,
                 bot_name=None, ply_count=0, game_mode="bot", max_moves=200,
                 time_left=None, increment=None, use_opening_book=True, use_tablebase=True):

        self.board = board
        self.color = color
        self.opponent_color = 'black' if color == 'white' else 'white'
        self.position_counts = position_counts
        self.comm_queue = comm_queue
        self.cancellation_event = cancellation_event
        self.ply_count = ply_count
        self.game_mode = game_mode
        self.max_moves = max_moves

        # --- TIME MANAGEMENT ---
        self.time_left = time_left
        self.increment = increment
        self.stop_time = None
        self.time_check_mask = 2047
        # -----------------------

        # --- OPENING BOOK FLAG ---
        self.use_opening_book = use_opening_book

        self.tb_manager = TablebaseManager()

        # --- 2. ADD THIS BLOCK ---
        if not use_tablebase:
            # Neuter the probe method so it does nothing and returns None,
            # causing the engine to fall back to its own search.
            self.tb_manager.probe = lambda b, t: None
        # -------------------------

        if bot_name is None:
            self.bot_name = "OP Bot" if self.__class__.__name__ == "OpponentAI" else "AI Bot"
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
        self.counter_moves = [[[None for _ in range(64)] for _ in range(64)] for _ in range(2)]

    def _store_tt(self, hash_val, score, depth, flag, move):
        existing = self.tt.get(hash_val)
        if not existing or depth >= existing.depth or flag == TT_FLAG_EXACT:
            best_move = move if move is not None else (existing.best_move if existing else None)
            self.tt[hash_val] = TTEntry(score, depth, flag, best_move)

    def _report_log(self, message): self.comm_queue.put(('log', message))
    def _report_eval(self, score, depth): self.comm_queue.put(('eval', score if self.color == 'white' else -score, depth))
    def _report_move(self, move): self.comm_queue.put(('move', move))

    def _calc_time_check_mask(self, allocated):
        if allocated <= 0.15: return 31
        if allocated <= 0.30: return 63
        if allocated <= 0.60: return 127
        if allocated <= 1.20: return 255
        if allocated <= 2.50: return 511
        return 2047

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
        if moving_piece is None: return 0
        (fr, fc), (tr, tc) = move
        from_center, to_center = abs(fr - 3.5) + abs(fc - 3.5), abs(tr - 3.5) + abs(tc - 3.5)
        bonus = 0
        if isinstance(moving_piece, Knight):
            bonus += int((from_center - to_center) * self.OPENING_KNIGHT_CENTER_WEIGHT)
            if (moving_piece.color == 'white' and fr == ROWS - 1) or (moving_piece.color == 'black' and fr == 0):
                bonus += self.OPENING_KNIGHT_DEVELOP_BONUS
        elif isinstance(moving_piece, Pawn):
            bonus += int((from_center - to_center) * self.OPENING_PAWN_CENTER_WEIGHT)
            if tc in self.OPENING_CENTRAL_FILES:
                bonus += self.OPENING_CENTER_PAWN_BONUS
        return bonus

    def _get_root_tb_eval_relative(self):
        root_abs = self.tb_manager.probe(self.board, self.color)
        if root_abs is None: return None
        self.tb_hits += 1
        return root_abs if self.color == 'white' else -root_abs

    def _get_best_tablebase_move_with_eval(self):
        if not is_insufficient_material(self.board) and self.tb_manager.probe(self.board, self.color) is None:
            return None, None
        best_move, best_score = None, -float('inf')
        for move in get_all_legal_moves(self.board, self.color):
            sim = self.board.clone()
            sim.make_move(move[0], move[1])
            if not sim.find_king_pos(self.opponent_color) or not has_legal_moves(sim, self.opponent_color):
                return move, self.MATE_SCORE - 1
            if len(sim.white_pieces) + len(sim.black_pieces) <= 2:
                score = 0
            else:
                score_abs = self.tb_manager.probe(sim, self.opponent_color)
                if score_abs is None: return None, None
                self.tb_hits += 1
                score = score_abs if self.color == 'white' else -score_abs
                if score > self.MATE_SCORE - 1000: score -= 1
                elif score < -self.MATE_SCORE + 1000: score += 1
            if score > best_score or (score == best_score and score == 0 and random.random() > 0.5):
                best_score, best_move = score, move
        return best_move, best_score

    def _run_depth_iteration(self, depth, root_moves, root_hash, pv_move, prev_iter_score=None, alpha_floor=None):
        iter_nodes = iter_tb_hits = 0
        any_heuristic_eval = False
        use_aspiration = (alpha_floor is None and prev_iter_score is not None and depth >= 2)

        if use_aspiration:
            window = self.ASP_WINDOW_INIT
            alpha_bound, beta_bound = prev_iter_score - window, prev_iter_score + window
            retries = 0
            while True:
                best_score, best_move = self._search_at_depth(depth, root_moves, root_hash, pv_move, aspiration_window=(alpha_bound, beta_bound))
                iter_nodes += self.nodes_searched; iter_tb_hits += self.tb_hits
                if self.used_heuristic_eval: any_heuristic_eval = True
                if self.cancellation_event.is_set(): raise SearchCancelledException()
                if best_score <= alpha_bound:
                    alpha_bound -= window; window *= 2; retries += 1
                elif best_score >= beta_bound:
                    beta_bound += window; window *= 2; retries += 1
                else: break
                if retries >= self.ASP_MAX_RETRIES:
                    best_score, best_move = self._search_at_depth(depth, root_moves, root_hash, pv_move, alpha_floor=alpha_floor)
                    iter_nodes += self.nodes_searched; iter_tb_hits += self.tb_hits
                    if self.used_heuristic_eval: any_heuristic_eval = True
                    break
        else:
            best_score, best_move = self._search_at_depth(depth, root_moves, root_hash, pv_move, alpha_floor=alpha_floor)
            iter_nodes, iter_tb_hits = self.nodes_searched, self.tb_hits
            if self.used_heuristic_eval: any_heuristic_eval = True

        self.nodes_searched, self.tb_hits, self.used_heuristic_eval = iter_nodes, iter_tb_hits, any_heuristic_eval
        return best_score, best_move

    def _age_history_table(self):
        for c in range(2):
            for f in range(64):
                for t in range(64):
                    self.history_heuristic_table[c][f][t] //= 2

    def _get_pv_data(self, max_depth, root_move):
        if not root_move: return [], []
        pv_san, pv_raw, board, turn, ply, seen = [], [], self.board.clone(), self.color, self.ply_count, set()
        move = root_move
        for i in range(max_depth):
            if not move: break
            move_num = (ply // 2) + 1
            san = self._format_move(board, move)
            pv_san.append(f"{move_num}. {san}" if turn == 'white' else (f"{move_num}... {san}" if i == 0 else f"{san}"))
            pv_raw.append(move)
            board.make_move(move[0], move[1]); turn = 'black' if turn == 'white' else 'white'; ply += 1
            h = board_hash(board, turn)
            if h in seen: break
            seen.add(h)
            tt_entry = self.tt.get(h)
            if not tt_entry or not tt_entry.best_move or not isinstance(tt_entry.best_move, tuple): break
            move = tt_entry.best_move
        return pv_san, pv_raw

    def _report_root_tb_solution(self, tb_move, tb_eval, perfect_play=False, emit_move=False):
        if not tb_move: return False
        root_tb_eval = self._get_root_tb_eval_relative()
        display_eval = root_tb_eval if root_tb_eval is not None else tb_eval
        if tb_eval > self.MATE_SCORE - 1000: display_eval = tb_eval
        eval_ui = display_eval if self.color == 'white' else -display_eval
        suffix = " (Perfect Play)" if perfect_play else ""
        self._report_log(f"  > {self.bot_name} (TB): {self._format_move(self.board, tb_move)}, Eval={eval_ui/100:+.2f}, TBhits={self.tb_hits}{suffix}")
        self._report_eval(display_eval, "TB")
        move_num = (self.ply_count // 2) + 1
        prefix = f"{move_num}. " if self.color == 'white' else f"{move_num}... "
        self.comm_queue.put(('pv', display_eval, "TB", [prefix + self._format_move(self.board, tb_move)], [tb_move]))
        if emit_move: self._report_move(tb_move)
        return True

    def make_move(self):
        try:
            self._age_history_table()
            if len(self.board.white_pieces) + len(self.board.black_pieces) <= 4:
                tb_move, tb_eval = self._get_best_tablebase_move_with_eval()
                if self._report_root_tb_solution(tb_move, tb_eval, emit_move=True): return
            root_moves = get_all_legal_moves(self.board, self.color)
            if not root_moves: self._report_move(None); return
            best_move_overall, prev_iter_score, total_nodes, root_hash = root_moves[0], None, 0, board_hash(self.board, self.color)
            if self.time_left is not None and self.increment is not None:
                alloc = (self.time_left/30.0)+(self.increment*0.8)
                buffer = max(TIME_BUFFER_SEC, self.time_left*TIME_BUFFER_PCT, self.increment*1.5)
                max_alloc = max(0.0, self.time_left - buffer)
                allocated = min(max_alloc, max(MIN_MOVE_TIME, alloc))
                self.stop_time, self.time_check_mask, target_depth = time.time() + allocated, self._calc_time_check_mask(allocated), 100
            else:
                self.stop_time, self.time_check_mask, target_depth = None, 2047, self.search_depth
            for current_depth in range(1, target_depth + 1):
                iter_start_time = time.time()
                best_score_this_iter, best_move_this_iter = self._run_depth_iteration(current_depth, root_moves, root_hash, best_move_overall, prev_iter_score=prev_iter_score)
                if self.cancellation_event.is_set(): raise SearchCancelledException()
                if self.stop_time and time.time() > self.stop_time:
                    best_move_overall = best_move_this_iter; break
                best_move_overall, prev_iter_score, total_nodes = best_move_this_iter, best_score_this_iter, total_nodes + self.nodes_searched
                iter_duration = time.time() - iter_start_time
                knps = (self.nodes_searched / iter_duration / 1000) if iter_duration > 0 else 0
                eval_ui = best_score_this_iter if self.color == 'white' else -best_score_this_iter
                depth_label = "TB" if not self.used_heuristic_eval else current_depth
                self._report_log(f"  > {self.bot_name} (D{depth_label}): {self._format_move(self.board, best_move_this_iter)}, Eval={eval_ui/100:+.2f}, NodesTotal={total_nodes}, KNPS={knps:.1f}, TBhits={self.tb_hits}, Time={iter_duration:.2f}s")
                self._report_eval(best_score_this_iter, depth_label)
                pv_str, pv_raw = self._get_pv_data(current_depth, best_move_this_iter)
                self.comm_queue.put(('pv', eval_ui, depth_label, pv_str, pv_raw))
                if best_score_this_iter > self.MATE_SCORE - 2000: break
            self._report_move(best_move_overall)
        except SearchCancelledException:
            if self.cancellation_event.is_set(): self._report_move(None)
            else: self._report_move(best_move_overall)

    def ponder_indefinitely(self):
        try:
            self._age_history_table()
            if is_insufficient_material(self.board): return
            if len(self.board.white_pieces) + len(self.board.black_pieces) <= 4:
                res = self._get_best_tablebase_move_with_eval()
                if self._report_root_tb_solution(*res, perfect_play=True):
                    while not self.cancellation_event.is_set(): time.sleep(0.1)
                    return
            root_moves = get_all_legal_moves(self.board, self.color)
            if not root_moves: return
            best_move_overall, root_hash, tb_alpha_floor, prev_iter_score, total_nodes = root_moves[0], board_hash(self.board, self.color), None, None, 0
            for current_depth in range(1, 100):
                if self.cancellation_event.is_set(): raise SearchCancelledException()
                iter_start_time = time.time()
                best_score, best_move = self._run_depth_iteration(current_depth, root_moves, root_hash, best_move_overall, prev_iter_score=prev_iter_score, alpha_floor=tb_alpha_floor)
                if not self.cancellation_event.is_set():
                    best_move_overall, prev_iter_score, total_nodes = best_move, best_score, total_nodes + self.nodes_searched
                    iter_duration = time.time() - iter_start_time
                    knps = (self.nodes_searched / iter_duration / 1000) if iter_duration > 0 else 0
                    eval_ui = best_score if self.color == 'white' else -best_score
                    depth_label = "TB" if not self.used_heuristic_eval else current_depth
                    self._report_log(f"  > {self.bot_name} (D{depth_label}): {self._format_move(self.board, best_move)}, Eval={eval_ui/100:+.2f}, NodesTotal={total_nodes}, KNPS={knps:.1f}, TBhits={self.tb_hits}, Time={iter_duration:.2f}s")
                    self._report_eval(best_score, depth_label)
                    pv_str, pv_raw = self._get_pv_data(current_depth, best_move)
                    self.comm_queue.put(('pv', eval_ui, depth_label, pv_str, pv_raw))
                    if depth_label == "TB":
                        while not self.cancellation_event.is_set(): time.sleep(0.1)
                        return
                    tb_alpha_floor = best_score if best_score > self.MATE_SCORE - 1000 and not self.used_heuristic_eval else None
                else: raise SearchCancelledException()
        except SearchCancelledException: pass

    def _search_at_depth(self, depth, root_moves, root_hash, pv_move, alpha_floor=None, aspiration_window=None):
        self.nodes_searched = self.used_heuristic_eval = self.tb_hits = 0
        if alpha_floor is not None:
            best_score, best_move, alpha, beta = alpha_floor, pv_move if pv_move in root_moves else (root_moves[0] if root_moves else None), alpha_floor, float('inf')
        elif aspiration_window:
            best_score, best_move, (alpha, beta) = -float('inf'), None, aspiration_window
        else:
            best_score, best_move, alpha, beta = -float('inf'), None, -float('inf'), float('inf')
        all_moves_draw = True
        for move in self.order_moves(self.board, root_moves, 0, pv_move, self.color):
            if self.cancellation_event.is_set(): raise SearchCancelledException()
            child = self.board.clone(); child.make_move(move[0], move[1])
            search_path, child_hash = {root_hash}, board_hash(child, self.opponent_color)
            if alpha_floor is not None:
                probe = -self.negamax(child, depth - 1, -(alpha_floor + 1), -alpha_floor, self.opponent_color, 1, search_path, child_hash, move)
                if probe <= alpha_floor: continue
                score = -self.negamax(child, depth - 1, -beta, -alpha, self.opponent_color, 1, search_path, child_hash, move)
            else:
                score = -self.negamax(child, depth - 1, -beta, -alpha, self.opponent_color, 1, search_path, child_hash, move)
            if score != self.DRAW_SCORE: all_moves_draw = False
            if score > best_score: best_score, best_move = score, move
            alpha = max(alpha, best_score)
        if alpha_floor is None and all_moves_draw: best_score = self.DRAW_SCORE
        return best_score, best_move

    def negamax(self, board, depth, alpha, beta, turn, ply, search_path, current_hash=None, prev_move=None):
        self.nodes_searched += 1
        if (self.nodes_searched & self.time_check_mask) == 0 and (self.cancellation_event.is_set() or (self.stop_time and time.time() > self.stop_time)):
            raise SearchCancelledException()
            
        # 1. Calculate hash first
        hash_val = current_hash or board_hash(board, turn)
        
        # 2. Check for draws by repetition BEFORE probing the Tablebase
        if ply > 0:
            # FIX: Change threshold from >= 3 to >= 2 for correct draw detection
            if hash_val in search_path or self.position_counts.get(hash_val, 0) >= 2:
                return self.DRAW_SCORE

        # 3. Probe the Tablebase ONLY if it's not a repetition draw
        if len(board.white_pieces) + len(board.black_pieces) <= 4:
            tb_score_abs = self.tb_manager.probe(board, turn)
            if tb_score_abs is not None:
                self.tb_hits += 1; tb_score = tb_score_abs if turn == 'white' else -tb_score_abs
                if tb_score > self.MATE_SCORE-1000: return tb_score-ply
                if tb_score < -self.MATE_SCORE+1000: return tb_score+ply
                return tb_score
            
        # 4. Check for insufficient material or max moves
        if is_insufficient_material(board) or self.ply_count + ply >= self.max_moves: return self.DRAW_SCORE
        
        original_alpha, tt_entry = alpha, self.tt.get(hash_val)
        if ply > 0 and tt_entry and tt_entry.depth >= depth:
            tt_score = tt_entry.score
            if tt_score > self.MATE_SCORE-1000: tt_score -= ply
            elif tt_score < -self.MATE_SCORE+1000: tt_score += ply
            self.used_heuristic_eval = True
            if tt_entry.flag == TT_FLAG_EXACT: return tt_score
            if tt_entry.flag == TT_FLAG_LOWERBOUND: alpha = max(alpha, tt_score)
            elif tt_entry.flag == TT_FLAG_UPPERBOUND: beta = min(beta, tt_score)
            if alpha >= beta: return tt_score
        if depth <= 0: return self.qsearch(board, alpha, beta, turn, ply)
        opp_turn, is_in_check_flag = ('black' if turn == 'white' else 'white'), is_in_check(board, turn)
        if is_in_check_flag: depth += 1
        path_added = False
        if hash_val not in search_path:
            search_path.add(hash_val); path_added = True
        try:
            if self.USE_PROBCUT and depth >= self.PROBCUT_MIN_DEPTH and not is_in_check_flag and abs(beta) < self.MATE_SCORE - 1000:
                shallow_beta = beta + self.PROBCUT_MARGIN
                prob_score = self.negamax(board, depth - self.PROBCUT_REDUCTION, shallow_beta - 1, shallow_beta, turn, ply, search_path, current_hash, prev_move)
                if prob_score >= shallow_beta:
                    sto = beta + ply if beta > self.MATE_SCORE-1000 else (beta - ply if beta < -self.MATE_SCORE+1000 else beta)
                    self._store_tt(hash_val, sto, depth, TT_FLAG_LOWERBOUND, None)
                    return beta
            if self.USE_NULL_MOVE_PRUNING and depth >= self.NMP_MIN_DEPTH and ply > 0 and not is_in_check_flag and abs(beta) < self.MATE_SCORE - 1000:
                pc = board.piece_counts
                if (pc['white'][Knight] + pc['white'][Bishop] + pc['white'][Rook] + pc['white'][Queen] > 0) and (pc['black'][Knight] + pc['black'][Bishop] + pc['black'][Rook] + pc['black'][Queen] > 0):
                    self.used_heuristic_eval, static_eval = True, self.evaluate_board(board, turn)
                    if static_eval >= beta:
                        reduction = self.NMP_BASE_REDUCTION + (depth // self.NMP_DEPTH_DIVISOR)
                        score = -self.negamax(board, depth - 1 - reduction, -beta, -beta + 1, opp_turn, ply + 1, search_path, hash_val ^ ZOBRIST_TURN)
                        if score >= beta: return beta
            pseudo_moves, hash_move = get_all_pseudo_legal_moves(board, turn), tt_entry.best_move if tt_entry else None
            prev_f, prev_t = (prev_move[0][0]*8+prev_move[0][1], prev_move[1][0]*8+prev_move[1][1]) if prev_move else (None, None)
            c_move = self.counter_moves[0 if turn=='white' else 1][prev_f][prev_t] if prev_f is not None else None
            ordered_entries, best_move, legal_moves_count, quiet_moves = self.order_moves(board, pseudo_moves, ply, hash_move, turn, True, c_move), None, 0, []
            for move, meta in ordered_entries:
                is_good, moving_piece = meta
                child_board = board.clone(); child_board.make_move(move[0], move[1])
                if not child_board.find_king_pos(opp_turn): return self.MATE_SCORE-ply
                if is_in_check(child_board, turn): continue
                legal_moves_count += 1
                if not is_good: quiet_moves.append(move)
                reduction = 0
                if depth >= 3 and legal_moves_count > 1 and not is_in_check_flag and not is_good:
                    reduction = 1
                    f,t = move[0][0]*8+move[0][1],move[1][0]*8+move[1][1]
                    if self.history_heuristic_table[0 if turn=='white' else 1][f][t] < 0: reduction += 1
                    if depth >= 8: reduction += 1
                child_hash, search_d = board_hash(child_board, opp_turn), max(0, depth - 1 - reduction)
                if legal_moves_count == 1:
                    score = -self.negamax(child_board, search_d, -beta, -alpha, opp_turn, ply + 1, search_path, child_hash, move)
                else:
                    score = -self.negamax(child_board, search_d, -(alpha + 1), -alpha, opp_turn, ply + 1, search_path, child_hash, move)
                    if alpha < score < beta:
                        score = -self.negamax(child_board, depth - 1, -beta, -alpha, opp_turn, ply + 1, search_path, child_hash, move)
                if score > alpha: alpha, best_move = score, move
                if alpha >= beta:
                    if not is_good:
                        if ply < len(self.killer_moves) and self.killer_moves[ply][0] != move:
                            self.killer_moves[ply][1], self.killer_moves[ply][0] = self.killer_moves[ply][0], move
                        if prev_f is not None:
                            self.counter_moves[0 if turn=='white' else 1][prev_f][prev_t] = move
                        if moving_piece:
                            c_idx, bonus, ht = (0 if turn=='white' else 1), depth*depth, self.history_heuristic_table[0 if turn=='white' else 1]
                            f_idx, t_idx = move[0][0]*8+move[0][1], move[1][0]*8+move[1][1]
                            ht[f_idx][t_idx] = min(2_000_000, ht[f_idx][t_idx] + bonus)
                            for f_move in quiet_moves:
                                if f_move != move:
                                    ff, ft = f_move[0][0]*8+f_move[0][1], f_move[1][0]*8+f_move[1][1]
                                    ht[ff][ft] = max(-2_000_000, ht[ff][ft] - bonus)
                    sto = beta + ply if beta > self.MATE_SCORE-1000 else (beta - ply if beta < -self.MATE_SCORE+1000 else beta)
                    self._store_tt(hash_val, sto, depth, TT_FLAG_LOWERBOUND, move); return beta
            if legal_moves_count == 0: return -self.MATE_SCORE+ply
            sto = alpha + ply if alpha > self.MATE_SCORE-1000 else (alpha - ply if alpha < -self.MATE_SCORE+1000 else alpha)
            self._store_tt(hash_val, sto, depth, TT_FLAG_EXACT if alpha > original_alpha else TT_FLAG_UPPERBOUND, best_move); return alpha
        finally:
            if path_added: search_path.remove(hash_val)

    def qsearch(self, board, alpha, beta, turn, ply):
        self.nodes_searched += 1
        if (self.nodes_searched & self.time_check_mask) == 0 and (self.cancellation_event.is_set() or (self.stop_time and time.time() > self.stop_time)):
            raise SearchCancelledException()
        if len(board.white_pieces)+len(board.black_pieces) <= 4:
            tb_score_abs = self.tb_manager.probe(board, turn)
            if tb_score_abs is not None:
                self.tb_hits += 1; tb_score = tb_score_abs if turn == 'white' else -tb_score_abs
                if tb_score > self.MATE_SCORE-1000: return tb_score-ply
                if tb_score < -self.MATE_SCORE+1000: return tb_score+ply
                return tb_score
        if is_insufficient_material(board): return self.DRAW_SCORE
        if ply >= self.MAX_Q_SEARCH_DEPTH: self.used_heuristic_eval = True; return self.evaluate_board(board, turn)
        self.used_heuristic_eval, stand_pat = True, self.evaluate_board(board, turn)
        is_in_check_flag = is_in_check(board, turn)
        if not is_in_check_flag:
            if stand_pat >= beta: return beta
            alpha = max(alpha, stand_pat)
        margin = self.Q_MARGIN_MAX if ply <= 4 else max(self.Q_MARGIN_MIN, self.Q_MARGIN_MAX - (ply-4)*117)
        promising = list(get_all_pseudo_legal_moves(board, turn)) if is_in_check_flag else list(generate_all_tactical_moves(board, turn))
        scored = []
        for move in promising:
            swing = self._ordering_tactical_swing(board, move, board.grid[move[0][0]][move[0][1]], board.grid[move[1][0]][move[1][1]])
            if not is_in_check_flag and (swing < 0 or stand_pat + swing + margin < alpha): continue
            scored.append((swing, move))
        scored.sort(key=lambda item: item[0], reverse=True)
        legal, opp_turn = 0, 'black' if turn == 'white' else 'white'
        for swing, move in scored:
            sim = board.clone(); sim.make_move(move[0], move[1])
            if not sim.find_king_pos(opp_turn): return self.MATE_SCORE-ply
            if is_in_check(sim, turn): continue
            legal += 1; score = -self.qsearch(sim, -beta, -alpha, opp_turn, ply + 1)
            if score >= beta: return beta
            alpha = max(alpha, score)
        if is_in_check_flag and legal == 0: return -self.MATE_SCORE+ply
        return alpha

    def order_moves(self, board, moves, ply, hash_move, turn, return_meta=False, counter_move=None):
        if not moves: return [] if return_meta else []
        scored, meta = [], {}
        killers, c_idx, is_opening = (self.killer_moves[ply] if ply < len(self.killer_moves) else [None,None]), (0 if turn == 'white' else 1), (ply <= self.OPENING_BONUS_MAX_PLY and self._is_opening_position(board))
        for move in moves:
            moving, target = board.grid[move[0][0]][move[0][1]], board.grid[move[1][0]][move[1][1]]
            swing = self._ordering_tactical_swing(board, move, moving, target)
            is_good = (swing > 0) or (swing == 0 and (target is not None or (isinstance(moving, Pawn) and (move[1][0]==0 or move[1][0]==ROWS-1))))
            meta[move] = (is_good, moving)
            if move == hash_move: score = self.BONUS_PV_MOVE
            elif (swing != 0) or (target is not None or (isinstance(moving, Pawn) and (move[1][0]==0 or move[1][0]==ROWS-1))): score = self.BONUS_CAPTURE + swing if swing >= 0 else self.BAD_TACTIC_PENALTY + swing
            elif move in killers: score = 4_000_000 if move == killers[0] else 3_000_000
            elif move == counter_move: score = 2_000_000
            else: score = self.history_heuristic_table[c_idx][move[0][0]*8+move[0][1]][move[1][0]*8+move[1][1]]
            if is_opening: score += self._opening_development_bonus(move, moving)
            scored.append((score, move))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [(m, meta[m]) for _, m in scored] if return_meta else [m for _, m in scored]

    def evaluate_board(self, board, turn_to_move):
        if is_insufficient_material(board): return self.DRAW_SCORE
        mg, eg, pcs, pawns, last, r_c, b_c, n_c, q_c = [0,0], [0,0], [0,0], [0,0], [None,None], [0,0], [0,0], [0,0], [0,0]
        kings, piece_lists, grid, total_phase = [board.white_king_pos, board.black_king_pos], [board.white_pieces, board.black_pieces], board.grid, 0
        for i in (0,1):
            is_white, my_color, enemy_king = (i==0), ('white' if i==0 else 'black'), kings[1-i]
            for piece in piece_lists[i]:
                pt, pr, pc = type(piece), piece.pos[0], piece.pos[1]
                if pt is Pawn: pawns[i]+=1
                elif pt is not King:
                    pcs[i]+=1; last[i]=pt; total_phase+=MG_PIECE_VALUES.get(pt,0)
                    if pt is Rook: r_c[i]+=1
                    elif pt is Bishop: b_c[i]+=1
                    elif pt is Knight: n_c[i]+=1
                    elif pt is Queen: q_c[i]+=1
                v_mg,v_eg,r_pst = MG_PIECE_VALUES[pt],EG_PIECE_VALUES[pt],(pr if is_white else 7-pr)
                if pt is King:
                    mg[i]+=PIECE_SQUARE_TABLES['king_midgame'][r_pst][pc]; eg[i]+=PIECE_SQUARE_TABLES['king_endgame'][r_pst][pc]
                else:
                    mg[i]+=v_mg; eg[i]+=v_eg; pst_val = PIECE_SQUARE_TABLES[pt][r_pst][pc]; mg[i]+=pst_val; eg[i]+=pst_val
                if pt is Pawn and ((pc>0 and isinstance(grid[pr][pc-1],Pawn) and grid[pr][pc-1].color==my_color) or (pc<COLS-1 and isinstance(grid[pr][pc+1],Pawn) and grid[pr][pc+1].color==my_color)):
                    mg[i]+=self.EVAL_PAWN_PHALANX_BONUS
                elif pt is Rook and enemy_king and (pr==enemy_king[0] or pc==enemy_king[1]):
                    mg[i]+=self.EVAL_ROOK_ALIGNMENT_BONUS
                elif pt is Knight:
                    for ar,ac in KNIGHT_ATTACKS_FROM[(pr,pc)]:
                        threat = grid[ar][ac]
                        if threat and threat.color!=my_color and type(threat) is not King:
                            mg[i]+=self.KNIGHT_ACTIVITY_BONUS; eg[i]+=self.KNIGHT_ACTIVITY_BONUS
        ph = min(256,(total_phase*256)//INITIAL_PHASE_MATERIAL) if INITIAL_PHASE_MATERIAL>0 else 0; inv = 256-ph
        total_p = pawns[0]+pawns[1]
        if pcs[0]>pcs[1]: eg[0]+=self.EVAL_PIECE_DOMINANCE_FACTOR//(pcs[1]+1)
        elif pcs[1]>pcs[0]: eg[1]+=self.EVAL_PIECE_DOMINANCE_FACTOR//(pcs[0]+1)
        LONE_R,LONE_B = [550,200,150,80,40],[650,250,170,100,50]
        for i in (0,1):
            if pawns[i]<4: pen = int(-250*(4-pawns[i])**2/16); mg[i]+=pen; eg[i]+=pen
            if pcs[i]==1 and pawns[i]<=4:
                pen=0
                if last[i] is Rook: pen=LONE_R[pawns[i]]
                elif last[i] is Bishop: pen=LONE_B[pawns[i]]
                if pen>0:
                    if i==0 and eg[0]>eg[1]: eg[0]=max(eg[1],eg[0]-pen)
                    elif i==1 and eg[1]>eg[0]: eg[1]=max(eg[0],eg[1]-pen)
            if b_c[i]>=2: mg[i]+=self.EVAL_PAIR_BONUS; eg[i]+=self.EVAL_PAIR_BONUS
            if n_c[i]>=2: mg[i]+=self.EVAL_PAIR_BONUS; eg[i]+=self.EVAL_PAIR_BONUS
            if r_c[i]>=2: mg[i]-=self.EVAL_DOUBLE_ROOK_PENALTY; eg[i]-=self.EVAL_DOUBLE_ROOK_PENALTY
            if r_c[i]>0: bonus = r_c[i]*total_p*self.EVAL_ROOK_PAWN_SCALING; mg[i]+=bonus; eg[i]+=bonus
        if kings[0] and kings[1]:
            dist = abs(kings[0][0]-kings[1][0])+abs(kings[0][1]-kings[1][1]); trop = (dist*dist*inv*50)//50176
            if eg[0]>eg[1]: eg[0]-=trop
            elif eg[1]>eg[0]: eg[1]-=trop
        mg_s,eg_s = mg[0]-mg[1],eg[0]-eg[1]; final = (mg_s*ph+eg_s*inv)>>8
        can_mate=[True,True]
        for i in(0,1):
            if (pawns[i]==0 and n_c[i]==0 and q_c[i]==0 and (r_c[i]+b_c[i])<2): can_mate[i]=False
        if final>0 and not can_mate[0]: final//=8
        elif final<0 and not can_mate[1]: final//=8
        return final if turn_to_move=='white' else -final
    

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