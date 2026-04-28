# AI.py (v116 - Try new knight PSTs and new Pieces values)

import json
import os
import time
import random
import glob
from collections import namedtuple
from GameLogic import *
from TablebaseManager import TablebaseManager
from operator import itemgetter

# --- TIME CONSTANTS ---
TIME_BUFFER_SEC = 0.50
TIME_BUFFER_PCT = 0.05
MIN_MOVE_TIME   = 0.03

# --- EVALUATION CONSTANTS (Tuned) ---
MG_PIECE_VALUES = {
    Pawn: 100,
    Knight: 1000,
    Bishop: 600,
    Rook: 600,
    Queen: 1300,
    King: 20000
}

EG_PIECE_VALUES = {
    Pawn: 100,
    Knight: 950,
    Bishop: 700,
    Rook: 750,
    Queen: 1000,
    King: 20000
}

# Array directly accessed via piece.z_idx
ORDERING_VALUES = [
    MG_PIECE_VALUES[Pawn],
    MG_PIECE_VALUES[Knight],
    MG_PIECE_VALUES[Bishop],
    MG_PIECE_VALUES[Rook],
    MG_PIECE_VALUES[Queen],
    MG_PIECE_VALUES[King]
]

INITIAL_PHASE_MATERIAL = (MG_PIECE_VALUES[Rook] * 4 + MG_PIECE_VALUES[Knight] * 4 +
                          MG_PIECE_VALUES[Bishop] * 4 + MG_PIECE_VALUES[Queen] * 2)

# --- ZOBRIST HASHING ---
ZOBRIST_ARRAY = None
ZOBRIST_TURN = None

def initialize_zobrist_table():
    global ZOBRIST_ARRAY, ZOBRIST_TURN
    if ZOBRIST_ARRAY is not None: return
    random.seed(42) # Set seed for stable Zobrist keys
    ZOBRIST_ARRAY = [[[[random.getrandbits(64) for _ in range(8)] for _ in range(8)] for _ in range(6)] for _ in range(2)]
    ZOBRIST_TURN = random.getrandbits(64)
    random.seed() # Restore true randomness for move selection

initialize_zobrist_table()

def run_ai_process(board, color, position_counts, comm_queue, cancellation_event,
                   bot_class, bot_name, search_depth, ply_count, game_mode,
                   time_left=None, increment=None, use_opening_book=True, use_tablebase=True):
    try:
        bot = bot_class(board, color, position_counts, comm_queue, cancellation_event,
                        bot_name, ply_count, game_mode, time_left=time_left, increment=increment,
                        use_opening_book=use_opening_book, use_tablebase=use_tablebase)
    except TypeError:
        try:
            # Fallback for bots missing the tablebase argument
            bot = bot_class(board, color, position_counts, comm_queue, cancellation_event,
                            bot_name, ply_count, game_mode, time_left=time_left, increment=increment,
                            use_opening_book=use_opening_book)
        except TypeError:
            try:
                # Fallback for bots missing the opening book argument
                bot = bot_class(board, color, position_counts, comm_queue, cancellation_event,
                                bot_name, ply_count, game_mode, time_left=time_left, increment=increment)
            except TypeError:
                # Fallback for very old bots
                bot = bot_class(board, color, position_counts, comm_queue, cancellation_event,
                                bot_name, ply_count, game_mode)

    bot.search_depth = search_depth
    if search_depth == 99:
        bot.ponder_indefinitely()
    else:
        bot.make_move()

def board_hash(board, turn):
    h = 0
    arr = ZOBRIST_ARRAY

    for piece in board.white_pieces:
        h ^= arr[0][piece.z_idx][piece.pos[0]][piece.pos[1]]
    for piece in board.black_pieces:
        h ^= arr[1][piece.z_idx][piece.pos[0]][piece.pos[1]]

    if turn == 'black':
        h ^= ZOBRIST_TURN
    return h

def incremental_hash(parent_hash, record_tuple):
    h = parent_hash ^ ZOBRIST_TURN
    arr = ZOBRIST_ARRAY

    start, end, mp, removed_pieces, added_pieces = record_tuple
    
    c_idx  = 0 if mp.color == 'white' else 1
    p_idx  = mp.z_idx
    sr, sc = start
    er, ec = end

    h ^= arr[c_idx][p_idx][sr][sc]
    
    mp_survived = True
    for piece, r, c in removed_pieces:
        if piece is mp:
            mp_survived = False
        else:
            pc_idx = 0 if piece.color == 'white' else 1
            h ^= arr[pc_idx][piece.z_idx][r][c]

    if mp_survived:
        h ^= arr[c_idx][p_idx][er][ec]

    for piece, r, c in added_pieces:
        pc_idx = 0 if piece.color == 'white' else 1
        h ^= arr[pc_idx][piece.z_idx][r][c]

    return h

# --- OPENING BOOK SETUP ---
_CLS_TO_CHAR = {Pawn: 'P', Knight: 'N', Bishop: 'B', Rook: 'R', Queen: 'Q', King: 'K'}

def board_to_fen(board, turn):
    fen = ''
    for r in range(ROWS):
        empty = 0
        for c in range(COLS):
            piece = board.grid[r][c]
            if piece is None:
                empty += 1
            else:
                if empty:
                    fen += str(empty)
                    empty = 0
                ch = _CLS_TO_CHAR[type(piece)]
                fen += ch if piece.color == 'white' else ch.lower()
        if empty:
            fen += str(empty)
        if r < ROWS - 1:
            fen += '/'
    return fen + (' w' if turn == 'white' else ' b')

OPENING_BOOK = {}


def _find_opening_book_files():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    patterns = (
        os.path.join(base_dir, "opening books", "opening_book*.json"),
        os.path.join(base_dir, "opening_book*.json"),
    )
    seen = set()
    matches = []
    for pattern in patterns:
        for path in glob.glob(pattern):
            norm = os.path.normcase(os.path.abspath(path))
            if norm in seen:
                continue
            seen.add(norm)
            matches.append(path)
    return sorted(
        matches,
        key=lambda path: (os.path.getmtime(path), os.path.basename(path)),
        reverse=True,
    )


for _book_filename in _find_opening_book_files():
    try:
        with open(_book_filename, "r", encoding="utf-8") as f:
            OPENING_BOOK = json.load(f)
        # print(f"Loaded Opening Book with {len(OPENING_BOOK)} positions from {_book_filename}.")
        break
    except Exception as e:
        print(f"Opening book not found or invalid at {_book_filename}: {e}")
# --------------------------

# --- SEARCH STRUCTURES ---
TTEntry = namedtuple('TTEntry', ['score', 'depth', 'flag', 'best_move'])
TT_FLAG_EXACT, TT_FLAG_LOWERBOUND, TT_FLAG_UPPERBOUND = 0, 1, 2

class SearchCancelledException(Exception): pass


class ChessBot:
    tb_probe_limit = 5
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

    USE_FUTILITY_PRUNING = True
    FUTILITY_MARGIN = 1000 # Lowered for speed at the cost of some tactical loss

    TT_MAX_SIZE = 50_000_000

    BONUS_PV_MOVE = 10_000_000
    BONUS_CAPTURE = 8_000_000
    BONUS_KILLER_1 = 4_000_000
    BONUS_KILLER_2 = 3_000_000
    BONUS_CONTINUATION = 2_500_000
    BAD_TACTIC_PENALTY = -2_000_000
    OPENING_TOTAL_PIECE_THRESHOLD = 23
    OPENING_BONUS_MAX_PLY = 1
    OPENING_KNIGHT_DEVELOP_BONUS = 100
    OPENING_KNIGHT_CENTER_WEIGHT = 22
    OPENING_PAWN_CENTER_WEIGHT = 10
    OPENING_CENTER_PAWN_BONUS = 28
    OPENING_CENTRAL_FILES = (COLS // 2 - 1, COLS // 2)
    ASP_WINDOW_INIT = 250
    ASP_MAX_RETRIES = 3

    MAX_EXTENSION_DEPTH = 12  # absolute ply ceiling including check extensions

    EVAL_PAWN_PHALANX_BONUS = 5
    EVAL_ROOK_ALIGNMENT_BONUS = 15
    EVAL_ROOK_OPEN_FILE_BONUS_MG = 25
    EVAL_ROOK_OPEN_FILE_BONUS_EG = 20
    EVAL_PIECE_DOMINANCE_FACTOR = 40
    EVAL_PAIR_BONUS = 20
    EVAL_DOUBLE_ROOK_PENALTY = 15
    EVAL_ROOK_PAWN_SCALING = 5
    KNIGHT_ACTIVITY_BONUS = 12
    EVAL_KING_ZONE_ATTACK_PENALTY = 50 #Stronger pentalty
    EVAL_PASSED_PAWN_PER_RANK = 10
    LONE_ROOK_PENALTIES = (550, 200, 150, 80, 40)
    LONE_BISHOP_PENALTIES = (650, 250, 170, 100, 50)
    EVAL_PAWN_VULNERABILITY_EG = 15

    EVAL_MOBILITY_ROOK   = 4
    EVAL_MOBILITY_QUEEN  = 2
    EVAL_MOBILITY_BISHOP = 4
    EVAL_MOBILITY_QUEEN  = 5

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
        
        # Always calculate a sensible mask, even if not using time controls.
        if time_left:
             allocated = (self.time_left / 30.0) + (self.increment * 0.8)
             self.time_check_mask = self._calc_time_check_mask(allocated)
        else:
             self.time_check_mask = 511 # A reasonable default for pondering/fixed-depth

        # --- OPENING BOOK FLAG ---
        self.use_opening_book = use_opening_book

        self.tb_manager = TablebaseManager()

        if not use_tablebase:
            # Neuter the probe method so it does nothing and returns None,
            # causing the engine to fall back to its own search.
            self.tb_manager.probe = lambda b, t: None

        if bot_name is None:
            self.bot_name = "OP Bot" if self.__class__.__name__ == "OpponentAI" else "AI Bot"
        else:
            self.bot_name = bot_name

        self._initialize_search_state()

    def _initialize_search_state(self):
        self.tt = {}
        self.eval_tt = {}
        self.nodes_searched = 0
        self.used_heuristic_eval = False
        self.tb_hits = 0
        self.killer_moves = [[None, None] for _ in range(max(200, self.max_moves))]
        self.history_heuristic_table = [[[0 for _ in range(64)] for _ in range(64)] for _ in range(2)]
        self.counter_moves = [[[None for _ in range(64)] for _ in range(64)] for _ in range(2)]
        # [color][prev_piece_type][prev_to_sq][my_piece_type][my_to_sq]
        self.continuation_history = [[[[[0] * 64 for _ in range(6)] for _ in range(64)] for _ in range(6)] for _ in range(2)]

    def _store_tt(self, hash_val, score, depth, flag, move):
        existing = self.tt.get(hash_val)
        if len(self.tt) > self.TT_MAX_SIZE:
            self.tt.clear()
        if not existing or depth >= existing.depth:
            best_move = move if move is not None else (existing.best_move if existing else None)
            self.tt[hash_val] = TTEntry(score, depth, flag, best_move)

    def _report_log(self, message):   self.comm_queue.put(('log', message))
    def _report_eval(self, score, depth): self.comm_queue.put(('eval', score if self.color == 'white' else -score, depth))
    def _report_move(self, move):     self.comm_queue.put(('move', move))

    def _calc_time_check_mask(self, allocated):
        if allocated <= 0.15: return 15
        if allocated <= 0.30: return 31
        if allocated <= 0.60: return 63
        if allocated <= 1.20: return 127
        if allocated <= 2.50: return 255
        return 511

    def _format_move(self, board_before, move):
        if not move: return "None"
        child = board_before.clone()
        child.make_move(move[0], move[1])
        return format_move_san(board_before, child, move)

    def _is_opening_position(self, board):
        return (len(board.white_pieces) + len(board.black_pieces)) >= self.OPENING_TOTAL_PIECE_THRESHOLD

    def _opening_development_bonus(self, move, moving_piece):
        if moving_piece is None: return 0
        (fr, fc), (tr, tc) = move
        from_center = abs(fr - 3.5) + abs(fc - 3.5)
        to_center   = abs(tr - 3.5) + abs(tc - 3.5)
        center_delta = from_center - to_center
        bonus = 0
        if type(moving_piece) is Knight:
            bonus += int(center_delta * self.OPENING_KNIGHT_CENTER_WEIGHT)
            if (moving_piece.color == 'white' and fr == ROWS - 1) or (moving_piece.color == 'black' and fr == 0):
                bonus += self.OPENING_KNIGHT_DEVELOP_BONUS
            return bonus
        if type(moving_piece) is Pawn:
            bonus += int(center_delta * self.OPENING_PAWN_CENTER_WEIGHT)
            if tc in self.OPENING_CENTRAL_FILES:
                bonus += self.OPENING_CENTER_PAWN_BONUS
            return bonus
        return 0

    def _get_root_tb_eval_relative(self):
        root_abs = self.tb_manager.probe(self.board, self.color)
        if root_abs is None: return None
        self.tb_hits += 1
        return root_abs if self.color == 'white' else -root_abs

    def _get_best_tablebase_move_with_eval(self):
        root_abs = self.tb_manager.probe(self.board, self.color)
        if root_abs is None and not is_insufficient_material(self.board): return None, None

        best_move  = None
        best_score = -float('inf')

        for move in get_all_legal_moves(self.board, self.color):
            sim = self.board.clone()
            sim.make_move(move[0], move[1])

            if not sim.find_king_pos(self.opponent_color): return move, self.MATE_SCORE - 1
            if not has_legal_moves(sim, self.opponent_color): return move, self.MATE_SCORE - 1

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
                best_score = score
                best_move  = move

        return best_move, best_score

    def _run_depth_iteration(self, depth, root_moves, root_hash, pv_move,
                             prev_iter_score=None, alpha_floor=None):
        iter_nodes       = 0
        iter_tb_hits     = 0
        any_heuristic_eval = False
        use_aspiration   = (alpha_floor is None and prev_iter_score is not None and depth >= 2)

        if use_aspiration:
            window      = self.ASP_WINDOW_INIT
            alpha_bound = prev_iter_score - window
            beta_bound  = prev_iter_score + window
            retries     = 0
            while True:
                best_score, best_move = self._search_at_depth(
                    depth, root_moves, root_hash, pv_move,
                    aspiration_window=(alpha_bound, beta_bound))
                iter_nodes   += self.nodes_searched
                iter_tb_hits += self.tb_hits
                if self.used_heuristic_eval: any_heuristic_eval = True

                if self.cancellation_event.is_set() or (self.stop_time and time.time() > self.stop_time):
                    raise SearchCancelledException()

                if best_score <= alpha_bound:
                    alpha_bound -= window; window *= 2; retries += 1
                elif best_score >= beta_bound:
                    beta_bound  += window; window *= 2; retries += 1
                else:
                    break

                if retries >= self.ASP_MAX_RETRIES:
                    best_score, best_move = self._search_at_depth(
                        depth, root_moves, root_hash, pv_move, alpha_floor=alpha_floor)
                    iter_nodes   += self.nodes_searched
                    iter_tb_hits += self.tb_hits
                    if self.used_heuristic_eval: any_heuristic_eval = True
                    break
        else:
            best_score, best_move = self._search_at_depth(
                depth, root_moves, root_hash, pv_move, alpha_floor=alpha_floor)
            iter_nodes       = self.nodes_searched
            iter_tb_hits     = self.tb_hits
            if self.used_heuristic_eval: any_heuristic_eval = True

        self.nodes_searched    = iter_nodes
        self.tb_hits           = iter_tb_hits
        self.used_heuristic_eval = any_heuristic_eval
        return best_score, best_move

    def _age_history_table(self):
        for color_idx in range(2):
            for from_sq in range(ROWS * COLS):
                for to_sq in range(ROWS * COLS):
                    self.history_heuristic_table[color_idx][from_sq][to_sq] //= 2

    def _get_pv_data(self, max_depth, root_move):
        if not root_move: return [], []

        pv_san  = []
        pv_raw  = []
        current_board = self.board.clone()
        current_turn  = self.color
        current_ply   = self.ply_count
        seen_hashes = set()
        move = root_move

        for i in range(max_depth):
            if not move: break
            san      = self._format_move(current_board, move)
            move_num = (current_ply // 2) + 1
            if current_turn == 'white':
                pv_san.append(f"{move_num}. {san}")
            else:
                pv_san.append(f"{move_num}... {san}" if i == 0 else san)

            pv_raw.append(move)
            current_board.make_move(move[0], move[1])
            current_turn = 'black' if current_turn == 'white' else 'white'
            current_ply += 1

            h = board_hash(current_board, current_turn)
            if h in seen_hashes: break
            seen_hashes.add(h)

            tt_entry = self.tt.get(h)
            if not tt_entry or not tt_entry.best_move: break
            if tt_entry.flag != TT_FLAG_EXACT: break
            move = tt_entry.best_move

        return pv_san, pv_raw

    def _report_root_tb_solution(self, tb_move, tb_eval, perfect_play=False, emit_move=False):
        if not tb_move: return False
        root_tb_eval  = self._get_root_tb_eval_relative()
        display_eval  = root_tb_eval if root_tb_eval is not None else tb_eval
        if tb_eval > self.MATE_SCORE - 1000: display_eval = tb_eval

        eval_for_ui = display_eval if self.color == 'white' else -display_eval
        suffix      = " (Perfect Play)" if perfect_play else ""
        self._report_log(f"  > {self.bot_name} (TB): {self._format_move(self.board, tb_move)}, Eval={eval_for_ui/100:+.2f}, TBhits={self.tb_hits}{suffix}")
        self._report_eval(display_eval, "TB")

        move_num = (self.ply_count // 2) + 1
        prefix   = f"{move_num}. " if self.color == 'white' else f"{move_num}... "
        pv_str   = prefix + self._format_move(self.board, tb_move)
        self.comm_queue.put(('pv', display_eval, "TB", [pv_str], [tb_move]))

        if emit_move: self._report_move(tb_move)
        return True

    def make_move(self):
        try:
            self._age_history_table()

            # 1. Check Tablebases
            if len(self.board.white_pieces) + len(self.board.black_pieces) <= self.tb_probe_limit:
                tb_move, tb_eval = self._get_best_tablebase_move_with_eval()
                if self._report_root_tb_solution(tb_move, tb_eval, emit_move=True): return

            # 2. Check Opening Book
            if self.use_opening_book and self.ply_count <= 12:
                fen = board_to_fen(self.board, self.color)
                if fen in OPENING_BOOK:
                    book_options = OPENING_BOOK[fen]
                    weights = [opt["weight"] for opt in book_options]
                    chosen = random.choices(book_options, weights=weights, k=1)[0]
                    move_tuple = (tuple(chosen["move"][0]), tuple(chosen["move"][1]))
                    abs_score = chosen['score']
                    rel_score = abs_score if self.color == 'white' else -abs_score
                    self._report_log(f"  > {self.bot_name} (Book): {chosen['san']}")
                    self._report_eval(rel_score, "Book")
                    self.comm_queue.put(('pv', abs_score, "Book", [chosen['san']], [move_tuple]))
                    self._report_move(move_tuple)
                    return

            root_moves = get_all_legal_moves(self.board, self.color)
            if not root_moves:
                self._report_move(None)
                return

            best_move_overall  = root_moves[0]
            prev_iter_score    = None
            total_nodes        = 0
            root_hash          = board_hash(self.board, self.color)

            # --- TIME ALLOCATION STRATEGY ---
            if self.time_left is not None and self.increment is not None:
                allocated = (self.time_left / 30.0) + (self.increment * 0.8)
                buffer = max(TIME_BUFFER_SEC, self.time_left * TIME_BUFFER_PCT, self.increment * 1.5)
                max_alloc = max(0.0, self.time_left - buffer)
                allocated = min(allocated, max_alloc)
                allocated = min(max_alloc, max(MIN_MOVE_TIME, allocated))
                self.stop_time = time.time() + allocated
                target_depth = 100
            else:
                self.stop_time = None
                # self.time_check_mask is now set in __init__
                target_depth = self.search_depth

            for current_depth in range(1, target_depth + 1):
                iter_start_time = time.time()
                best_score_this_iter, best_move_this_iter = self._run_depth_iteration(
                    current_depth, root_moves, root_hash, best_move_overall, prev_iter_score=prev_iter_score)

                if self.cancellation_event.is_set():
                    raise SearchCancelledException()

                if self.stop_time and time.time() > self.stop_time:
                    best_move_overall = best_move_this_iter
                    break

                best_move_overall = best_move_this_iter
                prev_iter_score   = best_score_this_iter
                total_nodes      += self.nodes_searched
                iter_duration     = time.time() - iter_start_time
                knps              = (self.nodes_searched / iter_duration / 1000) if iter_duration > 0 else 0
                # Prefer reporting a root tablebase evaluation if available —
                # this ensures TB-drawn positions are shown as TB and can be
                # auto-adjudicated by the UI. Fall back to heuristic reporting
                # when no root TB exists.
                root_tb_val = None
                if len(self.board.white_pieces) + len(self.board.black_pieces) <= self.tb_probe_limit:
                    root_tb_val = self.tb_manager.probe(self.board, self.color)

                if root_tb_val is not None:
                    # Root is in a tablebase — report TB as authoritative
                    self.tb_hits += 1
                    depth_label = "TB"
                    report_score = root_tb_val
                else:
                    depth_label = "TB" if (not self.used_heuristic_eval and self.tb_hits > 0) else current_depth
                    report_score = best_score_this_iter

                eval_for_ui = report_score if self.color == 'white' else -report_score
                self._report_log(f"  > {self.bot_name} (D{depth_label}): {self._format_move(self.board, best_move_this_iter)}, Eval={eval_for_ui/100:+.2f}, NodesTotal={total_nodes}, KNPS={knps:.1f}, TBhits={self.tb_hits}, Time={iter_duration:.2f}s")
                self._report_eval(report_score, depth_label)

                ui_eval        = report_score if self.color == 'white' else -report_score
                pv_str, pv_raw = self._get_pv_data(current_depth, best_move_this_iter)
                self.comm_queue.put(('pv', ui_eval, depth_label, pv_str, pv_raw))

                if report_score > self.MATE_SCORE - 2000: break

            self._report_move(best_move_overall)
        except SearchCancelledException:
            if self.cancellation_event.is_set():
                self._report_move(None)
            else:
                self._report_move(best_move_overall)

    def ponder_indefinitely(self):
        try:
            self._age_history_table()
            if is_insufficient_material(self.board): return
            if len(self.board.white_pieces) + len(self.board.black_pieces) <= self.tb_probe_limit:
                tb_move, tb_eval = self._get_best_tablebase_move_with_eval()
                if self._report_root_tb_solution(tb_move, tb_eval, perfect_play=True):
                    while not self.cancellation_event.is_set(): time.sleep(0.1)
                    return

            root_moves        = get_all_legal_moves(self.board, self.color)
            if not root_moves: return

            best_move_overall = root_moves[0]
            root_hash         = board_hash(self.board, self.color)
            tb_alpha_floor    = None
            prev_iter_score   = None
            total_nodes       = 0

            for current_depth in range(1, 100):
                if self.cancellation_event.is_set(): raise SearchCancelledException()
                iter_start_time = time.time()
                best_score_this_iter, best_move_this_iter = self._run_depth_iteration(
                    current_depth, root_moves, root_hash, best_move_overall,
                    prev_iter_score=prev_iter_score, alpha_floor=tb_alpha_floor)

                if not self.cancellation_event.is_set():
                    best_move_overall = best_move_this_iter
                    prev_iter_score   = best_score_this_iter
                    total_nodes      += self.nodes_searched
                    iter_duration     = time.time() - iter_start_time
                    knps              = (self.nodes_searched / iter_duration / 1000) if iter_duration > 0 else 0
                    # Prefer reporting a root tablebase evaluation if available —
                    # this ensures TB-drawn positions are shown as TB and can be
                    # auto-adjudicated by the UI. Fall back to heuristic reporting
                    # when no root TB exists.
                    root_tb_val = None
                    if len(self.board.white_pieces) + len(self.board.black_pieces) <= self.tb_probe_limit:
                        root_tb_val = self.tb_manager.probe(self.board, self.color)

                    if root_tb_val is not None:
                        # Root is in a tablebase — report TB as authoritative
                        self.tb_hits += 1
                        depth_label = "TB"
                        report_score = root_tb_val
                    else:
                        depth_label = "TB" if (not self.used_heuristic_eval and self.tb_hits > 0) else current_depth
                        report_score = best_score_this_iter

                    eval_for_ui = report_score if self.color == 'white' else -report_score
                    self._report_log(f"  > {self.bot_name} (D{depth_label}): {self._format_move(self.board, best_move_this_iter)}, Eval={eval_for_ui/100:+.2f}, NodesTotal={total_nodes}, KNPS={knps:.1f}, TBhits={self.tb_hits}, Time={iter_duration:.2f}s")
                    self._report_eval(report_score, depth_label)

                    ui_eval        = report_score if self.color == 'white' else -report_score
                    pv_str, pv_raw = self._get_pv_data(current_depth, best_move_this_iter)
                    self.comm_queue.put(('pv', ui_eval, depth_label, pv_str, pv_raw))

                    if depth_label == "TB":
                        while not self.cancellation_event.is_set(): time.sleep(0.1)
                        return

                    if best_score_this_iter > self.MATE_SCORE - 1000 and not self.used_heuristic_eval:
                        tb_alpha_floor = best_score_this_iter
                    else:
                        tb_alpha_floor = None
                else:
                    raise SearchCancelledException()
        except SearchCancelledException:
            pass

    def _search_at_depth(self, depth, root_moves, root_hash, pv_move, alpha_floor=None, aspiration_window=None):
        self.nodes_searched    = 0
        self.used_heuristic_eval = False
        self.tb_hits           = 0

        if alpha_floor is not None:
            best_score_this_iter  = alpha_floor
            best_move_this_iter   = pv_move if pv_move in root_moves else (root_moves[0] if root_moves else None)
            alpha = alpha_floor
            beta  = float('inf')
        elif aspiration_window is not None:
            best_score_this_iter, best_move_this_iter = -float('inf'), None
            alpha, beta = aspiration_window
        else:
            best_score_this_iter, best_move_this_iter = -float('inf'), None
            alpha = -float('inf')
            beta  =  float('inf')

        ordered_root_moves = self.order_moves(self.board, root_moves, 0, pv_move, self.color)
        board = self.board
        all_moves_draw = True

        for move in ordered_root_moves:
            if self.cancellation_event.is_set(): raise SearchCancelledException()

            record     = board.make_move_track(move[0], move[1])
            child_hash = incremental_hash(root_hash, record)

            search_path = {root_hash}
            try:
                mp = record[2]
                next_prev_tuple = (move, mp.z_idx)

                if alpha_floor is not None:
                    probe_score = -self.negamax(
                        board, depth - 1, -(alpha_floor + 1), -alpha_floor,
                        self.opponent_color, 1, search_path,
                        current_hash=child_hash, prev_move_tuple=next_prev_tuple)
                    if probe_score <= alpha_floor:
                        continue
                    score = -self.negamax(
                        board, depth - 1, -beta, -alpha,
                        self.opponent_color, 1, search_path,
                        current_hash=child_hash, prev_move_tuple=next_prev_tuple)
                else:
                    score = -self.negamax(
                        board, depth - 1, -beta, -alpha,
                        self.opponent_color, 1, search_path,
                        current_hash=child_hash, prev_move_tuple=next_prev_tuple)
            finally:
                board.unmake_move(record)

            if score != self.DRAW_SCORE: all_moves_draw = False

            if score > best_score_this_iter:
                best_score_this_iter = score
                best_move_this_iter  = move
            alpha = max(alpha, best_score_this_iter)

        if alpha_floor is None and all_moves_draw:
            best_score_this_iter = self.DRAW_SCORE
        return best_score_this_iter, best_move_this_iter

    def negamax(self, board, depth, alpha, beta, turn, ply, search_path, current_hash=None, prev_move_tuple=None, extensions=0):
        self.nodes_searched += 1
        if (self.nodes_searched & self.time_check_mask) == 0:
            if self.cancellation_event.is_set() or (self.stop_time and time.time() > self.stop_time):
                raise SearchCancelledException()

        # --- REPETITION CHECKS (must come before TB probe) ---
        hash_val = current_hash if current_hash is not None else board_hash(board, turn)
        if ply > 0:
            if self.position_counts.get(hash_val, 0) >= 2:
                return self.DRAW_SCORE
            if hash_val in search_path:
                return self.DRAW_SCORE

        if len(board.white_pieces) + len(board.black_pieces) <= self.tb_probe_limit:
            tb_score_absolute = self.tb_manager.probe(board, turn)
            if tb_score_absolute is not None:
                self.tb_hits += 1
                tb_score = tb_score_absolute if turn == 'white' else -tb_score_absolute
                if tb_score >  self.MATE_SCORE - 1000: return tb_score - ply
                elif tb_score < -self.MATE_SCORE + 1000: return tb_score + ply
                return tb_score

        if is_insufficient_material(board) or self.ply_count + ply >= self.max_moves:
            return self.DRAW_SCORE

        original_alpha = alpha
        tt_entry = self.tt.get(hash_val)
        if ply > 0 and tt_entry and tt_entry.depth >= depth:
            tt_score = tt_entry.score
            if tt_score >  self.MATE_SCORE - 1000: tt_score -= ply
            elif tt_score < -self.MATE_SCORE + 1000: tt_score += ply

            self.used_heuristic_eval = True

            if tt_entry.flag == TT_FLAG_EXACT:       return tt_score
            elif tt_entry.flag == TT_FLAG_LOWERBOUND: alpha = max(alpha, tt_score)
            elif tt_entry.flag == TT_FLAG_UPPERBOUND: beta  = min(beta,  tt_score)
            if alpha >= beta: return tt_score

        if depth <= 0: return self.qsearch(board, alpha, beta, turn, ply, current_hash=hash_val)

        opponent_turn    = 'black' if turn == 'white' else 'white'
        is_in_check_flag = is_in_check(board, turn)
        static_eval      = None

        # --- CHECK EXTENSION with absolute ceiling ---
        if is_in_check_flag and extensions < 16 and ply < self.MAX_EXTENSION_DEPTH:
            depth      += 1
            extensions += 1

        path_added = False
        if hash_val not in search_path:
            search_path.add(hash_val)
            path_added = True

        try:
            if (self.USE_NULL_MOVE_PRUNING and depth >= self.NMP_MIN_DEPTH and
                    ply > 0 and not is_in_check_flag and abs(beta) < self.MATE_SCORE - 1000):
                pc = board.piece_counts
                if (pc['white'][Knight] + pc['white'][Bishop] + pc['white'][Rook] + pc['white'][Queen] > 0 and
                        pc['black'][Knight] + pc['black'][Bishop] + pc['black'][Rook] + pc['black'][Queen] > 0):
                    self.used_heuristic_eval = True
                    static_eval = self.evaluate_board(board, turn)
                    if static_eval >= beta:
                        reduction  = self.NMP_BASE_REDUCTION + (depth // self.NMP_DEPTH_DIVISOR)
                        null_hash  = hash_val ^ ZOBRIST_TURN
                        score = -self.negamax(board, depth - 1 - reduction, -beta, -beta + 1,
                                            opponent_turn, ply + 1, search_path,
                                            null_hash, None, extensions)
                        if score >= beta: return beta

            # --- FORWARD FUTILITY PRUNING (Original Conservative Version) ---
            futility_prune = False
            if (self.USE_FUTILITY_PRUNING and depth == 1 and not is_in_check_flag and
                    abs(alpha) < self.MATE_SCORE - 1000):
                self.used_heuristic_eval = True
                if static_eval is None:
                    static_eval = self.evaluate_board(board, turn)
                if static_eval + self.FUTILITY_MARGIN < alpha:
                    futility_prune = True

            pseudo_moves = get_all_pseudo_legal_moves(board, turn)
            hash_move    = tt_entry.best_move if tt_entry else None
            
            if prev_move_tuple:
                (pr1, pc1), (pr2, pc2) = prev_move_tuple[0]
                c_move = self.counter_moves[0 if turn == 'white' else 1][pr1 * 8 + pc1][pr2 * 8 + pc2]
            else:
                c_move = None

            ordered_entries = self.order_moves(board, pseudo_moves, ply, hash_move, turn,
                                            return_meta=True, counter_move=c_move, prev_move_tuple=prev_move_tuple)
            best_move_for_node = None
            legal_moves_count  = 0
            quiet_moves_tried  = []
            history_table      = self.history_heuristic_table[0 if turn == 'white' else 1]

            for move, meta in ordered_entries:
                is_good_tactic, moving_piece = meta
                f_sq = move[0][0] * 8 + move[0][1]
                t_sq = move[1][0] * 8 + move[1][1]

                record     = board.make_move_track(move[0], move[1])
                child_hash = incremental_hash(hash_val, record)

                opp_king_alive    = (board.white_king_pos is not None) if opponent_turn == 'white' else (board.black_king_pos is not None)
                own_king_in_check = is_in_check(board, turn)

                if not opp_king_alive:
                    board.unmake_move(record)
                    return self.MATE_SCORE - ply

                if own_king_in_check:
                    board.unmake_move(record)
                    continue

                legal_moves_count += 1
                if not is_good_tactic: quiet_moves_tried.append((move, moving_piece))

                if futility_prune and not is_good_tactic and legal_moves_count > 1:
                    # SAFETY GUARD: Never prune a quiet move that delivers check.
                    # Thanks to the highly optimized is_square_attacked in GameLogic,
                    # this check is now fast enough to run here without tanking NPS.
                    if not is_in_check(board, opponent_turn):
                        board.unmake_move(record)
                        continue

                # --- CALIBRATED LATE MOVE REDUCTION ---
                reduction = 0
                if (depth >= self.LMR_DEPTH_THRESHOLD and
                        legal_moves_count > self.LMR_MOVE_COUNT_THRESHOLD and
                        not is_in_check_flag and not is_good_tactic):
                    
                    # 1. Base reduction with much gentler scaling
                    reduction = 1 + (depth // 7) + (legal_moves_count // 12)
                    
                    # 2. Protect likely refutations
                    if (ply < len(self.killer_moves) and move in self.killer_moves[ply]) or move == c_move:
                        reduction -= 1
                        
                    # 3. JUNGLE HEURISTIC: Protect volatile quiet pieces
                    if moving_piece.z_idx == 1: # Knight
                        reduction -= 1
                    elif moving_piece.z_idx == 4: # Queen
                        # Fast ray-cast from the Queen's new square. 
                        attacks_enemy = False
                        grid_ref = board.grid
                        for ray in RAYS[t_sq]:
                            for cr, cc in ray:
                                target = grid_ref[cr][cc]
                                if target is not None:
                                    if target.color == opponent_turn:
                                        attacks_enemy = True
                                    break
                            if attacks_enemy:
                                break
                        if attacks_enemy:
                            reduction -= 1
                        
                    # 4. History influence (only reward good moves, don't punish bad ones as hard)
                    if history_table[f_sq][t_sq] > 250_000:
                        reduction -= 1
                        
                    # 5. Safety Clamp
                    reduction = max(0, min(reduction, depth - 2))

                search_depth_child = depth - 1 - reduction

                next_prev_tuple = (move, moving_piece.z_idx)

                if legal_moves_count == 1:
                    score = -self.negamax(board, search_depth_child, -beta, -alpha,
                                        opponent_turn, ply + 1, search_path, child_hash, next_prev_tuple, extensions)
                else: # Principal Variation Search (PVS)
                    score = -self.negamax(board, search_depth_child, -(alpha + 1), -alpha,
                                        opponent_turn, ply + 1, search_path, child_hash, next_prev_tuple, extensions)
                    if score > alpha:
                        if reduction > 0 or score < beta:
                            score = -self.negamax(board, depth - 1, -beta, -alpha,
                                                opponent_turn, ply + 1, search_path, child_hash, next_prev_tuple, extensions)

                board.unmake_move(record)

                if score > alpha:
                    alpha, best_move_for_node = score, move

                if alpha >= beta:
                    if not is_good_tactic:
                        if ply < len(self.killer_moves) and self.killer_moves[ply][0] != move:
                            self.killer_moves[ply][1], self.killer_moves[ply][0] = \
                                self.killer_moves[ply][0], move
                        if prev_move_tuple:
                            (pr1, pc1), (pr2, pc2) = prev_move_tuple[0]
                            self.counter_moves[0 if turn == 'white' else 1][pr1 * 8 + pc1][pr2 * 8 + pc2] = move
                        
                        # --- CALIBRATED HISTORY UPDATES ---
                        if moving_piece:
                            c_idx = 0 if turn == 'white' else 1
                            bonus = depth * depth
                            ht    = self.history_heuristic_table[c_idx]
                            
                            # Gravity update for the successful move
                            ht[f_sq][t_sq] += bonus - (ht[f_sq][t_sq] * bonus) // 2_000_000
                            
                            # --- FIXED: Update Continuation History ---
                            if prev_move_tuple:
                                prev_move, prev_pt_idx = prev_move_tuple
                                pr, pc = prev_move[1]
                                prev_to_sq  = pr * 8 + pc
                                mp_idx      = moving_piece.z_idx
                                
                                ch_table = self.continuation_history[c_idx][prev_pt_idx][prev_to_sq][mp_idx]
                                ch_table[t_sq] += bonus - (ch_table[t_sq] * bonus) // 64_000
                            
                            # Gravity penalty for the failed quiet moves
                            for f_move, f_mp in quiet_moves_tried:
                                if f_move != move:
                                    (fr1, fc1), (fr2, fc2) = f_move
                                    ff = fr1 * 8 + fc1
                                    ft = fr2 * 8 + fc2
                                    ht[ff][ft] -= bonus + (ht[ff][ft] * bonus) // 2_000_000
                                    
                                    if prev_move_tuple:
                                        prev_move, prev_pt_idx = prev_move_tuple
                                        pr, pc = prev_move[1]
                                        prev_to_sq = pr * 8 + pc
                                        f_mp_idx = f_mp.z_idx
                                        ch_table = self.continuation_history[c_idx][prev_pt_idx][prev_to_sq][f_mp_idx]
                                        ch_table[ft] -= bonus + (ch_table[ft] * bonus) // 64_000

                    sto = beta
                    if sto >  self.MATE_SCORE - 1000: sto = beta + ply
                    elif sto < -self.MATE_SCORE + 1000: sto = beta - ply
                    self._store_tt(hash_val, sto, depth, TT_FLAG_LOWERBOUND, move)
                    return beta

            if legal_moves_count == 0:
                return -self.MATE_SCORE + ply

            sto = alpha
            if sto >  self.MATE_SCORE - 1000: sto = alpha + ply
            elif sto < -self.MATE_SCORE + 1000: sto = alpha - ply
            flag = TT_FLAG_EXACT if alpha > original_alpha else TT_FLAG_UPPERBOUND
            self._store_tt(hash_val, sto, depth, flag, best_move_for_node)
            return alpha

        finally:
            if path_added: search_path.discard(hash_val)


    def qsearch(self, board, alpha, beta, turn, ply, current_hash=None):
        self.nodes_searched += 1
        if (self.nodes_searched & self.time_check_mask) == 0:
            if self.cancellation_event.is_set() or (self.stop_time and time.time() > self.stop_time):
                raise SearchCancelledException()

        hash_val = current_hash if current_hash is not None else board_hash(board, turn)

        if len(board.white_pieces) + len(board.black_pieces) <= self.tb_probe_limit:
            tb_score_absolute = self.tb_manager.probe(board, turn)
            if tb_score_absolute is not None:
                self.tb_hits += 1
                tb_score = tb_score_absolute if turn == 'white' else -tb_score_absolute
                if tb_score >  self.MATE_SCORE - 1000: return tb_score - ply
                elif tb_score < -self.MATE_SCORE + 1000: return tb_score + ply
                return tb_score

        if is_insufficient_material(board): return self.DRAW_SCORE

        if ply >= self.MAX_Q_SEARCH_DEPTH:
            self.used_heuristic_eval = True
            if hash_val in self.eval_tt: return self.eval_tt[hash_val]
            score = self.evaluate_board(board, turn)
            if len(self.eval_tt) > 5_000_000: self.eval_tt.clear()
            self.eval_tt[hash_val] = score
            return score

        self.used_heuristic_eval = True
        
        if hash_val in self.eval_tt:
            stand_pat = self.eval_tt[hash_val]
        else:
            stand_pat = self.evaluate_board(board, turn)
            if len(self.eval_tt) > 5_000_000: self.eval_tt.clear()
            self.eval_tt[hash_val] = stand_pat
            
        is_in_check_flag = is_in_check(board, turn)

        if not is_in_check_flag:
            if stand_pat >= beta: return beta
            alpha = max(alpha, stand_pat)

        if ply <= 4:
            current_margin = self.Q_MARGIN_MAX
        else:
            current_margin = max(self.Q_MARGIN_MIN, self.Q_MARGIN_MAX - (ply - 4) * 117)

        promising_moves = get_all_pseudo_legal_moves(board, turn)

        scored_moves = []
        grid = board.grid
        for move in promising_moves:
            (r1, c1), (r2, c2) = move
            moving_piece = grid[r1][c1]
            target_piece = grid[r2][c2]
            swing, is_tactic = fast_approximate_material_swing(board, move, moving_piece, target_piece, ORDERING_VALUES)

            if not is_in_check_flag:
                # In Q-search, if not in check, we only look at tactical moves.
                if not is_tactic:
                    continue
                # Delta pruning (current_margin) will safely catch truly terrible tactics (like QxP hitting nothing else).
                if stand_pat + swing + current_margin < alpha:
                    continue

            scored_moves.append((swing, move))

        scored_moves.sort(key=itemgetter(0), reverse=True)

        legal_moves_count = 0
        opponent_turn     = 'black' if turn == 'white' else 'white'

        for swing, move in scored_moves:
            record = board.make_move_track(move[0], move[1])

            opp_king_alive = board.white_king_pos if opponent_turn == 'white' else board.black_king_pos
            if not opp_king_alive:
                board.unmake_move(record)
                return self.MATE_SCORE - ply

            if is_in_check(board, turn):
                board.unmake_move(record)
                continue

            legal_moves_count += 1
            child_hash = incremental_hash(hash_val, record)
            search_score = -self.qsearch(board, -beta, -alpha, opponent_turn, ply + 1, current_hash=child_hash)
            board.unmake_move(record)

            if search_score >= beta: return beta
            alpha = max(alpha, search_score)

        if is_in_check_flag and legal_moves_count == 0:
            return -self.MATE_SCORE + ply

        return alpha

    def order_moves(self, board, moves, ply, hash_move, turn, return_meta=False, counter_move=None, prev_move_tuple=None):
        if not moves: return []
        scored_moves = []
        killers   = self.killer_moves[ply] if ply < len(self.killer_moves) else [None, None]
        c_idx     = 0 if turn == 'white' else 1
        is_opening = (ply <= self.OPENING_BONUS_MAX_PLY and self._is_opening_position(board))
        history_table = self.history_heuristic_table[c_idx]

        grid = board.grid
        for move in moves:
            (r1, c1), (r2, c2) = move
            moving_piece = grid[r1][c1]
            target_piece = grid[r2][c2]

            swing, is_tactic = fast_approximate_material_swing(board, move, moving_piece, target_piece, ORDERING_VALUES)
            
            # In Jungle Chess, volatile moves (explosions, evaporations, piercings) 
            # are ALWAYS critical tactics, even if the net material swing is negative.
            is_good_tactic = is_tactic

            if move == hash_move:
                score = self.BONUS_PV_MOVE
            elif is_good_tactic:
                score = self.BONUS_CAPTURE + (swing * 100)
            elif move in killers:
                score = 4_000_000 if move == killers[0] else 3_000_000
            elif move == counter_move:
                score = 2_000_000
            else:
                score = 0
                if prev_move_tuple:
                    prev_move, prev_pt_idx = prev_move_tuple
                    pr, pc = prev_move[1]
                    prev_to_sq  = pr * 8 + pc
                    mp_idx      = moving_piece.z_idx
                    to_sq       = r2 * 8 + c2
                    
                    ch_score = self.continuation_history[c_idx][prev_pt_idx][prev_to_sq][mp_idx][to_sq]
                    if ch_score > 1000:
                        score = self.BONUS_CONTINUATION + ch_score
                
                if score == 0:
                    score = history_table[r1 * 8 + c1][r2 * 8 + c2]

            if is_opening: 
                score += self._opening_development_bonus(move, moving_piece)
            
            scored_moves.append((score, move, is_good_tactic, moving_piece))

        # C-optimized sorting that preserves stable order
        scored_moves.sort(key=itemgetter(0), reverse=True)

        if return_meta:
            return [(item[1], (item[2], item[3])) for item in scored_moves]
        else:
            return [item[1] for item in scored_moves]

    def evaluate_board(self, board, turn_to_move):
        if is_insufficient_material(board):
            return self.DRAW_SCORE

        grid = board.grid

        white_pawn_files = [False] * COLS
        black_pawn_files = [False] * COLS
        # For white pawn (moving toward row 0) to be passed, need no black pawn at row < r.
        # Track the MINIMUM row of black pawns per file.
        black_pawn_min_row = [8] * COLS
        # For black pawn (moving toward row 7) to be passed, need no white pawn at row > r.
        # Track the MAXIMUM row of white pawns per file.
        white_pawn_max_row = [-1] * COLS

        total_pawns = board.piece_counts['white'][Pawn] + board.piece_counts['black'][Pawn]
        if total_pawns > 0:
            for piece in board.white_pieces:
                if type(piece) is Pawn:
                    c, r = piece.pos[1], piece.pos[0]
                    white_pawn_files[c] = True
                    if r > white_pawn_max_row[c]: white_pawn_max_row[c] = r
            for piece in board.black_pieces:
                if type(piece) is Pawn:
                    c, r = piece.pos[1], piece.pos[0]
                    black_pawn_files[c] = True
                    if r < black_pawn_min_row[c]: black_pawn_min_row[c] = r

        scores_mg = [0, 0]; scores_eg = [0, 0]

        pc_w = board.piece_counts['white']
        pc_b = board.piece_counts['black']

        pawn_counts   = [pc_w[Pawn], pc_b[Pawn]]
        knight_counts = [pc_w[Knight], pc_b[Knight]]
        bishop_counts = [pc_w[Bishop], pc_b[Bishop]]
        rook_counts   = [pc_w[Rook], pc_b[Rook]]
        queen_counts  = [pc_w[Queen], pc_b[Queen]]

        piece_counts = [
            knight_counts[0] + bishop_counts[0] + rook_counts[0] + queen_counts[0],
            knight_counts[1] + bishop_counts[1] + rook_counts[1] + queen_counts[1]
        ]

        phase_material_score = (
            (knight_counts[0] + knight_counts[1]) * MG_PIECE_VALUES[Knight] +
            (bishop_counts[0] + bishop_counts[1]) * MG_PIECE_VALUES[Bishop] +
            (rook_counts[0] + rook_counts[1]) * MG_PIECE_VALUES[Rook] +
            (queen_counts[0] + queen_counts[1]) * MG_PIECE_VALUES[Queen]
        )

        king_pos    = [board.white_king_pos, board.black_king_pos]
        piece_lists = [board.white_pieces,   board.black_pieces]

        PAWN_PHALANX_BONUS       = self.EVAL_PAWN_PHALANX_BONUS
        ROOK_ALIGNMENT_BONUS     = self.EVAL_ROOK_ALIGNMENT_BONUS
        ROOK_OPEN_FILE_BONUS_MG  = self.EVAL_ROOK_OPEN_FILE_BONUS_MG
        ROOK_OPEN_FILE_BONUS_EG  = self.EVAL_ROOK_OPEN_FILE_BONUS_EG
        PIECE_DOMINANCE_FACTOR   = self.EVAL_PIECE_DOMINANCE_FACTOR
        PAIR_BONUS               = self.EVAL_PAIR_BONUS
        DOUBLE_ROOK_PENALTY      = self.EVAL_DOUBLE_ROOK_PENALTY
        ROOK_PAWN_SCALING        = self.EVAL_ROOK_PAWN_SCALING
        KNIGHT_ACTIVITY_BONUS    = self.KNIGHT_ACTIVITY_BONUS
        KING_ZONE_ATTACK_PENALTY = self.EVAL_KING_ZONE_ATTACK_PENALTY
        PASSED_PAWN_PER_RANK     = self.EVAL_PASSED_PAWN_PER_RANK
        LONE_ROOK_PENALTIES      = self.LONE_ROOK_PENALTIES
        LONE_BISHOP_PENALTIES    = self.LONE_BISHOP_PENALTIES
        PAWN_VULN_EG             = self.EVAL_PAWN_VULNERABILITY_EG

        king_zone_attacks = [0, 0]

        for color_idx in (0, 1):
            pieces   = piece_lists[color_idx]
            is_white = (color_idx == 0)
            my_color_name = 'white' if is_white else 'black'
            enemy_king    = king_pos[1 - color_idx]
            pst_mg = FLAT_PST_MG_WHITE if is_white else FLAT_PST_MG_BLACK
            pst_eg = FLAT_PST_EG_WHITE if is_white else FLAT_PST_EG_BLACK

            for piece in pieces:
                z       = piece.z_idx
                r, c    = piece.pos
                sq      = r * 8 + c

                scores_mg[color_idx] += pst_mg[z][sq]
                scores_eg[color_idx] += pst_eg[z][sq]

                if z == 0: # Pawn
                    left  = grid[r][c-1] if c > 0       else None
                    right = grid[r][c+1] if c < COLS-1  else None
                    if ((left  and left.z_idx == 0  and left.color  == my_color_name) or
                        (right and right.z_idx == 0 and right.color == my_color_name)):
                        scores_mg[color_idx] += PAWN_PHALANX_BONUS

                    is_passed = True
                    if is_white:
                        for pc in (c - 1, c, c + 1):
                            if 0 <= pc < COLS and black_pawn_files[pc]:
                                # Same file: enemy pawn must be strictly ahead (< r)
                                if pc == c and black_pawn_min_row[pc] < r:
                                    is_passed = False; break
                                # Adjacent file: enemy pawn can capture sideways if on the same rank (<= r)
                                elif pc != c and black_pawn_min_row[pc] <= r:
                                    is_passed = False; break
                    else:
                        for pc in (c - 1, c, c + 1):
                            if 0 <= pc < COLS and white_pawn_files[pc]:
                                if pc == c and white_pawn_max_row[pc] > r:
                                    is_passed = False; break
                                elif pc != c and white_pawn_max_row[pc] >= r:
                                    is_passed = False; break
                                    
                    if is_passed:
                        advance = max(0, (6 - r) if is_white else (r - 1))
                        scores_eg[color_idx] += advance * PASSED_PAWN_PER_RANK

                elif z == 3: # Rook
                    if enemy_king and (r == enemy_king[0] or c == enemy_king[1]):
                        scores_mg[color_idx] += ROOK_ALIGNMENT_BONUS
                    my_pawn_files = white_pawn_files if is_white else black_pawn_files
                    if not my_pawn_files[c]:
                        scores_mg[color_idx] += ROOK_OPEN_FILE_BONUS_MG
                        scores_eg[color_idx] += ROOK_OPEN_FILE_BONUS_EG

                    # --- JUNGLE-NATIVE MOBILITY (Piercing) ---
                    mobility = 0
                    for ray in RAYS[sq][:4]: # Orthogonal rays
                        for cr, cc in ray:
                            target = grid[cr][cc]
                            if target is not None:
                                if target.color == my_color_name:
                                    break # Blocked by friendly piece
                            mobility += 1
                    scores_mg[color_idx] += mobility * self.EVAL_MOBILITY_ROOK
                    scores_eg[color_idx] += mobility * self.EVAL_MOBILITY_ROOK

                elif z == 2: # Bishop
                    # --- JUNGLE-NATIVE MOBILITY (Sliding) ---
                    mobility = 0
                    for ray in RAYS[sq][4:]: # Diagonal rays
                        for cr, cc in ray:
                            target = grid[cr][cc]
                            if target is not None:
                                if target.color != my_color_name:
                                    mobility += 1 # Count the capture square
                                break # Stop at any piece
                            mobility += 1
                    scores_mg[color_idx] += mobility * self.EVAL_MOBILITY_BISHOP
                    scores_eg[color_idx] += mobility * self.EVAL_MOBILITY_BISHOP

                elif z == 1: # Knight
                    for ar, ac in KNIGHT_ATTACKS_FROM[(r, c)]:
                        threatened = grid[ar][ac]
                        if threatened and threatened.z_idx != 5 and threatened.color != my_color_name:
                            scores_mg[color_idx] += KNIGHT_ACTIVITY_BONUS
                            scores_eg[color_idx] += KNIGHT_ACTIVITY_BONUS
                    if enemy_king:
                        for ar, ac in KNIGHT_ATTACKS_FROM[(r, c)]:
                            if abs(ar - enemy_king[0]) <= 2 and abs(ac - enemy_king[1]) <= 2:
                                king_zone_attacks[1 - color_idx] += 1; break

                elif z == 4: # Queen
                    if enemy_king and (abs(r - enemy_king[0]) + abs(c - enemy_king[1]) <= 3):
                        king_zone_attacks[1 - color_idx] += 2
                        
                    # --- JUNGLE-NATIVE MOBILITY (Sliding) ---
                    mobility = 0
                    for ray in RAYS[sq]: # All 8 rays
                        for cr, cc in ray:
                            target = grid[cr][cc]
                            if target is not None:
                                if target.color != my_color_name:
                                    mobility += 1 # Count the capture square
                                break # Stop at any piece (capturing explodes her)
                            mobility += 1
                    scores_mg[color_idx] += mobility * self.EVAL_MOBILITY_QUEEN
                    scores_eg[color_idx] += mobility * self.EVAL_MOBILITY_QUEEN

        phase     = min(256, (phase_material_score * 256) // INITIAL_PHASE_MATERIAL) if INITIAL_PHASE_MATERIAL > 0 else 0
        inv_phase = 256 - phase

        for i in (0, 1):
            if king_zone_attacks[i] > 0:
                mg_penalty = (king_zone_attacks[i] * KING_ZONE_ATTACK_PENALTY * phase) >> 8
                scores_mg[i] -= mg_penalty

        total_pawns_on_board = pawn_counts[0] + pawn_counts[1]

        if piece_counts[0] > piece_counts[1]:
            scores_eg[0] += PIECE_DOMINANCE_FACTOR // (piece_counts[1] + 1)
        elif piece_counts[1] > piece_counts[0]:
            scores_eg[1] += PIECE_DOMINANCE_FACTOR // (piece_counts[0] + 1)

        for i in (0, 1):
            if pawn_counts[i] < 4:
                penalty = int(-250 * (4 - pawn_counts[i])**2 / 16)
                scores_mg[i] += penalty; scores_eg[i] += penalty

            if piece_counts[i] == 1 and pawn_counts[i] <= 4: #Not related to TB!
                penalty = 0
                if rook_counts[i] == 1:   penalty = LONE_ROOK_PENALTIES[pawn_counts[i]]
                elif bishop_counts[i] == 1: penalty = LONE_BISHOP_PENALTIES[pawn_counts[i]]
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

            # --- JUNGLE CHESS HEURISTIC: Pawn Vulnerability ---
            # Pawns lose value in the endgame if the opponent has Rooks (pierce) or Knights (evaporate)
            # because they can be safely destroyed even if defended by other pieces.
            enemy_killers = rook_counts[1 - i] + knight_counts[1 - i]
            if enemy_killers > 0:
                vuln_penalty = pawn_counts[i] * enemy_killers * PAWN_VULN_EG
                scores_eg[i] -= vuln_penalty

        if king_pos[0] and king_pos[1]:
            dist = abs(king_pos[0][0] - king_pos[1][0]) + abs(king_pos[0][1] - king_pos[1][1])
            tropism_penalty = (dist * dist * inv_phase * 50) // 50176
            if scores_eg[0] > scores_eg[1]: scores_eg[0] -= tropism_penalty
            elif scores_eg[1] > scores_eg[0]: scores_eg[1] -= tropism_penalty

        mg_score    = scores_mg[0] - scores_mg[1]
        eg_score    = scores_eg[0] - scores_eg[1]
        final_score = (mg_score * phase + eg_score * inv_phase) >> 8

        can_force_mate = [True, True]
        for i in (0, 1):
            if (pawn_counts[i] == 0 and knight_counts[i] == 0 and queen_counts[i] == 0 and
                    (rook_counts[i] + bishop_counts[i]) < 2):
                can_force_mate[i] = False

        if final_score > 0 and not can_force_mate[0]: final_score //= 8
        elif final_score < 0 and not can_force_mate[1]: final_score //= 8

        return final_score if turn_to_move == 'white' else -final_score


# --- Piece-Square Tables ---

pawn_pst = [
    [   0,    0,    0,    0,    0,    0,    0,    0],
    [  90,   90,   90,   90,   90,   90,   90,   90],
    [  50,   50,   50,   50,   55,   50,   50,   50],
    [  25,   30,   30,   45,   50,   30,   30,   25],
    [  15,   15,   20,   30,   35,   20,   15,   15],
    [  10,   10,   20,   25,   30,   20,   10,   10],
    [   0,    0,    0,   -5,  -10,   10,    0,    0],
    [   0,    0,    0,    0,    0,    0,    0,    0]
]

pawn_endgame_pst = [
    [   0,    0,    0,    0,    0,    0,    0,    0],
    [ 160,  160,  160,  160,  160,  160,  160,  160],
    [  80,   85,   85,   85,   85,   85,   85,   80],
    [  45,   50,   50,   55,   55,   50,   50,   45],
    [  25,   30,   30,   35,   35,   30,   30,   25],
    [  10,   15,   15,   20,   20,   15,   15,   10],
    [   0,    5,    5,    5,    5,   15,    5,    0],
    [   0,    0,    0,    0,    0,    0,    0,    0]
]

knight_pst = [
    [-120,  -80,  -60,  -50,  -50,  -60,  -80, -120],
    [ -50,  -10,   30,   40,   40,   30,  -10,  -50],
    [   0,   10,   40,   60,   60,   40,   10,    0],
    [  10,   20,   60,   90,   90,   60,   20,   10],
    [  30,   20,   70,   90,   90,   70,   20,   30],
    [   0,   10,   60,   40,   40,   60,   10,    0],
    [ -80,  -40,   10,   20,   20,   10,  -40,  -80],
    [-120,  -80,  -60,  -60,  -60,  -60,  -80, -120]
]

bishop_pst = [
    [ -30,  -15,  -15,  -15,  -15,  -15,  -15,  -30],
    [ -15,    0,    0,    0,    0,    0,    0,  -15],
    [ -15,    0,    8,   15,   15,    8,    0,  -15],
    [ -15,    8,    8,   15,   15,    8,    8,  -15],
    [ -15,    8,   22,   15,   15,   22,    8,  -15],
    [ -15,   15,   15,    8,    8,   15,   15,  -15],
    [ -15,    8,    0,    0,    0,    0,    8,  -15],
    [ -30,  -15,  -15,  -22,  -22,  -15,  -15,  -30]
]

rook_pst = [
    [  15,   15,   15,   22,   22,   15,   15,   15],
    [  22,   22,   22,   30,   30,   22,   22,   22],
    [   8,    0,    0,    8,    8,    0,    0,    8],
    [   8,    0,    0,    8,    8,    0,    0,    8],
    [   8,    0,    0,    8,    8,    0,    0,    8],
    [   8,    0,    0,    8,    8,    0,    0,    8],
    [   0,    0,    0,    8,    8,    0,    0,    0],
    [  10,   -5,    0,   15,   15,    0,   -5,   10]
]

queen_pst = [
    [ -45,  -30,  -15,  -15,  -15,  -15,  -30,  -45],
    [ -30,    0,    0,    0,    0,    0,    0,  -30],
    [ -15,    0,    8,   15,   15,    8,    0,  -15],
    [  -8,   15,   30,   45,   45,   30,   15,   -8],
    [  -8,    8,   30,   45,   45,   30,    8,   -8],
    [ -15,    8,   22,   22,   22,   22,    8,  -15],
    [ -30,  -15,    0,    8,    8,    0,  -15,  -30],
    [ -45,  -30,  -30,  -15,  -30,  -30,  -30,  -45]
]

king_midgame_pst = [
    [ -90,  -85,  -85,  -85,  -85,  -85,  -85,  -90],
    [ -70,  -75,  -75,  -90,  -90,  -75,  -75,  -70],
    [ -60,  -75,  -75,  -80,  -80,  -75,  -75,  -60],
    [ -60,  -75,  -75,  -80,  -80,  -75,  -75,  -60],
    [ -45,  -60,  -60,  -60,  -60,  -60,  -60,  -45],
    [ -15,  -15,  -30,  -30,  -30,  -30,  -15,  -15],
    [  -8,    0,    8,    8,    8,    8,    0,   -8],
    [ -30,   15,   15,   15,   30,   15,   15,  -30]
]

king_endgame_pst = [
    [ -60,  -45,  -45,  -45,  -45,  -45,  -45,  -60],
    [ -45,  -15,    0,    0,    0,    0,  -15,  -45],
    [ -45,    0,   15,   30,   30,   15,    0,  -45],
    [ -45,    8,   30,   30,   30,   30,    8,  -45],
    [ -45,    8,   22,   30,   30,   22,    8,  -45],
    [ -45,    0,   15,   15,   15,   15,    0,  -45],
    [ -45,    0,    8,    8,    8,    8,    0,  -45],
    [ -60,  -45,  -15,  -15,  -15,  -15,  -45,  -60]
]

PIECE_SQUARE_TABLES = {
    Pawn:           pawn_pst,
    'pawn_endgame': pawn_endgame_pst,
    Knight:         knight_pst,
    Bishop:         bishop_pst,
    Rook:           rook_pst,
    Queen:          queen_pst,
    'king_midgame': king_midgame_pst,
    'king_endgame': king_endgame_pst,
}

FLAT_PST_MG_WHITE = [None] * 6
FLAT_PST_MG_BLACK = [None] * 6
FLAT_PST_EG_WHITE = [None] * 6
FLAT_PST_EG_BLACK = [None] * 6

for pt in [Pawn, Knight, Bishop, Rook, Queen, King]:
    z = pt.z_idx
    FLAT_PST_MG_WHITE[z] = [0] * 64
    FLAT_PST_MG_BLACK[z] = [0] * 64
    FLAT_PST_EG_WHITE[z] = [0] * 64
    FLAT_PST_EG_BLACK[z] = [0] * 64
    
    mg_val = MG_PIECE_VALUES[pt]
    eg_val = EG_PIECE_VALUES[pt]
    
    if pt == Pawn:
        mg_table = PIECE_SQUARE_TABLES[Pawn]
        eg_table = PIECE_SQUARE_TABLES['pawn_endgame']
    elif pt == King:
        mg_table = PIECE_SQUARE_TABLES['king_midgame']
        eg_table = PIECE_SQUARE_TABLES['king_endgame']
    else:
        mg_table = PIECE_SQUARE_TABLES[pt]
        eg_table = PIECE_SQUARE_TABLES[pt]
        
    for r in range(8):
        for c in range(8):
            sq_w = r * 8 + c
            sq_b = (7 - r) * 8 + c
            FLAT_PST_MG_WHITE[z][sq_w] = mg_val + mg_table[r][c]
            FLAT_PST_MG_BLACK[z][sq_b] = mg_val + mg_table[r][c]
            FLAT_PST_EG_WHITE[z][sq_w] = eg_val + eg_table[r][c]
            FLAT_PST_EG_BLACK[z][sq_b] = eg_val + eg_table[r][c]