# AI.py (Max challenger v1)
# Variant-aware search/eval engine for Jungle Chess.

import time
import random
import os
import json
import glob
from collections import defaultdict

import GameLogic as GL
from GameLogic import *
from TablebaseManager import TablebaseManager

# ==============================================================================
# MODULE-LEVEL SETUP (DO NOT MODIFY)
# The UI's multiprocessing worker relies on these exact function signatures.
# ==============================================================================

ZOBRIST_ARRAY = None
ZOBRIST_TURN = None

def initialize_zobrist_table():
    global ZOBRIST_ARRAY, ZOBRIST_TURN
    if ZOBRIST_ARRAY is not None:
        return
    random.seed(42)
    ZOBRIST_ARRAY = [[[[random.getrandbits(64) for _ in range(8)] for _ in range(8)] for _ in range(6)] for _ in range(2)]
    ZOBRIST_TURN = random.getrandbits(64)
    random.seed()

initialize_zobrist_table()

def board_hash(board, turn):
    h = 0
    arr = ZOBRIST_ARRAY
    for piece in board.white_pieces:
        r, c = piece.pos
        h ^= arr[0][piece.z_idx][r][c]
    for piece in board.black_pieces:
        r, c = piece.pos
        h ^= arr[1][piece.z_idx][r][c]
    if turn == 'black':
        h ^= ZOBRIST_TURN
    return h

def incremental_hash(parent_hash, record_tuple):
    h = parent_hash ^ ZOBRIST_TURN
    arr = ZOBRIST_ARRAY
    start, end, mp, removed_pieces, added_pieces = record_tuple
    c_idx = 0 if mp.color == 'white' else 1
    p_idx = mp.z_idx
    sr, sc = start
    er, ec = end
    h ^= arr[c_idx][p_idx][sr][sc]
    mp_survived = True
    for piece, r, c in removed_pieces:
        if piece is mp:
            mp_survived = False
        else:
            h ^= arr[0 if piece.color == 'white' else 1][piece.z_idx][r][c]
    if mp_survived:
        h ^= arr[c_idx][p_idx][er][ec]
    for piece, r, c in added_pieces:
        h ^= arr[0 if piece.color == 'white' else 1][piece.z_idx][r][c]
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
            if norm not in seen:
                seen.add(norm)
                matches.append(path)
    return sorted(matches, key=lambda p: (os.path.getmtime(p), os.path.basename(p)), reverse=True)

for _book_filename in _find_opening_book_files():
    try:
        with open(_book_filename, "r", encoding="utf-8") as f:
            OPENING_BOOK = json.load(f)
        break
    except Exception:
        pass

def run_ai_process(board, color, position_counts, comm_queue, cancellation_event,
                   bot_class, bot_name, search_depth, ply_count, game_mode,
                   time_left=None, increment=None, use_opening_book=True, use_tablebase=True):
    import inspect
    accepted_params = set(inspect.signature(bot_class.__init__).parameters)
    kwargs = {
        'time_left': time_left,
        'increment': increment,
        'use_opening_book': use_opening_book,
        'use_tablebase': use_tablebase
    }
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted_params}
    bot = bot_class(board, color, position_counts, comm_queue, cancellation_event, bot_name, ply_count, game_mode, **filtered_kwargs)
    bot.search_depth = search_depth
    if search_depth == 99:
        bot.ponder_indefinitely()
    else:
        bot.make_move()

class SearchCancelledException(Exception):
    pass


# ==============================================================================
# ENGINE CONSTANTS
# ==============================================================================

INF = 10**12
TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2

PIECE_VALUES = [108, 430, 390, 760, 780, 0]
PHASE_WEIGHTS = [0, 1, 1, 2, 4, 0]
MOBILITY_WEIGHTS = [1, 4, 3, 2, 2, 1]
MAX_PHASE = 24
TEMPO_BONUS = 12

# Piece-square tables from White's perspective (row 0 = Black back rank).
PAWN_PST = (
    (0, 0, 0, 0, 0, 0, 0, 0),
    (62, 68, 72, 78, 78, 72, 68, 62),
    (42, 48, 54, 60, 60, 54, 48, 42),
    (28, 34, 40, 48, 48, 40, 34, 28),
    (18, 24, 30, 36, 36, 30, 24, 18),
    (8, 12, 18, 24, 24, 18, 12, 8),
    (2, 4, 8, 12, 12, 8, 4, 2),
    (0, 0, 0, 0, 0, 0, 0, 0),
)

KNIGHT_PST = (
    (-36, -20, -12, -10, -10, -12, -20, -36),
    (-20, -4, 6, 10, 10, 6, -4, -20),
    (-12, 6, 16, 20, 20, 16, 6, -12),
    (-10, 10, 20, 24, 24, 20, 10, -10),
    (-10, 10, 20, 24, 24, 20, 10, -10),
    (-12, 6, 16, 20, 20, 16, 6, -12),
    (-20, -4, 6, 10, 10, 6, -4, -20),
    (-36, -20, -12, -10, -10, -12, -20, -36),
)

BISHOP_PST = (
    (-18, -10, -8, -6, -6, -8, -10, -18),
    (-10, 0, 4, 8, 8, 4, 0, -10),
    (-8, 4, 10, 14, 14, 10, 4, -8),
    (-6, 8, 14, 18, 18, 14, 8, -6),
    (-6, 8, 14, 18, 18, 14, 8, -6),
    (-8, 4, 10, 14, 14, 10, 4, -8),
    (-10, 0, 4, 8, 8, 4, 0, -10),
    (-18, -10, -8, -6, -6, -8, -10, -18),
)

ROOK_PST = (
    (4, 6, 8, 10, 10, 8, 6, 4),
    (8, 10, 12, 16, 16, 12, 10, 8),
    (2, 4, 8, 12, 12, 8, 4, 2),
    (0, 2, 6, 10, 10, 6, 2, 0),
    (0, 2, 6, 10, 10, 6, 2, 0),
    (2, 4, 8, 12, 12, 8, 4, 2),
    (8, 10, 12, 16, 16, 12, 10, 8),
    (4, 6, 8, 10, 10, 8, 6, 4),
)

QUEEN_PST = (
    (-10, -6, -4, -2, -2, -4, -6, -10),
    (-6, 0, 2, 4, 4, 2, 0, -6),
    (-4, 2, 6, 8, 8, 6, 2, -4),
    (-2, 4, 8, 10, 10, 8, 4, -2),
    (-2, 4, 8, 10, 10, 8, 4, -2),
    (-4, 2, 6, 8, 8, 6, 2, -4),
    (-6, 0, 2, 4, 4, 2, 0, -6),
    (-10, -6, -4, -2, -2, -4, -6, -10),
)

KING_MID_PST = (
    (26, 30, 16, 4, 4, 16, 30, 26),
    (20, 18, 6, -6, -6, 6, 18, 20),
    (8, 0, -10, -18, -18, -10, 0, 8),
    (-2, -12, -20, -28, -28, -20, -12, -2),
    (-4, -14, -22, -30, -30, -22, -14, -4),
    (-2, -10, -18, -24, -24, -18, -10, -2),
    (8, 4, -4, -12, -12, -4, 4, 8),
    (18, 16, 8, 0, 0, 8, 16, 18),
)

KING_END_PST = (
    (-16, -8, 0, 8, 8, 0, -8, -16),
    (-8, 0, 8, 16, 16, 8, 0, -8),
    (0, 8, 16, 24, 24, 16, 8, 0),
    (8, 16, 24, 30, 30, 24, 16, 8),
    (8, 16, 24, 30, 30, 24, 16, 8),
    (0, 8, 16, 24, 24, 16, 8, 0),
    (-8, 0, 8, 16, 16, 8, 0, -8),
    (-16, -8, 0, 8, 8, 0, -8, -16),
)

PST_BY_Z = {
    0: PAWN_PST,
    1: KNIGHT_PST,
    2: BISHOP_PST,
    3: ROOK_PST,
    4: QUEEN_PST,
}


# ==============================================================================
# CHESS BOT CLASS
# ==============================================================================

class ChessBot:
    search_depth = 4  # Fixes the UI boot AttributeError
    MATE_SCORE = 1000000
    MATE_TT_THRESHOLD = 990000
    TT_MAX = 350000
    EVAL_CACHE_MAX = 100000
    MAX_PLY = 128

    def __init__(self, board, color, position_counts, comm_queue, cancellation_event,
                 bot_name="Challenger AI", ply_count=0, game_mode="bot", max_moves=200,
                 time_left=None, increment=None, use_opening_book=True, use_tablebase=True):

        self.board = board
        self.color = color
        self.opponent_color = 'black' if color == 'white' else 'white'
        self.position_counts = position_counts if position_counts is not None else {}
        self.comm_queue = comm_queue
        self.cancellation_event = cancellation_event
        self.ply_count = ply_count
        self.bot_name = bot_name
        self.max_moves = max_moves

        # Time management
        self.time_left = time_left
        self.increment = increment
        self.stop_time = None          # hard stop
        self.soft_stop_time = None     # soft stop
        self.search_depth = 4
        self.nodes_searched = 0

        # Search state
        self.tt = {}
        self.eval_cache = {}
        self.local_hash_counts = defaultdict(int)
        self.killers = [[None, None] for _ in range(self.MAX_PLY + 8)]
        self.history = [[[[0 for _ in range(8)] for _ in range(8)] for _ in range(6)] for _ in range(2)]
        self.root_order_hint = {}
        self.last_completed_depth = 0
        self.last_pv_moves = []
        self.last_pv_san = []

        # Databases
        self.use_opening_book = use_opening_book
        self.tb_manager = TablebaseManager()
        if not use_tablebase:
            self.tb_manager.probe = lambda b, t: None

    # --- UI COMMUNICATION HELPERS (Do not modify) ---
    def _report_log(self, message):
        self.comm_queue.put(('log', message))

    def _report_eval(self, score, depth):
        self.comm_queue.put(('eval', score if self.color == 'white' else -score, depth))

    def _report_move(self, move):
        self.comm_queue.put(('move', move))

    def _format_move(self, board_before, move):
        if not move:
            return "None"
        child = board_before.clone()
        child.make_move(move[0], move[1])
        return format_move_san(board_before, child, move)

    # --- MAIN ENTRY POINT ---
    def make_move(self):
        root_moves = None
        try:
            if self.use_opening_book and self.ply_count <= 12:
                fen = board_to_fen(self.board, self.color)
                if fen in OPENING_BOOK:
                    chosen = random.choices(
                        OPENING_BOOK[fen],
                        weights=[opt["weight"] for opt in OPENING_BOOK[fen]],
                        k=1
                    )[0]
                    move_tuple = (tuple(chosen["move"][0]), tuple(chosen["move"][1]))
                    self._report_log(f"  > {self.bot_name} (Book): {chosen['san']}")
                    self._report_eval(chosen['score'], "Book")
                    self.comm_queue.put(('pv', chosen['score'], "Book", [chosen['san']], [move_tuple]))
                    self._report_move(move_tuple)
                    return

            root_moves = get_all_legal_moves(self.board, self.color)
            if not root_moves:
                self._report_move(None)
                return

            if len(root_moves) == 1:
                san = self._format_move(self.board, root_moves[0])
                self.comm_queue.put(('pv', 0, "Forced", [san], [root_moves[0]]))
                self._report_move(root_moves[0])
                return

            search_start = time.time()
            target_depth = self.search_depth
            if self.time_left is not None and self.increment is not None:
                self._allocate_time(len(root_moves))
                target_depth = 64
            else:
                self.soft_stop_time = None
                self.stop_time = None

            best_move_overall = root_moves[0]
            best_score_overall = -INF
            root_hash = board_hash(self.board, self.color)

            previous_score = 0
            last_iter_duration = 0.0

            for current_depth in range(1, target_depth + 1):
                if self.cancellation_event.is_set():
                    break
                if self.soft_stop_time and current_depth > 1 and time.time() >= self.soft_stop_time:
                    break

                iter_start = time.time()
                self.nodes_searched = 0

                try:
                    if current_depth >= 3:
                        window = 60 if current_depth < 6 else 110
                        alpha = previous_score - window
                        beta = previous_score + window
                        while True:
                            score, best_move_this_iter, new_root_order = self._search_root(
                                current_depth, root_moves, root_hash, alpha, beta
                            )
                            if self.cancellation_event.is_set() or self._hard_time_exceeded():
                                raise SearchCancelledException()

                            if score <= alpha:
                                alpha -= window
                                window *= 2
                                continue
                            if score >= beta:
                                beta += window
                                window *= 2
                                continue
                            break
                    else:
                        score, best_move_this_iter, new_root_order = self._search_root(
                            current_depth, root_moves, root_hash, -INF, INF
                        )
                except SearchCancelledException:
                    break

                best_move_overall = best_move_this_iter
                best_score_overall = score
                previous_score = score
                root_moves = new_root_order
                self.last_completed_depth = current_depth

                pv_moves = self._extract_pv(root_hash, self.color, max_len=min(8, current_depth + 2))
                pv_san = self._pv_to_san(pv_moves)
                self.last_pv_moves = pv_moves
                self.last_pv_san = pv_san

                iter_duration = time.time() - iter_start
                last_iter_duration = iter_duration
                knps = (self.nodes_searched / max(iter_duration, 1e-9)) / 1000.0
                eval_ui = score if self.color == 'white' else -score
                pv_head = pv_san[0] if pv_san else self._format_move(self.board, best_move_overall)

                self._report_log(
                    f"  > {self.bot_name} (D{current_depth}): {pv_head}, "
                    f"Eval={eval_ui/100:+.2f}, Nodes={self.nodes_searched}, "
                    f"KNPS={knps:.1f}, Time={iter_duration:.2f}s"
                )
                self._report_eval(score, current_depth)
                self.comm_queue.put(('pv', eval_ui, current_depth, pv_san, pv_moves))

                if abs(score) > self.MATE_SCORE - 2000:
                    break

                if self.soft_stop_time:
                    now = time.time()
                    if now >= self.soft_stop_time:
                        break
                    remaining = self.soft_stop_time - now
                    if current_depth >= 3 and last_iter_duration * 1.8 > remaining:
                        break

            self._report_move(best_move_overall)

        except Exception as e:
            self._report_log(f"CRASH: {str(e)}")
            fallback = root_moves[0] if root_moves else None
            self._report_move(fallback)

    # ------------------------------------------------------------------
    # TIME MANAGEMENT
    # ------------------------------------------------------------------

    def _allocate_time(self, root_move_count):
        now = time.time()
        tl = max(0.01, float(self.time_left))
        inc = max(0.0, float(self.increment))

        remaining_fullmoves = max(8, 36 - (self.ply_count // 2))
        base = tl / (remaining_fullmoves + 4)
        alloc = (base * 0.9) + (inc * 0.85)

        if root_move_count <= 4:
            alloc *= 0.75
        elif root_move_count >= 24:
            alloc *= 1.20

        if tl < 2.0:
            alloc = min(alloc, 0.24 + inc * 0.6)
        if tl < 0.8:
            alloc = min(alloc, 0.08 + inc * 0.5)
        if tl < 0.3:
            alloc = min(alloc, 0.03 + inc * 0.35)

        alloc = max(0.015, min(alloc, tl * 0.30))
        hard = min(max(alloc * 2.2, alloc + 0.03), tl * 0.75)

        self.soft_stop_time = now + alloc
        self.stop_time = now + hard

    def _hard_time_exceeded(self):
        return self.stop_time is not None and time.time() >= self.stop_time

    def _periodic_abort_check(self):
        if self.cancellation_event.is_set() or self._hard_time_exceeded():
            raise SearchCancelledException()

    # ------------------------------------------------------------------
    # ROOT SEARCH
    # ------------------------------------------------------------------

    def _search_root(self, depth, root_moves, root_hash, alpha, beta):
        alpha_orig = alpha
        beta_orig = beta
        best_score = -INF
        best_move = root_moves[0]

        tt_entry = self.tt.get(root_hash)
        tt_move = tt_entry[3] if tt_entry else None
        ordered = self._order_moves(root_moves, self.color, 0, tt_move, root=True)

        move_results = []
        searched_moves = set()

        for index, info in enumerate(ordered):
            self._periodic_abort_check()
            move = info[1]
            searched_moves.add(move)

            record = self.board.make_move_track(move[0], move[1])
            child_hash = incremental_hash(root_hash, record)
            self.local_hash_counts[child_hash] += 1
            try:
                if index == 0:
                    score = -self.negamax(depth - 1, -beta, -alpha, self.opponent_color, 1, child_hash)
                else:
                    score = -self.negamax(depth - 1, -alpha - 1, -alpha, self.opponent_color, 1, child_hash)
                    if score > alpha and score < beta:
                        score = -self.negamax(depth - 1, -beta, -alpha, self.opponent_color, 1, child_hash)
            finally:
                self.local_hash_counts[child_hash] -= 1
                if self.local_hash_counts[child_hash] <= 0:
                    del self.local_hash_counts[child_hash]
                self.board.unmake_move(record)

            move_results.append((score, move))

            if score > best_score:
                best_score = score
                best_move = move

            if score > alpha:
                alpha = score

            if alpha >= beta:
                break

        if len(move_results) > 1:
            root_sc_map = {m: s for s, m in move_results}
            self.root_order_hint = root_sc_map
        else:
            self.root_order_hint = {best_move: best_score}

        ordered_finished = [m for _, m in sorted(move_results, key=lambda x: x[0], reverse=True)]
        for m in root_moves:
            if m not in searched_moves and m not in ordered_finished:
                ordered_finished.append(m)
        for _, m, *_ in ordered:
            if m not in ordered_finished:
                ordered_finished.append(m)

        flag = TT_EXACT
        if best_score <= alpha_orig:
            flag = TT_UPPER
        elif best_score >= beta_orig:
            flag = TT_LOWER
        self._store_tt(root_hash, depth, flag, best_score, best_move, 0, rep_sensitive=False)

        return best_score, best_move, ordered_finished

    # ------------------------------------------------------------------
    # NEGAMAX + QUIESCENCE
    # ------------------------------------------------------------------

    def negamax(self, depth, alpha, beta, turn, ply, current_hash):
        self.nodes_searched += 1
        if (self.nodes_searched & 1023) == 0:
            self._periodic_abort_check()

        if ply >= self.MAX_PLY:
            return self.evaluate_board(turn, current_hash)

        rep_count = self.position_counts.get(current_hash, 0) + self.local_hash_counts.get(current_hash, 0)
        if rep_count >= 3:
            return 0
        rep_sensitive = rep_count > 1

        if self.ply_count + ply >= self.max_moves:
            return 0

        if is_insufficient_material(self.board):
            return 0

        if len(self.board.white_pieces) + len(self.board.black_pieces) <= 5:
            tb_score_absolute = self.tb_manager.probe(self.board, turn)
            if tb_score_absolute is not None:
                tb_score = tb_score_absolute if turn == 'white' else -tb_score_absolute
                if tb_score > self.MATE_SCORE - 1000:
                    return tb_score - ply
                if tb_score < -self.MATE_SCORE + 1000:
                    return tb_score + ply
                return tb_score

        if depth <= 0:
            return self.quiescence(alpha, beta, turn, ply, current_hash)

        alpha_orig = alpha
        beta_orig = beta

        tt_entry = self.tt.get(current_hash)
        tt_move = None
        if tt_entry is not None:
            tt_depth, tt_flag, tt_score_raw, tt_move = tt_entry
            if tt_depth >= depth:
                tt_score = self._score_from_tt(tt_score_raw, ply)
                if tt_flag == TT_EXACT:
                    return tt_score
                if tt_flag == TT_LOWER:
                    alpha = max(alpha, tt_score)
                else:
                    beta = min(beta, tt_score)
                if alpha >= beta:
                    return tt_score

        in_check = is_in_check(self.board, turn)
        if in_check and depth < self.MAX_PLY - 4:
            depth += 1

        opponent = 'black' if turn == 'white' else 'white'
        pseudo_moves = get_all_pseudo_legal_moves(self.board, turn)
        ordered = self._order_moves(pseudo_moves, turn, ply, tt_move, root=False)

        best_score = -INF
        best_move = None
        legal_count = 0

        for move_index, info in enumerate(ordered):
            move = info[1]
            approx_swing = info[2]
            is_tactic = info[3]
            is_quiet = info[4]
            is_promo = info[6]
            piece_z = info[7]

            record = self.board.make_move_track(move[0], move[1])
            my_kp = self.board.white_king_pos if turn == 'white' else self.board.black_king_pos
            legal = (my_kp is not None and not is_square_attacked(self.board, my_kp[0], my_kp[1], opponent))
            if not legal:
                self.board.unmake_move(record)
                continue

            legal_count += 1
            child_hash = incremental_hash(current_hash, record)
            self.local_hash_counts[child_hash] += 1

            try:
                new_depth = depth - 1
                reduction = 0

                if (depth >= 4 and legal_count > 4 and not in_check and is_quiet and
                        not is_promo and piece_z != 5):
                    reduction = 1
                    if depth >= 6 and legal_count > 8:
                        reduction = 2 if approx_swing < 80 else 1

                if move_index == 0:
                    score = -self.negamax(new_depth, -beta, -alpha, opponent, ply + 1, child_hash)
                else:
                    if reduction > 0:
                        score = -self.negamax(new_depth - reduction, -alpha - 1, -alpha, opponent, ply + 1, child_hash)
                        if score > alpha:
                            score = -self.negamax(new_depth, -alpha - 1, -alpha, opponent, ply + 1, child_hash)
                            if score > alpha and score < beta:
                                score = -self.negamax(new_depth, -beta, -alpha, opponent, ply + 1, child_hash)
                    else:
                        score = -self.negamax(new_depth, -alpha - 1, -alpha, opponent, ply + 1, child_hash)
                        if score > alpha and score < beta:
                            score = -self.negamax(new_depth, -beta, -alpha, opponent, ply + 1, child_hash)
            finally:
                self.local_hash_counts[child_hash] -= 1
                if self.local_hash_counts[child_hash] <= 0:
                    del self.local_hash_counts[child_hash]
                self.board.unmake_move(record)

            if score > best_score:
                best_score = score
                best_move = move

            if score > alpha:
                alpha = score

            if alpha >= beta:
                if is_quiet:
                    self._update_killers_and_history(turn, ply, move, depth, piece_z)
                break

        if legal_count == 0:
            return -self.MATE_SCORE + ply

        flag = TT_EXACT
        if best_score <= alpha_orig:
            flag = TT_UPPER
        elif best_score >= beta_orig:
            flag = TT_LOWER
        self._store_tt(current_hash, depth, flag, best_score, best_move, ply, rep_sensitive)

        return best_score

    def quiescence(self, alpha, beta, turn, ply, current_hash):
        self.nodes_searched += 1
        if (self.nodes_searched & 1023) == 0:
            self._periodic_abort_check()

        if ply >= self.MAX_PLY:
            return self.evaluate_board(turn, current_hash)

        rep_count = self.position_counts.get(current_hash, 0) + self.local_hash_counts.get(current_hash, 0)
        if rep_count >= 3:
            return 0
        rep_sensitive = rep_count > 1

        if self.ply_count + ply >= self.max_moves:
            return 0

        if is_insufficient_material(self.board):
            return 0

        if len(self.board.white_pieces) + len(self.board.black_pieces) <= 5:
            tb_score_absolute = self.tb_manager.probe(self.board, turn)
            if tb_score_absolute is not None:
                tb_score = tb_score_absolute if turn == 'white' else -tb_score_absolute
                if tb_score > self.MATE_SCORE - 1000:
                    return tb_score - ply
                if tb_score < -self.MATE_SCORE + 1000:
                    return tb_score + ply
                return tb_score

        tt_entry = self.tt.get(current_hash)
        tt_move = None
        if tt_entry is not None:
            tt_depth, tt_flag, tt_score_raw, tt_move = tt_entry
            if tt_depth >= 0:
                tt_score = self._score_from_tt(tt_score_raw, ply)
                if tt_flag == TT_EXACT:
                    return tt_score
                if tt_flag == TT_LOWER:
                    alpha = max(alpha, tt_score)
                else:
                    beta = min(beta, tt_score)
                if alpha >= beta:
                    return tt_score

        if not has_legal_moves(self.board, turn):
            return -self.MATE_SCORE + ply

        in_check = is_in_check(self.board, turn)
        alpha_orig = alpha
        beta_orig = beta

        if not in_check:
            stand_pat = self.evaluate_board(turn, current_hash)
            if stand_pat >= beta:
                return stand_pat
            if stand_pat > alpha:
                alpha = stand_pat
            best_score = stand_pat
        else:
            best_score = -INF

        opponent = 'black' if turn == 'white' else 'white'
        pseudo_moves = get_all_pseudo_legal_moves(self.board, turn)
        ordered = self._order_moves(pseudo_moves, turn, ply, tt_move, root=False)

        best_move = None
        legal_count = 0
        enemy_king = self.board.black_king_pos if turn == 'white' else self.board.white_king_pos

        for info in ordered:
            move = info[1]
            approx_swing = info[2]
            is_tactic = info[3]
            is_quiet = info[4]
            is_checkish = info[5]
            is_promo = info[6]

            if not in_check:
                if not is_tactic and not is_checkish:
                    continue
                if approx_swing < -120 and not is_promo:
                    continue
                if best_score > -INF and best_score + approx_swing + 90 <= alpha and not is_checkish:
                    continue

            record = self.board.make_move_track(move[0], move[1])
            my_kp = self.board.white_king_pos if turn == 'white' else self.board.black_king_pos
            legal = (my_kp is not None and not is_square_attacked(self.board, my_kp[0], my_kp[1], opponent))
            if not legal:
                self.board.unmake_move(record)
                continue

            legal_count += 1

            if not in_check and not is_tactic:
                # Verify quiet checking candidates after legality.
                opp_kp = self.board.black_king_pos if turn == 'white' else self.board.white_king_pos
                if opp_kp is None or not is_square_attacked(self.board, opp_kp[0], opp_kp[1], turn):
                    self.board.unmake_move(record)
                    continue

            child_hash = incremental_hash(current_hash, record)
            self.local_hash_counts[child_hash] += 1
            try:
                score = -self.quiescence(-beta, -alpha, opponent, ply + 1, child_hash)
            finally:
                self.local_hash_counts[child_hash] -= 1
                if self.local_hash_counts[child_hash] <= 0:
                    del self.local_hash_counts[child_hash]
                self.board.unmake_move(record)

            if score > best_score:
                best_score = score
                best_move = move

            if score > alpha:
                alpha = score
            if alpha >= beta:
                break

        if in_check and legal_count == 0:
            return -self.MATE_SCORE + ply

        flag = TT_EXACT
        if best_score <= alpha_orig:
            flag = TT_UPPER
        elif best_score >= beta_orig:
            flag = TT_LOWER
        self._store_tt(current_hash, 0, flag, best_score, best_move, ply, rep_sensitive)

        return best_score

    # ------------------------------------------------------------------
    # MOVE ORDERING
    # ------------------------------------------------------------------

    def _order_moves(self, moves, turn, ply, tt_move, root=False):
        board = self.board
        grid = board.grid
        enemy_king = board.black_king_pos if turn == 'white' else board.white_king_pos
        color_idx = 0 if turn == 'white' else 1

        ordered = []
        for move in moves:
            start, end = move
            piece = grid[start[0]][start[1]]
            if piece is None or piece.color != turn:
                continue

            target = grid[end[0]][end[1]]
            approx_swing, is_tactic = self._approx_move_swing(move, piece, target, enemy_king)
            is_promo = (piece.z_idx == 0 and end[0] == piece.promo_rank)
            is_checkish = self._approx_gives_check_after_move(piece, start, end, target, enemy_king)
            is_quiet = not is_tactic and not is_promo and target is None and not is_checkish

            score = 0

            if tt_move is not None and move == tt_move:
                score += 2_000_000_000

            if root and move in self.root_order_hint:
                score += 1_000_000_000 + int(self.root_order_hint[move] * 8)

            if is_tactic or is_promo or is_checkish:
                score += 20_000_000
                score += approx_swing * 16
                if is_checkish:
                    score += 8000
                if is_promo:
                    score += 12000
            else:
                k0, k1 = self.killers[ply]
                if move == k0:
                    score += 9_000_000
                elif move == k1:
                    score += 8_000_000
                score += self.history[color_idx][piece.z_idx][end[0]][end[1]]
                score += self._quiet_positional_bonus(piece, start, end) * 8

            ordered.append((score, move, approx_swing, is_tactic, is_quiet, is_checkish, is_promo, piece.z_idx))

        ordered.sort(key=lambda x: x[0], reverse=True)
        return ordered

    def _approx_move_swing(self, move, piece, target_piece, enemy_king):
        swing, is_tactic = fast_approximate_material_swing(self.board, move, piece, target_piece, PIECE_VALUES)

        start, end = move
        z = piece.z_idx

        if enemy_king is not None:
            if z == 1 and enemy_king in KNIGHT_ATTACKS_FROM[end]:
                swing += 65
                is_tactic = True
            elif z == 3 and self._rook_attacks_square_after_move(start, end, piece.color, enemy_king):
                swing += 80
                is_tactic = True
            elif z == 4:
                if target_piece is not None and abs(end[0] - enemy_king[0]) <= 1 and abs(end[1] - enemy_king[1]) <= 1:
                    swing += 110
                    is_tactic = True
                elif self._queen_standard_attacks_square_after_move(start, end, enemy_king):
                    swing += 45
                    is_tactic = True
            elif z == 2 and self._bishop_standard_attacks_square_after_move(start, end, enemy_king):
                swing += 35
                is_tactic = True
            elif z == 0 and self._pawn_checks_square_from(end, piece.color, enemy_king):
                swing += 28
                is_tactic = True

        return swing, is_tactic

    def _approx_gives_check_after_move(self, piece, start, end, target_piece, enemy_king):
        if enemy_king is None:
            return False
        z = piece.z_idx
        if z == 1:
            return enemy_king in KNIGHT_ATTACKS_FROM[end]
        if z == 3:
            return self._rook_attacks_square_after_move(start, end, piece.color, enemy_king)
        if z == 4:
            if target_piece is not None and abs(end[0] - enemy_king[0]) <= 1 and abs(end[1] - enemy_king[1]) <= 1:
                return True
            return self._queen_standard_attacks_square_after_move(start, end, enemy_king)
        if z == 2:
            return self._bishop_standard_attacks_square_after_move(start, end, enemy_king)
        if z == 0:
            return self._pawn_checks_square_from(end, piece.color, enemy_king)
        return False

    def _rook_attacks_square_after_move(self, start, end, color, target):
        sr, sc = end
        tr, tc = target
        if sr != tr and sc != tc:
            return False
        if sr == tr:
            step = 1 if tc > sc else -1
            for c in range(sc + step, tc, step):
                if (sr, c) == start:
                    continue
                p = self.board.grid[sr][c]
                if p is not None and p.color == color:
                    return False
            return True
        step = 1 if tr > sr else -1
        for r in range(sr + step, tr, step):
            if (r, sc) == start:
                continue
            p = self.board.grid[r][sc]
            if p is not None and p.color == color:
                return False
        return True

    def _queen_standard_attacks_square_after_move(self, start, end, target):
        er, ec = end
        tr, tc = target
        dr = tr - er
        dc = tc - ec
        if dr == 0 and dc == 0:
            return False

        step_r = 0
        step_c = 0
        if dr == 0:
            step_c = 1 if dc > 0 else -1
        elif dc == 0:
            step_r = 1 if dr > 0 else -1
        elif abs(dr) == abs(dc):
            step_r = 1 if dr > 0 else -1
            step_c = 1 if dc > 0 else -1
        else:
            return False

        r, c = er + step_r, ec + step_c
        while (r, c) != (tr, tc):
            if (r, c) != start and self.board.grid[r][c] is not None:
                return False
            r += step_r
            c += step_c
        return True

    def _bishop_standard_attacks_square_after_move(self, start, end, target):
        er, ec = end
        tr, tc = target
        dr = tr - er
        dc = tc - ec
        if abs(dr) != abs(dc) or dr == 0:
            return False
        step_r = 1 if dr > 0 else -1
        step_c = 1 if dc > 0 else -1
        r, c = er + step_r, ec + step_c
        while (r, c) != (tr, tc):
            if (r, c) != start and self.board.grid[r][c] is not None:
                return False
            r += step_r
            c += step_c
        return True

    def _pawn_checks_square_from(self, pos, color, target):
        r, c = pos
        tr, tc = target
        direction = -1 if color == 'white' else 1

        if tr == r + direction and tc == c:
            return True
        if tr == r and (tc == c - 1 or tc == c + 1):
            return True

        starting_row = 6 if color == 'white' else 1
        if r == starting_row + direction and tc == c and tr == r + direction:
            return True
        return False

    def _quiet_positional_bonus(self, piece, start, end):
        z = piece.z_idx
        if z == 5:
            return 0
        start_ps = self._pst_value(z, piece.color, start[0], start[1], MAX_PHASE)
        end_ps = self._pst_value(z, piece.color, end[0], end[1], MAX_PHASE)
        return end_ps - start_ps

    def _update_killers_and_history(self, turn, ply, move, depth, piece_z):
        if ply < len(self.killers):
            if self.killers[ply][0] != move:
                self.killers[ply][1] = self.killers[ply][0]
                self.killers[ply][0] = move
        color_idx = 0 if turn == 'white' else 1
        bonus = depth * depth + 2 * depth
        r, c = move[1]
        self.history[color_idx][piece_z][r][c] += bonus
        if self.history[color_idx][piece_z][r][c] > 500000:
            for rr in range(8):
                for cc in range(8):
                    self.history[color_idx][piece_z][rr][cc] //= 2

    # ------------------------------------------------------------------
    # TRANSPOSITION TABLE
    # ------------------------------------------------------------------

    def _score_to_tt(self, score, ply):
        if score > self.MATE_TT_THRESHOLD:
            return score + ply
        if score < -self.MATE_TT_THRESHOLD:
            return score - ply
        return score

    def _score_from_tt(self, score, ply):
        if score > self.MATE_TT_THRESHOLD:
            return score - ply
        if score < -self.MATE_TT_THRESHOLD:
            return score + ply
        return score

    def _store_tt(self, key, depth, flag, score, move, ply, rep_sensitive):
        if rep_sensitive or move is None:
            return
        if len(self.tt) > self.TT_MAX:
            self.tt.clear()
        old = self.tt.get(key)
        if old is None or depth >= old[0]:
            self.tt[key] = (depth, flag, self._score_to_tt(score, ply), move)

    # ------------------------------------------------------------------
    # PV EXTRACTION
    # ------------------------------------------------------------------

    def _extract_pv(self, root_hash, turn, max_len=8):
        pv = []
        records = []
        seen_hashes = set()
        current_hash = root_hash
        side = turn

        try:
            for _ in range(max_len):
                entry = self.tt.get(current_hash)
                if entry is None:
                    break
                move = entry[3]
                if move is None:
                    break
                if current_hash in seen_hashes:
                    break
                seen_hashes.add(current_hash)

                start, end = move
                piece = self.board.grid[start[0]][start[1]]
                if piece is None or piece.color != side:
                    break

                record = self.board.make_move_track(start, end)
                my_kp = self.board.white_king_pos if side == 'white' else self.board.black_king_pos
                opponent = 'black' if side == 'white' else 'white'
                legal = (my_kp is not None and not is_square_attacked(self.board, my_kp[0], my_kp[1], opponent))
                if not legal:
                    self.board.unmake_move(record)
                    break

                pv.append(move)
                records.append(record)
                current_hash = incremental_hash(current_hash, record)
                side = opponent
        finally:
            for record in reversed(records):
                self.board.unmake_move(record)

        return pv

    def _pv_to_san(self, pv_moves):
        if not pv_moves:
            return []
        temp = self.board.clone()
        san_list = []
        for move in pv_moves:
            before = temp.clone()
            temp.make_move(move[0], move[1])
            san_list.append(format_move_san(before, temp, move))
        return san_list

    # ------------------------------------------------------------------
    # EVALUATION
    # ------------------------------------------------------------------

    def evaluate_board(self, turn_to_move, current_hash=None):
        abs_score = self._evaluate_absolute(current_hash, turn_to_move)
        signed = abs_score if turn_to_move == 'white' else -abs_score
        return signed + TEMPO_BONUS

    def _evaluate_absolute(self, current_hash=None, turn=None):
        key = self._board_only_key(current_hash, turn)
        cached = self.eval_cache.get(key)
        if cached is not None:
            return cached

        board = self.board
        phase = self._game_phase(board)
        score = 0

        for p in board.white_pieces:
            r, c = p.pos
            score += PIECE_VALUES[p.z_idx]
            score += self._pst_value(p.z_idx, 'white', r, c, phase)

        for p in board.black_pieces:
            r, c = p.pos
            score -= PIECE_VALUES[p.z_idx]
            score -= self._pst_value(p.z_idx, 'black', r, c, phase)

        if board.piece_counts_z['white'][2] >= 2:
            score += 22
        if board.piece_counts_z['black'][2] >= 2:
            score -= 22

        score += self._side_dynamic_eval('white', phase)
        score -= self._side_dynamic_eval('black', phase)

        if len(self.eval_cache) > self.EVAL_CACHE_MAX:
            self.eval_cache.clear()
        self.eval_cache[key] = int(score)
        return int(score)

    def _board_only_key(self, current_hash, turn):
        if current_hash is None:
            if turn is None:
                turn = self.color
            current_hash = board_hash(self.board, turn)
        return current_hash if turn == 'white' else (current_hash ^ ZOBRIST_TURN)

    def _game_phase(self, board):
        phase = 0
        for z in range(6):
            phase += PHASE_WEIGHTS[z] * (board.piece_counts_z['white'][z] + board.piece_counts_z['black'][z])
        if phase > MAX_PHASE:
            phase = MAX_PHASE
        return phase

    def _pst_value(self, z, color, r, c, phase):
        rr = r if color == 'white' else 7 - r

        if z == 5:
            mid = KING_MID_PST[rr][c]
            end = KING_END_PST[rr][c]
            return (mid * phase + end * (MAX_PHASE - phase)) // MAX_PHASE

        table = PST_BY_Z.get(z)
        return table[rr][c] if table is not None else 0

    def _side_dynamic_eval(self, color, phase):
        board = self.board
        pieces = board.white_pieces if color == 'white' else board.black_pieces
        opp = 'black' if color == 'white' else 'white'
        enemy_king = board.black_king_pos if color == 'white' else board.white_king_pos
        own_king = board.white_king_pos if color == 'white' else board.black_king_pos
        grid = board.grid

        total = 0
        top1 = 0
        top2 = 0

        for piece in pieces:
            pos = piece.pos
            if pos is None:
                continue

            moves = piece.get_valid_moves(board, pos)
            z = piece.z_idx
            total += MOBILITY_WEIGHTS[z] * len(moves)

            for move in moves:
                end = move[1]
                target = grid[end[0]][end[1]]
                is_promo = (z == 0 and end[0] == piece.promo_rank)

                likely = (
                    target is not None or
                    z in (1, 3) or
                    is_promo or
                    (enemy_king is not None and self._approx_gives_check_after_move(piece, pos, end, target, enemy_king))
                )
                if not likely:
                    continue

                swing, is_tactic = self._approx_move_swing(move, piece, target, enemy_king)
                if not is_tactic and not is_promo:
                    continue

                press = swing
                if z == 4 and press < 0:
                    press //= 2
                if press > top1:
                    top2 = top1
                    top1 = press
                elif press > top2:
                    top2 = press

        if own_king is None:
            return -50000

        if is_square_attacked(board, own_king[0], own_king[1], opp):
            total -= 140

        king_piece = grid[own_king[0]][own_king[1]]
        if king_piece is not None:
            king_moves = king_piece.get_valid_moves(board, own_king)
            total += len(king_moves) * (3 if phase < 10 else 1)

        adj_friendly = 0
        adj_enemy = 0
        for r, c in ADJACENT_SQUARES_MAP[own_king]:
            p = grid[r][c]
            if p is None:
                continue
            if p.color == color:
                adj_friendly += 1
            else:
                adj_enemy += 1

        if board.piece_counts_z[opp][4] > 0:
            total -= adj_friendly * 6
            total -= adj_enemy * 4

        total -= adj_enemy * 8
        total -= self._king_rook_exposure(color)

        if top1 > 0:
            total += (top1 * 22) // 100
        if top2 > 0:
            total += (top2 * 8) // 100

        return int(total)

    def _king_rook_exposure(self, color):
        opp = 'black' if color == 'white' else 'white'
        if self.board.piece_counts_z[opp][3] == 0:
            return 0

        kpos = self.board.white_king_pos if color == 'white' else self.board.black_king_pos
        if kpos is None:
            return 30

        exposure = 0
        idx = kpos[0] * 8 + kpos[1]
        for ray in RAYS[idx][:4]:
            first_enemy = None
            for r, c in ray:
                p = self.board.grid[r][c]
                if p is None:
                    continue
                if p.color != color:
                    first_enemy = p
                    break
            if first_enemy is None:
                exposure += 5
            elif first_enemy.z_idx == 3:
                exposure += 18
        return exposure

    # ------------------------------------------------------------------
    # IDLE
    # ------------------------------------------------------------------

    def ponder_indefinitely(self):
        while not self.cancellation_event.is_set():
            time.sleep(0.1)