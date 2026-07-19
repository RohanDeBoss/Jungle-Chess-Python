# AI.py (Max Challenger v2.2)

import time
import random
import os
import json
import glob
from collections import defaultdict

from GameLogic import *
from TablebaseManager import TablebaseManager

# ==============================================================================
# MODULE-LEVEL SETUP (DO NOT MODIFY)
# ==============================================================================

ZOBRIST_ARRAY = None
ZOBRIST_TURN = None

def initialize_zobrist_table():
    global ZOBRIST_ARRAY, ZOBRIST_TURN
    if ZOBRIST_ARRAY is not None: return
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
    sr, sc = start; er, ec = end
    h ^= arr[c_idx][p_idx][sr][sc]
    mp_survived = True
    for piece, r, c in removed_pieces:
        if piece is mp: mp_survived = False
        else: h ^= arr[0 if piece.color == 'white' else 1][piece.z_idx][r][c]
    if mp_survived: h ^= arr[c_idx][p_idx][er][ec]
    for piece, r, c in added_pieces:
        h ^= arr[0 if piece.color == 'white' else 1][piece.z_idx][r][c]
    return h

_CLS_TO_CHAR = {Pawn: 'P', Knight: 'N', Bishop: 'B', Rook: 'R', Queen: 'Q', King: 'K'}

def board_to_fen(board, turn):
    fen = ''
    for r in range(ROWS):
        empty = 0
        for c in range(COLS):
            piece = board.grid[r][c]
            if piece is None: empty += 1
            else:
                if empty: fen += str(empty); empty = 0
                ch = _CLS_TO_CHAR[type(piece)]
                fen += ch if piece.color == 'white' else ch.lower()
        if empty: fen += str(empty)
        if r < ROWS - 1: fen += '/'
    return fen + (' w' if turn == 'white' else ' b')

OPENING_BOOK = {}
def _find_opening_book_files():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    patterns = (os.path.join(base_dir, "opening books", "opening_book*.json"),
                os.path.join(base_dir, "opening_book*.json"))
    seen = set(); matches = []
    for pattern in patterns:
        for path in glob.glob(pattern):
            norm = os.path.normcase(os.path.abspath(path))
            if norm not in seen:
                seen.add(norm); matches.append(path)
    return sorted(matches, key=lambda p: (os.path.getmtime(p), os.path.basename(p)), reverse=True)

for _book_filename in _find_opening_book_files():
    try:
        with open(_book_filename, "r", encoding="utf-8") as f: OPENING_BOOK = json.load(f)
        break
    except Exception: pass

def run_ai_process(board, color, position_counts, comm_queue, cancellation_event,
                   bot_class, bot_name, search_depth, ply_count, game_mode,
                   time_left=None, increment=None, use_opening_book=True, use_tablebase=True):
    import inspect
    accepted_params = set(inspect.signature(bot_class.__init__).parameters)
    kwargs = {'time_left': time_left, 'increment': increment, 'use_opening_book': use_opening_book, 'use_tablebase': use_tablebase}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted_params}
    bot = bot_class(board, color, position_counts, comm_queue, cancellation_event, bot_name, ply_count, game_mode, **filtered_kwargs)
    bot.search_depth = search_depth
    if search_depth == 99: bot.ponder_indefinitely()
    else: bot.make_move()

class SearchCancelledException(Exception): pass

# ==============================================================================
# CONSTANTS
# ==============================================================================

_INF = 9999999
_TT_EXACT = 0
_TT_LOWER = 1
_TT_UPPER = 2

# Piece values tuned for Jungle Chess:
# Queen is kamikaze (dies on capture) so worth less than standard.
# Rook pierces everything so worth more. Knight evaporates so volatile.
# Jungle-tuned material (per user tuning: knights are devastating, queens die on capture)
_PV = [100, 950, 600, 600, 1300, 20000]  # P N B R Q K
_PV_LIST = _PV

# Phase weights for tapered eval
_PHASE_W = [0, 1, 1, 2, 3, 0]
_MAX_PHASE = 20

# PSTs from white's perspective (row 0 = rank 8 = black back rank)
_PAWN_MG = (
    ( 0,  0,  0,  0,  0,  0,  0,  0),
    (55, 60, 65, 70, 70, 65, 60, 55),
    (35, 40, 48, 55, 55, 48, 40, 35),
    (20, 26, 34, 42, 42, 34, 26, 20),
    (10, 16, 24, 32, 32, 24, 16, 10),
    ( 5,  8, 14, 20, 20, 14,  8,  5),
    ( 0,  2,  4,  8,  8,  4,  2,  0),
    ( 0,  0,  0,  0,  0,  0,  0,  0),
)
_PAWN_EG = (
    ( 0,  0,  0,  0,  0,  0,  0,  0),
    (75, 78, 80, 82, 82, 80, 78, 75),
    (55, 58, 62, 66, 66, 62, 58, 55),
    (35, 38, 42, 48, 48, 42, 38, 35),
    (20, 24, 28, 34, 34, 28, 24, 20),
    (10, 12, 16, 20, 20, 16, 12, 10),
    ( 2,  4,  6, 10, 10,  6,  4,  2),
    ( 0,  0,  0,  0,  0,  0,  0,  0),
)

_KNIGHT_MG = (
    (-40,-20,-12, -8, -8,-12,-20,-40),
    (-18, -4,  6, 10, 10,  6, -4,-18),
    (-10,  6, 18, 22, 22, 18,  6,-10),
    ( -8, 10, 22, 28, 28, 22, 10, -8),
    ( -8, 10, 22, 28, 28, 22, 10, -8),
    (-10,  6, 18, 22, 22, 18,  6,-10),
    (-18, -4,  6, 10, 10,  6, -4,-18),
    (-40,-20,-12, -8, -8,-12,-20,-40),
)

_BISHOP_MG = (
    (-16, -8, -6, -4, -4, -6, -8,-16),
    ( -8,  2,  6,  8,  8,  6,  2, -8),
    ( -6,  6, 12, 14, 14, 12,  6, -6),
    ( -4,  8, 14, 18, 18, 14,  8, -4),
    ( -4,  8, 14, 18, 18, 14,  8, -4),
    ( -6,  6, 12, 14, 14, 12,  6, -6),
    ( -8,  2,  6,  8,  8,  6,  2, -8),
    (-16, -8, -6, -4, -4, -6, -8,-16),
)

_ROOK_MG = (
    ( 6,  8, 10, 14, 14, 10,  8,  6),
    (12, 14, 16, 20, 20, 16, 14, 12),
    ( 4,  6, 10, 14, 14, 10,  6,  4),
    ( 0,  4,  8, 12, 12,  8,  4,  0),
    ( 0,  4,  8, 12, 12,  8,  4,  0),
    ( 4,  6, 10, 14, 14, 10,  6,  4),
    (12, 14, 16, 20, 20, 16, 14, 12),
    ( 6,  8, 10, 14, 14, 10,  8,  6),
)

_QUEEN_MG = (
    ( -8, -4, -2,  0,  0, -2, -4, -8),
    ( -4,  2,  4,  6,  6,  4,  2, -4),
    ( -2,  4,  8, 10, 10,  8,  4, -2),
    (  0,  6, 10, 12, 12, 10,  6,  0),
    (  0,  6, 10, 12, 12, 10,  6,  0),
    ( -2,  4,  8, 10, 10,  8,  4, -2),
    ( -4,  2,  4,  6,  6,  4,  2, -4),
    ( -8, -4, -2,  0,  0, -2, -4, -8),
)

_KING_MG = (
    ( 24, 28, 14,  2,  2, 14, 28, 24),
    ( 18, 16,  4, -8, -8,  4, 16, 18),
    (  6, -2,-12,-20,-20,-12, -2,  6),
    ( -4,-14,-22,-30,-30,-22,-14, -4),
    ( -6,-16,-24,-32,-32,-24,-16, -6),
    ( -4,-12,-20,-26,-26,-20,-12, -4),
    (  6,  2, -6,-14,-14, -6,  2,  6),
    ( 16, 14,  6, -2, -2,  6, 14, 16),
)

_KING_EG = (
    (-18,-10,  0,  8,  8,  0,-10,-18),
    (-10,  0,  8, 16, 16,  8,  0,-10),
    (  0,  8, 16, 24, 24, 16,  8,  0),
    (  8, 16, 24, 32, 32, 24, 16,  8),
    (  8, 16, 24, 32, 32, 24, 16,  8),
    (  0,  8, 16, 24, 24, 16,  8,  0),
    (-10,  0,  8, 16, 16,  8,  0,-10),
    (-18,-10,  0,  8,  8,  0,-10,-18),
)

# Precomputed PST lookups: _PST_MG[color_idx][z_idx][r][c]
# color_idx: 0=white, 1=black
def _build_pst_tables():
    mg = [[None]*6 for _ in range(2)]
    eg = [[None]*6 for _ in range(2)]
    
    tables_mg = [_PAWN_MG, _KNIGHT_MG, _BISHOP_MG, _ROOK_MG, _QUEEN_MG, _KING_MG]
    tables_eg = [_PAWN_EG, _KNIGHT_MG, _BISHOP_MG, _ROOK_MG, _QUEEN_MG, _KING_EG]
    
    for z in range(6):
        # White: row r maps to PST row r (row 0 = rank 8)
        w_mg = tuple(tuple(tables_mg[z][r][c] for c in range(8)) for r in range(8))
        w_eg = tuple(tuple(tables_eg[z][r][c] for c in range(8)) for r in range(8))
        # Black: mirror vertically
        b_mg = tuple(tuple(tables_mg[z][7-r][c] for c in range(8)) for r in range(8))
        b_eg = tuple(tuple(tables_eg[z][7-r][c] for c in range(8)) for r in range(8))
        mg[0][z] = w_mg
        mg[1][z] = b_mg
        eg[0][z] = w_eg
        eg[1][z] = b_eg
    return mg, eg

_PST_MG, _PST_EG = _build_pst_tables()

# Precomputed combined material+PST for incremental use
# _PIECE_PST_MG[color_idx][z_idx][r][c] = material + pst_mg
# _PIECE_PST_EG[color_idx][z_idx][r][c] = material + pst_eg
def _build_piece_pst():
    mg = [[None]*6 for _ in range(2)]
    eg = [[None]*6 for _ in range(2)]
    for ci in range(2):
        for z in range(6):
            mg[ci][z] = tuple(
                tuple(_PV[z] + _PST_MG[ci][z][r][c] for c in range(8))
                for r in range(8)
            )
            eg[ci][z] = tuple(
                tuple(_PV[z] + _PST_EG[ci][z][r][c] for c in range(8))
                for r in range(8)
            )
    return mg, eg

_PIECE_PST_MG, _PIECE_PST_EG = _build_piece_pst()


# ==============================================================================
# CHESS BOT CLASS
# ==============================================================================

class ChessBot:
    search_depth = 4  # class-level default for UI
    MATE_SCORE = 1000000
    MATE_BOUND = 990000
    TT_SIZE_LIMIT = 800000
    MAX_PLY = 100

    def __init__(self, board, color, position_counts, comm_queue, cancellation_event,
                 bot_name="Challenger AI", ply_count=0, game_mode="bot", max_moves=200,
                 time_left=None, increment=None, use_opening_book=True, use_tablebase=True):
        self.board = board
        self.color = color
        self.opponent_color = 'black' if color == 'white' else 'white'
        self.position_counts = position_counts if position_counts else {}
        self.comm_queue = comm_queue
        self.cancellation_event = cancellation_event
        self.ply_count = ply_count
        self.bot_name = bot_name
        self.max_moves = max_moves

        self.time_left = time_left
        self.increment = increment
        self.stop_time = None
        self.soft_stop = None
        self.nodes_searched = 0

        # TT: hash -> (depth, flag, score, best_move)
        self.tt = {}
        # Killer moves per ply
        self.killers = [[None, None] for _ in range(self.MAX_PLY + 8)]
        # History heuristic: [color_idx][z_idx][r][c]
        self.history = [[[[0]*8 for _ in range(8)] for _ in range(6)] for _ in range(2)]
        # Counter-move table: [color_idx][z_idx][r][c] -> move
        self.counter_move = [[[[None]*8 for _ in range(8)] for _ in range(6)] for _ in range(2)]
        self.prev_move = None  # last move made (for counter-move)

        self.use_opening_book = use_opening_book
        self.tb_manager = TablebaseManager()
        if not use_tablebase:
            self.tb_manager.probe = lambda b, t: None

    # --- UI COMMUNICATION (Do not modify) ---
    def _report_log(self, msg): self.comm_queue.put(('log', msg))
    def _report_eval(self, score, depth): self.comm_queue.put(('eval', score if self.color == 'white' else -score, depth))
    def _report_move(self, move): self.comm_queue.put(('move', move))

    def _format_move(self, board_before, move):
        if not move: return "None"
        child = board_before.clone()
        child.make_move(move[0], move[1])
        return format_move_san(board_before, child, move)

    # ------------------------------------------------------------------
    # MAIN ENTRY
    # ------------------------------------------------------------------
    def make_move(self):
        root_moves = None
        try:
            # Opening book
            if self.use_opening_book and self.ply_count <= 12:
                fen = board_to_fen(self.board, self.color)
                if fen in OPENING_BOOK:
                    chosen = random.choices(OPENING_BOOK[fen],
                                            weights=[o["weight"] for o in OPENING_BOOK[fen]], k=1)[0]
                    mt = (tuple(chosen["move"][0]), tuple(chosen["move"][1]))
                    self._report_log(f"  > {self.bot_name} (Book): {chosen['san']}")
                    self._report_eval(chosen['score'], "Book")
                    self.comm_queue.put(('pv', chosen['score'], "Book", [chosen['san']], [mt]))
                    self._report_move(mt)
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
                self.soft_stop = None
                self.stop_time = None

            best_move_overall = root_moves[0]
            root_hash = board_hash(self.board, self.color)
            prev_score = 0

            for depth in range(1, target_depth + 1):
                if self.cancellation_event.is_set():
                    break
                if depth > 1 and self.soft_stop and time.time() >= self.soft_stop:
                    break

                iter_start = time.time()
                self.nodes_searched = 0

                try:
                    if depth >= 4:
                        window = 200  # Jungle swings can be 1000+, tight windows waste re-searches
                        a, b = prev_score - window, prev_score + window
                        while True:
                            sc, bm, root_moves = self._search_root(depth, root_moves, root_hash, a, b)
                            if self.cancellation_event.is_set() or self._hard_stop():
                                raise SearchCancelledException()
                            if sc <= a:
                                a = max(sc - window * 2, -_INF)
                                window *= 2
                            elif sc >= b:
                                b = min(sc + window * 2, _INF)
                                window *= 2
                            else:
                                break
                    else:
                        sc, bm, root_moves = self._search_root(depth, root_moves, root_hash, -_INF, _INF)
                except SearchCancelledException:
                    break

                best_move_overall = bm
                prev_score = sc

                dur = time.time() - iter_start
                knps = (self.nodes_searched / max(dur, 1e-9)) / 1000
                eval_ui = sc if self.color == 'white' else -sc
                pv_san = self._format_move(self.board, best_move_overall)

                self._report_log(
                    f"  > {self.bot_name} (D{depth}): {pv_san}, Eval={eval_ui/100:+.2f}, "
                    f"Nodes={self.nodes_searched}, KNPS={knps:.1f}, Time={dur:.2f}s"
                )
                self._report_eval(sc, depth)
                self.comm_queue.put(('pv', eval_ui, depth, [pv_san], [best_move_overall]))

                if abs(sc) > self.MATE_SCORE - 2000:
                    break

                if self.soft_stop and depth >= 3:
                    remaining = self.soft_stop - time.time()
                    if remaining < dur * 1.6:
                        break

            self._report_move(best_move_overall)

        except Exception as e:
            self._report_log(f"CRASH: {str(e)}")
            self._report_move(root_moves[0] if root_moves else None)

    # ------------------------------------------------------------------
    # TIME MANAGEMENT
    # ------------------------------------------------------------------
    def _allocate_time(self, n_moves):
        now = time.time()
        tl = max(0.01, float(self.time_left))
        inc = max(0.0, float(self.increment))

        # Estimate moves remaining in the game (more generous than v302/v303)
        moves_left = max(12, 40 - self.ply_count // 3)
        
        # Base allocation: divide remaining time + banking on increment
        alloc = (tl / moves_left) + inc * 0.85

        # Complexity scaling
        if n_moves <= 4:
            alloc *= 0.75
        elif n_moves >= 25:
            alloc *= 1.20

        # Emergency time controls — but MUCH less aggressive than v303
        if tl < 3.0:
            alloc = min(alloc, tl * 0.15 + inc * 0.8)
        if tl < 1.0:
            alloc = min(alloc, tl * 0.10 + inc * 0.6)
        if tl < 0.3:
            alloc = min(alloc, tl * 0.08 + inc * 0.4)

        # Hard bounds — allow at least a real search
        alloc = max(0.05, min(alloc, tl * 0.35))
        hard = min(max(alloc * 3.0, alloc + 0.10), tl * 0.80)

        self.soft_stop = now + alloc
        self.stop_time = now + hard

    def _hard_stop(self):
        return self.stop_time is not None and time.time() >= self.stop_time

    def _check_abort(self):
        if self.cancellation_event.is_set() or self._hard_stop():
            raise SearchCancelledException()

    # ------------------------------------------------------------------
    # ROOT SEARCH
    # ------------------------------------------------------------------
    def _search_root(self, depth, root_moves, root_hash, alpha, beta):
        best_score = -_INF
        best_move = root_moves[0]
        alpha_orig = alpha
        move_scores = []

        tt_move = None
        tt_e = self.tt.get(root_hash)
        if tt_e: tt_move = tt_e[3]

        ordered = self._order_root(root_moves, tt_move)

        for idx, move in enumerate(ordered):
            if (self.nodes_searched & 2047) == 0:
                self._check_abort()

            record = self.board.make_move_track(move[0], move[1])
            child_hash = incremental_hash(root_hash, record)

            try:
                if idx == 0:
                    sc = -self._negamax(depth - 1, -beta, -alpha, self.opponent_color, 1, child_hash, move)
                else:
                    sc = -self._negamax(depth - 1, -alpha - 1, -alpha, self.opponent_color, 1, child_hash, move)
                    if sc > alpha and sc < beta:
                        sc = -self._negamax(depth - 1, -beta, -alpha, self.opponent_color, 1, child_hash, move)
            finally:
                self.board.unmake_move(record)

            move_scores.append((sc, move))

            if sc > best_score:
                best_score = sc
                best_move = move
            if sc > alpha:
                alpha = sc
            if alpha >= beta:
                break

        # Re-order root moves for next iteration
        scored_set = {id(m): s for s, m in move_scores}
        new_order = sorted(move_scores, key=lambda x: x[0], reverse=True)
        result = [m for _, m in new_order]
        for m in ordered:
            if id(m) not in scored_set:
                result.append(m)

        # Store TT
        flag = _TT_EXACT
        if best_score <= alpha_orig: flag = _TT_UPPER
        elif best_score >= beta: flag = _TT_LOWER
        self._tt_store(root_hash, depth, flag, best_score, best_move, 0)

        return best_score, best_move, result

    def _order_root(self, moves, tt_move):
        scored = []
        board = self.board
        grid = board.grid
        for move in moves:
            s, e = move
            piece = grid[s[0]][s[1]]
            target = grid[e[0]][e[1]]
            sc = 0
            if tt_move and move == tt_move:
                sc += 100000000
            if target is not None:
                sc += 10000000 + _PV[target.z_idx] * 16 - _PV[piece.z_idx]
            swing, tactic = fast_approximate_material_swing(board, move, piece, target, _PV_LIST)
            if tactic:
                sc += 5000000 + swing * 8
            if piece.z_idx == 0 and e[0] == piece.promo_rank:
                sc += 8000000
            ci = 0 if piece.color == 'white' else 1
            sc += self.history[ci][piece.z_idx][e[0]][e[1]]
            scored.append((sc, move))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored]

    # ------------------------------------------------------------------
    # NEGAMAX
    # ------------------------------------------------------------------
    def _negamax(self, depth, alpha, beta, turn, ply, h, last_move):
        self.nodes_searched += 1
        if (self.nodes_searched & 2047) == 0:
            self._check_abort()

        if ply >= self.MAX_PLY:
            return self._eval(turn)

        # Repetition
        total_rep = self.position_counts.get(h, 0)
        if total_rep >= 3:
            return 0
            
        # 2-fold repetition penalty when we're winning — avoid draws
        if total_rep >= 2 and ply > 0:
            static = self._eval(turn)
            if static > 50:  # We're winning, don't repeat
                return -20

        # Move limit
        if self.ply_count + ply >= self.max_moves:
            return 0

        # Insufficient material
        board = self.board
        total_pieces = len(board.white_pieces) + len(board.black_pieces)
        if total_pieces <= 2:
            return 0

        # Tablebase
        if total_pieces <= 5:
            tb = self.tb_manager.probe(board, turn)
            if tb is not None:
                tb_sc = tb if turn == 'white' else -tb
                if tb_sc > self.MATE_SCORE - 1000: return tb_sc - ply
                if tb_sc < -self.MATE_SCORE + 1000: return tb_sc + ply
                return tb_sc

        # Leaf
        if depth <= 0:
            return self._qsearch(alpha, beta, turn, ply, h)

        alpha_orig = alpha
        opp = 'black' if turn == 'white' else 'white'

        # TT probe
        tt_move = None
        tt_e = self.tt.get(h)
        if tt_e is not None:
            td, tf, ts_raw, tm = tt_e
            if td >= depth:
                ts = self._tt_score_out(ts_raw, ply)
                if tf == _TT_EXACT: return ts
                if tf == _TT_LOWER: alpha = max(alpha, ts)
                else: beta = min(beta, ts)
                if alpha >= beta: return ts
            tt_move = tm

        in_check = is_in_check(board, turn)

        # Check extension
        if in_check:
            depth += 1

        # Null move pruning - only if not in check, have enough material
        if (not in_check and depth >= 3 and ply > 0 and
                total_pieces > 5):
            # Verify we have non-pawn/king material
            ci = 0 if turn == 'white' else 1
            pcz = board.piece_counts_z[turn]
            non_pawn_king = pcz[1] + pcz[2] + pcz[3] + pcz[4]
            if non_pawn_king > 0:
                # Make null move (just flip turn)
                r_val = 3 if depth >= 6 else 2
                null_hash = h ^ ZOBRIST_TURN
                null_sc = -self._negamax(depth - 1 - r_val, -beta, -beta + 1, opp, ply + 1, null_hash, None)
                if null_sc >= beta:
                    if abs(null_sc) < self.MATE_SCORE - 1000:
                        return beta

        # Generate and search moves
        pseudo = get_all_pseudo_legal_moves(board, turn)
        ordered = self._order_moves(pseudo, turn, ply, tt_move, last_move)

        best_score = -_INF
        best_move = None
        legal_count = 0

        for idx, (mv_score, move, is_tactic, is_quiet, piece_z) in enumerate(ordered):
            record = board.make_move_track(move[0], move[1])
            my_kp = board.white_king_pos if turn == 'white' else board.black_king_pos
            legal = (my_kp is not None and not is_square_attacked(board, my_kp[0], my_kp[1], opp))
            if not legal:
                board.unmake_move(record)
                continue

            legal_count += 1
            child_hash = incremental_hash(h, record)

            # Late move reductions
            reduction = 0
            if (depth >= 3 and legal_count > 3 and not in_check and is_quiet
                    and piece_z != 5):
                if legal_count > 8:
                    reduction = 2 if depth >= 5 else 1
                else:
                    reduction = 1

            new_depth = depth - 1

            try:
                if idx == 0:
                    sc = -self._negamax(new_depth, -beta, -alpha, opp, ply + 1, child_hash, move)
                else:
                    if reduction > 0:
                        sc = -self._negamax(new_depth - reduction, -alpha - 1, -alpha, opp, ply + 1, child_hash, move)
                        if sc > alpha:
                            sc = -self._negamax(new_depth, -alpha - 1, -alpha, opp, ply + 1, child_hash, move)
                            if sc > alpha and sc < beta:
                                sc = -self._negamax(new_depth, -beta, -alpha, opp, ply + 1, child_hash, move)
                    else:
                        sc = -self._negamax(new_depth, -alpha - 1, -alpha, opp, ply + 1, child_hash, move)
                        if sc > alpha and sc < beta:
                            sc = -self._negamax(new_depth, -beta, -alpha, opp, ply + 1, child_hash, move)
            finally:
                board.unmake_move(record)

            if sc > best_score:
                best_score = sc
                best_move = move
            if sc > alpha:
                alpha = sc
            if alpha >= beta:
                if is_quiet:
                    self._update_killers(ply, move)
                    ci = 0 if turn == 'white' else 1
                    bonus = depth * depth
                    self.history[ci][piece_z][move[1][0]][move[1][1]] += bonus
                    if self.history[ci][piece_z][move[1][0]][move[1][1]] > 400000:
                        self._age_history()
                    if last_move is not None:
                        lp = self.board.grid[last_move[1][0]][last_move[1][1]]
                        if lp is not None:
                            oci = 1 - ci
                            self.counter_move[oci][lp.z_idx][last_move[1][0]][last_move[1][1]] = move
                break

        if legal_count == 0:
            return -self.MATE_SCORE + ply

        flag = _TT_EXACT
        if best_score <= alpha_orig: flag = _TT_UPPER
        elif best_score >= beta: flag = _TT_LOWER
        self._tt_store(h, depth, flag, best_score, best_move, ply)

        return best_score

    # ------------------------------------------------------------------
    # QUIESCENCE
    # ------------------------------------------------------------------
    def _qsearch(self, alpha, beta, turn, ply, h):
        # Qsearch depth limit — prevent explosion
        if ply > 20:
            return self._eval(turn)
        self.nodes_searched += 1
        if (self.nodes_searched & 4095) == 0:
            self._check_abort()

        if ply >= self.MAX_PLY:
            return self._eval(turn)

        board = self.board
        total_pieces = len(board.white_pieces) + len(board.black_pieces)
        if total_pieces <= 2:
            return 0

        if total_pieces <= 5:
            tb = self.tb_manager.probe(board, turn)
            if tb is not None:
                tb_sc = tb if turn == 'white' else -tb
                if tb_sc > self.MATE_SCORE - 1000: return tb_sc - ply
                if tb_sc < -self.MATE_SCORE + 1000: return tb_sc + ply
                return tb_sc

        in_check = is_in_check(board, turn)
        opp = 'black' if turn == 'white' else 'white'

        if not in_check:
            stand_pat = self._eval(turn)
            if stand_pat >= beta:
                return stand_pat
            if stand_pat > alpha:
                alpha = stand_pat
            best_score = stand_pat
        else:
            if not has_legal_moves(board, turn):
                return -self.MATE_SCORE + ply
            best_score = -_INF

        pseudo = get_all_pseudo_legal_moves(board, turn)
        grid = board.grid

        # Build and sort tactical moves
        tacticals = []
        for move in pseudo:
            s, e = move
            piece = grid[s[0]][s[1]]
            if piece is None or piece.color != turn:
                continue
            target = grid[e[0]][e[1]]
            is_promo = (piece.z_idx == 0 and e[0] == piece.promo_rank)

            if in_check:
                # Search all moves when in check
                swing, _ = fast_approximate_material_swing(board, move, piece, target, _PV_LIST)
                tacticals.append((swing + (10000 if is_promo else 0), move))
                continue

            # Only tactical moves in qsearch
            if target is None and not is_promo:
                swing, is_tactic = fast_approximate_material_swing(board, move, piece, target, _PV_LIST)
                if not is_tactic:
                    continue
                if swing < -50:
                    continue
                tacticals.append((swing, move))
            else:
                swing, _ = fast_approximate_material_swing(board, move, piece, target, _PV_LIST)
                        
                sc = swing
                if target is not None:
                    sc += _PV[target.z_idx] * 4
                if is_promo:
                    sc += 6000
                tacticals.append((sc, move))

        tacticals.sort(key=lambda x: x[0], reverse=True)

        for _, move in tacticals:
            record = board.make_move_track(move[0], move[1])
            my_kp = board.white_king_pos if turn == 'white' else board.black_king_pos
            legal = (my_kp is not None and not is_square_attacked(board, my_kp[0], my_kp[1], opp))
            if not legal:
                board.unmake_move(record)
                continue

            child_hash = incremental_hash(h, record)
            try:
                sc = -self._qsearch(-beta, -alpha, opp, ply + 1, child_hash)
            finally:
                board.unmake_move(record)

            if sc > best_score:
                best_score = sc
            if sc > alpha:
                alpha = sc
            if alpha >= beta:
                break

        if in_check and best_score == -_INF:
            if not has_legal_moves(board, turn):
                return -self.MATE_SCORE + ply

        return best_score

    # ------------------------------------------------------------------
    # MOVE ORDERING (non-root)
    # ------------------------------------------------------------------
    def _order_moves(self, pseudo, turn, ply, tt_move, last_move):
        board = self.board
        grid = board.grid
        ci = 0 if turn == 'white' else 1
        enemy_king = board.black_king_pos if turn == 'white' else board.white_king_pos
        k0, k1 = self.killers[ply] if ply < len(self.killers) else (None, None)

        cm = None
        if last_move is not None:
            lp = grid[last_move[1][0]][last_move[1][1]]
            if lp is not None:
                oci = 1 - ci
                cm = self.counter_move[oci][lp.z_idx][last_move[1][0]][last_move[1][1]]

        result = []
        for move in pseudo:
            s, e = move
            piece = grid[s[0]][s[1]]
            if piece is None or piece.color != turn:
                continue

            z = piece.z_idx
            target = grid[e[0]][e[1]]
            is_promo = (z == 0 and e[0] == piece.promo_rank)

            sc = 0
            is_tactic = False
            is_quiet = True

            if tt_move is not None and move == tt_move:
                sc += 200_000_000
                is_quiet = False

            if target is not None:
                is_quiet = False
                is_tactic = True
                sc += 30_000_000 + _PV[target.z_idx] * 16 - _PV[z]

            if is_promo:
                is_quiet = False
                is_tactic = True
                sc += 28_000_000

            # Detect AoE-based tactics: rook piercing, knight evap, queen blast
            swing = 0
            if target is not None or is_promo or z in (1, 3, 4):
                swing, tact = fast_approximate_material_swing(board, move, piece, target, _PV_LIST)
                if tact:
                    is_tactic = True
                    is_quiet = False
                    sc += 25_000_000 + swing * 8

            # KING THREAT DETECTION — moves that check enemy king (including AoE checks!)
            check_bonus = self._approx_check_bonus(move, piece, target, enemy_king)
            if check_bonus:
                sc += check_bonus
                is_quiet = False

            # KNIGHT LANDING NEAR ENEMY KING — evaporation check
            if z == 1 and enemy_king is not None:
                if enemy_king in KNIGHT_ATTACKS_FROM[e]:
                    sc += 15_000_000  # Direct evap check — often mate!
                    is_quiet = False

            # QUEEN EXPLOSION ADJACENT TO KING
            if z == 4 and target is not None and enemy_king is not None:
                if abs(e[0] - enemy_king[0]) <= 1 and abs(e[1] - enemy_king[1]) <= 1:
                    sc += 18_000_000  # Queen explodes next to king
                    is_quiet = False

            if is_quiet:
                # Tactical piece priority — knights are search-critical in Jungle
                if z == 1:
                    sc += 1000
                elif z == 4:
                    sc += 500

                if move == k0: sc += 9000000
                elif move == k1: sc += 8000000
                elif cm is not None and move == cm: sc += 7500000
                sc += self.history[ci][z][e[0]][e[1]]

            result.append((sc, move, is_tactic, is_quiet, z))

        result.sort(key=lambda x: x[0], reverse=True)
        return result
    
    def _approx_check_bonus(self, move, piece, target, enemy_king):
        """Return ordering bonus if this move puts enemy king in check."""
        if enemy_king is None:
            return 0

        start, end = move
        z = piece.z_idx

        # Knight evaporation check
        if z == 1:
            if enemy_king in KNIGHT_ATTACKS_FROM[end]:
                return 12_000_000
            return 0

        # Rook piercing check (shares rank/file with king)
        if z == 3:
            if end[0] == enemy_king[0] or end[1] == enemy_king[1]:
                return 10_000_000
            return 0

        # Queen: explosion check or line check
        if z == 4:
            if target is not None and abs(end[0] - enemy_king[0]) <= 1 and abs(end[1] - enemy_king[1]) <= 1:
                return 15_000_000
            # Line check
            if end[0] == enemy_king[0] or end[1] == enemy_king[1]:
                return 5_000_000
            if abs(end[0] - enemy_king[0]) == abs(end[1] - enemy_king[1]):
                return 4_500_000
            return 0

        # Bishop diagonal check
        if z == 2:
            if abs(end[0] - enemy_king[0]) == abs(end[1] - enemy_king[1]):
                return 3_000_000
            return 0

        # Pawn check
        if z == 0:
            direction = -1 if piece.color == 'white' else 1
            if enemy_king[0] == end[0] + direction and abs(enemy_king[1] - end[1]) == 1:
                return 2_000_000
            return 0

        return 0
    
    # ------------------------------------------------------------------
    # EVALUATION
    # ------------------------------------------------------------------
    def _eval(self, turn):
        board = self.board
        # Compute game phase
        phase = 0
        pcz_w = board.piece_counts_z['white']
        pcz_b = board.piece_counts_z['black']
        for z in range(6):
            phase += _PHASE_W[z] * (pcz_w[z] + pcz_b[z])
        if phase > _MAX_PHASE:
            phase = _MAX_PHASE

        score_mg = 0
        score_eg = 0

        pst_mg_w = _PIECE_PST_MG[0]
        pst_eg_w = _PIECE_PST_EG[0]
        pst_mg_b = _PIECE_PST_MG[1]
        pst_eg_b = _PIECE_PST_EG[1]

        for p in board.white_pieces:
            r, c = p.pos
            z = p.z_idx
            score_mg += pst_mg_w[z][r][c]
            score_eg += pst_eg_w[z][r][c]

        for p in board.black_pieces:
            r, c = p.pos
            z = p.z_idx
            score_mg -= pst_mg_b[z][r][c]
            score_eg -= pst_eg_b[z][r][c]

        # Tapered eval
        score = (score_mg * phase + score_eg * (_MAX_PHASE - phase)) // _MAX_PHASE

        # Bishop pair bonus
        if pcz_w[2] >= 2: score += 25
        if pcz_b[2] >= 2: score -= 25

        # Knight danger bonus: knights near enemy king are extra dangerous
        w_king = board.white_king_pos
        b_king = board.black_king_pos
        grid = board.grid

        if b_king is not None:
            # White knight pressure on black king
            bkr, bkc = b_king
            for r, c in KNIGHT_ATTACKS_FROM[b_king]:
                p = grid[r][c]
                if p is not None and p.z_idx == 1 and p.color == 'white':
                    score += 35  # Knight threatening king area

        if w_king is not None:
            wkr, wkc = w_king
            for r, c in KNIGHT_ATTACKS_FROM[w_king]:
                p = grid[r][c]
                if p is not None and p.z_idx == 1 and p.color == 'black':
                    score -= 35

        # Rook on open/semi-open file bonus
        for p in board.white_pieces:
            if p.z_idx == 3:
                r, c = p.pos
                blocked = False
                for rr in range(r - 1, -1, -1):
                    pp = grid[rr][c]
                    if pp is not None and pp.z_idx == 0 and pp.color == 'white':
                        blocked = True
                        break
                if not blocked:
                    score += 18

        for p in board.black_pieces:
            if p.z_idx == 3:
                r, c = p.pos
                blocked = False
                for rr in range(r + 1, 8):
                    pp = grid[rr][c]
                    if pp is not None and pp.z_idx == 0 and pp.color == 'black':
                        blocked = True
                        break
                if not blocked:
                    score -= 18

        # Queen explosion threat proximity to enemy king
        if b_king is not None:
            bkr, bkc = b_king
            for p in board.white_pieces:
                if p.z_idx == 4:
                    qr, qc = p.pos
                    dist = max(abs(qr - bkr), abs(qc - bkc))
                    if dist <= 3:
                        score += (4 - dist) * 12

        if w_king is not None:
            wkr, wkc = w_king
            for p in board.black_pieces:
                if p.z_idx == 4:
                    qr, qc = p.pos
                    dist = max(abs(qr - wkr), abs(qc - wkc))
                    if dist <= 3:
                        score -= (4 - dist) * 12

        # King safety: penalize open lines and count attackers around king in middlegame
        if phase > 10:
            if w_king is not None:
                score -= self._count_king_attackers(w_king, 'black') * phase // _MAX_PHASE
                score -= self._king_exposure(w_king, 'white', grid) * phase // _MAX_PHASE
            if b_king is not None:
                score += self._count_king_attackers(b_king, 'white') * phase // _MAX_PHASE
                score += self._king_exposure(b_king, 'black', grid) * phase // _MAX_PHASE

        # Pawn advancement bonus (passed pawn approximation)
        for p in board.white_pieces:
            if p.z_idx == 0:
                r, c = p.pos
                dist = r  # distance to rank 0 (promo)
                if dist <= 2:
                    score += (3 - dist) * 20

        for p in board.black_pieces:
            if p.z_idx == 0:
                r, c = p.pos
                dist = 7 - r  # distance to rank 7 (promo)
                if dist <= 2:
                    score -= (3 - dist) * 20

        # Return from perspective of turn
        signed = score if turn == 'white' else -score
        return signed + 10  # tempo

    def _count_king_attackers(self, king_pos, attacker_color):
        """Cheap count of enemy pieces attacking squares around the king."""
        if king_pos is None:
            return 0
        board = self.board
        grid = board.grid
        weight = 0
        
        # Check knights attacking the 9-square king zone
        for kr, kc in [king_pos] + list(ADJACENT_SQUARES_MAP[king_pos]):
            for r, c in KNIGHT_ATTACKS_FROM[(kr, kc)]:
                p = grid[r][c]
                if p is not None and p.color == attacker_color and p.z_idx == 1:
                    weight += 30
                    break  # count each square once
        
        # Check queens and rooks near the king zone
        kr, kc = king_pos
        pieces = board.white_pieces if attacker_color == 'white' else board.black_pieces
        for p in pieces:
            if p.z_idx == 4:
                qr, qc = p.pos
                dist = max(abs(qr - kr), abs(qc - kc))
                if dist <= 2:
                    weight += (3 - dist) * 20
            elif p.z_idx == 3:
                rr, rc = p.pos
                if rr == kr or rc == kc:
                    weight += 25
        
        return weight

    def _king_exposure(self, kpos, color, grid):
        """Rough measure of how exposed the king is."""
        kr, kc = kpos
        exposure = 0
        opp = 'black' if color == 'white' else 'white'
        idx = kr * 8 + kc
        # Check orthogonal rays for rook danger
        for i in range(4):
            for r, c in RAYS[idx][i]:
                p = grid[r][c]
                if p is None:
                    exposure += 1
                    continue
                if p.color != color and p.z_idx == 3:
                    exposure += 8
                break
        return exposure

    # ------------------------------------------------------------------
    # TT HELPERS
    # ------------------------------------------------------------------
    def _tt_score_in(self, score, ply):
        if score > self.MATE_BOUND: return score + ply
        if score < -self.MATE_BOUND: return score - ply
        return score

    def _tt_score_out(self, score, ply):
        if score > self.MATE_BOUND: return score - ply
        if score < -self.MATE_BOUND: return score + ply
        return score

    def _tt_store(self, key, depth, flag, score, move, ply):
        if move is None:
            return
        if len(self.tt) > self.TT_SIZE_LIMIT:
            self.tt.clear()
        old = self.tt.get(key)
        if old is None or depth >= old[0]:
            self.tt[key] = (depth, flag, self._tt_score_in(score, ply), move)

    # ------------------------------------------------------------------
    # KILLER HELPERS
    # ------------------------------------------------------------------
    def _update_killers(self, ply, move):
        if ply >= len(self.killers):
            return
        if self.killers[ply][0] != move:
            self.killers[ply][1] = self.killers[ply][0]
            self.killers[ply][0] = move

    def _age_history(self):
        for ci in range(2):
            for z in range(6):
                for r in range(8):
                    for c in range(8):
                        self.history[ci][z][r][c] //= 2

    # ------------------------------------------------------------------
    # PONDER
    # ------------------------------------------------------------------
    def ponder_indefinitely(self):
        while not self.cancellation_event.is_set():
            time.sleep(0.1)