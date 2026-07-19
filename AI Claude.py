# AI.py (Claude Challenger v3 — Book/Tablebase Foundation + NMP + LMR + Check Extensions + TT + QSearch)

import time
import random
import os
import json
import glob
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

# --- OPENING BOOK SETUP ---
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
# CHESS BOT CLASS
# ==============================================================================

class ChessBot:
    """
    Iterative-deepening alpha-beta search built on the book/tablebase
    foundation, adding:
      - Zobrist transposition table (mate-score-adjusted)
      - Tactically-aware move ordering (TT move -> AoE-swing captures ->
        killers -> history), reusing GameLogic's fast_approximate_material_swing
      - Null Move Pruning (off when in check / low material / near
        tablebase range, to avoid zugzwang and AoE blind spots)
      - Late Move Reductions restricted to quiet, non-tactical,
        non-check moves, with mandatory full-depth re-search verification
      - Budgeted check extensions for forced mating nets
      - Quiescence search resolving AoE swings past the horizon, itself
        also tablebase-aware
      - A tapered, AoE-aware evaluation function
    """
    # REQUIRED CLASS ATTRIBUTES - DO NOT REMOVE OR RENAME.
    search_depth = 4
    MATE_SCORE = 1000000

    # z_idx: 0=Pawn 1=Knight 2=Bishop 3=Rook 4=Queen 5=King
    PIECE_VALUES = [100, 340, 330, 550, 900, 0]

    TT_EXACT, TT_LOWER, TT_UPPER = 0, 1, 2
    MAX_QDEPTH = 20
    MAX_EXTENSIONS = 16

    # ---- Piece-square tables (white's perspective; row0=rank8 .. row7=rank1) --
    PAWN_PST = [
        [ 0,  0,  0,  0,  0,  0,  0,  0],
        [50, 50, 50, 50, 50, 50, 50, 50],
        [10, 10, 20, 30, 30, 20, 10, 10],
        [ 5,  5, 10, 25, 25, 10,  5,  5],
        [ 0,  0,  0, 20, 20,  0,  0,  0],
        [ 5, -5,-10,  0,  0,-10, -5,  5],
        [ 5, 10, 10,-20,-20, 10, 10,  5],
        [ 0,  0,  0,  0,  0,  0,  0,  0],
    ]
    KNIGHT_PST = [
        [-50,-40,-30,-30,-30,-30,-40,-50],
        [-40,-20,  0,  0,  0,  0,-20,-40],
        [-30,  0, 10, 15, 15, 10,  0,-30],
        [-30,  5, 15, 20, 20, 15,  5,-30],
        [-30,  0, 15, 20, 20, 15,  0,-30],
        [-30,  5, 10, 15, 15, 10,  5,-30],
        [-40,-20,  0,  5,  5,  0,-20,-40],
        [-50,-40,-30,-30,-30,-30,-40,-50],
    ]
    BISHOP_PST = [
        [-20,-10,-10,-10,-10,-10,-10,-20],
        [-10,  0,  0,  0,  0,  0,  0,-10],
        [-10,  0,  5, 10, 10,  5,  0,-10],
        [-10,  5,  5, 10, 10,  5,  5,-10],
        [-10,  0, 10, 10, 10, 10,  0,-10],
        [-10, 10, 10, 10, 10, 10, 10,-10],
        [-10,  5,  0,  0,  0,  0,  5,-10],
        [-20,-10,-10,-10,-10,-10,-10,-20],
    ]
    ROOK_PST = [
        [ 0,  0,  0,  0,  0,  0,  0,  0],
        [ 5, 10, 10, 10, 10, 10, 10,  5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [ 0,  0,  0,  5,  5,  0,  0,  0],
    ]
    QUEEN_PST = [
        [-20,-10,-10, -5, -5,-10,-10,-20],
        [-10,  0,  0,  0,  0,  0,  0,-10],
        [-10,  0,  5,  5,  5,  5,  0,-10],
        [ -5,  0,  5,  5,  5,  5,  0, -5],
        [  0,  0,  5,  5,  5,  5,  0, -5],
        [-10,  5,  5,  5,  5,  5,  0,-10],
        [-10,  0,  5,  0,  0,  0,  0,-10],
        [-20,-10,-10, -5, -5,-10,-10,-20],
    ]
    KING_MG = [
        [-30,-40,-40,-50,-50,-40,-40,-30],
        [-30,-40,-40,-50,-50,-40,-40,-30],
        [-30,-40,-40,-50,-50,-40,-40,-30],
        [-30,-40,-40,-50,-50,-40,-40,-30],
        [-20,-30,-30,-40,-40,-30,-30,-20],
        [-10,-20,-20,-20,-20,-20,-20,-10],
        [ 20, 20,  0,  0,  0,  0, 20, 20],
        [ 20, 30, 10,  0,  0, 10, 30, 20],
    ]
    KING_EG = [
        [-50,-40,-30,-20,-20,-30,-40,-50],
        [-30,-20,-10,  0,  0,-10,-20,-30],
        [-30,-10, 20, 30, 30, 20,-10,-30],
        [-30,-10, 30, 40, 40, 30,-10,-30],
        [-30,-10, 30, 40, 40, 30,-10,-30],
        [-30,-10, 20, 30, 30, 20,-10,-30],
        [-30,-30,  0,  0,  0,  0,-30,-30],
        [-50,-30,-30,-30,-30,-30,-30,-50],
    ]

    def __init__(self, board, color, position_counts, comm_queue, cancellation_event,
                 bot_name="Challenger AI", ply_count=0, game_mode="bot", max_moves=200,
                 time_left=None, increment=None, use_opening_book=True, use_tablebase=True):
        
        self.board = board
        self.color = color
        self.opponent_color = 'black' if color == 'white' else 'white'
        self.position_counts = position_counts
        self.comm_queue = comm_queue
        self.cancellation_event = cancellation_event
        self.ply_count = ply_count
        self.bot_name = bot_name
        self.max_moves = max_moves
        
        # --- TIME MANAGEMENT (10+0.1) ---
        self.time_left = time_left
        self.increment = increment
        self.stop_time = None
        self.nodes_searched = 0

        # --- DATABASE INTEGRATIONS ---
        self.use_opening_book = use_opening_book
        self.tb_manager = TablebaseManager()
        if not use_tablebase:
            self.tb_manager.probe = lambda b, t: None

        # --- SEARCH INFRASTRUCTURE ---
        self.tt = {}                # zobrist_hash -> (depth, score, flag, best_move)
        self.killers = {}           # ply -> [move, move]
        self.history = {}           # (color, start, end) -> score
        self._prev_best_move = None

        self.PST = [self.PAWN_PST, self.KNIGHT_PST, self.BISHOP_PST,
                    self.ROOK_PST, self.QUEEN_PST, None]

    # --- UI COMMUNICATION HELPERS (Do not modify) ---
    def _report_log(self, message):   self.comm_queue.put(('log', message))
    def _report_eval(self, score, depth): self.comm_queue.put(('eval', score if self.color == 'white' else -score, depth))
    def _report_move(self, move):     self.comm_queue.put(('move', move))
    
    def _format_move(self, board_before, move):
        if not move: return "None"
        child = board_before.clone()
        child.make_move(move[0], move[1])
        return format_move_san(board_before, child, move)

    # --- MAIN ENTRY POINT ---
    def make_move(self):
        root_moves = []
        try:
            root_moves = get_all_legal_moves(self.board, self.color)
            if not root_moves:
                self._report_move(None)
                return

            # --- Opening book ---
            if self.use_opening_book and self.ply_count <= 12:
                fen = board_to_fen(self.board, self.color)
                if fen in OPENING_BOOK:
                    try:
                        chosen = random.choices(
                            OPENING_BOOK[fen],
                            weights=[opt["weight"] for opt in OPENING_BOOK[fen]],
                            k=1
                        )[0]
                        move_tuple = (tuple(chosen["move"][0]), tuple(chosen["move"][1]))
                        if move_tuple in root_moves:
                            self._report_log(f"  > {self.bot_name} (Book): {chosen['san']}")
                            self._report_eval(chosen['score'], "Book")
                            self.comm_queue.put(('pv', chosen['score'], "Book", [chosen['san']], [move_tuple]))
                            self._report_move(move_tuple)
                            return
                    except Exception:
                        pass  # malformed/stale book entry -> fall through to search

            if len(root_moves) == 1:
                self.comm_queue.put(('pv', 0, "Forced", [self._format_move(self.board, root_moves[0])], [root_moves[0]]))
                self._report_move(root_moves[0])
                return

            # Keep the TT from growing without bound over a long game/session.
            if len(self.tt) > 2_000_000:
                self.tt.clear()
            # Fresh killer table each move; decay history so recent tactics dominate.
            self.killers = {}
            if self.history:
                for k in list(self.history.keys()):
                    v = self.history[k] // 2
                    if v <= 0:
                        del self.history[k]
                    else:
                        self.history[k] = v

            # 1. Time Management Calculation (Crucial for 10+0.1)
            search_start_time = time.time()
            if self.time_left is not None and self.increment is not None:
                base_time = self.time_left / 25.0
                allocated_time = base_time + (self.increment * 0.75)
                if self.time_left < 1.5:
                    # Emergency low-clock mode: play fast, bank on increment.
                    allocated_time = min(allocated_time, max(0.03, self.time_left * 0.4))
                safety_buffer = 0.15
                max_alloc = max(0.03, self.time_left - safety_buffer)
                allocated_time = max(0.03, min(allocated_time, max_alloc))
                self.stop_time = search_start_time + allocated_time
                target_depth = 100  # Go as deep as time allows
            else:
                self.stop_time = None
                target_depth = self.search_depth

            best_move_overall = root_moves[0]
            root_hash = board_hash(self.board, self.color)

            last_iter_duration = None

            # 2. Iterative Deepening Loop
            for current_depth in range(1, target_depth + 1):
                if self.stop_time:
                    now = time.time()
                    remaining = self.stop_time - now
                    if remaining <= 0:
                        break
                    # Predictive cutoff: don't start a deeper iteration we can't
                    # realistically finish (avoids wasting the whole iteration,
                    # which would otherwise get discarded on timeout anyway).
                    if last_iter_duration is not None and current_depth > 3:
                        projected = last_iter_duration * 4.0
                        if projected > remaining:
                            break

                iter_start_time = time.time()
                self.nodes_searched = 0
                
                try:
                    score, best_move_this_iter = self._search_root(current_depth, root_moves, root_hash)
                except SearchCancelledException:
                    break

                if self.stop_time and time.time() > self.stop_time:
                    break
                
                best_move_overall = best_move_this_iter
                last_iter_duration = time.time() - iter_start_time
                
                iter_duration = last_iter_duration
                knps = (self.nodes_searched / iter_duration / 1000) if iter_duration > 0 else 0
                eval_ui = score if self.color == 'white' else -score
                pv_san = self._format_move(self.board, best_move_overall)
                
                self._report_log(f"  > {self.bot_name} (D{current_depth}): {pv_san}, Eval={eval_ui/100:+.2f}, Nodes={self.nodes_searched}, KNPS={knps:.1f}, Time={iter_duration:.2f}s")
                self._report_eval(score, current_depth)
                self.comm_queue.put(('pv', eval_ui, current_depth, [pv_san], [best_move_overall]))

                if score > self.MATE_SCORE - 1000: 
                    break 

            self._report_move(best_move_overall)

        except Exception as e:
            self._report_log(f"CRASH: {str(e)}")
            self._report_move(root_moves[0] if root_moves else None)

    # ------------------------------------------------------------------
    # Move ordering helpers
    # ------------------------------------------------------------------
    def _score_move(self, board, move, color, tt_move, ply):
        """Returns (order_score, is_tactic). is_tactic is reused by the
        caller to gate LMR/extension decisions without recomputing the
        AoE-swing estimate twice."""
        start, end = move
        moving_piece = board.grid[start[0]][start[1]]
        target_piece = board.grid[end[0]][end[1]]

        swing, is_tactic = fast_approximate_material_swing(
            board, move, moving_piece, target_piece, self.PIECE_VALUES
        )

        if tt_move is not None and move == tt_move:
            return 20_000_000, is_tactic

        if is_tactic:
            return 10_000_000 + swing, is_tactic

        killers = self.killers.get(ply)
        if killers:
            if move == killers[0]:
                return 900_000, is_tactic
            if killers[1] is not None and move == killers[1]:
                return 800_000, is_tactic

        return self.history.get((color, start, end), 0), is_tactic

    def _store_killer(self, move, ply):
        k = self.killers.get(ply)
        if k is None:
            self.killers[ply] = [move, None]
        elif move != k[0]:
            k[1] = k[0]
            k[0] = move

    def _qsearch_moves(self, board, moves, include_all):
        scored = []
        for m in moves:
            start, end = m
            moving_piece = board.grid[start[0]][start[1]]
            target_piece = board.grid[end[0]][end[1]]
            swing, is_tactic = fast_approximate_material_swing(
                board, m, moving_piece, target_piece, self.PIECE_VALUES
            )
            if include_all or is_tactic:
                scored.append((swing, m))
        scored.sort(key=lambda x: -x[0])
        return scored

    def _has_non_pawn_material(self, board, turn):
        c = board.piece_counts_z[turn]
        return (c[1] + c[2] + c[3] + c[4]) > 0

    # ------------------------------------------------------------------
    # TT mate-score helpers (mate scores are ply-relative; must be
    # translated to/from "distance from root" when stored/retrieved).
    # ------------------------------------------------------------------
    def _tt_store_score(self, score, ply):
        if score >= self.MATE_SCORE - 1000:
            return score + ply
        elif score <= -self.MATE_SCORE + 1000:
            return score - ply
        return score

    def _tt_retrieve_score(self, score, ply):
        if score >= self.MATE_SCORE - 1000:
            return score - ply
        elif score <= -self.MATE_SCORE + 1000:
            return score + ply
        return score

    # ------------------------------------------------------------------
    # Tablebase probe helper (shared by negamax and qsearch). Returns a
    # ply-adjusted score relative to `turn`, or None if not applicable /
    # not found. The <=5-piece gate and the absolute->relative conversion
    # match the framework's required contract exactly.
    # ------------------------------------------------------------------
    def _probe_tablebase(self, turn, ply):
        if len(self.board.white_pieces) + len(self.board.black_pieces) > 5:
            return None
        tb_score_absolute = self.tb_manager.probe(self.board, turn)
        if tb_score_absolute is None:
            return None
        tb_score = tb_score_absolute if turn == 'white' else -tb_score_absolute
        if tb_score > self.MATE_SCORE - 1000:
            return tb_score - ply
        elif tb_score < -self.MATE_SCORE + 1000:
            return tb_score + ply
        return tb_score

    # ------------------------------------------------------------------
    # Root search
    # ------------------------------------------------------------------
    def _search_root(self, depth, root_moves, root_hash):
        """Root Alpha-Beta call. Returns (best_score, best_move)."""
        best_score = -float('inf')
        best_move = root_moves[0]
        alpha = -float('inf')
        beta = float('inf')

        tt_entry = self.tt.get(root_hash)
        tt_move = tt_entry[3] if tt_entry is not None else None

        scored_moves = []
        for move in root_moves:
            s, _ = self._score_move(self.board, move, self.color, tt_move, 0)
            if self._prev_best_move is not None and move == self._prev_best_move:
                s += 15_000_000
            scored_moves.append((s, move))
        scored_moves.sort(key=lambda x: -x[0])

        for _, move in scored_moves:
            if self.cancellation_event.is_set() or (self.stop_time and time.time() > self.stop_time):
                raise SearchCancelledException()

            # try/finally is mandatory here: if a deeper call raises
            # SearchCancelledException (a normal event under a real clock),
            # the board MUST still be unwound at every level of the stack,
            # or the position is left permanently corrupted.
            record = self.board.make_move_track(move[0], move[1])
            try:
                child_hash = incremental_hash(root_hash, record)
                score = -self.negamax(depth - 1, -beta, -alpha, self.opponent_color, 1, child_hash, 0)
            finally:
                self.board.unmake_move(record)

            if score > best_score:
                best_score = score
                best_move = move
            if best_score > alpha:
                alpha = best_score

        self._prev_best_move = best_move
        self.tt[root_hash] = (depth, self._tt_store_score(best_score, 0), self.TT_EXACT, best_move)

        return best_score, best_move

    # ------------------------------------------------------------------
    # Main search
    # ------------------------------------------------------------------
    def negamax(self, depth, alpha, beta, turn, ply, current_hash, ext):
        """Recursive Alpha-Beta Search (fail-soft, with tablebase probe,
        TT, NMP, LMR and check extensions)."""
        self.nodes_searched += 1
        
        if (self.nodes_searched & 1023) == 0:
            if self.cancellation_event.is_set() or (self.stop_time and time.time() > self.stop_time):
                raise SearchCancelledException()

        board = self.board

        # --- TABLEBASE PROBE ---
        # Exact endgame scores take priority over everything else whenever
        # <=5 pieces remain on the board.
        tb_score = self._probe_tablebase(turn, ply)
        if tb_score is not None:
            return tb_score

        in_check = is_in_check(board, turn)

        # --- Check extension: forced-sequence horizon safety, budgeted. ---
        if in_check and ext < self.MAX_EXTENSIONS:
            depth += 1
            ext += 1

        alpha_orig = alpha

        tt_move = None
        tt_entry = self.tt.get(current_hash)
        if tt_entry is not None:
            tt_depth, tt_score_raw, tt_flag, tt_move = tt_entry
            if tt_depth >= depth:
                tt_score2 = self._tt_retrieve_score(tt_score_raw, ply)
                if tt_flag == self.TT_EXACT:
                    return tt_score2
                elif tt_flag == self.TT_LOWER and tt_score2 >= beta:
                    return tt_score2
                elif tt_flag == self.TT_UPPER and tt_score2 <= alpha:
                    return tt_score2

        if depth <= 0:
            return self.qsearch(alpha, beta, turn, ply, current_hash, 0)

        opponent = 'black' if turn == 'white' else 'white'

        # --- Null Move Pruning -------------------------------------------------
        # Skipped when: in check, too shallow, side to move has no non-pawn
        # material (zugzwang risk), we're near tablebase range (exact search
        # matters more than pruning there), or the window already touches a
        # mate score. Board state doesn't need to change for the null move
        # since "turn" is passed as a parameter rather than stored on Board.
        if (not in_check and depth >= 3
                and self._has_non_pawn_material(board, turn)
                and (len(board.white_pieces) + len(board.black_pieces)) > 6
                and beta < self.MATE_SCORE - 1000
                and alpha > -self.MATE_SCORE + 1000):
            R = 3 if depth > 6 else 2
            null_depth = depth - 1 - R
            if null_depth >= 1:
                null_hash = current_hash ^ ZOBRIST_TURN
                null_score = -self.negamax(null_depth, -beta, -beta + 1, opponent, ply + 1, null_hash, ext)
                if null_score >= beta:
                    return null_score

        # Get pseudo-legal moves and order them
        moves = get_all_pseudo_legal_moves(board, turn)

        scored_moves = []
        for move in moves:
            s, is_tactic = self._score_move(board, move, turn, tt_move, ply)
            scored_moves.append((s, move, is_tactic))
        scored_moves.sort(key=lambda x: -x[0])

        best_score = -float('inf')
        best_move_here = None
        legal_moves_count = 0

        for _, move, is_tactic in scored_moves:
            start, end = move
            target_piece = board.grid[end[0]][end[1]]

            record = board.make_move_track(start, end)
            try:
                if is_in_check(board, turn):
                    continue

                legal_moves_count += 1
                child_hash = incremental_hash(current_hash, record)

                # --- Late Move Reductions ---------------------------------
                # Only ever applied to quiet, non-tactical, non-check-involved
                # moves ordered late in the list — AoE swings and checks
                # always get full-depth treatment. Any reduced move that
                # still beats alpha is re-searched at full depth before
                # being trusted. is_in_check(opponent) is only evaluated
                # when everything else already qualifies, so shallow nodes
                # (where LMR can never fire) don't pay for it.
                do_reduce = (
                    depth >= 3 and legal_moves_count > 4
                    and not is_tactic and not in_check
                    and move != tt_move
                    and not is_in_check(board, opponent)
                )

                if do_reduce:
                    reduction = 2 if (depth >= 6 and legal_moves_count > 8) else 1
                    score = -self.negamax(depth - 1 - reduction, -alpha - 1, -alpha, opponent, ply + 1, child_hash, ext)
                    if score > alpha:
                        score = -self.negamax(depth - 1, -beta, -alpha, opponent, ply + 1, child_hash, ext)
                else:
                    score = -self.negamax(depth - 1, -beta, -alpha, opponent, ply + 1, child_hash, ext)
            finally:
                board.unmake_move(record)

            if score > best_score:
                best_score = score
                best_move_here = move

            if best_score > alpha:
                alpha = best_score

            if alpha >= beta:
                if target_piece is None:
                    self._store_killer(move, ply)
                    key = (turn, start, end)
                    self.history[key] = self.history.get(key, 0) + depth * depth
                break

        if legal_moves_count == 0:
            # No legal moves is always an immediate loss in Jungle Chess,
            # whether or not the side to move is currently in check.
            return -self.MATE_SCORE + ply

        flag = self.TT_EXACT
        if best_score <= alpha_orig:
            flag = self.TT_UPPER
        elif best_score >= beta:
            flag = self.TT_LOWER
        self.tt[current_hash] = (depth, self._tt_store_score(best_score, ply), flag, best_move_here)

        return best_score

    # ------------------------------------------------------------------
    # Quiescence search — resolves tactical AoE swings (captures,
    # explosions, evaporations, piercing, promotions) past the main
    # search horizon to avoid horizon-effect blunders. Also tablebase-
    # aware, since captures inside qsearch routinely cross into <=5-piece
    # territory.
    # ------------------------------------------------------------------
    def qsearch(self, alpha, beta, turn, ply, current_hash, qdepth):
        self.nodes_searched += 1

        if (self.nodes_searched & 1023) == 0:
            if self.cancellation_event.is_set() or (self.stop_time and time.time() > self.stop_time):
                raise SearchCancelledException()

        if qdepth > 40:
            return self.evaluate_board(turn)

        tb_score = self._probe_tablebase(turn, ply)
        if tb_score is not None:
            return tb_score

        board = self.board
        in_check = is_in_check(board, turn)
        opponent = 'black' if turn == 'white' else 'white'

        if not in_check:
            stand_pat = self.evaluate_board(turn)
            if stand_pat >= beta:
                return stand_pat
            best_score = stand_pat
            if stand_pat > alpha:
                alpha = stand_pat
            if qdepth >= self.MAX_QDEPTH:
                return best_score
        else:
            best_score = -self.MATE_SCORE + ply

        moves = get_all_pseudo_legal_moves(board, turn)
        scored = self._qsearch_moves(board, moves, in_check)

        legal_count = 0
        for _, move in scored:
            record = board.make_move_track(move[0], move[1])
            try:
                if is_in_check(board, turn):
                    continue

                legal_count += 1
                child_hash = incremental_hash(current_hash, record)
                score = -self.qsearch(-beta, -alpha, opponent, ply + 1, child_hash, qdepth + 1)
            finally:
                board.unmake_move(record)

            if score > best_score:
                best_score = score
            if best_score > alpha:
                alpha = best_score
            if alpha >= beta:
                break

        if in_check and legal_count == 0:
            return -self.MATE_SCORE + ply

        return best_score

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate_board(self, turn_to_move):
        """
        Material + tapered piece-square tables + AoE-aware structural
        bonuses (bishop pair, rook open files / railgun alignment, knight
        aggression near the enemy king).
        """
        board = self.board

        if is_insufficient_material(board):
            return 0

        pcz = board.piece_counts_z
        w = pcz['white']
        b = pcz['black']

        phase = (w[1] + b[1]) * 1 + (w[2] + b[2]) * 1 + (w[3] + b[3]) * 2 + (w[4] + b[4]) * 4
        if phase > 24:
            phase = 24
        mg_weight = phase / 24.0
        eg_weight = 1.0 - mg_weight

        score = 0
        PV = self.PIECE_VALUES
        PST = self.PST
        KING_MG = self.KING_MG
        KING_EG = self.KING_EG

        for p in board.white_pieces:
            z = p.z_idx
            r, c = p.pos
            score += PV[z]
            if z == 5:
                score += KING_MG[r][c] * mg_weight + KING_EG[r][c] * eg_weight
            else:
                score += PST[z][r][c]

        for p in board.black_pieces:
            z = p.z_idx
            r, c = p.pos
            mr = 7 - r
            score -= PV[z]
            if z == 5:
                score -= KING_MG[mr][c] * mg_weight + KING_EG[mr][c] * eg_weight
            else:
                score -= PST[z][mr][c]

        if w[2] >= 2:
            score += 30
        if b[2] >= 2:
            score -= 30

        score += self._rook_file_bonus(board, 'white')
        score -= self._rook_file_bonus(board, 'black')

        if board.black_king_pos:
            score += self._railgun_alignment_bonus(board, 'white', board.black_king_pos)
            score += self._knight_aggression_bonus(board, 'white', board.black_king_pos)
        if board.white_king_pos:
            score -= self._railgun_alignment_bonus(board, 'black', board.white_king_pos)
            score -= self._knight_aggression_bonus(board, 'black', board.white_king_pos)

        return score if turn_to_move == 'white' else -score

    def _rook_file_bonus(self, board, color):
        counts = board.piece_counts_z[color]
        if counts[3] == 0:
            return 0
        bonus = 0
        grid = board.grid
        pieces = board.white_pieces if color == 'white' else board.black_pieces
        for p in pieces:
            if p.z_idx == 3 and p.pos is not None:
                c = p.pos[1]
                own_pawn = False
                enemy_pawn = False
                for r in range(ROWS):
                    sq = grid[r][c]
                    if sq is not None and sq.z_idx == 0:
                        if sq.color == color:
                            own_pawn = True
                        else:
                            enemy_pawn = True
                if not own_pawn and not enemy_pawn:
                    bonus += 20
                elif not own_pawn:
                    bonus += 10
        return bonus

    def _railgun_alignment_bonus(self, board, color, enemy_king_pos):
        kr, kc = enemy_king_pos
        bonus = 0
        pieces = board.white_pieces if color == 'white' else board.black_pieces
        for p in pieces:
            if p.z_idx == 3 and p.pos is not None:
                if p.pos[0] == kr or p.pos[1] == kc:
                    bonus += 12
        return bonus

    def _knight_aggression_bonus(self, board, color, enemy_king_pos):
        kr, kc = enemy_king_pos
        bonus = 0
        pieces = board.white_pieces if color == 'white' else board.black_pieces
        for p in pieces:
            if p.z_idx == 1 and p.pos is not None:
                dist = max(abs(p.pos[0] - kr), abs(p.pos[1] - kc))
                if dist <= 3:
                    bonus += (4 - dist) * 6
        return bonus

    def ponder_indefinitely(self):
        """Called when UI depth slider is set to 99."""
        while not self.cancellation_event.is_set():
            time.sleep(0.1)