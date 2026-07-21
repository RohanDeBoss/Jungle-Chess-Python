# AI.py (Gemini v9)
import time
import random
import os
import json
import glob
from operator import itemgetter
from collections import namedtuple
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
# JUNGLE CHESS EVALUATION TABLES & CUSTOM STRUCTURES (Synced from OPAI)
# ==============================================================================

MG_PIECE_VALUES = [100, 950, 600, 600, 1300, 20000]
EG_PIECE_VALUES = [100, 950, 700, 750, 1000, 20000]
INITIAL_PHASE_MATERIAL = (600 * 4 + 950 * 4 + 600 * 4 + 1300 * 2)

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

# Fast Pre-Calculated Flattened PSTs
FLAT_PST_MG_WHITE = [[0]*64 for _ in range(6)]
FLAT_PST_MG_BLACK = [[0]*64 for _ in range(6)]
FLAT_PST_EG_WHITE = [[0]*64 for _ in range(6)]
FLAT_PST_EG_BLACK = [[0]*64 for _ in range(6)]

tables_mg = [pawn_pst, knight_pst, bishop_pst, rook_pst, queen_pst, king_midgame_pst]
tables_eg = [pawn_endgame_pst, knight_pst, bishop_pst, rook_pst, queen_pst, king_endgame_pst]

for z in range(6):
    for r in range(8):
        for c in range(8):
            sq_w = r * 8 + c
            sq_b = (7 - r) * 8 + c
            FLAT_PST_MG_WHITE[z][sq_w] = MG_PIECE_VALUES[z] + tables_mg[z][r][c]
            FLAT_PST_MG_BLACK[z][sq_b] = MG_PIECE_VALUES[z] + tables_mg[z][r][c]
            FLAT_PST_EG_WHITE[z][sq_w] = EG_PIECE_VALUES[z] + tables_eg[z][r][c]
            FLAT_PST_EG_BLACK[z][sq_b] = EG_PIECE_VALUES[z] + tables_eg[z][r][c]

TTEntry = namedtuple('TTEntry', ['score', 'depth', 'flag', 'best_move'])
TT_FLAG_EXACT, TT_FLAG_LOWERBOUND, TT_FLAG_UPPERBOUND = 0, 1, 2
INFINITY = 10000000

# ==============================================================================
# CHESS BOT CLASS 
# ==============================================================================

class ChessBot:
    search_depth = 4  
    MATE_SCORE = 1000000
    DRAW_SCORE = 0
    
    ASP_WINDOW_INIT = 250
    ASP_MAX_RETRIES = 3
    MIN_MOVE_TIME = 0.03
    TIME_BUFFER_SEC = 0.50

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
        
        self.time_left = time_left
        self.increment = increment
        self.stop_time = None
        
        if time_left:
             allocated = (self.time_left / 30.0) + (self.increment * 0.8)
             self.time_check_mask = self._calc_time_check_mask(allocated)
        else:
             self.time_check_mask = 511

        self.use_opening_book = use_opening_book
        self.tb_manager = TablebaseManager()
        if not use_tablebase:
            self.tb_manager.probe = lambda b, t: None

        # --- OPAI DATA STRUCTURES ---
        self.tt = {}
        self.eval_tt = {} # Evaluation Cache
        self.history_table = {}
        self.killer_moves = [[None, None] for _ in range(128)]
        # Continuation History: 5D array [color][prev_z][prev_to][my_z][my_to]
        self.continuation_history = [[[[[0] * 64 for _ in range(6)] for _ in range(64)] for _ in range(6)] for _ in range(2)]
        
        self.nodes_searched = 0
        self.tb_hits = 0

    def _calc_time_check_mask(self, allocated):
        if allocated <= 0.15: return 15
        if allocated <= 0.30: return 31
        if allocated <= 0.60: return 63
        if allocated <= 1.20: return 127
        if allocated <= 2.50: return 255
        return 511

    def _report_log(self, message):       self.comm_queue.put(('log', message))
    def _report_eval(self, score, depth): self.comm_queue.put(('eval', score if self.color == 'white' else -score, depth))
    def _report_move(self, move):         self.comm_queue.put(('move', move))

    def _format_move(self, board_before, move):
        if not move: return "None"
        child = board_before.clone()
        child.make_move(move[0], move[1])
        # Reverted the hacky regex. Native casualty string rendering is allowed.
        return format_move_san(board_before, child, move)

    def _store_tt(self, hash_val, score, depth, flag, move):
        if len(self.tt) > 5000000:
            self.tt.clear()
        existing = self.tt.get(hash_val)
        if not existing or depth >= existing.depth:
            best_move = move if move is not None else (existing.best_move if existing else None)
            self.tt[hash_val] = TTEntry(score, depth, flag, best_move)

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

    def check_time(self):
        if self.cancellation_event.is_set():
            raise SearchCancelledException()
        if self.stop_time and time.time() > self.stop_time:
            raise SearchCancelledException()

    def make_move(self):
        try:
            if self.use_opening_book and self.ply_count <= 12:
                fen = board_to_fen(self.board, self.color)
                if fen in OPENING_BOOK:
                    chosen = random.choices(OPENING_BOOK[fen], weights=[opt["weight"] for opt in OPENING_BOOK[fen]], k=1)[0]
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
                self.comm_queue.put(('pv', 0, "Forced", [self._format_move(self.board, root_moves[0])], [root_moves[0]]))
                self._report_move(root_moves[0])
                return

            search_start_time = time.time()
            if self.time_left is not None and self.increment is not None:
                buffer = max(self.TIME_BUFFER_SEC, self.time_left * 0.05, self.increment * 1.5)
                clock_ceiling = max(0.0, self.time_left - buffer)
                
                buffer_health = max(0.0, min(1.0, clock_ceiling / max(0.1, self.time_left)))
                divisor = 50 - (20 * buffer_health)

                optimum_time = (self.time_left / divisor) + (self.increment * 0.8)
                optimum_time = min(max(self.MIN_MOVE_TIME, optimum_time), clock_ceiling)

                max_time = min(clock_ceiling, optimum_time * 3.5)
                max_time = max(max_time, min(self.MIN_MOVE_TIME, clock_ceiling))

                self.stop_time = search_start_time + max_time
                target_depth = 100
            else:
                self.stop_time = None
                optimum_time = float('inf')
                max_time = float('inf')
                target_depth = self.search_depth

            best_move_overall  = root_moves[0]
            prev_iter_score    = None
            prev_iter_duration = None
            total_nodes        = 0
            root_hash          = board_hash(self.board, self.color)

            for current_depth in range(1, target_depth + 1):
                iter_start_time = time.time()
                
                try:
                    best_score_this_iter, best_move_this_iter = self._run_depth_iteration(
                        current_depth, root_moves, root_hash, best_move_overall, prev_iter_score=prev_iter_score)
                except SearchCancelledException:
                    break

                best_move_overall = best_move_this_iter
                prev_iter_score   = best_score_this_iter
                total_nodes      += self.nodes_searched
                iter_duration     = time.time() - iter_start_time
                knps              = (self.nodes_searched / iter_duration / 1000) if iter_duration > 0 else 0
                
                root_tb_val = None
                if len(self.board.white_pieces) + len(self.board.black_pieces) <= 5:
                    root_tb_val = self.tb_manager.probe(self.board, self.color)

                if root_tb_val is not None:
                    self.tb_hits += 1
                    depth_label = "TB"
                    report_score = root_tb_val
                else:
                    depth_label = current_depth
                    report_score = best_score_this_iter

                eval_for_ui = report_score if self.color == 'white' else -report_score
                self._report_log(f"  > {self.bot_name} (D{depth_label}): {self._format_move(self.board, best_move_this_iter)}, Eval={eval_for_ui/100:+.2f}, NodesTotal={total_nodes}, KNPS={knps:.1f}, TBhits={self.tb_hits}, Time={iter_duration:.2f}s")
                self._report_eval(report_score, depth_label)

                pv_str, pv_raw = self._get_pv_data(current_depth, best_move_this_iter)
                self.comm_queue.put(('pv', eval_for_ui, depth_label, pv_str, pv_raw))

                if report_score > self.MATE_SCORE - 2000 or report_score < -self.MATE_SCORE + 2000: 
                    break 

                if self.stop_time:
                    time_used = time.time() - search_start_time
                    if time_used > optimum_time:
                        break

                    if prev_iter_duration and prev_iter_duration > 0:
                        effective_branching = iter_duration / prev_iter_duration
                        effective_branching = max(1.5, min(effective_branching, 8.0))
                    else:
                        effective_branching = 6.0

                    predicted_next_duration = iter_duration * effective_branching
                    time_remaining_to_max   = self.stop_time - time.time()

                    if predicted_next_duration > time_remaining_to_max * 0.85:
                        break

                prev_iter_duration = iter_duration

            self._report_move(best_move_overall)

        except Exception as e:
            self._report_log(f"CRASH: {str(e)}")
            self._report_move(root_moves[0] if root_moves else None)

    def _run_depth_iteration(self, depth, root_moves, root_hash, pv_move, prev_iter_score=None):
        self.nodes_searched = 0
        use_aspiration = (prev_iter_score is not None and depth >= 2)

        if use_aspiration:
            window = self.ASP_WINDOW_INIT
            alpha_bound = prev_iter_score - window
            beta_bound  = prev_iter_score + window
            retries = 0
            
            while True:
                best_score, best_move = self._search_at_depth(
                    depth, root_moves, root_hash, pv_move, alpha_bound, beta_bound)
                
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
                        depth, root_moves, root_hash, pv_move, -INFINITY, INFINITY)
                    break
        else:
            best_score, best_move = self._search_at_depth(
                depth, root_moves, root_hash, pv_move, -INFINITY, INFINITY)

        return best_score, best_move

    def _score_moves(self, moves, tt_move, ply, turn, prev_move_tuple=None):
        scored = []
        c_idx = 0 if turn == 'white' else 1
        
        for move in moves:
            if move == tt_move:
                scored.append((20000000, move, False, None))
                continue
                
            mp = self.board.grid[move[0][0]][move[0][1]]
            tp = self.board.grid[move[1][0]][move[1][1]]
            
            swing, is_tactic = fast_approximate_material_swing(self.board, move, mp, tp, MG_PIECE_VALUES)
            if is_tactic:
                attacker_penalty = MG_PIECE_VALUES[mp.z_idx] if mp.z_idx != 4 else 0
                if swing < 0:
                    score = 500000 + swing 
                else:
                    score = 10000000 + (swing * 10) - attacker_penalty
                scored.append((score, move, True, mp))
            else:
                score = 0
                if ply < 128:
                    if move == self.killer_moves[ply][0]: score = 900000
                    elif move == self.killer_moves[ply][1]: score = 800000
                if score == 0:
                    # Continuation History lookup
                    if prev_move_tuple:
                        prev_move, prev_pt_idx = prev_move_tuple
                        pr, pc = prev_move[1]
                        prev_to_sq = pr * 8 + pc
                        to_sq = move[1][0] * 8 + move[1][1]
                        ch_score = self.continuation_history[c_idx][prev_pt_idx][prev_to_sq][mp.z_idx][to_sq]
                        if ch_score > 1000:
                            score = 700000 + ch_score
                if score == 0:
                    score = min(self.history_table.get((mp.z_idx, move[1]), 0), 50000)
                scored.append((score, move, False, mp))
                
        scored.sort(key=itemgetter(0), reverse=True)
        return scored

    def _search_at_depth(self, depth, root_moves, root_hash, pv_move, alpha, beta):
        best_score_this_iter = -INFINITY
        best_move_this_iter = pv_move if pv_move in root_moves else (root_moves[0] if root_moves else None)

        scored_moves = self._score_moves(root_moves, pv_move, 0, self.color)

        for i, (move_score, move, is_tactic, mp) in enumerate(scored_moves):
            self.check_time()
            record = self.board.make_move_track(move[0], move[1])
            child_hash = incremental_hash(root_hash, record)
            next_prev_tuple = (move, mp.z_idx) if mp else None

            if i == 0:
                score = -self.negamax(depth - 1, -beta, -alpha, self.opponent_color, 1, child_hash, False, next_prev_tuple)
            else:
                score = -self.negamax(depth - 1, -alpha - 1, -alpha, self.opponent_color, 1, child_hash, False, next_prev_tuple)
                if score > alpha and score < beta: 
                    score = -self.negamax(depth - 1, -beta, -alpha, self.opponent_color, 1, child_hash, False, next_prev_tuple)
            
            self.board.unmake_move(record)

            if score > best_score_this_iter:
                best_score_this_iter = score
                best_move_this_iter = move
                
            if score > alpha:
                alpha = score

        return best_score_this_iter, best_move_this_iter

    def negamax(self, depth, alpha, beta, turn, ply, current_hash, is_null=False, prev_move_tuple=None):
        self.nodes_searched += 1
        if (self.nodes_searched & self.time_check_mask) == 0:
            self.check_time()

        if len(self.board.white_pieces) + len(self.board.black_pieces) <= 5:
            tb_score_absolute = self.tb_manager.probe(self.board, turn)
            if tb_score_absolute is not None:
                self.tb_hits += 1
                tb_score = tb_score_absolute if turn == 'white' else -tb_score_absolute
                if tb_score >  self.MATE_SCORE - 1000: return tb_score - ply
                elif tb_score < -self.MATE_SCORE + 1000: return tb_score + ply
                return tb_score

        alpha_orig = alpha
        
        tt_entry = self.tt.get(current_hash)
        tt_move = None
        if tt_entry is not None:
            tt_score = tt_entry.score
            tt_move = tt_entry.best_move
            
            if tt_score > self.MATE_SCORE - 1000: tt_score -= ply
            elif tt_score < -self.MATE_SCORE + 1000: tt_score += ply
            
            if tt_entry.depth >= depth:
                if tt_entry.flag == TT_FLAG_EXACT:
                    return tt_score
                elif tt_entry.flag == TT_FLAG_LOWERBOUND:
                    alpha = max(alpha, tt_score)
                elif tt_entry.flag == TT_FLAG_UPPERBOUND:
                    beta = min(beta, tt_score)
                if alpha >= beta:
                    return tt_score

        if depth <= 0:
            return self.qsearch(alpha, beta, turn, ply, current_hash)

        in_check = is_in_check(self.board, turn)
        opponent = 'black' if turn == 'white' else 'white'

        # --- Null Move Pruning (NMP) ---
        if depth >= 3 and not in_check and ply > 0 and not is_null and beta < INFINITY - 1000:
            pcz = self.board.piece_counts_z[turn]
            npm = pcz[1]*950 + pcz[2]*600 + pcz[3]*600 + pcz[4]*1300
            if npm > 0:
                null_hash = current_hash ^ ZOBRIST_TURN
                score = -self.negamax(depth - 1 - 2, -beta, -beta + 1, opponent, ply + 1, null_hash, True, None)
                if score >= beta:
                    return beta

        moves = get_all_pseudo_legal_moves(self.board, turn)
        scored_moves = self._score_moves(moves, tt_move, ply, turn, prev_move_tuple)

        legal_moves_count = 0
        best_move = None
        best_score = -INFINITY

        for move_score, move, is_tactic, mp in scored_moves:
            record = self.board.make_move_track(move[0], move[1])
            
            if is_in_check(self.board, turn):
                self.board.unmake_move(record)
                continue
                
            legal_moves_count += 1
            child_hash = incremental_hash(current_hash, record)
            next_prev_tuple = (move, mp.z_idx) if mp else None

            # --- Jungle-Aware Late Move Reductions (LMR) ---
            reduction = 0
            if depth >= 3 and legal_moves_count > 4 and not in_check and not is_tactic:
                reduction = 1 + (depth // 7) + (legal_moves_count // 12)
                
                # Protect King's safety by refusing to reduce Knights and attacking Queens
                if mp.z_idx == 1:
                    reduction -= 1
                elif mp.z_idx == 4:
                    attacks_enemy = False
                    grid_ref = self.board.grid
                    t_sq = move[1][0] * 8 + move[1][1]
                    for ray in RAYS[t_sq]:
                        for cr, cc in ray:
                            target = grid_ref[cr][cc]
                            if target is not None:
                                if target.color == opponent:
                                    attacks_enemy = True
                                break
                        if attacks_enemy:
                            break
                    if attacks_enemy:
                        reduction -= 1
                        
                reduction = max(0, min(reduction, depth - 2))

            # --- PVS Core ---
            if legal_moves_count == 1:
                score = -self.negamax(depth - 1, -beta, -alpha, opponent, ply + 1, child_hash, False, next_prev_tuple)
            else:
                if reduction > 0:
                    score = -self.negamax(depth - 1 - reduction, -alpha - 1, -alpha, opponent, ply + 1, child_hash, False, next_prev_tuple)
                    if score > alpha:
                        score = -self.negamax(depth - 1, -alpha - 1, -alpha, opponent, ply + 1, child_hash, False, next_prev_tuple)
                        if alpha < score < beta:
                            score = -self.negamax(depth - 1, -beta, -alpha, opponent, ply + 1, child_hash, False, next_prev_tuple)
                else:
                    score = -self.negamax(depth - 1, -alpha - 1, -alpha, opponent, ply + 1, child_hash, False, next_prev_tuple)
                    if alpha < score < beta: 
                        score = -self.negamax(depth - 1, -beta, -alpha, opponent, ply + 1, child_hash, False, next_prev_tuple)
            
            self.board.unmake_move(record)

            if score > best_score:
                best_score = score
                best_move = move
                
            if score > alpha:
                alpha = score
                
            if alpha >= beta:
                if not is_tactic and ply < 128:
                    if move != self.killer_moves[ply][0]:
                        self.killer_moves[ply][1] = self.killer_moves[ply][0]
                        self.killer_moves[ply][0] = move
                    if mp:
                        bonus = depth * depth
                        # Base History Update
                        self.history_table[(mp.z_idx, move[1])] = min(self.history_table.get((mp.z_idx, move[1]), 0) + bonus, 50000)
                        
                        # Continuation History Update
                        if prev_move_tuple:
                            c_idx = 0 if turn == 'white' else 1
                            prev_move, prev_pt_idx = prev_move_tuple
                            pr, pc = prev_move[1]
                            prev_to_sq = pr * 8 + pc
                            to_sq = move[1][0] * 8 + move[1][1]
                            ch_table = self.continuation_history[c_idx][prev_pt_idx][prev_to_sq][mp.z_idx]
                            ch_table[to_sq] = min(ch_table[to_sq] + bonus, 50000)
                break

        if legal_moves_count == 0:
            return -self.MATE_SCORE + ply 

        flag = TT_FLAG_EXACT
        if best_score <= alpha_orig:
            flag = TT_FLAG_UPPERBOUND
        elif best_score >= beta:
            flag = TT_FLAG_LOWERBOUND
            
        stored_score = best_score
        if stored_score > self.MATE_SCORE - 1000: stored_score += ply
        elif stored_score < -self.MATE_SCORE + 1000: stored_score -= ply
            
        self._store_tt(current_hash, stored_score, depth, flag, best_move)
        
        return best_score

    def qsearch(self, alpha, beta, turn, ply, current_hash):
        self.nodes_searched += 1
        if (self.nodes_searched & self.time_check_mask) == 0:
            self.check_time()
            
        in_check = is_in_check(self.board, turn)
        
        if not in_check:
            # Eval TT check
            if current_hash in self.eval_tt:
                stand_pat = self.eval_tt[current_hash]
            else:
                stand_pat = self.evaluate_board(turn)
                if len(self.eval_tt) > 5000000: self.eval_tt.clear()
                self.eval_tt[current_hash] = stand_pat
                
            if stand_pat >= beta:
                return beta
            if alpha < stand_pat:
                alpha = stand_pat
                
            # Delta Pruning
            if stand_pat + 1300 < alpha:
                return alpha
            
            best_score = stand_pat
        else:
            best_score = -INFINITY
                
        moves = get_all_pseudo_legal_moves(self.board, turn)
        tactical_moves = []
        opponent = 'black' if turn == 'white' else 'white'
        has_enemy_knights = self.board.piece_counts[opponent][Knight] > 0
        grid = self.board.grid
        
        for move in moves:
            (r1, c1), (r2, c2) = move
            mp = grid[r1][c1]
            tp = grid[r2][c2]
            
            # Fast pass check for quiet moves exactly like OPAI
            if not in_check and tp is None:
                if mp.z_idx in (2, 4, 5) or (mp.z_idx == 0 and r2 != mp.promo_rank):
                    gets_evaporated = False
                    if has_enemy_knights:
                        for kr, kc in KNIGHT_ATTACKS_FROM[(r2, c2)]:
                            kp = grid[kr][kc]
                            if kp is not None and kp.z_idx == 1 and kp.color == opponent:
                                gets_evaporated = True
                                break
                    if not gets_evaporated:
                        continue
            
            swing, is_tactic = fast_approximate_material_swing(self.board, move, mp, tp, MG_PIECE_VALUES)
            if in_check or is_tactic:
                attacker_penalty = MG_PIECE_VALUES[mp.z_idx] if mp.z_idx != 4 else 0
                score = (swing * 10) - attacker_penalty
                tactical_moves.append((score, move))
                
        tactical_moves.sort(key=itemgetter(0), reverse=True)
        
        legal_tactics_searched = 0
        
        for swing, move in tactical_moves:
            record = self.board.make_move_track(move[0], move[1])
            if is_in_check(self.board, turn):
                self.board.unmake_move(record)
                continue
                
            legal_tactics_searched += 1
            child_hash = incremental_hash(current_hash, record)
            
            score = -self.qsearch(-beta, -alpha, opponent, ply + 1, child_hash)
            self.board.unmake_move(record)
            
            if score > best_score:
                best_score = score
            if score > alpha:
                alpha = score
            if alpha >= beta:
                return beta

        if in_check and legal_tactics_searched == 0:
            return -self.MATE_SCORE + ply
            
        return best_score

    def evaluate_board(self, turn_to_move):
        if is_insufficient_material(self.board):
            return self.DRAW_SCORE

        scores_mg = [0, 0]
        scores_eg = [0, 0]
        
        pcz_w = self.board.piece_counts_z['white']
        pcz_b = self.board.piece_counts_z['black']
        
        phase_material = (pcz_w[1]+pcz_b[1])*950 + (pcz_w[2]+pcz_b[2])*600 + (pcz_w[3]+pcz_b[3])*600 + (pcz_w[4]+pcz_b[4])*1300
        phase = min(256, (phase_material * 256) // INITIAL_PHASE_MATERIAL) if INITIAL_PHASE_MATERIAL > 0 else 0
        inv_phase = 256 - phase
        
        king_zone_attacks = [0, 0]
        king_pos = [self.board.white_king_pos, self.board.black_king_pos]

        for color_idx in (0, 1):
            pieces = self.board.white_pieces if color_idx == 0 else self.board.black_pieces
            my_color_name = 'white' if color_idx == 0 else 'black'
            enemy_king = king_pos[1 - color_idx]
            pst_mg = FLAT_PST_MG_WHITE if color_idx == 0 else FLAT_PST_MG_BLACK
            pst_eg = FLAT_PST_EG_WHITE if color_idx == 0 else FLAT_PST_EG_BLACK
            
            for piece in pieces:
                z = piece.z_idx
                r, c = piece.pos
                sq = r * 8 + c
                
                scores_mg[color_idx] += pst_mg[z][sq]
                scores_eg[color_idx] += pst_eg[z][sq]
                
                if z == 1: # Knight
                    ekr, ekc = enemy_king if enemy_king else (None, None)
                    if ekr is not None:
                        for ar, ac in KNIGHT_ATTACKS_FROM[(r, c)]:
                            if abs(ar - ekr) <= 2 and abs(ac - ekc) <= 2:
                                king_zone_attacks[1 - color_idx] += 1
                                break
                                
                elif z == 4: # Queen
                    if enemy_king and (abs(r - enemy_king[0]) + abs(c - enemy_king[1]) <= 3):
                        king_zone_attacks[1 - color_idx] += 2
                        
        for i in (0, 1):
            if king_zone_attacks[i] > 0:
                mg_penalty = (king_zone_attacks[i] * 50 * phase) >> 8
                scores_mg[i] -= mg_penalty

        mg_score = scores_mg[0] - scores_mg[1]
        eg_score = scores_eg[0] - scores_eg[1]
        final_score = (mg_score * phase + eg_score * inv_phase) >> 8

        # --- Mating Material Verification ---
        can_force_mate = [True, True]
        for i, pcz in enumerate([pcz_w, pcz_b]):
            if pcz[0] == 0 and pcz[4] == 0 and pcz[3] == 0 and (pcz[1] + pcz[2]) < 2:
                can_force_mate[i] = False

        if final_score > 0 and not can_force_mate[0]: final_score //= 8
        elif final_score < 0 and not can_force_mate[1]: final_score //= 8

        return final_score if turn_to_move == 'white' else -final_score

    def ponder_indefinitely(self):
        while not self.cancellation_event.is_set():
            time.sleep(0.1)