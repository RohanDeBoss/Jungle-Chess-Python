# AI.py (Challenger Framework v2)
# Note to Challenger: Please label your version number when returning the code.

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
# You may add new methods, classes, and variables to this file as needed.
# ==============================================================================

class ChessBot:
    """
    The main AI class.
    """
    # REQUIRED CLASS ATTRIBUTES - DO NOT REMOVE OR RENAME. 
    # The UI will crash on boot if these are missing.
    search_depth = 4  
    MATE_SCORE = 1000000

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
                # Note: This is a highly simplified time allocation. 
                # Feel free to replace this with a more sophisticated algorithm.
                allocated_time = (self.time_left / 30.0) + (self.increment * 0.8) 
                self.stop_time = search_start_time + allocated_time
                target_depth = 100 
            else:
                self.stop_time = None
                target_depth = self.search_depth

            best_move_overall = root_moves[0]
            root_hash = board_hash(self.board, self.color)

            for current_depth in range(1, target_depth + 1):
                iter_start_time = time.time()
                self.nodes_searched = 0
                
                try:
                    score, best_move_this_iter = self._search_root(current_depth, root_moves, root_hash)
                except SearchCancelledException:
                    break

                if self.stop_time and time.time() > self.stop_time:
                    break
                
                best_move_overall = best_move_this_iter
                
                iter_duration = time.time() - iter_start_time
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

    def _search_root(self, depth, root_moves, root_hash):
        best_score = -float('inf')
        best_move = root_moves[0]
        alpha = -float('inf')
        beta = float('inf')

        # Note: Move ordering is absent here. Given the AoE mechanics, consider 
        # how you might prioritize high-impact tactical moves.

        for move in root_moves:
            if self.cancellation_event.is_set() or (self.stop_time and time.time() > self.stop_time):
                raise SearchCancelledException()

            record = self.board.make_move_track(move[0], move[1])
            child_hash = incremental_hash(root_hash, record)

            score = -self.negamax(depth - 1, -beta, -alpha, self.opponent_color, 1, child_hash)
            
            self.board.unmake_move(record)

            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, best_score)

        return best_score, best_move

    def negamax(self, depth, alpha, beta, turn, ply, current_hash):
        self.nodes_searched += 1
        
        if (self.nodes_searched & 1023) == 0:
            if self.cancellation_event.is_set() or (self.stop_time and time.time() > self.stop_time):
                raise SearchCancelledException()

        # --- TABLEBASE PROBE ---
        # The UI expects this logic to remain to gracefully handle exact endgame scores.
        if len(self.board.white_pieces) + len(self.board.black_pieces) <= 5:
            tb_score_absolute = self.tb_manager.probe(self.board, turn)
            if tb_score_absolute is not None:
                tb_score = tb_score_absolute if turn == 'white' else -tb_score_absolute
                if tb_score >  self.MATE_SCORE - 1000: return tb_score - ply
                elif tb_score < -self.MATE_SCORE + 1000: return tb_score + ply
                return tb_score

        if depth <= 0:
            # Note: A quiescence search is highly recommended here, as AoE captures 
            # and knight evaporations create immense horizon effects.
            return self.evaluate_board(turn)

        # Note: Pruning techniques (NMP, LMR) and a Transposition Table are strongly
        # suggested to reach competitive depths.

        moves = get_all_pseudo_legal_moves(self.board, turn)
        opponent = 'black' if turn == 'white' else 'white'
        legal_moves_count = 0

        for move in moves:
            record = self.board.make_move_track(move[0], move[1])
            
            if is_in_check(self.board, turn):
                self.board.unmake_move(record)
                continue
                
            legal_moves_count += 1
            child_hash = incremental_hash(current_hash, record)

            score = -self.negamax(depth - 1, -beta, -alpha, opponent, ply + 1, child_hash)
            
            self.board.unmake_move(record)

            if score >= beta:
                return beta 
            alpha = max(alpha, score)

        if legal_moves_count == 0:
            if is_in_check(self.board, turn) or not has_legal_moves(self.board, turn):
                return -self.MATE_SCORE + ply 

        return alpha

    def evaluate_board(self, turn_to_move):
        """
        Note: The current evaluation is a naive material counter. In Jungle Chess, 
        the relative value of pieces changes drastically based on proximity, 
        structural alignment, and AoE potential.
        """
        if is_insufficient_material(self.board):
            return 0

        score = 0
        for p in self.board.white_pieces:
            if type(p) == Queen: score += 900
            elif type(p) == Rook: score += 500
            elif type(p) == Bishop: score += 330
            elif type(p) == Knight: score += 320
            elif type(p) == Pawn: score += 100
        for p in self.board.black_pieces:
            if type(p) == Queen: score -= 900
            elif type(p) == Rook: score -= 500
            elif type(p) == Bishop: score -= 330
            elif type(p) == Knight: score -= 320
            elif type(p) == Pawn: score -= 100

        return score if turn_to_move == 'white' else -score

    def ponder_indefinitely(self):
        """Called when UI depth slider is set to 99."""
        while not self.cancellation_event.is_set():
            time.sleep(0.1)