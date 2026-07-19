# AI.py (Challenger Framework example! (label your version with a number!)

import time
import random
from GameLogic import *

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
    """Computes a Zobrist hash of the current board state."""
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
    """Fast incremental hash update for use inside the search tree."""
    h = parent_hash ^ ZOBRIST_TURN
    arr = ZOBRIST_ARRAY
    start, end, mp, removed_pieces, added_pieces = record_tuple
    
    c_idx = 0 if mp.color == 'white' else 1
    p_idx = mp.z_idx
    sr, sc = start; er, ec = end

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

def run_ai_process(board, color, position_counts, comm_queue, cancellation_event,
                   bot_class, bot_name, search_depth, ply_count, game_mode,
                   time_left=None, increment=None, use_opening_book=True, use_tablebase=True):
    """Entry point for the UI's persistent worker process."""
    import inspect
    accepted_params = set(inspect.signature(bot_class.__init__).parameters)
    kwargs = {
        'time_left': time_left, 'increment': increment,
        'use_opening_book': use_opening_book, 'use_tablebase': use_tablebase
    }
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted_params}

    bot = bot_class(board, color, position_counts, comm_queue, cancellation_event,
                    bot_name, ply_count, game_mode, **filtered_kwargs)

    bot.search_depth = search_depth
    if search_depth == 99:
        bot.ponder_indefinitely()
    else:
        bot.make_move()

class SearchCancelledException(Exception): pass

# ==============================================================================
# CHESS BOT CLASS (YOUR CHALLENGE STARTS HERE)
# ==============================================================================

class ChessBot:
    """
    The main AI class. You must implement advanced search and evaluation here.
    Do not change the __init__ signature.
    """
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
        
        # --- TIME MANAGEMENT ---
        # 10+0.1 Time Control variables. 
        self.time_left = time_left
        self.increment = increment
        self.stop_time = None
        
        self.search_depth = self.__class__.search_depth
        self.nodes_searched = 0

    # --- UI COMMUNICATION HELPERS ---
    def _report_log(self, message):   
        self.comm_queue.put(('log', message))
    def _report_eval(self, score, depth): 
        self.comm_queue.put(('eval', score if self.color == 'white' else -score, depth))
    def _report_move(self, move):     
        self.comm_queue.put(('move', move))
    def _format_move(self, board_before, move):
        if not move: return "None"
        child = board_before.clone()
        child.make_move(move[0], move[1])
        return format_move_san(board_before, child, move)

    # --- MAIN ENTRY POINT ---
    def make_move(self):
        try:
            root_moves = get_all_legal_moves(self.board, self.color)
            if not root_moves:
                self._report_move(None)
                return

            # 1. Time Management Calculation (Crucial for 10+0.1)
            search_start_time = time.time()
            if self.time_left is not None and self.increment is not None:
                # Basic fraction of remaining time.
                allocated_time = (self.time_left / 30.0) + (self.increment * 0.8)
                allocated_time = max(0.05, min(allocated_time, self.time_left - 0.2))
                self.stop_time = search_start_time + allocated_time
                target_depth = 100 # Go as deep as time allows
            else:
                self.stop_time = None
                target_depth = self.search_depth

            best_move_overall = root_moves[0]
            root_hash = board_hash(self.board, self.color)

            # 2. Iterative Deepening Loop
            for current_depth in range(1, target_depth + 1):
                iter_start_time = time.time()
                self.nodes_searched = 0
                
                try:
                    score, best_move_this_iter = self._search_root(current_depth, root_moves, root_hash)
                except SearchCancelledException:
                    break

                # If time ran out during the search, discard the incomplete iteration
                if self.stop_time and time.time() > self.stop_time:
                    break
                
                best_move_overall = best_move_this_iter
                
                # --- Send UI Updates ---
                iter_duration = time.time() - iter_start_time
                knps = (self.nodes_searched / iter_duration / 1000) if iter_duration > 0 else 0
                eval_ui = score if self.color == 'white' else -score
                pv_san = self._format_move(self.board, best_move_overall)
                
                self._report_log(f"  > {self.bot_name} (D{current_depth}): {pv_san}, Eval={eval_ui/100:+.2f}, NodesTotal={self.nodes_searched}, KNPS={knps:.1f}, Time={iter_duration:.2f}s")
                self._report_eval(score, current_depth)
                self.comm_queue.put(('pv', eval_ui, current_depth, [pv_san], [best_move_overall]))

                if score > self.MATE_SCORE - 1000: 
                    break # Found forced mate

            self._report_move(best_move_overall)

        except Exception as e:
            # Fallback to prevent UI crash
            self._report_log(f"CRASH: {str(e)}")
            self._report_move(root_moves[0] if root_moves else None)

    def _search_root(self, depth, root_moves, root_hash):
        """Root Alpha-Beta call. Returns (best_score, best_move)."""
        best_score = -float('inf')
        best_move = root_moves[0]
        alpha = -float('inf')
        beta = float('inf')

        # No move ordering here *hint*

        for move in root_moves:
            if self.cancellation_event.is_set() or (self.stop_time and time.time() > self.stop_time):
                raise SearchCancelledException()

            # Fast make/unmake
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
        """Recursive Alpha-Beta Search."""
        self.nodes_searched += 1
        
        # Check time periodically (every 1024 nodes to save CPU cycles)
        if (self.nodes_searched & 1023) == 0:
            if self.cancellation_event.is_set() or (self.stop_time and time.time() > self.stop_time):
                raise SearchCancelledException()

        if depth <= 0:
            # CHALLENGE: Implement Quiescence Search (QSearch) here to stop horizon effect blunders!
            return self.evaluate_board(turn)

        # Get pseudo-legal moves and filter them inside the loop
        moves = get_all_pseudo_legal_moves(self.board, turn)
        opponent = 'black' if turn == 'white' else 'white'
        
        legal_moves_count = 0

        for move in moves:
            record = self.board.make_move_track(move[0], move[1])
            
            # Filter illegal moves (leaving own king in check)
            if is_in_check(self.board, turn):
                self.board.unmake_move(record)
                continue
                
            legal_moves_count += 1
            child_hash = incremental_hash(current_hash, record)

            score = -self.negamax(depth - 1, -beta, -alpha, opponent, ply + 1, child_hash)
            
            self.board.unmake_move(record)

            if score >= beta:
                return beta # Beta cutoff
            alpha = max(alpha, score)

        if legal_moves_count == 0:
            if is_in_check(self.board, turn) or not has_legal_moves(self.board, turn):
                return -self.MATE_SCORE + ply # Checkmate or No-Legal-Moves loss

        return alpha

    def evaluate_board(self, turn_to_move):
        """
        CHALLENGE: Implement a robust evaluation function here! 
        """
        if is_insufficient_material(self.board):
            return 0

        score = 0
        # Basic material counting (EXTREMELY WEAK for Jungle Chess!)
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