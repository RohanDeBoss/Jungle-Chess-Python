# AI.py (Gemini v5)
import time
import random
import os
import json
import glob
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
# JUNGLE CHESS EVALUATION TABLES & CUSTOM STRUCTURES
# ==============================================================================

# Centralize Knights (Evaporation covers maximum squares in the middle)
PST_KNIGHT = [
    [-40, -30, -20, -20, -20, -20, -30, -40],
    [-30,  -5,  10,  15,  15,  10,  -5, -30],
    [-20,  10,  30,  40,  40,  30,  10, -20],
    [-20,  15,  40,  55,  55,  40,  15, -20],
    [-20,  15,  40,  55,  55,  40,  15, -20],
    [-20,  10,  30,  40,  40,  30,  10, -20],
    [-30,  -5,  10,  15,  15,  10,  -5, -30],
    [-40, -30, -20, -20, -20, -20, -30, -40]
]

PST_CENTER = [
    [-20, -10, -10, -10, -10, -10, -10, -20],
    [-10,   0,   0,   0,   0,   0,   0, -10],
    [-10,   0,  10,  15,  15,  10,   0, -10],
    [-10,   0,  15,  20,  20,  15,   0, -10],
    [-10,   0,  15,  20,  20,  15,   0, -10],
    [-10,   0,  10,  15,  15,  10,   0, -10],
    [-10,   0,   0,   0,   0,   0,   0, -10],
    [-20, -10, -10, -10, -10, -10, -10, -20]
]

PST_PAWN_WHITE = [
    [  0,   0,   0,   0,   0,   0,   0,   0],
    [150, 150, 150, 150, 150, 150, 150, 150],
    [ 50,  50,  60,  70,  70,  60,  50,  50],
    [ 30,  30,  40,  50,  50,  40,  30,  30],
    [ 15,  15,  20,  30,  30,  20,  15,  15],
    [  5,   5,  10,  15,  15,  10,   5,   5],
    [  0,   0,   0, -15, -15,   0,   0,   0],
    [  0,   0,   0,   0,   0,   0,   0,   0]
]
PST_PAWN_BLACK = PST_PAWN_WHITE[::-1]

TT_EXACT = 0
TT_LOWERBOUND = 1
TT_UPPERBOUND = 2

class TranspositionTable:
    def __init__(self, size=1048576):
        self.size = size
        self.keys = [None] * size
        self.entries = [None] * size
        
    def store(self, key, depth, score, flag, move):
        idx = key & (self.size - 1)
        self.keys[idx] = key
        self.entries[idx] = (depth, score, flag, move)
        
    def lookup(self, key):
        idx = key & (self.size - 1)
        if self.keys[idx] == key:
            return self.entries[idx]
        return None


# ==============================================================================
# CHESS BOT CLASS 
# ==============================================================================

class ChessBot:
    """
    Jungle Chess Engine (Gemini v5)
    Features AoE-Aware Move Ordering. Avoids double-penalizing explosive pieces,
    maximizing search depth and tactical acuity.
    """
    # REQUIRED CLASS ATTRIBUTES
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
        self.time_left = time_left
        self.increment = increment
        self.soft_time_limit = None
        self.stop_time = None
        self.nodes_searched = 0

        # --- DATABASE INTEGRATIONS ---
        self.use_opening_book = use_opening_book
        self.tb_manager = TablebaseManager()
        if not use_tablebase:
            self.tb_manager.probe = lambda b, t: None

        # --- ENGINE COMPONENTS ---
        # Matched exact OP Bot baseline
        self.PIECE_VALUES = [100, 900, 600, 600, 1300, 0] 
        self.tt = TranspositionTable(1048576)
        self.history_table = {}
        self.killer_moves = [[None, None] for _ in range(128)]

    # --- UI COMMUNICATION HELPERS ---
    def _report_log(self, message):       self.comm_queue.put(('log', message))
    def _report_eval(self, score, depth): self.comm_queue.put(('eval', score if self.color == 'white' else -score, depth))
    def _report_move(self, move):         self.comm_queue.put(('move', move))
    
    def _format_move(self, board_before, move):
        if not move: return "None"
        child = board_before.clone()
        child.make_move(move[0], move[1])
        return format_move_san(board_before, child, move)

    def check_time(self):
        """Immediately interrupt thread when out of time or cancelled by UI."""
        if self.cancellation_event.is_set():
            raise SearchCancelledException()
        if self.stop_time and time.time() > self.stop_time:
            raise SearchCancelledException()

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
                base_time = (self.time_left / 20.0) + (self.increment * 0.8) 
                self.soft_time_limit = search_start_time + base_time
                self.stop_time = search_start_time + min(base_time * 2.5, max(0.05, self.time_left - 0.2))
                target_depth = 100 
            else:
                self.soft_time_limit = None
                self.stop_time = None
                target_depth = self.search_depth

            best_move_overall = root_moves[0]
            root_hash = board_hash(self.board, self.color)

            for current_depth in range(1, target_depth + 1):
                iter_start_time = time.time()
                self.nodes_searched = 0
                
                try:
                    score, best_move_this_iter = self._search_root(current_depth, root_moves, root_hash, best_move_overall)
                except SearchCancelledException:
                    break
                
                best_move_overall = best_move_this_iter
                
                iter_duration = time.time() - iter_start_time
                knps = (self.nodes_searched / iter_duration / 1000) if iter_duration > 0 else 0
                eval_ui = score if self.color == 'white' else -score
                pv_san = self._format_move(self.board, best_move_overall)
                
                self._report_log(f"  > {self.bot_name} (D{current_depth}): {pv_san}, Eval={eval_ui/100:+.2f}, Nodes={self.nodes_searched}, KNPS={knps:.1f}, Time={iter_duration:.2f}s")
                self._report_eval(score, current_depth)
                self.comm_queue.put(('pv', eval_ui, current_depth, [pv_san], [best_move_overall]))

                if score > self.MATE_SCORE - 1000 or score < -self.MATE_SCORE + 1000: 
                    break 

                if self.soft_time_limit and time.time() > self.soft_time_limit:
                    break

            self._report_move(best_move_overall)

        except Exception as e:
            self._report_log(f"CRASH: {str(e)}")
            self._report_move(root_moves[0] if root_moves else None)

    def _score_moves(self, moves, tt_move, ply):
        """O(1) AoE-Aware Move Ordering."""
        scored = []
        
        for move in moves:
            if move == tt_move:
                scored.append((10000000, move))
                continue
                
            mp = self.board.grid[move[0][0]][move[0][1]]
            tp = self.board.grid[move[1][0]][move[1][1]]
            
            swing, is_tactic = fast_approximate_material_swing(self.board, move, mp, tp, self.PIECE_VALUES)
            if is_tactic:
                # If Queen captures, she explodes. Her -1300 death is ALREADY in the `swing` variable.
                # Do not double-penalize by subtracting her value again!
                attacker_penalty = self.PIECE_VALUES[mp.z_idx] if mp.z_idx != 4 else 0
                
                if swing < 0:
                    # Negative swing means sacrificing the piece (like a Queen) for garbage.
                    # Ranks it ~498,000. It will be searched, but it triggers LMR.
                    score = 500000 + swing 
                else:
                    # Positive swing (Profitable Queen Bomb/Normal Capture).
                    # Ranks ~1,000,000+. Exempt from LMR, highly prioritized.
                    score = 1000000 + (swing * 10) - attacker_penalty
                    
                scored.append((score, move))
            else:
                score = 0
                if ply < 128:
                    if move == self.killer_moves[ply][0]: score = 900000
                    elif move == self.killer_moves[ply][1]: score = 800000
                if score == 0:
                    score = self.history_table.get((mp.z_idx, move[1]), 0)
                scored.append((score, move))
                
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    def _search_root(self, depth, root_moves, root_hash, best_move_overall):
        alpha = -float('inf')
        beta = float('inf')

        scored_moves = self._score_moves(root_moves, best_move_overall, 0)
        best_score = -float('inf')
        best_move = scored_moves[0][1]

        for i, (move_score, move) in enumerate(scored_moves):
            self.check_time()
            record = self.board.make_move_track(move[0], move[1])
            child_hash = incremental_hash(root_hash, record)

            # Principal Variation Search (PVS)
            if i == 0:
                score = -self.negamax(depth - 1, -beta, -alpha, self.opponent_color, 1, child_hash)
            else:
                score = -self.negamax(depth - 1, -alpha - 1, -alpha, self.opponent_color, 1, child_hash)
                if score > alpha: 
                    score = -self.negamax(depth - 1, -beta, -alpha, self.opponent_color, 1, child_hash)
            
            self.board.unmake_move(record)

            if score > best_score:
                best_score = score
                best_move = move
                
            if score > alpha:
                alpha = score

        return best_score, best_move

    def negamax(self, depth, alpha, beta, turn, ply, current_hash):
        self.nodes_searched += 1
        
        if (self.nodes_searched & 1023) == 0:
            self.check_time()

        # --- Tablebase Probe ---
        if len(self.board.white_pieces) + len(self.board.black_pieces) <= 5:
            tb_score_absolute = self.tb_manager.probe(self.board, turn)
            if tb_score_absolute is not None:
                tb_score = tb_score_absolute if turn == 'white' else -tb_score_absolute
                if tb_score >  self.MATE_SCORE - 1000: return tb_score - ply
                elif tb_score < -self.MATE_SCORE + 1000: return tb_score + ply
                return tb_score

        alpha_orig = alpha
        
        # --- Transposition Table Lookup ---
        tt_entry = self.tt.lookup(current_hash)
        tt_move = None
        if tt_entry is not None:
            tt_depth, tt_score, tt_flag, tt_move = tt_entry
            
            # Mate score ply alignment
            if tt_score > self.MATE_SCORE - 1000: tt_score -= ply
            elif tt_score < -self.MATE_SCORE + 1000: tt_score += ply
            
            if tt_depth >= depth:
                if tt_flag == TT_EXACT:
                    return tt_score
                elif tt_flag == TT_LOWERBOUND:
                    alpha = max(alpha, tt_score)
                elif tt_flag == TT_UPPERBOUND:
                    beta = min(beta, tt_score)
                if alpha >= beta:
                    return tt_score

        # --- Quiescence Horizon ---
        if depth <= 0:
            return self.qsearch(alpha, beta, turn, ply, current_hash)

        in_check = is_in_check(self.board, turn)
        moves = get_all_pseudo_legal_moves(self.board, turn)
        scored_moves = self._score_moves(moves, tt_move, ply)

        opponent = 'black' if turn == 'white' else 'white'
        legal_moves_count = 0
        best_move = None
        best_score = -float('inf')

        for move_score, move in scored_moves:
            record = self.board.make_move_track(move[0], move[1])
            
            if is_in_check(self.board, turn):
                self.board.unmake_move(record)
                continue
                
            legal_moves_count += 1
            child_hash = incremental_hash(current_hash, record)

            # --- Smart Late Move Reductions (LMR) ---
            # move_score > 500000 correctly identifies neutral/positive expected material swings.
            # Suicidal explosions (< 500000) will be reduced here.
            is_tactic = move_score > 500000
            reduction = 0
            if depth >= 3 and legal_moves_count > 4 and not in_check and not is_tactic:
                reduction = 1
                if depth >= 5 and legal_moves_count > 10:
                    reduction = 2

            # --- PVS Core ---
            if legal_moves_count == 1:
                score = -self.negamax(depth - 1, -beta, -alpha, opponent, ply + 1, child_hash)
            else:
                if reduction > 0:
                    score = -self.negamax(depth - 1 - reduction, -alpha - 1, -alpha, opponent, ply + 1, child_hash)
                    if score > alpha:
                        score = -self.negamax(depth - 1, -beta, -alpha, opponent, ply + 1, child_hash)
                else:
                    score = -self.negamax(depth - 1, -alpha - 1, -alpha, opponent, ply + 1, child_hash)
                    if alpha < score < beta: 
                        score = -self.negamax(depth - 1, -beta, -alpha, opponent, ply + 1, child_hash)
            
            self.board.unmake_move(record)

            if score > best_score:
                best_score = score
                best_move = move
                
            if score > alpha:
                alpha = score
                
            if alpha >= beta:
                # History / Killer updates
                if not is_tactic and ply < 128:
                    if move != self.killer_moves[ply][0]:
                        self.killer_moves[ply][1] = self.killer_moves[ply][0]
                        self.killer_moves[ply][0] = move
                    mp = self.board.grid[move[0][0]][move[0][1]]
                    if mp:
                        self.history_table[(mp.z_idx, move[1])] = self.history_table.get((mp.z_idx, move[1]), 0) + depth * depth
                break

        if legal_moves_count == 0:
            return -self.MATE_SCORE + ply 

        # --- TT Store ---
        flag = TT_EXACT
        if best_score <= alpha_orig:
            flag = TT_UPPERBOUND
        elif best_score >= beta:
            flag = TT_LOWERBOUND
            
        stored_score = best_score
        if stored_score > self.MATE_SCORE - 1000: stored_score += ply
        elif stored_score < -self.MATE_SCORE + 1000: stored_score -= ply
            
        self.tt.store(current_hash, depth, stored_score, flag, best_move)
        
        return best_score

    def qsearch(self, alpha, beta, turn, ply, current_hash):
        self.nodes_searched += 1
        if (self.nodes_searched & 1023) == 0:
            self.check_time()
            
        in_check = is_in_check(self.board, turn)
        
        if not in_check:
            stand_pat = self.evaluate_board(turn)
            if stand_pat >= beta:
                return beta
            if alpha < stand_pat:
                alpha = stand_pat
            best_score = stand_pat
        else:
            best_score = -float('inf')
                
        moves = get_all_pseudo_legal_moves(self.board, turn)
        tactical_moves = []
        
        for move in moves:
            mp = self.board.grid[move[0][0]][move[0][1]]
            tp = self.board.grid[move[1][0]][move[1][1]]
            
            swing, is_tactic = fast_approximate_material_swing(self.board, move, mp, tp, self.PIECE_VALUES)
            if in_check or is_tactic:
                
                # Futility Pruning for bad sacrifices: 
                # If a Queen suicide results in a net negative material loss, ignore it.
                if not in_check and swing < 0:
                    continue
                    
                attacker_penalty = self.PIECE_VALUES[mp.z_idx] if mp.z_idx != 4 else 0
                score = (swing * 10) - attacker_penalty
                tactical_moves.append((score, move))
                
        tactical_moves.sort(key=lambda x: x[0], reverse=True)
        
        opponent = 'black' if turn == 'white' else 'white'
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
        """
        Fast, integer-only evaluation. Evaluates precise material differential, 
        and securely maps King Tropism logic when material significantly drops.
        """
        if is_insufficient_material(self.board):
            return 0

        score = 0
        
        pcz_w = self.board.piece_counts_z['white']
        pcz_b = self.board.piece_counts_z['black']
        
        # Base Material Balance
        score += (pcz_w[0] - pcz_b[0]) * 100
        score += (pcz_w[1] - pcz_b[1]) * 900
        score += (pcz_w[2] - pcz_b[2]) * 600
        score += (pcz_w[3] - pcz_b[3]) * 600
        score += (pcz_w[4] - pcz_b[4]) * 1300
        
        # Positional Features
        for p in self.board.white_pieces:
            r, c = p.pos
            z = p.z_idx
            if z == 1: score += PST_KNIGHT[r][c]
            elif z == 2 or z == 3 or z == 4: score += PST_CENTER[r][c]
            elif z == 0: score += PST_PAWN_WHITE[r][c]
                
        for p in self.board.black_pieces:
            r, c = p.pos
            z = p.z_idx
            if z == 1: score -= PST_KNIGHT[r][c]
            elif z == 2 or z == 3 or z == 4: score -= PST_CENTER[r][c]
            elif z == 0: score -= PST_PAWN_BLACK[r][c]

        # --- Integer-Only Endgame Mop-Up Logic ---
        wk = self.board.white_king_pos
        bk = self.board.black_king_pos
        
        if wk and bk:
            # Lightweight Game Phase calculation (Total Non-Pawn Material)
            npm = (pcz_w[1] + pcz_b[1])*900 + (pcz_w[2] + pcz_b[2])*600 + (pcz_w[3] + pcz_b[3])*600 + (pcz_w[4] + pcz_b[4])*1300
            
            if npm < 4000:
                # Linearly escalates from 1 to 5 as material drains
                eg_weight = 1 + (4000 - npm) // 1000 
                
                # King Tropism: Drive enemy king out of safe zones
                if score > 0:
                    dist = abs(wk[0] - bk[0]) + abs(wk[1] - bk[1])
                    score += (14 - dist) * 5 * eg_weight
                elif score < 0:
                    dist = abs(wk[0] - bk[0]) + abs(wk[1] - bk[1])
                    score -= (14 - dist) * 5 * eg_weight

        # --- Mating Material Verification ---
        # Checks if the dominant side has the required pieces to force a mate
        if score > 0:
            if pcz_w[0] == 0 and pcz_w[4] == 0 and pcz_w[3] == 0 and (pcz_w[1] + pcz_w[2]) < 2:
                score //= 8
        elif score < 0:
            if pcz_b[0] == 0 and pcz_b[4] == 0 and pcz_b[3] == 0 and (pcz_b[1] + pcz_b[2]) < 2:
                score //= 8

        return score if turn_to_move == 'white' else -score

    def ponder_indefinitely(self):
        while not self.cancellation_event.is_set():
            time.sleep(0.1)