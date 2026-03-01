# AI.py (v93 - Trying to fix SEE move ordering for variant)
import time
import random
from collections import namedtuple
from GameLogic import *
from TablebaseManager import TablebaseManager

# --- EVALUATION CONSTANTS ---

MG_PIECE_VALUES = {
    Pawn: 100, Knight: 950, Bishop: 650, Rook: 550, Queen: 850, King: 20000
}

# The Tablebase-Proven Meta
EG_PIECE_VALUES = {
    Pawn: 130,    # Dangerous, easily promotes or kills sideways
    Knight: 850,  # Just as good as a queen in endgames, but less pieces to fork.
    Bishop: 500,  # Lower value in endgames as it can't change colour or checkmate by itself
    Rook: 600,    # Kept high, good at eating pawns.
    Queen: 850,
    King: 20000
}

# Move-ordering tactical swing should track the same base scale as MG values.
ORDERING_VALUES = MG_PIECE_VALUES

INITIAL_PHASE_MATERIAL = (MG_PIECE_VALUES[Rook] * 4 + MG_PIECE_VALUES[Knight] * 4 +
                          MG_PIECE_VALUES[Bishop] * 4 + MG_PIECE_VALUES[Queen] * 2)

# --- ZOBRIST HASHING SETUP ---
ZOBRIST_TABLE = None
def initialize_zobrist_table():
    global ZOBRIST_TABLE
    if ZOBRIST_TABLE is not None: return
    random.seed(42)
    table = {}
    piece_types = [Pawn, Knight, Bishop, Rook, Queen, King, None]
    colors = ['white', 'black', None]
    for r in range(ROWS):
        for c in range(COLS):
            for piece_type in piece_types:
                for piece_color in colors:
                    key = (r, c, piece_type, piece_color)
                    table[key] = random.getrandbits(64)
    table['turn'] = random.getrandbits(64)
    ZOBRIST_TABLE = table

initialize_zobrist_table()

def board_hash(board, turn):
    h = 0
    zt = ZOBRIST_TABLE 
    for piece in board.white_pieces:
        if piece.pos:
            h ^= zt.get((piece.pos[0], piece.pos[1], type(piece), piece.color), 0)
    for piece in board.black_pieces:
        if piece.pos:
            h ^= zt.get((piece.pos[0], piece.pos[1], type(piece), piece.color), 0)
    if turn == 'black': h ^= zt['turn']
    return h

# --- SEARCH STRUCTURES ---
TTEntry = namedtuple('TTEntry', ['score', 'depth', 'flag', 'best_move'])
TT_FLAG_EXACT, TT_FLAG_LOWERBOUND, TT_FLAG_UPPERBOUND = 0, 1, 2

class SearchCancelledException(Exception): pass

class ChessBot:
    search_depth = 6
    MATE_SCORE = 1000000
    DRAW_SCORE = 0
    
    MAX_Q_SEARCH_DEPTH = 8
    LMR_DEPTH_THRESHOLD = 3
    LMR_MOVE_COUNT_THRESHOLD = 4
    LMR_REDUCTION = 1
    NMP_MIN_DEPTH = 3
    NMP_BASE_REDUCTION = 2
    NMP_DEPTH_DIVISOR = 6
    USE_NULL_MOVE_PRUNING = True
    Q_SEARCH_SAFETY_MARGIN = 600

    BONUS_PV_MOVE = 10_000_000
    BONUS_CAPTURE = 8_000_000
    BONUS_KILLER_1 = 4_000_000
    BONUS_KILLER_2 = 3_000_000
    BAD_TACTIC_PENALTY = -2_000_000
    # This bonus is removed in favor of the history heuristic: BONUS_Q_TACTIC = 3_500_000 
    OPENING_TOTAL_PIECE_THRESHOLD = 23
    OPENING_BONUS_MAX_PLY = 1
    OPENING_KNIGHT_DEVELOP_BONUS = 100
    OPENING_KNIGHT_CENTER_WEIGHT = 22
    OPENING_PAWN_CENTER_WEIGHT = 10
    OPENING_CENTER_PAWN_BONUS = 28
    OPENING_CENTRAL_FILES = (COLS // 2 - 1, COLS // 2)
    ASP_WINDOW_INIT = 150 # Because Jungle Chess is highly lethal, a window of 150 is still quite tight.
    ASP_MAX_RETRIES = 3 # Don't throw good time after bad; fallback to full search sooner
    
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
        self.killer_moves = [[None, None] for _ in range(50)]
        self.history_heuristic_table = [[[0 for _ in range(ROWS*COLS)] for _ in range(ROWS*COLS)] for _ in range(2)]

    def _report_log(self, message): self.comm_queue.put(('log', message))
    def _report_eval(self, score, depth): self.comm_queue.put(('eval', score if self.color == 'white' else -score, depth))
    def _report_move(self, move): self.comm_queue.put(('move', move))
    def _format_move(self, move): return format_move(move)

    def _is_tactical_move(self, board, move, moving_piece=None, target_piece=None):
        if moving_piece is None:
            moving_piece = board.grid[move[0][0]][move[0][1]]
        if moving_piece is None:
            return False

        if target_piece is None:
            target_piece = board.grid[move[1][0]][move[1][1]]
        if target_piece is not None:
            return True

        if isinstance(moving_piece, Pawn) and (move[1][0] == 0 or move[1][0] == ROWS - 1):
            return True
        if isinstance(moving_piece, Rook) and is_rook_piercing_capture(board, move):
            return True
        if isinstance(moving_piece, Knight) and is_quiet_knight_evaporation(board, move):
            return True
        if is_passive_knight_zone_evaporation(board, move):
            return True
        return False

    def _ordering_tactical_swing(self, board, move, moving_piece, target_piece):
            # Defers to the centralized, high-speed approximation in GameLogic.py
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
        """
        Return current position TB eval from the bot's perspective.
        This is the value we should display for the current root position.
        """
        root_abs = self.tb_manager.probe(self.board, self.color)
        if root_abs is None:
            return None
        self.tb_hits += 1
        return root_abs if self.color == 'white' else -root_abs

    def _get_best_tablebase_move_with_eval(self):
        """Finds the absolute best move when only 3 pieces remain."""
        best_move = None
        best_score = -float('inf')
        
        for move in get_all_legal_moves(self.board, self.color):
            sim = self.board.clone()
            sim.make_move(move[0], move[1])
            
            # Instant win check (Evaporation/Explosion)
            if not sim.find_king_pos(self.opponent_color):
                return move, self.MATE_SCORE - 1
            # Also treat immediate checkmate as mate-in-1 for display purposes.
            if is_in_check(sim, self.opponent_color) and not has_legal_moves(sim, self.opponent_color):
                return move, self.MATE_SCORE - 1
                 
            score_abs = self.tb_manager.probe(sim, self.opponent_color)
            
            if score_abs is None:
                # Capture resulted in King vs King
                score = 0
            else:
                self.tb_hits += 1
                score = score_abs if self.color == 'white' else -score_abs
                # Adjust for the 1-ply move we just made
                if score > self.MATE_SCORE - 1000: score -= 1
                elif score < -self.MATE_SCORE + 1000: score += 1
            
            # Tie breaker: add slight randomness to draws to avoid infinite shuffling
            if score > best_score or (score == best_score and score == 0 and random.random() > 0.5):
                best_score = score
                best_move = move
                
        return best_move, best_score

    def _run_depth_iteration(self, depth, root_moves, root_hash, pv_move, prev_iter_score=None, alpha_floor=None):
        iter_nodes = 0
        iter_tb_hits = 0
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
                if self.cancellation_event.is_set():
                    raise SearchCancelledException()

                # --- THE FIX IS HERE ---
                if best_score <= alpha_bound:       # Changed from < to <=
                    alpha_bound -= window
                    window *= 2
                    retries += 1
                elif best_score >= beta_bound:      # Changed from > to >=
                    beta_bound += window
                    window *= 2
                    retries += 1
                else:
                    break
                # -----------------------

                if retries >= self.ASP_MAX_RETRIES:
                    best_score, best_move = self._search_at_depth(
                        depth, root_moves, root_hash, pv_move, alpha_floor=alpha_floor
                    )
                    iter_nodes += self.nodes_searched
                    iter_tb_hits += self.tb_hits
                    break
        else:
            best_score, best_move = self._search_at_depth(
                depth, root_moves, root_hash, pv_move, alpha_floor=alpha_floor
            )
            iter_nodes = self.nodes_searched
            iter_tb_hits = self.tb_hits

        self.nodes_searched = iter_nodes
        self.tb_hits = iter_tb_hits
        return best_score, best_move

    def _report_root_tb_solution(self, tb_move, tb_eval, perfect_play=False, emit_move=False):
        if not tb_move:
            return False

        root_tb_eval = self._get_root_tb_eval_relative()
        display_eval = root_tb_eval if root_tb_eval is not None else tb_eval
        if tb_eval > self.MATE_SCORE - 1000:
            display_eval = tb_eval

        eval_for_ui = display_eval if self.color == 'white' else -display_eval
        suffix = " (Perfect Play)" if perfect_play else ""
        self._report_log(
            f"  > {self.bot_name} (TB): {self._format_move(tb_move)}, Eval={eval_for_ui/100:+.2f}, TBhits={self.tb_hits}{suffix}"
        )
        self._report_eval(display_eval, "TB")
        if emit_move:
            self._report_move(tb_move)
        return True

    def _age_history_table(self):
        # Decay all scores in the history table to prioritize recent information
        for color_idx in range(2):
            for from_sq in range(ROWS*COLS):
                for to_sq in range(ROWS*COLS):
                    self.history_heuristic_table[color_idx][from_sq][to_sq] //= 2

    def make_move(self):
        try:
            self._age_history_table()

            if len(self.board.white_pieces) + len(self.board.black_pieces) == 3:
                tb_move, tb_eval = self._get_best_tablebase_move_with_eval()
                if self._report_root_tb_solution(tb_move, tb_eval, emit_move=True):
                    return

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
                    move_str = self._format_move(best_move_this_iter)
                    depth_label = "TB" if not self.used_heuristic_eval else current_depth
                    
                    log_msg = f"  > {self.bot_name} (D{depth_label}): {move_str}, Eval={eval_for_ui/100:+.2f}, NodesTotal={total_nodes}, KNPS={knps:.1f}, TBhits={self.tb_hits}, Time={iter_duration:.2f}s"
                    
                    self._report_log(log_msg)
                    self._report_eval(best_score_this_iter, depth_label)

                    if best_score_this_iter > self.MATE_SCORE - 2000:
                        break
                else:
                    raise SearchCancelledException()
            
            self._report_move(best_move_overall)
        except SearchCancelledException:
            self._report_move(None)

    def ponder_indefinitely(self):
        try:
            self._age_history_table()
            
            if is_insufficient_material(self.board): return
            
            # Instantly solve 3-piece endgames in Analysis Mode
            if len(self.board.white_pieces) + len(self.board.black_pieces) == 3:
                tb_move, tb_eval = self._get_best_tablebase_move_with_eval()
                if self._report_root_tb_solution(tb_move, tb_eval, perfect_play=True):
                    # Sleep to prevent burning CPU since the position is perfectly solved
                    while not self.cancellation_event.is_set():
                        time.sleep(0.1)
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
                    current_depth,
                    root_moves,
                    root_hash,
                    best_move_overall,
                    prev_iter_score=prev_iter_score,
                    alpha_floor=tb_alpha_floor
                )
                
                if not self.cancellation_event.is_set():
                    best_move_overall = best_move_this_iter
                    prev_iter_score = best_score_this_iter
                    total_nodes += self.nodes_searched
                    iter_duration = time.time() - iter_start_time
                    knps = (self.nodes_searched / iter_duration / 1000) if iter_duration > 0 else 0
                    eval_for_ui = best_score_this_iter if self.color == 'white' else -best_score_this_iter
                    depth_label = "TB" if not self.used_heuristic_eval else current_depth
                    self._report_log(f"  > {self.bot_name} (D{depth_label}): {self._format_move(best_move_this_iter)}, Eval={eval_for_ui/100:+.2f}, NodesTotal={total_nodes}, KNPS={knps:.1f}, TBhits={self.tb_hits}, Time={iter_duration:.2f}s")
                    self._report_eval(best_score_this_iter, depth_label)

                    # If we already found a TB-level winning line, deeper iterations should
                    # only try to beat it (shorter win), not re-search longer/equal lines.
                    if best_score_this_iter > self.MATE_SCORE - 1000:
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
        
        ordered_root_moves = self.order_moves(self.board, root_moves, 0, pv_move)
        
        all_moves_draw = True
        for move in ordered_root_moves:
            if self.cancellation_event.is_set(): raise SearchCancelledException()
            
            child_board = self.board.clone()
            child_board.make_move(move[0], move[1])

            search_path = {root_hash}
            child_hash = board_hash(child_board, self.opponent_color)
            self.position_counts[child_hash] = self.position_counts.get(child_hash, 0) + 1

            if alpha_floor is not None:
                # Null-window root test: only continue if this move can beat current TB floor.
                probe_score = -self.negamax(
                    child_board, depth - 1, -(alpha_floor + 1), -alpha_floor,
                    self.opponent_color, 1, search_path
                )
                if probe_score <= alpha_floor:
                    self.position_counts[child_hash] -= 1
                    continue
                score = -self.negamax(
                    child_board, depth - 1, -beta, -alpha,
                    self.opponent_color, 1, search_path
                )
            else:
                score = -self.negamax(child_board, depth - 1, -beta, -alpha, self.opponent_color, 1, search_path)

            self.position_counts[child_hash] -= 1
            if score != self.DRAW_SCORE: all_moves_draw = False

            if score > best_score_this_iter:
                best_score_this_iter = score
                best_move_this_iter = move
            alpha = max(alpha, best_score_this_iter)
        
        if alpha_floor is None and all_moves_draw:
            best_score_this_iter = self.DRAW_SCORE
        return best_score_this_iter, best_move_this_iter

    def negamax(self, board, depth, alpha, beta, turn, ply, search_path):
        self.nodes_searched += 1
        if self.cancellation_event.is_set(): raise SearchCancelledException()

        if len(board.white_pieces) + len(board.black_pieces) == 3:
            tb_score_absolute = self.tb_manager.probe(board, turn)
            if tb_score_absolute is not None:
                self.tb_hits += 1
                tb_score = tb_score_absolute if turn == 'white' else -tb_score_absolute
                if tb_score > self.MATE_SCORE - 1000: return tb_score - ply
                elif tb_score < -self.MATE_SCORE + 1000: return tb_score + ply
                return tb_score

        # --- OPTIMIZATION: MATE DISTANCE PRUNING ---
        mate_value = self.MATE_SCORE - ply
        if beta > mate_value:
            beta = mate_value
            if alpha >= mate_value:
                return mate_value
        
        mated_value = -self.MATE_SCORE + ply
        if alpha < mated_value:
            alpha = mated_value
            if beta <= mated_value: return mated_value

        hash_val = board_hash(board, turn)
        
        if ply > 0:
            if hash_val in search_path: return self.DRAW_SCORE
            if self.position_counts.get(hash_val, 0) >= 3: return self.DRAW_SCORE
        
        if is_insufficient_material(board): return self.DRAW_SCORE
        if self.ply_count + ply >= self.max_moves: return self.DRAW_SCORE

        original_alpha = alpha
        tt_entry = self.tt.get(hash_val)
        if ply > 0 and tt_entry and tt_entry.depth >= depth:
            tt_score = tt_entry.score
            if tt_score > self.MATE_SCORE - 1000: tt_score -= ply
            elif tt_score < -self.MATE_SCORE + 1000: tt_score += ply

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
            search_path.add(hash_val)
            path_added = True

        try:
            if (self.USE_NULL_MOVE_PRUNING and depth >= self.NMP_MIN_DEPTH and ply > 0 and not is_in_check_flag and
                beta < self.MATE_SCORE - 200 and
                any(not isinstance(p, (Pawn, King)) for p in (board.white_pieces if turn == 'white' else board.black_pieces))):
                self.used_heuristic_eval = True
                static_eval = self.evaluate_board(board, turn)
                if static_eval >= beta:
                    nmp_reduction = self.NMP_BASE_REDUCTION + (depth // self.NMP_DEPTH_DIVISOR)
                    score = -self.negamax(board, depth - 1 - nmp_reduction, -beta, -beta + 1, opponent_turn, ply + 1, search_path)
                    if score >= beta:
                        store_score = beta
                        if store_score > self.MATE_SCORE - 1000: store_score += ply
                        elif store_score < -self.MATE_SCORE + 1000: store_score -= ply
                        self.tt[hash_val] = TTEntry(store_score, depth, TT_FLAG_LOWERBOUND, None)
                        return beta

            pseudo_moves = get_all_pseudo_legal_moves(board, turn)
            hash_move = tt_entry.best_move if tt_entry else None
            ordered_entries = self.order_moves(board, pseudo_moves, ply, hash_move, return_meta=True)

            best_move_for_node = None
            legal_moves_count = 0

            for move, meta in ordered_entries:
                is_tactical, moving_piece = meta

                child_board = board.clone()
                child_board.make_move(move[0], move[1])

                if is_in_check(child_board, turn): continue

                legal_moves_count += 1

                reduction = 0
                if (depth >= self.LMR_DEPTH_THRESHOLD and legal_moves_count > self.LMR_MOVE_COUNT_THRESHOLD and
                    not is_in_check_flag and not is_tactical):
                    reduction = self.LMR_REDUCTION

                search_depth = depth - 1 - reduction

                if legal_moves_count == 1 or alpha == -float('inf'):
                    score = -self.negamax(child_board, search_depth, -beta, -alpha, opponent_turn, ply + 1, search_path)
                    if reduction > 0 and score > alpha:
                        score = -self.negamax(child_board, depth - 1, -beta, -alpha, opponent_turn, ply + 1, search_path)
                else:
                    score = -self.negamax(child_board, search_depth, -(alpha + 1), -alpha, opponent_turn, ply + 1, search_path)
                    if reduction > 0 and score > alpha:
                        score = -self.negamax(child_board, depth - 1, -(alpha + 1), -alpha, opponent_turn, ply + 1, search_path)
                    if score > alpha and score < beta:
                        score = -self.negamax(child_board, depth - 1, -beta, -alpha, opponent_turn, ply + 1, search_path)

                if score > alpha:
                    alpha, best_move_for_node = score, move
                if alpha >= beta:
                    if not is_tactical:
                        if ply < len(self.killer_moves) and self.killer_moves[ply][0] != move:
                            self.killer_moves[ply][1], self.killer_moves[ply][0] = self.killer_moves[ply][0], move
                        if moving_piece:
                            color_index = 0 if moving_piece.color == 'white' else 1
                            from_sq, to_sq = move[0][0]*COLS+move[0][1], move[1][0]*COLS+move[1][1]
                            self.history_heuristic_table[color_index][from_sq][to_sq] += depth * depth

                    store_score = beta
                    if store_score > self.MATE_SCORE - 1000: store_score += ply
                    elif store_score < -self.MATE_SCORE + 1000: store_score -= ply
                    self.tt[hash_val] = TTEntry(store_score, depth, TT_FLAG_LOWERBOUND, move)
                    return beta

            if legal_moves_count == 0:
                if is_in_check_flag: return -self.MATE_SCORE + ply
                return self.DRAW_SCORE

            store_score = alpha
            if store_score > self.MATE_SCORE - 1000: store_score += ply
            elif store_score < -self.MATE_SCORE + 1000: store_score -= ply

            flag = TT_FLAG_EXACT if alpha > original_alpha else TT_FLAG_UPPERBOUND
            self.tt[hash_val] = TTEntry(store_score, depth, flag, best_move_for_node)
            return alpha
        finally:
            if path_added:
                search_path.remove(hash_val)
    

    def qsearch(self, board, alpha, beta, turn, ply):
        self.nodes_searched += 1
        if self.cancellation_event.is_set(): raise SearchCancelledException()

        if len(board.white_pieces) + len(board.black_pieces) == 3:
            tb_score_absolute = self.tb_manager.probe(board, turn)
            if tb_score_absolute is not None:
                self.tb_hits += 1
                tb_score = tb_score_absolute if turn == 'white' else -tb_score_absolute
                if tb_score > self.MATE_SCORE - 1000: return tb_score - ply
                elif tb_score < -self.MATE_SCORE + 1000: return tb_score + ply
                return tb_score

        if is_insufficient_material(board): return self.DRAW_SCORE
        if ply >= self.MAX_Q_SEARCH_DEPTH:
            self.used_heuristic_eval = True
            return self.evaluate_board(board, turn)
        
        self.used_heuristic_eval = True
        stand_pat = self.evaluate_board(board, turn)
        is_in_check_flag = is_in_check(board, turn)
        if not is_in_check_flag:
            if stand_pat >= beta: return beta
            alpha = max(alpha, stand_pat)

        if is_in_check_flag:
            promising_moves = list(get_all_pseudo_legal_moves(board, turn))
        else:
            promising_moves = list(generate_all_tactical_moves(board, turn))
        
        scored_moves =[]
        for move in promising_moves:
            moving_piece = board.grid[move[0][0]][move[0][1]]
            target_piece = board.grid[move[1][0]][move[1][1]]
            swing = self._ordering_tactical_swing(board, move, moving_piece, target_piece)
            scored_moves.append((swing, move))
        scored_moves.sort(key=lambda item: item[0], reverse=True)

        legal_moves_count = 0
        for swing, move in scored_moves:
            if not is_in_check_flag and stand_pat + swing + self.Q_SEARCH_SAFETY_MARGIN < alpha: continue
            
            sim_board = board.clone()
            sim_board.make_move(move[0], move[1])
            
            if is_in_check(sim_board, turn): continue
            
            legal_moves_count += 1
            search_score = -self.qsearch(sim_board, -beta, -alpha, ('black' if turn == 'white' else 'white'), ply + 1)
            if search_score >= beta: return beta
            alpha = max(alpha, search_score)
            
        if is_in_check_flag and legal_moves_count == 0: 
            return -self.MATE_SCORE + ply
            
        return alpha

    def order_moves(self, board, moves, ply, hash_move, return_meta=False):
        if not moves:
            return [] if return_meta else []

        scored_moves = []
        move_meta = {}
        killers = self.killer_moves[ply] if ply < len(self.killer_moves) else [None, None]
        color_index = 0 if (self.color if ply % 2 == 0 else self.opponent_color) == 'white' else 1
        opening_bonus_enabled = (ply <= self.OPENING_BONUS_MAX_PLY and self._is_opening_position(board))
        
        for move in moves:
            moving_piece = board.grid[move[0][0]][move[0][1]]
            target_piece = board.grid[move[1][0]][move[1][1]]
            is_tactical = self._is_tactical_move(board, move, moving_piece=moving_piece, target_piece=target_piece)
            move_meta[move] = (is_tactical, moving_piece)

            if move == hash_move:
                score = self.BONUS_PV_MOVE
            elif is_tactical:
                swing = self._ordering_tactical_swing(board, move, moving_piece, target_piece)
                # --- APPLY SEE (Static Exchange Evaluation) SPLIT ---
                if swing >= 0:
                    score = self.BONUS_CAPTURE + swing
                else:
                    score = self.BAD_TACTIC_PENALTY + swing
                # ----------------------------------------------------
            else:
                if move in killers:
                    score = self.BONUS_KILLER_1 if move == killers[0] else self.BONUS_KILLER_2
                else:
                    from_idx, to_idx = move[0][0]*COLS+move[0][1], move[1][0]*COLS+move[1][1]
                    score = self.history_heuristic_table[color_index][from_idx][to_idx]

            if opening_bonus_enabled:
                score += self._opening_development_bonus(move, moving_piece)
            scored_moves.append((score, move))

        scored_moves.sort(key=lambda item: item[0], reverse=True)
        if return_meta:
            return [(move, move_meta[move]) for _, move in scored_moves]
        return [move for _, move in scored_moves]

    def evaluate_board(self, board, turn_to_move):
        if is_insufficient_material(board):
            return self.DRAW_SCORE

        scores_mg = [0, 0]; scores_eg = [0, 0]
        piece_counts = [0, 0]; pawn_counts = [0, 0]; last_piece_type = [None, None]
        rook_counts = [0, 0]; bishop_counts = [0, 0]; knight_counts = [0, 0]; queen_counts = [0, 0]
        
        king_pos =[board.white_king_pos, board.black_king_pos]
        piece_lists =[board.white_pieces, board.black_pieces]
        grid = board.grid
        phase_material_score = 0
        
        PAWN_PHALANX_BONUS = 5
        ROOK_ALIGNMENT_BONUS = 15
        PIECE_DOMINANCE_FACTOR = 40
        PAIR_BONUS = 20
        DOUBLE_ROOK_PENALTY = 15
        ROOK_PAWN_SCALING = 5

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
                    
                    if ptype is Rook: rook_counts[color_idx] += 1
                    elif ptype is Bishop: bishop_counts[color_idx] += 1
                    elif ptype is Knight: knight_counts[color_idx] += 1
                    elif ptype is Queen: queen_counts[color_idx] += 1

                val_mg = MG_PIECE_VALUES[ptype]
                val_eg = EG_PIECE_VALUES[ptype] 
                
                r_pst = r if is_white else 7 - r
                
                if ptype is King:
                    scores_mg[color_idx] += PIECE_SQUARE_TABLES['king_midgame'][r_pst][c]
                    scores_eg[color_idx] += PIECE_SQUARE_TABLES['king_endgame'][r_pst][c]
                else:
                    scores_mg[color_idx] += val_mg
                    scores_eg[color_idx] += val_eg
                    if PIECE_SQUARE_TABLES.get(ptype):
                        pst_val = PIECE_SQUARE_TABLES[ptype][r_pst][c]
                        scores_mg[color_idx] += pst_val; scores_eg[color_idx] += pst_val

                # Variant Heuristics
                if ptype is Pawn:
                    if (c > 0 and isinstance(grid[r][c-1], Pawn) and grid[r][c-1].color == my_color_name) or \
                       (c < COLS-1 and isinstance(grid[r][c+1], Pawn) and grid[r][c+1].color == my_color_name):
                        scores_mg[color_idx] += PAWN_PHALANX_BONUS
                elif ptype is Rook:
                    if enemy_king and (r == enemy_king[0] or c == enemy_king[1]):
                        scores_mg[color_idx] += ROOK_ALIGNMENT_BONUS

        # 2. Global Calculations
        phase = min(256, (phase_material_score * 256) // INITIAL_PHASE_MATERIAL) if INITIAL_PHASE_MATERIAL > 0 else 0
        inv_phase = 256 - phase
        
        total_pawns_on_board = pawn_counts[0] + pawn_counts[1]

        if piece_counts[0] > piece_counts[1]: scores_eg[0] += PIECE_DOMINANCE_FACTOR // (piece_counts[1] + 1)
        elif piece_counts[1] > piece_counts[0]: scores_eg[1] += PIECE_DOMINANCE_FACTOR // (piece_counts[0] + 1)

        # Define Penalty Tables
        LONE_ROOK_PENALTIES   = [550, 200, 150, 80, 40]
        LONE_BISHOP_PENALTIES = [650, 250, 170, 100, 50]

        for i in (0, 1):
            if pawn_counts[i] < 4:
                penalty = int(-250 * (4 - pawn_counts[i])**2 / 16)
                scores_mg[i] += penalty; scores_eg[i] += penalty
            
            # Apply Lone Wolf "Draw Damping" LAST
            if piece_counts[i] == 1 and pawn_counts[i] <= 4:
                penalty = 0
                if last_piece_type[i] is Rook:
                     penalty = LONE_ROOK_PENALTIES[pawn_counts[i]]
                elif last_piece_type[i] is Bishop:
                     penalty = LONE_BISHOP_PENALTIES[pawn_counts[i]]
                
                if penalty > 0:
                    if i == 0 and scores_eg[0] > scores_eg[1]: # White winning
                        scores_eg[0] = max(scores_eg[1], scores_eg[0] - penalty)
                    elif i == 1 and scores_eg[1] > scores_eg[0]: # Black winning
                        scores_eg[1] = max(scores_eg[0], scores_eg[1] - penalty)

            # Synergy
            if bishop_counts[i] >= 2: 
                scores_mg[i] += PAIR_BONUS; scores_eg[i] += PAIR_BONUS
            if knight_counts[i] >= 2: 
                scores_mg[i] += PAIR_BONUS; scores_eg[i] += PAIR_BONUS
            if rook_counts[i] >= 2:
                scores_mg[i] -= DOUBLE_ROOK_PENALTY; scores_eg[i] -= DOUBLE_ROOK_PENALTY
            
            # Rook Scaling
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
        
        # ----------------------------------------------------------------
        # KNOWLEDGE BASE: UNWINNABLE ENDGAMES
        # ----------------------------------------------------------------
        can_force_mate = [True, True]
        for i in (0, 1):
            if (pawn_counts[i] == 0 and 
                knight_counts[i] == 0 and 
                queen_counts[i] == 0 and 
                (rook_counts[i] + bishop_counts[i]) < 2):
                can_force_mate[i] = False
                
        if final_score > 0 and not can_force_mate[0]:
            final_score //= 8
        elif final_score < 0 and not can_force_mate[1]:
            final_score //= 8
            
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