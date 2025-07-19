# Jungle_Chess/ai.py (or wherever OpponentAI is defined)
import time
from GameLogic import *
# We assume OpponentAI might be in the same file or can import ChessBot if needed,
# but based on the template, it seems intended to be self-contained with the v0.6 logic.
# If it's inheriting, remove the duplicated methods. If it's meant to be a distinct snapshot,
# keep the methods as copied below. Let's assume it's a snapshot.
import random # Needed for Zobrist

class OpponentAI: # Not inheriting from ChessBot to keep v0.6 logic separate
    # --- Copied from ChessBot v0.6 (Negamax + Killers, Slow Repetition Check) ---
    search_depth = 3 # Default search depth for opponent
    CENTER_SQUARES = {(3, 3), (3, 4), (4, 3), (4, 4)}
    PIECE_VALUES = {
        Pawn: 100, Knight: 700, Bishop: 600,
        Rook: 500, Queen: 900, King: 100000
    }
    MATE_SCORE = 1000000 # Use a large number for mate, distinct from evaluation bounds
    DRAW_SCORE = 0

    def __init__(self, board, color, app):
        # Initialization identical to v0.6 ChessBot
        self.board = board
        self.color = color
        self.opponent_color = 'black' if color == 'white' else 'white'
        self.app = app
        self.tt = {}
        self.nodes_searched = 0
        self.zobrist_table = self.initialize_zobrist_table()
        self.MAX_PLY_KILLERS = 30
        self.killer_moves = [[None, None] for _ in range(self.MAX_PLY_KILLERS)]
        # History/Counts are passed into negamax, not stored long-term on self here
        # self.position_history = [] # Not needed by this negamax implementation
        # self.position_counts = {} # Not needed by this negamax implementation

    def initialize_zobrist_table(self): # v0.6 version
        random.seed(42)
        return {
            (r, c, piece_type, color): random.getrandbits(64)
            for r in range(ROWS) for c in range(COLS)
            for piece_type in [Pawn, Knight, Bishop, Rook, Queen, King, None]
            for color in ['white', 'black', None]
        }

    def board_hash(self, board): # v0.6 version (no turn included)
        hash_val = 0
        for r in range(ROWS):
            for c in range(COLS):
                piece = board[r][c]
                key = (r, c, type(piece) if piece else None, piece.color if piece else None)
                hash_val ^= self.zobrist_table.get(key, 0)
        return hash_val

    # ====================================================
    # Threat Checking Methods (v0.6 version)
    # ====================================================
    def is_in_explosion_threat(self, board, color): # v0.6 version
        king_pos = None; enemy_color = self.opponent_color if color == self.color else self.color; enemy_queens = []
        for r in range(ROWS):
            for c_loop in range(COLS):
                p = board[r][c_loop]
                if isinstance(p, King) and p.color == color: king_pos = (r, c_loop)
                elif isinstance(p, Queen) and p.color == enemy_color: enemy_queens.append((r, c_loop))
        if not king_pos: return False
        for r_q, c_q in enemy_queens:
            qp = board[r_q][c_q]
            if not qp: continue
            for mv in qp.get_valid_moves(board, (r_q, c_q)):
                if max(abs(mv[0] - king_pos[0]), abs(mv[1] - king_pos[1])) == 1:
                    tp = board[mv[0]][mv[1]]
                    if tp and tp.color == color: return True
        return False

    def is_in_knight_evaporation_threat(self, board, color): # v0.6 version
        enemy_color = self.opponent_color if color == self.color else self.color
        for r in range(ROWS):
            for c_loop in range(COLS):
                p = board[r][c_loop]
                if isinstance(p, Knight) and p.color == color:
                    for dr, dc in DIRECTIONS['knight']:
                        nr, nc = r + dr, c_loop + dc
                        if 0 <= nr < ROWS and 0 <= nc < COLS:
                            tp = board[nr][nc]
                            if isinstance(tp, Knight) and tp.color == enemy_color: return True
        return False

    # ====================================================
    # Position Evaluation Methods (v0.6 Negamax version)
    # ====================================================
    def evaluate_board(self, board, depth, current_turn): # v0.6 Negamax version
        perspective_multiplier = 1 if current_turn == self.color else -1
        score_relative_to_ai = 0
        our_king_pos = None; enemy_king_pos = None; enemy_pieces = []
        for r_eval in range(ROWS):
            for c_eval in range(COLS):
                piece_eval = board[r_eval][c_eval]
                if not piece_eval: continue
                if isinstance(piece_eval, King):
                    if piece_eval.color == self.color: our_king_pos = (r_eval, c_eval)
                    else: enemy_king_pos = (r_eval, c_eval)
                    continue
                is_our_piece = (piece_eval.color == self.color)
                if not is_our_piece: enemy_pieces.append((r_eval, c_eval, piece_eval))
                value = self.PIECE_VALUES.get(type(piece_eval), 0)
                if isinstance(piece_eval, Knight):
                    if piece_eval: value += len(piece_eval.get_valid_moves(board, (r_eval, c_eval))) * 5
                elif isinstance(piece_eval, Queen) and is_our_piece:
                    atomic_threats = 0
                    if piece_eval:
                        atomic_threats = sum(
                            1 for move_q in piece_eval.get_valid_moves(board, (r_eval, c_eval))
                            if any(isinstance(board[adj_r][adj_c], King) and board[adj_r][adj_c].color != self.color
                                for adj_r, adj_c in self.get_adjacent_squares(move_q)))
                    value += atomic_threats * 15
                score_relative_to_ai += value if is_our_piece else -value
        if not enemy_king_pos: return self.MATE_SCORE * perspective_multiplier
        if not our_king_pos: return -self.MATE_SCORE * perspective_multiplier
        king_to_check_pos = our_king_pos if current_turn == self.color else enemy_king_pos
        pieces_that_threaten = []
        if current_turn == self.color: pieces_that_threaten = enemy_pieces
        else:
             for r_our in range(ROWS):
                 for c_our in range(COLS):
                     p_our = board[r_our][c_our]
                     if p_our and p_our.color == self.color and not isinstance(p_our, King):
                         pieces_that_threaten.append((r_our,c_our,p_our))
        threat_score_on_current_player = 0
        if king_to_check_pos:
            for r_p, c_p, p_obj in pieces_that_threaten:
                if p_obj and king_to_check_pos in p_obj.get_valid_moves(board, (r_p, c_p)):
                    threat_score_on_current_player += 200
        explosion_threat_on_current_player = self.is_in_explosion_threat(board, current_turn)
        final_score_ai_perspective = score_relative_to_ai
        if current_turn == self.color:
            final_score_ai_perspective -= threat_score_on_current_player
            final_score_ai_perspective -= (250 if explosion_threat_on_current_player else 0)
        else:
            final_score_ai_perspective += threat_score_on_current_player
            final_score_ai_perspective += (250 if explosion_threat_on_current_player else 0)
        return int(final_score_ai_perspective * perspective_multiplier)

    # ====================================================
    # Move Ordering Methods (v0.6 Negamax version + Killers)
    # ====================================================
    def evaluate_move(self, board, move): # v0.6 heuristic
        start, end = move; piece = board[start[0]][start[1]]; target = board[end[0]][end[1]]; score_val = 0
        if target: score_val = 1000 + self.PIECE_VALUES.get(type(target), 0)
        if end in self.CENTER_SQUARES: score_val += 50
        if isinstance(piece, Pawn):
            score_val += (end[0] if piece.color == "white" else (ROWS - 1 - end[0]))
            if end[0] == 0 or end[0] == ROWS - 1: score_val += self.PIECE_VALUES[Queen]
        return score_val
    
    def order_moves(self, board, moves, ply_searched=0): # v0.6 Negamax version
        if not moves: return moves
        board_key_ord = self.board_hash(board)
        tt_move = None
        tt_entry = self.tt.get(board_key_ord)
        if tt_entry and len(tt_entry) >= 3 and tt_entry[2] is not None: tt_move = tt_entry[2]
        move_scores = {}; valid_moves_set = set(moves)
        if tt_move and tt_move in valid_moves_set:
            move_scores[tt_move] = float('inf')
        if ply_searched < self.MAX_PLY_KILLERS:
            killers_for_ply = self.killer_moves[ply_searched]
            if killers_for_ply[0] and killers_for_ply[0] in valid_moves_set and killers_for_ply[0] not in move_scores:
                move_scores[killers_for_ply[0]] = 20000
            if killers_for_ply[1] and killers_for_ply[1] in valid_moves_set and killers_for_ply[1] not in move_scores:
                move_scores[killers_for_ply[1]] = 19999
        for move_val in moves:
            if move_val not in move_scores:
                move_scores[move_val] = self.evaluate_move(board, move_val)
        sorted_scored_moves = sorted(move_scores.items(), key=lambda item: item[1], reverse=True)
        return [move_val for move_val, score_val in sorted_scored_moves]

    # ====================================================
    # Search Methods (v0.6 Negamax Version - NO QSearch)
    # Uses dictionary COPYING for repetition check (slow but matches old baseline)
    # ====================================================
    def negamax(self, board, depth, alpha, beta,
                # Takes original history/counts parameters like old minimax
                position_history=None, # Passed but potentially unused if counts check done first
                position_counts=None,  # Passed and COPIED
                ply_searched=0,        # For killers
                turn_multiplier=1      # Implicitly 1 for initial call from OpponentAI's perspective
               ):
        self.nodes_searched += 1

        # --- Initialize history/counts with COPYING (as per old baseline) ---
        if position_history is None:
            # This assumes OpponentAI might be called without history/counts,
            # initialize from app state if so. In recursive calls, it receives copies.
            position_history = self.app.position_history.copy()
            position_counts = self.app.position_counts.copy()
        # --------------------------------------------------------------------

        current_turn = self.color if turn_multiplier == 1 else self.opponent_color

        # --- Repetition Check (using the PASSED counts dict) ---
        current_key = generate_position_key(board, current_turn)
        if position_counts.get(current_key, 0) >= 2: # Check if this is the 3rd time
            return self.DRAW_SCORE

        # --- Stalemate Check ---
        if is_stalemate(board, current_turn):
            return self.DRAW_SCORE

        # --- TT Lookup ---
        board_key_tt = self.board_hash(board)
        tt_entry = self.tt.get(board_key_tt)
        if tt_entry and tt_entry[0] >= depth:
            return tt_entry[1]

        # --- Depth Limit ---
        if depth == 0:
            # Evaluate board returns score relative to player whose turn it is
            return self.evaluate_board(board, depth, current_turn)

        # --- Null Move Pruning ---
        if depth >= 3 and not is_in_check(board, current_turn):
            null_move_reduction = 1
            # Null move needs to pass COPIES of history/counts
            pos_hist_null = position_history.copy()
            pos_counts_null = position_counts.copy()
            opponent_turn_after_null = self.opponent_color if turn_multiplier == 1 else self.color
            key_after_null_str = generate_position_key(board, opponent_turn_after_null)
            pos_counts_null[key_after_null_str] = pos_counts_null.get(key_after_null_str, 0) + 1

            null_value_score = -self.negamax(board, depth - 1 - null_move_reduction,
                                             -beta, -alpha,
                                             pos_hist_null, pos_counts_null, # Pass COPIES
                                             ply_searched + 1,
                                             -turn_multiplier) # Flip perspective

            # No need to backtrack copies
            if null_value_score >= beta:
                 return beta # Fail high

        # --- Move Generation & Loop ---
        moves = self.get_all_moves(board, current_turn)
        if not moves:
            if is_in_check(board, current_turn):
                return -self.MATE_SCORE + ply_searched
            else:
                return self.DRAW_SCORE

        ordered_moves = self.order_moves(board, moves, ply_searched)
        best_value_node = -float('inf')
        best_move_for_tt = None

        for i, move_tuple in enumerate(ordered_moves):
            start_pos, end_pos = move_tuple
            current_piece_moved = board[start_pos[0]][start_pos[1]]
            target_on_board = board[end_pos[0]][end_pos[1]]
            is_tactical = (target_on_board is not None) or \
                          (isinstance(current_piece_moved, Pawn) and \
                           (end_pos[0] == 0 or end_pos[0] == ROWS - 1))

            child_board = self.simulate_move(board, start_pos, end_pos)

            # --- Clone history/counts for child (original slow method) ---
            child_pos_hist = position_history.copy()
            child_pos_counts = position_counts.copy()
            child_turn_after_move = self.opponent_color if turn_multiplier == 1 else self.color
            child_key_str = generate_position_key(child_board, child_turn_after_move)
            child_pos_hist.append(child_key_str)
            child_pos_counts[child_key_str] = child_pos_counts.get(child_key_str, 0) + 1
            # ------------------------------------------------------------

            reduction_lmr = 0
            if not is_tactical and i >= (2 if depth >=4 else 1) and depth >= 2:
                reduction_lmr = 1
            child_search_depth = depth - 1 - reduction_lmr

            score_from_child_negated = 0
            if i == 0: # PV Node
                score_from_child_negated = -self.negamax(child_board, child_search_depth,
                                                         -beta, -alpha,
                                                         child_pos_hist, child_pos_counts, # Pass COPY
                                                         ply_searched + 1, -turn_multiplier)
            else: # Scout Node
                score_from_child_negated = -self.negamax(child_board, child_search_depth,
                                                         -alpha - 1, -alpha, # Zero window
                                                         child_pos_hist, child_pos_counts, # Pass COPY
                                                         ply_searched + 1, -turn_multiplier)
                if score_from_child_negated > alpha and score_from_child_negated < beta:
                    score_from_child_negated = -self.negamax(child_board, depth - 1, # Full depth re-search
                                                             -beta, -alpha,
                                                             child_pos_hist, child_pos_counts, # Pass COPY
                                                             ply_searched + 1, -turn_multiplier)

            # No need to backtrack copies

            if score_from_child_negated > best_value_node:
                best_value_node = score_from_child_negated
                best_move_for_tt = move_tuple
            alpha = max(alpha, best_value_node)
            if alpha >= beta:
                if not is_tactical and ply_searched < self.MAX_PLY_KILLERS:
                    km_ply_list = self.killer_moves[ply_searched]
                    if move_tuple != km_ply_list[0]:
                        km_ply_list[1] = km_ply_list[0]; km_ply_list[0] = move_tuple
                break

        self.tt[board_key_tt] = (depth, best_value_node, best_move_for_tt)
        return best_value_node


    def make_move(self): # v0.6 Negamax Root Call Structure
        overall_best_move = None; overall_best_value = -float('inf')
        total_start_time = time.time()
        
        # --- Use original history/counts COPYING for root calls ---
        # Required because this version of negamax expects copies
        root_position_history = self.app.position_history.copy()
        root_position_counts = self.app.position_counts.copy()
        # ----------------------------------------------------------
        
        self.killer_moves = [[None, None] for _ in range(self.MAX_PLY_KILLERS)] # Clear killers

        for current_depth_iter in range(1, self.search_depth + 1):
            self.nodes_searched = 0; iter_start_time = time.time()
            root_moves = self.get_all_moves(self.board, self.color)
            if not root_moves: return False
            
            ordered_root_moves = self.order_moves(self.board, root_moves, 0) 
            current_iter_best_move = None; current_iter_best_value = -float('inf')
            alpha_root = -float('inf'); beta_root = float('inf')

            for move_tuple_root in ordered_root_moves: 
                start_root, end_root = move_tuple_root
                child_board_root = self.simulate_move(self.board, start_root, end_root)
                
                # --- Clone history/counts for this root move call (original slow method) ---
                child_pos_hist_root = root_position_history.copy()
                child_pos_counts_root = root_position_counts.copy()
                child_turn_root = self.opponent_color
                child_key_str_root = generate_position_key(child_board_root, child_turn_root)
                child_pos_hist_root.append(child_key_str_root)
                child_pos_counts_root[child_key_str_root] = child_pos_counts_root.get(child_key_str_root, 0) + 1
                # --------------------------------------------------------------------------
                
                eval_for_move = 0 
                if child_pos_counts_root[child_key_str_root] >= 3: # Check repetition in the COPIED counts
                    eval_for_move = self.DRAW_SCORE 
                else:
                    # Call negamax for opponent's turn, passing COPIES of history/counts
                    score_from_opponent_view = self.negamax(
                        child_board_root, 
                        current_depth_iter - 1, 
                        -beta_root, -alpha_root, 
                        child_pos_hist_root, child_pos_counts_root, # Pass COPIES
                        0, # ply_searched = 0 for root children
                        -1 # turn_multiplier for opponent
                    )
                    eval_for_move = -score_from_opponent_view 
                
                # No need to backtrack copies
                
                if eval_for_move > current_iter_best_value:
                    current_iter_best_value = eval_for_move
                    current_iter_best_move = move_tuple_root
                alpha_root = max(alpha_root, current_iter_best_value)

            iter_time_val = time.time() - iter_start_time
            if current_iter_best_move is not None:
                overall_best_move = current_iter_best_move
                overall_best_value = current_iter_best_value
            reported_value_ui = overall_best_value
            if self.color == 'black': reported_value_ui = -overall_best_value
            eval_to_print_ui = f"{overall_best_value:.0f}"
            if overall_best_value > self.MATE_SCORE - 50 : eval_to_print_ui = f"Mate found" 
            elif overall_best_value < -self.MATE_SCORE + 50 : eval_to_print_ui = f"Mated"
            print(f"AI Depth {current_depth_iter}: {iter_time_val:.3f}s, Nodes: {self.nodes_searched}, Eval: {eval_to_print_ui} (AI: {self.color}), Move: {overall_best_move}")
            if self.app:
                ui_val = reported_value_ui
                if ui_val < -self.MATE_SCORE + 100 : ui_val = -99000 
                elif ui_val > self.MATE_SCORE - 100 : ui_val = 99000 
                self.app.master.after(0, lambda v=ui_val: self.app.draw_eval_bar(v))

        print(f"AI Total time: {(time.time() - total_start_time):.3f}s")
        if overall_best_move:
            start_final, end_final = overall_best_move
            moving_piece_final = self.board[start_final[0]][start_final[1]]
            self.board = moving_piece_final.move(self.board, start_final, end_final)
            check_evaporation(self.board)
            new_key_for_app = generate_position_key(self.board, self.opponent_color)
            self.app.position_history.append(new_key_for_app)
            self.app.position_counts[new_key_for_app] = self.app.position_counts.get(new_key_for_app, 0) + 1
            if self.app.position_counts[new_key_for_app] >= 3:
                self.app.game_over = True; self.app.game_result = ("repetition", None)
                if hasattr(self.app, 'turn_label'): self.app.turn_label.config(text="Draw by three-fold repetition!")
            return True
        return False

    # --- Original Helpers ---
    def get_adjacent_squares(self, pos): # Original
        r_pos, c_pos = pos; adj_squares = []
        for adj_r_loop in range(r_pos - 1, r_pos + 2):
            for adj_c_loop in range(c_pos - 1, c_pos + 2):
                if 0 <= adj_r_loop < ROWS and 0 <= adj_c_loop < COLS and \
                   (adj_r_loop, adj_c_loop) != (r_pos, c_pos): adj_squares.append((adj_r_loop, adj_c_loop))
        return adj_squares
    def simulate_move(self, board, start, end): # Original
        new_board = copy_board(board); piece_to_move = new_board[start[0]][start[1]]
        if not piece_to_move: return new_board
        new_board = piece_to_move.move(new_board, start, end); check_evaporation(new_board)
        return new_board
    def get_all_moves(self, board, color): # Original
        all_valid_moves = []
        for r_get in range(ROWS):
            for c_get in range(COLS):
                piece_get = board[r_get][c_get]
                if piece_get and piece_get.color == color:
                    for move_end_pos in piece_get.get_valid_moves(board, (r_get, c_get)):
                        if validate_move(board, color, (r_get, c_get), move_end_pos):
                            all_valid_moves.append(((r_get, c_get), move_end_pos))
        return all_valid_moves