import time
from GameLogic import *
import random

# v0.6: Converted Minimax to Negamax framework

class ChessBot:
    search_depth = 3 
    CENTER_SQUARES = {(3, 3), (3, 4), (4, 3), (4, 4)}
    PIECE_VALUES = {
        Pawn: 100, Knight: 700, Bishop: 600,
        Rook: 500, Queen: 900, King: 100000 
    }
    MATE_SCORE = 1000000 # Use a large number for mate, distinct from evaluation bounds
    DRAW_SCORE = 0

    def __init__(self, board, color, app):
        self.board = board
        self.color = color
        self.opponent_color = 'black' if color == 'white' else 'white'
        self.app = app
        self.tt = {}
        self.nodes_searched = 0
        self.zobrist_table = self.initialize_zobrist_table()
        self.MAX_PLY_KILLERS = 30 
        self.killer_moves = [[None, None] for _ in range(self.MAX_PLY_KILLERS)]
        self.position_history = [] # Kept for compatibility if needed elsewhere
        self.position_counts = {} # Kept for compatibility if needed elsewhere

    def initialize_zobrist_table(self): # Original
        random.seed(42)
        return {
            (r, c, piece_type, color): random.getrandbits(64)
            for r in range(ROWS) for c in range(COLS)
            for piece_type in [Pawn, Knight, Bishop, Rook, Queen, King, None]
            for color in ['white', 'black', None]
        }

    def board_hash(self, board): # Original - Note: Does NOT include turn
        hash_val = 0
        for r in range(ROWS):
            for c in range(COLS):
                piece = board[r][c]
                key = (r, c, type(piece) if piece else None, piece.color if piece else None)
                hash_val ^= self.zobrist_table.get(key, 0)
        return hash_val
        # Consider adding turn to hash for more precise TT entries if needed:
        # if turn_color == 'black': hash_val ^= some_black_turn_key

    # ====================================================
    # Threat Checking Methods (Original) - No changes needed
    # ====================================================
    def is_in_explosion_threat(self, board, color): # Original
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

    def is_in_knight_evaporation_threat(self, board, color): # Original
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
    # Position Evaluation Methods (MODIFIED for Negamax)
    # ====================================================
    def evaluate_board(self, board, depth, current_turn): 
        # MODIFIED: Returns score relative to current_turn
        perspective_multiplier = 1 if current_turn == self.color else -1
        
        score_relative_to_ai = 0 # Score calculated from AI's perspective first
        our_king_pos = None; enemy_king_pos = None; enemy_pieces = [] # enemy means opponent of self.color

        for r_eval in range(ROWS):
            for c_eval in range(COLS):
                piece_eval = board[r_eval][c_eval]
                if not piece_eval: continue
                if isinstance(piece_eval, King):
                    if piece_eval.color == self.color: our_king_pos = (r_eval, c_eval)
                    else: enemy_king_pos = (r_eval, c_eval)
                    continue
                
                is_our_piece = (piece_eval.color == self.color)
                if not is_our_piece:
                    enemy_pieces.append((r_eval, c_eval, piece_eval))

                value = self.PIECE_VALUES.get(type(piece_eval), 0)
                if isinstance(piece_eval, Knight):
                    if piece_eval: value += len(piece_eval.get_valid_moves(board, (r_eval, c_eval))) * 5 # Reduced weight slightly
                elif isinstance(piece_eval, Queen) and is_our_piece: # Only our Queen atomic threats
                    atomic_threats = 0
                    if piece_eval:
                        atomic_threats = sum(
                            1 for move_q in piece_eval.get_valid_moves(board, (r_eval, c_eval))
                            if any(isinstance(board[adj_r][adj_c], King) and board[adj_r][adj_c].color != self.color
                                for adj_r, adj_c in self.get_adjacent_squares(move_q)))
                    value += atomic_threats * 15 # Reduced weight slightly
                
                score_relative_to_ai += value if is_our_piece else -value

        # Check for king capture first (absolute win/loss)
        if not enemy_king_pos: return self.MATE_SCORE * perspective_multiplier
        if not our_king_pos: return -self.MATE_SCORE * perspective_multiplier

        # Calculate threats against the player whose turn it currently is
        king_to_check_pos = our_king_pos if current_turn == self.color else enemy_king_pos
        pieces_that_threaten = []
        if current_turn == self.color: # Opponent's pieces threaten our king
            pieces_that_threaten = enemy_pieces
        else: # Our pieces threaten opponent's king
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

        # Adjust final score based on threats to the current player
        # Score relative to AI, needs adjustment based on perspective AND threats
        final_score_ai_perspective = score_relative_to_ai
        if current_turn == self.color: # Threats against AI decrease AI's score
            final_score_ai_perspective -= threat_score_on_current_player
            final_score_ai_perspective -= (250 if explosion_threat_on_current_player else 0)
        else: # Threats against opponent increase AI's score (as it's opponent's score decrease)
            final_score_ai_perspective += threat_score_on_current_player
            final_score_ai_perspective += (250 if explosion_threat_on_current_player else 0)

        # Return score adjusted for the perspective of the player whose turn it is
        return int(final_score_ai_perspective * perspective_multiplier)


    # ====================================================
    # Move Ordering Methods (Killers, no maximizing_player flag needed)
    # ====================================================
    def evaluate_move(self, board, move): # Original heuristic
        start, end = move; piece = board[start[0]][start[1]]; target = board[end[0]][end[1]]; score_val = 0
        if target: score_val = 1000 + self.PIECE_VALUES.get(type(target), 0)
        if end in self.CENTER_SQUARES: score_val += 50
        if isinstance(piece, Pawn):
            score_val += (end[0] if piece.color == "white" else (ROWS - 1 - end[0]))
            if end[0] == 0 or end[0] == ROWS - 1: score_val += self.PIECE_VALUES[Queen]
        return score_val
    
    # Removed maximizing_player flag, sort always descending by score
    def order_moves(self, board, moves, ply_searched=0): 
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
    # Search Methods (CONVERTED TO NEGAMAX)
    # ====================================================
    def negamax(self, board, depth, alpha, beta, 
                position_counts_ref, # Passed by reference
                ply_searched,        # For killers
                turn_multiplier      # 1 for current player, -1 for opponent view needed by eval
               ):
        self.nodes_searched += 1

        # Determine current player color based on initial AI color and multiplier
        current_turn = self.color if turn_multiplier == 1 else self.opponent_color

        # --- Repetition Check ---
        # Need the state key for the player whose turn it is NOW
        current_key = generate_position_key(board, current_turn)
        if position_counts_ref.get(current_key, 0) >= 2: # Check if this is the 3rd time
            return self.DRAW_SCORE # Return 0 for draw

        # --- Stalemate Check ---
        # Note: is_stalemate needs the color of the player whose turn it is
        if is_stalemate(board, current_turn): 
            return self.DRAW_SCORE # Return 0 for draw

        # --- TT Lookup ---
        # TT key should ideally include turn if board_hash doesn't.
        # Using board_hash only for now as per original.
        board_key_tt = self.board_hash(board) 
        tt_entry = self.tt.get(board_key_tt)
        if tt_entry and tt_entry[0] >= depth:
            # Need to verify TT score perspective if storing explicit Max/Min scores
            # If TT stores Negamax score (relative to player at that node), return directly
            return tt_entry[1] 

        # --- Depth Limit ---
        if depth == 0:
            # evaluate_board returns score relative to current_turn
            return self.evaluate_board(board, depth, current_turn)

        # --- Null Move Pruning ---
        if depth >= 3 and not is_in_check(board, current_turn):
            null_move_reduction = 1
            # Manage position counts for null move state (opponent's turn)
            opponent_turn_after_null = self.opponent_color if turn_multiplier == 1 else self.color
            key_after_null_str = generate_position_key(board, opponent_turn_after_null)
            position_counts_ref[key_after_null_str] = position_counts_ref.get(key_after_null_str, 0) + 1
            
            # Negamax call for opponent: Pass negated and swapped bounds (-beta, -alpha)
            # Score returned will be from opponent's view, negate it.
            null_value_score = -self.negamax(board, depth - 1 - null_move_reduction, 
                                             -beta, -alpha, # Negated and swapped bounds
                                             position_counts_ref, 
                                             ply_searched + 1, # Increment ply
                                             -turn_multiplier) # Flip perspective
            
            position_counts_ref[key_after_null_str] -= 1
            if position_counts_ref[key_after_null_str] == 0: del position_counts_ref[key_after_null_str]

            # If null move score (from current player's view) causes beta cutoff
            if null_value_score >= beta:
                 # Optional: Store TT entry for beta cutoff
                 # self.tt[board_key_tt] = (depth, beta, None) # Store beta as lower bound
                 return beta # Fail high (beta cutoff)

        # --- Move Generation & Loop ---
        moves = self.get_all_moves(board, current_turn)
        if not moves: # No legal moves
            if is_in_check(board, current_turn): # Current player is checkmated
                # Return score from current player's perspective (very bad)
                # Adjust by ply to prefer faster mates
                return -self.MATE_SCORE + ply_searched # Lower score if mated deeper (higher ply)
            else: # Stalemate handled earlier, but double check
                return self.DRAW_SCORE

        # Order moves (no maximizing_player needed)
        ordered_moves = self.order_moves(board, moves, ply_searched) 

        best_value_node = -float('inf') # Always maximizing from current player's view
        best_move_for_tt = None
        
        for i, move_tuple in enumerate(ordered_moves):
            start_pos, end_pos = move_tuple
            current_piece_moved = board[start_pos[0]][start_pos[1]]
            target_on_board = board[end_pos[0]][end_pos[1]]
            is_tactical = (target_on_board is not None) or \
                          (isinstance(current_piece_moved, Pawn) and \
                           (end_pos[0] == 0 or end_pos[0] == ROWS - 1))

            child_board = self.simulate_move(board, start_pos, end_pos)
            
            # Manage position counts
            child_turn_after_move = self.opponent_color if turn_multiplier == 1 else self.color
            child_key_str = generate_position_key(child_board, child_turn_after_move)
            position_counts_ref[child_key_str] = position_counts_ref.get(child_key_str, 0) + 1
            
            # LMR Reduction
            reduction_lmr = 0
            if not is_tactical and i >= (2 if depth >=4 else 1) and depth >= 2:
                reduction_lmr = 1
            child_search_depth = depth - 1 - reduction_lmr
            
            score_from_child_negated = 0 # Score from CURRENT player's perspective
            # PVS Logic with Negamax
            if i == 0: # PV Node
                score_from_child_negated = -self.negamax(child_board, child_search_depth, 
                                                         -beta, -alpha, 
                                                         position_counts_ref, 
                                                         ply_searched + 1, -turn_multiplier)
            else: # Scout Node
                score_from_child_negated = -self.negamax(child_board, child_search_depth, 
                                                         -alpha - 1, -alpha, # Zero window
                                                         position_counts_ref, 
                                                         ply_searched + 1, -turn_multiplier)
                # If scout succeeded within window, re-search
                if score_from_child_negated > alpha and score_from_child_negated < beta:
                    score_from_child_negated = -self.negamax(child_board, depth - 1, # Full depth re-search
                                                             -beta, -alpha, # Use original full window
                                                             position_counts_ref, 
                                                             ply_searched + 1, -turn_multiplier)

            # Backtrack position counts
            position_counts_ref[child_key_str] -= 1
            if position_counts_ref[child_key_str] == 0: del position_counts_ref[child_key_str]

            # Update best score and alpha
            if score_from_child_negated > best_value_node:
                best_value_node = score_from_child_negated
                best_move_for_tt = move_tuple
            
            alpha = max(alpha, best_value_node)

            # Beta cutoff check
            if alpha >= beta:
                # Record Killer Move
                if not is_tactical and ply_searched < self.MAX_PLY_KILLERS:
                    km_ply_list = self.killer_moves[ply_searched]
                    if move_tuple != km_ply_list[0]: 
                        km_ply_list[1] = km_ply_list[0]; km_ply_list[0] = move_tuple
                break 

        # Store score relative to the player whose turn it was at this node
        self.tt[board_key_tt] = (depth, best_value_node, best_move_for_tt)
        return best_value_node

    def make_move(self): 
        overall_best_move = None; overall_best_value = -float('inf')
        total_start_time = time.time()
        current_game_turn_pos_counts = self.app.position_counts.copy()
        self.killer_moves = [[None, None] for _ in range(self.MAX_PLY_KILLERS)] # Clear killers

        for current_depth_iter in range(1, self.search_depth + 1):
            self.nodes_searched = 0; iter_start_time = time.time()
            root_moves = self.get_all_moves(self.board, self.color)
            if not root_moves: return False
            
            # Order moves (ply_searched=0 for root)
            ordered_root_moves = self.order_moves(self.board, root_moves, 0) 

            current_iter_best_move = None; current_iter_best_value = -float('inf')
            alpha_root = -float('inf'); beta_root = float('inf')

            for move_tuple_root in ordered_root_moves: 
                start_root, end_root = move_tuple_root
                child_board_root = self.simulate_move(self.board, start_root, end_root)
                
                child_turn_root = self.opponent_color
                child_key_str_root = generate_position_key(child_board_root, child_turn_root)
                current_game_turn_pos_counts[child_key_str_root] = current_game_turn_pos_counts.get(child_key_str_root, 0) + 1
                
                eval_for_move = 0 
                if current_game_turn_pos_counts[child_key_str_root] >= 3:
                    eval_for_move = self.DRAW_SCORE 
                else:
                    # Initial call to negamax. Score returned is from AI's perspective.
                    # turn_multiplier = 1 for the initial call representing AI's turn.
                    # However, the recursive call inside will be for opponent with -beta, -alpha and -multiplier.
                    # Let's simplify: the root call wants to maximize score for self.color.
                    # We call negamax for the child state (opponent's turn). Negamax returns score from opponent's view.
                    # We negate it to get the score from AI's view.
                    
                    score_from_opponent_view = self.negamax(
                        child_board_root, 
                        current_depth_iter - 1, 
                        -beta_root, -alpha_root, # Initial alpha/beta for opponent's search (-inf, +inf -> -inf, +inf)
                        current_game_turn_pos_counts, 
                        0, # ply_searched = 0 for root children
                        -1 # turn_multiplier for opponent is -1 relative to AI's initial 1
                    )
                    eval_for_move = -score_from_opponent_view # Score from AI's perspective
                
                current_game_turn_pos_counts[child_key_str_root] -= 1
                if current_game_turn_pos_counts[child_key_str_root] == 0: del current_game_turn_pos_counts[child_key_str_root]
                
                if eval_for_move > current_iter_best_value:
                    current_iter_best_value = eval_for_move
                    current_iter_best_move = move_tuple_root
                
                alpha_root = max(alpha_root, current_iter_best_value) # Update alpha for root

            iter_time_val = time.time() - iter_start_time
            if current_iter_best_move is not None:
                overall_best_move = current_iter_best_move
                overall_best_value = current_iter_best_value
            
            reported_value_ui = overall_best_value
            if self.color == 'black': reported_value_ui = -overall_best_value
            
            eval_to_print_ui = f"{overall_best_value:.0f}"
            # Mate score printing logic (adjust numbers if MATE_SCORE changes)
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