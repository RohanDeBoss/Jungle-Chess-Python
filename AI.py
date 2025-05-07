import time
from GameLogic import *
import random

#v0.4 Actually works!!

class ChessBot:
    search_depth = 2
    CENTER_SQUARES = {(3, 3), (3, 4), (4, 3), (4, 4)}
    PIECE_VALUES = {
        Pawn: 100,
        Knight: 700,
        Bishop: 600,
        Rook: 500,
        Queen: 900,
        King: 100000 # This is a material value. Checkmate is handled by search logic.
    }
    # NO MATE_SCORE or DRAW_SCORE constants here, use original float('-inf')/0

    def __init__(self, board, color, app):
        self.board = board
        self.color = color
        self.opponent_color = 'black' if color == 'white' else 'white' # Precompute
        self.tt = {}
        self.nodes_searched = 0
        self.app = app
        self.zobrist_table = self.initialize_zobrist_table()
        # These are from your original code, they were used in minimax signature
        # The optimized version will modify how minimax uses position_counts
        self.position_history = [] # Not strictly used by minimax logic but kept for signature match
        self.position_counts = {}  # This will be the one passed and mutated

    def initialize_zobrist_table(self):
        random.seed(42)
        return {
            (r, c, piece_type, color): random.getrandbits(64)
            for r in range(ROWS)
            for c in range(COLS)
            for piece_type in [Pawn, Knight, Bishop, Rook, Queen, King, None]
            for color in ['white', 'black', None]
        }

    def board_hash(self, board): # Original
        hash_val = 0
        for r in range(ROWS):
            for c in range(COLS):
                piece = board[r][c]
                key = (r, c, type(piece) if piece else None, 
                      piece.color if piece else None)
                hash_val ^= self.zobrist_table.get(key, 0)
        return hash_val

    # ====================================================
    # Threat Checking Methods (Original)
    # ====================================================
    def is_in_explosion_threat(self, board, color):
        king_pos = None
        enemy_color = self.opponent_color if color == self.color else self.color
        enemy_queens = []
        for r_loop in range(ROWS):
            for c_loop in range(COLS):
                piece_loop = board[r_loop][c_loop]
                if isinstance(piece_loop, King) and piece_loop.color == color:
                    king_pos = (r_loop, c_loop)
                elif isinstance(piece_loop, Queen) and piece_loop.color == enemy_color:
                    enemy_queens.append((r_loop, c_loop))
        if not king_pos: return False
        for r_q, c_q in enemy_queens:
            queen_piece = board[r_q][c_q]
            if not queen_piece: continue
            for move in queen_piece.get_valid_moves(board, (r_q, c_q)):
                if max(abs(move[0] - king_pos[0]), abs(move[1] - king_pos[1])) == 1:
                    target_piece = board[move[0]][move[1]]
                    if target_piece and target_piece.color == color:
                        return True
        return False

    def is_in_knight_evaporation_threat(self, board, color):
        enemy_color = self.opponent_color if color == self.color else self.color
        for r_loop in range(ROWS):
            for c_loop in range(COLS):
                piece_loop = board[r_loop][c_loop]
                if isinstance(piece_loop, Knight) and piece_loop.color == color:
                    for dr, dc in DIRECTIONS['knight']:
                        nr, nc = r_loop + dr, c_loop + dc
                        if 0 <= nr < ROWS and 0 <= nc < COLS:
                            target_piece = board[nr][nc]
                            if isinstance(target_piece, Knight) and target_piece.color == enemy_color:
                                return True
        return False

    # ====================================================
    # Position Evaluation Methods (Original)
    # ====================================================
    def evaluate_board(self, board, depth, current_turn):
        score = 0
        our_king_pos = None
        enemy_king_pos = None
        enemy_pieces = [] # Populated as list of (r,c, piece_obj)
        for r_eval in range(ROWS):
            for c_eval in range(COLS):
                piece_eval = board[r_eval][c_eval]
                if not piece_eval: continue
                if isinstance(piece_eval, King):
                    if piece_eval.color == self.color: our_king_pos = (r_eval, c_eval)
                    else: enemy_king_pos = (r_eval, c_eval)
                    continue
                if piece_eval.color != self.color:
                    enemy_pieces.append((r_eval, c_eval, piece_eval)) # Original way: (r, c, piece)
                value = self.PIECE_VALUES.get(type(piece_eval), 0)
                if isinstance(piece_eval, Knight):
                    if piece_eval: value += len(piece_eval.get_valid_moves(board, (r_eval, c_eval))) * 10
                elif isinstance(piece_eval, Queen) and piece_eval.color == self.color:
                    atomic_threats = 0
                    if piece_eval:
                        atomic_threats = sum(
                            1 for move_q in piece_eval.get_valid_moves(board, (r_eval, c_eval))
                            if any(isinstance(board[adj_r][adj_c], King) and board[adj_r][adj_c].color != self.color
                                for adj_r, adj_c in self.get_adjacent_squares(move_q))
                        )
                    value += atomic_threats * 20
                score += value if piece_eval.color == self.color else -value
        
        # Original logic for king missing:
        if not enemy_king_pos or not our_king_pos:
            return float('inf') if not enemy_king_pos else float('-inf') # From current_turn's perspective?
                                                                        # This means if current_turn is self.color and enemy_king_pos is None, return inf.
                                                                        # If current_turn is opponent and enemy_king_pos is None, return inf.
                                                                        # Let's assume it means score for self.color.
                                                                        # If enemy king (opponent of self.color) is gone, score for self.color is inf.
                                                                        # If our king is gone, score for self.color is -inf.
                                                                        # This seems to be the intent of your original.

        threat_score_val = 0
        if our_king_pos:
            for r_p, c_p, p_obj in enemy_pieces: # Unpack (r,c,p)
                if p_obj and our_king_pos in p_obj.get_valid_moves(board, (r_p, c_p)):
                    threat_score_val += 200
        
        explosion_threat_val = self.is_in_explosion_threat(board, self.color)
        return int(score - threat_score_val - (250 if explosion_threat_val else 0))


    # ====================================================
    # Move Ordering Methods (Original)
    # ====================================================
    def evaluate_move(self, board, move): # Original
        start, end = move
        piece = board[start[0]][start[1]]
        target = board[end[0]][end[1]]
        score_val = 0
        if target:
            score_val = 1000 + self.PIECE_VALUES.get(type(target), 0)
        if end in self.CENTER_SQUARES: # This was your original logic location
            score_val += 50
        if isinstance(piece, Pawn):
            score_val += (end[0] if piece.color == "white" else (ROWS - 1 - end[0]))
            # Promotion bonus was implicit in your original piece.move and check_evaporation.
            # If a pawn reaches the end, it promotes. Evaluate_move doesn't need to double-count typically,
            # unless it's a specific heuristic for ordering. Your original didn't have explicit promo bonus here.
        return score_val
    
    def order_moves(self, board, moves, maximizing_player=True): # Original
        if not moves: return moves
        board_key_ord = self.board_hash(board)
        best_tt_move = None
        tt_entry = self.tt.get(board_key_ord)
        if tt_entry and len(tt_entry) > 2 and tt_entry[2] is not None:
            best_tt_move = tt_entry[2]
        
        temp_moves_list = list(moves)
        if best_tt_move and best_tt_move in temp_moves_list:
            temp_moves_list.remove(best_tt_move)
        
        scored_moves = [(self.evaluate_move(board, move_val), move_val) for move_val in temp_moves_list]
        if len(set(score_val for score_val, _ in scored_moves)) > 1:
            scored_moves.sort(reverse=maximizing_player, key=lambda x: x[0]) # Original sort

        final_ordered_moves = [move_val for _, move_val in scored_moves]
        if best_tt_move and best_tt_move in moves: # Add TT move to front if valid
            if best_tt_move not in final_ordered_moves:
                 final_ordered_moves.insert(0, best_tt_move)
            elif final_ordered_moves and final_ordered_moves[0] != best_tt_move:
                 try: final_ordered_moves.remove(best_tt_move)
                 except ValueError: pass
                 final_ordered_moves.insert(0, best_tt_move)
            elif not final_ordered_moves : # if list became empty but TT move was valid
                final_ordered_moves = [best_tt_move]

        return final_ordered_moves


    # ====================================================
    # Search Methods (Minimax & Move Selection)
    # MODIFIED: `position_counts_ref` used instead of copying `position_counts`
    # `position_history` parameter is REMOVED from this minimax signature.
    # ====================================================
    def minimax(self, board, depth, maximizing_player, alpha, beta, 
                position_counts_ref # Passed by reference, mutated by this function and its children
               ):
        self.nodes_searched += 1

        current_turn = self.color if maximizing_player else self.opponent_color
        
        # Check if the current position would result in a threefold repetition
        current_key_str = generate_position_key(board, current_turn) # SLOW part
        # Check count *before* this instance. If it's already appeared twice, this is the third.
        if position_counts_ref.get(current_key_str, 0) >= 2: # Original was >=2, meaning this is the 3rd
            return 0  # Evaluate as draw

        if is_stalemate(board, current_turn):
            return 0

        board_key_tt = self.board_hash(board) # TT Key
        tt_entry = self.tt.get(board_key_tt)
        if tt_entry and tt_entry[0] >= depth:
            return tt_entry[1] # Return stored score

        if depth == 0:
            # evaluate_board returns score from self.color's perspective.
            # Minimax logic (Max/Min) correctly interprets this.
            return self.evaluate_board(board, depth, current_turn)

        # -------- Null Move Pruning (Original logic, but with position_counts_ref) --------
        if depth >= 3 and not is_in_check(board, current_turn):
            null_move_reduction = 1
            
            # Manage position_counts_ref for the null move's state
            # Turn flips for null move's child state
            key_after_null_str = generate_position_key(board, self.opponent_color if maximizing_player else self.color)
            position_counts_ref[key_after_null_str] = position_counts_ref.get(key_after_null_str, 0) + 1
            
            null_value_score = 0 # Renamed null_value
            if maximizing_player:
                null_value_score = self.minimax(board, depth - 1 - null_move_reduction, False, 
                                                alpha, beta, position_counts_ref)
            else:
                null_value_score = self.minimax(board, depth - 1 - null_move_reduction, True, 
                                                alpha, beta, position_counts_ref)
            
            # Backtrack position_counts_ref for null move
            position_counts_ref[key_after_null_str] -= 1
            if position_counts_ref[key_after_null_str] == 0:
                del position_counts_ref[key_after_null_str]

            if maximizing_player and null_value_score >= beta: return null_value_score
            if not maximizing_player and null_value_score <= alpha: return null_value_score
        # ------------------------------------

        moves = self.get_all_moves(board, current_turn)
        if not moves: # No legal moves
            if is_in_check(board, current_turn): # Current player is checkmated
                # Return score from current player's perspective (very bad)
                # The depth adjustment makes shallower mates (against current player) appear worse (more negative)
                return float('-inf') if maximizing_player else float('inf') # Original mate score
            else: # Stalemate
                return 0

        ordered_moves = self.order_moves(board, moves, maximizing_player)

        best_value_node = -float('inf') if maximizing_player else float('inf') # Renamed best_value
        best_move_for_tt = None # Your original TT stores the move
        first_move_pvs = True # Original PVS flag

        for i, move_tuple in enumerate(ordered_moves): # Renamed move
            start_pos, end_pos = move_tuple # Renamed start, end
            current_piece_moved = board[start_pos[0]][start_pos[1]] # Piece on current board
            
            # Simulate the move
            child_board = self.simulate_move(board, start_pos, end_pos)
            
            # --- Increment count for child's position string key ---
            child_turn_after_move = self.opponent_color if maximizing_player else self.color
            child_key_str = generate_position_key(child_board, child_turn_after_move)
            position_counts_ref[child_key_str] = position_counts_ref.get(child_key_str, 0) + 1
            # ---------------------------------------------------------
            
            # LMR from your original code
            target_on_board = board[end_pos[0]][end_pos[1]] # Target on original board for tactical check
            is_tactical = (target_on_board is not None) or \
                          (isinstance(current_piece_moved, Pawn) and \
                           (end_pos[0] == 0 or end_pos[0] == ROWS - 1))
            
            reduction_lmr = 0
            if not is_tactical and i >= 2 and depth >= 2 : # Original LMR: i >= 3 and depth >= 4. Use simpler for test.
                                                          # Reverting to a common simpler LMR condition
                reduction_lmr = 1
            
            child_search_depth = depth - 1 - reduction_lmr # Renamed new_depth
            
            score_from_child = 0 # Renamed score
            if first_move_pvs:
                score_from_child = self.minimax(child_board, child_search_depth, not maximizing_player, 
                                                alpha, beta, position_counts_ref)
                first_move_pvs = False
            else: # PVS scout search (original structure)
                if maximizing_player:
                    score_from_child = self.minimax(child_board, child_search_depth, False, 
                                                    alpha, alpha + 1, position_counts_ref)
                    if score_from_child > alpha and score_from_child < beta: # Re-search
                        # Your original re-search logic:
                        score_from_child = self.minimax(child_board, child_search_depth, False, 
                                                        score_from_child, beta, position_counts_ref)
                else: # Minimizing player
                    score_from_child = self.minimax(child_board, child_search_depth, True, 
                                                    beta - 1, beta, position_counts_ref)
                    if score_from_child < beta and score_from_child > alpha: # Re-search
                        # Your original re-search logic:
                        score_from_child = self.minimax(child_board, child_search_depth, True, 
                                                        alpha, score_from_child, position_counts_ref)
            
            # --- Decrement count for child's position string key (backtrack) ---
            position_counts_ref[child_key_str] -= 1
            if position_counts_ref[child_key_str] == 0:
                del position_counts_ref[child_key_str]
            # ------------------------------------------------------------------

            if maximizing_player:
                if score_from_child > best_value_node:
                    best_value_node = score_from_child
                    best_move_for_tt = move_tuple
                alpha = max(alpha, best_value_node)
            else: # Minimizing player
                if score_from_child < best_value_node:
                    best_value_node = score_from_child
                    best_move_for_tt = move_tuple
                beta = min(beta, best_value_node)

            if beta <= alpha: # Original condition was beta <= alpha
                break 

        self.tt[board_key_tt] = (depth, best_value_node, best_move_for_tt)
        return best_value_node

    def make_move(self): # Original make_move structure
        overall_best_move = None 
        overall_best_value = -float('inf') 

        total_start_time = time.time()

        # --- OPTIMIZATION: Create ONE mutable copy of position_counts for the entire search of this turn ---
        # This dictionary will be passed by reference and modified by minimax and its children.
        current_game_turn_pos_counts = self.app.position_counts.copy()
        # self.app.position_history is not passed to minimax in this optimized version.
        # -------------------------------------------------------------------------------------------------

        for current_depth_iter in range(1, self.search_depth + 1):
            self.nodes_searched = 0
            iter_start_time = time.time()

            root_moves = self.get_all_moves(self.board, self.color)
            if not root_moves: return False

            ordered_root_moves = self.order_moves(self.board, root_moves, True) # AI is maximizing

            current_iter_best_move = None 
            current_iter_best_value = -float('inf')

            for move_tuple_root in ordered_root_moves: 
                start_root, end_root = move_tuple_root
                
                child_board_root = self.simulate_move(self.board, start_root, end_root)
                
                # --- Manage current_game_turn_pos_counts for this root move ---
                child_turn_root = self.opponent_color
                child_key_str_root = generate_position_key(child_board_root, child_turn_root)
                current_game_turn_pos_counts[child_key_str_root] = current_game_turn_pos_counts.get(child_key_str_root, 0) + 1
                # ------------------------------------------------------------
                
                eval_for_move = 0 
                # Original logic for penalizing (checking) threefold repetition at root:
                if current_game_turn_pos_counts[child_key_str_root] >= 3: # If this move leads to 3rd repetition
                    eval_for_move = 0 # Draw value
                else:
                    # Pass the MUTABLE current_game_turn_pos_counts by reference
                    eval_for_move = self.minimax(child_board_root, current_depth_iter - 1, False, # Opponent's turn
                                                 -float('inf'), float('inf'), 
                                                 current_game_turn_pos_counts) 
                
                # --- Backtrack current_game_turn_pos_counts for this root move ---
                current_game_turn_pos_counts[child_key_str_root] -= 1
                if current_game_turn_pos_counts[child_key_str_root] == 0:
                    del current_game_turn_pos_counts[child_key_str_root]
                # -----------------------------------------------------------------

                if eval_for_move > current_iter_best_value:
                    current_iter_best_value = eval_for_move
                    current_iter_best_move = move_tuple_root
            
            iter_time_val = time.time() - iter_start_time # Renamed iter_time
            
            # Update overall best if a move was found in this iteration
            if current_iter_best_move is not None:
                overall_best_move = current_iter_best_move
                overall_best_value = current_iter_best_value # CRITICAL: update overall_best_value from current iteration

            reported_value_ui = overall_best_value # Use the best value from completed depths
            if self.color == 'black':
                reported_value_ui = -overall_best_value
            
            # Original print and eval bar logic
            print(f"AI Depth {current_depth_iter}: {iter_time_val:.3f}s, AI nodes: {self.nodes_searched}, AI Eval: {reported_value_ui if isinstance(reported_value_ui, (int,float)) else 'N/A'}")

            if self.app:
                 # Pass the potentially modified reported_value_ui to lambda
                self.app.master.after(0, lambda v_ui=reported_value_ui: self.app.draw_eval_bar(v_ui))


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
                self.app.game_over = True
                self.app.game_result = ("repetition", None)
                if hasattr(self.app, 'turn_label') and self.app.turn_label:
                    self.app.turn_label.config(text="Draw by three-fold repetition!")
            return True
        return False

    # ====================================================
    # Hashing and Helper Methods (Original)
    # ====================================================
    # initialize_zobrist_table is above
    # board_hash is above

    def get_adjacent_squares(self, pos): # Original
        r_pos, c_pos = pos
        adj_squares = []
        for adj_r_loop in range(r_pos - 1, r_pos + 2):
            for adj_c_loop in range(c_pos - 1, c_pos + 2):
                if 0 <= adj_r_loop < ROWS and 0 <= adj_c_loop < COLS and \
                   (adj_r_loop, adj_c_loop) != (r_pos, c_pos):
                    adj_squares.append((adj_r_loop, adj_c_loop))
        return adj_squares

    def simulate_move(self, board, start, end): # Original
        new_board = copy_board(board)
        piece_to_move = new_board[start[0]][start[1]]
        if not piece_to_move: return new_board
        new_board = piece_to_move.move(new_board, start, end)
        check_evaporation(new_board)
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