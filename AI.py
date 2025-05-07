import time
from GameLogic import * # Assuming GameLogic.py, ROWS, COLS, etc. are available
import random # For Zobrist

class ChessBot:
    search_depth = 2
    CENTER_SQUARES = {(3, 3), (3, 4), (4, 3), (4, 4)}
    PIECE_VALUES = {
        Pawn: 100, Knight: 700, Bishop: 600,
        Rook: 500, Queen: 900, King: 100000
    }

    def __init__(self, board, color, app):
        self.board = board # This is a reference to the app's board
        self.color = color
        self.opponent_color = 'black' if color == 'white' else 'white' # Store opponent color
        self.app = app
        self.tt = {}
        self.nodes_searched = 0
        self.zobrist_table = self.initialize_zobrist_table()
        # Note: self.position_history and self.position_counts are not used by the AI directly anymore
        # for its internal search repetition checks. They are for the main game state.

    def initialize_zobrist_table(self): # Renamed from initialize_zobrist_table
        random.seed(42)
        table = {}
        # Use None.__class__ for the type of None, or just None if keys are consistent
        piece_classes_for_hash = [Pawn, Knight, Bishop, Rook, Queen, King, None.__class__]
        for r_idx in range(ROWS):
            for c_idx in range(COLS):
                for piece_class in piece_classes_for_hash:
                    # For piece_class being None.__class__, color should also be None
                    colors_for_hash = ['white', 'black'] if piece_class != None.__class__ else [None]
                    for piece_color_key in colors_for_hash:
                        table[(piece_class, piece_color_key, r_idx, c_idx)] = random.getrandbits(64)
        table[('turn', 'black')] = random.getrandbits(64) # Hash component for black's turn
        # Add other game state components like castling rights, en passant target square if needed
        return table

    def board_hash(self, board_to_hash, turn_color_for_hash): # Renamed from board_hash
        h = 0
        for r_idx in range(ROWS):
            for c_idx in range(COLS):
                piece = board_to_hash[r_idx][c_idx]
                # Ensure key construction is robust for None piece
                piece_type_key = type(piece) if piece else None.__class__
                color_key = piece.color if piece else None
                h ^= self.zobrist_table[(piece_type_key, color_key, r_idx, c_idx)]
        
        if turn_color_for_hash == 'black':
            h ^= self.zobrist_table[('turn', 'black')]
        # XOR other state components (castling, en passant) here
        return h

    # ====================================================
    # Threat Checking Methods (Kept from original)
    # ====================================================
    def is_in_explosion_threat(self, board, color):
        king_pos = None
        enemy_color = 'black' if color == 'white' else 'white'
        enemy_queens = []
        for r in range(ROWS):
            for c in range(COLS):
                piece = board[r][c]
                if isinstance(piece, King) and piece.color == color:
                    king_pos = (r, c)
                elif isinstance(piece, Queen) and piece.color == enemy_color:
                    enemy_queens.append((r, c))
        if not king_pos: return False
        for r_q, c_q in enemy_queens: # Renamed r, c to avoid clash
            queen_piece = board[r_q][c_q] # Get the piece object
            if not queen_piece: continue # Should not happen if list is built correctly
            for move in queen_piece.get_valid_moves(board, (r_q, c_q)):
                if max(abs(move[0] - king_pos[0]), abs(move[1] - king_pos[1])) == 1:
                    target_piece = board[move[0]][move[1]]
                    if target_piece and target_piece.color == color: # Queen must capture piece of king's color
                        return True
        return False

    def is_in_knight_evaporation_threat(self, board, color): # color is of knights being checked
        enemy_color_attacker = 'black' if color == 'white' else 'white'
        for r_k_check in range(ROWS): # Renamed r, c
            for c_k_check in range(COLS):
                piece_k_check = board[r_k_check][c_k_check]
                if isinstance(piece_k_check, Knight) and piece_k_check.color == color:
                    for dr_adj, dc_adj in DIRECTIONS['knight']: # Renamed dr, dc
                        nr_adj, nc_adj = r_k_check + dr_adj, c_k_check + dc_adj
                        if 0 <= nr_adj < ROWS and 0 <= nc_adj < COLS:
                            target_piece_adj = board[nr_adj][nc_adj]
                            if isinstance(target_piece_adj, Knight) and target_piece_adj.color == enemy_color_attacker:
                                return True
        return False

    # ====================================================
    # Position Evaluation Methods (Kept from original)
    # ====================================================
    def evaluate_board(self, board_eval, depth_eval, current_turn_eval): # Renamed params
        score = 0
        our_king_pos_eval = None
        enemy_king_pos_eval = None # Relative to self.color
        enemy_pieces_list_eval = []

        for r_ev in range(ROWS): # Renamed r, c
            for c_ev in range(COLS):
                piece_ev = board_eval[r_ev][c_ev]
                if not piece_ev: continue

                if isinstance(piece_ev, King):
                    if piece_ev.color == self.color: our_king_pos_eval = (r_ev, c_ev)
                    else: enemy_king_pos_eval = (r_ev, c_ev)
                    continue

                if piece_ev.color != self.color:
                    enemy_pieces_list_eval.append(((r_ev, c_ev), piece_ev)) # Store (pos, piece_obj)

                value = self.PIECE_VALUES.get(type(piece_ev), 0)
                if isinstance(piece_ev, Knight): # Knight mobility bonus
                    value += len(piece_ev.get_valid_moves(board_eval, (r_ev, c_ev))) * 10
                elif isinstance(piece_ev, Queen) and piece_ev.color == self.color: # OUR Queen's atomic threats
                    atomic_threats = 0
                    # Check if piece_ev is not None before calling get_valid_moves
                    for move_q_eval in piece_ev.get_valid_moves(board_eval, (r_ev, c_ev)):
                        for adj_r_q, adj_c_q in self.get_adjacent_squares(move_q_eval):
                            adj_piece_q = board_eval[adj_r_q][adj_c_q]
                            if isinstance(adj_piece_q, King) and adj_piece_q.color != self.color: # Enemy king
                                atomic_threats += 1
                                break 
                    value += atomic_threats * 20
                score += value if piece_ev.color == self.color else -value

        if not enemy_king_pos_eval: return float('inf')
        if not our_king_pos_eval: return float('-inf')

        threat_score_on_us = 0
        if our_king_pos_eval:
            for (pos_enemy_r, pos_enemy_c), enemy_p_obj in enemy_pieces_list_eval:
                 if enemy_p_obj and our_king_pos_eval in enemy_p_obj.get_valid_moves(board_eval, (pos_enemy_r, pos_enemy_c)):
                    threat_score_on_us += 200
        
        # Explosion threat against OUR king
        is_expl_threat = self.is_in_explosion_threat(board_eval, self.color)
        
        # Score is always from self.color's perspective
        final_eval_score = score - threat_score_on_us - (250 if is_expl_threat else 0) # Made penalty higher
        return int(final_eval_score)

    # ====================================================
    # Move Ordering Methods (Kept from original, ensure board_hash is passed if used)
    # ====================================================
    def evaluate_move(self, board_ord, move_ord): # Renamed params
        start_ord, end_ord = move_ord
        piece_ord = board_ord[start_ord[0]][start_ord[1]]
        target_ord = board_ord[end_ord[0]][end_ord[1]]
        score = 0
        if target_ord: # Capture
            score = 1000 + self.PIECE_VALUES.get(type(target_ord), 0)
            # Simple MVV:
            # score -= self.PIECE_VALUES.get(type(piece_ord), 0) / 10 
        if end_ord in self.CENTER_SQUARES: score += 50
        if isinstance(piece_ord, Pawn):
            score += (ROWS - 1 - end_ord[0] if piece_ord.color == "white" else end_ord[0]) * 5 # Simpler advancement
            # Promotion bonus
            if end_ord[0] == 0 or end_ord[0] == ROWS -1 : score += self.PIECE_VALUES[Queen]
        return score
    
    def order_moves(self, board_to_order, moves_to_order, board_hash_for_tt, maximizing_player_flag=True): # Added board_hash
        if not moves_to_order: return moves_to_order
        
        tt_entry_ord = self.tt.get(board_hash_for_tt)
        best_tt_move_ord = None
        if tt_entry_ord and len(tt_entry_ord) > 2 and tt_entry_ord[2] is not None: # tt_entry[2] is move
            best_tt_move_ord = tt_entry_ord[2]

        ordered_list = []
        if best_tt_move_ord and best_tt_move_ord in moves_to_order:
            ordered_list.append(best_tt_move_ord)
            try:
                moves_to_order.remove(best_tt_move_ord)
            except ValueError: pass # Should not happen

        heuristic_scored_moves = []
        for m_to_score in moves_to_order:
            heuristic_scored_moves.append((self.evaluate_move(board_to_order, m_to_score), m_to_score))
        
        # Sort remaining moves by heuristic (higher scores better)
        heuristic_scored_moves.sort(key=lambda x: x[0], reverse=True)
        
        ordered_list.extend([m_tuple for _, m_tuple in heuristic_scored_moves])
        return ordered_list

    # ====================================================
    # Search Methods (Minimax & Move Selection)
    # ====================================================
    def minimax(self, board_mm, depth_mm, maximizing_player_mm, 
                alpha_mm, beta_mm, 
                current_board_hash_mm, # Zobrist hash of board_mm + turn
                path_hashes_set_mm):   # Set of Zobrist hashes in current search path
        self.nodes_searched += 1
        original_alpha_for_tt_flag = alpha_mm # For TT flag storage

        # 1. Repetition check for current search path
        if current_board_hash_mm in path_hashes_set_mm:
            return 0  # Draw by repetition in this path

        # 2. Transposition Table Lookup
        tt_entry = self.tt.get(current_board_hash_mm)
        if tt_entry and tt_entry[0] >= depth_mm: # tt_entry = (depth, score, best_move)
            # Add TT flags (EXACT, LOWERBOUND, UPPERBOUND) for more precise pruning
            return tt_entry[1] # Return stored score

        # 3. Terminal Node Checks (Stalemate, Depth Limit)
        current_player_at_node = self.color if maximizing_player_mm else self.opponent_color
        if is_stalemate(board_mm, current_player_at_node): # GameLogic.is_stalemate
            return 0
        if depth_mm == 0:
            # evaluate_board returns score from self.color's perspective.
            # If current node is minimizer (opponent's turn), this score is what AI gets.
            # If current node is maximizer (AI's turn), this score is what AI gets.
            # So, no negation needed here if minimax handles the score correctly based on maximizing_player_mm.
            return self.evaluate_board(board_mm, depth_mm, current_player_at_node)

        # 4. Null Move Pruning
        if depth_mm >= 3 and not is_in_check(board_mm, current_player_at_node): # GameLogic.is_in_check
            # Hash for state after null move (turn flips)
            hash_after_null = current_board_hash_mm ^ self.zobrist_table[('turn', 'black')]
            path_hashes_set_mm.add(hash_after_null) # Add to path for recursive check

            # Null move: opponent plays. Score returned is from opponent's perspective.
            # Minimax called with maximizing_player_flag = not maximizing_player_mm
            # Alpha/beta window is passed normally for opponent.
            null_move_score = self.minimax(board_mm, depth_mm - 1 - 1, # R=1 reduction
                                           not maximizing_player_mm,
                                           alpha_mm, beta_mm, # Pass original alpha/beta
                                           hash_after_null,
                                           path_hashes_set_mm)
            path_hashes_set_mm.remove(hash_after_null) # Backtrack

            # Interpretation of null_move_score:
            # If current node is Max (AI), null_move_score is what Min opponent achieved.
            # If this score is >= beta_mm, it means opponent achieved a score good enough for them
            # to cause a beta cutoff for us.
            if maximizing_player_mm:
                if null_move_score >= beta_mm: return beta_mm # Or null_move_score
            else: # Current node is Min (Opponent)
                  # null_move_score is what Max (AI) achieved in sub-search.
                  # If this score is <= alpha_mm, it means AI achieved a score bad enough for them
                  # to cause an alpha cutoff for us (the Min node).
                if null_move_score <= alpha_mm: return alpha_mm # Or null_move_score
        
        # 5. Move Generation & Loop
        moves_list_mm = self.get_all_moves(board_mm, current_player_at_node)
        if not moves_list_mm: # No legal moves
            if is_in_check(board_mm, current_player_at_node): # Checkmate against current_player_at_node
                # Score is from current_player_at_node's perspective: very bad.
                # Depth adjustment prefers faster mates (found at higher depth_mm).
                return -100000 + (self.search_depth - depth_mm) 
            else:
                return 0 # Stalemate

        ordered_moves = self.order_moves(board_mm, moves_list_mm, current_board_hash_mm, maximizing_player_mm)
        
        best_score_node = -float('inf') if maximizing_player_mm else float('inf')
        best_move_node = None
        
        path_hashes_set_mm.add(current_board_hash_mm) # Add current state hash before iterating children

        for i, move_tuple in enumerate(ordered_moves):
            start_pos, end_pos = move_tuple
            sim_board = self.simulate_move(board_mm, start_pos, end_pos) # Uses copy_board
            # Hash for child node (turn flips)
            hash_for_child = self.board_hash(sim_board, self.opponent_color if maximizing_player_mm else self.color)
            
            score_child = 0
            # PVS logic - maintaining original structure as much as possible
            if i == 0: # First move (PV node)
                score_child = self.minimax(sim_board, depth_mm - 1, not maximizing_player_mm,
                                           alpha_mm, beta_mm, hash_for_child, path_hashes_set_mm)
            else: # Scout search for other moves
                reduction_lmr = 0
                is_capture = board_mm[end_pos[0]][end_pos[1]] is not None # Check original board
                if depth_mm >= 2 and i >= 2 and not is_capture: # Simplified LMR
                    reduction_lmr = 1
                
                # Scout search window logic needs to be careful with alpha/beta for non-negamax PVS
                if maximizing_player_mm:
                    score_child = self.minimax(sim_board, depth_mm - 1 - reduction_lmr, False, # Min child
                                               alpha_mm, alpha_mm + 1, # Test if score > alpha
                                               hash_for_child, path_hashes_set_mm)
                    if score_child > alpha_mm and score_child < beta_mm: # If promising, re-search
                        score_child = self.minimax(sim_board, depth_mm - 1, False, # Full depth
                                                   score_child, beta_mm, # Narrowed window
                                                   hash_for_child, path_hashes_set_mm)
                else: # Minimizing player current node
                    score_child = self.minimax(sim_board, depth_mm - 1 - reduction_lmr, True, # Max child
                                               beta_mm - 1, beta_mm,   # Test if score < beta
                                               hash_for_child, path_hashes_set_mm)
                    if score_child < beta_mm and score_child > alpha_mm: # If promising, re-search
                        score_child = self.minimax(sim_board, depth_mm - 1, True, # Full depth
                                                   alpha_mm, score_child, # Narrowed window
                                                   hash_for_child, path_hashes_set_mm)
            
            if maximizing_player_mm:
                if score_child > best_score_node:
                    best_score_node = score_child
                    best_move_node = move_tuple
                alpha_mm = max(alpha_mm, best_score_node)
            else: # Minimizing player
                if score_child < best_score_node:
                    best_score_node = score_child
                    best_move_node = move_tuple
                beta_mm = min(beta_mm, best_score_node)

            if alpha_mm >= beta_mm:
                break # Alpha-beta cutoff
        
        path_hashes_set_mm.remove(current_board_hash_mm) # Backtrack

        # Store in TT: score is from perspective of player at current_board_hash_mm
        self.tt[current_board_hash_mm] = (depth_mm, best_score_node, best_move_node)
        return best_score_node


    def make_move(self): # Root search function
        overall_best_move_from_id = None
        overall_best_eval_for_ai = -float('inf') # AI is always maximizing its own score

        total_start_time = time.time()
        # Hash of the current actual game board (AI's turn, so self.color)
        root_board_hash_val = self.board_hash(self.board, self.color)

        # For threefold repetition check at root against app's main game history
        current_game_pos_history_strings = self.app.position_history.copy()
        current_game_pos_counts_strings = self.app.position_counts.copy()

        for current_search_depth_iter in range(1, self.search_depth + 1):
            self.nodes_searched = 0
            iter_start_time = time.time()
            
            root_moves = self.get_all_moves(self.board, self.color)
            if not root_moves: return False

            # Order root moves (AI is maximizing_player = True)
            ordered_root_moves = self.order_moves(self.board, root_moves, root_board_hash_val, True)

            current_iter_best_move = None
            current_iter_best_eval = -float('inf') # AI wants to maximize this

            alpha_at_root = -float('inf')
            beta_at_root = float('inf')
            # path_hashes_set for recursive calls (starts empty for children of root)
            # Root itself isn't part of its children's path repetition check set initially.
            path_hashes_for_children = set() 

            for i, move_tuple_at_root in enumerate(ordered_root_moves):
                # Simulate the move on a copy
                sim_board_after_root_move = self.simulate_move(self.board, move_tuple_at_root[0], move_tuple_at_root[1])
                
                # Check global threefold repetition for this simulated state (using string keys)
                # This is the original repetition check logic from root.
                key_for_sim_board_str = generate_position_key(sim_board_after_root_move, self.opponent_color)
                temp_pos_counts = current_game_pos_counts_strings.copy()
                temp_pos_counts[key_for_sim_board_str] = temp_pos_counts.get(key_for_sim_board_str, 0) + 1
                if temp_pos_counts[key_for_sim_board_str] >= 3:
                    eval_for_this_move = 0 # Draw by repetition
                else:
                    # Hash for the board state AFTER AI's move (so it's opponent's turn)
                    hash_for_sim_board = self.board_hash(sim_board_after_root_move, self.opponent_color)
                    
                    # Call minimax for opponent's turn (maximizing_player_flag = False for opponent)
                    # The score returned is from opponent's perspective.
                    eval_for_this_move = self.minimax(
                        sim_board_after_root_move, 
                        current_search_depth_iter - 1, 
                        False, # Opponent's turn (minimizer from AI's POV)
                        alpha_at_root, beta_at_root, 
                        hash_for_sim_board,
                        path_hashes_for_children # Pass empty set for first recursive level
                    )
                
                # eval_for_this_move is the score the Minimizing opponent achieved.
                # The AI (Maximizer at root) wants to choose the move where opponent's outcome
                # (which is this eval_for_this_move) is highest.
                if eval_for_this_move > current_iter_best_eval:
                    current_iter_best_eval = eval_for_this_move
                    current_iter_best_move = move_tuple_at_root
                
                alpha_at_root = max(alpha_at_root, current_iter_best_eval)
                # No beta cutoff for root moves typically

            # After checking all root moves for this depth
            if current_iter_best_move is not None: # If a move was found for this depth
                overall_best_move_from_id = current_iter_best_move
                overall_best_eval_for_ai = current_iter_best_eval

            iter_time_taken = time.time() - iter_start_time
            
            # Eval bar should show score from White's perspective consistently.
            # overall_best_eval_for_ai is from AI's (self.color) perspective.
            reported_val_for_bar_ui = overall_best_eval_for_ai
            if self.color == 'black': # If AI is black, negate its eval for white's perspective for bar
                reported_val_for_bar_ui = -overall_best_eval_for_ai
            
            eval_to_print_str_ui = f"{overall_best_eval_for_ai:.0f}"
            if overall_best_eval_for_ai == -float('inf'): eval_to_print_str_ui = "-inf"
            elif overall_best_eval_for_ai == float('inf'): eval_to_print_str_ui = "+inf"

            print(f"AI Depth {current_search_depth_iter}: {iter_time_taken:.3f}s, Nodes: {self.nodes_searched}, Eval: {eval_to_print_str_ui} (AI: {self.color})")
            
            if self.app:
                final_ui_report_val = reported_val_for_bar_ui
                if final_ui_report_val == -float('inf'): final_ui_report_val = -9900 
                elif final_ui_report_val == float('inf'): final_ui_report_val = 9900
                self.app.master.after(0, lambda v_bar=final_ui_report_val: self.app.draw_eval_bar(v_bar))

        total_time_taken = time.time() - total_start_time
        print(f"AI Total time: {total_time_taken:.3f}s")

        if overall_best_move_from_id:
            start_final, end_final = overall_best_move_from_id
            moving_piece_final = self.board[start_final[0]][start_final[1]] # self.board is app.board
            self.board = moving_piece_final.move(self.board, start_final, end_final) # GameLogic.move
            check_evaporation(self.board) # GameLogic global check
            
            next_turn_color_key = self.opponent_color
            new_main_key_str = generate_position_key(self.board, next_turn_color_key)
            self.app.position_history.append(new_main_key_str)
            self.app.position_counts[new_main_key_str] = self.app.position_counts.get(new_main_key_str, 0) + 1
            if self.app.position_counts[new_main_key_str] >= 3:
                self.app.game_over = True; self.app.game_result = ("repetition", None)
                if hasattr(self.app, 'turn_label'): self.app.turn_label.config(text="Draw by three-fold repetition!")
            return True
        else:
            return False # No move found

    # simulate_move, get_all_moves, get_adjacent_squares are kept as they were in your original code
    def simulate_move(self, board, start, end):
        new_board = copy_board(board)
        piece = new_board[start[0]][start[1]]
        if not piece: return new_board # Should not happen if start is from a valid piece
        new_board = piece.move(new_board, start, end) # Uses GameLogic piece.move
        check_evaporation(new_board) # Uses GameLogic check_evaporation
        return new_board

    def get_all_moves(self, board_param, color_param): # Renamed params
        legal_moves_found = []
        for r_param in range(ROWS):
            for c_param in range(COLS):
                piece_param = board_param[r_param][c_param]
                if piece_param and piece_param.color == color_param:
                    for end_pos_param in piece_param.get_valid_moves(board_param, (r_param, c_param)):
                        if validate_move(board_param, color_param, (r_param, c_param), end_pos_param): # GameLogic
                            legal_moves_found.append(((r_param, c_param), end_pos_param))
        return legal_moves_found

    def get_adjacent_squares(self, pos): # Kept original name
        r_pos, c_pos = pos # Renamed r,c
        adj_sq_list = []
        for adj_r_iter in range(r_pos - 1, r_pos + 2): # Renamed adj_r, adj_c
            for adj_c_iter in range(c_pos - 1, c_pos + 2):
                if 0 <= adj_r_iter < ROWS and 0 <= adj_c_iter < COLS and (adj_r_iter, adj_c_iter) != (r_pos, c_pos):
                    adj_sq_list.append((adj_r_iter, adj_c_iter))
        return adj_sq_list