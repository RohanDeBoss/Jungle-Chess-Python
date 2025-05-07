import time
from GameLogic import *
import random

#v0.3 - Fully working, slow ai with 6s for depth 4, but low node count

class ChessBot:
    search_depth = 2
    CENTER_SQUARES = {(3, 3), (3, 4), (4, 3), (4, 4)}
    PIECE_VALUES = {
        Pawn: 100, Knight: 700, Bishop: 600,
        Rook: 500, Queen: 900, King: 100000 # Checkmate/King missing should override material
    }
    # Ensure King has a very high value, but checkmate detection is better
    MATE_SCORE = 100000 # Base score for checkmate
    DRAW_SCORE = 0 # Score for draws (stalemate, repetition)


    def __init__(self, board, color, app):
        self.board = board
        self.color = color
        self.opponent_color = 'black' if color == 'white' else 'white'
        self.app = app
        self.tt = {}
        self.nodes_searched = 0
        self.zobrist_table = self.initialize_zobrist_table()

    def initialize_zobrist_table(self):
        random.seed(42)
        table = {}
        piece_classes_for_hash = [Pawn, Knight, Bishop, Rook, Queen, King, None.__class__]
        for r_idx in range(ROWS):
            for c_idx in range(COLS):
                for piece_class in piece_classes_for_hash:
                    colors_for_hash = ['white', 'black'] if piece_class != None.__class__ else [None]
                    for piece_color_key in colors_for_hash:
                        table[(piece_class, piece_color_key, r_idx, c_idx)] = random.getrandbits(64)
        table[('turn', 'black')] = random.getrandbits(64)
        return table
    
    # Hashing Function (Zobrist) - Use the Zobrist table to compute a hash for the board state
    def board_hash(self, board_to_hash, turn_color_for_hash):
        h = 0
        for r_idx in range(ROWS):
            for c_idx in range(COLS):
                piece = board_to_hash[r_idx][c_idx]
                piece_type_key = type(piece) if piece else None.__class__
                color_key = piece.color if piece else None
                # Ensure the key components exist in the table
                if (piece_type_key, color_key, r_idx, c_idx) in self.zobrist_table:
                     h ^= self.zobrist_table[(piece_type_key, color_key, r_idx, c_idx)]
                # else: print(f"Warning: Zobrist key missing for {piece_type_key}, {color_key} at ({r_idx},{c_idx})") # Debug if needed
        
        if turn_color_for_hash == 'black':
            h ^= self.zobrist_table[('turn', 'black')]
        return h

    # --- Threat Checking Methods (Keep originals) ---
    def is_in_explosion_threat(self, board, color):
        king_pos = None
        enemy_color = 'black' if color == 'white' else 'white'
        enemy_queens = []
        for r in range(ROWS):
            for c_in_loop in range(COLS):
                piece = board[r][c_in_loop]
                if isinstance(piece, King) and piece.color == color:
                    king_pos = (r, c_in_loop)
                elif isinstance(piece, Queen) and piece.color == enemy_color:
                    enemy_queens.append((r, c_in_loop))
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
        enemy_color = 'black' if color == 'white' else 'white'
        for r in range(ROWS):
            for c_in_loop in range(COLS):
                piece = board[r][c_in_loop]
                if isinstance(piece, Knight) and piece.color == color:
                    for dr, dc in DIRECTIONS['knight']:
                        nr, nc = r + dr, c_in_loop + dc
                        if 0 <= nr < ROWS and 0 <= nc < COLS:
                            target_piece = board[nr][nc]
                            if isinstance(target_piece, Knight) and target_piece.color == enemy_color:
                                return True
        return False

    # --- Evaluation (Keep original features, ensure it's AI-centric) ---
    def evaluate_board(self, board_eval, depth_eval, current_turn_eval):
        # Returns score from the perspective of self.color (the AI bot)
        score = 0
        our_king_pos_eval = None
        enemy_king_pos_eval = None 
        enemy_pieces_list_eval = []

        for r_eval in range(ROWS):
            for c_eval in range(COLS):
                piece_eval = board_eval[r_eval][c_eval]
                if not piece_eval: continue

                if isinstance(piece_eval, King):
                    if piece_eval.color == self.color: our_king_pos_eval = (r_eval, c_eval)
                    else: enemy_king_pos_eval = (r_eval, c_eval)
                    continue

                if piece_eval.color != self.color:
                    enemy_pieces_list_eval.append(((r_eval, c_eval), piece_eval))

                value = self.PIECE_VALUES.get(type(piece_eval), 0)
                if isinstance(piece_eval, Knight):
                    # Check piece_eval exists before calling method
                    if piece_eval: value += len(piece_eval.get_valid_moves(board_eval, (r_eval, c_eval))) * 10
                elif isinstance(piece_eval, Queen) and piece_eval.color == self.color: # Only OUR Queen's threats add bonus
                    atomic_threats = 0
                    if piece_eval: # Check exists
                        for move_q in piece_eval.get_valid_moves(board_eval, (r_eval, c_eval)):
                            for adj_r_q, adj_c_q in self.get_adjacent_squares(move_q):
                                adj_piece_q = board_eval[adj_r_q][adj_c_q]
                                if isinstance(adj_piece_q, King) and adj_piece_q.color != self.color:
                                    atomic_threats += 1
                                    break 
                        value += atomic_threats * 20
                
                score += value if piece_eval.color == self.color else -value

        # If opponent king is missing, it's a win (very high score)
        if not enemy_king_pos_eval: return self.MATE_SCORE 
        # If our king is missing, it's a loss (very low score) - should be caught by game over
        if not our_king_pos_eval: return -self.MATE_SCORE 

        threat_score_on_us = 0
        if our_king_pos_eval:
            for (pos_enemy_r, pos_enemy_c), enemy_p_obj in enemy_pieces_list_eval:
                 if enemy_p_obj and our_king_pos_eval in enemy_p_obj.get_valid_moves(board_eval, (pos_enemy_r, pos_enemy_c)):
                    threat_score_on_us += 200
        
        is_expl_threat_on_us = self.is_in_explosion_threat(board_eval, self.color)
        
        final_eval_score = score - threat_score_on_us - (250 if is_expl_threat_on_us else 0)
        return int(final_eval_score)

    # --- Move Ordering (Keep original heuristic) ---
    def evaluate_move(self, board, move):
        start, end = move
        piece = board[start[0]][start[1]]
        target = board[end[0]][end[1]]
        score = 0 
        if target:
            score = 1000 + self.PIECE_VALUES.get(type(target), 0)
        if end in self.CENTER_SQUARES and not target:
             score += 50
        if isinstance(piece, Pawn):
            advancement = (ROWS - 1 - end[0]) if piece.color == "white" else end[0]
            score += advancement * 5
            if end[0] == 0 or end[0] == ROWS -1 :
                score += self.PIECE_VALUES[Queen]
        return score
    
    def order_moves(self, board, moves, board_hash_for_tt, maximizing_player=True):
        if not moves: return moves
        tt_entry = self.tt.get(board_hash_for_tt)
        best_tt_move = None
        if tt_entry and len(tt_entry) > 2 and tt_entry[2] is not None:
            best_tt_move = tt_entry[2]
        ordered_list = []
        if best_tt_move and best_tt_move in moves:
            ordered_list.append(best_tt_move)
            try: moves.remove(best_tt_move)
            except ValueError: pass
        heuristic_scored_moves = []
        for m in moves:
            heuristic_scored_moves.append((self.evaluate_move(board, m), m))
        heuristic_scored_moves.sort(key=lambda x: x[0], reverse=True)
        ordered_list.extend([m_tuple for _, m_tuple in heuristic_scored_moves])
        return ordered_list

    # --- Search (Minimax with explicit Max/Min, corrected repetition/mate logic) ---
    def minimax(self, board, depth, maximizing_player, alpha, beta,
                # Pass the dictionary of game position counts for 3-fold check
                game_position_counts, # Dict[str, int] from app history
                path_hashes_set # Set of hashes in current search path
               ):
        self.nodes_searched += 1
        
        current_player_node = self.color if maximizing_player else self.opponent_color
        current_board_hash = self.board_hash(board, current_player_node)

        # 1. Repetition Checks
        # Check 1: Repetition within the current search path
        if current_board_hash in path_hashes_set:
            return self.DRAW_SCORE # Draw by cycle in search

        # Check 2: Threefold repetition based on actual game history
        current_key_str = generate_position_key(board, current_player_node)
        # We need the count *before* adding this instance, so check >= 2
        if game_position_counts.get(current_key_str, 0) >= 2:
            return self.DRAW_SCORE # Draw by threefold repetition

        # 2. Transposition Table Lookup
        tt_entry = self.tt.get(current_board_hash)
        if tt_entry and tt_entry[0] >= depth:
            return tt_entry[1] 

        # 3. Terminal Node Checks (Stalemate/Checkmate BEFORE depth limit if possible)
        # Generate moves first to check for terminal state
        moves = self.get_all_moves(board, current_player_node)
        if not moves:
            if is_in_check(board, current_player_node): # GameLogic.is_in_check
                # Checkmate against current_player_node. Score is very bad for them.
                # Return score from the perspective of the *parent* node (who delivered mate)
                # This score should be relative to self.color. A high score is good for AI.
                # Mate score adjusted by depth (faster mate is better).
                # (self.search_depth - depth) = plies from root to this node.
                # We add this to make MATE_SCORE slightly worse if found deeper.
                mate_delivered_score = self.MATE_SCORE - (self.search_depth - depth)
                # If AI (Max) delivered mate (current node is Min), return high score.
                # If Opponent (Min) delivered mate (current node is Max), return low score.
                return mate_delivered_score if not maximizing_player else -mate_delivered_score
            else:
                return self.DRAW_SCORE # Stalemate

        # 4. Depth Limit Check
        if depth == 0:
            # Return evaluation score from AI's (self.color's) perspective
            return self.evaluate_board(board, depth, current_player_node)

        # 5. Null Move Pruning (Optional but kept from original)
        # Ensure score comparison is correct for explicit Max/Min nodes
        if depth >= 3 and not is_in_check(board, current_player_node):
            hash_after_null = current_board_hash ^ self.zobrist_table[('turn', 'black')]
            # Pass game_position_counts down, but add hash_after_null to path set copy
            temp_path_hashes_null = path_hashes_set.copy()
            temp_path_hashes_null.add(hash_after_null)

            # Recursive call for opponent. Score is from opponent's perspective.
            null_score = self.minimax(board, depth - 1 - 1, # R=1
                                      not maximizing_player,
                                      alpha, beta, # Pass window normally
                                      game_position_counts, # Pass main game counts
                                      temp_path_hashes_null) # Pass path set
            
            if maximizing_player: # We are Max, null_score is what Min achieved. Low is good for Min.
                if null_score >= beta: return beta # If Min couldn't force below beta, prune.
            else: # We are Min, null_score is what Max achieved. High is good for Max.
                if null_score <= alpha: return alpha # If Max couldn't force above alpha, prune.
        
        # 6. Iterate through moves (PVS)
        ordered_moves = self.order_moves(board, moves, current_board_hash, maximizing_player)
        best_node_score = -float('inf') if maximizing_player else float('inf')
        best_node_move = None
        
        path_hashes_set.add(current_board_hash) # Add current hash to path before exploring children

        for i, move_tuple in enumerate(ordered_moves):
            start_pos, end_pos = move_tuple
            child_board = self.simulate_move(board, start_pos, end_pos)
            child_turn = self.opponent_color if maximizing_player else self.color
            child_hash = self.board_hash(child_board, child_turn)
            
            # Update game_position_counts for the child node simulation
            child_game_pos_counts = game_position_counts.copy()
            child_key_str = generate_position_key(child_board, child_turn)
            child_game_pos_counts[child_key_str] = child_game_pos_counts.get(child_key_str, 0) + 1

            score_child = 0
            # PVS logic (explicit Max/Min)
            if i == 0: # PV node
                score_child = self.minimax(child_board, depth - 1, not maximizing_player,
                                           alpha, beta, 
                                           child_game_pos_counts, # Pass updated counts
                                           path_hashes_set.copy()) # Pass copy of path set
            else: # Scout nodes
                reduction = 0
                is_capture = board[end_pos[0]][end_pos[1]] is not None
                if depth >= 2 and i >= 2 and not is_capture: reduction = 1

                if maximizing_player: # We are Max, child is Min
                    # Scout with (alpha, alpha+1) window for Min child
                    score_child = self.minimax(child_board, depth - 1 - reduction, False,
                                               alpha, alpha + 1,
                                               child_game_pos_counts, path_hashes_set.copy())
                    if score_child > alpha and score_child < beta: # Re-search if promising
                        score_child = self.minimax(child_board, depth - 1, False,
                                                   score_child, beta, # Use score as new alpha for Min
                                                   child_game_pos_counts, path_hashes_set.copy())
                else: # We are Min, child is Max
                     # Scout with (beta-1, beta) window for Max child
                    score_child = self.minimax(child_board, depth - 1 - reduction, True,
                                               beta - 1, beta,
                                               child_game_pos_counts, path_hashes_set.copy())
                    if score_child < beta and score_child > alpha: # Re-search if promising
                        score_child = self.minimax(child_board, depth - 1, True,
                                                   alpha, score_child, # Use score as new beta for Max
                                                   child_game_pos_counts, path_hashes_set.copy())

            # Update best score based on Max/Min player
            if maximizing_player:
                if score_child > best_node_score:
                    best_node_score = score_child
                    best_node_move = move_tuple
                alpha = max(alpha, best_node_score)
            else: # Minimizing player
                if score_child < best_node_score:
                    best_node_score = score_child
                    best_node_move = move_tuple
                beta = min(beta, best_node_score)

            if alpha >= beta: break # Cutoff

        path_hashes_set.remove(current_board_hash) # Backtrack

        self.tt[current_board_hash] = (depth, best_node_score, best_node_move)
        return best_node_score

    def make_move(self):
        overall_best_move = None
        overall_best_eval_for_ai = -float('inf')

        total_start = time.time() # Define total_start here
        root_board_hash = self.board_hash(self.board, self.color)

        # Get game history counts from app ONCE for root checks
        root_game_pos_counts = self.app.position_counts.copy()

        for current_search_depth in range(1, self.search_depth + 1):
            self.nodes_searched = 0
            iter_start_time = time.time()
            
            root_moves = self.get_all_moves(self.board, self.color)
            if not root_moves: return False

            ordered_root_moves = self.order_moves(self.board, root_moves, root_board_hash, True)

            current_iter_best_move = None
            current_iter_best_eval = -float('inf')

            alpha_root = -float('inf')
            beta_root = float('inf')
            root_path_hashes = set() # Path hashes for children of root

            for i, move_tuple_root in enumerate(ordered_root_moves):
                sim_board_child = self.simulate_move(self.board, move_tuple_root[0], move_tuple_root[1])
                
                # Create the position counts dictionary as it would be AFTER this move
                child_game_pos_counts = root_game_pos_counts.copy()
                child_key_str = generate_position_key(sim_board_child, self.opponent_color)
                child_game_pos_counts[child_key_str] = child_game_pos_counts.get(child_key_str, 0) + 1

                eval_for_this_move = 0
                # Check repetition using the updated counts BEFORE calling minimax
                if child_game_pos_counts[child_key_str] >= 3:
                    eval_for_this_move = self.DRAW_SCORE # Draw by repetition
                else:
                    child_hash = self.board_hash(sim_board_child, self.opponent_color)
                    
                    # Call minimax for opponent's turn (Min node from AI's perspective)
                    eval_for_this_move = self.minimax(
                        sim_board_child, 
                        current_search_depth - 1, 
                        False, # Opponent's turn (Min node)
                        alpha_root, beta_root, 
                        child_game_pos_counts, # Pass the updated game counts
                        root_path_hashes.copy() # Pass empty set for this branch start
                    )
                
                # AI (root is Max node) wants to maximize the score returned by the Min node child
                if eval_for_this_move > current_iter_best_eval:
                    current_iter_best_eval = eval_for_this_move
                    current_iter_best_move = move_tuple_root
                
                alpha_root = max(alpha_root, current_iter_best_eval)

            # Update overall best move if a move was found in this iteration
            if current_iter_best_move is not None:
                overall_best_move = current_iter_best_move
                overall_best_eval_for_ai = current_iter_best_eval

            iter_time = time.time() - iter_start_time # Correct variable name
            
            reported_val_for_bar = overall_best_eval_for_ai 
            if self.color == 'black': reported_val_for_bar = -overall_best_eval_for_ai
            
            eval_to_print_str = f"{overall_best_eval_for_ai:.0f}"
            if overall_best_eval_for_ai == -float('inf'): eval_to_print_str = "-inf"
            elif overall_best_eval_for_ai == float('inf'): eval_to_print_str = "+inf"
            elif overall_best_eval_for_ai == self.MATE_SCORE - (self.search_depth - current_search_depth): eval_to_print_str = f"MATE_IN_{(self.search_depth - (current_search_depth -1)) // 2}" # Approx mate plies
            elif overall_best_eval_for_ai == -self.MATE_SCORE + (self.search_depth - current_search_depth): eval_to_print_str = f"MATED_IN_{(self.search_depth - (current_search_depth -1)) // 2}"

            print(f"AI Depth {current_search_depth}: {iter_time:.3f}s, Nodes: {self.nodes_searched}, Eval: {eval_to_print_str} (AI: {self.color})")
            
            if self.app:
                ui_report_val = reported_val_for_bar
                if ui_report_val <= -self.MATE_SCORE + self.search_depth : ui_report_val = -9900 
                elif ui_report_val >= self.MATE_SCORE - self.search_depth : ui_report_val = 9900
                self.app.master.after(0, lambda v_bar=ui_report_val: self.app.draw_eval_bar(v_bar))

        total_time_taken = time.time() - total_start # Correct variable name
        print(f"AI Total time: {total_time_taken:.3f}s")

        if overall_best_move:
            start_final, end_final = overall_best_move
            moving_piece_final = self.board[start_final[0]][start_final[1]]
            self.board = moving_piece_final.move(self.board, start_final, end_final)
            check_evaporation(self.board)
            
            next_turn_color_key = self.opponent_color
            new_main_key_str = generate_position_key(self.board, next_turn_color_key)
            self.app.position_history.append(new_main_key_str)
            self.app.position_counts[new_main_key_str] = self.app.position_counts.get(new_main_key_str, 0) + 1
            if self.app.position_counts[new_main_key_str] >= 3:
                self.app.game_over = True; self.app.game_result = ("repetition", None)
                if hasattr(self.app, 'turn_label'): self.app.turn_label.config(text="Draw by three-fold repetition!")
            return True
        else:
            return False

    # --- Other helpers (simulate_move, get_all_moves, get_adjacent_squares) ---
    def simulate_move(self, board, start, end):
        new_board = copy_board(board)
        piece = new_board[start[0]][start[1]]
        if not piece: return new_board
        new_board = piece.move(new_board, start, end)
        check_evaporation(new_board)
        return new_board

    def get_all_moves(self, board, color):
        legal_moves = []
        for r in range(ROWS):
            for c_loop in range(COLS):
                piece = board[r][c_loop]
                if piece and piece.color == color:
                    for end_pos in piece.get_valid_moves(board, (r, c_loop)):
                        if validate_move(board, color, (r, c_loop), end_pos):
                            legal_moves.append(((r, c_loop), end_pos))
        return legal_moves

    def get_adjacent_squares(self, pos):
        r_pos_adj, c_pos_adj = pos
        adj_sq_list = []
        for adj_r in range(r_pos_adj - 1, r_pos_adj + 2):
            for adj_c in range(c_pos_adj - 1, c_pos_adj + 2):
                if 0 <= adj_r < ROWS and 0 <= adj_c < COLS and (adj_r, adj_c) != (r_pos_adj, c_pos_adj):
                    adj_sq_list.append((adj_r, adj_c))
        return adj_sq_list