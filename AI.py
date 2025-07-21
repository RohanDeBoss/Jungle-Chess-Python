# AI.py v1.0

import time
from GameLogic import *
import random
from collections import namedtuple

### NEW: Zobrist Hashing moved to global scope for shared use by UI and AI
def initialize_zobrist_table():
    """Creates a table of random numbers for each piece/square combination."""
    random.seed(42)
    table = {
        (r, c, piece_type, color): random.getrandbits(64)
        for r in range(ROWS) for c in range(COLS)
        for piece_type in [Pawn, Knight, Bishop, Rook, Queen, King, None]
        for color in ['white', 'black', None]
    }
    # Add a key to hash whose turn it is
    table['turn'] = random.getrandbits(64)
    return table

# Create a single, global Zobrist table to be used by the entire application.
ZOBRIST_TABLE = initialize_zobrist_table()

def board_hash(board, turn):
    """Calculates the Zobrist hash for a given board state and turn."""
    hash_val = 0
    for r in range(ROWS):
        for c in range(COLS):
            piece = board[r][c]
            key = (r, c, type(piece) if piece else None, piece.color if piece else None)
            hash_val ^= ZOBRIST_TABLE.get(key, 0)
    if turn == 'black':
        hash_val ^= ZOBRIST_TABLE['turn']
    return hash_val


TTEntry = namedtuple('TTEntry', ['score', 'depth', 'flag', 'best_move'])
TT_FLAG_EXACT = 0
TT_FLAG_LOWERBOUND = 1
TT_FLAG_UPPERBOUND = 2

class SearchCancelledException(Exception):
    """Exception raised when the AI search is cancelled by the UI."""
    pass

class ChessBot:
    search_depth = 3
    CENTER_SQUARES = {(3, 3), (3, 4), (4, 3), (4, 4)}
    PIECE_VALUES = {
        Pawn: 100, Knight: 700, Bishop: 600,
        Rook: 500, Queen: 900, King: 100000
    }
    MATE_SCORE = 1000000
    DRAW_SCORE = 0
    
    QSEARCH_CHECKS_MAX_DEPTH = 4

    CAPTURE_SCORE_BONUS = 10000
    PROMOTION_SCORE_BONUS = 9000
    CHECK_SCORE_BONUS = 5000
    HASH_MOVE_SCORE_BONUS = 100000

    def __init__(self, board, color, app, cancellation_event):
        self.board = board
        self.color = color
        self.opponent_color = 'black' if color == 'white' else 'white'
        self.app = app
        self.cancellation_event = cancellation_event
        self.tt = {}
        self.nodes_searched = 0
        self.MAX_PLY_KILLERS = 30
        self.killer_moves = [[None, None] for _ in range(self.MAX_PLY_KILLERS)]
        self.position_history = []
        self.position_counts = {}
    
    def evaluate_board(self, board, current_turn):
        perspective_multiplier = 1 if current_turn == self.color else -1
        score_relative_to_ai = 0
        our_king_pos = find_king_pos(board, self.color)
        enemy_king_pos = find_king_pos(board, self.opponent_color)
        
        if not enemy_king_pos: return self.MATE_SCORE * perspective_multiplier
        if not our_king_pos: return -self.MATE_SCORE * perspective_multiplier

        enemy_pieces = []
        for r_eval in range(ROWS):
            for c_eval in range(COLS):
                piece_eval = board[r_eval][c_eval]
                if not piece_eval or isinstance(piece_eval, King):
                    continue
                is_our_piece = (piece_eval.color == self.color)
                if not is_our_piece:
                    enemy_pieces.append((r_eval, c_eval, piece_eval))
                value = self.PIECE_VALUES.get(type(piece_eval), 0)
                if isinstance(piece_eval, Knight):
                    value += len(piece_eval.get_valid_moves(board, (r_eval, c_eval))) * 5
                elif isinstance(piece_eval, Queen) and is_our_piece:
                    atomic_threats = sum(
                        1 for move_q in piece_eval.get_valid_moves(board, (r_eval, c_eval))
                        if any(isinstance(board[adj_r][adj_c], King) and board[adj_r][adj_c].color != self.color
                            for adj_r, adj_c in self.get_adjacent_squares(move_q)))
                    value += atomic_threats * 15
                score_relative_to_ai += value if is_our_piece else -value

        king_to_check_pos = our_king_pos if current_turn == self.color else enemy_king_pos
        pieces_that_threaten = enemy_pieces if current_turn == self.color else [
            (r, c, p) for r in range(ROWS) for c in range(COLS) 
            if (p := board[r][c]) and p.color == self.color and not isinstance(p, King)
        ]
        threat_score = sum(200 for r,c,p in pieces_that_threaten if king_to_check_pos in p.get_valid_moves(board, (r,c)))
        
        ### MODIFIED: Call the canonical function from GameLogic
        explosion_threat = is_in_explosion_threat(board, current_turn)
        final_score = score_relative_to_ai
        
        if current_turn == self.color:
            final_score -= threat_score
            final_score -= (250 if explosion_threat else 0)
        else:
            final_score += threat_score
            final_score += (250 if explosion_threat else 0)
            
        return int(final_score * perspective_multiplier)

    def is_move_a_check(self, board, move, moving_player_color):
        # We can use our new tactical checker for this, since it simulates the move.
        # This avoids having to create a sim_board in two different places.
        return is_move_tactical(board, move, is_qsearch_check=True)

    def evaluate_move(self, board, move, moving_player_color):
        start, end = move
        piece = board[start[0]][start[1]]
        target = board[end[0]][end[1]]
        score = 0

        # Standard capture value
        if target:
            score = self.CAPTURE_SCORE_BONUS + (self.PIECE_VALUES.get(type(target), 0) - self.PIECE_VALUES.get(type(piece), 0))
        
        # Promotion bonus
        if isinstance(piece, Pawn) and (end[0] == 0 or end[0] == ROWS - 1):
            score += self.PROMOTION_SCORE_BONUS

        # Check bonus
        if self.is_move_a_check(board, move, moving_player_color):
            score += self.CHECK_SCORE_BONUS
            
        return score

    def order_moves(self, board, moves, moving_player_color, ply_searched=0, hash_move=None):
        # ... (This function is now simpler as evaluate_move handles all bonuses)
        move_scores = {move: self.evaluate_move(board, move, moving_player_color) for move in moves}
        
        if hash_move and hash_move in move_scores:
            move_scores[hash_move] += self.HASH_MOVE_SCORE_BONUS
            
        if ply_searched < self.MAX_PLY_KILLERS:
            killers = self.killer_moves[ply_searched]
            if killers[0] and killers[0] in move_scores:
                move_scores[killers[0]] += 50000
            if killers[1] and killers[1] in move_scores:
                move_scores[killers[1]] += 40000

        return sorted(moves, key=lambda m: move_scores.get(m, 0), reverse=True)

    ### MODIFIED: qsearch is now beautifully simple and correct. ###
    def qsearch(self, board, alpha, beta, turn_multiplier, depth):
        if self.cancellation_event.is_set(): raise SearchCancelledException()
        self.nodes_searched += 1

        current_turn = self.color if turn_multiplier == 1 else self.opponent_color
        stand_pat = self.evaluate_board(board, current_turn)

        if stand_pat >= beta:
            return beta
        alpha = max(alpha, stand_pat)

        all_moves = self.get_all_moves(board, current_turn)
        
        # Use our new centralized function to find all tactical moves.
        # The is_qsearch_check flag includes checks, but only if we have depth.
        tactical_moves = [
            m for m in all_moves 
            if is_move_tactical(board, m, is_qsearch_check=(depth > 0))
        ]

        for move in self.order_moves(board, tactical_moves, current_turn):
            # We can still reduce depth for non-capture/non-promo tactical moves
            piece = board[move[0][0]][move[0][1]]
            target = board[move[1][0]][move[1][1]]
            is_promo = isinstance(piece, Pawn) and (move[1][0] == 0 or move[1][0] == ROWS-1)

            next_depth = depth if (target or is_promo) else depth - 1
            
            child_board = self.simulate_move(board, move[0], move[1])
            score = -self.qsearch(child_board, -beta, -alpha, -turn_multiplier, next_depth)
            
            if score >= beta: return beta
            alpha = max(alpha, score)
            
        return alpha

    def negamax(self, board, depth, alpha, beta, turn_multiplier, ply):
        if self.cancellation_event.is_set(): raise SearchCancelledException()
        self.nodes_searched += 1
        
        current_turn = self.color if turn_multiplier == 1 else self.opponent_color
        original_alpha = alpha
        
        hash_val = board_hash(board, current_turn)

        if ply > 0 and self.app.position_counts.get(hash_val, 0) >= 2:
            return self.DRAW_SCORE

        hash_move = None
        tt_entry = self.tt.get(hash_val)
        if tt_entry and tt_entry.depth >= depth:
            if tt_entry.flag == TT_FLAG_EXACT: return tt_entry.score
            elif tt_entry.flag == TT_FLAG_LOWERBOUND: alpha = max(alpha, tt_entry.score)
            elif tt_entry.flag == TT_FLAG_UPPERBOUND: beta = min(beta, tt_entry.score)
            if alpha >= beta: return tt_entry.score
            hash_move = tt_entry.best_move

        is_in_check_now = is_in_check(board, current_turn)
        if is_in_check_now: depth += 1

        if depth <= 0:
            return self.qsearch(board, alpha, beta, turn_multiplier, self.QSEARCH_CHECKS_MAX_DEPTH)

        moves = self.get_all_moves(board, current_turn)
        if not moves:
            return -self.MATE_SCORE + ply if is_in_check_now else self.DRAW_SCORE

        ordered_moves = self.order_moves(board, moves, current_turn, ply, hash_move=hash_move)
        best_move_for_node = None
        
        for move in ordered_moves:
            child_board = self.simulate_move(board, move[0], move[1])
            child_turn = self.opponent_color if turn_multiplier == 1 else self.color
            child_hash = board_hash(child_board, child_turn)
            
            self.app.position_counts[child_hash] = self.app.position_counts.get(child_hash, 0) + 1
            score = -self.negamax(child_board, depth - 1, -beta, -alpha, -turn_multiplier, ply + 1)
            self.app.position_counts[child_hash] -= 1
            
            if score > alpha:
                alpha = score
                best_move_for_node = move
            
            if alpha >= beta:
                is_capture = board[move[1][0]][move[1][1]] is not None
                if not is_capture and ply < self.MAX_PLY_KILLERS and self.killer_moves[ply][0] != move:
                    self.killer_moves[ply][1] = self.killer_moves[ply][0]
                    self.killer_moves[ply][0] = move
                
                self.tt[hash_val] = TTEntry(beta, depth, TT_FLAG_LOWERBOUND, move)
                return beta
            
        flag = TT_FLAG_UPPERBOUND if alpha <= original_alpha else TT_FLAG_EXACT
        self.tt[hash_val] = TTEntry(alpha, depth, flag, best_move_for_node)
        return alpha

    def make_move(self):
        try:
            best_move_found = None
            total_start_time = time.time()
            self.killer_moves = [[None, None] for _ in range(self.MAX_PLY_KILLERS)]
            self.tt.clear()

            for current_depth in range(1, self.search_depth + 1):
                if self.cancellation_event.is_set(): raise SearchCancelledException()
                self.nodes_searched = 0
                
                root_moves = self.get_all_moves(self.board, self.color)
                if not root_moves: return False
                
                root_hash = board_hash(self.board, self.color)
                hash_move = self.tt.get(root_hash, None)
                if hash_move: hash_move = hash_move.best_move

                ordered_root_moves = self.order_moves(self.board, root_moves, self.color, 0, hash_move=hash_move)
                
                best_score = -float('inf')
                alpha, beta = -float('inf'), float('inf')

                for i, move in enumerate(ordered_root_moves):
                    if self.cancellation_event.is_set(): raise SearchCancelledException()

                    child_board = self.simulate_move(self.board, move[0], move[1])
                    child_hash = board_hash(child_board, self.opponent_color)
                    self.app.position_counts[child_hash] = self.app.position_counts.get(child_hash, 0) + 1
                    
                    score = -self.negamax(child_board, current_depth - 1, -beta, -alpha, -1, 1)
                    
                    self.app.position_counts[child_hash] -= 1

                    if score > best_score:
                        best_score = score
                        best_move_found = move
                    alpha = max(alpha, best_score)

                if not self.cancellation_event.is_set():
                    eval_for_ui = best_score * (1 if self.color == 'white' else -1)
                    print(f"Depth {current_depth}: Best={best_move_found}, Eval={eval_for_ui/100:.2f}, Nodes={self.nodes_searched}")
                    if self.app: self.app.master.after(0, self.app.draw_eval_bar, eval_for_ui)

            if best_move_found:
                piece = self.board[best_move_found[0][0]][best_move_found[0][1]]
                self.board = piece.move(self.board, best_move_found[0], best_move_found[1])
                check_evaporation(self.board)
                return True
            return False
            
        except SearchCancelledException:
            print(f"AI ({self.color}): Search cancelled.")
            return False

    def get_adjacent_squares(self, pos):
        r, c = pos; squares = []
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr==0 and dc==0: continue
                if 0 <= r+dr < ROWS and 0 <= c+dc < COLS: squares.append((r+dr, c+dc))
        return squares

    def simulate_move(self, board, start, end):
        new_board = copy_board(board)
        piece = new_board[start[0]][start[1]]
        if piece:
            new_board = piece.move(new_board, start, end)
            check_evaporation(new_board)
        return new_board
        
    def get_all_moves(self, board, color):
        moves = []
        for r in range(ROWS):
            for c in range(COLS):
                piece = board[r][c]
                if piece and piece.color == color:
                    for move in piece.get_valid_moves(board, (r, c)):
                        if validate_move(board, color, (r, c), move):
                            moves.append(((r, c), move))
        return moves

    def negamax(self, board, depth, alpha, beta, turn_multiplier, ply):
        if self.cancellation_event.is_set(): raise SearchCancelledException()
        self.nodes_searched += 1
        
        current_turn = self.color if turn_multiplier == 1 else self.opponent_color
        original_alpha = alpha
        
        ### MODIFIED: Call the global board_hash function
        hash_val = board_hash(board, current_turn)

        if ply > 0 and self.app.position_counts.get(hash_val, 0) >= 2:
            return self.DRAW_SCORE

        hash_move = None
        tt_entry = self.tt.get(hash_val)
        if tt_entry and tt_entry.depth >= depth:
            if tt_entry.flag == TT_FLAG_EXACT: return tt_entry.score
            elif tt_entry.flag == TT_FLAG_LOWERBOUND: alpha = max(alpha, tt_entry.score)
            elif tt_entry.flag == TT_FLAG_UPPERBOUND: beta = min(beta, tt_entry.score)
            if alpha >= beta: return tt_entry.score
            hash_move = tt_entry.best_move

        is_in_check_now = is_in_check(board, current_turn)
        if is_in_check_now: depth += 1

        if depth <= 0:
            return self.qsearch(board, alpha, beta, turn_multiplier, self.QSEARCH_CHECKS_MAX_DEPTH)

        moves = self.get_all_moves(board, current_turn)
        if not moves:
            return -self.MATE_SCORE + ply if is_in_check_now else self.DRAW_SCORE

        ordered_moves = self.order_moves(board, moves, current_turn, ply, hash_move=hash_move)
        best_move_for_node = None
        
        for move in ordered_moves:
            child_board = self.simulate_move(board, move[0], move[1])
            child_turn = self.opponent_color if turn_multiplier == 1 else self.color
            ### MODIFIED: Call the global board_hash function
            child_hash = board_hash(child_board, child_turn)
            
            self.app.position_counts[child_hash] = self.app.position_counts.get(child_hash, 0) + 1
            score = -self.negamax(child_board, depth - 1, -beta, -alpha, -turn_multiplier, ply + 1)
            self.app.position_counts[child_hash] -= 1
            
            if score > alpha:
                alpha = score
                best_move_for_node = move
            
            if alpha >= beta:
                is_capture = board[move[1][0]][move[1][1]] is not None
                if not is_capture and ply < self.MAX_PLY_KILLERS and self.killer_moves[ply][0] != move:
                    self.killer_moves[ply][1] = self.killer_moves[ply][0]
                    self.killer_moves[ply][0] = move
                
                self.tt[hash_val] = TTEntry(beta, depth, TT_FLAG_LOWERBOUND, move)
                return beta
            
        flag = TT_FLAG_UPPERBOUND if alpha <= original_alpha else TT_FLAG_EXACT
        self.tt[hash_val] = TTEntry(alpha, depth, flag, best_move_for_node)
        return alpha

    def make_move(self):
        try:
            best_move_found = None
            total_start_time = time.time()
            self.killer_moves = [[None, None] for _ in range(self.MAX_PLY_KILLERS)]
            self.tt.clear()

            for current_depth in range(1, self.search_depth + 1):
                if self.cancellation_event.is_set(): raise SearchCancelledException()
                self.nodes_searched = 0
                
                root_moves = self.get_all_moves(self.board, self.color)
                if not root_moves: return False
                
                ### MODIFIED: Call the global board_hash function
                root_hash = board_hash(self.board, self.color)
                hash_move = self.tt.get(root_hash, None)
                if hash_move: hash_move = hash_move.best_move

                ordered_root_moves = self.order_moves(self.board, root_moves, self.color, 0, hash_move=hash_move)
                
                best_score = -float('inf')
                alpha, beta = -float('inf'), float('inf')

                for i, move in enumerate(ordered_root_moves):
                    if self.cancellation_event.is_set(): raise SearchCancelledException()

                    child_board = self.simulate_move(self.board, move[0], move[1])
                    ### MODIFIED: Call the global board_hash function
                    child_hash = board_hash(child_board, self.opponent_color)
                    self.app.position_counts[child_hash] = self.app.position_counts.get(child_hash, 0) + 1
                    
                    score = -self.negamax(child_board, current_depth - 1, -beta, -alpha, -1, 1)
                    
                    self.app.position_counts[child_hash] -= 1

                    if score > best_score:
                        best_score = score
                        best_move_found = move
                    alpha = max(alpha, best_score)

                if not self.cancellation_event.is_set():
                    eval_for_ui = best_score * (1 if self.color == 'white' else -1)
                    print(f"Depth {current_depth}: Best={best_move_found}, Eval={eval_for_ui/100:.2f}, Nodes={self.nodes_searched}")
                    if self.app: self.app.master.after(0, self.app.draw_eval_bar, eval_for_ui)

            if best_move_found:
                piece = self.board[best_move_found[0][0]][best_move_found[0][1]]
                self.board = piece.move(self.board, best_move_found[0], best_move_found[1])
                check_evaporation(self.board)
                return True
            return False
            
        except SearchCancelledException:
            print(f"AI ({self.color}): Search cancelled.")
            return False

    def get_adjacent_squares(self, pos):
        r, c = pos; squares = []
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr==0 and dc==0: continue
                if 0 <= r+dr < ROWS and 0 <= c+dc < COLS: squares.append((r+dr, c+dc))
        return squares

    def simulate_move(self, board, start, end):
        new_board = copy_board(board)
        piece = new_board[start[0]][start[1]]
        if piece:
            new_board = piece.move(new_board, start, end)
            check_evaporation(new_board)
        return new_board
        
    def get_all_moves(self, board, color):
        moves = []
        for r in range(ROWS):
            for c in range(COLS):
                piece = board[r][c]
                if piece and piece.color == color:
                    for move in piece.get_valid_moves(board, (r, c)):
                        if validate_move(board, color, (r, c), move):
                            moves.append(((r, c), move))
        return moves