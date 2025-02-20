# Jungle_Chess/ai.py
import time
from GameLogic import *

class ChessBot:
    search_depth = 2
    CENTER_SQUARES = {(3, 3), (3, 4), (4, 3), (4, 4)}
    PIECE_VALUES = {
        Pawn: 100,
        Knight: 700,
        Bishop: 600,
        Rook: 500,
        Queen: 900,
        King: 100000
    }

    def __init__(self, board, color, app):
        self.board = board
        self.color = color
        self.tt = {}
        self.nodes_searched = 0
        self.app = app
        self.zobrist_table = self.initialize_zobrist_table()

    # ====================================================
    # Threat Checking Methods
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

        if not king_pos:
            return False

        for r, c in enemy_queens:
            queen_moves = board[r][c].get_valid_moves(board, (r, c))
            for move in queen_moves:
                if max(abs(move[0] - king_pos[0]), abs(move[1] - king_pos[1])) == 1:
                    target_piece = board[move[0]][move[1]]
                    if target_piece and target_piece.color == color:
                        return True
        return False

    def is_in_knight_evaporation_threat(self, board, color):
        """Check if any of the player's knights are in positions where enemy knights can attack via evaporation."""
        enemy_color = 'black' if color == 'white' else 'white'
        for r in range(ROWS):
            for c in range(COLS):
                piece = board[r][c]
                if isinstance(piece, Knight) and piece.color == color:
                    for dr, dc in DIRECTIONS['knight']:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < ROWS and 0 <= nc < COLS:
                            target_piece = board[nr][nc]
                            if isinstance(target_piece, Knight) and target_piece.color == enemy_color:
                                return True
        return False

    # ====================================================
    # Position Evaluation Methods
    # ====================================================
    def evaluate_board(self, board, depth, current_turn):
        score = 0
        our_king_pos = enemy_king_pos = None
        enemy_pieces = []

        for r in range(ROWS):
            for c in range(COLS):
                piece = board[r][c]
                if not piece:
                    continue

                if isinstance(piece, King):
                    if piece.color == self.color:
                        our_king_pos = (r, c)
                    else:
                        enemy_king_pos = (r, c)
                    continue

                if piece.color != self.color:
                    enemy_pieces.append((r, c, piece))

                value = self.PIECE_VALUES.get(type(piece), 0)
                if isinstance(piece, Knight):
                    value += len(piece.get_valid_moves(board, (r, c))) * 10
                elif isinstance(piece, Queen):
                    atomic_threats = sum(
                        1 for move in piece.get_valid_moves(board, (r, c))
                        if any(isinstance(board[adj_r][adj_c], King)
                               for adj_r, adj_c in self.get_adjacent_squares(move))
                    )
                    value += atomic_threats * 20

                score += value if piece.color == self.color else -value

        if not enemy_king_pos or not our_king_pos:
            return float('inf') if not enemy_king_pos else float('-inf')

        threat_score = 0
        if our_king_pos:
            for r, c, p in enemy_pieces:
                if our_king_pos in p.get_valid_moves(board, (r, c)):
                    threat_score += 200

        explosion_threat = self.is_in_explosion_threat(board, self.color)
        return int(score - threat_score - explosion_threat)

    # ====================================================
    # Move Ordering Methods
    # ====================================================
    def evaluate_move(self, board, move):
        start, end = move
        piece = board[start[0]][start[1]]
        target = board[end[0]][end[1]]

        if target:
            return 1000 + self.PIECE_VALUES.get(type(target), 0)

        if end in self.CENTER_SQUARES:
            return 50

        if isinstance(piece, Pawn):
            return end[0] if piece.color == "white" else 7 - end[0]

        return 0

    def order_moves(self, board, moves, maximizing_player=True):
        """
        Order moves using a heuristic evaluation and transposition table (TT) best move.
        If the TT contains a best move for the current board, that move is promoted.
        """
        if not moves:
            return moves

        board_key = self.board_hash(board)
        best_tt_move = None

        if board_key in self.tt and len(self.tt[board_key]) > 2:
            best_tt_move = self.tt[board_key][2]

        if best_tt_move and best_tt_move in moves:
            moves.remove(best_tt_move)
            moves.insert(0, best_tt_move)

        scored_moves = [(self.evaluate_move(board, move), move) for move in moves]

        if len(set(score for score, _ in scored_moves)) > 1:
            scored_moves.sort(reverse=maximizing_player, key=lambda x: x[0])

        return [move for _, move in scored_moves]

    # ====================================================
    # Search Methods (Minimax & Move Selection)
    # ====================================================
    def minimax(self, board, depth, maximizing_player, alpha, beta):
        """Optimized minimax with alpha-beta pruning and various pruning techniques."""
        self.nodes_searched += 1

        current_turn = self.color if maximizing_player else ('black' if self.color == 'white' else 'white')
        
        # Threefold repetition check using game_logic's function.
        if is_threefold_repetition(self.app.position_history):
            return 0

        if is_stalemate(board, current_turn):
            return 0

        board_key = self.board_hash(board)
        if board_key in self.tt and self.tt[board_key][0] >= depth:
            return self.tt[board_key][1]

        if depth == 0:
            return self.evaluate_board(board, depth, current_turn)

        # Null Move Pruning
        if depth >= 3 and not is_in_check(board, current_turn):
            null_move_reduction = 2
            if maximizing_player:
                null_value = self.minimax(board, depth - 1 - null_move_reduction, False, alpha, beta)
                if null_value >= beta:
                    return null_value
            else:
                null_value = self.minimax(board, depth - 1 - null_move_reduction, True, alpha, beta)
                if null_value <= alpha:
                    return null_value

        moves = self.get_all_moves(board, current_turn)
        if not moves:
            if is_in_check(board, current_turn):
                return float('-inf') if maximizing_player else float('inf')
            else:
                return 0

        moves = self.order_moves(board, moves, maximizing_player)

        best_value = float('-inf') if maximizing_player else float('inf')
        first_move = True

        for i, move in enumerate(moves):
            start, end = move
            piece = board[start[0]][start[1]]
            target = board[end[0]][end[1]]
            is_tactical = (target is not None)
            if not is_tactical and isinstance(piece, Pawn) and (end[0] == 0 or end[0] == ROWS - 1):
                is_tactical = True

            reduction = 0
            if not is_tactical and i >= 3 and depth >= 4:
                reduction = 1

            new_depth = depth - 1 - reduction
            new_board = self.simulate_move(board, start, end)
            if first_move:
                score = self.minimax(new_board, new_depth, not maximizing_player, alpha, beta)
                first_move = False
            else:
                if maximizing_player:
                    score = self.minimax(new_board, new_depth, not maximizing_player, alpha, alpha + 1)
                    if score > alpha and score < beta:
                        score = self.minimax(new_board, new_depth, not maximizing_player, score, beta)
                else:
                    score = self.minimax(new_board, new_depth, not maximizing_player, beta - 1, beta)
                    if score < beta and score > alpha:
                        score = self.minimax(new_board, new_depth, not maximizing_player, alpha, score)

            if maximizing_player:
                if score > best_value:
                    best_value = score
                alpha = max(alpha, best_value)
            else:
                if score < best_value:
                    best_value = score
                beta = min(beta, best_value)

            if beta <= alpha:
                break

        self.tt[board_key] = (depth, best_value)
        return best_value

    def make_move(self):
        """Make the best move found by the bot."""
        best_move = None
        best_value = float('-inf')
        total_start = time.time()

        for current_depth in range(1, self.search_depth + 1):
            self.nodes_searched = 0
            iteration_start = time.time()

            moves = self.get_all_moves(self.board, self.color)
            if not moves:
                return False

            moves = self.order_moves(self.board, moves, maximizing_player=True)
            current_best_move = None
            current_best_value = float('-inf')

            for move in moves:
                start, end = move
                new_board = self.simulate_move(self.board, start, end)
                value = self.minimax(new_board, current_depth - 1, False, float('-inf'), float('inf'))
                if value > current_best_value:
                    current_best_value = value
                    current_best_move = move

            iteration_time = time.time() - iteration_start
            reported_value = -current_best_value if self.color == 'black' else current_best_value
            print(f"AI Depth {current_depth}: {iteration_time:.3f}s, AI nodes: {self.nodes_searched}, AI Eval: {reported_value}")

            self.app.master.after(0, lambda: self.app.draw_eval_bar(reported_value))
            best_move = current_best_move
            best_value = current_best_value

        print(f"AI Total time: {(time.time() - total_start):.3f}s")

        # Before applying the best move, check if the game history shows threefold repetition.
        if is_threefold_repetition(self.app.position_history):
            self.app.turn_label.config(text="Draw detected: Threefold repetition imminent!")

        if best_move:
            start, end = best_move
            moving_piece = self.board[start[0]][start[1]]
            self.board = moving_piece.move(self.board, start, end)
            check_evaporation(self.board)
            return True
        return False

    # ====================================================
    # Hashing and Helper Methods
    # ====================================================
    def initialize_zobrist_table(self):
        import random
        random.seed(42)
        return {
            (r, c, piece_type, color): random.getrandbits(64)
            for r in range(ROWS)
            for c in range(COLS)
            for piece_type in [Pawn, Knight, Bishop, Rook, Queen, King, None]
            for color in ['white', 'black', None]
        }

    def board_hash(self, board):
        hash_val = 0
        for r in range(ROWS):
            for c in range(COLS):
                piece = board[r][c]
                key = (r, c, type(piece) if piece else None, 
                      piece.color if piece else None)
                hash_val ^= self.zobrist_table.get(key, 0)
        return hash_val

    def get_adjacent_squares(self, pos):
        r, c = pos
        return [
            (adj_r, adj_c)
            for adj_r in range(r - 1, r + 2)
            for adj_c in range(c - 1, c + 2)
            if 0 <= adj_r < ROWS and 0 <= adj_c < COLS and (adj_r, adj_c) != (r, c)
        ]

    def simulate_move(self, board, start, end):
        new_board = copy_board(board)
        piece = new_board[start[0]][start[1]]
        new_board = piece.move(new_board, start, end)
        check_evaporation(new_board)
        return new_board

    def get_all_moves(self, board, color):
        return [
            ((r, c), move)
            for r in range(ROWS) for c in range(COLS)
            if (piece := board[r][c]) and piece.color == color
            for move in piece.get_valid_moves(board, (r, c))
            if validate_move(board, color, (r, c), move)
        ]
