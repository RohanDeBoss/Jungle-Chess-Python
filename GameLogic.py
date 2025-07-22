# game_logic.py

# --- Versioning ---
# v1.3
# - Renamed get_all_moves to generate_pseudo_legal_moves for clarity. This function is now
#   used by the AI's optimized search loop to quickly get all potential moves.
# - The core move validation and check logic remains robust and unchanged.
# - No other functional changes were made.

# -----------------------------
# Global Constants
# -----------------------------
ROWS, COLS = 8, 8
SQUARE_SIZE = 65
BOARD_COLOR_1 = "#D2B48C"
BOARD_COLOR_2 = "#8B5A2B"

DIRECTIONS = {
    'king': ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)),
    'queen': ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)),
    'rook': ((0, 1), (0, -1), (1, 0), (-1, 0)),
    'bishop': ((-1, -1), (-1, 1), (1, -1), (1, 1)),
    'knight': ((2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2))
}

ADJACENT_DIRS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


# -----------------------------
# Piece Base Class and Subclasses
# -----------------------------
class Piece:
    def __init__(self, color):
        self.color = color
        self.has_moved = False
        self.opponent_color = "black" if color == "white" else "white"

    def clone(self):
        new_piece = self.__class__(self.color)
        new_piece.has_moved = self.has_moved
        return new_piece

    def symbol(self):
        return "?"

    def get_valid_moves(self, board, pos):
        return []

    def move(self, board, start, end):
        board[end[0]][end[1]] = self
        board[start[0]][start[1]] = None
        self.has_moved = True
        return board

    def _is_valid_square(self, r, c):
        return 0 <= r < ROWS and 0 <= c < COLS


class King(Piece):
    def symbol(self): return "♔" if self.color == "white" else "♚"

    def get_valid_moves(self, board, pos):
        moves = []
        r_start, c_start = pos
        for dr, dc in DIRECTIONS['king']:
            for step in (1, 2):
                new_r, new_c = r_start + dr * step, c_start + dc * step
                if self._is_valid_square(new_r, new_c):
                    if step == 2:
                        inter_r, inter_c = r_start + dr, c_start + dc
                        if board[inter_r][inter_c] is not None:
                            break
                    target = board[new_r][new_c]
                    if target is None or target.color != self.color:
                        moves.append((new_r, new_c))
                    if target is not None:
                        break
                else:
                    break
        return moves


class Queen(Piece):
    def symbol(self): return "♕" if self.color == "white" else "♛"

    def get_valid_moves(self, board, pos):
        moves = []
        r_start, c_start = pos
        for dr, dc in DIRECTIONS['queen']:
            r, c = r_start, c_start
            while True:
                r += dr
                c += dc
                if not self._is_valid_square(r, c):
                    break
                target = board[r][c]
                if target is None:
                    moves.append((r, c))
                else:
                    if target.color != self.color:
                        moves.append((r, c))
                    break
        return moves

    def move(self, board, start, end):
        piece_at_destination = board[end[0]][end[1]]
        is_capture = piece_at_destination and piece_at_destination.color != self.color
        if is_capture:
            board[end[0]][end[1]] = None
            for dr_adj, dc_adj in ADJACENT_DIRS:
                adj_r, adj_c = end[0] + dr_adj, end[1] + dc_adj
                if self._is_valid_square(adj_r, adj_c):
                    piece_on_adj = board[adj_r][adj_c]
                    if piece_on_adj and piece_on_adj.color != self.color:
                        board[adj_r][adj_c] = None
            board[start[0]][start[1]] = None
            self.has_moved = True
        else:
            super().move(board, start, end)
        return board


class Rook(Piece):
    def symbol(self): return "♖" if self.color == "white" else "♜"

    def get_valid_moves(self, board, pos):
        moves = []
        r_start, c_start = pos
        for dr, dc in DIRECTIONS['rook']:
            r, c = r_start, c_start
            enemy_encountered = False
            while True:
                r += dr
                c += dc
                if not self._is_valid_square(r,c):
                    break
                target = board[r][c]
                if not enemy_encountered:
                    if target is None:
                        moves.append((r, c))
                    else:
                        if target.color != self.color:
                            moves.append((r, c))
                            enemy_encountered = True
                        else:
                            break
                else:
                    if target is None or target.color != self.color:
                        moves.append((r, c))
                    else:
                        break
        return moves

    def move(self, board, start, end):
        if start[0] == end[0]:
            d = (0, 1 if end[1] > start[1] else -1)
        else:
            d = (1 if end[0] > start[0] else -1, 0)
        
        current_r, current_c = start[0] + d[0], start[1] + d[1]
        path_contained_enemy = False
        path_to_clear = []
        
        while (current_r, current_c) != end:
            if not self._is_valid_square(current_r, current_c):
                break
            piece_on_path = board[current_r][current_c]
            if piece_on_path and piece_on_path.color != self.color:
                path_contained_enemy = True
            path_to_clear.append((current_r, current_c))
            current_r += d[0]
            current_c += d[1]
            
        if path_contained_enemy:
            for r_path, c_path in path_to_clear:
                piece_to_clear = board[r_path][c_path]
                if piece_to_clear and piece_to_clear.color != self.color:
                    board[r_path][c_path] = None
        
        super().move(board, start, end)
        return board


class Bishop(Piece):
    def symbol(self): return "♗" if self.color == "white" else "♝"

    def get_valid_moves(self, board, pos):
        return list(set(get_zigzag_moves(board, pos, self.color) + get_diagonal_moves(board, pos, self.color)))


class Knight(Piece):
    def symbol(self): return "♘" if self.color == "white" else "♞"

    def get_valid_moves(self, board, pos):
        moves = []
        r_start, c_start = pos
        for dr, dc in DIRECTIONS['knight']:
            nr, nc = r_start + dr, c_start + dc
            if self._is_valid_square(nr, nc):
                target = board[nr][nc]
                if not target or target.color != self.color:
                    moves.append((nr,nc))
        return moves

    def move(self, board, start, end):
        super().move(board, start, end)
        self.evaporate(board, end)
        return board

    def evaporate(self, board, pos):
        enemy_knights_evaporated = []
        r_knight, c_knight = pos
        for dr, dc in DIRECTIONS['knight']:
            r_adj, c_adj = r_knight + dr, c_knight + dc
            if self._is_valid_square(r_adj, c_adj):
                piece_on_adj = board[r_adj][c_adj]
                if piece_on_adj and piece_on_adj.color != self.color:
                    if isinstance(piece_on_adj, Knight):
                        enemy_knights_evaporated.append((r_adj, c_adj))
                    board[r_adj][c_adj] = None
        if enemy_knights_evaporated:
            board[r_knight][c_knight] = None


class Pawn(Piece):
    def symbol(self): return "♙" if self.color == "white" else "♟"

    def get_valid_moves(self, board, pos):
        moves = []
        r_start, c_start = pos
        direction = -1 if self.color == "white" else 1
        starting_row = 6 if self.color == "white" else 1
        
        for steps in [1, 2]:
            if steps == 2 and r_start != starting_row:
                continue
            new_r, new_c = r_start + (direction * steps), c_start
            if self._is_valid_square(new_r, new_c):
                fwd_target = board[new_r][new_c]
                if fwd_target and fwd_target.color != self.color:
                    moves.append((new_r, new_c))
                if fwd_target is None:
                    moves.append((new_r, new_c))
                if fwd_target is not None:
                    break
            else:
                break
                
        for dc_offset in [-1, 1]:
            new_c_side = c_start + dc_offset
            if self._is_valid_square(r_start, new_c_side):
                side_target = board[r_start][new_c_side]
                if side_target and side_target.color != self.color:
                    moves.append((r_start, new_c_side))
        return moves

    def move(self, board, start, end):
        super().move(board, start, end)
        promotion_rank = 0 if self.color == "white" else (ROWS - 1)
        if end[0] == promotion_rank:
            board[end[0]][end[1]] = Queen(self.color)
        return board


# -----------------------------
# Helper Functions
# -----------------------------
def get_zigzag_moves(board, pos, color):
    moves = set()
    r_start, c_start = pos
    direction_pairs = (
        ((-1, 1), (-1, -1)), ((-1, -1), (-1, 1)), ((1, 1), (1, -1)), ((1, -1), (1, 1)),
        ((-1, 1), (1, 1)), ((1, 1), (-1, 1)), ((-1, -1), (1, -1)), ((1, -1), (-1, -1))
    )
    for d1, d2 in direction_pairs:
        cr, cc, cd = r_start, c_start, d1
        while True:
            cr += cd[0]
            cc += cd[1]
            if not (0 <= cr < ROWS and 0 <= cc < COLS):
                break
            target = board[cr][cc]
            if target:
                if target.color != color:
                    moves.add((cr, cc))
                break
            moves.add((cr, cc))
            cd = d2 if cd == d1 else d1
    return list(moves)


def get_diagonal_moves(board, pos, color):
    moves = []
    r_start, c_start = pos
    for dr, dc in DIRECTIONS['bishop']:
        r, c = r_start, c_start
        while True:
            r += dr
            c += dc
            if not (0 <= r < ROWS and 0 <= c < COLS):
                break
            target = board[r][c]
            if target:
                if target.color != color:
                    moves.append((r, c))
                break
            moves.append((r, c))
    return moves


def copy_board(board):
    return [[p.clone() if p else None for p in row] for row in board]


# -----------------------------
# Game Logic
# -----------------------------
def create_initial_board():
    board = [[None for _ in range(COLS)] for _ in range(ROWS)]
    board[0][0]=Rook("black"); board[0][1]=Knight("black"); board[0][2]=Bishop("black"); board[0][3]=Queen("black")
    board[0][4]=King("black"); board[0][5]=Bishop("black"); board[0][6]=Knight("black"); board[0][7]=Rook("black")
    for i in range(COLS): board[1][i] = Pawn("black")
    board[7][0]=Rook("white"); board[7][1]=Knight("white"); board[7][2]=Bishop("white"); board[7][3]=Queen("white")
    board[7][4]=King("white"); board[7][5]=Bishop("white"); board[7][6]=Knight("white"); board[7][7]=Rook("white")
    for i in range(COLS): board[6][i] = Pawn("white")
    return board


def find_king_pos(board, color):
    for r in range(ROWS):
        for c in range(COLS):
            piece = board[r][c]
            if isinstance(piece, King) and piece.color == color:
                return (r, c)
    return None


def has_legal_moves(board, color):
    # This check is used to determine stalemate/checkmate. It needs to be fully correct.
    for start_pos, end_pos in generate_pseudo_legal_moves(board, color):
        if validate_move(board, color, start_pos, end_pos):
            return True
    return False


def check_game_over(board, turn_color_who_just_moved):
    if not find_king_pos(board, "white"): return "king_capture", "black"
    if not find_king_pos(board, "black"): return "king_capture", "white"
    
    next_player_color = "black" if turn_color_who_just_moved == "white" else "white"
    
    if not has_legal_moves(board, next_player_color):
        if is_in_check(board, next_player_color):
            return "checkmate", turn_color_who_just_moved
        else:
            return "stalemate", None
            
    return None, None


def check_evaporation(board):
    knights_on_board = []
    for r in range(ROWS):
        for c in range(COLS):
            piece = board[r][c]
            if isinstance(piece, Knight):
                knights_on_board.append(((r,c), piece))

    for pos, knight_instance in knights_on_board:
        if board[pos[0]][pos[1]] is knight_instance:
             knight_instance.evaporate(board, pos)


def is_in_check(board, color):
    """A comprehensive check for all threats against a king of a given color."""
    king_pos = find_king_pos(board, color)
    if not king_pos:
        return True # A captured king is the ultimate check

    enemy_color = 'black' if color == 'white' else 'white'

    for r_attacker in range(ROWS):
        for c_attacker in range(COLS):
            piece = board[r_attacker][c_attacker]
            if not piece or piece.color != enemy_color:
                continue

            # Case 1: Standard move-based attacks (covers Rooks, Bishops, Pawns, sliding Queens, Kings)
            if king_pos in piece.get_valid_moves(board, (r_attacker, c_attacker)):
                return True

            # Case 2: Static Knight check (King is an L-shape away from a Knight)
            if isinstance(piece, Knight):
                for dr, dc in DIRECTIONS['knight']:
                    if (r_attacker + dr, c_attacker + dc) == king_pos:
                        return True
            
            # Case 3: Special "potential" threats that create an attack zone
            if isinstance(piece, Queen):
                for move_dest in piece.get_valid_moves(board, (r_attacker, c_attacker)):
                    target = board[move_dest[0]][move_dest[1]]
                    if target and target.color == color:
                        if max(abs(move_dest[0] - king_pos[0]), abs(move_dest[1] - king_pos[1])) == 1:
                            return True
            
            if isinstance(piece, Knight):
                for move_dest in piece.get_valid_moves(board, (r_attacker, c_attacker)):
                    for dr_evap, dc_evap in DIRECTIONS['knight']:
                        if (move_dest[0] + dr_evap, move_dest[1] + dc_evap) == king_pos:
                            return True
                            
    return False


def validate_move(board, color, start, end):
    """The ultimate authority on whether a single move is legal."""
    piece_to_move = board[start[0]][start[1]]
    if not piece_to_move or piece_to_move.color != color:
        return False
    
    if end not in piece_to_move.get_valid_moves(board, start):
        return False
        
    sim_board = copy_board(board)
    sim_piece = sim_board[start[0]][start[1]]
    sim_piece.move(sim_board, start, end)
    check_evaporation(sim_board)
    
    # If the simulated board leaves our king in check, the move is illegal.
    if is_in_check(sim_board, color):
        return False
        
    return True

def generate_pseudo_legal_moves(board, color):
    """Generates all moves a player can make, without checking for check legality."""
    moves = []
    for r in range(ROWS):
        for c in range(COLS):
            piece = board[r][c]
            if piece and piece.color == color:
                # The piece's own get_valid_moves logic is inherently pseudo-legal
                for end_pos in piece.get_valid_moves(board, (r, c)):
                    moves.append(((r, c), end_pos))
    return moves

# --- Tactical Move Checkers for AI ---

def _is_rook_path_capture(board, start_pos, end_pos, piece_color):
    if start_pos[0] == end_pos[0]:
        d = (0, 1 if end_pos[1] > start_pos[1] else -1)
    else:
        d = (1 if end_pos[0] > start_pos[0] else -1, 0)
        
    cr, cc = start_pos[0] + d[0], start_pos[1] + d[1]
    
    while (cr, cc) != end_pos:
        target = board[cr][cc]
        if target and target.color != piece_color:
            return True
        cr += d[0]
        cc += d[1]
        
    return False


def _is_knight_evaporation_move(board, start_pos, end_pos, piece_color):
    for dr, dc in DIRECTIONS['knight']:
        nr, nc = end_pos[0] + dr, end_pos[1] + dc
        if 0 <= nr < ROWS and 0 <= nc < COLS and (nr, nc) != start_pos:
            target = board[nr][nc]
            if target and target.color != piece_color:
                return True
                
    for dr_adj, dc_adj in ADJACENT_DIRS:
        adj_r, adj_c = end_pos[0] + dr_adj, end_pos[1] + dc_adj
        if 0 <= adj_r < ROWS and 0 <= adj_c < COLS:
            target = board[adj_r][adj_c]
            if isinstance(target, Knight) and target.color != piece_color:
                return True
                
    return False


def is_move_tactical(board, move, is_qsearch_check=False):
    start_pos, end_pos = move
    piece = board[start_pos[0]][start_pos[1]]
    if not piece:
        return False

    if board[end_pos[0]][end_pos[1]] is not None:
        return True
        
    if isinstance(piece, Pawn):
        promo_rank = 0 if piece.color == 'white' else ROWS - 1
        if end_pos[0] == promo_rank:
            return True
    elif isinstance(piece, Rook):
        if _is_rook_path_capture(board, start_pos, end_pos, piece.color):
            return True
    elif isinstance(piece, Knight):
        if _is_knight_evaporation_move(board, start_pos, end_pos, piece.color):
            return True
    
    if is_qsearch_check:
        sim_board = copy_board(board)
        sim_piece = sim_board[start_pos[0]][start_pos[1]]
        sim_piece.move(sim_board, start_pos, end_pos)
        check_evaporation(sim_board)
        if is_in_check(sim_board, piece.opponent_color):
            return True
            
    return False