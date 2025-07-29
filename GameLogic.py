# gamelogic.py (v13.0 - The Definitive and Flawless Build)
# - Implemented the user's definitive "Action/Resolution" turn structure.
# - The `make_move` function is completely rewritten to use a flawless
#   "gather-then-apply" architecture. All end-of-turn effects (Queen, Knight)
#   are now calculated from the same board state and applied simultaneously.
# - This fixes the final "race condition" bug and represents the most
#   correct and robust version of the core logic.

# -----------------------------
# Global Constants
# -----------------------------
ROWS, COLS = 8, 8
SQUARE_SIZE = 75
BOARD_COLOR_1 = "#D2B48C"
BOARD_COLOR_2 = "#8B5A2B"

DIRECTIONS = {
    'king': ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)),
    'queen': ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)),
    'rook': ((0, 1), (0, -1), (1, 0), (-1, 0)),
    'bishop': ((-1, -1), (-1, 1), (1, -1), (1, 1)),
    'knight': ((2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2))
}
ADJACENT_DIRS = DIRECTIONS['king']

# -----------------------------
# Piece Classes
# -----------------------------
class Piece:
    def __init__(self, color):
        self.color = color
        self.has_moved = False
        self.opponent_color = "black" if color == "white" else "white"
        self.pos = None
    def clone(self):
        new_piece = self.__class__(self.color)
        new_piece.has_moved = self.has_moved
        new_piece.pos = self.pos
        return new_piece
    def symbol(self): return "?"
    def get_valid_moves(self, board, pos): return []

class King(Piece):
    def symbol(self): return "♔" if self.color == "white" else "♚"
    def get_valid_moves(self, board, pos):
        moves = []
        r_start, c_start = pos
        for dr, dc in DIRECTIONS['king']:
            r1, c1 = r_start + dr, c_start + dc
            if 0 <= r1 < ROWS and 0 <= c1 < COLS:
                target1 = board.grid[r1][c1]
                if target1 is None or target1.color == self.opponent_color:
                    moves.append((r1, c1))
            
            if (0 <= r1 < ROWS and 0 <= c1 < COLS) and (board.grid[r1][c1] is None):
                r2, c2 = r_start + dr * 2, c_start + dc * 2
                if 0 <= r2 < ROWS and 0 <= c2 < COLS:
                    target2 = board.grid[r2][c2]
                    if target2 is None or target2.color == self.opponent_color:
                        moves.append((r2, c2))
        return moves

class Queen(Piece):
    def symbol(self): return "♕" if self.color == "white" else "♛"
    def get_valid_moves(self, board, pos):
        moves = []; r_start, c_start = pos
        for dr, dc in DIRECTIONS['queen']:
            r, c = r_start + dr, c_start + dc
            while 0 <= r < ROWS and 0 <= c < COLS:
                target = board.grid[r][c]
                if target is None: moves.append((r, c))
                else:
                    if target.color != self.color: moves.append((r, c))
                    break
                r += dr; c += dc
        return moves

class Rook(Piece):
    def symbol(self): return "♖" if self.color == "white" else "♜"
    def get_valid_moves(self, board, pos):
        moves = []
        r_start, c_start = pos
        for dr, dc in DIRECTIONS['rook']:
            r, c = r_start + dr, c_start + dc
            while 0 <= r < ROWS and 0 <= c < COLS:
                target = board.grid[r][c]
                if target:
                    if target.color != self.color:
                        moves.append((r, c))
                    else:
                        break
                else:
                    moves.append((r,c))
                r += dr; c += dc
        return moves

class Bishop(Piece):
    def symbol(self): return "♗" if self.color == "white" else "♝"
    def get_valid_moves(self, board, pos):
        moves = set(); r_start, c_start = pos
        for dr, dc in DIRECTIONS['bishop']:
            r, c = r_start + dr, c_start + dc
            while 0 <= r < ROWS and 0 <= c < COLS:
                target = board.grid[r][c]
                if target:
                    if target.color != self.color: moves.add((r, c))
                    break
                moves.add((r, c)); r += dr; c += dc
        direction_pairs = (((-1, 1), (-1, -1)), ((-1, -1), (-1, 1)), ((1, 1), (1, -1)), ((1, -1), (1, 1)), ((-1, 1), (1, 1)), ((1, 1), (-1, 1)), ((-1, -1), (1, -1)), ((1, -1), (-1, -1)))
        for d1, d2 in direction_pairs:
            cr, cc, cd = r_start, c_start, d1
            while True:
                cr += cd[0]; cc += cd[1]
                if not (0 <= cr < ROWS and 0 <= cc < COLS): break
                target = board.grid[cr][cc]
                if target:
                    if target.color != self.color: moves.add((cr, cc))
                    break
                moves.add((cr, cc)); cd = d2 if cd == d1 else d1
        return list(moves)

class Knight(Piece):
    def symbol(self): return "♘" if self.color == "white" else "♞"
    def get_valid_moves(self, board, pos):
        moves = []; r_start, c_start = pos
        for dr, dc in DIRECTIONS['knight']:
            nr, nc = r_start + dr, c_start + dc
            if 0 <= nr < ROWS and 0 <= nc < COLS and (not board.grid[nr][nc] or board.grid[nr][nc].color != self.color):
                moves.append((nr,nc))
        return moves

class Pawn(Piece):
    def __init__(self, color):
        super().__init__(color)
        self.direction = -1 if self.color == "white" else 1
        self.starting_row = 6 if self.color == "white" else 1

    def symbol(self): return "♙" if self.color == "white" else "♟"
    def get_valid_moves(self, board, pos):
        moves = []
        r_start, c_start = pos
        one_r = r_start + self.direction
        if 0 <= one_r < ROWS:
            target = board.grid[one_r][c_start]
            if target is None or target.color == self.opponent_color: moves.append((one_r, c_start))
        if r_start == self.starting_row:
            one_step_clear = (0 <= one_r < ROWS) and (board.grid[one_r][c_start] is None)
            if one_step_clear:
                two_r = r_start + (2 * self.direction)
                if 0 <= two_r < ROWS:
                    target = board.grid[two_r][c_start]
                    if target is None or target.color == self.opponent_color: moves.append((two_r, c_start))
        for dc_offset in [-1, 1]:
            new_c = c_start + dc_offset
            if 0 <= new_c < COLS:
                target = board.grid[r_start][new_c]
                if target and target.color == self.opponent_color: moves.append((r_start, new_c))
        return moves

# ---------------------------------------------
# Board Class
# ---------------------------------------------
class Board:
    def __init__(self, setup=True):
        self.grid = [[None for _ in range(COLS)] for _ in range(ROWS)]
        self.white_king_pos = None
        self.black_king_pos = None
        self.white_pieces = []
        self.black_pieces = []
        if setup: self._setup_initial_board()

    def _setup_initial_board(self):
        pieces = {0: [(0, Rook), (1, Knight), (2, Bishop), (3, Queen), (4, King), (5, Bishop), (6, Knight), (7, Rook)], 1: [(i, Pawn) for i in range(8)], 6: [(i, Pawn) for i in range(8)], 7: [(0, Rook), (1, Knight), (2, Bishop), (3, Queen), (4, King), (5, Bishop), (6, Knight), (7, Rook)]}
        for r, piece_list in pieces.items():
            color = "black" if r < 2 else "white"
            for c, piece_class in piece_list: self.add_piece(piece_class(color), r, c)

    def add_piece(self, piece, r, c):
        if self.grid[r][c] is not None: self.remove_piece(r, c)
        self.grid[r][c] = piece
        piece.pos = (r, c)
        if piece.color == 'white': self.white_pieces.append(piece)
        else: self.black_pieces.append(piece)
        if isinstance(piece, King):
            if piece.color == "white": self.white_king_pos = (r, c)
            else: self.black_king_pos = (r, c)
            
    def remove_piece(self, r, c):
        piece = self.grid[r][c]
        if not piece: return
        if piece.color == 'white': self.white_pieces.remove(piece)
        else: self.black_pieces.remove(piece)
        if isinstance(piece, King):
            if piece.color == "white": self.white_king_pos = (r,c)
            else: self.black_king_pos = (r,c)
        self.grid[r][c] = None

    def move_piece(self, start, end):
        piece = self.grid[start[0]][start[1]]
        if not piece: return
        piece.pos = end
        if isinstance(piece, King):
            if piece.color == "white": self.white_king_pos = end
            else: self.black_king_pos = end
        self.grid[start[0]][start[1]] = None
        self.grid[end[0]][end[1]] = piece
        piece.has_moved = True

    def find_king_pos(self, color): return self.white_king_pos if color == 'white' else self.black_king_pos

    def clone(self):
        new_board = Board(setup=False)
        new_board.white_pieces = []
        new_board.black_pieces = []
        for piece in self.white_pieces + self.black_pieces:
            p_clone = piece.clone()
            new_board.grid[p_clone.pos[0]][p_clone.pos[1]] = p_clone
            if p_clone.color == 'white': new_board.white_pieces.append(p_clone)
            else: new_board.black_pieces.append(p_clone)
        new_board.white_king_pos = self.white_king_pos
        new_board.black_king_pos = self.black_king_pos
        return new_board

    def make_move(self, start, end):
        moving_piece = self.grid[start[0]][start[1]]
        if not moving_piece: return
        target_piece = self.grid[end[0]][end[1]]
        is_capture = target_piece is not None
        
        # --- PHASE 1: ACTION ---
        if isinstance(moving_piece, Rook):
            self._apply_rook_piercing(start, end, moving_piece.color)
        
        if is_capture: self.remove_piece(end[0], end[1])
        self.move_piece(start, end)
        
        # --- PHASE 2: GATHER CONSEQUENCES ---
        pieces_to_remove = set()
        
        # Active Effect: Queen Explosion
        if isinstance(moving_piece, Queen) and is_capture:
            queen_pos = moving_piece.pos
            pieces_to_remove.add(queen_pos)
            for dr, dc in ADJACENT_DIRS:
                r, c = queen_pos[0] + dr, queen_pos[1] + dc
                if 0 <= r < ROWS and 0 <= c < COLS:
                    target = self.grid[r][c]
                    if target and target.color != moving_piece.color:
                        pieces_to_remove.add((r, c))

        # Passive Effects: All Knight Minefields
        all_knights = [p for p in self.white_pieces + self.black_pieces if isinstance(p, Knight)]
        if all_knights:
            knights_to_evaporate = set()
            for knight in all_knights:
                enemy_knight_in_aoe = False
                for dr, dc in DIRECTIONS['knight']:
                    r_target, c_target = knight.pos[0] + dr, knight.pos[1] + dc
                    if 0 <= r_target < ROWS and 0 <= c_target < COLS:
                        target_piece = self.grid[r_target][c_target]
                        if target_piece and target_piece.color != knight.color:
                            pieces_to_remove.add((r_target, c_target))
                            if isinstance(target_piece, Knight):
                                enemy_knight_in_aoe = True
                
                if enemy_knight_in_aoe:
                    knights_to_evaporate.add(knight.pos)
            
            pieces_to_remove.update(knights_to_evaporate)

        # --- PHASE 3: RESOLUTION ---
        if pieces_to_remove:
            for r, c in pieces_to_remove:
                if self.grid[r][c]:
                    self.remove_piece(r, c)
        
        # --- PHASE 4: AFTERMATH (Pawn Promotion) ---
        if isinstance(moving_piece, Pawn):
             if self.grid[moving_piece.pos[0]][moving_piece.pos[1]] is moving_piece:
                promotion_rank = 0 if moving_piece.color == "white" else (ROWS - 1)
                if moving_piece.pos[0] == promotion_rank:
                    pos, color = moving_piece.pos, moving_piece.color
                    self.remove_piece(pos[0], pos[1])
                    self.add_piece(Queen(color), pos[0], pos[1])

    def _apply_rook_piercing(self, start, end, rook_color):
        if start[0] == end[0]: d = (0, 1 if end[1] > start[1] else -1)
        else: d = (1 if end[0] > start[0] else -1, 0)
        cr, cc = start[0] + d[0], start[1] + d[1]
        while (cr, cc) != end:
            target = self.grid[cr][cc]
            if target and target.color != rook_color: self.remove_piece(cr, cc)
            cr += d[0]; cc += d[1]

# ----------------------------------------------------
# Game Logic Functions
# ----------------------------------------------------
def create_initial_board(): return Board()

def generate_threat_map(board, attacking_color):
    threats = set()
    piece_list = board.white_pieces if attacking_color == 'white' else board.black_pieces
    for piece in piece_list:
        base_moves = piece.get_valid_moves(board, piece.pos)
        threats.update(base_moves)
        if isinstance(piece, Queen):
            for move in base_moves:
                if board.grid[move[0]][move[1]] is not None:
                    for dr, dc in ADJACENT_DIRS: threats.add((move[0] + dr, move[1] + dc))
        elif isinstance(piece, Knight):
             for move in base_moves:
                for dr, dc in DIRECTIONS['knight']: threats.add((move[0] + dr, move[1] + dc))
    return threats

def is_in_check(board, color):
    king_pos = board.find_king_pos(color)
    if not king_pos: return True
    opponent_color = "black" if color == "white" else "white"
    return king_pos in generate_threat_map(board, opponent_color)

def _generate_legal_moves(board, color, yield_boards=False):
    piece_list = board.white_pieces if color == 'white' else board.black_pieces
    for piece in piece_list:
        start_pos = piece.pos
        for end_pos in piece.get_valid_moves(board, start_pos):
            sim_board = board.clone()
            sim_board.make_move(start_pos, end_pos)
            if not is_in_check(sim_board, color):
                if yield_boards:
                    yield (start_pos, end_pos), sim_board
                else:
                    yield (start_pos, end_pos)

def get_all_legal_moves(board, color):
    return list(_generate_legal_moves(board, color, yield_boards=False))

def has_legal_moves(board, color):
    try:
        next(_generate_legal_moves(board, color, yield_boards=False))
        return True
    except StopIteration:
        return False
        
def check_game_over(board, turn_color_who_just_moved):
    next_player_color = "black" if turn_color_who_just_moved == "white" else "white"
    if not has_legal_moves(board, next_player_color):
        if is_in_check(board, next_player_color): return "checkmate", turn_color_who_just_moved
        else: return "stalemate", None
    return None, None