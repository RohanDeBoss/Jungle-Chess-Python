# gamelogic.py (v6.0 - Definitive Performance & Correctness)
# - Implemented Knight position tracking on the Board object, similar to the
#   King tracker. This is the definitive optimization.
# - The slow, 64-square scan for Knights has been eliminated. The Knight
#   resolution logic now uses the tracked position lists, providing a massive
#   performance boost while retaining the critical "simultaneous effects" bugfix.

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
    def clone(self):
        new_piece = self.__class__(self.color); new_piece.has_moved = self.has_moved; return new_piece
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
        self.white_king_pos = None; self.black_king_pos = None
        
        self.white_knight_positions = []
        self.black_knight_positions = []

        if setup: self._setup_initial_board()

    def _setup_initial_board(self):
        pieces = {0: [(0, Rook), (1, Knight), (2, Bishop), (3, Queen), (4, King), (5, Bishop), (6, Knight), (7, Rook)], 1: [(i, Pawn) for i in range(8)], 6: [(i, Pawn) for i in range(8)], 7: [(0, Rook), (1, Knight), (2, Bishop), (3, Queen), (4, King), (5, Bishop), (6, Knight), (7, Rook)]}
        for r, piece_list in pieces.items():
            color = "black" if r < 2 else "white"
            for c, piece_class in piece_list: self.add_piece(piece_class(color), r, c)

    def add_piece(self, piece, r, c):
        self.grid[r][c] = piece
        if isinstance(piece, King):
            if piece.color == "white": self.white_king_pos = (r, c)
            else: self.black_king_pos = (r, c)
        elif isinstance(piece, Knight):
            if piece.color == "white": self.white_knight_positions.append((r, c))
            else: self.black_knight_positions.append((r, c))
            
    def remove_piece(self, r, c):
        piece = self.grid[r][c]
        if not piece: return
        pos = (r, c)
        
        if isinstance(piece, King):
            if piece.color == "white" and self.white_king_pos == pos: self.white_king_pos = None
            elif piece.color == "black" and self.black_king_pos == pos: self.black_king_pos = None
        elif isinstance(piece, Knight):
            if piece.color == "white": self.white_knight_positions.remove(pos)
            else: self.black_knight_positions.remove(pos)
            
        self.grid[r][c] = None

    def move_piece(self, start, end):
        piece = self.grid[start[0]][start[1]]
        if not piece: return
        
        # Update position trackers BEFORE the move
        if isinstance(piece, King):
            if piece.color == "white": self.white_king_pos = end
            else: self.black_king_pos = end
        elif isinstance(piece, Knight):
            if piece.color == "white": self.white_knight_positions.remove(start); self.white_knight_positions.append(end)
            else: self.black_knight_positions.remove(start); self.black_knight_positions.append(end)
        
        self.grid[start[0]][start[1]] = None
        self.grid[end[0]][end[1]] = piece
        piece.has_moved = True

    def find_king_pos(self, color): return self.white_king_pos if color == 'white' else self.black_king_pos

    def clone(self):
        new_board = Board(setup=False)
        new_board.grid = [[p.clone() if p else None for p in row] for row in self.grid]
        new_board.white_king_pos = self.white_king_pos
        new_board.black_king_pos = self.black_king_pos
        new_board.white_knight_positions = self.white_knight_positions[:]
        new_board.black_knight_positions = self.black_knight_positions[:]
        return new_board

    def make_move(self, start, end):
        moving_piece = self.grid[start[0]][start[1]]
        if not moving_piece: return
        target_piece = self.grid[end[0]][end[1]]
        is_capture = target_piece is not None
        
        if isinstance(moving_piece, Rook):
            self._apply_rook_piercing(start, end, moving_piece.color)
        
        if is_capture: self.remove_piece(end[0], end[1])
        self.move_piece(start, end)
        
        if isinstance(moving_piece, Queen) and is_capture:
            self._apply_queen_aoe(end, moving_piece.color)
        elif isinstance(moving_piece, Pawn):
            promotion_rank = 0 if moving_piece.color == "white" else (ROWS - 1)
            if end[0] == promotion_rank:
                self.remove_piece(end[0], end[1])
                self.add_piece(Queen(moving_piece.color), end[0], end[1])
        
        if self.white_knight_positions or self.black_knight_positions:
            self._resolve_all_knight_effects()

    def _apply_queen_aoe(self, pos, queen_color):
        self.remove_piece(pos[0], pos[1])
        for dr, dc in ADJACENT_DIRS:
            adj_r, adj_c = pos[0] + dr, pos[1] + dc
            if 0 <= adj_r < ROWS and 0 <= adj_c < COLS:
                adj_piece = self.grid[adj_r][adj_c]
                if adj_piece and adj_piece.color != queen_color: self.remove_piece(adj_r, adj_c)

    def _apply_rook_piercing(self, start, end, rook_color):
        if start[0] == end[0]: d = (0, 1 if end[1] > start[1] else -1)
        else: d = (1 if end[0] > start[0] else -1, 0)
        cr, cc = start[0] + d[0], start[1] + d[1]
        while (cr, cc) != end:
            target = self.grid[cr][cc]
            if target and target.color != rook_color: self.remove_piece(cr, cc)
            cr += d[0]; cc += d[1]

    def _resolve_all_knight_effects(self):
        all_knight_positions = self.white_knight_positions + self.black_knight_positions
        if not all_knight_positions: return

        pieces_to_remove = set()
        knights_to_evaporate = set()
        
        for pos in all_knight_positions:
            knight = self.grid[pos[0]][pos[1]]
            if not knight: continue # Should not happen if lists are correct

            enemy_knight_in_aoe = False
            aoe_targets = []
            
            for dr, dc in DIRECTIONS['knight']:
                r_target, c_target = pos[0] + dr, pos[1] + dc
                if 0 <= r_target < ROWS and 0 <= c_target < COLS:
                    target_piece = self.grid[r_target][c_target]
                    if target_piece and target_piece.color != knight.color:
                        aoe_targets.append((r_target, c_target))
                        if isinstance(target_piece, Knight):
                            enemy_knight_in_aoe = True
            
            if aoe_targets:
                for target_pos in aoe_targets:
                    pieces_to_remove.add(target_pos)
            
            if enemy_knight_in_aoe:
                knights_to_evaporate.add(pos)

        # Apply all removals in a single pass to ensure simultaneity
        # Combine the sets, handling cases where a Knight is both a target and evaporating
        final_removals = pieces_to_remove.union(knights_to_evaporate)

        for r, c in final_removals:
            self.remove_piece(r, c)

# ----------------------------------------------------
# Game Logic Functions
# ----------------------------------------------------
def create_initial_board(): return Board()
def generate_threat_map(board, attacking_color):
    threats = set()
    for r_start in range(ROWS):
        for c_start in range(COLS):
            piece = board.grid[r_start][c_start]
            if not piece or piece.color != attacking_color: continue
            pos = (r_start, c_start)
            base_moves = piece.get_valid_moves(board, pos)
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
    for r in range(ROWS):
        for c in range(COLS):
            piece = board.grid[r][c]
            if piece and piece.color == color:
                start_pos = (r, c)
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