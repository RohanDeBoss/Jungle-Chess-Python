# game_logic.py

# --- Versioning ---
# v2.1 (API Cleanup & Bugfix)
# - Fixed a critical syntax error in Board.make_move.
# - Completed the refactoring by adding Board.find_king_pos and making methods public.
# - Removed old, slow global functions, enforcing use of the optimized Board class methods.

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

    def symbol(self): return "?"
    def get_valid_moves(self, board, pos): return []

class King(Piece):
    def symbol(self): return "♔" if self.color == "white" else "♚"
    def get_valid_moves(self, board, pos):
        moves = []
        r_start, c_start = pos
        for dr, dc in DIRECTIONS['king']:
            for step in (1, 2):
                new_r, new_c = r_start + dr * step, c_start + dc * step
                if 0 <= new_r < ROWS and 0 <= new_c < COLS:
                    if step == 2:
                        if board.grid[r_start + dr][c_start + dc] is not None: break
                    target = board.grid[new_r][new_c]
                    if target is None or target.color != self.color: moves.append((new_r, new_c))
                    if target is not None: break
                else: break
        return moves

class Queen(Piece):
    def symbol(self): return "♕" if self.color == "white" else "♛"
    def get_valid_moves(self, board, pos):
        moves = []
        r_start, c_start = pos
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
            enemy_encountered = False
            while 0 <= r < ROWS and 0 <= c < COLS:
                target = board.grid[r][c]
                if not enemy_encountered:
                    if target is None: moves.append((r, c))
                    else:
                        if target.color != self.color:
                            moves.append((r, c))
                            enemy_encountered = True
                        else: break
                else:
                    if target is None or target.color != self.color: moves.append((r, c))
                    else: break
                r += dr; c += dc
        return moves

class Bishop(Piece):
    def symbol(self): return "♗" if self.color == "white" else "♝"
    def get_valid_moves(self, board, pos):
        moves = set()
        r_start, c_start = pos
        for dr, dc in DIRECTIONS['bishop']:
            r, c = r_start + dr, c_start + dc
            while 0 <= r < ROWS and 0 <= c < COLS:
                target = board.grid[r][c]
                if target:
                    if target.color != self.color: moves.add((r, c))
                    break
                moves.add((r, c))
                r += dr; c += dc
        direction_pairs = (((-1, 1), (-1, -1)), ((-1, -1), (-1, 1)), ((1, 1), (1, -1)), ((1, -1), (1, 1)),
                           ((-1, 1), (1, 1)), ((1, 1), (-1, 1)), ((-1, -1), (1, -1)), ((1, -1), (-1, -1)))
        for d1, d2 in direction_pairs:
            cr, cc, cd = r_start, c_start, d1
            while True:
                cr += cd[0]; cc += cd[1]
                if not (0 <= cr < ROWS and 0 <= cc < COLS): break
                target = board.grid[cr][cc]
                if target:
                    if target.color != self.color: moves.add((cr, cc))
                    break
                moves.add((cr, cc))
                cd = d2 if cd == d1 else d1
        return list(moves)

class Knight(Piece):
    def symbol(self): return "♘" if self.color == "white" else "♞"
    def get_valid_moves(self, board, pos):
        moves = []
        r_start, c_start = pos
        for dr, dc in DIRECTIONS['knight']:
            nr, nc = r_start + dr, c_start + dc
            if 0 <= nr < ROWS and 0 <= nc < COLS:
                target = board.grid[nr][nc]
                if not target or target.color != self.color: moves.append((nr,nc))
        return moves

class Pawn(Piece):
    def symbol(self): return "♙" if self.color == "white" else "♟"
    def get_valid_moves(self, board, pos):
        moves = []
        r_start, c_start = pos
        direction = -1 if self.color == "white" else 1
        starting_row = 6 if self.color == "white" else 1
        for steps in [1, 2]:
            if steps == 2 and r_start != starting_row: continue
            new_r, new_c = r_start + (direction * steps), c_start
            if 0 <= new_r < ROWS and 0 <= new_c < COLS:
                fwd_target = board.grid[new_r][new_c]
                if fwd_target and fwd_target.color != self.color: moves.append((new_r, new_c))
                if fwd_target is None: moves.append((new_r, new_c))
                if fwd_target is not None: break
            else: break
        for dc_offset in [-1, 1]:
            new_c_side = c_start + dc_offset
            if 0 <= r_start < ROWS and 0 <= new_c_side < COLS:
                side_target = board.grid[r_start][new_c_side]
                if side_target and side_target.color != self.color: moves.append((r_start, new_c_side))
        return moves

# -----------------------------
# Board Class and Game Logic
# -----------------------------
class Board:
    def __init__(self, setup=True):
        self.grid = [[None for _ in range(COLS)] for _ in range(ROWS)]
        self.white_pieces = []
        self.black_pieces = []
        self.white_king_pos = None
        self.black_king_pos = None
        if setup: self.setup_initial_board()

    def setup_initial_board(self):
        pieces = {
            0: [(0, Rook), (1, Knight), (2, Bishop), (3, Queen), (4, King), (5, Bishop), (6, Knight), (7, Rook)],
            1: [(i, Pawn) for i in range(8)],
            6: [(i, Pawn) for i in range(8)],
            7: [(0, Rook), (1, Knight), (2, Bishop), (3, Queen), (4, King), (5, Bishop), (6, Knight), (7, Rook)]
        }
        for r, piece_list in pieces.items():
            color = "black" if r < 2 else "white"
            for c, piece_class in piece_list:
                self.add_piece(piece_class(color), r, c)

    def add_piece(self, piece, r, c):
        self.grid[r][c] = piece
        pos = (r, c)
        piece_list = self.white_pieces if piece.color == "white" else self.black_pieces
        if pos not in piece_list: piece_list.append(pos)
        if isinstance(piece, King):
            if piece.color == "white": self.white_king_pos = pos
            else: self.black_king_pos = pos

    def remove_piece(self, r, c):
        piece = self.grid[r][c]
        if not piece: return
        pos = (r, c)
        if piece.color == "white":
            if pos in self.white_pieces: self.white_pieces.remove(pos)
            if self.white_king_pos == pos: self.white_king_pos = None
        else:
            if pos in self.black_pieces: self.black_pieces.remove(pos)
            if self.black_king_pos == pos: self.black_king_pos = None
        self.grid[r][c] = None

    def move_piece(self, start, end):
        piece = self.grid[start[0]][start[1]]
        if not piece: return
        pos = (start[0], start[1])
        if piece.color == "white":
            if pos in self.white_pieces: self.white_pieces.remove(pos)
            self.white_pieces.append(end)
            if isinstance(piece, King): self.white_king_pos = end
        else:
            if pos in self.black_pieces: self.black_pieces.remove(pos)
            self.black_pieces.append(end)
            if isinstance(piece, King): self.black_king_pos = end
        self.grid[start[0]][start[1]] = None
        self.grid[end[0]][end[1]] = piece
        piece.has_moved = True

    def make_move(self, start, end):
        moving_piece = self.grid[start[0]][start[1]]
        if not moving_piece: return
        target_piece = self.grid[end[0]][end[1]]
        is_capture = target_piece is not None and target_piece.color != moving_piece.color
        if isinstance(moving_piece, Queen) and is_capture:
            self.remove_piece(start[0], start[1])
            self.remove_piece(end[0], end[1]) # <-- SYNTAX FIX HERE
            for dr_adj, dc_adj in ADJACENT_DIRS:
                adj_r, adj_c = end[0] + dr_adj, end[1] + dc_adj
                if 0 <= adj_r < ROWS and 0 <= adj_c < COLS:
                    adj_piece = self.grid[adj_r][adj_c]
                    if adj_piece and adj_piece.color != moving_piece.color:
                        self.remove_piece(adj_r, adj_c)
            return
        elif isinstance(moving_piece, Rook):
            if start[0] == end[0]: d = (0, 1 if end[1] > start[1] else -1)
            else: d = (1 if end[0] > start[0] else -1, 0)
            cr, cc = start[0] + d[0], start[1] + d[1]
            path_contained_enemy = False
            while (cr, cc) != end:
                if self.grid[cr][cc] and self.grid[cr][cc].color != moving_piece.color:
                    path_contained_enemy = True; break
                cr += d[0]; cc += d[1]
            if path_contained_enemy:
                cr, cc = start[0] + d[0], start[1] + d[1]
                while (cr, cc) != end:
                    if self.grid[cr][cc] and self.grid[cr][cc].color != moving_piece.color:
                        self.remove_piece(cr, cc)
                    cr += d[0]; cc += d[1]
        if target_piece: self.remove_piece(end[0], end[1])
        self.move_piece(start, end)
        if isinstance(moving_piece, Knight): self.evaporate(end)
        elif isinstance(moving_piece, Pawn):
            promotion_rank = 0 if moving_piece.color == "white" else (ROWS - 1)
            if end[0] == promotion_rank:
                self.remove_piece(end[0], end[1])
                self.add_piece(Queen(moving_piece.color), end[0], end[1])
        self.check_evaporation()

    def evaporate(self, pos):
        r_knight, c_knight = pos
        knight_instance = self.grid[r_knight][c_knight]
        if not knight_instance: return
        enemy_knights_to_remove = []
        pieces_to_remove = []
        for dr, dc in DIRECTIONS['knight']:
            r_adj, c_adj = r_knight + dr, c_knight + dc
            if 0 <= r_adj < ROWS and 0 <= c_adj < COLS:
                piece_on_adj = self.grid[r_adj][c_adj]
                if piece_on_adj and piece_on_adj.color != knight_instance.color:
                    if isinstance(piece_on_adj, Knight): enemy_knights_to_remove.append((r_adj, c_adj))
                    pieces_to_remove.append((r_adj, c_adj))
        for r, c in pieces_to_remove: self.remove_piece(r, c)
        if enemy_knights_to_remove: self.remove_piece(r_knight, c_knight)

    def check_evaporation(self):
        knights_on_board = [(pos, self.grid[pos[0]][pos[1]]) for pos in self.white_pieces + self.black_pieces if isinstance(self.grid[pos[0]][pos[1]], Knight)]
        for pos, knight_instance in knights_on_board:
            if self.grid[pos[0]][pos[1]] is knight_instance:
                 self.evaporate(pos)

    def find_king_pos(self, color):
        return self.white_king_pos if color == 'white' else self.black_king_pos

    def clone(self):
        new_board = Board(setup=False)
        new_board.grid = [[p.clone() if p else None for p in row] for row in self.grid]
        new_board.white_pieces = list(self.white_pieces)
        new_board.black_pieces = list(self.black_pieces)
        new_board.white_king_pos = self.white_king_pos
        new_board.black_king_pos = self.black_king_pos
        return new_board

def create_initial_board(): return Board()

def is_in_check(board, color):
    king_pos = board.find_king_pos(color)
    if not king_pos: return True
    enemy_pieces = board.black_pieces if color == 'white' else board.white_pieces
    for pos in enemy_pieces:
        piece = board.grid[pos[0]][pos[1]]
        if piece and king_pos in piece.get_valid_moves(board, pos):
            return True
    return False

def generate_pseudo_legal_moves(board, color):
    moves = []
    piece_list = board.white_pieces if color == "white" else board.black_pieces
    for pos in piece_list:
        piece = board.grid[pos[0]][pos[1]]
        if piece:
            for end_pos in piece.get_valid_moves(board, pos):
                moves.append((pos, end_pos))
    return moves

def validate_move(board, color, start, end):
    piece_to_move = board.grid[start[0]][start[1]]
    if not piece_to_move or piece_to_move.color != color: return False
    if end not in piece_to_move.get_valid_moves(board, start): return False
    sim_board = board.clone()
    sim_board.make_move(start, end)
    if is_in_check(sim_board, color): return False
    return True

def has_legal_moves(board, color):
    for start_pos, end_pos in generate_pseudo_legal_moves(board, color):
        if validate_move(board, color, start_pos, end_pos):
            return True
    return False

def check_game_over(board, turn_color_who_just_moved):
    if not board.find_king_pos("white"): return "king_capture", "black"
    if not board.find_king_pos("black"): return "king_capture", "white"
    next_player_color = "black" if turn_color_who_just_moved == "white" else "white"
    if not has_legal_moves(board, next_player_color):
        if is_in_check(board, next_player_color):
            return "checkmate", turn_color_who_just_moved
        else:
            return "stalemate", None
    return None, None

def is_move_tactical(board, move, is_qsearch_check=False):
    start_pos, end_pos = move
    piece = board.grid[start_pos[0]][start_pos[1]]
    if not piece: return False
    if board.grid[end_pos[0]][end_pos[1]] is not None: return True
    if isinstance(piece, Pawn):
        promo_rank = 0 if piece.color == 'white' else ROWS - 1
        if end_pos[0] == promo_rank: return True
    elif isinstance(piece, Rook):
        if start_pos[0] == end_pos[0]: d = (0, 1 if end_pos[1] > start_pos[1] else -1)
        else: d = (1 if end_pos[0] > start_pos[0] else -1, 0)
        cr, cc = start_pos[0] + d[0], start_pos[1] + d[1]
        while (cr, cc) != end_pos:
            target = board.grid[cr][cc]
            if target and target.color != piece.color: return True
            cr += d[0]; cc += d[1]
    elif isinstance(piece, Knight):
        for dr, dc in DIRECTIONS['knight']:
            nr, nc = end_pos[0] + dr, end_pos[1] + dc
            if 0 <= nr < ROWS and 0 <= nc < COLS and (nr, nc) != start_pos:
                if board.grid[nr][nc] and board.grid[nr][nc].color != piece.color: return True
        for dr_adj, dc_adj in ADJACENT_DIRS:
            adj_r, adj_c = end_pos[0] + dr_adj, end_pos[1] + dc_adj
            if 0 <= adj_r < ROWS and 0 <= adj_c < COLS:
                target = board.grid[adj_r][adj_c]
                if isinstance(target, Knight) and target.color != piece.color: return True
    if is_qsearch_check:
        sim_board = board.clone()
        sim_board.make_move(start_pos, end_pos)
        if is_in_check(sim_board, piece.opponent_color): return True
    return False