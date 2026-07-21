# GameLogic.py (v67.1 - precalculated knight rays with free performance)


# -----------------------------------------------------------------------
# Global constants
# z_idx reference: 0=Pawn 1=Knight 2=Bishop 3=Rook 4=Queen(kamikaze) 5=King
# -----------------------------------------------------------------------
ROWS, COLS = 8, 8
SQUARE_SIZE = 75
BOARD_COLOR_1 = "#D2B48C"
BOARD_COLOR_2 = "#8B5A2B"

DIRECTIONS = {
    'king':   ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)),
    'queen':  ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)),
    'rook':   ((0, 1), (0, -1), (1, 0), (-1, 0)),
    'bishop': ((-1, -1), (-1, 1), (1, -1), (1, 1)),
    'knight': ((2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)),
}
ADJACENT_DIRS = DIRECTIONS['king']

BISHOP_ZIGZAG_DIRS = (
    ((-1, 1), (-1, -1)), ((-1, -1), (-1, 1)),
    ((1, 1),  (1, -1)),  ((1, -1),  (1, 1)),
    ((-1, 1), (1, 1)),   ((1, 1),   (-1, 1)),
    ((-1, -1), (1, -1)), ((1, -1),  (-1, -1)),
)

# Stored as tuples so CPython uses the faster tuple-iteration C path.
KNIGHT_ATTACKS_FROM = {
    (r, c): tuple(
        (r + dr, c + dc) for dr, dc in DIRECTIONS['knight']
        if 0 <= r + dr < ROWS and 0 <= c + dc < COLS
    )
    for r in range(ROWS) for c in range(COLS)
}
ADJACENT_SQUARES_MAP = {
    (r, c): tuple(
        (r + dr, c + dc) for dr, dc in ADJACENT_DIRS
        if 0 <= r + dr < ROWS and 0 <= c + dc < COLS
    )
    for r in range(ROWS) for c in range(COLS)
}

# RAYS[sq_index][direction_index] — inner sequences are tuples (read-only)
RAYS = [[None] * 8 for _ in range(64)]
BISHOP_ZIGZAG_RAYS = [None] * 64

def _init_rays():
    dy_dx = [(-1, 0), (1, 0), (0, 1), (0, -1), (-1, 1), (-1, -1), (1, 1), (1, -1)]
    tmp = [[[] for _ in range(8)] for _ in range(64)]
    zigzag_tmp = [[[] for _ in range(len(BISHOP_ZIGZAG_DIRS))] for _ in range(64)]
    for r in range(ROWS):
        for c in range(COLS):
            idx = r * COLS + c
            for i, (dr, dc) in enumerate(dy_dx):
                cr, cc = r + dr, c + dc
                while 0 <= cr < ROWS and 0 <= cc < COLS:
                    tmp[idx][i].append((cr, cc))
                    cr += dr
                    cc += dc
            for i, (d1, d2) in enumerate(BISHOP_ZIGZAG_DIRS):
                cr, cc, cd = r, c, d1
                while True:
                    cr += cd[0]
                    cc += cd[1]
                    if not (0 <= cr < ROWS and 0 <= cc < COLS):
                        break
                    zigzag_tmp[idx][i].append((cr, cc))
                    cd = d2 if cd == d1 else d1
    for sq in range(64):
        RAYS[sq] = tuple(tuple(ray) for ray in tmp[sq])
        BISHOP_ZIGZAG_RAYS[sq] = tuple(tuple(ray) for ray in zigzag_tmp[sq])

_init_rays()


# --- PRECOMPUTED KNIGHT EVAPORATION TABLE ---
KNIGHT_EVAP_SQUARES = [[None] * 64 for _ in range(64)]

def _init_knight_evap():
    for r1 in range(8):
        for c1 in range(8):
            idx1 = r1 * 8 + c1
            jumps1 = set(KNIGHT_ATTACKS_FROM[(r1, c1)])
            for r2 in range(8):
                for c2 in range(8):
                    idx2 = r2 * 8 + c2
                    if idx1 == idx2:
                        continue
                    sq2 = (r2, c2)
                    
                    # 1. Direct threat (1 jump away)
                    if sq2 in jumps1:
                        KNIGHT_EVAP_SQUARES[idx1][idx2] = True
                    else:
                        # 2. Indirect threat (2 jumps away - precompute shared intermediate squares)
                        jumps2 = set(KNIGHT_ATTACKS_FROM[sq2])
                        shared = jumps1.intersection(jumps2)
                        if shared:
                            KNIGHT_EVAP_SQUARES[idx1][idx2] = tuple(shared)

_init_knight_evap()


def _clone_piece_fast(piece):
    cls       = piece.__class__
    new_piece = cls.__new__(cls)
    new_piece.color          = piece.color
    new_piece.opponent_color = piece.opponent_color
    new_piece.pos            = piece.pos
    new_piece._list_pos      = piece._list_pos   # preserve index in new board's list
    if cls is Pawn:
        new_piece.direction    = piece.direction
        new_piece.starting_row = piece.starting_row
        new_piece.promo_rank   = piece.promo_rank
    elif cls is Knight:
        new_piece._knight_list_pos = getattr(piece, '_knight_list_pos', -1)
    return new_piece


# -----------------------------------------------------------------------
# Piece classes
# -----------------------------------------------------------------------
class Piece:
    def __init__(self, color):
        self.color          = color
        self.opponent_color = "black" if color == "white" else "white"
        self.pos            = None
        self._list_pos      = -1   # index in Board.white_pieces / black_pieces

    def clone(self):
        new_piece     = self.__class__(self.color)
        new_piece.pos = self.pos
        return new_piece

    def symbol(self):                        return "?"
    def get_valid_moves(self, board, pos):   return []


class King(Piece):
    z_idx = 5
    def symbol(self): return "♔" if self.color == "white" else "♚"

    def get_valid_moves(self, board, pos):
        moves = []
        r_start, c_start = pos
        opp = self.opponent_color
        grid = board.grid
        for dr, dc in DIRECTIONS['king']:
            r1, c1 = r_start + dr, c_start + dc
            if 0 <= r1 < ROWS and 0 <= c1 < COLS:
                target = grid[r1][c1]
                if target is None or target.color == opp:
                    moves.append((pos, (r1, c1)))
                    if target is None:
                        r2, c2 = r1 + dr, c1 + dc
                        if 0 <= r2 < ROWS and 0 <= c2 < COLS:
                            t2 = grid[r2][c2]
                            if t2 is None or t2.color == opp:
                                moves.append((pos, (r2, c2)))
        return moves


class Queen(Piece):
    z_idx = 4
    def symbol(self): return "♕" if self.color == "white" else "♛"

    def get_valid_moves(self, board, pos):
        moves = []
        grid = board.grid
        start_index = pos[0] * COLS + pos[1]
        for i in range(8):
            for r, c in RAYS[start_index][i]:
                target = grid[r][c]
                if target is None:
                    moves.append((pos, (r, c)))
                else:
                    if target.color != self.color:
                        moves.append((pos, (r, c)))
                    break
        return moves


class Rook(Piece):
    z_idx = 3
    def symbol(self): return "♖" if self.color == "white" else "♜"

    def get_valid_moves(self, board, pos):
        moves = []
        grid = board.grid
        start_index = pos[0] * COLS + pos[1]
        for i in range(4):
            for r, c in RAYS[start_index][i]:
                target = grid[r][c]
                if target and target.color == self.color:
                    break
                moves.append((pos, (r, c)))
        return moves


class Bishop(Piece):
    z_idx = 2
    def symbol(self): return "♗" if self.color == "white" else "♝"

    def get_valid_moves(self, board, pos):
        moves       = set()
        grid        = board.grid
        r_start, c_start = pos
        start_index = r_start * COLS + c_start

        for i in range(4, 8):
            for r, c in RAYS[start_index][i]:
                target = grid[r][c]
                if target:
                    if target.color != self.color:
                        moves.add((pos, (r, c)))
                    break
                moves.add((pos, (r, c)))

        for ray in BISHOP_ZIGZAG_RAYS[start_index]:
            for r, c in ray:
                target = grid[r][c]
                if target:
                    if target.color != self.color:
                        moves.add((pos, (r, c)))
                    break
                moves.add((pos, (r, c)))

        return sorted(moves, key=lambda m: m[1])


class Knight(Piece):
    z_idx = 1
    def symbol(self): return "♘" if self.color == "white" else "♞"

    def get_valid_moves(self, board, pos):
        moves = []
        grid = board.grid
        for r, c in KNIGHT_ATTACKS_FROM[pos]:
            if grid[r][c] is None:
                moves.append((pos, (r, c)))
        return moves


class Pawn(Piece):
    z_idx = 0
    def __init__(self, color):
        super().__init__(color)
        self.direction    = -1 if color == "white" else 1
        self.starting_row = 6  if color == "white" else 1
        self.promo_rank   = 0  if color == "white" else ROWS - 1

    def symbol(self): return "♙" if self.color == "white" else "♟"

    def get_valid_moves(self, board, pos):
        moves = []
        r, c      = pos
        direction = self.direction
        grid      = board.grid
        opp       = self.opponent_color

        one_r = r + direction
        if 0 <= one_r < ROWS:
            target1 = grid[one_r][c]
            if target1 is None or target1.color == opp:
                moves.append((pos, (one_r, c)))
                if r == self.starting_row and target1 is None:
                    two_r   = r + (2 * direction)
                    target2 = grid[two_r][c]
                    if target2 is None or target2.color == opp:
                        moves.append((pos, (two_r, c)))

        if c > 0:
            target = grid[r][c - 1]
            if target and target.color == opp:
                moves.append((pos, (r, c - 1)))
        if c < COLS - 1:
            target = grid[r][c + 1]
            if target and target.color == opp:
                moves.append((pos, (r, c + 1)))
        return moves


# -----------------------------------------------------------------------
# Board
# -----------------------------------------------------------------------
class Board:
    def __init__(self, setup=True):
        self.grid           = [[None] * COLS for _ in range(ROWS)]
        self.white_king_pos = None
        self.black_king_pos = None
        self.white_pieces   = []
        self.black_pieces   = []
        self.white_knights  = []
        self.black_knights  = []
        self.piece_counts   = {
            'white': {Pawn: 0, Knight: 0, Bishop: 0, Rook: 0, Queen: 0, King: 0},
            'black': {Pawn: 0, Knight: 0, Bishop: 0, Rook: 0, Queen: 0, King: 0},
        }
        self.piece_counts_z = {'white': [0] * 6, 'black': [0] * 6}
        if setup:
            self._setup_initial_board()

    def _setup_initial_board(self):
        pieces = {
            0: [(0, Rook), (1, Knight), (2, Bishop), (3, Queen), (4, King),
                (5, Bishop), (6, Knight), (7, Rook)],
            1: [(i, Pawn) for i in range(8)],
            6: [(i, Pawn) for i in range(8)],
            7: [(0, Rook), (1, Knight), (2, Bishop), (3, Queen), (4, King),
                (5, Bishop), (6, Knight), (7, Rook)],
        }
        for r, piece_list in pieces.items():
            color = "black" if r < 2 else "white"
            for c, piece_class in piece_list:
                self.add_piece(piece_class(color), r, c)

    # ---- O(1) piece-list helpers ------------------------------------------

    def _list_append(self, piece):
        """Append to the colour list and record the resulting index."""
        lst             = self.white_pieces if piece.color == 'white' else self.black_pieces
        piece._list_pos = len(lst)
        lst.append(piece)
        if piece.z_idx == 1:
            klst = self.white_knights if piece.color == 'white' else self.black_knights
            piece._knight_list_pos = len(klst)
            klst.append(piece)

    def _list_remove(self, piece):
        """
        O(1) swap-and-pop removal.  Swaps piece with the last element so the
        list stays compact, then pops the tail.  Safe to call on a piece that
        is already absent (_list_pos == -1) — treated as a no-op.
        """
        idx = piece._list_pos
        if idx < 0:
            return
        lst            = self.white_pieces if piece.color == 'white' else self.black_pieces
        last           = lst[-1]
        lst[idx]       = last
        last._list_pos = idx
        lst.pop()
        piece._list_pos = -1
        
        if piece.z_idx == 1:
            kidx = getattr(piece, '_knight_list_pos', -1)
            if kidx >= 0:
                klst = self.white_knights if piece.color == 'white' else self.black_knights
                klast = klst[-1]
                klst[kidx] = klast
                klast._knight_list_pos = kidx
                klst.pop()
                piece._knight_list_pos = -1

    # ---- Board mutation primitives ----------------------------------------

    def add_piece(self, piece, r, c):
        if self.grid[r][c] is not None:
            self.remove_piece(r, c)
        self.grid[r][c] = piece
        piece.pos       = (r, c)
        self._list_append(piece)
        self.piece_counts[piece.color][type(piece)] += 1
        self.piece_counts_z[piece.color][piece.z_idx] += 1
        if type(piece) is King:
            if piece.color == 'white': self.white_king_pos = (r, c)
            else:                      self.black_king_pos = (r, c)

    def remove_piece(self, r, c):
        piece = self.grid[r][c]
        if not piece:
            return
        self._list_remove(piece)
        self.piece_counts[piece.color][type(piece)] -= 1
        self.piece_counts_z[piece.color][piece.z_idx] -= 1
        if type(piece) is King:
            if piece.color == 'white': self.white_king_pos = None
            else:                      self.black_king_pos = None
        piece.pos       = None
        self.grid[r][c] = None

    def move_piece(self, start, end):
        piece = self.grid[start[0]][start[1]]
        if not piece:
            return
        piece.pos = end
        if type(piece) is King:
            if piece.color == 'white': self.white_king_pos = end
            else:                      self.black_king_pos = end
        self.grid[start[0]][start[1]] = None
        self.grid[end[0]][end[1]]     = piece

    def find_king_pos(self, color):
        return self.white_king_pos if color == 'white' else self.black_king_pos

    def clone(self):
        new_board               = Board.__new__(Board)
        new_board.grid          = [[None] * COLS for _ in range(ROWS)]
        new_board.white_king_pos = self.white_king_pos
        new_board.black_king_pos = self.black_king_pos

        white_pieces = [_clone_piece_fast(p) for p in self.white_pieces]
        black_pieces = [_clone_piece_fast(p) for p in self.black_pieces]
        new_board.white_pieces = white_pieces
        new_board.black_pieces = black_pieces
        
        new_board.white_knights = [p for p in white_pieces if p.z_idx == 1]
        for i, p in enumerate(new_board.white_knights):
            p._knight_list_pos = i
        new_board.black_knights = [p for p in black_pieces if p.z_idx == 1]
        for i, p in enumerate(new_board.black_knights):
            p._knight_list_pos = i

        grid = new_board.grid
        for p in white_pieces:
            r, c = p.pos; grid[r][c] = p
        for p in black_pieces:
            r, c = p.pos; grid[r][c] = p

        pc = self.piece_counts
        new_board.piece_counts = {
            'white': pc['white'].copy(),
            'black': pc['black'].copy(),
        }
        pcz = self.piece_counts_z
        new_board.piece_counts_z = {
            'white': pcz['white'].copy(),
            'black': pcz['black'].copy(),
        }
        return new_board

    # -----------------------------------------------------------------------
    # make_move  (UI path — delegates to make_move_track)
    # -----------------------------------------------------------------------
    def make_move(self, start, end):
        self.make_move_track(start, end)

    # -----------------------------------------------------------------------
    # make_move_track / unmake_move  (search path — no cloning)
    # -----------------------------------------------------------------------
    def make_move_track(self, start, end):
        moving_piece = self.grid[start[0]][start[1]]
        removed = []
        added = []
        mc      = moving_piece.color
        mp_z = moving_piece.z_idx

        target_piece = self.grid[end[0]][end[1]]
        is_capture = target_piece is not None

        # ── 1. Rook piercing ──
        if mp_z == 3:   # Rook
            dr = (end[0] > start[0]) - (start[0] > end[0])
            dc = (end[1] > start[1]) - (start[1] > end[1])
            cr, cc = start[0] + dr, start[1] + dc
            while (cr, cc) != end:
                t = self.grid[cr][cc]
                if t is not None and t.color != mc:
                    removed.append((t, cr, cc))
                    self.remove_piece(cr, cc)
                cr += dr
                cc += dc

        # ── 2. Standard capture ──
        if is_capture:
            removed.append((target_piece, end[0], end[1]))
            self.remove_piece(end[0], end[1])

        # ── 3. Move ──
        self.move_piece(start, end)

        # ── 4. Queen AOE explosion ──
        if mp_z == 4 and is_capture:   # Queen
            removed.append((moving_piece, end[0], end[1]))
            self.remove_piece(end[0], end[1])
            for r, c in ADJACENT_SQUARES_MAP[end]:
                adj = self.grid[r][c]
                if adj is not None and adj.color != mc:
                    removed.append((adj, r, c))
                    self.remove_piece(r, c)

        # ── 5. Pawn promotion ──
        elif mp_z == 0 and end[0] == moving_piece.promo_rank:   # Pawn
            removed.append((moving_piece, end[0], end[1]))
            self.remove_piece(end[0], end[1])
            new_queen = Queen(mc)
            self.add_piece(new_queen, end[0], end[1])
            added.append((new_queen, end[0], end[1]))

        # ── 6. Knight AOE ──
        grid = self.grid
        if mp_z == 1:   # Knight
            enemy_knight_coords = []
            for r, c in KNIGHT_ATTACKS_FROM[end]:
                target = grid[r][c]
                if target is not None and target.color != mc:
                    if target.z_idx == 1:
                        enemy_knight_coords.append((r, c))
            
            for ek_r, ek_c in enemy_knight_coords:
                for r, c in KNIGHT_ATTACKS_FROM[(ek_r, ek_c)]:
                    target = grid[r][c]
                    if target is not None and target.color == mc:
                        removed.append((target, r, c))
                        self.remove_piece(r, c)
            
            for r, c in KNIGHT_ATTACKS_FROM[end]:
                target = grid[r][c]
                if target is not None and target.color != mc:
                    removed.append((target, r, c))
                    self.remove_piece(r, c)
        else:
            # Passive evaporation check
            victim = grid[end[0]][end[1]]
            if victim is not None:
                for r, c in KNIGHT_ATTACKS_FROM[end]:
                    killer = grid[r][c]
                    if killer is not None and killer.z_idx == 1 and killer.color != mc:
                        removed.append((victim, end[0], end[1]))
                        self.remove_piece(end[0], end[1])
                        break

        return (start, end, moving_piece, removed, added)

    def unmake_move(self, record_tuple):
        """Restore the board to the exact state before make_move_track()."""
        start, end, moving_piece, removed, added = record_tuple
        added_ids = {id(p) for p, _r, _c in added} if added else set()

        # ── 1. Undo promoted pieces ──
        for piece, r, c in reversed(added):
            if piece.pos is not None:
                self.grid[r][c] = None
                piece.pos = None
                self._list_remove(piece)
                self.piece_counts[piece.color][type(piece)] -= 1
                self.piece_counts_z[piece.color][piece.z_idx] -= 1

        mp_pos = moving_piece.pos
        if mp_pos is not None:
            self.grid[mp_pos[0]][mp_pos[1]] = None
            self.grid[start[0]][start[1]] = moving_piece
            moving_piece.pos = start
            if moving_piece.z_idx == 5:
                if moving_piece.color == 'white': self.white_king_pos = start
                else:                             self.black_king_pos = start
        else:
            self.grid[start[0]][start[1]] = moving_piece
            moving_piece.pos = start
            self._list_append(moving_piece)
            self.piece_counts[moving_piece.color][type(moving_piece)] += 1
            self.piece_counts_z[moving_piece.color][moving_piece.z_idx] += 1
            if moving_piece.z_idx == 5:
                if moving_piece.color == 'white': self.white_king_pos = start
                else:                             self.black_king_pos = start

        for piece, r, c in removed:
            if piece is moving_piece:   continue
            if id(piece) in added_ids:  continue
            self.grid[r][c] = piece
            piece.pos = (r, c)
            self._list_append(piece)
            self.piece_counts[piece.color][type(piece)] += 1
            self.piece_counts_z[piece.color][piece.z_idx] += 1
            if piece.z_idx == 5:
                if piece.color == 'white': self.white_king_pos = (r, c)
                else:                      self.black_king_pos = (r, c)


# -----------------------------------------------------------------------
# Global game logic
# -----------------------------------------------------------------------
def _bishop_attacks_square(board, start, tr, tc, bishop_color):
    if ((start[0] + start[1] - tr - tc) & 1) != 0:
        return False

    grid = board.grid
    start_index = start[0] * COLS + start[1]

    for ray in RAYS[start_index][4:]:
        for r, c in ray:
            piece = grid[r][c]
            if r == tr and c == tc:
                return (piece is None) or (piece.color != bishop_color)
            if piece is not None:
                break

    for ray in BISHOP_ZIGZAG_RAYS[start_index]:
        for r, c in ray:
            piece = grid[r][c]
            if r == tr and c == tc:
                return (piece is None) or (piece.color != bishop_color)
            if piece is not None:
                break
    return False


def is_square_attacked(board, r, c, attacking_color):
    grid            = board.grid
    defending_color = 'black' if attacking_color == 'white' else 'white'
    attacking_pieces = board.white_pieces if attacking_color == 'white' else board.black_pieces
    attacker_counts = board.piece_counts_z[attacking_color]
    attacking_king_pos = board.white_king_pos if attacking_color == 'white' else board.black_king_pos

    # 1. PAWN ATTACKS (Correct sideways and forward checks)
    pawn_move_dir = -1 if attacking_color == 'white' else 1
    pr = r - pawn_move_dir
    if 0 <= pr < 8:
        p = grid[pr][c]
        if p is not None and p.z_idx == 0 and p.color == attacking_color:
            return True
        if p is None:
            two_pr = r - (2 * pawn_move_dir)
            starting_row = 6 if attacking_color == 'white' else 1
            if two_pr == starting_row:
                p2 = grid[two_pr][c]
                if p2 is not None and p2.z_idx == 0 and p2.color == attacking_color:
                    return True
                    
    if c > 0:
        p = grid[r][c - 1]
        if p is not None and p.z_idx == 0 and p.color == attacking_color:
            return True
    if c < 7:
        p = grid[r][c + 1]
        if p is not None and p.z_idx == 0 and p.color == attacking_color:
            return True

    # 2. KNIGHT ATTACKS (Strictly Non-Functional, O(1) Precomputed Evaporation)
    if attacker_counts[1] > 0:
        target_idx = r * 8 + c
        attacking_knights = board.white_knights if attacking_color == 'white' else board.black_knights
        for piece in attacking_knights:
            if piece.pos:
                p_idx = piece.pos[0] * 8 + piece.pos[1]
                evap = KNIGHT_EVAP_SQUARES[p_idx][target_idx]
                if evap is True:
                    return True
                elif evap is not None:
                    for zr, zc in evap:
                        if grid[zr][zc] is None:
                            return True

    # 3. KING ATTACKS (Inlined, no abs() calls)
    if attacking_king_pos:
        kr, kc = attacking_king_pos
        dr = r - kr
        dc = c - kc
        abs_dr = dr if dr >= 0 else -dr
        abs_dc = dc if dc >= 0 else -dc
        if abs_dr <= 2 and abs_dc <= 2:
            m_dist = abs_dr if abs_dr > abs_dc else abs_dc
            if m_dist == 1:
                return True
            if m_dist == 2 and (abs_dr == abs_dc or abs_dr == 0 or abs_dc == 0):
                if grid[kr + dr // 2][kc + dc // 2] is None:
                    return True
                    
    if len(attacking_pieces) == attacker_counts[5]:
        return False

    # 4. ROOKS & BISHOPS (Original Raycast - perfectly stable, but optimized layout)
    has_rooks = attacker_counts[3] > 0
    has_bishops = attacker_counts[2] > 0
    if has_rooks or has_bishops:
        start_index = r * 8 + c
        if has_rooks:
            for i in range(4):
                for cr, cc in RAYS[start_index][i]:
                    piece = grid[cr][cc]
                    if piece is not None:
                        if piece.color == attacking_color:
                            if piece.z_idx == 3: return True
                            break
        if has_bishops:
            for i in range(4, 8):
                for cr, cc in RAYS[start_index][i]:
                    piece = grid[cr][cc]
                    if piece is not None:
                        if piece.color == attacking_color and piece.z_idx == 2:
                            return True
                        break

    # 5. BISHOP ZIG-ZAG
    if has_bishops:
        target_parity = (r + c) & 1
        for piece in attacking_pieces:
            if piece.z_idx == 2 and piece.pos:
                if ((piece.pos[0] + piece.pos[1]) & 1) == target_parity:
                    if _bishop_attacks_square(board, piece.pos, r, c, attacking_color):
                        return True

    # 6. QUEEN EXPLOSIONS (Original Raycast - perfectly stable, but optimized math)
    if attacker_counts[4] > 0:
        for piece in attacking_pieces:
            if piece.z_idx == 4 and piece.pos:
                q_idx = piece.pos[0] * 8 + piece.pos[1]
                for i in range(8):
                    for cr, cc in RAYS[q_idx][i]:
                        target = grid[cr][cc]
                        if target is not None:
                            if target.color == defending_color:
                                dr = cr - r
                                dc = cc - c
                                if (dr >= -1 and dr <= 1) and (dc >= -1 and dc <= 1):
                                    return True
                            break
                            
    return False


def is_in_check(board, color):
    king_pos = board.white_king_pos if color == 'white' else board.black_king_pos
    if not king_pos:
        return True
    opponent_color = "black" if color == "white" else "white"
    return is_square_attacked(board, king_pos[0], king_pos[1], opponent_color)


def generate_legal_moves_generator(board, color, yield_boards=False):
    opp_color = "black" if color == "white" else "white"
    piece_list = list(board.white_pieces if color == 'white' else board.black_pieces)
    for piece in piece_list:
        if piece.pos is None: continue
        for move in piece.get_valid_moves(board, piece.pos):
            record   = board.make_move_track(move[0], move[1])

            my_kp = board.white_king_pos if color == 'white' else board.black_king_pos
            legal = (my_kp is not None and not is_square_attacked(board, my_kp[0], my_kp[1], opp_color))
            
            if legal and yield_boards:
                result_board = board.clone()
            board.unmake_move(record)
            if legal:
                if yield_boards: yield move, result_board
                else: yield move


def get_all_legal_moves(board, color):
    return list(generate_legal_moves_generator(board, color))


def get_all_pseudo_legal_moves(board, color):
    moves = []
    piece_list = board.white_pieces if color == 'white' else board.black_pieces
    for piece in piece_list:
        if piece.pos is not None:
            moves.extend(piece.get_valid_moves(board, piece.pos))
    return moves


def has_legal_moves(board, color):
    try:
        next(generate_legal_moves_generator(board, color))
        return True
    except StopIteration:
        return False


def is_insufficient_material(board):
    return (len(board.white_pieces) + len(board.black_pieces)) <= 2


_board_hash_fn = None
def _get_board_hash():
    global _board_hash_fn
    if _board_hash_fn is None:
        from AI import board_hash
        _board_hash_fn = board_hash
    return _board_hash_fn

def get_game_state(board, turn_to_move, position_counts, ply_count, max_moves):
    if not has_legal_moves(board, turn_to_move):
        winner = 'black' if turn_to_move == 'white' else 'white'
        return ("checkmate", winner)

    if is_insufficient_material(board):
        return ("insufficient_material", None)

    try:
        bh_fn = _get_board_hash()
        if position_counts.get(bh_fn(board, turn_to_move), 0) >= 3:
            return ("repetition", None)
    except ImportError:
        pass

    if ply_count >= max_moves:
        return ("move_limit", None)

    return "ongoing", None


def is_draw(board, turn_to_move, position_counts, ply_count, max_moves):
    state, _ = get_game_state(board, turn_to_move, position_counts, ply_count, max_moves)
    return state in ("insufficient_material", "repetition", "move_limit")


def fast_approximate_material_swing(board, move, moving_piece, target_piece, piece_values_list):
    swing   = 0
    is_tactic = False
    my_z = moving_piece.z_idx
    my_color = moving_piece.color

    if target_piece is not None:
        swing += piece_values_list[target_piece.z_idx]
        is_tactic = True

    if my_z == 0 and move[1][0] == moving_piece.promo_rank:
        swing += piece_values_list[4] - piece_values_list[0]
        is_tactic = True

    if my_z == 4 and target_piece is not None:
        swing -= piece_values_list[4]
        for r, c in ADJACENT_SQUARES_MAP[move[1]]:
            adj = board.grid[r][c]
            if adj and adj.color != my_color:
                swing += piece_values_list[adj.z_idx]
                is_tactic = True
        return swing, is_tactic

    pierced_knights_mask = 0
    if my_z == 3:
        start, end = move
        dr = (end[0] > start[0]) - (start[0] > end[0])
        dc = (end[1] > start[1]) - (start[1] > end[1])
        cr, cc = start[0] + dr, start[1] + dc
        while (cr, cc) != end:
            target = board.grid[cr][cc]
            if target and target.color != my_color:
                swing += piece_values_list[target.z_idx]
                is_tactic = True
                if target.z_idx == 1:
                    pierced_knights_mask |= (1 << (cr * 8 + cc))
            cr += dr
            cc += dc

    if my_z == 1:
        seen_passive_mask = 0
        for r, c in KNIGHT_ATTACKS_FROM[move[1]]:
            target = board.grid[r][c]
            if target and target.color != my_color:
                swing += piece_values_list[target.z_idx]
                is_tactic = True
                if target.z_idx == 1:
                    for pr, pc in KNIGHT_ATTACKS_FROM[(r, c)]:
                        if (pr, pc) == move[1]:
                            if not (seen_passive_mask & 2): # 1 << 1 (Knight z_idx is 1)
                                swing -= piece_values_list[1]
                                seen_passive_mask |= 2
                        else:
                            ptarget = board.grid[pr][pc]
                            if ptarget and ptarget.color == my_color:
                                pz = ptarget.z_idx
                                bit = 1 << pz
                                if not (seen_passive_mask & bit):
                                    swing -= piece_values_list[pz]
                                    seen_passive_mask |= bit
        return swing, is_tactic

    for r, c in KNIGHT_ATTACKS_FROM[move[1]]:
        pk = board.grid[r][c]
        if (pk and pk.z_idx == 1
                and pk.color != my_color
                and not (pierced_knights_mask & (1 << (r * 8 + c)))):
            evap_z = 4 if (my_z == 0 and move[1][0] == moving_piece.promo_rank) else my_z
            swing -= piece_values_list[evap_z]
            is_tactic = True
            break

    return swing, is_tactic


def format_move(move):
    if not move:
        return "None"
    (r1, c1), (r2, c2) = move
    return f"{'abcdefgh'[c1]}{'87654321'[r1]}-{'abcdefgh'[c2]}{'87654321'[r2]}"


def format_move_san(board_before, board_after, move):
    if not move:
        return "None"
    start_pos, end_pos = move
    moving_piece = board_before.grid[start_pos[0]][start_pos[1]]
    if not moving_piece:
        return format_move(move)

    ptype = type(moving_piece)

    is_capture = False
    if ptype is not Knight:
        is_capture = board_before.grid[end_pos[0]][end_pos[1]] is not None

    def file_of(c):   return "abcdefgh"[c]
    def rank_of(r):   return "87654321"[r]
    def sq_name(pos): return file_of(pos[1]) + rank_of(pos[0])

    # Disambiguation — O(n_pieces) piece-list scan instead of O(64) grid scan
    disambig = ""
    if ptype not in (Pawn, King):
        others     = []
        piece_list = (board_before.white_pieces if moving_piece.color == 'white'
                      else board_before.black_pieces)
        for p in piece_list:
            if type(p) is ptype and p.pos != start_pos:
                if any(m[1] == end_pos for m in p.get_valid_moves(board_before, p.pos)):
                    others.append(p.pos)
        if others:
            same_file = any(pos[1] == start_pos[1] for pos in others)
            same_rank = any(pos[0] == start_pos[0] for pos in others)
            if not same_file:
                disambig = file_of(start_pos[1])
            elif not same_rank:
                disambig = rank_of(start_pos[0])
            else:
                disambig = sq_name(start_pos)

    if ptype is Pawn:
        base_str = (file_of(start_pos[1]) + "x" + sq_name(end_pos)) if is_capture \
                   else sq_name(end_pos)
    else:
        p_char   = {King: 'K', Queen: 'Q', Rook: 'R', Bishop: 'B', Knight: 'N'}[ptype]
        cap_str  = "x" if is_capture else ""
        base_str = p_char + disambig + cap_str + sq_name(end_pos)

    if ptype is Pawn and end_pos[0] == moving_piece.promo_rank:
        base_str += "=Q"

    # Dead squares — O(n_pieces) piece-list scan (retained from v52)
    after_grid   = board_after.grid
    end_r, end_c = end_pos
    dead_squares = []

    if after_grid[end_r][end_c] is None:
        dead_squares.append(end_pos)

    for piece in board_before.white_pieces + board_before.black_pieces:
        pr, pc = piece.pos
        if (pr, pc) == start_pos or (pr, pc) == end_pos:
            continue
        if after_grid[pr][pc] is None:
            dead_squares.append((pr, pc))

    if dead_squares:
        dead_squares.sort(key=lambda pos: (8 - pos[0], pos[1]))
        cas_str   = " ".join(f"x{sq_name(pos)}" for pos in dead_squares)
        base_str += f" ({cas_str})"

    opp_color = "black" if moving_piece.color == "white" else "white"
    is_mate   = not has_legal_moves(board_after, opp_color)

    if is_mate:
        base_str += "#"
    elif is_in_check(board_after, opp_color):
        base_str += "+"

    return base_str