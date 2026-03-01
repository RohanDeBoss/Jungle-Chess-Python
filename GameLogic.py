# GameLogic.py (v40.2 - Raycasting fully implemented)

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

# BISHOP_ZIGZAG_DIRS pulled out to a constant to avoid reallocation inside the loop
BISHOP_ZIGZAG_DIRS = (
    ((-1, 1), (-1, -1)), ((-1, -1), (-1, 1)), # Forward zig-zags
    ((1, 1), (1, -1)), ((1, -1), (1, 1)),     # Backward zig-zags
    ((-1, 1), (1, 1)), ((1, 1), (-1, 1)),     # Rightward zig-zags
    ((-1, -1), (1, -1)), ((1, -1), (-1, -1))  # Leftward zig-zags
)

# --- Pre-computation Maps for Performance ---
KNIGHT_ATTACKS_FROM = { (r, c):[(r+dr, c+dc) for dr, dc in DIRECTIONS['knight'] if 0 <= r+dr < ROWS and 0 <= c+dc < COLS] for r in range(ROWS) for c in range(COLS) }
ADJACENT_SQUARES_MAP = { (r, c):[(r+dr, c+dc) for dr, dc in ADJACENT_DIRS if 0 <= r+dr < ROWS and 0 <= c+dc < COLS] for r in range(ROWS) for c in range(COLS) }

# ------------------------------------------------------------
# STEP 1: Precomputed Rays for ultra-fast threat detection
# RAYS[square_index][direction_index] = list of (r, c) coordinates
# ------------------------------------------------------------
RAYS = [[[] for _ in range(8)] for _ in range(64)]

def _init_rays():
    # Order: N, S, E, W (Orthogonal 0-3), NE, NW, SE, SW (Diagonal 4-7)
    dy_dx = [(-1, 0), (1, 0), (0, 1), (0, -1), (-1, 1), (-1, -1), (1, 1), (1, -1)]
    for r in range(ROWS):
        for c in range(COLS):
            for i, (dr, dc) in enumerate(dy_dx):
                cr, cc = r + dr, c + dc
                while 0 <= cr < ROWS and 0 <= cc < COLS:
                    RAYS[r * COLS + c][i].append((cr, cc))
                    cr += dr
                    cc += dc

_init_rays()
# ------------------------------------------------------------

def _clone_piece_fast(piece):
    """
    Fast clone for search boards: bypass __init__ and copy stable fields directly.
    """
    cls = piece.__class__
    new_piece = cls.__new__(cls)
    new_piece.color = piece.color
    new_piece.opponent_color = piece.opponent_color
    new_piece.pos = piece.pos
    if cls is Pawn:
        new_piece.direction = piece.direction
        new_piece.starting_row = piece.starting_row
    return new_piece

# ---------------------------------------------------
# PIECE CLASSES: THE SINGLE SOURCE OF TRUTH
# ---------------------------------------------------
class Piece:
    def __init__(self, color):
        self.color = color
        self.opponent_color = "black" if color == "white" else "white"
        self.pos = None

    def clone(self):
        new_piece = self.__class__(self.color)
        new_piece.pos = self.pos
        return new_piece

    def symbol(self): return "?"

    def get_valid_moves(self, board, pos):
        return[]

    def get_threats(self, board, pos):
        # By default, a piece threatens exactly the squares it can validly move to
        return set(self.get_valid_moves(board, pos))

class King(Piece):
    def symbol(self): return "♔" if self.color == "white" else "♚"
    def get_valid_moves(self, board, pos):
        moves =[]
        r_start, c_start = pos
        for dr, dc in DIRECTIONS['king']:
            r1, c1 = r_start + dr, c_start + dc
            if 0 <= r1 < ROWS and 0 <= c1 < COLS and (board.grid[r1][c1] is None or board.grid[r1][c1].color == self.opponent_color):
                moves.append((r1, c1))
                # Only allow moving a 2nd square if the 1st square was empty (Kings can't jump captured pieces)
                if board.grid[r1][c1] is None:
                    r2, c2 = r1 + dr, c1 + dc
                    if 0 <= r2 < ROWS and 0 <= c2 < COLS and (board.grid[r2][c2] is None or board.grid[r2][c2].color == self.opponent_color):
                        moves.append((r2, c2))
        return moves

class Queen(Piece):
    def symbol(self): return "♕" if self.color == "white" else "♛"

    def get_valid_moves(self, board, pos):
        moves = []
        grid = board.grid
        start_index = pos[0] * COLS + pos[1]

        # Check all 8 directions (0-7) using precomputed rays
        for i in range(8):
            for (r, c) in RAYS[start_index][i]:
                target = grid[r][c]
                if target is None:
                    moves.append((r, c))
                else:
                    # Capture enemy, then stop (Queens don't pierce)
                    if target.color != self.color:
                        moves.append((r, c))
                    break
        return moves

    def get_threats(self, board, pos):
        threats = set()
        # We can reuse get_valid_moves logic efficiently here
        valid_moves = self.get_valid_moves(board, pos)
        for move in valid_moves:
            threats.add(move)
            target_piece = board.grid[move[0]][move[1]]
            # Explosive Proxy logic
            if target_piece is not None and target_piece.color == self.opponent_color:
                threats.update(ADJACENT_SQUARES_MAP.get(move, []))
        return threats

class Rook(Piece):
    def symbol(self): return "♖" if self.color == "white" else "♜"

    def get_valid_moves(self, board, pos):
        moves = []
        grid = board.grid
        start_index = pos[0] * COLS + pos[1]

        # Check only Orthogonal directions (Indices 0, 1, 2, 3)
        for i in range(4):
            for (r, c) in RAYS[start_index][i]:
                target = grid[r][c]
                # Railgun: Stops ONLY at friendly pieces.
                # It continues through enemies and empty squares.
                if target and target.color == self.color:
                    break
                moves.append((r, c))
        return moves
        
    def get_threats(self, board, pos):
        return set(self.get_valid_moves(board, pos))

class Bishop(Piece):
    def symbol(self): return "♗" if self.color == "white" else "♝"

    def get_valid_moves(self, board, pos):
        moves = {} # Dict to ensure uniqueness between normal and zigzag paths
        grid = board.grid
        r_start, c_start = pos
        start_index = r_start * COLS + c_start

        # 1. Normal Diagonal (Indices 4, 5, 6, 7)
        for i in range(4, 8):
            for (r, c) in RAYS[start_index][i]:
                target = grid[r][c]
                if target:
                    if target.color != self.color:
                        moves[(r, c)] = None
                    break
                moves[(r, c)] = None
                
        # 2. Zig-Zag Movement (Keep existing logic)
        for d1, d2 in BISHOP_ZIGZAG_DIRS:
            cr, cc, cd = r_start, c_start, d1
            while True:
                cr += cd[0]; cc += cd[1]
                if not (0 <= cr < ROWS and 0 <= cc < COLS): break
                target = grid[cr][cc]
                if target:
                    if target.color != self.color: moves[(cr, cc)] = None
                    break
                moves[(cr, cc)] = None; cd = d2 if cd == d1 else d1
                
        return list(moves)
    
class Knight(Piece):
    def symbol(self): return "♘" if self.color == "white" else "♞"
    def get_valid_moves(self, board, pos):
        # Walrus operator assignment optimization
        return[(r, c) for r, c in KNIGHT_ATTACKS_FROM[pos] 
                if (p := board.grid[r][c]) is None or p.color != self.color]

    def get_threats(self, board, pos):
        threats = set()
        valid_moves = self.get_valid_moves(board, pos)
        for move in valid_moves:
            threats.add(move)
            # Active Evaporation Check: Threatens the squares it would land on AND their blast radii
            threats.update(KNIGHT_ATTACKS_FROM.get(move,[]))
        return threats

class Pawn(Piece):
    def __init__(self, color):
        super().__init__(color)
        self.direction = -1 if self.color == "white" else 1
        self.starting_row = 6 if self.color == "white" else 1

    def symbol(self): return "♙" if self.color == "white" else "♟"

    def get_valid_moves(self, board, pos):
        moves =[]
        r, c = pos
        direction = self.direction
        grid = board.grid
        
        # 1. Forward Moves & Captures
        one_r = r + direction
        if 0 <= one_r < ROWS:
            target1 = grid[one_r][c]
            # Check 1st square (Empty OR Enemy Capture)
            if target1 is None or target1.color == self.opponent_color:
                moves.append((one_r, c))
                
                # 2. Forward 2 (Only if 1st was empty so pawn isn't jumping pieces)
                if r == self.starting_row and target1 is None:
                    two_r = r + (2 * direction)
                    target2 = grid[two_r][c]
                    if target2 is None or target2.color == self.opponent_color:
                        moves.append((two_r, c))

        # 3. Sideways Captures (Unrolled for speed)
        if c > 0:
            target = grid[r][c-1]
            if target and target.color == self.opponent_color:
                moves.append((r, c-1))
        if c < COLS - 1:
            target = grid[r][c+1]
            if target and target.color == self.opponent_color:
                moves.append((r, c+1))
                
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
        self.black_pieces =[]
        if setup: self._setup_initial_board()

    def _setup_initial_board(self):
        pieces = {0:[(0, Rook), (1, Knight), (2, Bishop), (3, Queen), (4, King), (5, Bishop), (6, Knight), (7, Rook)], 1:[(i, Pawn) for i in range(8)], 6: [(i, Pawn) for i in range(8)], 7:[(0, Rook), (1, Knight), (2, Bishop), (3, Queen), (4, King), (5, Bishop), (6, Knight), (7, Rook)]}
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
        
        if piece.color == 'white':
            if piece in self.white_pieces: self.white_pieces.remove(piece)
        else: 
            if piece in self.black_pieces: self.black_pieces.remove(piece)
            
        if isinstance(piece, King):
            if piece.color == "white": self.white_king_pos = None
            else: self.black_king_pos = None
                
        piece.pos = None 
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

    def find_king_pos(self, color): return self.white_king_pos if color == 'white' else self.black_king_pos

    def clone(self):
        new_board = Board(setup=False)
        
        # Clone pieces with a fast constructor-bypass path.
        new_board.white_pieces = [_clone_piece_fast(p) for p in self.white_pieces]
        new_board.black_pieces = [_clone_piece_fast(p) for p in self.black_pieces]
        
        # Populate grid directly (Removed slow isinstance(King) checks from loop)
        for p in new_board.white_pieces:
            new_board.grid[p.pos[0]][p.pos[1]] = p
        for p in new_board.black_pieces:
            new_board.grid[p.pos[0]][p.pos[1]] = p
            
        # Directly copy King positions from the parent board (Massive Speedup)
        new_board.white_king_pos = self.white_king_pos
        new_board.black_king_pos = self.black_king_pos
            
        return new_board

    def make_move(self, start, end):
        moving_piece = self.grid[start[0]][start[1]]
        if not moving_piece: return
        
        target_piece = self.grid[end[0]][end[1]]
        is_capture = target_piece is not None
        
        if isinstance(moving_piece, Rook): self._apply_rook_piercing(start, end, moving_piece.color)
        if is_capture: self.remove_piece(end[0], end[1])
        self.move_piece(start, end)
        
        if isinstance(moving_piece, Queen) and is_capture:
            self._apply_queen_aoe(end, moving_piece.color)
        elif isinstance(moving_piece, Pawn):
            promotion_rank = 0 if moving_piece.color == "white" else (ROWS - 1)
            if end[0] == promotion_rank:
                self.remove_piece(end[0], end[1])
                self.add_piece(Queen(moving_piece.color), end[0], end[1])

        self._apply_knight_aoe(end, is_active_move=isinstance(moving_piece, Knight))

    def _apply_queen_aoe(self, pos, queen_color):
        if self.grid[pos[0]][pos[1]]: self.remove_piece(pos[0], pos[1])
        for r, c in ADJACENT_SQUARES_MAP.get(pos, set()):
            adj_piece = self.grid[r][c]
            if adj_piece and adj_piece.color != queen_color: self.remove_piece(r, c)

    def _apply_rook_piercing(self, start, end, rook_color):
        # Optimized vector calculation
        dr = (end[0] > start[0]) - (start[0] > end[0])
        dc = (end[1] > start[1]) - (start[1] > end[1])
        
        cr, cc = start[0] + dr, start[1] + dc
        while (cr, cc) != end:
            target = self.grid[cr][cc]
            if target and target.color != rook_color: self.remove_piece(cr, cc)
            cr += dr; cc += dc

    def _apply_knight_aoe(self, pos, is_active_move):
        grid = self.grid
        
        # 1. Active Evaporation
        if is_active_move:
            knight_instance = grid[pos[0]][pos[1]]
            if not knight_instance: return
            
            to_remove, enemy_knights_destroyed =[], False
            for r, c in KNIGHT_ATTACKS_FROM[pos]:
                target = grid[r][c]
                if target and target.color != knight_instance.color:
                    to_remove.append(target)
                    if isinstance(target, Knight):
                        enemy_knights_destroyed = True
            
            for piece in to_remove:
                self.remove_piece(piece.pos[0], piece.pos[1])

            if enemy_knights_destroyed:
                self.remove_piece(pos[0], pos[1])
                
        # 2. Passive Evaporation
        else:
            victim = grid[pos[0]][pos[1]]
            if not victim: return
            
            for r, c in KNIGHT_ATTACKS_FROM[pos]:
                potential_killer = grid[r][c]
                if (potential_killer and isinstance(potential_killer, Knight) and potential_killer.color != victim.color):
                    self.remove_piece(pos[0], pos[1])
                    return

    def _get_rook_piercing_captures(self, start, end, rook_color):
        captured = []
        # Optimized vector calculation
        dr = (end[0] > start[0]) - (start[0] > end[0])
        dc = (end[1] > start[1]) - (start[1] > end[1])
        
        cr, cc = start[0] + dr, start[1] + dc
        while (cr, cc) != end:
            target = self.grid[cr][cc]
            if target and target.color != rook_color:
                captured.append(target)
            cr += dr; cc += dc
        return captured

    def _get_queen_aoe_captures(self, pos, queen_color):
        captured =[]
        for r, c in ADJACENT_SQUARES_MAP.get(pos, set()):
            adj_piece = self.grid[r][c]
            if adj_piece and adj_piece.color != queen_color:
                captured.append(adj_piece)
        return captured
    
    def _get_knight_aoe_outcome(self, knight_pos, knight_color):
        captured, self_evaporates =[], False
        for r, c in KNIGHT_ATTACKS_FROM.get(knight_pos, set()):
            target = self.grid[r][c]
            if target and target.color != knight_color:
                captured.append(target)
                if isinstance(target, Knight):
                    self_evaporates = True
        return captured, self_evaporates

    def get_move_outcome(self, move):
        start_pos, end_pos = move
        moving_piece = self.grid[start_pos[0]][start_pos[1]]
        if not moving_piece:
            return set(), set(), None
            
        friendly_lost, opponent_captured, promotion_type = set(), set(), None
        target_piece = self.grid[end_pos[0]][end_pos[1]]
        is_capture = target_piece is not None

        if is_capture:
            opponent_captured.add(target_piece)

        if isinstance(moving_piece, Rook):
            opponent_captured.update(self._get_rook_piercing_captures(start_pos, end_pos, moving_piece.color))
        elif isinstance(moving_piece, Queen) and is_capture:
            friendly_lost.add(moving_piece)
            opponent_captured.update(self._get_queen_aoe_captures(end_pos, moving_piece.color))
        elif isinstance(moving_piece, Knight):
            captures, self_evaporates = self._get_knight_aoe_outcome(end_pos, moving_piece.color)
            opponent_captured.update(captures)
            if self_evaporates: friendly_lost.add(moving_piece)
        elif isinstance(moving_piece, Pawn):
            promotion_rank = 0 if moving_piece.color == "white" else (ROWS - 1)
            if end_pos[0] == promotion_rank:
                promotion_type = Queen
                friendly_lost.add(moving_piece)

        # Passive knight evaporation: non-knights can evaporate by landing inside
        # an enemy knight's zone, unless that knight was already captured by this move.
        if not isinstance(moving_piece, Knight):
            if isinstance(moving_piece, Queen) and is_capture:
                passive_victim_type = None  # Explosive queen capture already self-removes.
            elif promotion_type is not None:
                passive_victim_type = Queen  # Pawn promoted, so victim would be the promoted queen.
            else:
                passive_victim_type = type(moving_piece)

            if passive_victim_type is not None:
                for r, c in KNIGHT_ATTACKS_FROM.get(end_pos, set()):
                    potential_killer = self.grid[r][c]
                    if (potential_killer and isinstance(potential_killer, Knight) and
                        potential_killer.color != moving_piece.color and
                        potential_killer not in opponent_captured):
                        friendly_lost.add(passive_victim_type(moving_piece.color))
                        break
        
        return friendly_lost, opponent_captured, promotion_type

# ----------------------------------------------------
# GLOBAL GAME LOGIC: ROBUST & CENTRALIZED
# ----------------------------------------------------
def _bishop_attacks_square(board, start, target, bishop_color):
    tr, tc = target

    # 1) Normal diagonal movement
    for dr, dc in DIRECTIONS['bishop']:
        r, c = start[0] + dr, start[1] + dc
        while 0 <= r < ROWS and 0 <= c < COLS:
            piece = board.grid[r][c]
            if r == tr and c == tc:
                return (piece is None) or (piece.color != bishop_color)
            if piece is not None:
                break
            r += dr
            c += dc

    # 2) Zig-zag movement
    for d1, d2 in BISHOP_ZIGZAG_DIRS:
        cr, cc, cd = start[0], start[1], d1
        while True:
            cr += cd[0]
            cc += cd[1]
            if not (0 <= cr < ROWS and 0 <= cc < COLS):
                break

            piece = board.grid[cr][cc]
            if cr == tr and cc == tc:
                return (piece is None) or (piece.color != bishop_color)
            if piece is not None:
                break

            cd = d2 if cd == d1 else d1

    return False

def _knight_attacks_square(board, start, target, knight_color):
    tr, tc = target
    grid = board.grid

    for dst in KNIGHT_ATTACKS_FROM[start]:
        piece_at_dst = grid[dst[0]][dst[1]]
        # Knight valid-move filter (cannot land on friendly piece).
        if piece_at_dst is not None and piece_at_dst.color == knight_color:
            continue

        # Direct knight threat.
        if dst[0] == tr and dst[1] == tc:
            return True

        # Active evaporation proxy threat from the landing square.
        for rr, cc in KNIGHT_ATTACKS_FROM[dst]:
            if rr == tr and cc == tc:
                return True

    return False

def _king_attacks_square(board, start, target, king_color):
    tr, tc = target
    grid = board.grid

    for dr, dc in DIRECTIONS['king']:
        r1, c1 = start[0] + dr, start[1] + dc
        if not (0 <= r1 < ROWS and 0 <= c1 < COLS):
            continue

        p1 = grid[r1][c1]
        # 1-step king move threat.
        if r1 == tr and c1 == tc:
            if p1 is None or p1.color != king_color:
                return True

        # 2-step king move threat only if first square is empty.
        if p1 is not None:
            continue

        r2, c2 = r1 + dr, c1 + dc
        if not (0 <= r2 < ROWS and 0 <= c2 < COLS):
            continue
        p2 = grid[r2][c2]
        if r2 == tr and c2 == tc and (p2 is None or p2.color != king_color):
            return True

    return False

def _pawn_attacks_square(board, start, target, pawn):
    tr, tc = target
    r, c = start
    grid = board.grid
    direction = pawn.direction

    # 1-step forward (empty or enemy)
    one_r = r + direction
    if 0 <= one_r < ROWS:
        p1 = grid[one_r][c]
        if one_r == tr and c == tc and (p1 is None or p1.color != pawn.color):
            return True

        # 2-step forward from starting rank (first square must be empty)
        if r == pawn.starting_row and p1 is None:
            two_r = r + 2 * direction
            if 0 <= two_r < ROWS:
                p2 = grid[two_r][c]
                if two_r == tr and c == tc and (p2 is None or p2.color != pawn.color):
                    return True

    # Sideways captures (only if enemy piece exists there)
    if r == tr and abs(c - tc) == 1:
        pt = grid[tr][tc]
        if pt is not None and pt.color == pawn.opponent_color:
            return True

    return False

def is_square_attacked(board, r, c, attacking_color):
    """
    Optimized 'Reverse-Raycast' check using precomputed `RAYS`.
    Looks OUTWARDS from the target square to find threats.
    """
    grid = board.grid

    # 1. Check Knights (Standard Lookup - O(1))
    for nr, nc in KNIGHT_ATTACKS_FROM.get((r, c), []):
        piece = grid[nr][nc]
        if piece and piece.color == attacking_color and type(piece) is Knight:
            return True

    # 2. Check Sliding Pieces (Rook/Queen/Bishop) using Precomputed Rays
    # We look out from (r,c). If we hit a piece, we check if it attacks us.
    start_index = r * COLS + c

    for direction_idx, ray_path in enumerate(RAYS[start_index]):
        is_orthogonal = direction_idx < 4  # 0-3 are N,S,E,W

        # Track if we are looking through a "Defender" (Friendly piece)
        passed_defender = False
        defender_is_adjacent = False

        for step_idx, (cr, cc) in enumerate(ray_path):
            piece = grid[cr][cc]
            if piece is None:
                continue

            p_type = type(piece)

            if piece.color == attacking_color:
                # --- FOUND AN ATTACKER ---

                if is_orthogonal:
                    if p_type is Rook:
                        # Rooks pierce through defenders to hit the target
                        return True
                    if p_type is Queen:
                        # Direct hit OR Proxy Explosion
                        if not passed_defender:
                            return True  # Direct line of sight
                        elif defender_is_adjacent:
                            return True  # Proxy: Queen captures the adjacent defender -> Explosion hits King
                        # If passed_defender is True but NOT adjacent, Queen captures that piece
                        # but the explosion is too far away to hurt us.
                        return False
                    # Bishops don't attack orthogonally
                    break

                else:
                    # Diagonal (Bishop/Queen)
                    if p_type is Bishop or p_type is Queen:
                        # Note: Diagonal pieces usually don't pierce.
                        # Rule Check: Do Queens explode on diagonals? Yes.
                        if not passed_defender:
                            return True  # Direct line of sight
                        elif defender_is_adjacent and p_type is Queen:
                            return True  # Proxy: Queen captures adjacent defender -> Explosion
                        return False
                    break
            else:
                # --- FOUND A DEFENDER (Friendly Piece) ---
                if passed_defender:
                    # We already hit one defender, now we hit a second one.
                    # Line is fully blocked (even for piercing Rooks).
                    break
                else:
                    passed_defender = True
                    # If this is the very first square in the ray, the defender is adjacent.
                    if step_idx == 0:
                        defender_is_adjacent = True
                    # Continue loop to check for Piercing Rook or Proxy Queen behind this defender

    # 3. Check Pawns
    # Pawns capture Forward-Diagonal or Sideways?
    # Variant Rules: "Captures forward or sideways (but not diagonally)"
    # We need to see if a pawn is in a position to capture (r,c).

    # Forward Check: (If attacker is White, they are at r+1 looking "up" at us)
    pawn_attack_dir = 1 if attacking_color == 'white' else -1
    pr = r + pawn_attack_dir
    if 0 <= pr < ROWS:
        p = grid[pr][c]
        if p and p.color == attacking_color and type(p) is Pawn:
            return True

    # Sideways Check:
    for dc in [-1, 1]:
        pc = c + dc
        if 0 <= pc < COLS:
            p = grid[r][pc]
            if p and p.color == attacking_color and type(p) is Pawn:
                return True

    # 4. Check King (Range 2)
    # Optimization: Just check distance to enemy king.
    enemy_king_pos = board.find_king_pos(attacking_color)
    if enemy_king_pos:
        dist = max(abs(r - enemy_king_pos[0]), abs(c - enemy_king_pos[1]))
        if dist <= 2:
            return True

    # 5. Zig-Zag Bishop Fallback
    # Zig-Zags are non-linear, so rays don't work easily. Use standard iteration for this rare case.
    attacking_pieces = board.white_pieces if attacking_color == 'white' else board.black_pieces
    for piece in attacking_pieces:
        if type(piece) is Bishop:
            if _bishop_attacks_square(board, piece.pos, (r, c), piece.color):
                return True

    return False

def is_in_check(board, color):
    king_pos = board.find_king_pos(color)
    if not king_pos: return True
    opponent_color = "black" if color == "white" else "white"
    return is_square_attacked(board, king_pos[0], king_pos[1], opponent_color)

def generate_legal_moves_generator(board, color, yield_boards=False):
    piece_list = board.white_pieces if color == 'white' else board.black_pieces
    opponent_color = "black" if color == "white" else "white"
    for piece in piece_list: 
        start_pos = piece.pos
        if start_pos is None: continue
        for end_pos in piece.get_valid_moves(board, start_pos):
            sim_board = board.clone()
            sim_board.make_move(start_pos, end_pos)
            king_pos = sim_board.find_king_pos(color)
            if king_pos and not is_square_attacked(sim_board, king_pos[0], king_pos[1], opponent_color):
                if yield_boards:
                    yield (start_pos, end_pos), sim_board
                else:
                    yield (start_pos, end_pos)

def get_all_legal_moves(board, color):
    return list(generate_legal_moves_generator(board, color))

def get_all_pseudo_legal_moves(board, color):
    moves =[]
    piece_list = board.white_pieces if color == 'white' else board.black_pieces
    for piece in piece_list:
        if piece.pos is not None:
            moves.extend([(piece.pos, end_pos) for end_pos in piece.get_valid_moves(board, piece.pos)])
    return moves

def has_legal_moves(board, color):
    try:
        next(generate_legal_moves_generator(board, color))
        return True
    except StopIteration:
        return False
        
def is_insufficient_material(board):
    """
    In Jungle Chess, pieces are extremely lethal. 
    We ONLY declare a draw if only the two Kings remain.
    As long as any other piece exists, a win is mathematically possible.
    """
    return (len(board.white_pieces) + len(board.black_pieces)) <= 2

def get_game_state(board, turn_to_move, position_counts, ply_count, max_moves):
    """
    Adjudicates the game state. Responds correctly to Jungle Chess 
    piece lethality and tablebase outcomes.
    """
    # 1. CHECK legal moves first (Checkmate or Stalemate)
    if not has_legal_moves(board, turn_to_move):
        winner = 'black' if turn_to_move == 'white' else 'white'
        if is_in_check(board, turn_to_move):
            return ("checkmate", winner)
        else:
            return ("stalemate", None)
    
    # 2. Physical Insufficiency (Only King vs King)
    if is_insufficient_material(board):
        return ("insufficient_material", None)
    
    # 3. Three-fold Repetition
    # Important: We use 3-fold (not 2-fold) to allow for triangulation 
    # to reach tablebase wins.
    try:
        from AI import board_hash
        current_hash = board_hash(board, turn_to_move)
        if position_counts.get(current_hash, 0) >= 3:
            return ("repetition", None)
    except ImportError:
        pass
        
    # 4. Hard Move Limit (The 50-move rule equivalent)
    if ply_count >= max_moves:
        return ("move_limit", None)
        
    return "ongoing", None

def calculate_material_swing(board, move, tapered_vals_by_type):
    friendly_lost, opponent_captured, promotion_type = board.get_move_outcome(move)
    if not friendly_lost and not opponent_captured and promotion_type is None: return 0
    swing = 0
    for piece in opponent_captured:
        swing += tapered_vals_by_type.get(type(piece), 0)
    for piece in friendly_lost:
        swing -= tapered_vals_by_type.get(type(piece), 0)
    if promotion_type is not None:
        swing += tapered_vals_by_type.get(promotion_type, 0)
    return swing

def is_draw(board, turn_to_move, position_counts, ply_count, max_moves):
    """Simple wrapper for UI/Search checks."""
    state, _ = get_game_state(board, turn_to_move, position_counts, ply_count, max_moves)
    return state in ["stalemate", "insufficient_material", "repetition", "move_limit"]

def is_rook_piercing_capture(board, move):
    start, end = move
    moving_piece = board.grid[start[0]][start[1]]
    if not isinstance(moving_piece, Rook): return False
    if board.grid[end[0]][end[1]] is not None: return False
    dr = (end[0] > start[0]) - (start[0] > end[0])
    dc = (end[1] > start[1]) - (start[1] > end[1])
    cr, cc = start[0] + dr, start[1] + dc
    while (cr, cc) != end:
        target = board.grid[cr][cc]
        if target and target.color != moving_piece.color:
            return True
        cr += dr; cc += dc
    return False

def generate_all_captures(board, color):
    piece_list = board.white_pieces if color == 'white' else board.black_pieces
    for piece in piece_list:
        start_pos = piece.pos
        if start_pos is None: continue
        if isinstance(piece, Pawn):
            promotion_rank = 0 if piece.color == "white" else (ROWS - 1)
            for end_pos in piece.get_valid_moves(board, start_pos):
                if board.grid[end_pos[0]][end_pos[1]] is not None or end_pos[0] == promotion_rank:
                    yield (start_pos, end_pos)
        else:
            for end_pos in piece.get_valid_moves(board, start_pos):
                # Short-circuit checking ensures is_rook_piercing is only called when needed
                if board.grid[end_pos[0]][end_pos[1]] is not None or (isinstance(piece, Rook) and is_rook_piercing_capture(board, (start_pos, end_pos))):
                    yield (start_pos, end_pos)

def is_quiet_knight_evaporation(board, move):
    start_pos, end_pos = move
    moving_piece = board.grid[start_pos[0]][start_pos[1]]
    if not isinstance(moving_piece, Knight) or board.grid[end_pos[0]][end_pos[1]] is not None:
        return False
    for r, c in KNIGHT_ATTACKS_FROM.get(end_pos, set()):
        target = board.grid[r][c]
        if target and target.color == moving_piece.opponent_color:
            return True
    return False

def is_passive_knight_zone_evaporation(board, move):
    start_pos, end_pos = move
    moving_piece = board.grid[start_pos[0]][start_pos[1]]
    if moving_piece is None or isinstance(moving_piece, Knight):
        return False

    for r, c in KNIGHT_ATTACKS_FROM.get(end_pos, set()):
        potential_killer = board.grid[r][c]
        if (potential_killer and isinstance(potential_killer, Knight) and
            potential_killer.color != moving_piece.color):
            return True
    return False

def _first_piece_in_direction(board, start, dr, dc):
    r, c = start[0] + dr, start[1] + dc
    while 0 <= r < ROWS and 0 <= c < COLS:
        piece = board.grid[r][c]
        if piece is not None:
            return piece, (r, c)
        r += dr
        c += dc
    return None, None

def _is_between(a, b, x):
    dr = (b[0] > a[0]) - (b[0] < a[0])
    dc = (b[1] > a[1]) - (b[1] < a[1])
    r, c = a[0] + dr, a[1] + dc
    while (r, c) != b:
        if (r, c) == x:
            return True
        r += dr
        c += dc
    return False

def is_discovered_slider_unlock(board, move):
    """
    Moving a blocker can uncover a friendly Queen/Rook line to an enemy piece.
    Knights are excluded because they do not depend on line-of-sight.
    """
    start_pos, end_pos = move
    moving_piece = board.grid[start_pos[0]][start_pos[1]]
    if moving_piece is None or isinstance(moving_piece, Knight):
        return False

    my_color = moving_piece.color
    opp_color = moving_piece.opponent_color

    for dr, dc in DIRECTIONS['queen']:
        p_plus, pos_plus = _first_piece_in_direction(board, start_pos, dr, dc)
        p_minus, pos_minus = _first_piece_in_direction(board, start_pos, -dr, -dc)
        if p_plus is None or p_minus is None:
            continue

        # Check both orientations around the moved blocker.
        for slider, slider_pos, target, target_pos in (
            (p_plus, pos_plus, p_minus, pos_minus),
            (p_minus, pos_minus, p_plus, pos_plus),
        ):
            if slider.color != my_color or target.color != opp_color:
                continue

            slider_is_queen = isinstance(slider, Queen)
            slider_is_rook = isinstance(slider, Rook) and (dr == 0 or dc == 0)
            if not (slider_is_queen or slider_is_rook):
                continue

            # If destination still lies between slider and target, line stays blocked.
            if _is_between(slider_pos, target_pos, end_pos):
                continue
            return True

    return False

def generate_all_tactical_moves(board, color):
    piece_list = board.white_pieces if color == 'white' else board.black_pieces
    
    for piece in piece_list:
        start_pos = piece.pos
        if start_pos is None: continue

        for end_pos in piece.get_valid_moves(board, start_pos):
            is_capture = board.grid[end_pos[0]][end_pos[1]] is not None
            is_promotion = isinstance(piece, Pawn) and (end_pos[0] == 0 or end_pos[0] == ROWS - 1)

            if is_capture or is_promotion:
                yield (start_pos, end_pos)
                continue

            # Include variant-specific tactical motifs without full outcome simulation.
            move = (start_pos, end_pos)
            if isinstance(piece, Rook) and is_rook_piercing_capture(board, move):
                yield (start_pos, end_pos)
            elif isinstance(piece, Knight) and is_quiet_knight_evaporation(board, move):
                yield (start_pos, end_pos)
            elif is_passive_knight_zone_evaporation(board, move):
                yield (start_pos, end_pos)

def fast_approximate_material_swing(board, move, moving_piece, target_piece, piece_values):
    """
    A highly optimized, allocation-free version of get_move_outcome.
    Used purely by the AI for move ordering (Static Exchange Evaluation).
    """
    swing = 0
    
    # 1. Base Capture Value
    if target_piece is not None:
        swing += piece_values.get(type(target_piece), 0)

    # Cache type for performance (faster than calling type() repeatedly)
    my_type = type(moving_piece)

    # 2. Handle Pawn Promotion
    # We must check this first because a promoting pawn changes type to Queen for calculation purposes
    if my_type is Pawn and (move[1][0] == 0 or move[1][0] == ROWS - 1):
        swing += piece_values.get(Queen, 0) - piece_values.get(Pawn, 0)
        # Note: We do NOT change my_type to Queen here for the logic below.
        # A pawn capturing on the last rank does NOT explode on that turn. 
        # It promotes *after* the move.

    # 3. Queen Explosive Capture
    if my_type is Queen and target_piece is not None:
        swing -= piece_values.get(Queen, 0)  # Queen dies in the explosion
        for r, c in ADJACENT_SQUARES_MAP.get(move[1], []):
            adj = board.grid[r][c]
            if adj and adj.color != moving_piece.color:
                swing += piece_values.get(type(adj), 0)
        return swing # Queen explodes, no further passive evaporation applies

    # 4. Rook Piercing
    # We must track pierced knights so we don't fear evaporation from ghosts
    pierced_knights = []
    if my_type is Rook:
        start, end = move
        dr = (end[0] > start[0]) - (start[0] > end[0])
        dc = (end[1] > start[1]) - (start[1] > end[1])
        cr, cc = start[0] + dr, start[1] + dc
        
        while (cr, cc) != end:
            target = board.grid[cr][cc]
            if target and target.color != moving_piece.color:
                swing += piece_values.get(type(target), 0)
                if type(target) is Knight:
                    pierced_knights.append((cr, cc))
            cr += dr; cc += dc

    # 5. Knight Active Evaporation (The Knight moves)
    if my_type is Knight:
        evaporates_self = False
        for r, c in KNIGHT_ATTACKS_FROM.get(move[1], []):
            adj = board.grid[r][c]
            if adj and adj.color != moving_piece.color:
                swing += piece_values.get(type(adj), 0)
                if type(adj) is Knight:
                    evaporates_self = True
        if evaporates_self:
            swing -= piece_values.get(Knight, 0)
        return swing

    # 6. Passive Knight Zone Evaporation (The Piece lands)
    # Applies to Rooks, Bishops, Kings, and Pawns landing in a death zone.
    if my_type is not Knight:
        for r, c in KNIGHT_ATTACKS_FROM.get(move[1], []):
            potential_killer = board.grid[r][c]
            if potential_killer and type(potential_killer) is Knight and potential_killer.color != moving_piece.color:
                # Critical Check: Did we just pierce/kill this knight with a Rook?
                # If so, it cannot kill us.
                if (r, c) not in pierced_knights:
                    swing -= piece_values.get(my_type, 0)
                    break 

    return swing

def format_move(move):
    if not move: return "None"
    (r1, c1), (r2, c2) = move
    return f"{'abcdefgh'[c1]}{'87654321'[r1]}-{'abcdefgh'[c2]}{'87654321'[r2]}"
