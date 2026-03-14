# GameLogic.py (v52 - Performance Optimised)
#
# Key changes vs v51:
#   1. generate_legal_moves_generator — uses make_move_track/unmake_move instead of
#      Board.clone() for every pseudo-legal move.  This is the single largest
#      performance improvement: eliminates ~150 full board clones per legal-move query,
#      which the AI calls hundreds of thousands of times per search.
#
#   2. is_square_attacked — knight check rewritten to use KNIGHT_ATTACKS_FROM as a
#      reverse-lookup table.  Direct attack drops from O(n_pieces × 8) to O(8);
#      two-hop evaporation check drops from O(n_pieces × 64) to O(72).
#
#   3. Pawn.promo_rank cached — removes a conditional from every make_move call.
#
#   4. Bishop.get_valid_moves — uses set() for deduplication (faster than dict).
#
#   5. format_move_san dead_squares — iterates pieces (O(n)) instead of the grid
#      (O(64)), and handles the special end_pos case directly.

# -----------------------------
# Global Constants
# -----------------------------
ROWS, COLS = 8, 8
SQUARE_SIZE = 75
BOARD_COLOR_1 = "#D2B48C"
BOARD_COLOR_2 = "#8B5A2B"

DIRECTIONS = {
    'king':   ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)),
    'queen':  ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)),
    'rook':   ((0, 1), (0, -1), (1, 0), (-1, 0)),
    'bishop': ((-1, -1), (-1, 1), (1, -1), (1, 1)),
    'knight': ((2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2))
}
ADJACENT_DIRS = DIRECTIONS['king']

BISHOP_ZIGZAG_DIRS = (
    ((-1, 1), (-1, -1)), ((-1, -1), (-1, 1)),
    ((1, 1),  (1, -1)),  ((1, -1),  (1, 1)),
    ((-1, 1), (1, 1)),   ((1, 1),   (-1, 1)),
    ((-1, -1), (1, -1)), ((1, -1),  (-1, -1))
)

KNIGHT_ATTACKS_FROM = {
    (r, c): [(r + dr, c + dc) for dr, dc in DIRECTIONS['knight']
             if 0 <= r + dr < ROWS and 0 <= c + dc < COLS]
    for r in range(ROWS) for c in range(COLS)
}
ADJACENT_SQUARES_MAP = {
    (r, c): [(r + dr, c + dc) for dr, dc in ADJACENT_DIRS
             if 0 <= r + dr < ROWS and 0 <= c + dc < COLS]
    for r in range(ROWS) for c in range(COLS)
}

RAYS = [[[] for _ in range(8)] for _ in range(64)]

def _init_rays():
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

def _clone_piece_fast(piece):
    cls = piece.__class__
    new_piece = cls.__new__(cls)
    new_piece.color          = piece.color
    new_piece.opponent_color = piece.opponent_color
    new_piece.pos            = piece.pos
    if cls is Pawn:
        new_piece.direction    = piece.direction
        new_piece.starting_row = piece.starting_row
        new_piece.promo_rank   = piece.promo_rank   # cached attribute
    return new_piece


# ---------------------------------------------------
# PIECE CLASSES
# ---------------------------------------------------
class Piece:
    def __init__(self, color):
        self.color          = color
        self.opponent_color = "black" if color == "white" else "white"
        self.pos            = None

    def clone(self):
        new_piece     = self.__class__(self.color)
        new_piece.pos = self.pos
        return new_piece

    def symbol(self):             return "?"
    def get_valid_moves(self, board, pos): return []


class King(Piece):
    def symbol(self): return "♔" if self.color == "white" else "♚"

    def get_valid_moves(self, board, pos):
        moves = []
        r_start, c_start = pos
        for dr, dc in DIRECTIONS['king']:
            r1, c1 = r_start + dr, c_start + dc
            if 0 <= r1 < ROWS and 0 <= c1 < COLS:
                target = board.grid[r1][c1]
                if target is None or target.color == self.opponent_color:
                    moves.append((r1, c1))
                    if target is None:
                        r2, c2 = r1 + dr, c1 + dc
                        if 0 <= r2 < ROWS and 0 <= c2 < COLS:
                            t2 = board.grid[r2][c2]
                            if t2 is None or t2.color == self.opponent_color:
                                moves.append((r2, c2))
        return moves


class Queen(Piece):
    def symbol(self): return "♕" if self.color == "white" else "♛"

    def get_valid_moves(self, board, pos):
        moves = []
        grid  = board.grid
        start_index = pos[0] * COLS + pos[1]
        for i in range(8):
            for r, c in RAYS[start_index][i]:
                target = grid[r][c]
                if target is None:
                    moves.append((r, c))
                else:
                    if target.color != self.color:
                        moves.append((r, c))
                    break
        return moves


class Rook(Piece):
    def symbol(self): return "♖" if self.color == "white" else "♜"

    def get_valid_moves(self, board, pos):
        moves = []
        grid  = board.grid
        start_index = pos[0] * COLS + pos[1]
        for i in range(4):
            for r, c in RAYS[start_index][i]:
                target = grid[r][c]
                if target and target.color == self.color:
                    break
                moves.append((r, c))
        return moves


class Bishop(Piece):
    def symbol(self): return "♗" if self.color == "white" else "♝"

    def get_valid_moves(self, board, pos):
        # Use set for deduplication (straight diagonal + zigzag can overlap)
        moves = set()
        grid  = board.grid
        r_start, c_start = pos
        start_index = r_start * COLS + c_start

        for i in range(4, 8):
            for r, c in RAYS[start_index][i]:
                target = grid[r][c]
                if target:
                    if target.color != self.color:
                        moves.add((r, c))
                    break
                moves.add((r, c))

        for d1, d2 in BISHOP_ZIGZAG_DIRS:
            cr, cc, cd = r_start, c_start, d1
            while True:
                cr += cd[0]
                cc += cd[1]
                if not (0 <= cr < ROWS and 0 <= cc < COLS):
                    break
                target = grid[cr][cc]
                if target:
                    if target.color != self.color:
                        moves.add((cr, cc))
                    break
                moves.add((cr, cc))
                cd = d2 if cd == d1 else d1

        return list(moves)


class Knight(Piece):
    def symbol(self): return "♘" if self.color == "white" else "♞"

    def get_valid_moves(self, board, pos):
        return [(r, c) for r, c in KNIGHT_ATTACKS_FROM[pos]
                if board.grid[r][c] is None]


class Pawn(Piece):
    def __init__(self, color):
        super().__init__(color)
        self.direction    = -1 if color == "white" else 1
        self.starting_row = 6  if color == "white" else 1
        self.promo_rank   = 0  if color == "white" else ROWS - 1  # cached

    def symbol(self): return "♙" if self.color == "white" else "♟"

    def get_valid_moves(self, board, pos):
        moves     = []
        r, c      = pos
        direction = self.direction
        grid      = board.grid

        one_r = r + direction
        if 0 <= one_r < ROWS:
            target1 = grid[one_r][c]
            if target1 is None or target1.color == self.opponent_color:
                moves.append((one_r, c))
                if r == self.starting_row and target1 is None:
                    two_r   = r + (2 * direction)
                    target2 = grid[two_r][c]
                    if target2 is None or target2.color == self.opponent_color:
                        moves.append((two_r, c))

        if c > 0:
            target = grid[r][c - 1]
            if target and target.color == self.opponent_color:
                moves.append((r, c - 1))
        if c < COLS - 1:
            target = grid[r][c + 1]
            if target and target.color == self.opponent_color:
                moves.append((r, c + 1))
        return moves


# ---------------------------------------------------
# MOVE RECORD — lightweight snapshot for make/unmake
# ---------------------------------------------------
class MoveRecord:
    """
    Records all board mutations caused by one call to make_move_track() so they
    can be reversed exactly by unmake_move().

    removed_pieces : list[(piece, r, c)]  — every piece removed from the board
    added_pieces   : list[(piece, r, c)]  — every piece added  (promotions only)

    Design invariants
    -----------------
    * The moving piece always appears in removed_pieces when it self-destructs
      (queen explosion, knight self-evap, pawn promotion).  unmake_move skips it
      there and restores it via the 'mp_pos is None' branch instead.
    * Pieces created during this move (promoted queens) are tracked in added_pieces
      AND possibly in removed_pieces (if immediately evaporated).  unmake_move
      uses added_ids to skip re-adding them in step 3.
    """
    __slots__ = ('start', 'end', 'moving_piece', 'removed_pieces', 'added_pieces')

    def __init__(self, start, end, moving_piece):
        self.start         = start
        self.end           = end
        self.moving_piece  = moving_piece
        self.removed_pieces = []   # (piece, r, c)
        self.added_pieces   = []   # (piece, r, c)  — promotion queens only


# ---------------------------------------------
# Board Class
# ---------------------------------------------
class Board:
    def __init__(self, setup=True):
        self.grid        = [[None] * COLS for _ in range(ROWS)]
        self.white_king_pos = None
        self.black_king_pos = None
        self.white_pieces   = []
        self.black_pieces   = []
        self.piece_counts   = {
            'white': {Pawn: 0, Knight: 0, Bishop: 0, Rook: 0, Queen: 0, King: 0},
            'black': {Pawn: 0, Knight: 0, Bishop: 0, Rook: 0, Queen: 0, King: 0},
        }
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

    def add_piece(self, piece, r, c):
        if self.grid[r][c] is not None:
            self.remove_piece(r, c)
        self.grid[r][c] = piece
        piece.pos = (r, c)
        if piece.color == 'white':
            self.white_pieces.append(piece)
        else:
            self.black_pieces.append(piece)
        self.piece_counts[piece.color][type(piece)] += 1
        if isinstance(piece, King):
            if piece.color == 'white': self.white_king_pos = (r, c)
            else:                      self.black_king_pos = (r, c)

    def remove_piece(self, r, c):
        piece = self.grid[r][c]
        if not piece:
            return
        try:
            if piece.color == 'white': self.white_pieces.remove(piece)
            else:                      self.black_pieces.remove(piece)
        except ValueError:
            pass
        self.piece_counts[piece.color][type(piece)] -= 1
        if isinstance(piece, King):
            if piece.color == 'white': self.white_king_pos = None
            else:                      self.black_king_pos = None
        piece.pos       = None
        self.grid[r][c] = None

    def move_piece(self, start, end):
        piece = self.grid[start[0]][start[1]]
        if not piece:
            return
        piece.pos = end
        if isinstance(piece, King):
            if piece.color == 'white': self.white_king_pos = end
            else:                      self.black_king_pos = end
        self.grid[start[0]][start[1]] = None
        self.grid[end[0]][end[1]]     = piece

    def find_king_pos(self, color):
        return self.white_king_pos if color == 'white' else self.black_king_pos

    def clone(self):
        new_board = Board.__new__(Board)
        new_board.grid          = [[None] * COLS for _ in range(ROWS)]
        new_board.white_king_pos = self.white_king_pos
        new_board.black_king_pos = self.black_king_pos

        white_pieces = [_clone_piece_fast(p) for p in self.white_pieces]
        black_pieces = [_clone_piece_fast(p) for p in self.black_pieces]
        new_board.white_pieces = white_pieces
        new_board.black_pieces = black_pieces

        grid = new_board.grid
        for p in white_pieces:
            r, c = p.pos
            grid[r][c] = p
        for p in black_pieces:
            r, c = p.pos
            grid[r][c] = p

        pc = self.piece_counts
        new_board.piece_counts = {
            'white': pc['white'].copy(),
            'black': pc['black'].copy(),
        }
        return new_board

    def make_move(self, start, end):
        moving_piece = self.grid[start[0]][start[1]]
        if not moving_piece:
            return

        target_piece = self.grid[end[0]][end[1]]
        is_capture   = target_piece is not None

        if isinstance(moving_piece, Rook):
            self._apply_rook_piercing(start, end, moving_piece.color)
        if is_capture:
            self.remove_piece(end[0], end[1])
        self.move_piece(start, end)

        if isinstance(moving_piece, Queen) and is_capture:
            self._apply_queen_aoe(end, moving_piece.color)
        elif isinstance(moving_piece, Pawn) and end[0] == moving_piece.promo_rank:
            self.remove_piece(end[0], end[1])
            self.add_piece(Queen(moving_piece.color), end[0], end[1])

        self._apply_knight_aoe(end, is_active_move=isinstance(moving_piece, Knight))

    # ------------------------------------------------------------------
    # make_move_track / unmake_move  (used by search — no allocation)
    # ------------------------------------------------------------------

    def make_move_track(self, start, end):
        """
        Execute a move exactly like make_move() but return a MoveRecord that
        allows the board to be restored perfectly via unmake_move().
        """
        moving_piece = self.grid[start[0]][start[1]]
        record  = MoveRecord(start, end, moving_piece)
        removed = record.removed_pieces
        mc      = moving_piece.color

        target_piece = self.grid[end[0]][end[1]]
        is_capture   = target_piece is not None

        # ── 1. Rook piercing ──
        if isinstance(moving_piece, Rook):
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

        # ── 2. Capture ──
        if is_capture:
            removed.append((target_piece, end[0], end[1]))
            self.remove_piece(end[0], end[1])

        # ── 3. Move ──
        self.move_piece(start, end)

        # ── 4. Queen AOE ──
        if isinstance(moving_piece, Queen) and is_capture:
            removed.append((moving_piece, end[0], end[1]))
            self.remove_piece(end[0], end[1])
            for r, c in ADJACENT_SQUARES_MAP.get(end, ()):
                adj = self.grid[r][c]
                if adj is not None and adj.color != mc:
                    removed.append((adj, r, c))
                    self.remove_piece(r, c)

        # ── 5. Pawn promotion ──
        elif isinstance(moving_piece, Pawn) and end[0] == moving_piece.promo_rank:
            removed.append((moving_piece, end[0], end[1]))
            self.remove_piece(end[0], end[1])
            new_queen = Queen(mc)
            self.add_piece(new_queen, end[0], end[1])
            record.added_pieces.append((new_queen, end[0], end[1]))

        # ── 6. Knight AOE ──
        grid = self.grid
        if isinstance(moving_piece, Knight):
            knight_instance = grid[end[0]][end[1]]
            if knight_instance is not None:
                captured_ev, passive_losses, enemy_knights = \
                    self._collect_knight_evaporation(end, mc, knight_instance)
                delayed = set(enemy_knights)
                done    = set()

                for piece in captured_ev + passive_losses:
                    if (piece is None or piece is knight_instance or
                            piece in delayed or piece in done or piece.pos is None):
                        continue
                    removed.append((piece, piece.pos[0], piece.pos[1]))
                    self.remove_piece(piece.pos[0], piece.pos[1])
                    done.add(piece)

                for ek in enemy_knights:
                    if ek.pos is not None:
                        removed.append((ek, ek.pos[0], ek.pos[1]))
                        self.remove_piece(ek.pos[0], ek.pos[1])

                if knight_instance in passive_losses and knight_instance.pos is not None:
                    removed.append((knight_instance, end[0], end[1]))
                    self.remove_piece(end[0], end[1])

        else:
            # Passive evaporation
            victim = grid[end[0]][end[1]]
            if victim is not None:
                for r, c in KNIGHT_ATTACKS_FROM[end]:
                    killer = grid[r][c]
                    if (killer is not None and type(killer) is Knight
                            and killer.color != victim.color):
                        removed.append((victim, end[0], end[1]))
                        self.remove_piece(end[0], end[1])
                        break

        return record

    def unmake_move(self, record):
        """Restore the board to the exact state before make_move_track()."""
        start        = record.start
        moving_piece = record.moving_piece
        removed      = record.removed_pieces

        added_ids = {id(p) for p, _r, _c in record.added_pieces} if record.added_pieces else set()

        # ── 1. Undo promoted pieces ──
        for piece, r, c in reversed(record.added_pieces):
            if piece.pos is not None:
                self.grid[r][c] = None
                piece.pos = None
                try:
                    if piece.color == 'white': self.white_pieces.remove(piece)
                    else:                      self.black_pieces.remove(piece)
                except ValueError:
                    pass
                self.piece_counts[piece.color][type(piece)] -= 1

        # ── 2. Restore moving piece to start ──
        mp_pos = moving_piece.pos
        if mp_pos is not None:
            self.grid[mp_pos[0]][mp_pos[1]] = None
            self.grid[start[0]][start[1]]   = moving_piece
            moving_piece.pos = start
            if isinstance(moving_piece, King):
                if moving_piece.color == 'white': self.white_king_pos = start
                else:                             self.black_king_pos = start
        else:
            self.grid[start[0]][start[1]] = moving_piece
            moving_piece.pos = start
            if moving_piece.color == 'white': self.white_pieces.append(moving_piece)
            else:                             self.black_pieces.append(moving_piece)
            self.piece_counts[moving_piece.color][type(moving_piece)] += 1
            if isinstance(moving_piece, King):
                if moving_piece.color == 'white': self.white_king_pos = start
                else:                             self.black_king_pos = start

        # ── 3. Restore removed pieces ──
        for piece, r, c in removed:
            if piece is moving_piece:    continue
            if id(piece) in added_ids:   continue
            self.grid[r][c] = piece
            piece.pos = (r, c)
            if piece.color == 'white': self.white_pieces.append(piece)
            else:                      self.black_pieces.append(piece)
            self.piece_counts[piece.color][type(piece)] += 1
            if isinstance(piece, King):
                if piece.color == 'white': self.white_king_pos = (r, c)
                else:                      self.black_king_pos = (r, c)

    # ------------------------------------------------------------------
    # Special-effect helpers
    # ------------------------------------------------------------------

    def _apply_queen_aoe(self, pos, queen_color):
        if self.grid[pos[0]][pos[1]]:
            self.remove_piece(pos[0], pos[1])
        for r, c in ADJACENT_SQUARES_MAP.get(pos, set()):
            adj_piece = self.grid[r][c]
            if adj_piece and adj_piece.color != queen_color:
                self.remove_piece(r, c)

    def _apply_rook_piercing(self, start, end, rook_color):
        dr = (end[0] > start[0]) - (start[0] > end[0])
        dc = (end[1] > start[1]) - (start[1] > end[1])
        cr, cc = start[0] + dr, start[1] + dc
        while (cr, cc) != end:
            target = self.grid[cr][cc]
            if target and target.color != rook_color:
                self.remove_piece(cr, cc)
            cr += dr
            cc += dc

    def _apply_knight_aoe(self, pos, is_active_move):
        grid = self.grid
        if is_active_move:
            knight_instance = grid[pos[0]][pos[1]]
            if not knight_instance:
                return
            captured, passive_losses, enemy_knights = self._collect_knight_evaporation(
                pos, knight_instance.color, knight_instance)
            delayed_knights = set(enemy_knights)
            removed = set()
            for piece in captured + passive_losses:
                if (piece is None or piece is knight_instance or piece in delayed_knights
                        or piece in removed or piece.pos is None):
                    continue
                self.remove_piece(piece.pos[0], piece.pos[1])
                removed.add(piece)
            for enemy_knight in enemy_knights:
                if enemy_knight.pos is not None:
                    self.remove_piece(enemy_knight.pos[0], enemy_knight.pos[1])
            if knight_instance in passive_losses and knight_instance.pos is not None:
                self.remove_piece(pos[0], pos[1])
        else:
            victim = grid[pos[0]][pos[1]]
            if not victim:
                return
            for r, c in KNIGHT_ATTACKS_FROM[pos]:
                potential_killer = grid[r][c]
                if (potential_killer and isinstance(potential_killer, Knight)
                        and potential_killer.color != victim.color):
                    self.remove_piece(pos[0], pos[1])
                    return

    def _get_rook_piercing_captures(self, start, end, rook_color):
        captured = []
        dr = (end[0] > start[0]) - (start[0] > end[0])
        dc = (end[1] > start[1]) - (start[1] > end[1])
        cr, cc = start[0] + dr, start[1] + dc
        while (cr, cc) != end:
            target = self.grid[cr][cc]
            if target and target.color != rook_color:
                captured.append(target)
            cr += dr
            cc += dc
        return captured

    def _get_queen_aoe_captures(self, pos, queen_color):
        captured = []
        for r, c in ADJACENT_SQUARES_MAP.get(pos, set()):
            adj_piece = self.grid[r][c]
            if adj_piece and adj_piece.color != queen_color:
                captured.append(adj_piece)
        return captured

    def _collect_knight_evaporation(self, knight_pos, knight_color, self_piece=None):
        captured      = []
        passive_losses = []
        enemy_knights  = []
        if self_piece is None:
            self_piece = self.grid[knight_pos[0]][knight_pos[1]]

        for r, c in KNIGHT_ATTACKS_FROM.get(knight_pos, ()):
            target = self.grid[r][c]
            if target and target.color != knight_color:
                captured.append(target)
                if type(target) is Knight:
                    enemy_knights.append(target)

        if enemy_knights:
            for enemy_knight in enemy_knights:
                if enemy_knight.pos is None:
                    continue
                for r, c in KNIGHT_ATTACKS_FROM.get(enemy_knight.pos, ()):
                    target = self_piece if (r, c) == knight_pos else self.grid[r][c]
                    if target and target.color == knight_color and target not in passive_losses:
                        passive_losses.append(target)

        return captured, passive_losses, enemy_knights

    def _get_knight_aoe_outcome(self, knight_pos, knight_color, self_piece=None):
        captured, passive_losses, _ = self._collect_knight_evaporation(
            knight_pos, knight_color, self_piece)
        return captured, passive_losses

    def get_move_outcome(self, move):
        start_pos, end_pos = move
        moving_piece = self.grid[start_pos[0]][start_pos[1]]
        if not moving_piece:
            return set(), set(), None

        friendly_lost    = set()
        opponent_captured = set()
        promotion_type   = None

        target_piece = self.grid[end_pos[0]][end_pos[1]]
        is_capture   = target_piece is not None
        if is_capture:
            opponent_captured.add(target_piece)

        if isinstance(moving_piece, Rook):
            opponent_captured.update(
                self._get_rook_piercing_captures(start_pos, end_pos, moving_piece.color))
        elif isinstance(moving_piece, Queen) and is_capture:
            friendly_lost.add(moving_piece)
            opponent_captured.update(
                self._get_queen_aoe_captures(end_pos, moving_piece.color))
        elif isinstance(moving_piece, Knight):
            captures, passive_losses = self._get_knight_aoe_outcome(
                end_pos, moving_piece.color, moving_piece)
            opponent_captured.update(captures)
            friendly_lost.update(passive_losses)
        elif isinstance(moving_piece, Pawn) and end_pos[0] == moving_piece.promo_rank:
            promotion_type = Queen
            friendly_lost.add(moving_piece)

        if not isinstance(moving_piece, Knight):
            if isinstance(moving_piece, Queen) and is_capture:
                passive_victim_type = None
            elif promotion_type is not None:
                passive_victim_type = Queen
            else:
                passive_victim_type = type(moving_piece)

            if passive_victim_type is not None:
                for r, c in KNIGHT_ATTACKS_FROM.get(end_pos, set()):
                    potential_killer = self.grid[r][c]
                    if (potential_killer and isinstance(potential_killer, Knight)
                            and potential_killer.color != moving_piece.color
                            and potential_killer not in opponent_captured):
                        friendly_lost.add(passive_victim_type(moving_piece.color))
                        break

        return friendly_lost, opponent_captured, promotion_type


# ----------------------------------------------------
# GLOBAL GAME LOGIC
# ----------------------------------------------------
def _bishop_attacks_square(board, start, target, bishop_color):
    tr, tc = target
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


def is_square_attacked(board, r, c, attacking_color):
    grid            = board.grid
    defending_color = 'black' if attacking_color == 'white' else 'white'
    attacker_counts = board.piece_counts[attacking_color]

    non_king_pieces = (
        attacker_counts.get(Pawn,   0) + attacker_counts.get(Knight, 0) +
        attacker_counts.get(Bishop, 0) + attacker_counts.get(Rook,   0) +
        attacker_counts.get(Queen,  0)
    )
    if non_king_pieces == 0:
        enemy_king_pos = board.find_king_pos(attacking_color)
        if enemy_king_pos:
            kr, kc = enemy_king_pos
            dr, dc = r - kr, c - kc
            abs_dr, abs_dc = abs(dr), abs(dc)
            m_dist = max(abs_dr, abs_dc)
            if m_dist == 1:
                return True
            if m_dist == 2 and (abs_dr == abs_dc or abs_dr == 0 or abs_dc == 0):
                if grid[kr + dr // 2][kc + dc // 2] is None:
                    return True
        return False

    attacking_pieces = board.white_pieces if attacking_color == 'white' else board.black_pieces

    # ── Knight check (OPTIMISED) ──────────────────────────────────────────────
    # Instead of iterating all attacking_pieces to find knights — O(n × 8) — we
    # use KNIGHT_ATTACKS_FROM as a reverse lookup.  Any square in
    # KNIGHT_ATTACKS_FROM[(r,c)] that holds an enemy knight is a direct attacker;
    # any EMPTY square there that is itself reachable by a knight in one move
    # is a two-hop evaporation threat.  Worst case: O(8 + 8×8) = O(72).
    if attacker_counts[Knight] > 0:
        for pr, pc in KNIGHT_ATTACKS_FROM[(r, c)]:
            p = grid[pr][pc]
            if p is not None:
                # Direct knight attack
                if type(p) is Knight and p.color == attacking_color:
                    return True
            else:
                # Empty square — can an enemy knight move here and evaporate (r,c)?
                for qr, qc in KNIGHT_ATTACKS_FROM[(pr, pc)]:
                    q = grid[qr][qc]
                    if q is not None and type(q) is Knight and q.color == attacking_color:
                        return True

    # ── Queen AOE check ──────────────────────────────────────────────────────
    if attacker_counts[Queen] > 0:
        for piece in attacking_pieces:
            if type(piece) is Queen and piece.pos:
                qr, qc  = piece.pos
                q_idx   = qr * COLS + qc
                for i in range(8):
                    for cr, cc in RAYS[q_idx][i]:
                        target = grid[cr][cc]
                        if target is not None:
                            if target.color == defending_color:
                                if abs(cr - r) <= 1 and abs(cc - c) <= 1:
                                    return True
                            break

    has_rooks   = attacker_counts[Rook] > 0
    start_index = r * COLS + c

    # ── Sliding pieces (rooks on orthogonal rays, bishops on diagonal rays) ──
    for direction_idx, ray_path in enumerate(RAYS[start_index]):
        is_orthogonal    = direction_idx < 4
        defenders_passed = 0

        for cr, cc in ray_path:
            piece = grid[cr][cc]
            if piece is None:
                continue
            p_type = type(piece)

            if piece.color == attacking_color:
                if is_orthogonal:
                    if p_type is Rook:
                        return True
                else:
                    if p_type is Bishop:
                        if defenders_passed == 0:
                            return True
                break
            else:
                defenders_passed += 1
                if not has_rooks:
                    break

    # ── Pawn check ────────────────────────────────────────────────────────────
    pawn_move_dir = -1 if attacking_color == 'white' else 1
    pr = r - pawn_move_dir
    if 0 <= pr < ROWS:
        p = grid[pr][c]
        if p is not None and p.color == attacking_color and type(p) is Pawn:
            return True
        elif p is None:
            two_pr       = r - (2 * pawn_move_dir)
            starting_row = 6 if attacking_color == 'white' else 1
            if 0 <= two_pr < ROWS and two_pr == starting_row:
                p2 = grid[two_pr][c]
                if p2 is not None and p2.color == attacking_color and type(p2) is Pawn:
                    return True

    for dc_off in (-1, 1):
        pc = c + dc_off
        if 0 <= pc < COLS:
            p = grid[r][pc]
            if p is not None and p.color == attacking_color and type(p) is Pawn:
                return True

    # ── Enemy king check ──────────────────────────────────────────────────────
    enemy_king_pos = board.find_king_pos(attacking_color)
    if enemy_king_pos:
        kr, kc = enemy_king_pos
        dr, dc = r - kr, c - kc
        abs_dr, abs_dc = abs(dr), abs(dc)
        m_dist = max(abs_dr, abs_dc)
        if m_dist == 1:
            return True
        if m_dist == 2 and (abs_dr == abs_dc or abs_dr == 0 or abs_dc == 0):
            if grid[kr + dr // 2][kc + dc // 2] is None:
                return True

    # ── Bishop zigzag check ───────────────────────────────────────────────────
    if attacker_counts[Bishop] > 0:
        for piece in attacking_pieces:
            if type(piece) is Bishop and piece.pos:
                if _bishop_attacks_square(board, piece.pos, (r, c), attacking_color):
                    return True

    return False


def is_in_check(board, color):
    king_pos = board.find_king_pos(color)
    if not king_pos:
        return True
    opponent_color = "black" if color == "white" else "white"
    return is_square_attacked(board, king_pos[0], king_pos[1], opponent_color)


def generate_legal_moves_generator(board, color, yield_boards=False):
    """
    Yields legal moves for `color`.

    OPTIMISED: uses make_move_track / unmake_move so no Board.clone() is needed
    on the hot path (yield_boards=False).  A clone is only made when the caller
    explicitly requests post-move boards AND the move is legal — i.e. we no
    longer clone for every pseudo-legal candidate.

    The piece list is snapshotted before iteration because make_move_track
    temporarily mutates board.white_pieces / board.black_pieces.
    """
    piece_list = list(board.white_pieces if color == 'white' else board.black_pieces)
    for piece in piece_list:
        start_pos = piece.pos
        if start_pos is None:
            continue
        for end_pos in piece.get_valid_moves(board, start_pos):
            record   = board.make_move_track(start_pos, end_pos)
            king_pos = board.find_king_pos(color)
            legal    = king_pos is not None and not is_in_check(board, color)
            if legal and yield_boards:
                result_board = board.clone()
            board.unmake_move(record)
            if legal:
                if yield_boards:
                    yield (start_pos, end_pos), result_board
                else:
                    yield (start_pos, end_pos)


def get_all_legal_moves(board, color):
    return list(generate_legal_moves_generator(board, color))


def get_all_pseudo_legal_moves(board, color):
    moves      = []
    piece_list = board.white_pieces if color == 'white' else board.black_pieces
    for piece in piece_list:
        if piece.pos is not None:
            moves.extend(
                [(piece.pos, end_pos) for end_pos in piece.get_valid_moves(board, piece.pos)])
    return moves


def has_legal_moves(board, color):
    try:
        next(generate_legal_moves_generator(board, color))
        return True
    except StopIteration:
        return False


def is_insufficient_material(board):
    return (len(board.white_pieces) + len(board.black_pieces)) <= 2


def get_game_state(board, turn_to_move, position_counts, ply_count, max_moves):
    if not has_legal_moves(board, turn_to_move):
        winner = 'black' if turn_to_move == 'white' else 'white'
        return ("checkmate", winner)

    if is_insufficient_material(board):
        return ("insufficient_material", None)

    try:
        from AI import board_hash
        if position_counts.get(board_hash(board, turn_to_move), 0) >= 3:
            return ("repetition", None)
    except ImportError:
        pass

    if ply_count >= max_moves:
        return ("move_limit", None)

    return "ongoing", None


def calculate_material_swing(board, move, tapered_vals_by_type):
    friendly_lost, opponent_captured, promotion_type = board.get_move_outcome(move)
    if not friendly_lost and not opponent_captured and promotion_type is None:
        return 0
    swing = 0
    for piece in opponent_captured:
        swing += tapered_vals_by_type.get(type(piece), 0)
    for piece in friendly_lost:
        swing -= tapered_vals_by_type.get(type(piece), 0)
    if promotion_type is not None:
        swing += tapered_vals_by_type.get(promotion_type, 0)
    return swing


def is_draw(board, turn_to_move, position_counts, ply_count, max_moves):
    state, _ = get_game_state(board, turn_to_move, position_counts, ply_count, max_moves)
    return state in ("insufficient_material", "repetition", "move_limit")


def is_rook_piercing_capture(board, move):
    start, end   = move
    moving_piece = board.grid[start[0]][start[1]]
    if not isinstance(moving_piece, Rook):
        return False
    if board.grid[end[0]][end[1]] is not None:
        return False
    dr = (end[0] > start[0]) - (start[0] > end[0])
    dc = (end[1] > start[1]) - (start[1] > end[1])
    cr, cc = start[0] + dr, start[1] + dc
    while (cr, cc) != end:
        target = board.grid[cr][cc]
        if target and target.color != moving_piece.color:
            return True
        cr += dr
        cc += dc
    return False


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
        if (potential_killer and isinstance(potential_killer, Knight)
                and potential_killer.color != moving_piece.color):
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
    start_pos, end_pos = move
    moving_piece = board.grid[start_pos[0]][start_pos[1]]
    if moving_piece is None or isinstance(moving_piece, Knight):
        return False

    my_color  = moving_piece.color
    opp_color = moving_piece.opponent_color

    for dr, dc in DIRECTIONS['queen']:
        p_plus,  pos_plus  = _first_piece_in_direction(board, start_pos,  dr,  dc)
        p_minus, pos_minus = _first_piece_in_direction(board, start_pos, -dr, -dc)
        if p_plus is None or p_minus is None:
            continue

        for slider, slider_pos, target, target_pos in (
                (p_plus,  pos_plus,  p_minus, pos_minus),
                (p_minus, pos_minus, p_plus,  pos_plus)):
            if slider.color != my_color or target.color != opp_color:
                continue
            slider_is_queen = isinstance(slider, Queen)
            slider_is_rook  = isinstance(slider, Rook) and (dr == 0 or dc == 0)
            if not (slider_is_queen or slider_is_rook):
                continue
            if _is_between(slider_pos, target_pos, end_pos):
                continue
            return True
    return False


def generate_all_tactical_moves(board, color):
    piece_list = board.white_pieces if color == 'white' else board.black_pieces
    for piece in piece_list:
        start_pos = piece.pos
        if start_pos is None:
            continue
        for end_pos in piece.get_valid_moves(board, start_pos):
            is_capture   = board.grid[end_pos[0]][end_pos[1]] is not None
            is_promotion = isinstance(piece, Pawn) and end_pos[0] == piece.promo_rank

            if is_capture or is_promotion:
                yield (start_pos, end_pos)
                continue

            move = (start_pos, end_pos)
            if isinstance(piece, Rook) and is_rook_piercing_capture(board, move):
                yield (start_pos, end_pos)
            elif isinstance(piece, Knight) and is_quiet_knight_evaporation(board, move):
                yield (start_pos, end_pos)
            elif is_passive_knight_zone_evaporation(board, move):
                yield (start_pos, end_pos)


def fast_approximate_material_swing(board, move, moving_piece, target_piece, piece_values):
    swing   = 0
    my_type = type(moving_piece)

    if target_piece is not None:
        swing += piece_values.get(type(target_piece), 0)

    if my_type is Pawn and move[1][0] == moving_piece.promo_rank:
        swing += piece_values.get(Queen, 0) - piece_values.get(Pawn, 0)

    if my_type is Queen and target_piece is not None:
        swing -= piece_values.get(Queen, 0)
        for r, c in ADJACENT_SQUARES_MAP.get(move[1], []):
            adj = board.grid[r][c]
            if adj and adj.color != moving_piece.color:
                swing += piece_values.get(type(adj), 0)
        return swing

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
            cr += dr
            cc += dc

    if my_type is Knight:
        captures, passive_losses = board._get_knight_aoe_outcome(
            move[1], moving_piece.color, moving_piece)
        for piece in captures:
            swing += piece_values.get(type(piece), 0)
        for piece in passive_losses:
            swing -= piece_values.get(type(piece), 0)
        return swing

    # Passive evaporation check for non-knights
    for r, c in KNIGHT_ATTACKS_FROM.get(move[1], []):
        potential_killer = board.grid[r][c]
        if (potential_killer and type(potential_killer) is Knight
                and potential_killer.color != moving_piece.color
                and (r, c) not in pierced_knights):
            evap_type = Queen if (my_type is Pawn and move[1][0] == moving_piece.promo_rank) \
                        else my_type
            swing -= piece_values.get(evap_type, 0)
            break

    return swing


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
    if ptype != Knight:
        is_capture = board_before.grid[end_pos[0]][end_pos[1]] is not None

    def file_of(c):  return "abcdefgh"[c]
    def rank_of(r):  return "87654321"[r]
    def sq_name(pos): return file_of(pos[1]) + rank_of(pos[0])

    disambig = ""
    if ptype not in (Pawn, King):
        others = []
        for r in range(ROWS):
            for c in range(COLS):
                p = board_before.grid[r][c]
                if (p and type(p) == ptype and p.color == moving_piece.color
                        and (r, c) != start_pos):
                    if end_pos in p.get_valid_moves(board_before, (r, c)):
                        others.append((r, c))
        if others:
            same_file = any(pos[1] == start_pos[1] for pos in others)
            same_rank = any(pos[0] == start_pos[0] for pos in others)
            if not same_file:
                disambig = file_of(start_pos[1])
            elif not same_rank:
                disambig = rank_of(start_pos[0])
            else:
                disambig = sq_name(start_pos)

    if ptype == Pawn:
        base_str = (file_of(start_pos[1]) + "x" + sq_name(end_pos)) if is_capture \
                   else sq_name(end_pos)
    else:
        p_char   = {King: 'K', Queen: 'Q', Rook: 'R', Bishop: 'B', Knight: 'N'}[ptype]
        cap_str  = "x" if is_capture else ""
        base_str = p_char + disambig + cap_str + sq_name(end_pos)

    if ptype == Pawn and end_pos[0] == moving_piece.promo_rank:
        base_str += "=Q"

    # ── Dead squares (OPTIMISED) ──────────────────────────────────────────────
    # Iterate pieces (O(n)) instead of the full 8×8 grid (O(64)).
    #
    # Two sources of dead squares:
    #   A. end_pos — if empty in board_after, something died there (queen explosion
    #      or promoted pawn immediately evaporated).  Note: a simple capture that
    #      the mover survives leaves the mover at end_pos (not empty), so it is
    #      correctly NOT added here.
    #   B. All other squares that had a piece before but are empty after
    #      (AOE, rook piercing, knight evaporation casualties).
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
    # ─────────────────────────────────────────────────────────────────────────

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