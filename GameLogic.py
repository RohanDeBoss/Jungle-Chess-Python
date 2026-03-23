# GameLogic.py (v54 - Performance 2)


# -----------------------------------------------------------------------
# Global constants
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

def _init_rays():
    dy_dx = [(-1, 0), (1, 0), (0, 1), (0, -1), (-1, 1), (-1, -1), (1, 1), (1, -1)]
    tmp = [[[] for _ in range(8)] for _ in range(64)]
    for r in range(ROWS):
        for c in range(COLS):
            idx = r * COLS + c
            for i, (dr, dc) in enumerate(dy_dx):
                cr, cc = r + dr, c + dc
                while 0 <= cr < ROWS and 0 <= cc < COLS:
                    tmp[idx][i].append((cr, cc))
                    cr += dr
                    cc += dc
    for sq in range(64):
        RAYS[sq] = tuple(tuple(ray) for ray in tmp[sq])

_init_rays()


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
        moves       = []
        grid        = board.grid
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
        moves       = []
        grid        = board.grid
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
        moves       = set()
        grid        = board.grid
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
        self.promo_rank   = 0  if color == "white" else ROWS - 1

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


# -----------------------------------------------------------------------
# MoveRecord
# -----------------------------------------------------------------------
class MoveRecord:
    """
    Lightweight snapshot of every board mutation made by make_move_track(),
    used by unmake_move() to restore the position without cloning.

    removed_pieces : list[(piece, r, c)]  — every piece taken off the board
    added_pieces   : list[(piece, r, c)]  — pieces placed (promotions only)

    Design invariants
    -----------------
    * The moving piece appears in removed_pieces when it self-destructs
      (queen explosion, knight self-evap, pawn promotion).  unmake_move skips
      it there and restores it via the 'mp_pos is None' branch.
    * Pieces created this move (promoted queens) appear in added_pieces AND
      possibly in removed_pieces (if immediately evaporated).  added_ids in
      unmake_move guards against double-restore.
    """
    __slots__ = ('start', 'end', 'moving_piece', 'removed_pieces', 'added_pieces')

    def __init__(self, start, end, moving_piece):
        self.start          = start
        self.end            = end
        self.moving_piece   = moving_piece
        self.removed_pieces = []
        self.added_pieces   = []


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

    # ---- O(1) piece-list helpers ------------------------------------------

    def _list_append(self, piece):
        """Append to the colour list and record the resulting index."""
        lst             = self.white_pieces if piece.color == 'white' else self.black_pieces
        piece._list_pos = len(lst)
        lst.append(piece)

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

    # ---- Board mutation primitives ----------------------------------------

    def add_piece(self, piece, r, c):
        if self.grid[r][c] is not None:
            self.remove_piece(r, c)
        self.grid[r][c] = piece
        piece.pos       = (r, c)
        self._list_append(piece)
        self.piece_counts[piece.color][type(piece)] += 1
        if type(piece) is King:
            if piece.color == 'white': self.white_king_pos = (r, c)
            else:                      self.black_king_pos = (r, c)

    def remove_piece(self, r, c):
        piece = self.grid[r][c]
        if not piece:
            return
        self._list_remove(piece)
        self.piece_counts[piece.color][type(piece)] -= 1
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
        return new_board

    # -----------------------------------------------------------------------
    # make_move  (UI path — no undo needed)
    # -----------------------------------------------------------------------
    def make_move(self, start, end):
        moving_piece = self.grid[start[0]][start[1]]
        if not moving_piece:
            return

        target_piece = self.grid[end[0]][end[1]]
        is_capture   = target_piece is not None
        mp_type      = type(moving_piece)

        if mp_type is Rook:
            self._apply_rook_piercing(start, end, moving_piece.color)
        if is_capture:
            self.remove_piece(end[0], end[1])
        self.move_piece(start, end)

        if mp_type is Queen and is_capture:
            self._apply_queen_aoe(end, moving_piece.color)
        elif mp_type is Pawn and end[0] == moving_piece.promo_rank:
            self.remove_piece(end[0], end[1])
            self.add_piece(Queen(moving_piece.color), end[0], end[1])

        self._apply_knight_aoe(end, is_active_move=(mp_type is Knight))

    # -----------------------------------------------------------------------
    # make_move_track / unmake_move  (search path — no cloning)
    # -----------------------------------------------------------------------
    def make_move_track(self, start, end):
        """
        Execute the move exactly like make_move() but return a MoveRecord that
        allows the board to be restored perfectly via unmake_move().
        """
        moving_piece = self.grid[start[0]][start[1]]
        record  = MoveRecord(start, end, moving_piece)
        removed = record.removed_pieces
        mc      = moving_piece.color
        mp_type = type(moving_piece)

        target_piece = self.grid[end[0]][end[1]]
        is_capture   = target_piece is not None

        # ── 1. Rook piercing ──
        if mp_type is Rook:
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
        if mp_type is Queen and is_capture:
            removed.append((moving_piece, end[0], end[1]))
            self.remove_piece(end[0], end[1])
            for r, c in ADJACENT_SQUARES_MAP[end]:
                adj = self.grid[r][c]
                if adj is not None and adj.color != mc:
                    removed.append((adj, r, c))
                    self.remove_piece(r, c)

        # ── 5. Pawn promotion ──
        elif mp_type is Pawn and end[0] == moving_piece.promo_rank:
            removed.append((moving_piece, end[0], end[1]))
            self.remove_piece(end[0], end[1])
            new_queen = Queen(mc)
            self.add_piece(new_queen, end[0], end[1])
            record.added_pieces.append((new_queen, end[0], end[1]))

        # ── 6. Knight AOE ──
        grid = self.grid
        if mp_type is Knight:
            knight_instance = grid[end[0]][end[1]]
            if knight_instance is not None:
                captured_ev, passive_losses, enemy_knights = \
                    self._collect_knight_evaporation(end, mc, knight_instance)
                delayed = set(enemy_knights)
                done    = set()
                for piece in captured_ev + passive_losses:
                    if (piece is None or piece is knight_instance
                            or piece in delayed or piece in done
                            or piece.pos is None):
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
            # Passive evaporation check
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
                piece.pos       = None
                self._list_remove(piece)   # O(1)
                self.piece_counts[piece.color][type(piece)] -= 1

        # ── 2. Restore moving piece to start ──
        mp_pos = moving_piece.pos
        if mp_pos is not None:
            self.grid[mp_pos[0]][mp_pos[1]] = None
            self.grid[start[0]][start[1]]   = moving_piece
            moving_piece.pos = start
            if type(moving_piece) is King:
                if moving_piece.color == 'white': self.white_king_pos = start
                else:                             self.black_king_pos = start
        else:
            # Was removed this move (queen explosion / knight self-evap / promo)
            self.grid[start[0]][start[1]] = moving_piece
            moving_piece.pos = start
            self._list_append(moving_piece)   # O(1)
            self.piece_counts[moving_piece.color][type(moving_piece)] += 1
            if type(moving_piece) is King:
                if moving_piece.color == 'white': self.white_king_pos = start
                else:                             self.black_king_pos = start

        # ── 3. Restore all removed pieces ──
        for piece, r, c in removed:
            if piece is moving_piece:   continue
            if id(piece) in added_ids:  continue
            self.grid[r][c] = piece
            piece.pos       = (r, c)
            self._list_append(piece)   # O(1)
            self.piece_counts[piece.color][type(piece)] += 1
            if type(piece) is King:
                if piece.color == 'white': self.white_king_pos = (r, c)
                else:                      self.black_king_pos = (r, c)

    # -----------------------------------------------------------------------
    # Special-effect helpers
    # -----------------------------------------------------------------------
    def _apply_queen_aoe(self, pos, queen_color):
        if self.grid[pos[0]][pos[1]]:
            self.remove_piece(pos[0], pos[1])
        for r, c in ADJACENT_SQUARES_MAP[pos]:
            adj = self.grid[r][c]
            if adj and adj.color != queen_color:
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
            captured, passive_losses, enemy_knights = \
                self._collect_knight_evaporation(pos, knight_instance.color, knight_instance)
            delayed = set(enemy_knights)
            done    = set()
            for piece in captured + passive_losses:
                if (piece is None or piece is knight_instance or piece in delayed
                        or piece in done or piece.pos is None):
                    continue
                self.remove_piece(piece.pos[0], piece.pos[1])
                done.add(piece)
            for ek in enemy_knights:
                if ek.pos is not None:
                    self.remove_piece(ek.pos[0], ek.pos[1])
            if knight_instance in passive_losses and knight_instance.pos is not None:
                self.remove_piece(pos[0], pos[1])
        else:
            victim = grid[pos[0]][pos[1]]
            if not victim:
                return
            for r, c in KNIGHT_ATTACKS_FROM[pos]:
                pk = grid[r][c]
                if pk and type(pk) is Knight and pk.color != victim.color:
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
        return [
            self.grid[r][c]
            for r, c in ADJACENT_SQUARES_MAP[pos]
            if self.grid[r][c] and self.grid[r][c].color != queen_color
        ]

    def _collect_knight_evaporation(self, knight_pos, knight_color, self_piece=None):
        captured       = []
        passive_losses = []
        enemy_knights  = []
        if self_piece is None:
            self_piece = self.grid[knight_pos[0]][knight_pos[1]]

        for r, c in KNIGHT_ATTACKS_FROM[knight_pos]:
            target = self.grid[r][c]
            if target and target.color != knight_color:
                captured.append(target)
                if type(target) is Knight:
                    enemy_knights.append(target)

        if enemy_knights:
            for ek in enemy_knights:
                if ek.pos is None:
                    continue
                for r, c in KNIGHT_ATTACKS_FROM[ek.pos]:
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

        friendly_lost     = set()
        opponent_captured = set()
        promotion_type    = None
        mp_type           = type(moving_piece)

        target_piece = self.grid[end_pos[0]][end_pos[1]]
        is_capture   = target_piece is not None
        if is_capture:
            opponent_captured.add(target_piece)

        if mp_type is Rook:
            opponent_captured.update(
                self._get_rook_piercing_captures(start_pos, end_pos, moving_piece.color))
        elif mp_type is Queen and is_capture:
            friendly_lost.add(moving_piece)
            opponent_captured.update(
                self._get_queen_aoe_captures(end_pos, moving_piece.color))
        elif mp_type is Knight:
            captures, passive_losses = self._get_knight_aoe_outcome(
                end_pos, moving_piece.color, moving_piece)
            opponent_captured.update(captures)
            friendly_lost.update(passive_losses)
        elif mp_type is Pawn and end_pos[0] == moving_piece.promo_rank:
            promotion_type = Queen
            friendly_lost.add(moving_piece)

        if mp_type is not Knight:
            if mp_type is Queen and is_capture:
                passive_victim_type = None
            elif promotion_type is not None:
                passive_victim_type = Queen
            else:
                passive_victim_type = mp_type
            if passive_victim_type is not None:
                for r, c in KNIGHT_ATTACKS_FROM[end_pos]:
                    pk = self.grid[r][c]
                    if (pk and type(pk) is Knight
                            and pk.color != moving_piece.color
                            and pk not in opponent_captured):
                        friendly_lost.add(passive_victim_type(moving_piece.color))
                        break

        return friendly_lost, opponent_captured, promotion_type


# -----------------------------------------------------------------------
# Global game logic
# -----------------------------------------------------------------------
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

    # Direct [X] access — all keys always present; avoids .get() overhead
    non_king_pieces = (
        attacker_counts[Pawn]   + attacker_counts[Knight] +
        attacker_counts[Bishop] + attacker_counts[Rook]   +
        attacker_counts[Queen]
    )
    if non_king_pieces == 0:
        enemy_king_pos = board.find_king_pos(attacking_color)
        if enemy_king_pos:
            kr, kc         = enemy_king_pos
            dr, dc         = r - kr, c - kc
            abs_dr, abs_dc = abs(dr), abs(dc)
            m_dist         = max(abs_dr, abs_dc)
            if m_dist == 1:
                return True
            if m_dist == 2 and (abs_dr == abs_dc or abs_dr == 0 or abs_dc == 0):
                if grid[kr + dr // 2][kc + dc // 2] is None:
                    return True
        return False

    attacking_pieces = board.white_pieces if attacking_color == 'white' else board.black_pieces

    # ── Knight check: reverse lookup — O(8 direct + 8×8 two-hop) ──
    if attacker_counts[Knight] > 0:
        for pr, pc in KNIGHT_ATTACKS_FROM[(r, c)]:
            p = grid[pr][pc]
            if p is not None:
                if type(p) is Knight and p.color == attacking_color:
                    return True
            else:
                for qr, qc in KNIGHT_ATTACKS_FROM[(pr, pc)]:
                    q = grid[qr][qc]
                    if q is not None and type(q) is Knight and q.color == attacking_color:
                        return True

    # ── Queen AOE check ──
    if attacker_counts[Queen] > 0:
        for piece in attacking_pieces:
            if type(piece) is Queen and piece.pos:
                qr, qc = piece.pos
                q_idx  = qr * COLS + qc
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

    # ── Sliding pieces (rooks on orthogonal rays, bishops on diagonal) ──
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
                    if p_type is Bishop and defenders_passed == 0:
                        return True
                break
            else:
                defenders_passed += 1
                if not has_rooks:
                    break

    # ── Pawn check ──
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

    # ── Enemy king check ──
    enemy_king_pos = board.find_king_pos(attacking_color)
    if enemy_king_pos:
        kr, kc         = enemy_king_pos
        dr, dc         = r - kr, c - kc
        abs_dr, abs_dc = abs(dr), abs(dc)
        m_dist         = max(abs_dr, abs_dc)
        if m_dist == 1:
            return True
        if m_dist == 2 and (abs_dr == abs_dc or abs_dr == 0 or abs_dc == 0):
            if grid[kr + dr // 2][kc + dc // 2] is None:
                return True

    # ── Bishop zigzag check ──
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

    Uses make_move_track/unmake_move — no Board.clone() on the hot path.
    A clone is only made when yield_boards=True AND the move is legal.
    Piece list is snapshotted because make_move_track mutates white/black_pieces.
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
    append     = moves.append
    piece_list = board.white_pieces if color == 'white' else board.black_pieces
    for piece in piece_list:
        start_pos = piece.pos
        if start_pos is None:
            continue
        for end_pos in piece.get_valid_moves(board, start_pos):
            append((start_pos, end_pos))
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
    for r, c in KNIGHT_ATTACKS_FROM[end_pos]:
        target = board.grid[r][c]
        if target and target.color == moving_piece.opponent_color:
            return True
    return False


def is_passive_knight_zone_evaporation(board, move):
    start_pos, end_pos = move
    moving_piece = board.grid[start_pos[0]][start_pos[1]]
    if moving_piece is None or isinstance(moving_piece, Knight):
        return False
    for r, c in KNIGHT_ATTACKS_FROM[end_pos]:
        pk = board.grid[r][c]
        if pk and isinstance(pk, Knight) and pk.color != moving_piece.color:
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
            if not (isinstance(slider, Queen) or
                    (isinstance(slider, Rook) and (dr == 0 or dc == 0))):
                continue
            if _is_between(slider_pos, target_pos, end_pos):
                continue
            return True
    return False


def generate_all_tactical_moves(board, color):
    grid       = board.grid
    piece_list = board.white_pieces if color == 'white' else board.black_pieces
    for piece in piece_list:
        start_pos = piece.pos
        if start_pos is None:
            continue
        mp_type = type(piece)
        my_color = piece.color
        opp_color = piece.opponent_color
        for end_pos in piece.get_valid_moves(board, start_pos):
            end_r, end_c = end_pos
            is_capture   = grid[end_r][end_c] is not None
            is_promotion = mp_type is Pawn and end_r == piece.promo_rank

            if is_capture or is_promotion:
                yield (start_pos, end_pos)
                continue

            if mp_type is Rook:
                dr = (end_r > start_pos[0]) - (start_pos[0] > end_r)
                dc = (end_c > start_pos[1]) - (start_pos[1] > end_c)
                cr, cc = start_pos[0] + dr, start_pos[1] + dc
                while (cr, cc) != end_pos:
                    target = grid[cr][cc]
                    if target is not None and target.color != my_color:
                        yield (start_pos, end_pos)
                        break
                    cr += dr
                    cc += dc
                continue

            if mp_type is Knight:
                for r, c in KNIGHT_ATTACKS_FROM[end_pos]:
                    target = grid[r][c]
                    if target is not None and target.color == opp_color:
                        yield (start_pos, end_pos)
                        break
                continue

            for r, c in KNIGHT_ATTACKS_FROM[end_pos]:
                pk = grid[r][c]
                if pk is not None and type(pk) is Knight and pk.color != my_color:
                    yield (start_pos, end_pos)
                    break


def fast_approximate_material_swing(board, move, moving_piece, target_piece, piece_values):
    swing   = 0
    my_type = type(moving_piece)

    if target_piece is not None:
        swing += piece_values.get(type(target_piece), 0)

    if my_type is Pawn and move[1][0] == moving_piece.promo_rank:
        swing += piece_values.get(Queen, 0) - piece_values.get(Pawn, 0)

    if my_type is Queen and target_piece is not None:
        swing -= piece_values.get(Queen, 0)
        for r, c in ADJACENT_SQUARES_MAP[move[1]]:
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

    for r, c in KNIGHT_ATTACKS_FROM[move[1]]:
        pk = board.grid[r][c]
        if (pk and type(pk) is Knight
                and pk.color != moving_piece.color
                and (r, c) not in pierced_knights):
            evap_type = (Queen if (my_type is Pawn and move[1][0] == moving_piece.promo_rank)
                         else my_type)
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
                if end_pos in p.get_valid_moves(board_before, p.pos):
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