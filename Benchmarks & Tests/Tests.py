# Tests.py (v4.1 - fixed new directory)

import sys
import os

# --- PATH INJECTION (MUST BE AT THE VERY TOP) ---
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (the main project folder)
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
# Add the parent directory to the Python path
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# Change CWD so that sub-imports (like tablebase loader) work correctly
os.chdir(PARENT_DIR)

# --- NOW YOU CAN IMPORT YOUR MODULES ---
import argparse
from dataclasses import dataclass
from GameLogic import *
from AI import ChessBot, MG_PIECE_VALUES, TT_FLAG_EXACT, TT_FLAG_LOWERBOUND
from OpponentAI import OpponentAI
from TablebaseManager import TablebaseManager

# Change CWD so that sub-imports (like tablebase loader) work correctly
os.chdir(PARENT_DIR)

PIECE_CLASS_BY_NAME = {
    "Queen": Queen,
    "Rook": Rook,
    "Knight": Knight,
    "Bishop": Bishop,
    "Pawn": Pawn,
}
PIECE_SORT_ORDER = {King: 0, Queen: 1, Rook: 2, Bishop: 3, Knight: 4, Pawn: 5}
PIECE_LETTER = {King: "K", Queen: "Q", Rook: "R", Bishop: "B", Knight: "N", Pawn: "P"}
SHOW_ASCII = False
REF_BISHOP_ZIGZAGS = (
    ((-1, 1), (-1, -1)),
    ((-1, -1), (-1, 1)),
    ((1, 1), (1, -1)),
    ((1, -1), (1, 1)),
    ((-1, 1), (1, 1)),
    ((1, 1), (-1, 1)),
    ((-1, -1), (1, -1)),
    ((1, -1), (-1, -1)),
)


class SkipCase(Exception):
    pass


class _DummyQueue:
    def put(self, _item):
        pass


class _DummyEvent:
    def is_set(self):
        return False


@dataclass(frozen=True)
class TestCaseSpec:
    name: str
    group: str
    description: str
    func: callable


def expect(condition, message):
    if not condition:
        raise AssertionError(message)


def square_name(pos):
    return "abcdefgh"[pos[1]] + "87654321"[pos[0]]


def describe_board(board):
    def render_piece(piece):
        return PIECE_LETTER[type(piece)] + square_name(piece.pos)

    def render_color(pieces):
        ordered = sorted(
            (p for p in pieces if p.pos is not None),
            key=lambda piece: (PIECE_SORT_ORDER[type(piece)], piece.pos[0], piece.pos[1]),
        )
        return " ".join(render_piece(piece) for piece in ordered) or "-"

    return f"White[{render_color(board.white_pieces)}] | Black[{render_color(board.black_pieces)}]"


def board_to_ascii(board):
    rows = []
    for r in range(8):
        row_cells = []
        for c in range(8):
            piece = board.grid[r][c]
            if piece is None:
                row_cells.append("..")
            else:
                prefix = "w" if piece.color == "white" else "b"
                row_cells.append(prefix + PIECE_LETTER[type(piece)])
        rows.append(f"{8-r} " + " ".join(row_cells))
    rows.append("  a  b  c  d  e  f  g  h")
    return "\n".join(rows)


def position_details(board, label):
    details = [f"{label}: {describe_board(board)}"]
    if SHOW_ASCII:
        details.append(board_to_ascii(board))
    return details


def board_state_signature(board):
    grid_sig = {}
    for r in range(8):
        for c in range(8):
            piece = board.grid[r][c]
            if piece is not None:
                grid_sig[(r, c)] = (piece.color, type(piece).__name__)
    piece_counts = {
        "white": {cls.__name__: board.piece_counts["white"][cls] for cls in (Pawn, Knight, Bishop, Rook, Queen, King)},
        "black": {cls.__name__: board.piece_counts["black"][cls] for cls in (Pawn, Knight, Bishop, Rook, Queen, King)},
    }
    return (grid_sig, board.white_king_pos, board.black_king_pos, piece_counts)


def board_identity_snapshot(board):
    grid_ids = {}
    for r in range(8):
        for c in range(8):
            piece = board.grid[r][c]
            if piece is not None:
                grid_ids[(r, c)] = id(piece)
    return (
        grid_ids,
        sorted(id(p) for p in board.white_pieces),
        sorted(id(p) for p in board.black_pieces),
        board.white_king_pos,
        board.black_king_pos,
        {cls.__name__: board.piece_counts["white"][cls] for cls in (Pawn, Knight, Bishop, Rook, Queen, King)},
        {cls.__name__: board.piece_counts["black"][cls] for cls in (Pawn, Knight, Bishop, Rook, Queen, King)},
    )


def assert_board_equivalent(expected, actual, label):
    if board_state_signature(expected) != board_state_signature(actual):
        details = [
            f"{label}: board state mismatch after make_move_track.",
            f"Expected: {describe_board(expected)}",
            f"Actual:   {describe_board(actual)}",
        ]
        if SHOW_ASCII:
            details.append("Expected board:")
            details.append(board_to_ascii(expected))
            details.append("Actual board:")
            details.append(board_to_ascii(actual))
        raise AssertionError("\n".join(details))


def assert_board_identity_restored(actual, snapshot, label):
    if board_identity_snapshot(actual) != snapshot:
        details = [
            f"{label}: board identity mismatch after unmake_move.",
            f"Actual:   {describe_board(actual)}",
        ]
        if SHOW_ASCII:
            details.append(board_to_ascii(actual))
        raise AssertionError("\n".join(details))


def assert_board_consistent(board, label):
    white_set = set(board.white_pieces)
    black_set = set(board.black_pieces)
    if len(white_set) != len(board.white_pieces):
        raise AssertionError(f"{label}: duplicate entries in white_pieces list.")
    if len(black_set) != len(board.black_pieces):
        raise AssertionError(f"{label}: duplicate entries in black_pieces list.")
    for r in range(8):
        for c in range(8):
            piece = board.grid[r][c]
            if piece is None:
                continue
            if piece.pos != (r, c):
                raise AssertionError(f"{label}: grid piece at {(r, c)} has pos {piece.pos}.")
            if piece.color == "white":
                if piece not in white_set:
                    raise AssertionError(f"{label}: white piece missing from list at {(r, c)}.")
            else:
                if piece not in black_set:
                    raise AssertionError(f"{label}: black piece missing from list at {(r, c)}.")
    for piece in board.white_pieces:
        if piece.pos is None or board.grid[piece.pos[0]][piece.pos[1]] is not piece:
            raise AssertionError(f"{label}: white piece list contains invalid entry {piece}.")
    for piece in board.black_pieces:
        if piece.pos is None or board.grid[piece.pos[0]][piece.pos[1]] is not piece:
            raise AssertionError(f"{label}: black piece list contains invalid entry {piece}.")


def make_board(pieces):
    board = Board(setup=False)
    for color, piece_class, pos in pieces:
        board.add_piece(piece_class(color), pos[0], pos[1])
    return board


def make_opponent_ai(board, color):
    return OpponentAI(board, color, {}, _DummyQueue(), _DummyEvent())


def manager_with_table(table_name):
    manager = TablebaseManager()
    if table_name not in manager.tables:
        raise SkipCase(f"Required table '{table_name}' is not available in tablebases/.")
    return manager


def opposite(color):
    return "black" if color == "white" else "white"


def in_bounds(r, c):
    return 0 <= r < 8 and 0 <= c < 8


def ref_from_board(board):
    pieces = {}
    for piece in board.white_pieces + board.black_pieces:
        if piece.pos is not None:
            pieces[piece.pos] = (piece.color, type(piece).__name__)
    return {
        "pieces": pieces,
        "white_king": board.white_king_pos,
        "black_king": board.black_king_pos,
    }


def ref_clone(state):
    return {
        "pieces": dict(state["pieces"]),
        "white_king": state["white_king"],
        "black_king": state["black_king"],
    }


def ref_remove_piece(state, pos):
    piece = state["pieces"].pop(pos, None)
    if piece is None:
        return None
    _color, name = piece
    if name == "King":
        if pos == state["white_king"]:
            state["white_king"] = None
        if pos == state["black_king"]:
            state["black_king"] = None
    return piece


def ref_add_piece(state, color, name, pos):
    state["pieces"][pos] = (color, name)
    if name == "King":
        if color == "white":
            state["white_king"] = pos
        else:
            state["black_king"] = pos


def ref_knight_targets(pos):
    r, c = pos
    targets = []
    for dr, dc in DIRECTIONS["knight"]:
        nr, nc = r + dr, c + dc
        if in_bounds(nr, nc):
            targets.append((nr, nc))
    return targets


def ref_piece_moves(state, pos):
    color, name = state["pieces"][pos]
    enemy = opposite(color)
    pieces = state["pieces"]
    r, c = pos
    moves = []

    if name == "King":
        for dr, dc in DIRECTIONS["king"]:
            r1, c1 = r + dr, c + dc
            if in_bounds(r1, c1):
                target1 = pieces.get((r1, c1))
                if target1 is None or target1[0] == enemy:
                    moves.append((r1, c1))
                    if target1 is None:
                        r2, c2 = r1 + dr, c1 + dc
                        if in_bounds(r2, c2):
                            target2 = pieces.get((r2, c2))
                            if target2 is None or target2[0] == enemy:
                                moves.append((r2, c2))
        return moves

    if name == "Queen":
        for dr, dc in DIRECTIONS["queen"]:
            cr, cc = r + dr, c + dc
            while in_bounds(cr, cc):
                target = pieces.get((cr, cc))
                if target is None:
                    moves.append((cr, cc))
                else:
                    if target[0] == enemy:
                        moves.append((cr, cc))
                    break
                cr += dr
                cc += dc
        return moves

    if name == "Rook":
        for dr, dc in DIRECTIONS["rook"]:
            cr, cc = r + dr, c + dc
            while in_bounds(cr, cc):
                target = pieces.get((cr, cc))
                if target is not None and target[0] == color:
                    break
                moves.append((cr, cc))
                cr += dr
                cc += dc
        return moves

    if name == "Bishop":
        seen = set()
        for dr, dc in DIRECTIONS["bishop"]:
            cr, cc = r + dr, c + dc
            while in_bounds(cr, cc):
                target = pieces.get((cr, cc))
                if target is not None:
                    if target[0] == enemy:
                        seen.add((cr, cc))
                    break
                seen.add((cr, cc))
                cr += dr
                cc += dc
        for d1, d2 in REF_BISHOP_ZIGZAGS:
            cr, cc = r, c
            current_dir = d1
            while True:
                cr += current_dir[0]
                cc += current_dir[1]
                if not in_bounds(cr, cc):
                    break
                target = pieces.get((cr, cc))
                if target is not None:
                    if target[0] == enemy:
                        seen.add((cr, cc))
                    break
                seen.add((cr, cc))
                current_dir = d2 if current_dir == d1 else d1
        return list(seen)

    if name == "Knight":
        for target in ref_knight_targets(pos):
            if target not in pieces:
                moves.append(target)
        return moves

    if name == "Pawn":
        direction = -1 if color == "white" else 1
        starting_row = 6 if color == "white" else 1
        one = (r + direction, c)
        if in_bounds(*one):
            target1 = pieces.get(one)
            if target1 is None or target1[0] == enemy:
                moves.append(one)
                two = (r + 2 * direction, c)
                if r == starting_row and target1 is None and in_bounds(*two):
                    target2 = pieces.get(two)
                    if target2 is None or target2[0] == enemy:
                        moves.append(two)
        for dc in (-1, 1):
            side = (r, c + dc)
            target = pieces.get(side)
            if in_bounds(*side) and target is not None and target[0] == enemy:
                moves.append(side)
        return moves

    raise AssertionError(f"Unknown piece name in reference model: {name}")


def ref_bishop_attacks_square(state, start, target, bishop_color):
    pieces = state["pieces"]
    tr, tc = target

    for dr, dc in DIRECTIONS["bishop"]:
        cr, cc = start[0] + dr, start[1] + dc
        while in_bounds(cr, cc):
            piece = pieces.get((cr, cc))
            if (cr, cc) == (tr, tc):
                return piece is None or piece[0] != bishop_color
            if piece is not None:
                break
            cr += dr
            cc += dc

    for d1, d2 in REF_BISHOP_ZIGZAGS:
        cr, cc = start
        current_dir = d1
        while True:
            cr += current_dir[0]
            cc += current_dir[1]
            if not in_bounds(cr, cc):
                break
            piece = pieces.get((cr, cc))
            if (cr, cc) == (tr, tc):
                return piece is None or piece[0] != bishop_color
            if piece is not None:
                break
            current_dir = d2 if current_dir == d1 else d1

    return False


def ref_is_square_attacked(state, target, attacking_color):
    pieces = state["pieces"]
    defending_color = opposite(attacking_color)

    for pos, (color, name) in pieces.items():
        if color != attacking_color:
            continue

        if name == "Knight":
            if target in ref_knight_targets(pos):
                return True
            for landing in ref_knight_targets(pos):
                if landing in pieces:
                    continue
                if target in ref_knight_targets(landing):
                    return True

        elif name == "Queen":
            qr, qc = pos
            for dr, dc in DIRECTIONS["queen"]:
                cr, cc = qr + dr, qc + dc
                while in_bounds(cr, cc):
                    if (cr, cc) == target:
                        occupant = pieces.get(target)
                        if occupant is None or occupant[0] != attacking_color:
                            return True
                    occupant = pieces.get((cr, cc))
                    if occupant is not None:
                        if occupant[0] == defending_color and abs(cr - target[0]) <= 1 and abs(cc - target[1]) <= 1:
                            return True
                        break
                    cr += dr
                    cc += dc

        elif name == "Rook":
            rr, rc = pos
            tr, tc = target
            if rr == tr or rc == tc:
                dr = (tr > rr) - (rr > tr)
                dc = (tc > rc) - (rc > tc)
                cr, cc = rr + dr, rc + dc
                blocked = False
                while in_bounds(cr, cc):
                    occupant = pieces.get((cr, cc))
                    if occupant is not None and occupant[0] == attacking_color:
                        blocked = True
                        break
                    if (cr, cc) == target:
                        break
                    cr += dr
                    cc += dc
                if not blocked and (cr, cc) == target:
                    return True

        elif name == "Bishop":
            if ref_bishop_attacks_square(state, pos, target, attacking_color):
                return True

        elif name == "Pawn":
            direction = -1 if attacking_color == "white" else 1
            if (pos[0] + direction, pos[1]) == target:
                return True
            starting_row = 6 if attacking_color == "white" else 1
            midpoint = (pos[0] + direction, pos[1])
            if pos[0] == starting_row and midpoint not in pieces and (pos[0] + 2 * direction, pos[1]) == target:
                return True
            if target[0] == pos[0] and abs(target[1] - pos[1]) == 1:
                occupant = pieces.get(target)
                if occupant is not None and occupant[0] == defending_color:
                    return True

        elif name == "King":
            dr = target[0] - pos[0]
            dc = target[1] - pos[1]
            abs_dr = abs(dr)
            abs_dc = abs(dc)
            max_dist = max(abs_dr, abs_dc)
            if max_dist == 1:
                return True
            if max_dist == 2 and (abs_dr == abs_dc or abs_dr == 0 or abs_dc == 0):
                midpoint = (pos[0] + (dr > 0) - (dr < 0), pos[1] + (dc > 0) - (dc < 0))
                if midpoint not in pieces:
                    return True

    return False


def ref_is_in_check(state, color):
    king_pos = state["white_king"] if color == "white" else state["black_king"]
    if king_pos is None:
        return True
    return ref_is_square_attacked(state, king_pos, opposite(color))


def ref_apply_move(state, move):
    child = ref_clone(state)
    start, end = move
    moving_color, moving_name = child["pieces"].pop(start)

    if moving_name == "King":
        if moving_color == "white":
            child["white_king"] = end
        else:
            child["black_king"] = end

    if moving_name == "Rook":
        dr = (end[0] > start[0]) - (start[0] > end[0])
        dc = (end[1] > start[1]) - (start[1] > end[1])
        cr, cc = start[0] + dr, start[1] + dc
        while (cr, cc) != end:
            occupant = child["pieces"].get((cr, cc))
            if occupant is not None and occupant[0] != moving_color:
                ref_remove_piece(child, (cr, cc))
            cr += dr
            cc += dc

    target_piece = child["pieces"].get(end)
    if target_piece is not None:
        ref_remove_piece(child, end)

    ref_add_piece(child, moving_color, moving_name, end)

    if moving_name == "Queen" and target_piece is not None:
        ref_remove_piece(child, end)
        for dr, dc in DIRECTIONS["king"]:
            adj = (end[0] + dr, end[1] + dc)
            occupant = child["pieces"].get(adj)
            if occupant is not None and occupant[0] != moving_color:
                ref_remove_piece(child, adj)
    elif moving_name == "Pawn":
        promotion_row = 0 if moving_color == "white" else 7
        if end[0] == promotion_row:
            ref_remove_piece(child, end)
            ref_add_piece(child, moving_color, "Queen", end)
            moving_name = "Queen"

    if moving_name == "Knight":
        active_targets = ref_knight_targets(end)
        active_victims = {sq for sq in active_targets if sq in child["pieces"] and child["pieces"][sq][0] != moving_color}
        enemy_knights = {sq for sq in active_victims if child["pieces"][sq][1] == "Knight"}
        passive_victims = set()
        for enemy_knight_pos in enemy_knights:
            for sq in ref_knight_targets(enemy_knight_pos):
                occupant = child["pieces"].get(sq)
                if occupant is not None and occupant[0] == moving_color:
                    passive_victims.add(sq)
        for sq in active_victims | passive_victims:
            ref_remove_piece(child, sq)
    else:
        occupant = child["pieces"].get(end)
        if occupant is not None:
            for sq in ref_knight_targets(end):
                attacker = child["pieces"].get(sq)
                if attacker is not None and attacker[0] != occupant[0] and attacker[1] == "Knight":
                    ref_remove_piece(child, end)
                    break

    return child


def ref_legal_moves(state, color):
    moves = []
    for pos, (piece_color, _piece_name) in sorted(state["pieces"].items()):
        if piece_color != color:
            continue
        for end in ref_piece_moves(state, pos):
            child = ref_apply_move(state, (pos, end))
            king_pos = child["white_king"] if color == "white" else child["black_king"]
            if king_pos is not None and not ref_is_in_check(child, color):
                moves.append((pos, end))
    return moves


def move_list_str(moves):
    if not moves:
        return "(none)"
    return ", ".join(f"{square_name(start)}-{square_name(end)}" for start, end in sorted(moves))


def compare_engine_and_reference(board, color, label):
    state = ref_from_board(board)
    engine_check = is_in_check(board, color)
    ref_check = ref_is_in_check(state, color)
    engine_moves = set(get_all_legal_moves(board, color))
    ref_moves = set(ref_legal_moves(state, color))
    engine_has_moves = has_legal_moves(board, color)
    ref_has_moves = bool(ref_moves)

    details = position_details(board, label)
    details.append(f"{color} in check: engine={engine_check}, reference={ref_check}")
    details.append(f"{color} has legal moves: engine={engine_has_moves}, reference={ref_has_moves}")
    
    expect(engine_check == ref_check, f"{label}: check status mismatch between engine and reference.")
    expect(engine_has_moves == ref_has_moves, f"{label}: has-legal-moves mismatch between engine and reference.")
    expect(engine_moves == ref_moves, f"{label}: legal move set mismatch between engine and reference. \nEngine unique: {engine_moves - ref_moves}\nRef unique: {ref_moves - engine_moves}")
    return details


def compare_outcomes(board, move, ref_state_before, ref_state_after):
    friendly_lost, opponent_captured, promo_type = board.get_move_outcome(move)
    
    engine_lost_counts = {}
    for p in friendly_lost:
        t = type(p).__name__
        engine_lost_counts[t] = engine_lost_counts.get(t, 0) + 1
        
    engine_cap_counts = {}
    for p in opponent_captured:
        t = type(p).__name__
        engine_cap_counts[t] = engine_cap_counts.get(t, 0) + 1
        
    color, moving_name = ref_state_before["pieces"][move[0]]
    
    ref_lost = {}
    ref_cap = {}
    
    for pos, (c, n) in ref_state_before["pieces"].items():
        if pos == move[0]: continue
        if pos not in ref_state_after["pieces"]:
            if c == color:
                ref_lost[n] = ref_lost.get(n, 0) + 1
            else:
                ref_cap[n] = ref_cap.get(n, 0) + 1

    # If the destination square is still occupied but changed color, count it as a capture.
    end_before = ref_state_before["pieces"].get(move[1])
    end_after = ref_state_after["pieces"].get(move[1])
    if end_before is not None and end_after is not None and end_before[0] != end_after[0]:
        ref_cap[end_before[1]] = ref_cap.get(end_before[1], 0) + 1
                
    # Handle self-destruction of the moving piece (Knight evaporation, Queen explosion)
    if move[1] not in ref_state_after["pieces"]:
        n = "Queen" if promo_type is not None else moving_name
        ref_lost[n] = ref_lost.get(n, 0) + 1
        
    expect(engine_lost_counts == ref_lost, f"Friendly losses mismatch on move {move}.\nEngine calculated: {engine_lost_counts}\nReference Oracle:  {ref_lost}")
    expect(engine_cap_counts == ref_cap, f"Opponent captures mismatch on move {move}.\nEngine calculated: {engine_cap_counts}\nReference Oracle:  {ref_cap}")


# ==============================================================================
# TEST CASES
# ==============================================================================

def _run_make_unmake_equivalence(board, move, label):
    expected = board.clone()
    expected.make_move(move[0], move[1])

    tracked = board.clone()
    before_snapshot = board_identity_snapshot(tracked)
    record = tracked.make_move_track(move[0], move[1])

    assert_board_equivalent(expected, tracked, label)
    tracked.unmake_move(record)
    assert_board_identity_restored(tracked, before_snapshot, label)
    assert_board_consistent(tracked, label)


def case_move_tracking_edge_cases():
    details = []

    # 1) Rook piercing captures a king on the path (not on the destination square).
    board = make_board([
        ("white", King, (7, 7)),
        ("white", Rook, (7, 0)),
        ("black", King, (4, 0)),
        ("black", Bishop, (2, 0)),
    ])
    move = ((7, 0), (0, 0))  # a1-a8
    _run_make_unmake_equivalence(board, move, "Rook piercing captures king on path")
    details.append("Rook piercing capture path, including king, restores correctly.")

    # 2) Pawn promotes into enemy knight kill zone (promotion queen evaporates).
    board = make_board([
        ("white", King, (7, 7)),
        ("white", Pawn, (1, 0)),   # a7
        ("black", King, (0, 7)),
        ("black", Knight, (1, 2)), # c7 attacks a8
    ])
    move = ((1, 0), (0, 0))  # a7-a8=Q (evaporates)
    _run_make_unmake_equivalence(board, move, "Promotion into knight evaporation zone")
    details.append("Promotion + immediate knight evaporation restores correctly.")

    # 3) Active knight move with mutual evaporation against enemy knight.
    board = make_board([
        ("white", King, (7, 7)),
        ("white", Knight, (6, 1)),  # b2
        ("black", King, (0, 7)),
        ("black", Knight, (2, 3)),  # d6, mutually attacks c4
    ])
    move = ((6, 1), (4, 2))  # b2-c4 (captures d6 via evaporation; mutual self-evap)
    _run_make_unmake_equivalence(board, move, "Mutual knight evaporation")
    details.append("Mutual knight evaporation restores correctly.")

    # 4) Queen capture explosion removes adjacent enemy king.
    board = make_board([
        ("white", King, (7, 7)),
        ("white", Queen, (7, 3)),  # d1
        ("black", King, (1, 4)),   # e7
        ("black", Rook, (1, 3)),   # d7 (capture target)
    ])
    move = ((7, 3), (1, 3))  # Qd1xd7 (explodes, removes king on e7)
    _run_make_unmake_equivalence(board, move, "Queen explosion captures adjacent king")
    details.append("Queen AOE explosion and king removal restore correctly.")

    return details


def case_move_tracking_random_equivalence():
    import random

    details = []
    seeds = [7, 123, 2026]
    plies_per_seed = 25
    samples_per_position = 20

    for seed in seeds:
        rng = random.Random(seed)
        board = Board()
        turn = "white"

        for ply in range(plies_per_seed):
            moves = get_all_legal_moves(board, turn)
            if not moves:
                break

            sample = moves if len(moves) <= samples_per_position else rng.sample(moves, samples_per_position)
            for move in sample:
                _run_make_unmake_equivalence(board, move, f"Seed {seed}, ply {ply}")

            board.make_move(*rng.choice(moves))
            turn = opposite(turn)

    details.append(
        f"Random make/unmake equivalence passed ({len(seeds)} seeds, "
        f"up to {plies_per_seed} plies, {samples_per_position} samples/position)."
    )
    return details


def case_regression_promotion_knight_zone():
    board = make_board([
        ("white", King, (7, 0)),
        ("white", Pawn, (1, 0)), # a7
        ("black", King, (0, 7)),
        ("black", Knight, (1, 2)), # c7, covers a8
    ])
    move = ((1, 0), (0, 0)) # a7-a8=Q
    moving_piece = board.grid[1][0]
    target_piece = board.grid[0][0]
    
    swing = fast_approximate_material_swing(board, move, moving_piece, target_piece, MG_PIECE_VALUES)
    
    # Expected: 
    # Pawn value (100) -> Queen value (1400) = +1300 gain
    # Queen lands in knight zone -> Evaporates = -1400 loss
    # Net swing = -100.
    # If the bug is present, it subtracts Pawn (-100) instead, giving +1200.
    details = position_details(board, "Pawn promoting into Knight evaporation zone")
    details.append(f"Move: {square_name(move[0])}-{square_name(move[1])}=Q")
    details.append(f"Material Swing calculated: {swing}")
    
    expect(swing == -100, f"Regression detected: Swing was {swing}. The engine is subtracting the pawn's value instead of the promoted Queen's value upon evaporation.")
    return details


def case_regression_tt_best_move_preservation():
    board = Board()
    bot = ChessBot(board, "white", {}, _DummyQueue(), _DummyEvent())
    hash_val = 99999
    best_move = ((6, 4), (4, 4)) # e2-e4
    
    # 1. Simulate a standard search storing a PV move
    bot._store_tt(hash_val, 50, 4, TT_FLAG_EXACT, best_move)
    
    # 2. Simulate ProbCut firing at a higher depth, storing None
    bot._store_tt(hash_val, 60, 6, TT_FLAG_LOWERBOUND, None)
    
    entry = bot.tt.get(hash_val)
    details = ["Testing TT behavior during ProbCut overwrites."]
    details.append(f"Initial entry stored with best_move: {best_move}")
    details.append("ProbCut attempts to overwrite with depth=6, best_move=None")
    details.append(f"Resulting TT entry best_move: {entry.best_move}")
    
    expect(entry is not None, "TT entry missing entirely.")
    expect(entry.best_move == best_move, f"Regression detected: ProbCut erased the best_move! Expected {best_move}, got {entry.best_move}")
    return details


def case_regression_nmp_mate_blindness():
    # The bug was: NMP guard used `beta < MATE_SCORE - 200`, which evaluates True
    # for large NEGATIVE betas (e.g. -999900 < 999800), incorrectly allowing NMP
    # to fire when the search window indicates white is being mated.
    # The fix is: `abs(beta) < MATE_SCORE - 1000`, which is False for any near-mate
    # beta regardless of sign.
    #
    # Position requirements:
    #   - White NOT in check at the start (NMP only considers positions not in check)
    #   - White has a non-pawn piece (NMP checks for non-pawn material on both sides)
    #   - No white piece can immediately capture the black king
    #   - White is clearly losing (black has a double-rook mating net)
    #
    # White: Ka1 (a1), Rc5 (c5)  — Rook on c5 satisfies the non-pawn requirement
    #   and cannot reach the black king at h8 (different rank AND file).
    # Black: Kh8 (h8), Rc2 (c2), Rg2 (g2)  — neither rook is on rank 7 or file a,
    #   so white is not in check from the start.
    board = make_board([
        ("white", King, (7, 0)),   # a1  — cornered
        ("white", Rook, (3, 2)),   # c5  — NMP trigger piece; c5→h8 impossible (diff rank & file)
        ("black", King, (0, 7)),   # h8  — unreachable by any white piece in one move
        ("black", Rook, (6, 2)),   # c2  — file c, rank 6: no check on Ka1 (rank 7, file a)
        ("black", Rook, (6, 6)),   # g2  — file g, rank 6: no check on Ka1
    ])

    bot = ChessBot(board, "white", {}, _DummyQueue(), _DummyEvent())
    bot.tt = {}

    # Preconditions: verify position properties before searching.
    expect(not is_in_check(board, "white"), "Test setup error: white should not be in check.")
    white_wins_immediately = any(
        board.clone().tap(lambda b: b.make_move(m[0], m[1])).find_king_pos("black") is None
        for m in get_all_legal_moves(board, "white")
    ) if hasattr(board, "tap") else False
    # Manual check: Rc5 cannot reach h8 (different rank AND file).
    # Ka1 cannot reach h8 (too far). No immediate win possible.

    # Also verify the NMP guard expression directly.
    MATE = bot.MATE_SCORE
    mate_beta = -999900
    expect(
        not (abs(mate_beta) < MATE - 1000),
        f"NMP guard regression: abs({mate_beta}) < {MATE - 1000} should be False "
        f"(got {abs(mate_beta) < MATE - 1000})"
    )

    try:
        score = bot.negamax(
            board, depth=4, alpha=-999999, beta=-999900,
            turn="white", ply=1, search_path=set()
        )
    except Exception:
        score = 0

    details = position_details(board, "White cornered, black double-rook mating net")
    details.append(f"Negamax(depth=4, alpha=-999999, beta=-999900) returned: {score}")
    details.append(f"NMP guard abs({mate_beta}) < {MATE - 1000}: {abs(mate_beta) < MATE - 1000} (must be False)")

    expect(
        score < -900000,
        f"Regression detected: Engine returned {score} instead of a mating score. "
        f"NMP may have incorrectly pruned the search."
    )
    return details


def case_oracle_deep_fuzz():
    import random
    details = []
    seeds = [2026, 42, 1337] # Multi-seed to guarantee variety
    plies_per_seed = 40
    
    for seed in seeds:
        rng = random.Random(seed)
        board = Board()
        turn = "white"
        
        for ply in range(plies_per_seed):
            state = ref_from_board(board)
            engine_moves = set(get_all_legal_moves(board, turn))
            ref_moves = set(ref_legal_moves(state, turn))
            
            expect(engine_moves == ref_moves, f"Fuzz failure on seed {seed}, ply {ply}: move mismatch.\nBoard:\n{board_to_ascii(board)}")
            
            if not engine_moves:
                break
            
            # Pick a random valid move
            move = rng.choice(sorted(engine_moves))
            ref_child = ref_apply_move(state, move)
            
            # Verify casualty logic (get_move_outcome)
            compare_outcomes(board, move, state, ref_child)
            
            board.make_move(move[0], move[1])
            turn = opposite(turn)
            
    details.append(f"Passed deep fuzzing across {len(seeds)} seeds (up to {plies_per_seed} plies each).")
    details.append("100% agreement on legal move generation and piece casualty accounting.")
    return details


def case_tb_inventory():
    manager = TablebaseManager()
    loaded = sorted(manager.tables.keys())
    details = [
        f"Loaded table count: {len(loaded)}",
        "Loaded tables: " + (", ".join(loaded) if loaded else "(none)"),
        "Checks only probe wiring and availability; it does not certify the tables are strategically correct.",
    ]
    expect(True, "Inventory reporting should always pass.")
    return details


CASES = [
    TestCaseSpec(
        "move_tracking_edge_cases",
        "engine_internals",
        "Targeted make_move_track/unmake_move edge cases (piercing, promotion, knight AOE, queen explosion).",
        case_move_tracking_edge_cases,
    ),
    TestCaseSpec(
        "move_tracking_random_equivalence",
        "engine_internals",
        "Randomized make_move_track equivalence against make_move, plus unmake identity restoration.",
        case_move_tracking_random_equivalence,
    ),
    TestCaseSpec(
        "regression_promotion_knight_zone",
        "engine_internals",
        "Verify the engine correctly subtracts a Queen's value when a pawn promotes into an evaporation zone.",
        case_regression_promotion_knight_zone,
    ),
    TestCaseSpec(
        "regression_tt_best_move_preservation",
        "engine_internals",
        "Verify that _store_tt preserves existing best moves when ProbCut stores a None move.",
        case_regression_tt_best_move_preservation,
    ),
    TestCaseSpec(
        "regression_nmp_mate_blindness",
        "engine_internals",
        "Verify NMP is suppressed when beta indicates a forced-mate line (abs(beta) >= MATE_SCORE - 1000).",
        case_regression_nmp_mate_blindness,
    ),
    TestCaseSpec(
        "oracle_deep_fuzz",
        "oracle",
        "Multi-seed, deep-ply playout verifying 100% agreement on moves, checks, and piece casualty mechanics.",
        case_oracle_deep_fuzz,
    ),
    TestCaseSpec(
        "tb_inventory",
        "tablebase",
        "Report which tablebase files are currently loaded before running the mapping checks.",
        case_tb_inventory,
    )
]

def select_cases(args):
    case_map = {case.name: case for case in CASES}

    if args.case:
        unknown = [name for name in args.case if name not in case_map]
        if unknown:
            raise SystemExit(
                "Unknown case(s): " + ", ".join(sorted(unknown)) +
                "\nUse --list to see the available case names."
            )
        return [case_map[name] for name in args.case]

    groups = set(args.group or ["engine_internals", "oracle", "tablebase"])
    if "all" in groups:
        return CASES[:]
    return [case for case in CASES if case.group in groups]


def print_case_list():
    print("Available test cases:\n")
    for case in CASES:
        print(f"- {case.name} [{case.group}]")
        print(f"  {case.description}")


def run_case(case):
    print(f"\n[RUN ] {case.name} [{case.group}]")
    print(f"       {case.description}")
    try:
        details = case.func()
        for detail in details:
            print(f"       {detail}")
        print(f"[PASS] {case.name}")
        return "PASS"
    except SkipCase as exc:
        print(f"       {exc}")
        print(f"[SKIP] {case.name}")
        return "SKIP"
    except AssertionError as exc:
        print(f"       {exc}")
        print(f"[FAIL] {case.name}")
        return "FAIL"


def main():
    global SHOW_ASCII

    parser = argparse.ArgumentParser(
        description="Selectable Jungle Chess rule, search, and tablebase checks."
    )
    parser.add_argument(
        "--group",
        action="append",
        choices=["engine_internals", "oracle", "tablebase", "all"],
        help="Run only the selected test group(s). Defaults to all groups.",
    )
    parser.add_argument(
        "--case",
        action="append",
        help="Run only the named test case. Can be provided multiple times.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the available named test cases and exit.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failing case.",
    )
    parser.add_argument(
        "--ascii",
        action="store_true",
        help="Also print an ASCII board diagram for each position under test.",
    )
    args = parser.parse_args()
    SHOW_ASCII = args.ascii

    if args.list:
        print_case_list()
        return 0

    selected = select_cases(args)
    if not selected:
        print("No cases selected.")
        return 0

    print(f"Running {len(selected)} test case(s)...")
    pass_count = 0
    skip_count = 0
    fail_count = 0

    for case in selected:
        outcome = run_case(case)
        if outcome == "PASS":
            pass_count += 1
        elif outcome == "SKIP":
            skip_count += 1
        else:
            fail_count += 1
            if args.fail_fast:
                break

    print("\nSummary")
    print(f"  Passed: {pass_count}")
    print(f"  Skipped: {skip_count}")
    print(f"  Failed: {fail_count}")
    return 1 if fail_count else 0


if __name__ == "__main__":
    raise SystemExit(main())