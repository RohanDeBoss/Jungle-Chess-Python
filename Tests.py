# Tests.py (v2.1 - Oracle asessment, selectable Jungle rule, search, and tablebase checks)

import argparse
from dataclasses import dataclass

from GameLogic import (
    Board,
    Bishop,
    DIRECTIONS,
    King,
    Knight,
    Pawn,
    Queen,
    Rook,
    format_move_san,
    get_all_legal_moves,
    has_legal_moves,
    is_in_check,
    is_passive_knight_zone_evaporation,
)
from OpponentAI import OpponentAI
from TablebaseManager import TablebaseManager


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
    details.append(f"Engine legal moves ({len(engine_moves)}): {move_list_str(engine_moves)}")
    details.append(f"Reference legal moves ({len(ref_moves)}): {move_list_str(ref_moves)}")

    expect(engine_check == ref_check, f"{label}: check status mismatch between engine and reference.")
    expect(engine_has_moves == ref_has_moves, f"{label}: has-legal-moves mismatch between engine and reference.")
    expect(engine_moves == ref_moves, f"{label}: legal move set mismatch between engine and reference.")
    return details


def case_mutual_knight_resolution():
    board = make_board([
        ("white", King, (7, 0)),    # a1
        ("white", Knight, (6, 1)),  # b2
        ("white", Pawn, (6, 4)),    # e2
        ("black", King, (0, 7)),    # h8
        ("black", Knight, (4, 5)),  # f4
        ("black", Pawn, (3, 4)),    # e5
    ])
    move = ((6, 1), (5, 3))         # Nb2-d3

    details = position_details(board, "Start")
    board.make_move(*move)
    details.extend(position_details(board, "After Nb2-d3"))

    expect(board.grid[6][4] is None, "Expected e2 to evaporate from the passive enemy knight.")
    expect(board.grid[5][3] is None, "Expected the moving knight on d3 to self-evaporate.")
    expect(board.grid[4][5] is None, "Expected the enemy knight on f4 to be removed.")
    expect(board.grid[3][4] is None, "Expected e5 to die from the active knight evaporation.")
    details.append("Verified: e2, d3, f4, and e5 are all empty after the mutual-knight cascade.")
    return details


def case_mutual_knight_outcome_and_san():
    board = make_board([
        ("white", King, (7, 0)),
        ("white", Knight, (6, 1)),
        ("white", Pawn, (6, 4)),
        ("black", King, (0, 7)),
        ("black", Knight, (4, 5)),
        ("black", Pawn, (3, 4)),
    ])
    move = ((6, 1), (5, 3))

    friendly_lost, opponent_captured, promotion_type = board.get_move_outcome(move)
    child = board.clone()
    child.make_move(*move)
    san = format_move_san(board, child, move)

    details = position_details(board, "Start")
    details.append(f"Friendly losses: {sorted(type(piece).__name__ for piece in friendly_lost)}")
    details.append(f"Opponent captures: {sorted(type(piece).__name__ for piece in opponent_captured)}")
    details.append(f"SAN: {san}")

    expect(sorted(type(piece).__name__ for piece in friendly_lost) == ["Knight", "Pawn"],
           "Expected the moving knight and the pawn on e2 to be counted as friendly losses.")
    expect(sorted(type(piece).__name__ for piece in opponent_captured) == ["Knight", "Pawn"],
           "Expected the black knight on f4 and pawn on e5 to be counted as captures.")
    expect(promotion_type is None, "Did not expect a promotion in the knight cascade case.")
    expect(san == "Nd3 (xe2 xd3 xf4 xe5)",
           "Expected SAN to list the passive and active casualties in deterministic order.")
    return details


def case_opai_trap_mate():
    board = make_board([
        ("white", King, (0, 2)),    # c8
        ("white", Rook, (3, 1)),    # b5
        ("white", Pawn, (0, 0)),    # a8
        ("black", King, (1, 0)),    # a7
        ("black", Pawn, (7, 0)),    # a1
    ])
    bot = make_opponent_ai(board, "black")
    score = bot.qsearch(board, -10**9, 10**9, "black", 0)

    details = position_details(board, "Trap position")
    details.append(f"is_in_check(black)={is_in_check(board, 'black')}")
    details.append(f"has_legal_moves(black)={has_legal_moves(board, 'black')}")
    details.append(f"OpponentAI.qsearch score={score}")

    expect(not is_in_check(board, "black"),
           "This position should be a no-check Jungle trap, not a standard in-check mate.")
    expect(not has_legal_moves(board, "black"),
           "Black should have zero legal moves in this trap position.")
    expect(score == -bot.MATE_SCORE,
           "OpponentAI qsearch should score no-legal-move trap positions as immediate mate.")
    return details


def case_opai_passive_knight_tactical():
    board = make_board([
        ("white", King, (7, 0)),    # a1
        ("white", Rook, (6, 0)),    # a2
        ("black", King, (0, 7)),    # h8
        ("black", Knight, (7, 2)),  # c1
    ])
    move = ((6, 0), (6, 4))         # Ra2-e2 into the knight's passive zone
    bot = make_opponent_ai(board, "white")
    is_passive = is_passive_knight_zone_evaporation(board, move)
    is_tactical = bot._is_tactical_move(board, move)

    details = position_details(board, "Passive knight-zone test")
    details.append(f"Move under test: {square_name(move[0])}-{square_name(move[1])}")
    details.append(f"is_passive_knight_zone_evaporation={is_passive}")
    details.append(f"OpponentAI._is_tactical_move={is_tactical}")

    expect(is_passive, "Expected Ra2-e2 to be recognized as a passive knight-zone evaporation.")
    expect(is_tactical, "OpponentAI should classify passive knight-zone evaporations as tactical.")
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


def case_tb_white_3man_lookup():
    manager = manager_with_table("K_Queen_K")
    board = make_board([
        ("white", King, (7, 4)),    # e1
        ("white", Queen, (6, 3)),   # d2
        ("black", King, (0, 0)),    # a8
    ])
    wk = board.white_king_pos
    wq = next(piece for piece in board.white_pieces if type(piece) is Queen).pos
    bk = board.black_king_pos

    details = position_details(board, "3-man white probe wiring")
    for turn, t_idx in (("white", 0), ("black", 1)):
        raw = int(manager.tables["K_Queen_K"][wk[0] * 8 + wk[1], wq[0] * 8 + wq[1], bk[0] * 8 + bk[1], t_idx])
        expected = manager._tb_score_to_ai_score(raw, "white")
        actual = manager.probe(board, turn)
        details.append(f"{turn} to move: raw={raw}, expected_probe={expected}, actual_probe={actual}")
        expect(actual == expected,
               f"Tablebase probe mismatch for K_Queen_K with {turn} to move.")
    return details


def case_tb_black_3man_lookup():
    manager = manager_with_table("K_Rook_K")
    board = make_board([
        ("white", King, (7, 0)),    # a1
        ("black", King, (0, 4)),    # e8
        ("black", Rook, (1, 3)),    # d7
    ])
    bk = board.black_king_pos
    br = next(piece for piece in board.black_pieces if type(piece) is Rook).pos
    wk = board.white_king_pos

    def flip(pos):
        return (7 - pos[0], pos[1])

    details = position_details(board, "3-man black probe wiring")
    for turn, t_idx in (("black", 0), ("white", 1)):
        idx = (
            flip(bk)[0] * 8 + flip(bk)[1],
            flip(br)[0] * 8 + flip(br)[1],
            flip(wk)[0] * 8 + flip(wk)[1],
            t_idx,
        )
        raw = int(manager.tables["K_Rook_K"][idx])
        expected = manager._tb_score_to_ai_score(raw, "black")
        actual = manager.probe(board, turn)
        details.append(f"{turn} to move: raw={raw}, expected_probe={expected}, actual_probe={actual}")
        expect(actual == expected,
               f"Tablebase probe mismatch for black-attacker K_Rook_K with {turn} to move.")
    return details


def case_tb_white_4man_lookup():
    manager = manager_with_table("K_Queen_Rook_K")
    board = make_board([
        ("white", King, (7, 4)),    # e1
        ("white", Queen, (6, 3)),   # d2
        ("white", Rook, (5, 2)),    # c3
        ("black", King, (0, 0)),    # a8
    ])
    w_pieces = [piece for piece in board.white_pieces if not isinstance(piece, King)]
    w_pieces.sort(key=lambda piece: type(piece).__name__)
    wk = board.white_king_pos
    bk = board.black_king_pos

    details = position_details(board, "4-man same-side probe wiring")
    for turn, t_idx in (("white", 0), ("black", 1)):
        idx = manager._flat_idx_raw_4(
            wk[0] * 8 + wk[1],
            w_pieces[0].pos[0] * 8 + w_pieces[0].pos[1],
            w_pieces[1].pos[0] * 8 + w_pieces[1].pos[1],
            bk[0] * 8 + bk[1],
            t_idx,
        )
        raw = int(manager.tables["K_Queen_Rook_K"].flat[idx])
        expected = manager._tb_score_to_ai_score(raw, "white")
        actual = manager.probe(board, turn)
        details.append(f"{turn} to move: raw={raw}, expected_probe={expected}, actual_probe={actual}")
        expect(actual == expected,
               f"Tablebase probe mismatch for K_Queen_Rook_K with {turn} to move.")
    return details


def case_tb_missing_cross_returns_none():
    manager = TablebaseManager()
    pieces = ["Queen", "Rook", "Knight", "Bishop", "Pawn"]
    missing_pair = None
    for white_name in pieces:
        for black_name in pieces:
            if f"K_{white_name}_vs_{black_name}_K" not in manager.tables:
                missing_pair = (white_name, black_name)
                break
        if missing_pair is not None:
            break

    if missing_pair is None:
        raise SkipCase("No missing cross table found; every K_X_vs_Y_K table is present.")

    white_name, black_name = missing_pair
    board = make_board([
        ("white", King, (7, 0)),
        ("white", PIECE_CLASS_BY_NAME[white_name], (6, 1)),
        ("black", King, (0, 7)),
        ("black", PIECE_CLASS_BY_NAME[black_name], (1, 6)),
    ])
    actual = manager.probe(board, "white")

    details = position_details(board, "Missing cross-table fallback")
    details.append(f"Missing table: K_{white_name}_vs_{black_name}_K")
    details.append(f"Probe result: {actual}")

    expect(actual is None,
           "Expected TablebaseManager.probe to return None when the required cross table is missing.")
    return details


def case_oracle_curated_positions():
    positions = [
        (
            "Queen proxy explosion check",
            "black",
            make_board([
                ("white", King, (7, 7)),    # h1
                ("white", Queen, (4, 7)),   # h4
                ("black", King, (6, 4)),    # e2
                ("black", Pawn, (6, 5)),    # f2
            ]),
        ),
        (
            "Rook railgun check through enemy screen",
            "black",
            make_board([
                ("white", King, (7, 7)),    # h1
                ("white", Rook, (3, 7)),    # h5
                ("black", King, (3, 0)),    # a5
                ("black", Pawn, (3, 4)),    # e5
            ]),
        ),
        (
            "Bishop zig-zag mobility",
            "white",
            make_board([
                ("white", King, (7, 0)),    # a1
                ("white", Bishop, (4, 3)),  # d4
                ("black", King, (0, 7)),    # h8
                ("black", Pawn, (2, 4)),    # e6
                ("black", Pawn, (5, 4)),    # e3
            ]),
        ),
        (
            "King two-step through attacked midpoint",
            "white",
            make_board([
                ("white", King, (7, 4)),    # e1
                ("black", King, (0, 7)),    # h8
                ("black", Rook, (0, 5)),    # f8 attacks f-file midpoint
            ]),
        ),
    ]

    details = []
    for label, color, board in positions:
        details.extend(compare_engine_and_reference(board, color, label))

    e1_g1 = ((7, 4), (7, 6))
    king_board = positions[3][2]
    engine_moves = set(get_all_legal_moves(king_board, "white"))
    expect(e1_g1 in engine_moves,
           "Expected the white king to be allowed to move e1-g1 through an attacked midpoint if g1 is safe.")
    details.append("Verified: the curated oracle positions all matched the independent reference model.")
    return details


def case_oracle_playout_consistency():
    import random

    rng = random.Random(20260307)
    details = []
    board = Board()
    turn = "white"
    checked_positions = 0

    for ply in range(16):
        details.extend(compare_engine_and_reference(board, turn, f"Oracle playout ply {ply + 1}"))
        checked_positions += 1

        legal_moves = sorted(get_all_legal_moves(board, turn))
        if not legal_moves:
            break
        move = rng.choice(legal_moves)
        details.append(f"Chosen continuation: {square_name(move[0])}-{square_name(move[1])}")
        board.make_move(*move)
        turn = opposite(turn)

    details.append(f"Verified {checked_positions} consecutive reachable positions from the starting position.")
    return details


CASES = [
    TestCaseSpec(
        "mutual_knight_resolution",
        "rules",
        "Verify that mutual knight evaporation removes both knights and their secondary victims on the board.",
        case_mutual_knight_resolution,
    ),
    TestCaseSpec(
        "mutual_knight_outcome_san",
        "rules",
        "Verify move-outcome accounting and SAN output for the mutual-knight cascade case.",
        case_mutual_knight_outcome_and_san,
    ),
    TestCaseSpec(
        "opai_trap_mate",
        "search",
        "Verify that OpponentAI q-search scores a no-legal-moves trap as mate even when the side to move is not in check.",
        case_opai_trap_mate,
    ),
    TestCaseSpec(
        "opai_passive_knight_tactical",
        "search",
        "Verify that OpponentAI treats passive knight-zone evaporations as tactical moves.",
        case_opai_passive_knight_tactical,
    ),
    TestCaseSpec(
        "oracle_curated_positions",
        "oracle",
        "Cross-check the optimized engine against an independent reference model on curated Jungle-specific quirk positions.",
        case_oracle_curated_positions,
    ),
    TestCaseSpec(
        "oracle_playout_consistency",
        "oracle",
        "Cross-check the optimized engine against an independent reference model across a deterministic legal playout from the starting position.",
        case_oracle_playout_consistency,
    ),
    TestCaseSpec(
        "tb_inventory",
        "tablebase",
        "Report which tablebase files are currently loaded before running the mapping checks.",
        case_tb_inventory,
    ),
    TestCaseSpec(
        "tb_white_3man_lookup",
        "tablebase",
        "Verify that TablebaseManager.probe matches a direct K_Queen_K raw lookup for a white-attacker 3-man position.",
        case_tb_white_3man_lookup,
    ),
    TestCaseSpec(
        "tb_black_3man_lookup",
        "tablebase",
        "Verify that TablebaseManager.probe matches a direct K_Rook_K raw lookup for a black-attacker 3-man position.",
        case_tb_black_3man_lookup,
    ),
    TestCaseSpec(
        "tb_white_4man_lookup",
        "tablebase",
        "Verify that TablebaseManager.probe matches a direct K_Queen_Rook_K raw lookup for a 4-man same-side position.",
        case_tb_white_4man_lookup,
    ),
    TestCaseSpec(
        "tb_missing_cross_returns_none",
        "tablebase",
        "Verify that missing cross-table positions return None instead of a fake score.",
        case_tb_missing_cross_returns_none,
    ),
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

    groups = set(args.group or ["rules", "search", "oracle", "tablebase"])
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
        choices=["rules", "search", "oracle", "tablebase", "all"],
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
