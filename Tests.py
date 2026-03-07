# Tests.py (v2.0 - selectable Jungle rule, search, and tablebase checks)

import argparse
from dataclasses import dataclass

from GameLogic import (
    Board,
    Bishop,
    King,
    Knight,
    Pawn,
    Queen,
    Rook,
    format_move_san,
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

    groups = set(args.group or ["rules", "search", "tablebase"])
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
        choices=["rules", "search", "tablebase", "all"],
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
