import argparse
import json
import os
import re
from dataclasses import dataclass

import numpy as np

from GameLogic import Board, King, Queen, Rook, Bishop, Knight, Pawn, is_in_check
from TablebaseManager import TablebaseManager


OLD_SUFFIX = "_xsml.bin"
NEW_SUFFIX = "_tb16.bin"
OLD_LONGEST = "longest_mates_xsml.tsv"
NEW_LONGEST = "longest_mates_tb16.tsv"
PIECE_CLASSES = {
    "King": King,
    "Queen": Queen,
    "Rook": Rook,
    "Bishop": Bishop,
    "Knight": Knight,
    "Pawn": Pawn,
}
PIECE_CHARS = {
    "King": "K",
    "Queen": "Q",
    "Rook": "R",
    "Bishop": "B",
    "Knight": "N",
    "Pawn": "P",
}
PAWN_WK_SQUARES = [r * 8 + c for r in range(8) for c in range(4)]
NON_PAWN_WK_SQUARES = []
for _c in range(4):
    for _r in range(_c + 1):
        NON_PAWN_WK_SQUARES.append(_r * 8 + _c)


@dataclass
class TableMetadata:
    base_name: str
    category: str
    white_pieces: list[str]
    black_pieces: list[str]
    has_pawn: bool
    shape: tuple[int, ...]


class ProbeManager(TablebaseManager):
    def __init__(self, tables):
        self.tables = tables
        self.tb_dir = ""


def parse_longest_mates(path):
    if not os.path.exists(path):
        return {}
    records = {}
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or any(line.startswith(x) for x in ("#", "Table", "---")):
                continue
            parts = re.split(r"\s{2,}|\t", line)
            if len(parts) < 6:
                continue
            key = parts[0]
            if key.startswith("regen_"):
                key = key[len("regen_") :]
            records[key] = {
                "dtm": int(parts[1]),
                "decisive": int(parts[2].replace(",", "")),
                "drawn": int(parts[3].replace(",", "")),
                "minutes": float(parts[4]),
                "updated_utc": parts[5],
            }
    return records


def list_table_bases(directory, suffix):
    return {name[: -len(suffix)] for name in os.listdir(directory) if name.endswith(suffix)}


def parse_table_metadata(base_name):
    parts = base_name.split("_")
    if len(parts) < 3 or parts[0] != "K" or parts[-1] != "K":
        raise ValueError(f"Unrecognized table name: {base_name}")

    if "vs" in parts:
        vs_idx = parts.index("vs")
        white_pieces = parts[1:vs_idx]
        black_pieces = parts[vs_idx + 1 : -1]
        if len(white_pieces) == 1 and len(black_pieces) == 1:
            category = "4-Man Cross"
        elif len(white_pieces) == 2 and len(black_pieces) == 1:
            category = "5-Man Cross"
        else:
            raise ValueError(f"Unsupported cross table name: {base_name}")
    else:
        white_pieces = parts[1:-1]
        black_pieces = []
        if len(white_pieces) == 1:
            category = "3-Man"
        elif len(white_pieces) == 2:
            category = "4-Man Same-Side"
        elif len(white_pieces) == 3:
            category = "5-Man Same-Side"
        else:
            raise ValueError(f"Unsupported same-side table name: {base_name}")

    has_pawn = ("Pawn" in white_pieces) or ("Pawn" in black_pieces)
    wk_size = 32 if has_pawn else 10
    num_pieces = len(white_pieces) + len(black_pieces)
    shape = tuple([wk_size] + [64] * (num_pieces + 1) + [2])
    return TableMetadata(base_name, category, white_pieces, black_pieces, has_pawn, shape)


def load_table(directory, base_name, suffix, dtype, shape):
    filename = os.path.join(directory, f"{base_name}{suffix}")
    return np.memmap(filename, dtype=dtype, mode="r", shape=shape)


def decode_placements(flat, metadata):
    turn = flat % 2
    rest = flat // 2
    wk_squares = PAWN_WK_SQUARES if metadata.has_pawn else NON_PAWN_WK_SQUARES

    if metadata.category == "3-Man":
        bk = rest % 64
        p1 = (rest // 64) % 64
        wk_idx = rest // 4096
        wk = wk_squares[wk_idx]
        placements = [("K", wk), ("k", bk), (PIECE_CHARS[metadata.white_pieces[0]], p1)]
    elif metadata.category == "4-Man Same-Side":
        bk = rest % 64
        p2 = (rest // 64) % 64
        p1 = (rest // 4096) % 64
        wk_idx = rest // 262144
        wk = wk_squares[wk_idx]
        placements = [
            ("K", wk),
            ("k", bk),
            (PIECE_CHARS[metadata.white_pieces[0]], p1),
            (PIECE_CHARS[metadata.white_pieces[1]], p2),
        ]
    elif metadata.category == "4-Man Cross":
        bp = rest % 64
        bk = (rest // 64) % 64
        wp = (rest // 4096) % 64
        wk_idx = rest // 262144
        wk = wk_squares[wk_idx]
        placements = [
            ("K", wk),
            ("k", bk),
            (PIECE_CHARS[metadata.white_pieces[0]], wp),
            (PIECE_CHARS[metadata.black_pieces[0]].lower(), bp),
        ]
    elif metadata.category == "5-Man Same-Side":
        bk = rest % 64
        p3 = (rest // 64) % 64
        p2 = (rest // 4096) % 64
        p1 = (rest // 262144) % 64
        wk_idx = rest // 16777216
        wk = wk_squares[wk_idx]
        placements = [
            ("K", wk),
            ("k", bk),
            (PIECE_CHARS[metadata.white_pieces[0]], p1),
            (PIECE_CHARS[metadata.white_pieces[1]], p2),
            (PIECE_CHARS[metadata.white_pieces[2]], p3),
        ]
    elif metadata.category == "5-Man Cross":
        bp = rest % 64
        bk = (rest // 64) % 64
        wp2 = (rest // 4096) % 64
        wp1 = (rest // 262144) % 64
        wk_idx = rest // 16777216
        wk = wk_squares[wk_idx]
        placements = [
            ("K", wk),
            ("k", bk),
            (PIECE_CHARS[metadata.white_pieces[0]], wp1),
            (PIECE_CHARS[metadata.white_pieces[1]], wp2),
            (PIECE_CHARS[metadata.black_pieces[0]].lower(), bp),
        ]
    else:
        raise ValueError(f"Unsupported category: {metadata.category}")

    return placements, turn


def build_board(placements):
    board = Board(setup=False)
    for char, square in placements:
        piece_class = {
            "K": King,
            "Q": Queen,
            "R": Rook,
            "B": Bishop,
            "N": Knight,
            "P": Pawn,
        }[char.upper()]
        color = "white" if char.isupper() else "black"
        board.add_piece(piece_class(color), square // 8, square % 8)
    return board


def build_fen(placements, turn):
    board = [["" for _ in range(8)] for _ in range(8)]
    for char, pos in placements:
        board[pos // 8][pos % 8] = char

    rows = []
    for r in range(8):
        empty = 0
        row_str = ""
        for c in range(8):
            if board[r][c] == "":
                empty += 1
            else:
                if empty:
                    row_str += str(empty)
                    empty = 0
                row_str += board[r][c]
        if empty:
            row_str += str(empty)
        rows.append(row_str)
    return "/".join(rows) + f" {'w' if turn == 0 else 'b'} - - 0 1"


def positions_not_in_check(board):
    return (not is_in_check(board, "white")) and (not is_in_check(board, "black"))


def compare_longest_mates(old_records, new_records, common_bases):
    comparison = {
        "shared_note_entries": 0,
        "metadata_match_count": 0,
        "metadata_mismatches": [],
        "old_only_note_entries": sorted(set(old_records) - set(new_records)),
        "new_only_note_entries": sorted(set(new_records) - set(old_records)),
    }
    for base_name in common_bases:
        old = old_records.get(base_name)
        new = new_records.get(base_name)
        if old is None or new is None:
            continue
        comparison["shared_note_entries"] += 1
        if old["dtm"] == new["dtm"] and old["decisive"] == new["decisive"] and old["drawn"] == new["drawn"]:
            comparison["metadata_match_count"] += 1
        else:
            comparison["metadata_mismatches"].append(
                {
                    "table": base_name,
                    "old": old,
                    "new": new,
                }
            )
    return comparison


def table_actual_summary(arr):
    flat = np.asarray(arr).reshape(-1)
    abs_flat = np.abs(flat.astype(np.int32, copy=False))
    decisive = int(np.count_nonzero(flat))
    max_dtm = int(abs_flat.max()) if decisive else 0
    return {
        "max_abs_value": max_dtm,
        "decisive_entries": decisive,
        "zero_entries_total": int(flat.size - decisive),
    }


def full_compare_table(base_name, old_arr, new_arr):
    old_flat = np.asarray(old_arr).reshape(-1).astype(np.int32, copy=False)
    new_flat = np.asarray(new_arr).reshape(-1).astype(np.int32, copy=False)
    mismatch_mask = old_flat != new_flat
    mismatch_count = int(np.count_nonzero(mismatch_mask))
    summary = {
        "table": base_name,
        "state_count": int(old_flat.size),
        "identical": mismatch_count == 0,
        "mismatch_count": mismatch_count,
        "max_abs_diff": int(np.max(np.abs(old_flat - new_flat))) if mismatch_count else 0,
        "old_actual": table_actual_summary(old_flat),
        "new_actual": table_actual_summary(new_flat),
        "mismatch_examples": [],
    }
    if mismatch_count:
        mismatch_indices = np.flatnonzero(mismatch_mask)[:10]
        for idx in mismatch_indices:
            summary["mismatch_examples"].append(
                {
                    "flat_index": int(idx),
                    "old": int(old_flat[idx]),
                    "new": int(new_flat[idx]),
                }
            )
    return summary


def sample_probe_checks(base_name, metadata, old_arr, new_arr, sample_count, rng_seed):
    old_flat = np.asarray(old_arr).reshape(-1).astype(np.int32, copy=False)
    new_flat = np.asarray(new_arr).reshape(-1).astype(np.int32, copy=False)
    mismatch_indices = np.flatnonzero(old_flat != new_flat)
    candidate_indices = np.flatnonzero((old_flat != 0) | (new_flat != 0))

    old_mgr = ProbeManager({base_name: old_arr})
    new_mgr = ProbeManager({base_name: new_arr})

    chosen = []
    if len(mismatch_indices):
        chosen.extend(int(x) for x in mismatch_indices[:sample_count])

    if len(chosen) < sample_count and len(candidate_indices):
        rng = np.random.default_rng(rng_seed)
        pool_size = min(len(candidate_indices), max(sample_count * 20, sample_count))
        pool = rng.choice(candidate_indices, size=pool_size, replace=False)
        for idx in pool:
            if len(chosen) >= sample_count:
                break
            idx = int(idx)
            if idx not in chosen:
                chosen.append(idx)

    checks = []
    for idx in chosen:
        placements, turn = decode_placements(idx, metadata)
        board = build_board(placements)
        if not positions_not_in_check(board):
            continue
        turn_name = "white" if turn == 0 else "black"
        old_score = old_mgr.probe(board, turn_name)
        new_score = new_mgr.probe(board, turn_name)
        checks.append(
            {
                "flat_index": idx,
                "turn": turn_name,
                "fen": build_fen(placements, turn),
                "old_table_value": int(old_flat[idx]),
                "new_table_value": int(new_flat[idx]),
                "old_probe_score": None if old_score is None else int(old_score),
                "new_probe_score": None if new_score is None else int(new_score),
                "probe_match": old_score == new_score,
            }
        )
        if len(checks) >= sample_count:
            break
    return checks


def main():
    parser = argparse.ArgumentParser(
        description="Compare old and new Jungle Chess tablebases using note files, full value diffs, and live probe checks."
    )
    parser.add_argument("--old-dir", default="old tablebases")
    parser.add_argument("--new-dir", default="tablebases")
    parser.add_argument("--old-suffix", default=OLD_SUFFIX)
    parser.add_argument("--new-suffix", default=NEW_SUFFIX)
    parser.add_argument("--old-longest", default=OLD_LONGEST)
    parser.add_argument("--new-longest", default=NEW_LONGEST)
    parser.add_argument("--samples-per-table", type=int, default=12)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output", default="tablebase_equivalence_report.json")
    args = parser.parse_args()

    old_bases = list_table_bases(args.old_dir, args.old_suffix)
    new_bases = list_table_bases(args.new_dir, args.new_suffix)
    common_bases = sorted(old_bases & new_bases)

    old_longest = parse_longest_mates(os.path.join(args.old_dir, args.old_longest))
    new_longest = parse_longest_mates(os.path.join(args.new_dir, args.new_longest))

    report = {
        "old_dir": args.old_dir,
        "new_dir": args.new_dir,
        "old_suffix": args.old_suffix,
        "new_suffix": args.new_suffix,
        "common_table_count": len(common_bases),
        "common_tables": common_bases,
        "old_only_tables": sorted(old_bases - new_bases),
        "new_only_tables": sorted(new_bases - old_bases),
        "longest_mate_comparison": compare_longest_mates(old_longest, new_longest, common_bases),
        "table_comparisons": [],
    }

    for offset, base_name in enumerate(common_bases):
        metadata = parse_table_metadata(base_name)
        old_arr = load_table(args.old_dir, base_name, args.old_suffix, np.int8, metadata.shape)
        new_arr = load_table(args.new_dir, base_name, args.new_suffix, np.int16, metadata.shape)

        table_report = full_compare_table(base_name, old_arr, new_arr)
        table_report["metadata"] = {
            "category": metadata.category,
            "shape": list(metadata.shape),
            "has_pawn": metadata.has_pawn,
        }
        table_report["old_note"] = old_longest.get(base_name)
        table_report["new_note"] = new_longest.get(base_name)
        table_report["old_note_matches_actual"] = (
            table_report["old_note"] is not None
            and table_report["old_note"]["dtm"] == table_report["old_actual"]["max_abs_value"]
            and table_report["old_note"]["decisive"] == table_report["old_actual"]["decisive_entries"]
        )
        table_report["new_note_matches_actual"] = (
            table_report["new_note"] is not None
            and table_report["new_note"]["dtm"] == table_report["new_actual"]["max_abs_value"]
            and table_report["new_note"]["decisive"] == table_report["new_actual"]["decisive_entries"]
        )
        table_report["sample_probe_checks"] = sample_probe_checks(
            base_name, metadata, old_arr, new_arr, args.samples_per_table, args.seed + offset
        )
        table_report["sample_probe_mismatch_count"] = sum(
            1 for item in table_report["sample_probe_checks"] if not item["probe_match"]
        )
        report["table_comparisons"].append(table_report)

    report["summary"] = {
        "fully_identical_tables": sum(1 for item in report["table_comparisons"] if item["identical"]),
        "tables_with_value_mismatches": sum(1 for item in report["table_comparisons"] if not item["identical"]),
        "sample_probe_checks_run": sum(len(item["sample_probe_checks"]) for item in report["table_comparisons"]),
        "sample_probe_mismatches": sum(item["sample_probe_mismatch_count"] for item in report["table_comparisons"]),
    }

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"Common tables: {report['common_table_count']}")
    print(f"Fully identical tables: {report['summary']['fully_identical_tables']}")
    print(f"Tables with value mismatches: {report['summary']['tables_with_value_mismatches']}")
    print(f"Sample probe checks run: {report['summary']['sample_probe_checks_run']}")
    print(f"Sample probe mismatches: {report['summary']['sample_probe_mismatches']}")
    print(f"Report written to: {args.output}")

    if report["longest_mate_comparison"]["metadata_mismatches"]:
        print("\nLongest mate metadata mismatches:")
        for item in report["longest_mate_comparison"]["metadata_mismatches"]:
            print(
                f"  {item['table']}: old DTM {item['old']['dtm']} vs new DTM {item['new']['dtm']} | "
                f"old decisive {item['old']['decisive']} vs new decisive {item['new']['decisive']} | "
                f"old drawn {item['old']['drawn']} vs new drawn {item['new']['drawn']}"
            )

    mismatch_tables = [item for item in report["table_comparisons"] if not item["identical"]]
    if mismatch_tables:
        print("\nTables with direct value mismatches:")
        for item in mismatch_tables:
            print(
                f"  {item['table']}: {item['mismatch_count']} mismatching states, max abs diff {item['max_abs_diff']}"
            )


if __name__ == "__main__":
    main()
