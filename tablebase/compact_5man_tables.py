#!/usr/bin/env python3
"""Create side-by-side compact copies of eligible five-man tablebases.

Every source is checked for diagonal-transpose consistency and every retained
value is round-tripped before a compact copy is installed. Dense sources stay
untouched for review.
"""

import argparse
import os
import re
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tablebase.DiagonalCompressedTable import (
    convert_dense_table,
    DiagonalCompressedTable,
    expand_compressed_table,
)


TABLE_RE = re.compile(r"^K_(Bishop|Knight|Queen|Rook)_(Bishop|Knight|Queen|Rook)_(Bishop|Knight|Queen|Rook)_K_tb8\.bin$")
SHAPE = (10, 64, 64, 64, 64, 2)


def eligible_tables(tb_dir):
    result = []
    for name in sorted(os.listdir(tb_dir)):
        match = TABLE_RE.fullmatch(name)
        if not match:
            continue
        path = os.path.join(tb_dir, name)
        if not os.path.isfile(path):
            continue
        if os.path.exists(path + ".diagmap"):
            raise RuntimeError(f"Expected a dense source but found a compact map: {path}")
        compact_path = path.replace("_tb8.bin", "_tb8(compressed).bin")
        if os.path.exists(compact_path) or os.path.exists(compact_path + ".diagmap"):
            continue
        result.append((path, name[:-len("_tb8.bin")], match.groups()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tb-dir", default=os.path.join(PROJECT_ROOT, "TBs"))
    parser.add_argument("--table", action="append", help="base table name; repeat to select several")
    parser.add_argument("--restore-dense", help="restore dense form after an earlier in-place conversion")
    args = parser.parse_args()

    lock = os.path.join(args.tb_dir, "generator.lock")
    if os.path.exists(lock):
        raise SystemExit(f"Refusing to compact while generator lock exists: {lock}")

    if args.restore_dense:
        base_name = args.restore_dense
        dense_path = os.path.join(args.tb_dir, base_name + "_tb8.bin")
        compact_path = os.path.join(args.tb_dir, base_name + "_tb8(compressed).bin")
        map_path = dense_path + ".diagmap"
        if not (os.path.exists(dense_path) and os.path.exists(map_path)):
            raise SystemExit(f"No in-place compacted table and map found for {base_name}")
        if os.path.exists(compact_path) or os.path.exists(compact_path + ".diagmap"):
            raise SystemExit(f"Compressed copy already exists for {base_name}")
        expanded_tmp = dense_path + ".restore.tmp"
        try:
            expand_compressed_table(dense_path, expanded_tmp, SHAPE, base_name)
            os.replace(dense_path, compact_path)
            os.replace(map_path, compact_path + ".diagmap")
            os.replace(expanded_tmp, dense_path)
        except Exception:
            if not os.path.exists(dense_path) and os.path.exists(compact_path):
                os.replace(compact_path, dense_path)
            if not os.path.exists(map_path) and os.path.exists(compact_path + ".diagmap"):
                os.replace(compact_path + ".diagmap", map_path)
            if os.path.exists(expanded_tmp):
                os.remove(expanded_tmp)
            raise
        print(f"RESTORED dense form and preserved compressed copy: {base_name}", flush=True)
        return

    candidates = eligible_tables(args.tb_dir)
    if args.table:
        selected = set(args.table)
        candidates = [item for item in candidates if item[1] in selected]
        missing = selected - {item[1] for item in candidates}
        if missing:
            raise SystemExit("Missing eligible dense tables: " + ", ".join(sorted(missing)))
    if not candidates:
        raise SystemExit("No eligible dense non-pawn same-side five-man tables found.")

    prepared = []
    try:
        for index, (path, base_name, piece_names) in enumerate(candidates, 1):
            print(f"[{index}/{len(candidates)}] Validating and packing {base_name}...", flush=True)
            compact, map_file, count, itemsize = convert_dense_table(path, base_name, SHAPE, piece_names)
            prepared.append((path, base_name, compact, map_file, count, itemsize))
            old_size = SHAPE[0] * int(np.prod(SHAPE[1:], dtype=np.int64)) * itemsize
            new_size = os.path.getsize(compact)
            print(f"  verified {count:,} stored values; {old_size:,} -> {new_size:,} bytes", flush=True)

        # Install separate compact files; never replace a dense source.
        installed = []
        try:
            for path, base_name, compact, map_file, count, itemsize in prepared:
                compact_path = path.replace("_tb8.bin", "_tb8(compressed).bin")
                compact_map = compact_path + ".diagmap"
                if os.path.exists(compact_path) or os.path.exists(compact_map):
                    raise RuntimeError(f"Compressed output already exists: {compact_path}")
                try:
                    os.replace(compact, compact_path)
                    os.replace(map_file, compact_map)
                    loaded = DiagonalCompressedTable(compact_path, SHAPE, base_name)
                    if loaded.compact_count != count or loaded.dtype.itemsize != itemsize:
                        raise RuntimeError(f"Loaded metadata mismatch for {base_name}")
                    loaded._flat._mmap.close()
                    loaded._map._mmap.close()
                except Exception:
                    if os.path.exists(compact_path):
                        os.remove(compact_path)
                    if os.path.exists(compact_map):
                        os.remove(compact_map)
                    raise
                installed.append((compact_path, compact_map))
        except Exception:
            for compact_path, compact_map in reversed(installed):
                if os.path.exists(compact_path):
                    os.remove(compact_path)
                if os.path.exists(compact_map):
                    os.remove(compact_map)
            raise
    finally:
        for _, _, compact, map_file, _, _ in prepared:
            for temporary in (compact, map_file):
                if os.path.exists(temporary):
                    os.remove(temporary)

    print(f"COMPLETE: {len(prepared)} compacted copies created; dense sources preserved.", flush=True)


if __name__ == "__main__":
    main()
