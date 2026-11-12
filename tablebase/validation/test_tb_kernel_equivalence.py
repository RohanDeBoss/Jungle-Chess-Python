"""
Differential test: Numba SameSideKernel (tablebase/TBTransition.py) vs the Python
transition workers in tablebase/TablebaseGenerator.py.

Run these commands from the project root:

    python tablebase/validation/test_tb_kernel_equivalence.py
    python tablebase/validation/test_tb_kernel_equivalence.py Queen,Queen,Queen
    python tablebase/validation/test_tb_kernel_equivalence.py --states 50000
    python tablebase/validation/test_tb_kernel_equivalence.py --count

Populate TBs/ with the 3-man and 4-man tables first so promotion/capture lookups are exercised.
Exit code is non-zero if any material mismatches.
"""
import argparse
import itertools
import os
import random
import signal
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isfile(os.path.join(ROOT, "GameLogic.py")):
    parent = os.path.dirname(ROOT)
    if parent == ROOT:
        raise RuntimeError("Could not find the project root containing GameLogic.py")
    ROOT = parent
for p in (ROOT, os.path.join(ROOT, "tablebase")):
    if p not in sys.path:
        sys.path.insert(0, p)

from tablebase import TablebaseGenerator as G
from tablebase import TBTransition as K


def order_key(name):
    return (G._PIECE_CANONICAL_ORDER[G.PIECE_CLASS_BY_NAME[name]], name)


def all_materials():
    for r in (1, 2, 3):
        for combo in itertools.combinations_with_replacement(list(G.PIECE_CLASS_BY_NAME), r):
            if G.is_possible([G.PIECE_CLASS_BY_NAME[x] for x in combo]):
                yield tuple(sorted(combo, key=order_key))


def restore_interrupts():
    # The generator's worker initialisers set SIG_IGN on console interrupts; undo that here.
    for sig in G._CONSOLE_INTERRUPT_SIGNALS:
        try:
            signal.signal(sig, signal.default_int_handler if sig == signal.SIGINT else signal.SIG_DFL)
        except Exception:
            pass


def py_setup(names):
    """Initialise the Python worker globals for this material; return (worker_fn, promo_path)."""
    n = len(names)
    if n == 1:
        gen = G.Generator(G.PIECE_CLASS_BY_NAME[names[0]])
        G._init_transition_worker(names[0], gen.queen_tb_file)
        worker, promo = G._build_transition_worker, None
    elif n == 2:
        gen = G.Generator4(*[G.PIECE_CLASS_BY_NAME[x] for x in names])
        G._init_transition_worker_4(gen.p1_name, gen.p2_name, gen.promo_tb_file)
        worker, promo = G._build_transition_worker_4, gen.promo_tb_file
    else:
        G._init_transition_worker_5(*names)
        worker, promo = G._build_transition_worker_5, None
    restore_interrupts()
    return worker, promo


def sample_flats(kern, n, rng, n_states, width=64):
    total = (32 if kern.has_pawn else 10) * 64 ** (n + 1) * 2
    windows = max(1, n_states // 25)
    parts = []
    for _ in range(windows):
        lo = rng.randrange(0, total - width) & ~1
        parts.append(K.enum_valid(lo, lo + width, n, kern.gz, kern.has_pawn))
    return np.unique(np.concatenate(parts))


def normalize(packed):
    """Packed worker output -> {flat: (table, deg, children, promo_win, promo_losses)}."""
    flats, table, deg, wpi, wpv, bpi, bpv, e_b2w, e_w2b = packed
    kids = [[] for _ in range(len(flats))]
    for edges in (e_b2w, e_w2b):
        for child, parent in edges.tolist():
            kids[parent].append(child)
    win = dict(zip(wpi.tolist(), wpv.tolist()))
    loss = {}
    for i, v in zip(bpi.tolist(), bpv.tolist()):
        loss.setdefault(i, []).append(v)
    out = {}
    for i, f in enumerate(flats.tolist()):
        tab = int(table[i])
        k = sorted(kids[i])
        if (f & 1) == 0 and tab == 1:
            k = []  # immediate win: Python keeps a partial edge list, the kernel drops it (harmless)
        out[f] = (tab, int(deg[i]), tuple(k), win.get(i, 0), tuple(sorted(loss.get(i, []))))
    return out


def check_counts(names, kern):
    n = len(names)
    total = (32 if kern.has_pawn else 10) * 64 ** (n + 1) * 2
    chunk = 4_000_000
    k_count = sum(len(K.enum_valid(lo, min(lo + chunk, total), n, kern.gz, kern.has_pawn))
                  for lo in range(0, total, chunk))
    if n == 2:
        p_count = sum(len(c) for c in G._gen_valid_4man_indices_numpy(
            total, names[0], names[1], names[0] == names[1], kern.has_pawn))
    elif n == 3:
        p_count = G._count_valid_5same_indices_numpy(total, names[0], names[1], names[2], kern.has_pawn)
    else:
        return True
    ok = k_count == p_count
    print(f"    enumeration count: kernel={k_count:,} python={p_count:,} {'OK' if ok else 'MISMATCH'}")
    return ok


def compare(names, n_states, rng, do_count):
    n = len(names)
    worker, promo = py_setup(names)
    kern = K.SameSideKernel(list(names), G.TB_DIR, G.TB_SUFFIX, promo)
    flats = sample_flats(kern, n, rng, n_states)

    kern.raw(flats[:16])  # JIT warm-up, excluded from timing
    t0 = time.time()
    packed_k = K._compact(flats, *kern.raw(flats))
    t_kernel = time.time() - t0

    t0 = time.time()
    results = [r for r in map(worker, flats.tolist()) if r is not None]
    packed_p = G._pack_same_side(results)
    t_python = time.time() - t0

    a, b = normalize(packed_p), normalize(packed_k)
    bad = [f for f in sorted(set(a) | set(b)) if a.get(f) != b.get(f)]
    label = "+".join(names)
    speed = (t_python / t_kernel) if t_kernel > 0 else float("inf")
    status = "OK" if not bad else f"MISMATCH ({len(bad)})"
    print(f"{label:<28} states={len(a):>7,}  python={t_python:6.1f}s  kernel={t_kernel:6.2f}s  x{speed:5.1f}  {status}")
    for f in bad[:5]:
        print(f"    flat {f}: python={a.get(f)}  kernel={b.get(f)}")
    ok = not bad
    if do_count and n >= 2:
        ok = check_counts(names, kern) and ok
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("materials", nargs="*", help="comma-separated piece names, e.g. Queen,Queen,Queen")
    ap.add_argument("--states", type=int, default=20000, help="approximate sampled states per material")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--count", action="store_true", help="also compare enumeration counts (slow)")
    args = ap.parse_args()

    if args.materials:
        mats = [tuple(sorted(m.split(","), key=order_key)) for m in args.materials]
    else:
        mats = list(all_materials())

    rng = random.Random(args.seed)
    failures = [m for m in mats if not compare(m, args.states, rng, args.count)]
    print()
    if failures:
        print("FAILED:", ", ".join("+".join(m) for m in failures))
        sys.exit(1)
    print("All materials match.")


if __name__ == "__main__":
    main()
