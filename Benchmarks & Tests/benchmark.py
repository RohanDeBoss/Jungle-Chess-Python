# Benchmark.py (v5.0 - High Performance / Real-World Path)

import time
import os
import json
import sys
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════════
# PATH INJECTION
# ═══════════════════════════════════════════════════════════════════════════════
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

os.chdir(PARENT_DIR)

from GameLogic import Board, Pawn, Knight, Bishop, Rook, Queen, King, get_all_legal_moves
from AI import ChessBot, board_hash

# Define output paths before we start
LOG_FILE = os.path.join(SCRIPT_DIR, "benchmarks.tsv")
JSON_DIR = os.path.join(SCRIPT_DIR, "benchmark_reports")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
TARGET_DEPTH = 6  
NUM_RUNS = 1      

class DummyQueue:
    def put(self, msg): pass
    def empty(self): return True
    def get_nowait(self): return None

class DummyEvent:
    def is_set(self): return False

def load_fen(fen):
    parts = fen.split()
    board = Board(setup=False)
    r, c = 0, 0
    for char in parts[0]:
        if char == '/': r += 1; c = 0
        elif char.isdigit(): c += int(char)
        else:
            pc = {'p':Pawn,'n':Knight,'b':Bishop,'r':Rook,'q':Queen,'k':King}.get(char.lower())
            if pc: board.add_piece(pc("white" if char.isupper() else "black"), r, c)
            c += 1
    return board, ('white' if parts[1].lower() == 'w' else 'black')

POSITIONS = {
    "01. Startpos": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w",
    "02. Midgame 1": "1rbqk2r/pp1p1ppp/4p3/8/1P1P4/4B3/4PPPP/R3KBNR w",
    "03. Midgame 2": "r2qk3/pppb4/2npp3/8/6p1/2B1P3/PPPQ3P/R3KB2 b",
    "04. Midgame 3 (Complex)": "2b3kr/7p/pq1p4/1r2p1p1/3PP2P/2B5/4BPP1/R3K2R w",
    "05. Midgame 4": "2b3kr/7p/p2p4/3rp2p/8/8/4BPP1/R3K2R w",
    "06. Midgame 5": "r1bqk2r/ppp3pp/n3pp2/3p4/8/1P2KNP1/P1PP3P/RNBQ4 w",
    "07. Midgame 6": "r2qk2r/1p2bpp1/2p1bn1p/p7/3P4/2NQ1NP1/PPP2P1P/R1K4R w",
    "08. Midgame 7": "3rk2r/b4pp1/5n1p/pp6/8/5NP1/PPP1KP1P/R3R3 b",
    "09. Midgame 8 (complexyyy)": "3r1n2/2r2k1p/R7/2p1b1p1/1p3p2/2P5/2K1QBP1/1R6 w",
    "10. Endgame 1 (K+R+P vs K+B+P)": "2b5/8/p2p2k1/2K1p3/8/8/5PPr/R7 w",
    "11. Endgame 2 (Rook vs Pawns)": "8/8/p7/3ppk2/8/R3K2P/5P2/8 b",
}

def run_benchmark():
    print("=" * 70)
    print(f" JUNGLE CHESS ENGINE BENCHMARK (BOOM VERSION)")
    print(f" Target Depth: {TARGET_DEPTH} | Suite Passes: {NUM_RUNS}")
    print("=" * 70)

    # CREATE ONE PERSISTENT BOT (Reuses Transposition Table for efficiency)
    # We set use_tablebase=False to test raw calculation speed.
    dummy_board = Board()
    bot = ChessBot(dummy_board, 'white', {}, DummyQueue(), DummyEvent(), use_tablebase=False)

    pass_metrics = []
    for run in range(1, NUM_RUNS + 1):
        print(f"\n[ Executing Pass {run} / {NUM_RUNS} ]", flush=True)
        pass_nodes, pass_time = 0, 0.0

        for name, fen in POSITIONS.items():
            board, turn = load_fen(fen)
            
            # Update bot state for this position
            bot.board = board
            bot.color = turn
            bot.opponent_color = 'black' if turn == 'white' else 'white'
            bot.position_counts = {board_hash(board, turn): 1}
            
            root_moves = get_all_legal_moves(board, turn)
            best_move = root_moves[0]
            r_hash = board_hash(board, turn)
            
            # RAW SEARCH PATH (Matches UI Iterative Deepening exactly)
            start_t = time.time()
            prev_score = None
            nodes_this_pos = 0
            
            for d in range(1, TARGET_DEPTH + 1):
                # We call the internal search directly to avoid time-limit overhead
                score, best_move = bot._run_depth_iteration(
                    d, root_moves, r_hash, best_move, prev_iter_score=prev_score
                )
                prev_score = score
                nodes_this_pos += bot.nodes_searched
            
            elapsed = time.time() - start_t
            pass_nodes += nodes_this_pos
            pass_time += elapsed
            
            knps = (nodes_this_pos / elapsed / 1000) if elapsed > 0 else 0
            print(f"  {name:<30} | {nodes_this_pos:>8,} nodes | {elapsed:>6.3f}s | {knps:>5.1f} kN/s")

        pass_knps = (pass_nodes / pass_time / 1000) if pass_time > 0 else 0
        pass_metrics.append({"nodes": pass_nodes, "time": pass_time, "knps": pass_knps})

    # --- SUMMARY ---
    avg_nodes = sum(m["nodes"] for m in pass_metrics) / NUM_RUNS
    avg_time = sum(m["time"] for m in pass_metrics) / NUM_RUNS
    avg_knps = (avg_nodes / avg_time / 1000) if avg_time > 0 else 0

    print("\n" + "=" * 70 + "\n BENCHMARK SUMMARY\n" + "=" * 70)
    print(f" Avg Nodes/Pass: {avg_nodes:12,.0f}")
    print(f" Avg Time/Pass:  {avg_time:12.3f} s")
    print(f" Avg Speed:      {avg_knps:12.1f} kN/s\n" + "=" * 70)

    # --- SAVING ---
    v_name = input("\nEnter a version name: ").strip() or "untagged"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not os.path.isfile(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            f.write("Timestamp\tVersion\tDepth\tRuns\tAvgNodes\tAvgTime\tAvg_kNPS\n")
    with open(LOG_FILE, "a") as f:
        f.write(f"{timestamp}\t{v_name}\t{TARGET_DEPTH}\t{NUM_RUNS}\t{avg_nodes:.0f}\t{avg_time:.3f}\t{avg_knps:.1f}\n")
        
    os.makedirs(JSON_DIR, exist_ok=True)
    with open(os.path.join(JSON_DIR, f"bench_{v_name}.json"), "w") as f:
        json.dump({"timestamp": timestamp, "version": v_name, "passes": pass_metrics}, f, indent=4)

if __name__ == "__main__":
    run_benchmark()