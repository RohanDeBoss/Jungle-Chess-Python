# Benchmark.py (v2.1 - Streamlined Output & Pass Variance)

import time
import os
import json
from datetime import datetime

from GameLogic import Board, Pawn, Knight, Bishop, Rook, Queen, King, get_all_legal_moves
from AI import ChessBot, board_hash
import TablebaseManager

BENCHMARK_LOG_FILE = "benchmarks.tsv"
JSON_DIR = "benchmark_reports"

# --- CONFIGURATION ---
TARGET_DEPTH = 6  # How deep to search each position
NUM_RUNS = 1      # How many times to run the entire suite

# Force the Tablebase to return None so we test Python's raw calculation speed
TablebaseManager.TablebaseManager.probe = lambda self, board, turn: None

class DummyQueue:
    def put(self, msg): pass
    def empty(self): return True
    def get_nowait(self): return None

class DummyEvent:
    def is_set(self): return False

def load_fen(fen):
    parts = fen.split()
    board_part = parts[0]
    turn = 'white' if len(parts) > 1 and parts[1].lower() == 'w' else 'black'
    
    board = Board(setup=False)
    r, c = 0, 0
    for char in board_part:
        if char == '/':
            r += 1; c = 0
        elif char.isdigit():
            c += int(char)
        else:
            color = "white" if char.isupper() else "black"
            char_lower = char.lower()
            piece_class = {'p':Pawn, 'n':Knight, 'b':Bishop, 'r':Rook, 'q':Queen, 'k':King}.get(char_lower)
            if piece_class: board.add_piece(piece_class(color), r, c)
            c += 1
    return board, turn

# --- Curated Practical Jungle Chess Benchmark Positions ---
POSITIONS = {
    "01. Startpos": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w",
    "02. Midgame 1": "1rbqk2r/pp1p1ppp/4p3/8/1P1P4/4B3/4PPPP/R3KBNR w",
    "03. Midgame 2": "r2qk3/pppb4/2npp3/8/6p1/2B1P3/PPPQ3P/R3KB2 b",
    "04. Midgame 3 (Complex)": "2b3kr/7p/pq1p4/1r2p1p1/3PP2P/2B5/4BPP1/R3K2R w",
    "05. Midgame 4": "2b3kr/7p/p2p4/3rp2p/8/8/4BPP1/R3K2R w",
    "06. Midgame 5": "r2qkb1r/ppp2pp1/4bn1p/8/3P4/2N2N2/PPP2PPP/R2QK2R w",
    "07. Midgame 6": "r2qk2r/1p2bpp1/2p1bn1p/p7/3P4/2NQ1NP1/PPP2P1P/R1K4R w",
    "08. Midgame 7": "3rk2r/b4pp1/5n1p/pp6/8/5NP1/PPP1KP1P/R3R3 b",
    "09. Endgame 1 (K+R+P vs K+B+P)": "2b5/8/p2p2k1/2K1p3/8/8/5PPr/R7 w",
    "10. Endgame 2 (Rook vs Pawns)": "8/8/p7/3ppk2/8/R3K2P/5P2/8 b",
    "11. Endgame 3 (Rook vs Pawn)": "8/8/8/3R4/4p3/3K4/5k2/8 b"
}

def run_benchmark():
    print("=" * 70)
    print(f" JUNGLE CHESS RAW ENGINE SPEED BENCHMARK")
    print(f" Target Depth: {TARGET_DEPTH} | Suite Passes: {NUM_RUNS}")
    print(" (Tablebases disabled. Cache cleared between positions.)")
    print("=" * 70)

    pass_metrics = []

    for run in range(1, NUM_RUNS + 1):
        print(f"\n[ Executing Pass {run} / {NUM_RUNS} ]", flush=True)
        
        pass_nodes = 0
        pass_time = 0.0

        for name, fen in POSITIONS.items():
            board, turn = load_fen(fen)
            position_counts = {board_hash(board, turn): 1}
            bot = ChessBot(board, turn, position_counts, DummyQueue(), DummyEvent())
            
            root_moves = get_all_legal_moves(board, turn)
            if not root_moves: continue
                
            best_move = root_moves[0]
            r_hash = board_hash(board, turn)
            
            for d in range(1, TARGET_DEPTH + 1):
                start_time = time.time()
                score, best_move = bot._run_depth_iteration(d, root_moves, r_hash, best_move)
                elapsed = max(time.time() - start_time, 0.00001)
                
                pass_nodes += bot.nodes_searched
                pass_time += elapsed

        pass_knps = (pass_nodes / pass_time / 1000) if pass_time > 0 else 0
        pass_metrics.append({"nodes": pass_nodes, "time": pass_time, "knps": pass_knps})
        
        print(f"  -> Pass {run} Complete | Nodes: {pass_nodes:,} | Time: {pass_time:.3f}s | Speed: {pass_knps:.1f} kN/s")

    print("\n" + "=" * 70)
    print(" BENCHMARK SUMMARY")
    print("=" * 70)
    
    total_nodes_all = sum(m["nodes"] for m in pass_metrics)
    total_time_all = sum(m["time"] for m in pass_metrics)
    
    avg_nodes_per_pass = total_nodes_all / NUM_RUNS
    avg_time_per_pass = total_time_all / NUM_RUNS
    avg_knps = (avg_nodes_per_pass / avg_time_per_pass / 1000) if avg_time_per_pass > 0 else 0

    print(f" Average Nodes/Pass: {avg_nodes_per_pass:12,.0f}")
    print(f" Average Time/Pass:  {avg_time_per_pass:12.3f} s")
    print(f" Average Speed:      {avg_knps:12.1f} kN/s")
    print("=" * 70)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    version_name = input("\nEnter a version name/tag for this run (e.g., 'v96'): ").strip()
    if not version_name:
        version_name = "untagged"

    file_exists = os.path.isfile(BENCHMARK_LOG_FILE)
    with open(BENCHMARK_LOG_FILE, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write("Timestamp\tVersion\tDepth\tRuns\tAvgNodesPerPass\tAvgTimeSec\tAvg_kNPS\n")
        f.write(f"{timestamp}\t{version_name}\t{TARGET_DEPTH}\t{NUM_RUNS}\t{avg_nodes_per_pass:.0f}\t{avg_time_per_pass:.3f}\t{avg_knps:.1f}\n")
        
    os.makedirs(JSON_DIR, exist_ok=True)
    json_filename = os.path.join(JSON_DIR, f"bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{version_name}.json")
    
    report = {
        "timestamp": timestamp,
        "version": version_name,
        "target_depth": TARGET_DEPTH,
        "num_runs": NUM_RUNS,
        "passes": pass_metrics,
        "overall_averages": {
            "total_nodes_per_pass": avg_nodes_per_pass,
            "total_time_sec_per_pass": avg_time_per_pass,
            "avg_knps": avg_knps
        }
    }
    
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
        
    print(f"\nData saved to {BENCHMARK_LOG_FILE} and {json_filename}")

if __name__ == "__main__":
    run_benchmark()