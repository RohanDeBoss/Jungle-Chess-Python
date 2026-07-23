# book_generator.py (v4.2 - No TT wipe + New Partial Save Logic and root_depth_bonus + bug fixes)

import os
import json
import time
import math
import multiprocessing as mp
from GameLogic import Board, Pawn, Knight, Bishop, Rook, Queen, King, get_all_legal_moves, format_move_san, ROWS, COLS
import AI
from AI import board_hash


# Ensure the book saves to the correct directory
BOOK_DIR         = os.path.join(os.path.dirname(os.path.abspath(__file__)), "opening books")
if not os.path.exists(BOOK_DIR):
    os.makedirs(BOOK_DIR)
    
BOOK_FILE        = os.path.join(BOOK_DIR, "opening_book_6-3-9-2-250.json")
PARTIAL_BOOK_FILE= os.path.join(BOOK_DIR, "opening_book_6-3-9-2-250_partial.json")
BOOK_PLY_DEPTH   = 6    # 6 plies = 3 full moves
BRANCHING_FACTOR = 3    # Top 3 responses prevents exponential explosion
SEARCH_DEPTH     = 9    # Deep enough to avoid tactical blunders in side-lines
ROOT_DEPTH_BONUS = 2    # Extra depth added ONLY to Ply 0 and Ply 1
EVAL_TOLERANCE   = 250  # Discard moves that are worse than the best by 3 pawns


_CLS_TO_CHAR = {Pawn: 'P', Knight: 'N', Bishop: 'B', Rook: 'R', Queen: 'Q', King: 'K'}
_FEN_CHAR_TO_CLASS = {'p': Pawn, 'n': Knight, 'b': Bishop, 'r': Rook, 'q': Queen, 'k': King}

def board_to_fen(board: Board, turn: str) -> str:
    fen = ''
    for r in range(ROWS):
        empty = 0
        for c in range(COLS):
            piece = board.grid[r][c]
            if piece is None:
                empty += 1
            else:
                if empty:
                    fen += str(empty)
                    empty = 0
                ch = _CLS_TO_CHAR[type(piece)]
                fen += ch if piece.color == 'white' else ch.lower()
        if empty:
            fen += str(empty)
        if r < ROWS - 1:
            fen += '/'
    return fen + (' w' if turn == 'white' else ' b')

def fen_to_board(fen: str):
    parts = fen.split()
    board = Board(setup=False)
    r = c = 0
    for ch in parts[0]:
        if ch == '/':
            r += 1; c = 0
        elif ch.isdigit():
            c += int(ch)
        else:
            pc = _FEN_CHAR_TO_CLASS[ch.lower()]
            board.add_piece(pc("white" if ch.isupper() else "black"), r, c)
            c += 1
    turn = 'white' if (parts[1] if len(parts) > 1 else 'w').lower() == 'w' else 'black'
    return board, turn

# ═══════════════════════════════════════════════════════════════════════════════
# PARALLEL WORKER (Evaluates ONE full position, finding the Top N moves)
# ═══════════════════════════════════════════════════════════════════════════════

class DummyQueue:
    def put(self, msg): pass
    def empty(self): return True
    def get_nowait(self): return None

class DummyEvent:
    def is_set(self): return False

def evaluate_position(fen, ply, branch_factor, search_depth, eval_tol):
    """
    Worker function: Takes a FEN, runs iterative deepening up to SEARCH_DEPTH 
    to find the top N moves in a single parallel process.
    """
    import gc
    board, turn = fen_to_board(fen)
    
    # Initialize a silent bot with zero-overhead dummy IPC classes
    bot = AI.ChessBot(board, turn, {}, DummyQueue(), DummyEvent(), bot_name='Worker', ply_count=ply, use_opening_book=False, use_tablebase=False)
    bot._report_log = lambda msg: None
    bot._report_eval = lambda s, d: None
    bot._report_move = lambda m: None
    bot.ply_count = ply 

    # Bulletproof RAM caps (~15MB per core) to guarantee completion.
    bot.TT_MAX_SIZE = 150_000
    bot.EVAL_TT_MAX_SIZE = 50_000

    legal_moves = get_all_legal_moves(board, turn)
    root_hash = board_hash(board, turn)
    
    raw_branches = []

    for branch in range(branch_factor):
        if not legal_moves:
            break

        bot.current_age += 1 

        b_move = legal_moves[0]
        p_score = None
        
        # Iterative Deepening to target depth
        for d in range(1, search_depth + 1):
            p_score, b_move = bot._run_depth_iteration(d, legal_moves, root_hash, b_move, prev_iter_score=p_score)

        if p_score is None: 
            break

        abs_score = p_score if turn == 'white' else -p_score
        
        child = board.clone()
        child.make_move(b_move[0], b_move[1])
        san = format_move_san(board, child, b_move)

        raw_branches.append({
            "b_move": b_move,
            "san": san,
            "p_score": p_score,
            "abs_score": abs_score,
            "branch_num": branch + 1
        })
        
        # Remove best move to find the next best move in the next loop
        legal_moves.remove(b_move)

    if not raw_branches:
        return fen, [], ""

    # Pass 2: SORT branches descending by score so the absolute best move is always #1
    raw_branches.sort(key=lambda b: b["p_score"], reverse=True)
    max_p_score = raw_branches[0]["p_score"]

    # Pass 3: Filter by tolerance & scale weights relative to the TRUE best move
    node_moves = []
    logs = [f"\n[Worker] Evaluating Ply {ply} | FEN: {fen.split()[0]}"]

    for idx, b in enumerate(raw_branches, start=1):
        diff = max_p_score - b["p_score"] # Always >= 0
        
        if diff > eval_tol:
            msg = f"  [X] Discarded #{idx}: {b['san']:<8} | Eval: {b['abs_score']/100:+.2f} (Diff: {diff/100:.2f} > {eval_tol/100:.2f})"
            logs.append(msg)
            print(f"[Core] {fen.split()[0]} -> {msg.strip()}")
            break # Safe to break now because the list is sorted descending!

        weight = max(1, int(100 * math.exp(-diff / 100.0)))
        msg = f"  [✓] Saved     #{idx}: {b['san']:<8} | Eval: {b['abs_score']/100:+.2f} | Weight: {weight}"
        logs.append(msg)
        print(f"[Core] {fen.split()[0]} -> {msg.strip()}")

        node_moves.append({
            "move": [b["b_move"][0], b["b_move"][1]],
            "san": b["san"],
            "score": b["abs_score"],
            "weight": weight
        })

    ret = (fen, node_moves, "\n".join(logs))
    
    # Brutally force CPython to drop all dictionary arrays back to the OS
    bot.tt.clear()
    bot.eval_tt.clear()
    del bot.tt
    del bot.eval_tt
    del bot
    gc.collect() 
    
    return ret

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN GENERATOR CONTROLLER
# ═══════════════════════════════════════════════════════════════════════════════

def generate_book():
    print("═" * 70)
    print("  JUNGLE CHESS OPENING BOOK GENERATOR [PRO MULTI-CORE]")
    print(f"  Target Depth: {BOOK_PLY_DEPTH} plies | Branching: Top {BRANCHING_FACTOR}")
    print(f"  Eval Depth: {SEARCH_DEPTH} (with +{ROOT_DEPTH_BONUS} bonus for Plies 0 & 1)")
    print(f"  Using {mp.cpu_count()} CPU Cores for Level-Synchronous Processing")
    print("═" * 70)

    book = {}
    visited_fens = set()
    
    # --- AUTO-RESUME LOGIC ---
    resume_target = PARTIAL_BOOK_FILE if os.path.exists(PARTIAL_BOOK_FILE) else (BOOK_FILE if os.path.exists(BOOK_FILE) else None)
    if resume_target:
        try:
            with open(resume_target, 'r') as f:
                book = json.load(f)
            print(f"[INFO] Resuming from '{os.path.basename(resume_target)}' ({len(book)} positions).")
        except json.JSONDecodeError:
            print(f"[WARNING] File '{resume_target}' is corrupted. Starting fresh.")
    
    # Start queue with the initial FEN string (Keeps RAM usage extremely low)
    start_board = Board()
    current_level_queue = [(board_to_fen(start_board, 'white'), 0)]
    
    start_time = time.time()
    positions_evaluated = 0

    with mp.Pool(processes=mp.cpu_count()) as pool:
        
        for current_ply in range(BOOK_PLY_DEPTH):
            if not current_level_queue:
                break
                
            print(f"\n{'=' * 40}")
            print(f" STARTING PLY {current_ply} (Positions in queue: {len(current_level_queue)})")
            print(f"{'=' * 40}")
            
            next_level_queue = []
            args = []
            cached_results = []
            
            # Determine the depth for this specific ply level
            current_search_depth = SEARCH_DEPTH + ROOT_DEPTH_BONUS if current_ply < 2 else SEARCH_DEPTH

            # Deduplicate and sort cache vs new evaluations
            for fen, ply in current_level_queue:
                if fen in visited_fens:
                    continue
                visited_fens.add(fen)
                
                if fen in book:
                    # Instantly retrieve from cache instead of re-evaluating!
                    cached_results.append((fen, book[fen], f"[Cached] Ply {ply} | {fen.split()[0]}"))
                else:
                    args.append((fen, ply, BRANCHING_FACTOR, current_search_depth, EVAL_TOLERANCE))

            print(f"  -> Retrieving {len(cached_results)} from cache.")
            print(f"  -> Evaluating {len(args)} new positions across {mp.cpu_count()} cores...")

            # Execute all UNCACHED positions for this ply in parallel
            # chunksize=1 ensures CPUs never sit idle waiting for a hard batch to finish
            new_results = pool.starmap(evaluate_position, args, chunksize=1)
            
            # Combine cached results and new results
            all_results = cached_results + new_results
            
            # Process results to build the next ply level
            for fen, node_moves, logs in all_results:
                if fen not in book:
                    print(logs) # Only print logs for newly calculated positions
                
                if node_moves:
                    if fen not in book:
                        book[fen] = node_moves
                        positions_evaluated += 1
                    
                    # Generate the child FENs for the NEXT ply level
                    board, turn = fen_to_board(fen)
                    next_turn = 'black' if turn == 'white' else 'white'
                            
                    for move_data in node_moves:
                        child = board.clone()
                        child.make_move(move_data['move'][0], move_data['move'][1])
                        child_fen = board_to_fen(child, next_turn)
                        next_level_queue.append((child_fen, current_ply + 1))
            
            # Save progress incrementally after every full ply level to the _partial file
            with open(PARTIAL_BOOK_FILE, 'w') as f:
                json.dump(book, f, indent=4)
            print(f"\n[INFO] Saved {len(book)} total positions to '{os.path.basename(PARTIAL_BOOK_FILE)}' after Ply {current_ply}.")

            # Set up the queue for the next iteration
            current_level_queue = next_level_queue

    # Save the completed final book
    with open(BOOK_FILE, 'w') as f:
        json.dump(book, f, indent=4)
        
    # Remove the partial progress file
    if os.path.exists(PARTIAL_BOOK_FILE):
        os.remove(PARTIAL_BOOK_FILE)

    elapsed = (time.time() - start_time) / 60
    print("\n" + "═" * 70)
    print(f"  FINISHED! Evaluated {positions_evaluated} NEW positions in {elapsed:.1f} minutes.")
    print(f"  Final Book saved to '{os.path.basename(BOOK_FILE)}' ({len(book)} total positions).")
    print("═" * 70)

if __name__ == '__main__':
    mp.freeze_support()
    generate_book()