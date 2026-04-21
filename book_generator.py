# book_generator.py (v3.1 - Compatible with AI v111.3)

import os
import json
import time
import math
import multiprocessing as mp
from GameLogic import Board, Pawn, Knight, Bishop, Rook, Queen, King, get_all_legal_moves, format_move_san, ROWS, COLS
import AI
from AI import board_hash

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

BOOK_FILE        = "opening_book_5-4-7-300.json"
BOOK_PLY_DEPTH   = 5    # 4 plies = 2 full turns (White, Black, White, Black)
BRANCHING_FACTOR = 4    # Top 3 responses is the gold standard for books
SEARCH_DEPTH     = 7    # Deep Pro-Level Calculation
EVAL_TOLERANCE   = 250  # Discard alternative moves if they are 2.5 pawns worse

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

_CLS_TO_CHAR = {Pawn: 'P', Knight: 'N', Bishop: 'B', Rook: 'R', Queen: 'Q', King: 'K'}

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

# ═══════════════════════════════════════════════════════════════════════════════
# PARALLEL WORKER (Evaluates ONE full position, finding the Top N moves)
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_position(board_data, turn, ply, branch_factor, search_depth, eval_tol):
    """
    Worker function: Takes a board state, runs iterative deepening up to SEARCH_DEPTH 
    to find the best move, removes it, clears TT, and repeats to find the Top N moves.
    """
    board = board_data.clone()
    fen = board_to_fen(board, turn)
    
    # Initialize a silent bot
    bot = AI.ChessBot(board, turn, {}, mp.Queue(), mp.Event(), bot_name='Worker', ply_count=ply, use_opening_book=False, use_tablebase=False)
    bot._report_log = lambda msg: None
    bot._report_eval = lambda s, d: None
    bot._report_move = lambda m: None
    bot.ply_count = ply 

    legal_moves = get_all_legal_moves(board, turn)
    root_hash = board_hash(board, turn)
    
    best_overall_score = None
    node_moves = []
    logs = []
    
    logs.append(f"\n[Worker] Evaluating Ply {ply} | FEN: {fen.split()[0]}")

    for branch in range(branch_factor):
        if not legal_moves:
            break

        # CRITICAL CORRECTNESS: Clear Transposition Table between branches so 
        # bounds from the previous best-move don't cause false cutoffs.
        bot.tt.clear()

        b_move = legal_moves[0]
        p_score = None
        
        # Iterative Deepening to target depth
        for d in range(1, search_depth + 1):
            p_score, b_move = bot._run_depth_iteration(d, legal_moves, root_hash, b_move, prev_iter_score=p_score)

        if p_score is None: 
            break

        if best_overall_score is None:
            best_overall_score = p_score

        diff = abs(best_overall_score - p_score)
        
        # CRITICAL CORRECTNESS: Absolute White Evaluation Sync
        abs_score = p_score if turn == 'white' else -p_score
        
        child = board.clone()
        child.make_move(b_move[0], b_move[1])
        san = format_move_san(board, child, b_move)

        if diff > eval_tol:
            logs.append(f"  [X] Discarded #{branch+1}: {san:<8} | Eval: {abs_score/100:+.2f} (Diff: {diff/100:.2f} > {eval_tol/100:.2f})")
            break 

        weight = max(1, int(100 * math.exp(-diff / 100.0)))
        logs.append(f"  [✓] Saved     #{branch+1}: {san:<8} | Eval: {abs_score/100:+.2f} | Weight: {weight}")

        node_moves.append({
            "move": [b_move[0], b_move[1]],
            "san": san,
            "score": abs_score,
            "weight": weight
        })
        
        # Remove best move to find the next best move in the next loop
        legal_moves.remove(b_move)

    return fen, node_moves, "\n".join(logs)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN GENERATOR CONTROLLER
# ═══════════════════════════════════════════════════════════════════════════════

def generate_book():
    print("═" * 70)
    print("  JUNGLE CHESS OPENING BOOK GENERATOR [PRO MULTI-CORE]")
    print(f"  Target Depth: {BOOK_PLY_DEPTH} plies | Branching: Top {BRANCHING_FACTOR} | Eval Depth: {SEARCH_DEPTH}")
    print(f"  Using {mp.cpu_count()} CPU Cores for Level-Synchronous Processing")
    print("═" * 70)

    book = {}
    visited_fens = set()
    
    # We process level by level (Ply 0 -> Ply 1 -> Ply 2 -> etc)
    current_level_queue = [(Board(), 'white', 0)]
    
    start_time = time.time()
    positions_evaluated = 0

    with mp.Pool(processes=mp.cpu_count()) as pool:
        
        for current_ply in range(BOOK_PLY_DEPTH):
            if not current_level_queue:
                break
                
            print(f"\n{'=' * 40}")
            print(f" STARTING PLY {current_ply} (Positions to evaluate: {len(current_level_queue)})")
            print(f"{'=' * 40}")
            
            next_level_queue = []
            
            # Prepare arguments for the parallel workers
            args = []
            for board, turn, ply in current_level_queue:
                fen = board_to_fen(board, turn)
                if fen not in visited_fens:
                    visited_fens.add(fen)
                    args.append((board, turn, ply, BRANCHING_FACTOR, SEARCH_DEPTH, EVAL_TOLERANCE))

            # Execute all positions for this ply in parallel!
            results = pool.starmap(evaluate_position, args)
            
            # Process results
            for fen, node_moves, logs in results:
                print(logs) # Print the worker's logs sequentially
                
                if node_moves:
                    book[fen] = node_moves
                    positions_evaluated += 1
                    
                    # Generate the child boards for the NEXT ply level
                    turn = 'black' if fen.endswith('w') else 'white'
                    # We have to reconstruct the parent board to apply the moves
                    # (This is very fast and saves us sending huge objects over IPC)
                    parts = fen.split()
                    board = Board(setup=False)
                    r = c = 0
                    for ch in parts[0]:
                        if ch == '/':
                            r += 1; c = 0
                        elif ch.isdigit():
                            c += int(ch)
                        else:
                            _FEN_CHAR_TO_CLASS = {'p': Pawn, 'n': Knight, 'b': Bishop, 'r': Rook, 'q': Queen, 'k': King}
                            pc = _FEN_CHAR_TO_CLASS[ch.lower()]
                            board.add_piece(pc("white" if ch.isupper() else "black"), r, c)
                            c += 1
                            
                    for move_data in node_moves:
                        child = board.clone()
                        child.make_move(move_data['move'][0], move_data['move'][1])
                        next_level_queue.append((child, turn, current_ply + 1))
            
            # Save progress after every full ply level
            with open(BOOK_FILE, 'w') as f:
                json.dump(book, f, indent=4)
            print(f"\n[INFO] Saved {positions_evaluated} positions to {BOOK_FILE} after Ply {current_ply}.")

            # Set up the queue for the next iteration
            current_level_queue = next_level_queue

    elapsed = (time.time() - start_time) / 60
    print("\n" + "═" * 70)
    print(f"  FINISHED! Processed {positions_evaluated} positions in {elapsed:.1f} minutes.")
    print(f"  Final Book saved to {BOOK_FILE}")
    print("═" * 70)

if __name__ == '__main__':
    mp.freeze_support()
    generate_book()