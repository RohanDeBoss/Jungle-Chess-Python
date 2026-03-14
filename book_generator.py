# BookGenerator.py (v1.0 - Jungle Chess Opening Book Builder)

import os
import json
import time
import multiprocessing as mp
from GameLogic import Board, Pawn, Knight, Bishop, Rook, Queen, King, get_all_legal_moves, format_move_san, ROWS, COLS
import AI
from AI import board_hash

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

BOOK_FILE        = "opening_book.json"
BOOK_PLY_DEPTH   = 4    # How deep into the game to explore (4 plies = 2 full turns)
BRANCHING_FACTOR = 3    # Keep up to the Top N moves for every position
SEARCH_DEPTH     = 6    # How deep the AI calculates to evaluate the moves
EVAL_TOLERANCE   = 50   # If the 2nd best move is >50cp worse than the 1st, discard it.

# ═══════════════════════════════════════════════════════════════════════════════
# FEN GENERATOR (To use as Dictionary Keys)
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
# BOOK GENERATOR ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def generate_book():
    print("═" * 60)
    print("  Jungle Chess Opening Book Generator")
    print(f"  Target Depth: {BOOK_PLY_DEPTH} plies | Branching: Top {BRANCHING_FACTOR} | Eval Depth: {SEARCH_DEPTH}")
    print("═" * 60)

    # Initialize a silent bot
    dummy_q = mp.Queue()
    dummy_e = mp.Event()
    bot = AI.ChessBot(Board(), 'white', {}, dummy_q, dummy_e, 'BookBuilder')
    bot._report_log = lambda msg: None
    bot._report_eval = lambda s, d: None
    bot._report_move = lambda m: None

    book = {}
    visited_fens = set()
    
    # Breadth-First Search Queue: (Board, Turn, Current_Ply)
    queue = [(Board(), 'white', 0)]
    
    start_time = time.time()
    positions_evaluated = 0

    while queue:
        current_board, turn, ply = queue.pop(0)
        fen = board_to_fen(current_board, turn)

        # Skip if we already evaluated this exact position via a transposition
        if fen in visited_fens:
            continue
        visited_fens.add(fen)

        if ply >= BOOK_PLY_DEPTH:
            continue

        legal_moves = get_all_legal_moves(current_board, turn)
        if not legal_moves:
            continue

        # Clear TT for a completely fresh evaluation of this node
        bot.tt.clear()
        
        bot.board = current_board
        bot.color = turn
        bot.opponent_color = 'black' if turn == 'white' else 'white'
        root_hash = board_hash(current_board, turn)

        best_overall_score = None
        node_moves = []

        print(f"Evaluating Ply {ply} | Positions Done: {positions_evaluated} | FEN: {fen.split()[0]}")

        # --- THE MULTI-PV TRICK ---
        for branch in range(BRANCHING_FACTOR):
            if not legal_moves:
                break

            # Iterative Deepening
            b_move = legal_moves[0]
            p_score = None
            for d in range(1, SEARCH_DEPTH + 1):
                p_score, b_move = bot._run_depth_iteration(d, legal_moves, root_hash, b_move, prev_iter_score=p_score)

            if p_score is None: 
                break

            if best_overall_score is None:
                best_overall_score = p_score

            # Check Tolerance (Don't save blunders just to fill the branching factor)
            if abs(best_overall_score - p_score) > EVAL_TOLERANCE:
                break 

            # Format Move
            child = current_board.clone()
            child.make_move(b_move[0], b_move[1])
            san = format_move_san(current_board, child, b_move)

            # Assign a weight (moves closer to the best score are picked more often)
            weight = max(1, 100 - abs(best_overall_score - p_score))

            node_moves.append({
                "move": [b_move[0], b_move[1]],
                "san": san,
                "score": p_score,
                "weight": weight
            })

            # Queue child position to be evaluated later
            queue.append((child, 'black' if turn == 'white' else 'white', ply + 1))

            # Remove this move from the legal list, so the next loop finds the 2nd best move
            legal_moves.remove(b_move)

        if node_moves:
            book[fen] = node_moves
            positions_evaluated += 1

    # Save to disk
    with open(BOOK_FILE, 'w') as f:
        json.dump(book, f, indent=4)

    elapsed = time.time() - start_time
    print("═" * 60)
    print(f"  FINISHED! Generated {positions_evaluated} book positions in {elapsed:.1f} seconds.")
    print(f"  Saved to {BOOK_FILE}")
    print("═" * 60)

if __name__ == '__main__':
    mp.freeze_support()
    generate_book()