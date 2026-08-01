# EngineRuntime.py v1.0 - shared backend/runtime plumbing for Jungle Chess engines

import gc
import glob
import inspect
import json
import os
import random
import traceback

from GameLogic import ROWS, COLS, Pawn, Knight, Bishop, Rook, Queen, King


TIME_BUFFER_SEC = 0.50
TIME_BUFFER_PCT = 0.05
MIN_MOVE_TIME = 0.03

OPTIONAL_BOT_KWARGS = (
    "time_left",
    "increment",
    "use_opening_book",
    "use_tablebase",
    "show_tt_fullness",
)


# ---------------------------------------------------------------------------
# Zobrist hashing
# ---------------------------------------------------------------------------
ZOBRIST_ARRAY = None
ZOBRIST_TURN = None


def initialize_zobrist_table():
    global ZOBRIST_ARRAY, ZOBRIST_TURN
    if ZOBRIST_ARRAY is not None:
        return
    random.seed(42)
    ZOBRIST_ARRAY = [[[[random.getrandbits(64) for _ in range(8)] for _ in range(8)]
                      for _ in range(6)] for _ in range(2)]
    ZOBRIST_TURN = random.getrandbits(64)
    random.seed()


initialize_zobrist_table()


def board_hash(board, turn):
    h = 0
    arr = ZOBRIST_ARRAY

    for piece in board.white_pieces:
        r, c = piece.pos
        h ^= arr[0][piece.z_idx][r][c]
    for piece in board.black_pieces:
        r, c = piece.pos
        h ^= arr[1][piece.z_idx][r][c]

    if turn == "black":
        h ^= ZOBRIST_TURN
    return h


def incremental_hash(parent_hash, record_tuple):
    h = parent_hash ^ ZOBRIST_TURN
    arr = ZOBRIST_ARRAY

    start, end, mp, removed_pieces, added_pieces = record_tuple

    c_idx = 0 if mp.color == "white" else 1
    p_idx = mp.z_idx
    sr, sc = start
    er, ec = end

    h ^= arr[c_idx][p_idx][sr][sc]

    mp_survived = True
    for piece, r, c in removed_pieces:
        if piece is mp:
            mp_survived = False
        else:
            pc_idx = 0 if piece.color == "white" else 1
            h ^= arr[pc_idx][piece.z_idx][r][c]

    if mp_survived:
        h ^= arr[c_idx][p_idx][er][ec]

    for piece, r, c in added_pieces:
        pc_idx = 0 if piece.color == "white" else 1
        h ^= arr[pc_idx][piece.z_idx][r][c]

    return h


# ---------------------------------------------------------------------------
# Opening book / FEN helpers
# ---------------------------------------------------------------------------
_CLS_TO_CHAR = {Pawn: "P", Knight: "N", Bishop: "B", Rook: "R", Queen: "Q", King: "K"}
OPENING_BOOK = {}


def board_to_fen(board, turn):
    fen = ""
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
                fen += ch if piece.color == "white" else ch.lower()
        if empty:
            fen += str(empty)
        if r < ROWS - 1:
            fen += "/"
    return fen + (" w" if turn == "white" else " b")


def _find_opening_book_files():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    patterns = (
        os.path.join(base_dir, "opening books", "opening_book*.json"),
        os.path.join(base_dir, "opening_book*.json"),
    )
    seen = set()
    matches = []
    for pattern in patterns:
        for path in glob.glob(pattern):
            norm = os.path.normcase(os.path.abspath(path))
            if norm in seen:
                continue
            seen.add(norm)
            matches.append(path)
    return sorted(
        matches,
        key=lambda path: (os.path.getmtime(path), os.path.basename(path)),
        reverse=True,
    )


for _book_filename in _find_opening_book_files():
    try:
        with open(_book_filename, "r", encoding="utf-8") as f:
            OPENING_BOOK = json.load(f)
        break
    except Exception as e:
        print(f"Opening book not found or invalid at {_book_filename}: {e}")


# ---------------------------------------------------------------------------
# Common bot lifecycle helpers
# ---------------------------------------------------------------------------
class SearchCancelledException(Exception):
    pass


def calc_time_check_mask(allocated):
    if allocated <= 0.15:
        return 15
    if allocated <= 0.30:
        return 31
    if allocated <= 0.60:
        return 63
    if allocated <= 1.20:
        return 127
    if allocated <= 2.50:
        return 255
    return 511


def search_time_budget(time_left, increment):
    buffer = max(TIME_BUFFER_SEC, time_left * TIME_BUFFER_PCT, increment * 1.5)
    clock_ceiling = max(0.0, time_left - buffer)

    if time_left > 0:
        buffer_health = max(0.0, min(1.0, clock_ceiling / time_left))
    else:
        buffer_health = 0.0

    divisor = 50 - (20 * buffer_health)
    optimum_time = (time_left / divisor) + (increment * 0.8)
    optimum_time = max(MIN_MOVE_TIME, optimum_time)
    optimum_time = min(optimum_time, clock_ceiling)

    max_time = min(clock_ceiling, optimum_time * 3.5)
    max_time = max(max_time, min(MIN_MOVE_TIME, clock_ceiling))
    return optimum_time, max_time


def should_reset_search_memory(bot, ply_count):
    previous_ply = getattr(bot, "ply_count", None)
    return previous_ply is not None and ply_count < previous_ply


def configure_tablebase(bot, use_tablebase):
    if not hasattr(bot, "tb_manager"):
        return
    if not hasattr(bot, "_real_tb_probe"):
        bot._real_tb_probe = bot.tb_manager.probe
    bot.use_tablebase = use_tablebase
    bot.tb_manager.probe = bot._real_tb_probe if use_tablebase else (lambda b, t: None)


def update_bot_runtime_state(bot, board, color, position_counts, comm_queue,
                             cancellation_event, bot_name, ply_count, game_mode,
                             **kwargs):
    if should_reset_search_memory(bot, ply_count):
        bot._initialize_search_state()
        gc.collect()

    bot.board = board
    bot.color = color
    bot.opponent_color = "black" if color == "white" else "white"
    bot.position_counts = position_counts
    bot.comm_queue = comm_queue
    bot.cancellation_event = cancellation_event
    bot.bot_name = bot_name
    bot.ply_count = ply_count
    bot.game_mode = game_mode

    bot.time_left = kwargs.get("time_left")
    bot.increment = kwargs.get("increment")
    bot.use_opening_book = kwargs.get("use_opening_book", True)
    bot.show_tt_fullness = kwargs.get("show_tt_fullness", False)
    configure_tablebase(bot, kwargs.get("use_tablebase", True))

    if bot.time_left:
        allocated = (bot.time_left / 30.0) + (bot.increment * 0.8)
        bot.time_check_mask = calc_time_check_mask(allocated)
    else:
        bot.time_check_mask = 511

    bot.current_age += 1


def accepted_bot_kwargs(bot_class, values):
    accepted_params = set(inspect.signature(bot_class.__init__).parameters)
    return {k: values[k] for k in OPTIONAL_BOT_KWARGS
            if k in values and k in accepted_params}


def run_bot_turn(bot):
    if bot.search_depth == 99:
        bot.ponder_indefinitely()
    else:
        bot.make_move()


def run_ai_process(board, color, position_counts, comm_queue, cancellation_event,
                   bot_class, bot_name, search_depth, ply_count, game_mode,
                   time_left=None, increment=None, use_opening_book=True,
                   use_tablebase=True, show_tt_fullness=False):
    values = {
        "time_left": time_left,
        "increment": increment,
        "use_opening_book": use_opening_book,
        "use_tablebase": use_tablebase,
        "show_tt_fullness": show_tt_fullness,
    }
    bot = bot_class(board, color, position_counts, comm_queue, cancellation_event,
                    bot_name, ply_count, game_mode,
                    **accepted_bot_kwargs(bot_class, values))
    bot.search_depth = search_depth
    run_bot_turn(bot)


# ---------------------------------------------------------------------------
# Persistent worker - one bot instance per worker process
# ---------------------------------------------------------------------------
class TaskQueueWrapper:
    """Intercept worker messages and tag them with the active task_id."""

    def __init__(self, real_queue, task_id):
        self.real_queue = real_queue
        self.task_id = task_id

    def put(self, item):
        if isinstance(item, tuple) and item and item[0] in {"move", "log", "eval", "pv"}:
            self.real_queue.put(item + (self.task_id,))
        else:
            self.real_queue.put(item)


class EngineWorker:
    def __init__(self, bot_class):
        self.bot_class = bot_class
        self.bot = None

    def handle_task(self, task, comm_queue, cancel_event):
        cancel_event.clear()
        wrapped_comm = TaskQueueWrapper(comm_queue, task.get("task_id", -1))

        values = {
            "time_left": task.get("time_left"),
            "increment": task.get("increment"),
            "use_opening_book": task.get("use_opening_book", True),
            "use_tablebase": task.get("use_tablebase", True),
            "show_tt_fullness": task.get("show_tt_fullness", False),
        }
        filtered_kwargs = accepted_bot_kwargs(self.bot_class, values)

        if self.bot is None or task.get("clear_hash", False):
            self.bot = self.bot_class(
                task["board"], task["color"], task["position_counts"],
                wrapped_comm, cancel_event, task["bot_name"],
                task["ply_count"], task["game_mode"], **filtered_kwargs
            )
        else:
            self.bot.update_state(
                task["board"], task["color"], task["position_counts"],
                wrapped_comm, cancel_event, task["bot_name"],
                task["ply_count"], task["game_mode"], **filtered_kwargs
            )

        self.bot.search_depth = task["search_depth"]
        run_bot_turn(self.bot)


def persistent_worker(work_queue, comm_queue, cancel_event, bot_class):
    worker = EngineWorker(bot_class)

    while True:
        task = work_queue.get()
        if task is None:
            break

        try:
            worker.handle_task(task, comm_queue, cancel_event)
        except Exception:
            traceback.print_exc()
            TaskQueueWrapper(comm_queue, task.get("task_id", -1)).put(("move", None))
