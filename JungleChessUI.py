# JungleChessUI.py (v14.91 - Persistent Worker Processes + IPC Race Condition Fixes)

import tkinter as tk
from tkinter import ttk, messagebox
import math
import random
import time
import re
from GameLogic import *
from AI import ChessBot, board_hash
from OpponentAI import OpponentAI
from enum import Enum
import multiprocessing as mp

class GameMode(Enum):
    HUMAN_VS_BOT   = "bot"
    HUMAN_VS_HUMAN = "human"
    AI_VS_AI       = "ai_vs_ai"

_CASUALTIES_RE = re.compile(r'\s*\(.*?\)')
_FEN_CHAR_TO_CLASS = {'p': Pawn, 'n': Knight, 'b': Bishop, 'r': Rook, 'q': Queen, 'k': King}
_CLASS_TO_FEN_CHAR = {Pawn:'P', Knight:'N', Bishop:'B', Rook:'R', Queen:'Q', King:'K'}


# ---------------------------------------------------------------------------
# Persistent worker — runs in a subprocess, imports happen ONCE at startup.
# ---------------------------------------------------------------------------
class TaskQueueWrapper:
    """Intercepts messages from the bot to attach the current task_id to moves."""
    def __init__(self, real_queue, task_id):
        self.real_queue = real_queue
        self.task_id = task_id

    def put(self, item):
        # Attach the task_id only to 'move' messages
        if isinstance(item, tuple) and item[0] == 'move':
            self.real_queue.put(('move', item[1], self.task_id))
        else:
            self.real_queue.put(item)

def persistent_worker(work_queue, comm_queue, cancel_event, bot_class):
    """
    Sits in a loop waiting for task dicts. Each task dict contains everything
    the bot needs. Sending None shuts the worker down.
    """
    while True:
        task = work_queue.get()          # blocks until a task arrives
        if task is None:                 # shutdown signal
            break

        # The worker clears the event AFTER receiving the task. 
        # This prevents the UI from accidentally "un-cancelling" an aborting task.
        cancel_event.clear()             

        task_id = task.get('task_id', -1)
        wrapped_comm = TaskQueueWrapper(comm_queue, task_id)

        try:
            try:
                bot = bot_class(
                    task['board'], task['color'], task['position_counts'],
                    wrapped_comm, cancel_event,
                    task['bot_name'], task['ply_count'], task['game_mode'],
                    time_left=task.get('time_left'),
                    increment=task.get('increment'),
                    use_opening_book=task.get('use_opening_book', True),
                )
            except TypeError:
                try:
                    bot = bot_class(
                        task['board'], task['color'], task['position_counts'],
                        wrapped_comm, cancel_event,
                        task['bot_name'], task['ply_count'], task['game_mode'],
                        time_left=task.get('time_left'),
                        increment=task.get('increment'),
                    )
                except TypeError:
                    bot = bot_class(
                        task['board'], task['color'], task['position_counts'],
                        wrapped_comm, cancel_event,
                        task['bot_name'], task['ply_count'], task['game_mode'],
                    )

            bot.search_depth = task['search_depth']
            if task['search_depth'] == 99:
                bot.ponder_indefinitely()
            else:
                bot.make_move()
                
        except Exception as e:
            # Prevents a silent crash from locking up the UI forever
            import traceback
            traceback.print_exc()
            wrapped_comm.put(('move', None))


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
class EnhancedChessApp:
    MAIN_AI_NAME     = "AI Bot"
    OPPONENT_AI_NAME = "OP Bot"
    ANALYSIS_AI_NAME = "Analysis"
    slidermaxvalue   = 12
    MAX_GAME_MOVES   = 200
    AI_SERIES_GAMES  = 300

    def __init__(self, master):
        self.master = master
        self.master.title("Jungle Chess")
        random.seed()

        # --- COMMUNICATION ---
        self.comm_queue = mp.Queue()

        # --- PERSISTENT WORKER STATE ---
        self.current_task_id   = 0
        self.main_work_queue   = mp.Queue()
        self.op_work_queue     = mp.Queue()
        self.main_cancel_event = mp.Event()
        self.op_cancel_event   = mp.Event()
        self.active_worker_name = None   # 'main' | 'op' | None
        self.analysis_thinking  = False
        self.main_worker        = None   
        self.op_worker          = None

        # --- BOARD / GAME STATE ---
        self.board        = Board()
        self.turn         = "white"
        self.selected     = None
        self.valid_moves  = []
        self.game_over    = False
        self.game_result  = None
        self.dragging     = False
        self.drag_piece_ghost = None
        self.drag_start   = None

        self.full_history         = []
        self.history_pointer      = -1
        self.position_counts      = {}
        self.current_opening_sequence = []
        self.square_size          = 75
        self.base_sidebar_width   = 280

        self.game_mode           = tk.StringVar(value=GameMode.HUMAN_VS_BOT.value)
        self.analysis_mode_var   = tk.BooleanVar(value=True)
        self.ai_series_running   = False
        self.ai_series_stats     = {'game_count': 0, 'my_ai_wins': 0, 'op_ai_wins': 0, 'draws': 0}
        self.depth_stats         = {}
        self.auto_save_stats_var = tk.BooleanVar(value=True)
        self.show_pv_var         = tk.BooleanVar(value=True)
        self.long_notation_var   = tk.BooleanVar(value=False)
        self.instant_move        = tk.BooleanVar(value=False)
        self.use_opening_book_var = tk.BooleanVar(value=True)

        self.current_pv_raw  = []
        self.current_pv_san  = []
        self.last_pv_message = None

        self.white_playing_bot_type = "main"
        self.human_color            = "white"
        self.board_orientation      = "white"
        self.last_move_timestamp    = None
        self.game_started           = False

        self.last_eval_score = 0.0
        self.last_eval_depth = None
        self.last_eval_bar_w = 0
        self.last_eval_bar_h = 0

        # --- TIME STATE ---
        self.time_control_seconds = tk.IntVar(value=300)
        self.white_time      = 0.0
        self.black_time      = 0.0
        self.increment       = 0.0
        self.last_clock_tick = None
        self.clock_running   = False
        self.use_clock_var   = tk.BooleanVar(value=True)

        self.COLORS = self.setup_styles()
        self.master.configure(bg=self.COLORS['bg_dark'])
        self.build_ui()
        self.master.bind("<Key>", self.handle_key_press)
        self.master.protocol("WM_DELETE_WINDOW", self._on_close)
        self._start_persistent_workers()
        self.process_comm_queue()
        self.reset_game()

    # ------------------------------------------------------------------ workers
    def _start_persistent_workers(self):
        self.main_worker = mp.Process(
            target=persistent_worker,
            args=(self.main_work_queue, self.comm_queue,
                  self.main_cancel_event, ChessBot),
            daemon=True,
        )
        self.op_worker = mp.Process(
            target=persistent_worker,
            args=(self.op_work_queue, self.comm_queue,
                  self.op_cancel_event, OpponentAI),
            daemon=True,
        )
        self.main_worker.start()
        self.op_worker.start()

    def _on_close(self):
        """Gracefully shut down workers before destroying the window."""
        try:
            self.main_work_queue.put(None)
            self.op_work_queue.put(None)
        except Exception:
            pass
        self.master.destroy()

    # ------------------------------------------------------------------ helpers
    def _format_san_display(self, s):
        return s if (self.long_notation_var.get() or not s) else _CASUALTIES_RE.sub('', s)

    def _on_notation_toggle(self):
        self.update_moves_list()
        self._render_pv()

    # ------------------------------------------------------------------ clock helpers
    def _start_clock(self):
        if not self.use_clock_var.get() or self.game_over or self.clock_running:
            return
        self.last_clock_tick = time.time()
        self.clock_running   = True
        self._tick_clock()

    def _pause_clock(self):
        was_running        = self.clock_running
        self.clock_running = False
        return was_running

    def _reset_clock_state(self):
        base = float(self.time_control_seconds.get())
        self.white_time      = base
        self.black_time      = base
        self.increment       = base / 60.0
        self.clock_running   = False
        self.last_clock_tick = None

    # ------------------------------------------------------------------ UI build
    def build_ui(self):
        sw, sh = self.master.winfo_screenwidth(), self.master.winfo_screenheight()
        self.master.geometry(f"{sw}x{sh}+0+0")
        self.master.state('zoomed')

        self.main_frame = ttk.Frame(self.master, style='Left.TFrame')
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- LEFT PANEL ---
        self.left_panel = ttk.Frame(self.main_frame, style='Left.TFrame')
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        self.left_panel.pack_propagate(False)
        ttk.Label(self.left_panel, text="JUNGLE CHESS", style='Header.TLabel',
                  font=('Helvetica', 22, 'bold')).pack(pady=(0, 5))
        self.pv_text = tk.Text(self.left_panel, height=6, bg=self.COLORS['bg_medium'],
                               fg=self.COLORS['text_light'], font=('Helvetica', 10),
                               wrap=tk.WORD, borderwidth=1, relief="solid")
        self.pv_text.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=10)
        self.pv_text.config(state=tk.DISABLED)
        self._build_control_widgets(self.left_panel)

        # --- CENTER PANEL ---
        self.center_panel = ttk.Frame(self.main_frame, style='Right.TFrame')
        self.center_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.board_column = ttk.Frame(self.center_panel, style='Right.TFrame')
        self.board_column.pack(expand=True, fill=tk.BOTH)

        self.eval_frame = ttk.Frame(self.board_column, style='Right.TFrame',
                                    width=COLS * self.square_size, height=58)
        self.eval_frame.pack(side=tk.TOP, anchor=tk.CENTER, pady=(6, 5))
        self.eval_frame.pack_propagate(False)
        self.eval_score_label = ttk.Label(self.eval_frame, text="Even",
                                          style='Status.TLabel', anchor="center")
        self.eval_score_label.pack(side=tk.TOP, pady=(0, 4))
        self.eval_bar_canvas = tk.Canvas(self.eval_frame, width=COLS * self.square_size,
                                         height=20, bg=self.COLORS['bg_light'],
                                         highlightthickness=1,
                                         highlightbackground=self.COLORS['text_dark'])
        self.eval_bar_canvas.pack(side=tk.TOP, anchor=tk.CENTER)
        self.eval_bar_canvas.bind("<Configure>", self.redraw_eval_bar_on_resize)

        self.board_row_frame = ttk.Frame(self.board_column, style='Right.TFrame')
        self.board_row_frame.pack(expand=True, fill=tk.BOTH)
        self.canvas_frame = ttk.Frame(self.board_row_frame, style='Canvas.TFrame')
        self.canvas_frame.pack(expand=True, fill=tk.BOTH)

        self.canvas = tk.Canvas(self.canvas_frame,
                                width=COLS * self.square_size, height=ROWS * self.square_size,
                                bg=self.COLORS['bg_medium'], highlightthickness=0)
        self.board_image    = self.create_board_image()
        self.board_image_id = self.canvas.create_image(0, 0, anchor='nw', tags="board")
        self.canvas.pack(expand=True)

        for attr in ('top_bot_label', 'bottom_bot_label'):
            setattr(self, attr, ttk.Label(
                self.board_row_frame, text="", font=("Helvetica", 11, "bold"),
                background=self.COLORS['bg_medium'], foreground=self.COLORS['text_light'],
                anchor="center", justify=tk.CENTER))

        # Navigation bar
        self.navigation_frame = ttk.Frame(self.center_panel, style='Right.TFrame')
        self.navigation_frame.pack(fill=tk.X, pady=(5, 10))
        self.start_button, self.undo_button, self.redo_button, self.end_button = [
            ttk.Button(self.navigation_frame, text=t, command=c,
                       style='Nav.TButton', state=tk.DISABLED)
            for t, c in [("«", self.go_to_start), ("‹", self.undo_move),
                          ("›", self.redo_move),   ("»", self.go_to_end)]]
        self.navigation_frame.columnconfigure(0, weight=1)
        self.navigation_frame.columnconfigure(5, weight=1)
        for col, btn in enumerate([self.start_button, self.undo_button,
                                    self.redo_button,  self.end_button], start=1):
            btn.grid(row=0, column=col, padx=5)

        # --- RIGHT PANEL ---
        self.right_panel = ttk.Frame(self.main_frame, style='Left.TFrame')
        self.right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        self.right_panel.pack_propagate(False)
        self._build_right_sidebar_widgets(self.right_panel)

        self.main_frame.bind("<Configure>",   self.handle_main_resize)
        self.center_panel.bind("<Configure>", self.handle_board_resize)

    def _build_control_widgets(self, parent):
        gf = ttk.Frame(parent, style='Left.TFrame')
        gf.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(gf, text="GAME MODE", style='Header.TLabel').pack(anchor=tk.W)
        for mode in GameMode:
            ttk.Radiobutton(gf, text=mode.name.replace("_", " ").title(),
                            variable=self.game_mode, value=mode.value,
                            command=self.on_mode_changed,
                            style='Custom.TRadiobutton').pack(anchor=tk.W, pady=(2, 0))

        cf = ttk.Frame(parent, style='Left.TFrame')
        cf.pack(fill=tk.X, pady=10)
        self.controls_frame = cf
        for txt, cmd in [("NEW GAME",        self.reset_game),
                          ("SWAP SIDES",      self.swap_sides),
                          ("AI vs OP Series", self.start_ai_series)]:
            ttk.Button(cf, text=txt, command=cmd, style='Control.TButton').pack(fill=tk.X, pady=3)
        self.flip_view_btn = ttk.Button(cf, text="FLIP VIEW",
                                        command=self.toggle_board_view, style='Control.TButton')
        self.flip_view_btn.pack(fill=tk.X, pady=3)

        ttk.Label(cf, text="Depth:", style='SmallHeader.TLabel').pack(anchor=tk.W, pady=(10, 0))
        self.bot_depth_slider = tk.Scale(cf, from_=1, to=self.slidermaxvalue,
                                         orient=tk.HORIZONTAL, bg=self.COLORS['bg_dark'],
                                         fg=self.COLORS['text_light'],
                                         highlightthickness=0, relief='flat')
        self.bot_depth_slider.set(ChessBot.search_depth)
        self.bot_depth_slider.pack(fill=tk.X, pady=(0, 5))

        for text, var, cmd in [
            ("Use Opening Book",           self.use_opening_book_var, None),
            ("Instant Moves",              self.instant_move,         None),
            ("Analysis Mode (H-vs-H)",     self.analysis_mode_var,    self._update_analysis_after_state_change),
            ("Auto-save Depth Stats",      self.auto_save_stats_var,  None),
            ("Show Engine Lines (PV)",     self.show_pv_var,          self._render_pv),
            ("Long Notation (Casualties)", self.long_notation_var,    self._on_notation_toggle),
        ]:
            kw = {'command': cmd} if cmd else {}
            ttk.Checkbutton(cf, text=text, variable=var,
                            style='Custom.TCheckbutton', **kw).pack(anchor=tk.W, pady=(2, 2))

    def _build_right_sidebar_widgets(self, parent):
        info = ttk.Frame(parent, style='Left.TFrame')
        info.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))
        self.info_frame = info
        self.game_info_label = ttk.Label(info, text="Match Info", style='Header.TLabel')
        self.game_info_label.pack(anchor=tk.W)
        self.turn_label = ttk.Label(info, text="WHITE'S TURN", style='Status.TLabel')
        self.turn_label.pack(fill=tk.X, pady=(5, 5))
        ttk.Checkbutton(info, text="Use Clock", variable=self.use_clock_var,
                        command=self._toggle_clock).pack(anchor=tk.W, pady=(2, 2))

        self.clock_frame = ttk.Frame(info, style='Left.TFrame')
        self.clock_frame.pack(fill=tk.X, pady=(5, 5))
        self.black_clock_lbl = tk.Label(self.clock_frame, text="00:00.0",
                                        font=('Courier', 18, 'bold'),
                                        bg=self.COLORS['bg_medium'],
                                        fg=self.COLORS['text_light'], pady=2)
        self.black_clock_lbl.pack(side=tk.TOP, fill=tk.X, pady=1)
        self.white_clock_lbl = tk.Label(self.clock_frame, text="00:00.0",
                                        font=('Courier', 18, 'bold'),
                                        bg=self.COLORS['bg_light'],
                                        fg=self.COLORS['text_light'], pady=2)
        self.white_clock_lbl.pack(side=tk.BOTTOM, fill=tk.X, pady=1)

        self.time_control_frame = ttk.Frame(info, style='Left.TFrame')
        self.time_control_frame.pack(fill=tk.X, pady=(5, 5))
        self.time_control_label = ttk.Label(self.time_control_frame,
                                            text="Time Control: 05:00",
                                            style='SmallHeader.TLabel')
        self.time_control_label.pack(anchor=tk.W)
        self.time_control_slider = tk.Scale(
            self.time_control_frame, from_=10, to=600, orient=tk.HORIZONTAL,
            bg=self.COLORS['bg_dark'], fg=self.COLORS['text_light'],
            highlightthickness=0, relief='flat', showvalue=False,
            variable=self.time_control_seconds,
            command=lambda _=None: self._update_time_control_label())
        self.time_control_slider.set(int(self.time_control_seconds.get()))
        self.time_control_slider.pack(fill=tk.X, pady=(2, 2))
        self.time_control_slider.bind("<ButtonRelease-1>", lambda e: self.reset_game())

        self.bottom_tools_frame = ttk.Frame(parent, style='Left.TFrame')
        self.bottom_tools_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(0, 10))
        self.fen_entry = self._create_import_export_widget(
            self.bottom_tools_frame, "FEN String:", self.load_fen_from_entry, self.copy_fen_to_clipboard)
        self.pgn_entry = self._create_import_export_widget(
            self.bottom_tools_frame, "PGN Record:", self.load_pgn_from_entry, self.copy_pgn_to_clipboard)

        self.scoreboard_label = ttk.Label(parent, text="", font=("Helvetica", 11),
                                          justify=tk.LEFT, background=self.COLORS['bg_dark'],
                                          foreground=self.COLORS['text_light'])
        self.scoreboard_label.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 5))

        ttk.Label(parent, text="Move History", style='SmallHeader.TLabel').pack(side=tk.TOP, anchor=tk.W)
        self.tree_frame = tk.Frame(parent, bg=self.COLORS['bg_medium'])
        self.tree_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(2, 10))
        hdr = tk.Frame(self.tree_frame, bg=self.COLORS['bg_light'])
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text=" #  " + "White".center(14) + "Black".center(14),
                 bg=self.COLORS['bg_light'], fg=self.COLORS['text_light'],
                 font=('Courier', 11, 'bold'), anchor=tk.W).pack(side=tk.LEFT, fill=tk.X)
        self.moves_text = tk.Text(self.tree_frame, font=('Courier', 11),
                                  bg=self.COLORS['bg_medium'], fg=self.COLORS['text_light'],
                                  borderwidth=0, highlightthickness=0,
                                  state=tk.DISABLED, cursor="arrow", wrap=tk.NONE)
        sb = ttk.Scrollbar(self.tree_frame, orient=tk.VERTICAL, command=self.moves_text.yview)
        self.moves_text.configure(yscrollcommand=sb.set)
        self.moves_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

    def _create_import_export_widget(self, parent, label, load_cmd, copy_cmd):
        frame = ttk.Frame(parent, style='Left.TFrame')
        frame.pack(fill=tk.X, pady=(2, 2))
        ttk.Label(frame, text=label, style='SmallHeader.TLabel').pack(anchor=tk.W)
        entry = ttk.Entry(frame, font=('Courier', 10), style='TEntry')
        entry.pack(fill=tk.X, pady=(2, 2))
        bf = ttk.Frame(frame, style='Left.TFrame')
        bf.pack(fill=tk.X)
        prefix = label.split()[0]
        ttk.Button(bf, text=f"Load {prefix}", command=load_cmd,
                   style='Control.TButton').pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ttk.Button(bf, text=f"Copy {prefix}", command=copy_cmd,
                   style='Control.TButton').pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))
        return entry

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        C = {'bg_dark':    '#1a1a2e', 'bg_medium':  '#16213e', 'bg_light':   '#0f3460',
             'accent':     '#e94560', 'text_light': '#ffffff', 'text_dark':  '#a2a2a2',
             'warning':    '#FF8C00'}

        style.configure('.',             background=C['bg_dark'],   foreground=C['text_light'])
        style.configure('TFrame',        background=C['bg_dark'])
        style.configure('Left.TFrame',   background=C['bg_dark'])
        style.configure('Right.TFrame',  background=C['bg_medium'])
        style.configure('Canvas.TFrame', background=C['bg_medium'])

        style.configure('Header.TLabel',      background=C['bg_dark'],  foreground=C['text_light'], font=('Helvetica', 14, 'bold'), padding=(0, 5))
        style.configure('SmallHeader.TLabel', background=C['bg_dark'],  foreground=C['text_light'], font=('Helvetica', 12, 'bold'), padding=(0, 1))
        style.configure('Status.TLabel',      background=C['bg_light'], foreground=C['text_light'], font=('Helvetica', 14, 'bold'), padding=(6, 4), relief='solid', borderwidth=1)

        style.configure('Nav.TButton', background=C['bg_light'], foreground=C['text_light'],
                        font=('Helvetica', 16, 'bold'), padding=(10, 5), borderwidth=0)
        style.map('Nav.TButton', background=[('active', C['bg_light']), ('pressed', C['bg_medium'])],
                                 foreground=[('disabled', C['text_dark'])])

        for name, bg, pressed in [('Control', C['accent'], '#d13550'),
                                   ('Flipped', C['warning'], '#E07B00')]:
            style.configure(f'{name}.TButton', background=bg, foreground=C['text_light'],
                            font=('Helvetica', 11, 'bold'), padding=(8, 4), borderwidth=0)
            style.map(f'{name}.TButton', background=[('active', bg), ('pressed', pressed)])

        for name in ('Custom.TRadiobutton', 'Custom.TCheckbutton'):
            style.configure(name, background=C['bg_dark'], foreground=C['text_light'],
                            font=('Helvetica', 11))
            style.map(name, background=[('active', C['bg_dark'])],
                      indicatorcolor=[('selected', C['accent'])])

        style.configure('TEntry', fieldbackground='#FFFFFF', foreground='#000000', insertcolor='#000000')
        return C

    # ------------------------------------------------------------------ resize
    def handle_main_resize(self, event):
        w = max(240, int(event.width * 0.20))
        if w != self.left_panel.winfo_width():
            self.left_panel.config(width=w)
            self.right_panel.config(width=w + 20)

    def handle_board_resize(self, event):
        eval_h = max(self.eval_frame.winfo_height(), self.eval_frame.winfo_reqheight())
        nav_h  = max(self.navigation_frame.winfo_height(), self.navigation_frame.winfo_reqheight())
        vw, vh = event.width - 40, event.height - eval_h - nav_h - 35
        if vw <= 1 or vh <= 1:
            return
        new_sq = min(vw // COLS, vh // ROWS)
        bw = COLS * self.square_size
        self.eval_frame.config(width=bw)
        self.eval_bar_canvas.config(width=bw)
        if new_sq != self.square_size and new_sq > 0:
            self.square_size = new_sq
            bw = COLS * self.square_size
            self.canvas.config(width=bw, height=ROWS * self.square_size)
            self.eval_frame.config(width=bw)
            self.eval_bar_canvas.config(width=bw)
            self.board_image = self.create_board_image()
            self.draw_board()
        self._position_side_labels()

    def handle_key_press(self, event):
        if self.is_ai_thinking() and not self.analysis_thinking:
            return
        action = {'Left': self.undo_move, 'Right': self.redo_move,
                  'Home': self.go_to_start, 'End': self.go_to_end}.get(event.keysym)
        if action:
            action()

    def redraw_eval_bar_on_resize(self, event):
        self.draw_eval_bar(self.last_eval_score, self.last_eval_depth)

    # ------------------------------------------------------------------ flip / swap / mode
    def _update_flip_view_button_style(self):
        mode = self.game_mode.get()
        warn = (mode == GameMode.HUMAN_VS_BOT.value and self.board_orientation != self.human_color) or \
               (mode != GameMode.HUMAN_VS_BOT.value and self.board_orientation == "black")
        self.flip_view_btn.configure(style='Flipped.TButton' if warn else 'Control.TButton')

    def toggle_board_view(self):
        self.board_orientation = "black" if self.board_orientation == "white" else "white"
        self._update_flip_view_button_style()
        self.update_bot_labels()
        self.draw_board()

    def on_mode_changed(self):
        self._stop_ai_process()
        mode = self.game_mode.get()
        if mode == GameMode.HUMAN_VS_BOT.value:
            self.board_orientation = self.human_color
            if not self.game_over and self.turn != self.human_color:
                self.master.after(self._get_ai_move_delay(), self._make_game_ai_move)
        elif mode == GameMode.AI_VS_AI.value:
            if not self.game_over:
                self.master.after(self._get_ai_move_delay(), self._make_game_ai_move)
        else:
            self._update_analysis_after_state_change()
        self._update_flip_view_button_style()
        self.update_ui_after_state_change()

    def swap_sides(self):
        self._stop_ai_process()
        if self.game_mode.get() == GameMode.HUMAN_VS_BOT.value:
            self.human_color       = "black" if self.human_color == "white" else "white"
            self.board_orientation = self.human_color
            self._update_flip_view_button_style()
            self.update_ui_after_state_change()
            if not self.game_over and self.turn != self.human_color:
                print(f"Swapped sides. AI ({self.turn}) taking over...")
                self.master.after(self._get_ai_move_delay(), self._make_game_ai_move)

    def _reset_game_state_vars(self):
        self.full_history    = [(self.board.clone(), self.turn, None)]
        self.history_pointer = 0
        self.position_counts = {board_hash(self.board, self.turn): 1}
        self.game_over       = False
        self.game_result     = None
        self.last_eval_score = 0.0
        self.last_eval_depth = None
        self.draw_eval_bar(0)
        self.current_pv_raw  = []
        self.last_pv_message = None
        if hasattr(self, 'pv_text'):
            self.pv_text.config(state=tk.NORMAL)
            self.pv_text.delete(1.0, tk.END)
            self.pv_text.config(state=tk.DISABLED)

    # ------------------------------------------------------------------ FEN / PGN
    def get_current_fen(self):
        rows = []
        for r in range(ROWS):
            row, empty = "", 0
            for c in range(COLS):
                p = self.board.grid[r][c]
                if p is None:
                    empty += 1
                else:
                    if empty:
                        row += str(empty)
                        empty = 0
                    ch = _CLASS_TO_FEN_CHAR[type(p)]
                    row += ch if p.color == "white" else ch.lower()
            if empty:
                row += str(empty)
            rows.append(row)
        return "/".join(rows) + f" {'w' if self.turn == 'white' else 'b'} - - 0 1"

    def copy_fen_to_clipboard(self):
        fen = self.get_current_fen()
        self.fen_entry.delete(0, tk.END)
        self.fen_entry.insert(0, fen)
        self.master.clipboard_clear()
        self.master.clipboard_append(fen)

    def load_fen_from_entry(self):
        fen = self.fen_entry.get().strip()
        if not fen:
            return
        parts = fen.split()
        self._stop_ai_process()
        self.board = Board(setup=False)
        r = c = 0
        for ch in parts[0]:
            if ch == '/':
                r += 1; c = 0
            elif ch.isdigit():
                c += int(ch)
            else:
                pc = _FEN_CHAR_TO_CLASS.get(ch.lower())
                if pc:
                    self.board.add_piece(pc("white" if ch.isupper() else "black"), r, c)
                c += 1
        self.turn         = "white" if (parts[1] if len(parts) > 1 else 'w').lower() == 'w' else "black"
        self.game_started = True
        self._reset_clock_state()
        self.render_clocks()
        self._reset_game_state_vars()
        status, winner = get_game_state(self.board, self.turn, self.position_counts,
                                        self.history_pointer, self.MAX_GAME_MOVES)
        if status != "ongoing":
            self.game_over   = True
            self.game_result = (status, winner)
        self.board_orientation = self.human_color
        self._update_flip_view_button_style()
        self.update_ui_after_state_change()
        self._update_analysis_after_state_change()
        if not self.game_over and self.game_mode.get() == GameMode.HUMAN_VS_BOT.value \
                and self.turn != self.human_color:
            self.master.after(self._get_ai_move_delay(), self._make_game_ai_move)

    def get_current_pgn(self):
        moves      = []
        start_turn = self.full_history[0][1]
        for i in range(1, len(self.full_history)):
            m = self.full_history[i][2]
            if m:
                moves.append(format_move_san(self.full_history[i-1][0], self.full_history[i][0], m))
        pgn, move_num = "", 1
        if start_turn == 'black' and moves:
            pgn      += f"{move_num}... {moves[0]} "
            moves     = moves[1:]
            move_num += 1
        for i in range(0, len(moves), 2):
            w, b = moves[i], moves[i+1] if i+1 < len(moves) else None
            pgn += f"{move_num}. {w}, {b} " if b else f"{move_num}. {w} "
            move_num += 1
        if self.game_result:
            r = self.game_result[1]
            pgn += "1-0" if r == 'white' else "0-1" if r == 'black' else "1/2-1/2"
        else:
            pgn += "*"
        return pgn.strip()

    def copy_pgn_to_clipboard(self):
        pgn = self.get_current_pgn()
        self.pgn_entry.delete(0, tk.END)
        self.pgn_entry.insert(0, pgn)
        self.master.clipboard_clear()
        self.master.clipboard_append(pgn)

    def load_pgn_from_entry(self):
        pgn_text = self.pgn_entry.get().strip()
        if not pgn_text:
            return
        self.reset_game()
        self._pause_clock()
        self.last_clock_tick = None
        for res in ["1-0", "0-1", "1/2-1/2", "*"]:
            pgn_text = pgn_text.replace(res, "")
        pgn_text = re.sub(r'\d+\.+', '', pgn_text).replace(',', ' ')
        while pgn_text.strip():
            pgn_text = pgn_text.strip()
            san_map  = {}
            for m in get_all_legal_moves(self.board, self.turn):
                child = self.board.clone()
                child.make_move(m[0], m[1])
                san_map[format_move_san(self.board, child, m)] = m
            matched_move = matched_san = None
            for san in sorted(san_map, key=len, reverse=True):
                if pgn_text.startswith(san) and \
                        (len(pgn_text) == len(san) or pgn_text[len(san)].isspace()):
                    matched_move = san_map[san]
                    matched_san  = san
                    break
            if matched_move:
                self.board.make_move(matched_move[0], matched_move[1])
                self.execute_move_and_check_state(self.turn, matched_move)
                pgn_text = pgn_text[len(matched_san):]
                if self.game_over:
                    break
            else:
                messagebox.showwarning("PGN Error", f"Could not parse: {pgn_text[:20]}...")
                break
        self.last_clock_tick = time.time()

    # ------------------------------------------------------------------ move history UI
    def update_moves_list(self):
        self.moves_text.config(state=tk.NORMAL)
        self.moves_text.delete(1.0, tk.END)
        for tag in self.moves_text.tag_names():
            if tag.startswith("ply_"):
                self.moves_text.tag_delete(tag)

        formatted  = []
        start_turn = self.full_history[0][1]
        for i in range(1, len(self.full_history)):
            m = self.full_history[i][2]
            if m:
                formatted.append(format_move_san(self.full_history[i-1][0], self.full_history[i][0], m))

        pairs = []
        if start_turn == 'black' and formatted:
            pairs.append(["...", formatted[0]])
            formatted = formatted[1:]
        for i in range(0, len(formatted), 2):
            pairs.append([formatted[i], formatted[i+1] if i+1 < len(formatted) else ""])

        for i, pair in enumerate(pairs):
            self.moves_text.insert(tk.END, f"{i+1}.".ljust(4), "num")
            w_ptr = (i * 2) + 1 if start_turn == 'white' else (i * 2)
            b_ptr = w_ptr + 1
            w_tag, b_tag = f"ply_{w_ptr}", f"ply_{b_ptr}"
            self.moves_text.insert(tk.END, self._format_san_display(pair[0]).center(14), w_tag)
            self.moves_text.insert(
                tk.END,
                self._format_san_display(pair[1]).center(14) if pair[1] else " " * 14,
                b_tag if pair[1] else "")
            self.moves_text.insert(tk.END, "\n")
            if pair[0] != "...":
                self.moves_text.tag_bind(w_tag, "<Button-1>", lambda e, p=w_ptr: self._navigate_history(p))
                self.moves_text.tag_bind(w_tag, "<Enter>", lambda e: self.moves_text.config(cursor="hand2"))
                self.moves_text.tag_bind(w_tag, "<Leave>", lambda e: self.moves_text.config(cursor="arrow"))
            if pair[1]:
                self.moves_text.tag_bind(b_tag, "<Button-1>", lambda e, p=b_ptr: self._navigate_history(p))
                self.moves_text.tag_bind(b_tag, "<Enter>", lambda e: self.moves_text.config(cursor="hand2"))
                self.moves_text.tag_bind(b_tag, "<Leave>", lambda e: self.moves_text.config(cursor="arrow"))

        self.moves_text.tag_configure("num", foreground=self.COLORS['text_dark'])
        for tag in self.moves_text.tag_names():
            if tag.startswith("ply_"):
                self.moves_text.tag_configure(tag, background=self.COLORS['bg_medium'],
                                              foreground=self.COLORS['text_light'])
        if self.history_pointer > 0:
            atag = f"ply_{self.history_pointer}"
            self.moves_text.tag_configure(atag, background=self.COLORS['accent'],
                                          foreground=self.COLORS['text_light'])
            try:
                self.moves_text.see(f"{atag}.first")
            except tk.TclError:
                pass
        self.moves_text.config(state=tk.DISABLED)

    # ------------------------------------------------------------------ core gameplay
    def execute_move_and_check_state(self, player_who_moved, move):
        if self.use_clock_var.get() and not self.game_over and self.increment:
            if player_who_moved == 'white':
                self.white_time += self.increment
            else:
                self.black_time += self.increment
            self.render_clocks()
        self.switch_turn()
        self._start_clock()
        if self.history_pointer < len(self.full_history) - 1:
            self.full_history = self.full_history[:self.history_pointer + 1]
            self.position_counts.clear()
            for board, turn, _ in self.full_history:
                h = board_hash(board, turn)
                self.position_counts[h] = self.position_counts.get(h, 0) + 1
        self.full_history.append((self.board.clone(), self.turn, move))
        self.history_pointer += 1
        key = board_hash(self.board, self.turn)
        self.position_counts[key] = self.position_counts.get(key, 0) + 1
        status, winner = get_game_state(self.board, self.turn, self.position_counts,
                                        self.history_pointer, self.MAX_GAME_MOVES)
        if status != "ongoing":
            self.game_over   = True
            self.game_result = (status, winner)
        self.update_ui_after_state_change()
        if self.game_over:
            print(f"Game Over! Result: {self.game_result[0]}")
            self._stop_ai_process()
            if self.game_mode.get() == GameMode.AI_VS_AI.value and self.ai_series_running:
                self.process_ai_series_result()

    def _execute_ai_move(self, the_move):
        if self.game_over:
            return
        if the_move:
            self.board.make_move(the_move[0], the_move[1])
            self.execute_move_and_check_state(self.turn, the_move)
            if not self.game_over and self.game_mode.get() == GameMode.AI_VS_AI.value:
                self.master.after(self._get_ai_move_delay(), self._make_game_ai_move)
        else:
            print("AI reported no valid move.")
        self._stop_ai_process()
        self.update_bot_labels()
        self.set_interactivity(True)

    def on_drag_start(self, event):
        if self.game_over:
            return
        if self.is_ai_thinking() and not self.analysis_thinking:
            return
        r, c = self.canvas_to_board(event.x, event.y)
        if r == -1 or not self.board.grid[r][c]:
            return
        piece = self.board.grid[r][c]
        if piece.color != self.turn:
            return
        if self.game_mode.get() == GameMode.HUMAN_VS_BOT.value and self.turn != self.human_color:
            return
        self.selected  = (r, c)
        self.drag_start = (r, c)
        self.dragging  = True
        self.valid_moves = get_all_legal_moves(self.board, self.turn)
        self.valid_moves_for_highlight = [e for s, e in self.valid_moves if s == self.selected]
        self.drag_piece_ghost = self.canvas.create_text(
            event.x, event.y, text=piece.symbol(),
            font=("Arial Unicode MS", int(self.square_size * 0.7)),
            fill=self.turn, tags="drag_ghost")
        self.draw_board()
        self.canvas.tag_raise("drag_ghost")

    def on_drag_motion(self, event):
        if self.dragging:
            self.canvas.coords(self.drag_piece_ghost, event.x, event.y)

    def on_drag_end(self, event):
        is_analysis = self.is_ai_thinking() and self.analysis_thinking
        if self.is_ai_thinking() and not is_analysis:
            self.valid_moves = []
            self.draw_board()
            return
        if not self.dragging:
            self.valid_moves = []
            self.draw_board()
            return
        self.dragging = False
        self.canvas.delete("drag_ghost")
        row, col = self.canvas_to_board(event.x, event.y)
        if row == -1 or not self.drag_start:
            self.update_ui_after_state_change()
            self.set_interactivity(True)
            return
        start_pos, end_pos = self.drag_start, (row, col)
        if (start_pos, end_pos) in self.valid_moves:
            self.board.make_move(start_pos, end_pos)
            self.execute_move_and_check_state(self.turn, (start_pos, end_pos))
            if not self.game_over:
                mode = self.game_mode.get()
                if mode == GameMode.HUMAN_VS_BOT.value and self.turn != self.human_color:
                    self.drag_start = None
                    self.set_interactivity(False)
                    self.master.after(self._get_ai_move_delay(), self._make_game_ai_move)
                    return
                elif mode == GameMode.HUMAN_VS_HUMAN.value:
                    self._update_analysis_after_state_change()
        self.drag_start = None
        self.update_ui_after_state_change()
        self.set_interactivity(True)

    def reset_game(self):
        if self.game_mode.get() != GameMode.AI_VS_AI.value:
            self.ai_series_running = False
        self._stop_ai_process()
        self.board               = Board()
        self.turn                = "white"
        self.game_started        = False
        self.last_move_timestamp = time.time()
        self.selected            = None
        self.valid_moves         = []
        self._reset_game_state_vars()
        self._reset_clock_state()
        self._update_time_control_label()
        self.render_clocks()
        self._toggle_clock()
        self.fen_entry.delete(0, tk.END)
        self.pgn_entry.delete(0, tk.END)
        self.game_started = True
        mode, delay = self.game_mode.get(), self._get_ai_move_delay()
        if mode == GameMode.AI_VS_AI.value:
            self.white_playing_bot_type = "op" if (
                self.ai_series_running and self.ai_series_stats['game_count'] % 2 == 1) else "main"
            self.board_orientation = "white" if self.white_playing_bot_type == "main" else "black"
            if self.ai_series_running:
                self.apply_series_opening_move()
            if not self.game_over:
                self.master.after(delay, self._make_game_ai_move)
        elif mode == GameMode.HUMAN_VS_BOT.value:
            self.board_orientation = self.human_color
            if self.turn != self.human_color:
                self.master.after(delay, self._make_game_ai_move)
        else:
            self.board_orientation = "white"
        self.update_ui_after_state_change()
        self._update_analysis_after_state_change()

    def _make_game_ai_move(self):
        if self.game_over:
            return
        print(f"\n--- Turn {self.history_pointer + 1} ({self.turn.capitalize()}) ---")
        self.last_move_timestamp = time.time()
        mode      = self.game_mode.get()
        bot_class = bot_name = None
        if mode == GameMode.HUMAN_VS_BOT.value:
            if self.turn != self.human_color:
                bot_class, bot_name = ChessBot, self.MAIN_AI_NAME
        elif mode == GameMode.AI_VS_AI.value:
            main_color  = "white" if self.white_playing_bot_type == "main" else "black"
            bot_class, bot_name = (ChessBot, self.MAIN_AI_NAME) if self.turn == main_color \
                               else (OpponentAI, self.OPPONENT_AI_NAME)
        if bot_class:
            self._start_ai_process(bot_class, bot_name, self.bot_depth_slider.get())

    def update_ui_after_state_change(self):
        self.selected                  = None
        self.valid_moves               = []
        self.valid_moves_for_highlight = []
        self.update_turn_label()
        self.update_game_info_label()
        self.update_bot_labels()
        self.update_moves_list()
        self.draw_board()
        self.update_navigation_buttons()

    def _navigate_history(self, target_index):
        if self.game_mode.get() == GameMode.AI_VS_AI.value:
            return
        new_index = max(0, min(target_index, len(self.full_history) - 1))
        if new_index != self.history_pointer:
            self.history_pointer = new_index
            self._load_state_from_history()

    def _load_state_from_history(self):
        self._stop_ai_process()
        was_running = self._pause_clock()
        board_state, turn_state, _ = self.full_history[self.history_pointer]
        self.board       = board_state.clone()
        self.turn        = turn_state
        self.game_over   = False
        self.game_result = None
        self.position_counts.clear()
        for i in range(self.history_pointer + 1):
            b, t, _ = self.full_history[i]
            h = board_hash(b, t)
            self.position_counts[h] = self.position_counts.get(h, 0) + 1
        status, winner = get_game_state(self.board, self.turn, self.position_counts,
                                        self.history_pointer, self.MAX_GAME_MOVES)
        if status != "ongoing":
            self.game_over   = True
            self.game_result = (status, winner)
        if was_running and self.history_pointer == len(self.full_history) - 1 and not self.game_over:
            self.last_clock_tick = time.time()
            self.clock_running   = True
            self._tick_clock()
        self.update_ui_after_state_change()
        self._update_analysis_after_state_change()

    def update_game_info_label(self):
        text = {GameMode.HUMAN_VS_BOT.value:  f"Human vs {self.MAIN_AI_NAME}",
                GameMode.AI_VS_AI.value:       f"{self.MAIN_AI_NAME} vs {self.OPPONENT_AI_NAME}",
                GameMode.HUMAN_VS_HUMAN.value: "Human vs Human Analysis"}.get(self.game_mode.get())
        self.game_info_label.config(text=text)

    # ------------------------------------------------------------------ board drawing
    def create_board_image(self):
        if self.square_size <= 0:
            return None
        img = tk.PhotoImage(width=COLS * self.square_size, height=ROWS * self.square_size)
        C1, C2 = "#D2B48C", "#8B5A2B"
        for r in range(ROWS):
            for c in range(COLS):
                x1, y1 = c * self.square_size, r * self.square_size
                img.put(C1 if (r + c) % 2 == 0 else C2,
                        to=(x1, y1, x1 + self.square_size, y1 + self.square_size))
        return img

    def draw_board(self):
        if not self.board_image:
            return
        self.canvas.itemconfig(self.board_image_id, image=self.board_image)
        self.canvas.delete("highlight", "piece", "check_highlight", "border_highlight")
        mode = self.game_mode.get()
        warn = (mode == GameMode.HUMAN_VS_BOT.value and self.board_orientation != self.human_color) or \
               (mode != GameMode.HUMAN_VS_BOT.value and self.board_orientation == "black")
        if warn:
            w, h = COLS * self.square_size, ROWS * self.square_size
            self.canvas.create_rectangle(2, 2, w - 2, h - 2, outline=self.COLORS['warning'],
                                         width=4, tags="border_highlight")
        for r_m, c_m in getattr(self, 'valid_moves_for_highlight', []):
            x1, y1 = self.board_to_canvas(r_m, c_m)
            rd      = self.square_size // 5
            cx, cy  = x1 + self.square_size // 2, y1 + self.square_size // 2
            self.canvas.create_oval(cx - rd, cy - rd, cx + rd, cy + rd,
                                    fill="#1E90FF", outline="", tags="highlight")
        for r in range(ROWS):
            for c in range(COLS):
                piece = self.board.grid[r][c]
                if not piece:
                    continue
                if isinstance(piece, King):
                    is_lost = (self.game_over and self.game_result
                               and self.game_result[0] == "checkmate"
                               and self.game_result[1] != piece.color)
                    clr = "darkred" if is_lost else ("red" if is_in_check(self.board, piece.color) else None)
                    if clr:
                        x1, y1 = self.board_to_canvas(r, c)
                        self.canvas.create_rectangle(
                            x1, y1, x1 + self.square_size, y1 + self.square_size,
                            outline=clr, width=4, tags="check_highlight")
                if (r, c) != self.drag_start:
                    x, y  = self.board_to_canvas(r, c)
                    cx    = x + self.square_size // 2
                    cy    = y + self.square_size // 2 + 2
                    font  = ("Arial Unicode MS", int(self.square_size * 0.67))
                    sym   = piece.symbol()
                    self.canvas.create_text(cx + 1, cy + 1, text=sym, font=font,
                                            fill="#888888", tags="piece")
                    self.canvas.create_text(cx, cy, text=sym, font=font,
                                            fill="#000000" if piece.color == "black" else "#FFFFFF",
                                            tags="piece")
        self._position_side_labels()

    def _position_side_labels(self):
        if not hasattr(self, "canvas"):
            return
        self.board_row_frame.update_idletasks()
        cx, cy  = self.canvas.winfo_x(), self.canvas.winfo_y()
        label_w = min(max(96, int(self.square_size * 1.9)), max(1, cx - 10))
        if label_w < 20:
            self.top_bot_label.place_forget()
            self.bottom_bot_label.place_forget()
            return
        for lbl in (self.top_bot_label, self.bottom_bot_label):
            lbl.config(wraplength=max(1, label_w - 6), anchor="center", justify=tk.CENTER)
        lx = 2 + max(0, (max(3, cx - 8) - 2 - label_w) // 2)
        bh = ROWS * self.square_size
        self.top_bot_label   .place(in_=self.board_row_frame, x=lx, y=cy + 4,      width=label_w, anchor="nw")
        self.bottom_bot_label.place(in_=self.board_row_frame, x=lx, y=cy + bh - 4, width=label_w, anchor="sw")

    def draw_eval_bar(self, eval_score, depth=None):
        score = eval_score / 100.0
        w, h  = self.eval_bar_canvas.winfo_width(), self.eval_bar_canvas.winfo_height()
        if w <= 1 or h <= 1:
            return
        if w != self.last_eval_bar_w or h != self.last_eval_bar_h:
            self.eval_bar_canvas.delete("gradient")
            for x_px in range(w):
                i = int(255 * x_px / (w - 1))
                self.eval_bar_canvas.create_line(x_px, 0, x_px, h,
                                                 fill=f"#{i:02x}{i:02x}{i:02x}", tags="gradient")
            self.last_eval_bar_w, self.last_eval_bar_h = w, h
        self.eval_bar_canvas.delete("marker")
        mx = int(((max(-1.0, min(1.0, math.tanh(score / 10.0))) + 1) / 2.0) * w)
        self.eval_bar_canvas.create_line(mx,   0, mx,   h, fill="#FF0000", width=3, tags="marker")
        self.eval_bar_canvas.create_line(w//2, 0, w//2, h, fill="#00FF00", width=2, tags="marker")
        sfx = f" (D{depth})" if depth is not None else ""
        self.eval_score_label.config(text=f"Even{sfx}" if abs(score) < 0.05
                                     else f"{'+' if score > 0 else ''}{score:.2f}{sfx}")

    # ------------------------------------------------------------------ comm queue / PV
    def process_comm_queue(self):
        try:
            while not self.comm_queue.empty():
                msg  = self.comm_queue.get_nowait()
                kind = msg[0]
                if kind == 'log':
                    print(msg[1])
                    if self.auto_save_stats_var.get() and self.game_mode.get() == GameMode.AI_VS_AI.value:
                        m = re.search(r'> (.*?) \(D(\d+|TB)\).*?Time=([0-9.]+)s', msg[1])
                        if m:
                            for cat in (m.group(1).strip(), 'Global'):
                                self.depth_stats.setdefault(cat, {}).setdefault(
                                    m.group(2), []).append(float(m.group(3)))
                elif kind == 'eval':
                    self.last_eval_score, self.last_eval_depth = msg[1], msg[2]
                    self.draw_eval_bar(msg[1], msg[2])
                elif kind == 'pv':
                    self.last_pv_message = msg
                    self._render_pv()
                elif kind == 'move':
                    msg_task_id = msg[2] if len(msg) > 2 else -1
                    
                    # Only accept the move if it matches the current generation ID
                    if self.active_worker_name is not None and msg_task_id == self.current_task_id:
                        self.active_worker_name = None
                        self.analysis_thinking  = False
                        self._execute_ai_move(msg[1])
        except Exception:
            pass
        finally:
            self.master.after(20, self.process_comm_queue)

    def _render_pv(self):
        if not getattr(self, 'show_pv_var', None) or not self.show_pv_var.get():
            self.pv_text.pack_forget()
            return
        self.pv_text.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=10)
        msg = getattr(self, 'last_pv_message', None)
        if not msg:
            return
        score, depth, pv_san_list, pv_raw = msg[1], msg[2], msg[3], msg[4]
        self.current_pv_raw = pv_raw
        self.current_pv_san = pv_san_list
        if score > 990000:
            sd = f"+M{(1000000 - score + 1) // 2}"
        elif score < -990000:
            sd = f"-M{(score + 1000000 + 1) // 2}"
        else:
            sd = f"{score / 100:+.2f}"
        self.pv_text.config(state=tk.NORMAL)
        self.pv_text.delete(1.0, tk.END)
        self.pv_text.insert(tk.END, f"[{sd}] (D{depth}):\n")
        for i, (san, _) in enumerate(zip(pv_san_list, pv_raw)):
            tag = f"pv_move_{i}"
            self.pv_text.insert(tk.END, self._format_san_display(san) + " ", tag)
            self.pv_text.tag_bind(tag, "<Enter>", lambda e, idx=i, t=tag: self.on_pv_hover_enter(e, idx, t))
            self.pv_text.tag_bind(tag, "<Leave>", lambda e, t=tag: self.on_pv_hover_leave(e, t))
        self.pv_text.config(state=tk.DISABLED)

    # ------------------------------------------------------------------ AI process
    def _start_ai_process(self, bot_class, bot_name, search_depth):
        """Submit a task to the appropriate persistent worker."""
        if self.active_worker_name is not None:
            return   

        time_left = (self.white_time if self.turn == 'white' else self.black_time) \
                    if self.use_clock_var.get() else None
        inc = self.increment if self.use_clock_var.get() else None

        self.current_task_id += 1  # Increment task generation

        task = {
            'board':            self.board.clone(),
            'color':            self.turn,
            'position_counts':  self.position_counts.copy(),
            'bot_name':         bot_name,
            'ply_count':        self.history_pointer,
            'game_mode':        self.game_mode.get(),
            'search_depth':     search_depth,
            'time_left':        time_left,
            'increment':        inc,
            'use_opening_book': self.use_opening_book_var.get(),
            'task_id':          self.current_task_id  # Pass it to the worker
        }

        self.analysis_thinking = (bot_name == self.ANALYSIS_AI_NAME)

        if bot_class is ChessBot:
            self.active_worker_name = 'main'
            self.main_work_queue.put(task)
        else:
            self.active_worker_name = 'op'
            self.op_work_queue.put(task)

        if not self.analysis_thinking:
            self.set_interactivity(False)
        self.update_bot_labels()

    def _stop_ai_process(self):
        """Cancel the current task (worker stays alive)."""
        if self.active_worker_name == 'main':
            self.main_cancel_event.set()
        elif self.active_worker_name == 'op':
            self.op_cancel_event.set()

        self.active_worker_name = None
        self.analysis_thinking  = False

        # Drain any messages already in the queue from the cancelled task.
        while not self.comm_queue.empty():
            try:
                self.comm_queue.get_nowait()
            except Exception:
                break

        if self.game_mode.get() == GameMode.HUMAN_VS_HUMAN.value and not self.analysis_mode_var.get():
            self.last_eval_score, self.last_eval_depth = 0.0, None
            self.draw_eval_bar(0)
            self.eval_score_label.config(text="Even")

        self.set_interactivity(True)
        self.update_bot_labels()

    def _update_analysis_after_state_change(self):
        self._stop_ai_process()
        if self.analysis_mode_var.get() and \
                self.game_mode.get() == GameMode.HUMAN_VS_HUMAN.value and not self.game_over:
            fullmove = (self.history_pointer + 1) // 2 + 1
            print(f"\n--- Analysis: Move {fullmove}, Ply {self.history_pointer}, {self.turn.capitalize()} ---")
            self.master.after(50, lambda: self._start_ai_process(ChessBot, self.ANALYSIS_AI_NAME, 99))

    # ------------------------------------------------------------------ misc helpers
    def _update_time_control_label(self):
        t = int(self.time_control_seconds.get())
        self.time_control_label.config(text=f"Time Control: {t // 60:02d}:{t % 60:02d}")

    def _toggle_clock(self):
        if self.use_clock_var.get():
            self.clock_frame.pack(after=self.turn_label, fill=tk.X, pady=(5, 5))
            self.time_control_frame.pack(after=self.clock_frame, fill=tk.X, pady=(5, 5))
            self._update_time_control_label()
            self.render_clocks()
            if self.game_started and not self.game_over and self.history_pointer > 0:
                self._start_clock()
        else:
            self.clock_frame.pack_forget()
            self.time_control_frame.pack_forget()
            self._pause_clock()
            self.last_clock_tick = None

    def _get_ai_move_delay(self):
        return 0 if self.use_clock_var.get() else (4 if self.instant_move.get() else 20)

    def render_clocks(self):
        if not self.use_clock_var.get():
            return

        def fmt(t):
            t = max(0, t)
            return f"{int(t) // 60:02d}:{int(t) % 60:02d}.{int((t - int(t)) * 10)}"

        self.white_clock_lbl.config(text=f"W: {fmt(self.white_time)}")
        self.black_clock_lbl.config(text=f"B: {fmt(self.black_time)}")
        if not self.game_over:
            self.white_clock_lbl.config(
                bg=self.COLORS['accent']     if self.turn == 'white' else self.COLORS['bg_light'],
                fg=self.COLORS['text_light'] if self.turn == 'white' else self.COLORS['text_dark'])
            self.black_clock_lbl.config(
                bg=self.COLORS['accent']     if self.turn == 'black' else self.COLORS['bg_medium'],
                fg=self.COLORS['text_light'] if self.turn == 'black' else self.COLORS['text_dark'])
        else:
            self.white_clock_lbl.config(bg=self.COLORS['bg_light'],  fg=self.COLORS['text_light'])
            self.black_clock_lbl.config(bg=self.COLORS['bg_medium'], fg=self.COLORS['text_light'])

    def _tick_clock(self):
        if not self.use_clock_var.get() or not self.clock_running or self.game_over:
            self.clock_running = False
            return
        now              = time.time()
        elapsed          = now - self.last_clock_tick
        self.last_clock_tick = now
        if self.turn == 'white':
            self.white_time -= elapsed
        else:
            self.black_time -= elapsed
        timed_out = (self.turn == 'white' and self.white_time <= 0) or \
                    (self.turn == 'black' and self.black_time  <= 0)
        if timed_out:
            if self.turn == 'white': self.white_time = 0
            else:                    self.black_time  = 0
            self.handle_timeout(self.turn)
            return
        self.render_clocks()
        self.master.after(25, self._tick_clock)

    def handle_timeout(self, color):
        self.game_over     = True
        self.clock_running = False
        self.game_result   = ('timeout', 'black' if color == 'white' else 'white')
        self.update_ui_after_state_change()
        print("Game Over! Result: timeout")
        self._stop_ai_process()
        if self.game_mode.get() == GameMode.AI_VS_AI.value and self.ai_series_running:
            self.process_ai_series_result()

    def update_turn_label(self):
        self.turn_label.config(text=f"GAME OVER: {self.game_result[0].upper()}"
                               if self.game_result else f"TURN: {self.turn.upper()}")

    def update_bot_labels(self):
        mode = self.game_mode.get()
        if mode == GameMode.AI_VS_AI.value:
            wl = self.MAIN_AI_NAME     if self.white_playing_bot_type == "main" else self.OPPONENT_AI_NAME
            bl = self.OPPONENT_AI_NAME if self.white_playing_bot_type == "main" else self.MAIN_AI_NAME
        elif mode == GameMode.HUMAN_VS_BOT.value:
            wl = "Human" if self.human_color == "white" else self.MAIN_AI_NAME
            bl = "Human" if self.human_color == "black" else self.MAIN_AI_NAME
        else:
            wl, bl = "White", "Black"
        if self.turn == "white":
            wl += "\n(to move)"
        else:
            bl += "\n(to move)"
        bottom, top = (wl, bl) if self.board_orientation == 'white' else (bl, wl)
        self.bottom_bot_label.config(text=bottom)
        self.top_bot_label   .config(text=top)
        self._position_side_labels()

    def set_interactivity(self, on):
        if on:
            self.canvas.bind("<Button-1>",        self.on_drag_start)
            self.canvas.bind("<B1-Motion>",       self.on_drag_motion)
            self.canvas.bind("<ButtonRelease-1>", self.on_drag_end)
        else:
            self.canvas.unbind("<Button-1>")
            self.canvas.unbind("<B1-Motion>")
            self.canvas.unbind("<ButtonRelease-1>")

    def is_ai_thinking(self):
        return self.active_worker_name is not None

    def switch_turn(self):
        if not self.game_over:
            self.turn = "black" if self.turn == "white" else "white"

    def board_to_canvas(self, r, c):
        if self.board_orientation == "black":
            return (COLS - 1 - c) * self.square_size, (ROWS - 1 - r) * self.square_size
        return c * self.square_size, r * self.square_size

    def canvas_to_board(self, x, y):
        if self.board_orientation == "black":
            c = (COLS - 1) - x // self.square_size
            r = (ROWS - 1) - y // self.square_size
        else:
            c = x // self.square_size
            r = y // self.square_size
        return (r, c) if 0 <= r < ROWS and 0 <= c < COLS else (-1, -1)

    def undo_move(self):   self._navigate_history(self.history_pointer - 1)
    def redo_move(self):   self._navigate_history(self.history_pointer + 1)
    def go_to_start(self): self._navigate_history(0)
    def go_to_end(self):   self._navigate_history(len(self.full_history) - 1)

    def update_navigation_buttons(self):
        if self.game_mode.get() == GameMode.AI_VS_AI.value:
            for b in (self.start_button, self.undo_button, self.redo_button, self.end_button):
                b.config(state=tk.DISABLED)
            return
        can_back = self.history_pointer > 0
        can_fwd  = self.history_pointer < len(self.full_history) - 1
        self.start_button.config(state=tk.NORMAL if can_back else tk.DISABLED)
        self.undo_button .config(state=tk.NORMAL if can_back else tk.DISABLED)
        self.redo_button .config(state=tk.NORMAL if can_fwd  else tk.DISABLED)
        self.end_button  .config(state=tk.NORMAL if can_fwd  else tk.DISABLED)

    # ------------------------------------------------------------------ AI series
    def process_ai_series_result(self):
        self.ai_series_stats['game_count'] += 1
        _, wc = self.game_result
        if wc:
            main_color = 'white' if self.white_playing_bot_type == 'main' else 'black'
            self.ai_series_stats['my_ai_wins' if wc == main_color else 'op_ai_wins'] += 1
        else:
            self.ai_series_stats['draws'] += 1
        self.update_scoreboard()
        if self.auto_save_stats_var.get():
            self.save_depth_stats_to_file()
        if self.ai_series_running and self.ai_series_stats['game_count'] < self.AI_SERIES_GAMES:
            self.master.after(1000, self.reset_game)
        else:
            self.ai_series_running = False
            self.turn_label.config(text="AI SERIES COMPLETE!")

    def start_ai_series(self):
        self._stop_ai_process()
        self.game_mode.set(GameMode.AI_VS_AI.value)
        self.ai_series_stats          = {'game_count': 0, 'my_ai_wins': 0, 'op_ai_wins': 0, 'draws': 0}
        self.depth_stats              = {}
        self.ai_series_running        = True
        self.current_opening_sequence = []
        self.update_scoreboard()
        self.reset_game()

    def apply_series_opening_move(self):
        if self.ai_series_stats['game_count'] % 2 == 0:
            print("\n--- Generating new 2-ply opening sequence ---")
            self.current_opening_sequence = []
            temp_board = self.board.clone()
            temp_turn  = "white"
            for _ in range(2):
                moves = get_all_legal_moves(temp_board, temp_turn)
                if not moves:
                    break
                move = random.choice(moves)
                self.current_opening_sequence.append(move)
                temp_board.make_move(move[0], move[1])
                temp_turn = "black" if temp_turn == "white" else "white"
        self._pause_clock()
        for move in self.current_opening_sequence:
            child = self.board.clone()
            child.make_move(move[0], move[1])
            print(f"Opening: {format_move_san(self.board, child, move)}")
            self.board.make_move(move[0], move[1])
            self.execute_move_and_check_state(self.turn, move)
            if self.game_over:
                break
        self.last_clock_tick = time.time()
        self.clock_running   = False

    def update_scoreboard(self):
        if self.game_mode.get() == GameMode.AI_VS_AI.value and self.ai_series_running:
            s = self.ai_series_stats
            self.scoreboard_label.config(text=(
                f"{self.MAIN_AI_NAME} vs {self.OPPONENT_AI_NAME} "
                f"({s['game_count']}/{self.AI_SERIES_GAMES} games)\n"
                f"  {self.MAIN_AI_NAME}: {s['my_ai_wins']}  "
                f"{self.OPPONENT_AI_NAME}: {s['op_ai_wins']}  Draws: {s['draws']}"))
        else:
            self.scoreboard_label.config(text="")

    def save_depth_stats_to_file(self):
        if not self.depth_stats or not self.depth_stats.get('Global'):
            return
        sort_key = lambda k: int(k) if k.isdigit() else 999
        try:
            with open("AI_Series_Depth_Averages.txt", "w") as f:
                s = self.ai_series_stats
                f.write(f"=== AI vs OP Series Depth Stats ===\n"
                        f"Games: {s['game_count']}  Score: {self.MAIN_AI_NAME} "
                        f"{s['my_ai_wins']}-{s['op_ai_wins']} {self.OPPONENT_AI_NAME} "
                        f"(Draws: {s['draws']})\n\n")
                for cat in ['Global', self.MAIN_AI_NAME, self.OPPONENT_AI_NAME]:
                    data = self.depth_stats.get(cat)
                    if not data:
                        continue
                    f.write(f"--- {cat} ---\n")
                    for d in sorted(data, key=sort_key):
                        ts = data[d]
                        f.write(f"  D{d:<3} Avg:{sum(ts)/len(ts):.3f}s  Max:{max(ts):.3f}s  N:{len(ts)}\n")
                    f.write("\n")
        except Exception as e:
            print(f"Failed to save stats: {e}")

    # ------------------------------------------------------------------ PV hover mini-board
    def on_pv_hover_enter(self, event, move_idx, tag):
        self.pv_text.tag_config(tag, background=self.COLORS['accent'],
                                foreground=self.COLORS['text_light'])
        if getattr(self, 'pv_tooltip', None):
            self.pv_tooltip.destroy()
        self.pv_tooltip = tk.Toplevel(self.master)
        self.pv_tooltip.wm_overrideredirect(True)
        self.pv_tooltip.wm_geometry(f"+{event.x_root + 15}+{event.y_root - ROWS * 25 - 20}")
        self.tt_sq_size = 25
        self.tt_canvas  = tk.Canvas(self.pv_tooltip,
                                    width=COLS * self.tt_sq_size, height=ROWS * self.tt_sq_size,
                                    bg=self.COLORS['bg_medium'], highlightthickness=2,
                                    highlightbackground=self.COLORS['accent'])
        self.tt_canvas.pack()
        tt_sim = self.board.clone()
        for i in range(move_idx + 1):
            tt_sim.make_move(*self.current_pv_raw[i])
        self._draw_tt_board_static(tt_sim, self.current_pv_raw[move_idx])

    def on_pv_hover_leave(self, event, tag):
        self.pv_text.tag_config(tag, background="", foreground="")
        if getattr(self, 'pv_tooltip', None):
            self.pv_tooltip.destroy()
            self.pv_tooltip = None

    def _draw_tt_board_static(self, sim_board, last_move):
        self.tt_canvas.delete("all")
        sq     = self.tt_sq_size
        C1, C2 = "#D2B48C", "#8B5A2B"
        for r in range(ROWS):
            for c in range(COLS):
                dr = (ROWS - 1 - r) if self.board_orientation == "black" else r
                dc = (COLS - 1 - c) if self.board_orientation == "black" else c
                x1, y1 = dc * sq, dr * sq
                self.tt_canvas.create_rectangle(x1, y1, x1 + sq, y1 + sq,
                                                fill=C1 if (r + c) % 2 == 0 else C2, outline="")
                if last_move and (r, c) in last_move:
                    self.tt_canvas.create_rectangle(x1, y1, x1 + sq, y1 + sq,
                                                    fill="#F0E68C", stipple="gray50", outline="")
                piece = sim_board.grid[r][c]
                if piece:
                    font = ("Arial Unicode MS", int(sq * 0.7))
                    sym  = piece.symbol()
                    self.tt_canvas.create_text(x1 + sq // 2 + 1, y1 + sq // 2 + 2,
                                               text=sym, font=font, fill="#888888")
                    self.tt_canvas.create_text(x1 + sq // 2, y1 + sq // 2 + 1,
                                               text=sym, font=font,
                                               fill="#000" if piece.color == "black" else "#FFF")


if __name__ == "__main__":
    mp.freeze_support()
    root = tk.Tk()
    app  = EnhancedChessApp(root)
    root.mainloop()