# JungleChessUI.py (v14 - New time management)

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

# Compiled once at module level — used by _format_san_display
_CASUALTIES_RE = re.compile(r'\s*\(.*?\)')

# FEN character → piece class (shared by load_fen and get_current_fen)
_FEN_CHAR_TO_CLASS = {
    'p': Pawn, 'n': Knight, 'b': Bishop, 'r': Rook, 'q': Queen, 'k': King,
}
_CLASS_TO_FEN_CHAR = {
    Pawn: 'P', Knight: 'N', Bishop: 'B', Rook: 'R', Queen: 'Q', King: 'K',
}

def run_ai_process(board, color, position_counts, comm_queue, cancellation_event,
                   bot_class, bot_name, search_depth, ply_count, game_mode,
                   time_left=None, increment=None):
    try:
        # Pass time controls down to the new AI
        bot = bot_class(board, color, position_counts, comm_queue, cancellation_event,
                        bot_name, ply_count, game_mode, time_left=time_left, increment=increment)
    except TypeError:
        # Fallback just in case you haven't updated OpponentAI.py yet!
        bot = bot_class(board, color, position_counts, comm_queue, cancellation_event,
                        bot_name, ply_count, game_mode)
    bot.search_depth = search_depth
    if search_depth == 99:
        bot.ponder_indefinitely()
    else:
        bot.make_move()


class EnhancedChessApp:
    MAIN_AI_NAME     = "AI Bot"
    OPPONENT_AI_NAME = "OP Bot"
    ANALYSIS_AI_NAME = "Analysis"
    slidermaxvalue   = 12
    MAX_GAME_MOVES   = 200

    def __init__(self, master):
        self.master = master
        self.master.title("Jungle Chess")
        random.seed()

        self.comm_queue            = mp.Queue()
        self.ai_process            = None
        self.ai_cancellation_event = mp.Event()

        self.board = Board()

        self.turn             = "white"
        self.selected         = None
        self.valid_moves      = []
        self.game_over        = False
        self.game_result      = None
        self.dragging         = False
        self.drag_piece_ghost = None
        self.drag_start       = None

        self.full_history    = []
        self.history_pointer = -1
        self.position_counts = {}

        self.current_opening_sequence = []
        self.square_size       = 75
        self.base_sidebar_width = 280

        self.game_mode           = tk.StringVar(value=GameMode.HUMAN_VS_BOT.value)
        self.analysis_mode_var   = tk.BooleanVar(value=True)
        self.ai_series_running   = False
        self.ai_series_stats     = {'game_count': 0, 'my_ai_wins': 0, 'op_ai_wins': 0, 'draws': 0}
        self.depth_stats         = {}
        self.auto_save_stats_var = tk.BooleanVar(value=True)
        self.show_pv_var         = tk.BooleanVar(value=True)
        self.long_notation_var   = tk.BooleanVar(value=False)

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
        self.white_time = 0.0
        self.black_time = 0.0
        self.increment = 0.0
        self.last_clock_tick = None
        self.clock_running = False
        # ------------------

        self.COLORS = self.setup_styles()
        self.master.configure(bg=self.COLORS['bg_dark'])
        self.build_ui()

        self.master.bind("<Key>", self.handle_key_press)
        self.process_comm_queue()
        self.reset_game()

    # ------------------------------------------------------------------ helpers
    def _format_san_display(self, san_str):
        """Strip casualty brackets when long notation is off."""
        if self.long_notation_var.get() or not san_str:
            return san_str
        return _CASUALTIES_RE.sub('', san_str)

    def _on_notation_toggle(self):
        self.update_moves_list()
        self._render_pv()

    # ------------------------------------------------------------------ UI build
    def build_ui(self):
        screen_w, screen_h = self.master.winfo_screenwidth(), self.master.winfo_screenheight()
        self.master.geometry(f"{screen_w}x{screen_h}+0+0")
        self.master.state('zoomed')

        self.main_frame = ttk.Frame(self.master, style='Left.TFrame')
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- LEFT PANEL ---
        self.left_panel = ttk.Frame(self.main_frame, style='Left.TFrame')
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        self.left_panel.pack_propagate(False)

        self.title_label = ttk.Label(
            self.left_panel, text="JUNGLE CHESS", style='Header.TLabel',
            font=('Helvetica', 22, 'bold'))
        self.title_label.pack(pady=(0, 15))

        self.pv_text = tk.Text(
            self.left_panel, height=6,
            bg=self.COLORS['bg_medium'], fg=self.COLORS['text_light'],
            font=('Helvetica', 10), wrap=tk.WORD, borderwidth=1, relief="solid")
        self.pv_text.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=10)
        self.pv_text.config(state=tk.DISABLED)

        self._build_control_widgets(self.left_panel)

        # --- CENTER PANEL ---
        self.center_panel = ttk.Frame(self.main_frame, style='Right.TFrame')
        self.center_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.board_column = ttk.Frame(self.center_panel, style='Right.TFrame')
        self.board_column.pack(expand=True, fill=tk.BOTH)

        self.eval_frame = ttk.Frame(
            self.board_column, style='Right.TFrame',
            width=COLS * self.square_size, height=58)
        self.eval_frame.pack(side=tk.TOP, anchor=tk.CENTER, pady=(6, 5))
        self.eval_frame.pack_propagate(False)

        self.eval_score_label = ttk.Label(
            self.eval_frame, text="Even", style='Status.TLabel', anchor="center")
        self.eval_score_label.pack(side=tk.TOP, pady=(0, 4))

        self.eval_bar_canvas = tk.Canvas(
            self.eval_frame, width=COLS * self.square_size, height=20,
            bg=self.COLORS['bg_light'], highlightthickness=1,
            highlightbackground=self.COLORS['text_dark'])
        self.eval_bar_canvas.pack(side=tk.TOP, anchor=tk.CENTER)
        self.eval_bar_canvas.bind("<Configure>", self.redraw_eval_bar_on_resize)

        self.board_row_frame = ttk.Frame(self.board_column, style='Right.TFrame')
        self.board_row_frame.pack(expand=True, fill=tk.BOTH)

        self.canvas_frame = ttk.Frame(self.board_row_frame, style='Canvas.TFrame')
        self.canvas_frame.pack(expand=True, fill=tk.BOTH)

        self.canvas = tk.Canvas(
            self.canvas_frame,
            width=COLS * self.square_size, height=ROWS * self.square_size,
            bg=self.COLORS['bg_medium'], highlightthickness=0)
        self.board_image_white = self.create_board_image("white")
        self.board_image_black = self.create_board_image("black")
        self.board_image_id = self.canvas.create_image(0, 0, anchor='nw', tags="board")
        self.canvas.pack(expand=True)

        self.top_bot_label = ttk.Label(
            self.board_row_frame, text="",
            font=("Helvetica", 11, "bold"),
            background=self.COLORS['bg_medium'], foreground=self.COLORS['text_light'],
            anchor="center", justify=tk.CENTER)
        self.bottom_bot_label = ttk.Label(
            self.board_row_frame, text="",
            font=("Helvetica", 11, "bold"),
            background=self.COLORS['bg_medium'], foreground=self.COLORS['text_light'],
            anchor="center", justify=tk.CENTER)

        # Navigation bar 
        self.navigation_frame = ttk.Frame(self.center_panel, style='Right.TFrame')
        self.navigation_frame.pack(fill=tk.X, pady=(5, 10))

        self.start_button = ttk.Button(self.navigation_frame, text="«", command=self.go_to_start,  style='Nav.TButton', state=tk.DISABLED)
        self.undo_button  = ttk.Button(self.navigation_frame, text="‹", command=self.undo_move,    style='Nav.TButton', state=tk.DISABLED)
        self.redo_button  = ttk.Button(self.navigation_frame, text="›", command=self.redo_move,    style='Nav.TButton', state=tk.DISABLED)
        self.end_button   = ttk.Button(self.navigation_frame, text="»", command=self.go_to_end,    style='Nav.TButton', state=tk.DISABLED)

        self.navigation_frame.columnconfigure(0, weight=1)
        self.navigation_frame.columnconfigure(5, weight=1)
        self.start_button.grid(row=0, column=1, padx=5)
        self.undo_button .grid(row=0, column=2, padx=5)
        self.redo_button .grid(row=0, column=3, padx=5)
        self.end_button  .grid(row=0, column=4, padx=5)

        # --- RIGHT PANEL ---
        self.right_panel = ttk.Frame(self.main_frame, style='Left.TFrame')
        self.right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        self.right_panel.pack_propagate(False)
        self._build_right_sidebar_widgets(self.right_panel)

        self.main_frame.bind("<Configure>", self.handle_main_resize)
        self.center_panel.bind("<Configure>", self.handle_board_resize)

    def _build_control_widgets(self, parent_frame):
        self.game_mode_frame = ttk.Frame(parent_frame, style='Left.TFrame')
        self.game_mode_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(self.game_mode_frame, text="GAME MODE", style='Header.TLabel').pack(anchor=tk.W)
        for mode in GameMode:
            ttk.Radiobutton(
                self.game_mode_frame,
                text=mode.name.replace("_", " ").title(),
                variable=self.game_mode, value=mode.value,
                command=self.on_mode_changed, style='Custom.TRadiobutton'
            ).pack(anchor=tk.W, pady=(2, 0))

        self.controls_frame = ttk.Frame(parent_frame, style='Left.TFrame')
        self.controls_frame.pack(fill=tk.X, pady=10)

        ttk.Button(self.controls_frame, text="NEW GAME",       command=self.reset_game,        style='Control.TButton').pack(fill=tk.X, pady=3)
        ttk.Button(self.controls_frame, text="SWAP SIDES",     command=self.swap_sides,        style='Control.TButton').pack(fill=tk.X, pady=3)
        self.flip_view_btn = ttk.Button(self.controls_frame, text="FLIP VIEW", command=self.toggle_board_view, style='Control.TButton')
        self.flip_view_btn.pack(fill=tk.X, pady=3)
        ttk.Button(self.controls_frame, text="AI vs OP Series", command=self.start_ai_series,  style='Control.TButton').pack(fill=tk.X, pady=3)

        ttk.Label(self.controls_frame, text="Depth:", style='SmallHeader.TLabel').pack(anchor=tk.W, pady=(10, 0))
        self.bot_depth_slider = tk.Scale(
            self.controls_frame, from_=1, to=self.slidermaxvalue, orient=tk.HORIZONTAL,
            bg=self.COLORS['bg_dark'], fg=self.COLORS['text_light'],
            highlightthickness=0, relief='flat')
        self.bot_depth_slider.set(ChessBot.search_depth)
        self.bot_depth_slider.pack(fill=tk.X, pady=(0, 5))

        self.instant_move = tk.BooleanVar(value=False)
        checkboxes = [
            ("Instant Moves",               self.instant_move,          None),
            ("Analysis Mode (H-vs-H)",      self.analysis_mode_var,     self._update_analysis_after_state_change),
            ("Auto-save Depth Stats",       self.auto_save_stats_var,   None),
            ("Show Engine Lines (PV)",      self.show_pv_var,           self._render_pv),
            ("Long Notation (Casualties)",  self.long_notation_var,     self._on_notation_toggle),
        ]
        for text, var, cmd in checkboxes:
            kw = {'command': cmd} if cmd else {}
            ttk.Checkbutton(
                self.controls_frame, text=text, variable=var,
                style='Custom.TCheckbutton', **kw
            ).pack(anchor=tk.W, pady=(2, 2))

    def _build_right_sidebar_widgets(self, parent_frame):
        # 1. Info Frame
        self.info_frame = ttk.Frame(parent_frame, style='Left.TFrame')
        self.info_frame.pack(fill=tk.X, pady=(0, 5))
        self.game_info_label = ttk.Label(self.info_frame, text="Match Info", style='Header.TLabel')
        self.game_info_label.pack(anchor=tk.W)
        self.turn_label = ttk.Label(self.info_frame, text="WHITE'S TURN", style='Status.TLabel')
        self.turn_label.pack(fill=tk.X, pady=(5, 5))

        # --- CLOCK UI ---
        self.clock_frame = ttk.Frame(self.info_frame, style='Left.TFrame')
        self.clock_frame.pack(fill=tk.X, pady=(5, 5))

        self.black_clock_lbl = tk.Label(
            self.clock_frame, text="00:00.0", font=('Courier', 20, 'bold'),
            bg=self.COLORS['bg_medium'], fg=self.COLORS['text_light'], pady=4)
        self.black_clock_lbl.pack(side=tk.TOP, fill=tk.X, pady=2)

        self.white_clock_lbl = tk.Label(
            self.clock_frame, text="00:00.0", font=('Courier', 20, 'bold'),
            bg=self.COLORS['bg_light'], fg=self.COLORS['text_light'], pady=4)
        self.white_clock_lbl.pack(side=tk.BOTTOM, fill=tk.X, pady=2)
        # ----------------

        # --- TIME CONTROL SLIDER ---
        self.time_control_frame = ttk.Frame(self.info_frame, style='Left.TFrame')
        self.time_control_frame.pack(fill=tk.X, pady=(5, 5))

        self.time_control_label = ttk.Label(
            self.time_control_frame, text="Time Control: 05:00",
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
        # ---------------------------

        # 2. Move History
        ttk.Label(parent_frame, text="Move History", style='SmallHeader.TLabel').pack(anchor=tk.W)

        self.tree_frame = tk.Frame(parent_frame, bg=self.COLORS['bg_medium'])
        self.tree_frame.pack(fill=tk.X, expand=False, pady=(2, 10))

        self.history_header = tk.Frame(self.tree_frame, bg=self.COLORS['bg_light'])
        self.history_header.pack(fill=tk.X)
        header_str = " #  " + "White".center(14) + "Black".center(14)
        tk.Label(
            self.history_header, text=header_str,
            bg=self.COLORS['bg_light'], fg=self.COLORS['text_light'],
            font=('Courier', 11, 'bold'), anchor=tk.W, justify=tk.LEFT
        ).pack(side=tk.LEFT, fill=tk.X)

        self.moves_text = tk.Text(
            self.tree_frame, font=('Courier', 11),
            bg=self.COLORS['bg_medium'], fg=self.COLORS['text_light'],
            borderwidth=0, highlightthickness=0, state=tk.DISABLED,
            cursor="arrow", wrap=tk.NONE, height=16) # Fixed height
        scrollbar = ttk.Scrollbar(self.tree_frame, orient=tk.VERTICAL, command=self.moves_text.yview)
        self.moves_text.configure(yscrollcommand=scrollbar.set)
        self.moves_text.pack(side=tk.LEFT, fill=tk.X, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 3. Scoreboard (Top-aligned, just below history)
        self.scoreboard_label = ttk.Label(
            parent_frame, text="", font=("Helvetica", 11), justify=tk.LEFT,
            background=self.COLORS['bg_dark'], foreground=self.COLORS['text_light'])
        self.scoreboard_label.pack(fill=tk.X, pady=(5, 5))

        # 4. Bottom Tools (Clamped permanently to bottom)
        self.bottom_tools_frame = ttk.Frame(parent_frame, style='Left.TFrame')
        self.bottom_tools_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(0, 10))

        self.fen_entry = self._create_import_export_widget(self.bottom_tools_frame, "FEN String:", self.load_fen_from_entry, self.copy_fen_to_clipboard)
        self.pgn_entry = self._create_import_export_widget(self.bottom_tools_frame, "PGN Record:", self.load_pgn_from_entry, self.copy_pgn_to_clipboard)

    def _create_import_export_widget(self, parent_frame, label_text, load_cmd, copy_cmd):
        frame = ttk.Frame(parent_frame, style='Left.TFrame')
        frame.pack(fill=tk.X, pady=(2, 2))
        ttk.Label(frame, text=label_text, style='SmallHeader.TLabel').pack(anchor=tk.W)
        entry = ttk.Entry(frame, font=('Courier', 10), style='TEntry')
        entry.pack(fill=tk.X, pady=(2, 2))
        btn_frame = ttk.Frame(frame, style='Left.TFrame')
        btn_frame.pack(fill=tk.X)
        prefix = label_text.split()[0]
        ttk.Button(btn_frame, text=f"Load {prefix}", command=load_cmd, style='Control.TButton').pack(side=tk.LEFT,  fill=tk.X, expand=True, padx=(0, 2))
        ttk.Button(btn_frame, text=f"Copy {prefix}", command=copy_cmd, style='Control.TButton').pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))
        return entry

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        C = {
            'bg_dark':    '#1a1a2e',
            'bg_medium':  '#16213e',
            'bg_light':   '#0f3460',
            'accent':     '#e94560',
            'text_light': '#ffffff',
            'text_dark':  '#a2a2a2',
            'warning':    '#FF8C00',
        }
        style.configure('.',              background=C['bg_dark'],   foreground=C['text_light'])
        style.configure('TFrame',         background=C['bg_dark'])
        style.configure('Left.TFrame',    background=C['bg_dark'])
        style.configure('Right.TFrame',   background=C['bg_medium'])
        style.configure('Canvas.TFrame',  background=C['bg_medium'])

        style.configure('Header.TLabel',      background=C['bg_dark'],  foreground=C['text_light'], font=('Helvetica', 14, 'bold'), padding=(0, 5))
        style.configure('SmallHeader.TLabel', background=C['bg_dark'],  foreground=C['text_light'], font=('Helvetica', 12, 'bold'), padding=(0, 1))
        style.configure('Status.TLabel',      background=C['bg_light'], foreground=C['text_light'], font=('Helvetica', 14, 'bold'), padding=(6, 4), relief='solid', borderwidth=1)

        style.configure('Control.TButton', background=C['accent'],   foreground=C['text_light'], font=('Helvetica', 11, 'bold'), padding=(8, 6), borderwidth=0)
        style.map('Control.TButton',       background=[('active', C['accent']), ('pressed', '#d13550')])

        style.configure('Flipped.TButton', background=C['warning'],  foreground=C['text_light'], font=('Helvetica', 11, 'bold'), padding=(8, 6), borderwidth=0)
        style.map('Flipped.TButton',       background=[('active', C['warning']), ('pressed', '#E07B00')])

        style.configure('Nav.TButton',     background=C['bg_light'], foreground=C['text_light'], font=('Helvetica', 16, 'bold'), padding=(10, 5), borderwidth=0)
        style.map('Nav.TButton',           background=[('active', C['bg_light']), ('pressed', C['bg_medium'])],
                                           foreground=[('disabled', C['text_dark'])])

        style.configure('Custom.TRadiobutton', background=C['bg_dark'], foreground=C['text_light'], font=('Helvetica', 11))
        style.map('Custom.TRadiobutton',       background=[('active', C['bg_dark'])], indicatorcolor=[('selected', C['accent'])])

        style.configure('Custom.TCheckbutton', background=C['bg_dark'], foreground=C['text_light'], font=('Helvetica', 11))
        style.map('Custom.TCheckbutton',       background=[('active', C['bg_dark'])], indicatorcolor=[('selected', C['accent'])])

        style.configure('TEntry', fieldbackground='#FFFFFF', foreground='#000000', insertcolor='#000000')
        return C

    # ------------------------------------------------------------------ resize
    def handle_main_resize(self, event):
        new_sidebar_width = max(240, int(event.width * 0.20))
        if new_sidebar_width == self.left_panel.winfo_width():
            return
        self.left_panel.config(width=new_sidebar_width)
        self.right_panel.config(width=new_sidebar_width + 20)

    def handle_board_resize(self, event):
        eval_h = self.eval_frame.winfo_height() if self.eval_frame.winfo_height() > 1 else self.eval_frame.winfo_reqheight()
        nav_h  = self.navigation_frame.winfo_height() if self.navigation_frame.winfo_height() > 1 else self.navigation_frame.winfo_reqheight()
        reserved_height = eval_h + nav_h + 35

        view_width  = event.width - 40
        view_height = event.height - reserved_height
        if view_width <= 1 or view_height <= 1:
            return

        new_square_size  = min(view_width // COLS, view_height // ROWS)
        board_pixel_width = COLS * self.square_size
        self.eval_frame.config(width=board_pixel_width)
        self.eval_bar_canvas.config(width=board_pixel_width)

        if new_square_size != self.square_size and new_square_size > 0:
            self.square_size  = new_square_size
            board_pixel_width = COLS * self.square_size
            self.canvas.config(width=board_pixel_width, height=ROWS * self.square_size)
            self.eval_frame.config(width=board_pixel_width)
            self.eval_bar_canvas.config(width=board_pixel_width)
            self.board_image_white = self.create_board_image("white")
            self.board_image_black = self.create_board_image("black")
            self.draw_board()
        self._position_side_labels()

    def handle_key_press(self, event):
        if self.is_ai_thinking() and self.ai_process.name != self.ANALYSIS_AI_NAME:
            return
        actions = {'Left': self.undo_move, 'Right': self.redo_move,
                   'Home': self.go_to_start, 'End': self.go_to_end}
        action = actions.get(event.keysym)
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
                self.master.after(20, self._make_game_ai_move)
        elif mode == GameMode.AI_VS_AI.value:
            if not self.game_over:
                self.master.after(20, self._make_game_ai_move)
        elif mode == GameMode.HUMAN_VS_HUMAN.value:
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
                delay = 4 if self.instant_move.get() else 20
                self.master.after(delay, self._make_game_ai_move)

    def _reset_game_state_vars(self):
        self.full_history    = [(self.board.clone(), self.turn, None)]
        self.history_pointer = 0
        self.position_counts = {board_hash(self.board, self.turn): 1}
        self.game_over   = False
        self.game_result = None
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
        fen = ""
        for r in range(ROWS):
            empty = 0
            for c in range(COLS):
                piece = self.board.grid[r][c]
                if piece is None:
                    empty += 1
                else:
                    if empty:
                        fen += str(empty)
                        empty = 0
                    ch = _CLASS_TO_FEN_CHAR[type(piece)]
                    fen += ch if piece.color == "white" else ch.lower()
            if empty:
                fen += str(empty)
            if r < ROWS - 1:
                fen += "/"
        turn_char = 'w' if self.turn == 'white' else 'b'
        return f"{fen} {turn_char} - - 0 1"

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
        parts      = fen.split()
        board_part = parts[0]
        turn_part  = parts[1] if len(parts) > 1 else 'w'

        self._stop_ai_process()
        self.board = Board(setup=False)
        r = c = 0
        for ch in board_part:
            if ch == '/':
                r += 1; c = 0
            elif ch.isdigit():
                c += int(ch)
            else:
                color       = "white" if ch.isupper() else "black"
                piece_class = _FEN_CHAR_TO_CLASS.get(ch.lower())
                if piece_class:
                    self.board.add_piece(piece_class(color), r, c)
                c += 1

        self.turn         = "white" if turn_part.lower() == 'w' else "black"
        self.game_started = True
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

        if (not self.game_over
                and self.game_mode.get() == GameMode.HUMAN_VS_BOT.value
                and self.turn != self.human_color):
            self.master.after(20, self._make_game_ai_move)

    def get_current_pgn(self):
        moves = []
        for i in range(1, len(self.full_history)):
            board_before = self.full_history[i - 1][0]
            board_after  = self.full_history[i][0]
            move         = self.full_history[i][2]
            if move:
                moves.append(format_move_san(board_before, board_after, move))

        pgn        = ""
        move_num   = 1
        start_turn = self.full_history[0][1]

        if start_turn == 'black' and moves:
            pgn    += f"{move_num}... {moves[0]} "
            moves   = moves[1:]
            move_num += 1

        for i in range(0, len(moves), 2):
            w   = moves[i]
            b   = moves[i + 1] if i + 1 < len(moves) else None
            pgn += f"{move_num}. {w}, {b} " if b else f"{move_num}. {w} "
            move_num += 1

        if self.game_result:
            res = self.game_result[1]
            pgn += "1-0" if res == 'white' else "0-1" if res == 'black' else "1/2-1/2"
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

        for res in ["1-0", "0-1", "1/2-1/2", "*"]:
            pgn_text = pgn_text.replace(res, "")
        pgn_text = re.sub(r'\d+\.+', '', pgn_text).replace(',', ' ')

        while pgn_text.strip():
            pgn_text    = pgn_text.strip()
            legal_moves = get_all_legal_moves(self.board, self.turn)
            san_map     = {}
            for m in legal_moves:
                child = self.board.clone()
                child.make_move(m[0], m[1])
                san_map[format_move_san(self.board, child, m)] = m

            matched_move = None
            matched_san  = ""
            for san in sorted(san_map, key=len, reverse=True):
                if pgn_text.startswith(san):
                    if len(pgn_text) == len(san) or pgn_text[len(san)].isspace():
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
                messagebox.showwarning("PGN Error", f"Could not parse next move from: {pgn_text[:20]}...")
                break

    # ------------------------------------------------------------------ move history UI
    def update_moves_list(self):
        self.moves_text.config(state=tk.NORMAL)
        self.moves_text.delete(1.0, tk.END)

        formatted_moves = []
        for i in range(1, len(self.full_history)):
            bb   = self.full_history[i - 1][0]
            ba   = self.full_history[i][0]
            move = self.full_history[i][2]
            if move:
                formatted_moves.append(format_move_san(bb, ba, move))

        start_turn = self.full_history[0][1]
        pairs      = []
        if start_turn == 'black' and formatted_moves:
            pairs.append(["...", formatted_moves[0]])
            formatted_moves = formatted_moves[1:]
        for i in range(0, len(formatted_moves), 2):
            w = formatted_moves[i]
            b = formatted_moves[i + 1] if i + 1 < len(formatted_moves) else ""
            pairs.append([w, b])

        for i, pair in enumerate(pairs):
            num_str = f"{i + 1}.".ljust(4)
            w_str   = self._format_san_display(pair[0]).center(14)
            b_str   = self._format_san_display(pair[1]).center(14)

            self.moves_text.insert(tk.END, num_str, "num")

            w_ptr = (i * 2) + 1 if start_turn == 'white' else (i * 2)
            b_ptr = (i * 2) + 2 if start_turn == 'white' else (i * 2) + 1
            w_tag = f"ply_{w_ptr}"
            b_tag = f"ply_{b_ptr}"

            self.moves_text.insert(tk.END, w_str, w_tag)
            if pair[1]:
                self.moves_text.insert(tk.END, b_str, b_tag)
            else:
                self.moves_text.insert(tk.END, b_str)
            self.moves_text.insert(tk.END, "\n")

            if pair[0] != "...":
                self.moves_text.tag_bind(w_tag, "<Button-1>", lambda e, p=w_ptr: self._navigate_history(p))
                self.moves_text.tag_bind(w_tag, "<Enter>",    lambda e: self.moves_text.config(cursor="hand2"))
                self.moves_text.tag_bind(w_tag, "<Leave>",    lambda e: self.moves_text.config(cursor="arrow"))
            if pair[1]:
                self.moves_text.tag_bind(b_tag, "<Button-1>", lambda e, p=b_ptr: self._navigate_history(p))
                self.moves_text.tag_bind(b_tag, "<Enter>",    lambda e: self.moves_text.config(cursor="hand2"))
                self.moves_text.tag_bind(b_tag, "<Leave>",    lambda e: self.moves_text.config(cursor="arrow"))

        self.moves_text.tag_configure("num", foreground=self.COLORS['text_dark'])
        for tag in self.moves_text.tag_names():
            if tag.startswith("ply_"):
                self.moves_text.tag_configure(tag, background=self.COLORS['bg_medium'],
                                              foreground=self.COLORS['text_light'])
        if self.history_pointer > 0:
            active_tag = f"ply_{self.history_pointer}"
            self.moves_text.tag_configure(active_tag, background=self.COLORS['accent'],
                                          foreground=self.COLORS['text_light'])
            try:
                self.moves_text.see(f"{active_tag}.first")
            except tk.TclError:
                pass

        self.moves_text.config(state=tk.DISABLED)

    # ------------------------------------------------------------------ core gameplay
    def execute_move_and_check_state(self, player_who_moved, move):
        # --- ADD INCREMENT ---
        if not self.game_over and self.increment:
            if player_who_moved == 'white':
                self.white_time += self.increment
            else:
                self.black_time += self.increment
            self.render_clocks()
        # ---------------------

        self.switch_turn()
        if not self.game_over and not self.clock_running:
            self.last_clock_tick = time.time()
            self.clock_running = True
            self._tick_clock()
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
            self._log_game_over()
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
                delay = 4 if self.instant_move.get() else 20
                self.master.after(delay, self._make_game_ai_move)
        else:
            print("AI reported no valid move.")

        self._stop_ai_process()
        self.update_bot_labels()
        self.set_interactivity(True)

    def on_drag_start(self, event):
        if self.game_over:
            return
        if self.is_ai_thinking() and self.ai_process.name != self.ANALYSIS_AI_NAME:
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
        self.dragging  = True
        self.drag_start = (r, c)
        self.valid_moves = get_all_legal_moves(self.board, self.turn)
        self.valid_moves_for_highlight = [end for start, end in self.valid_moves if start == self.selected]
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
        is_analysis = self.is_ai_thinking() and self.ai_process.name == self.ANALYSIS_AI_NAME
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
                    self.master.after(4 if self.instant_move.get() else 20, self._make_game_ai_move)
                elif mode == GameMode.HUMAN_VS_HUMAN.value:
                    self._update_analysis_after_state_change()

        self.drag_start = None
        self.update_ui_after_state_change()
        self.set_interactivity(True)

    def reset_game(self):
        if self.game_mode.get() != GameMode.AI_VS_AI.value:
            self.ai_series_running = False
        self._stop_ai_process()
        self.board        = Board()
        self.turn         = "white"
        self.game_started = False
        self.last_move_timestamp = time.time()
        self.selected, self.valid_moves = None, []

        self._reset_game_state_vars()

        # --- TIME RESET ---
        base_seconds = float(self.time_control_seconds.get())
        self.white_time = base_seconds
        self.black_time = base_seconds
        self.increment = base_seconds / 60.0
        self.clock_running = False # Starts on first move
        self._update_time_control_label()
        self.render_clocks()
        # ------------------

        self.fen_entry.delete(0, tk.END)
        self.pgn_entry.delete(0, tk.END)
        self._start_game_if_needed()

        mode  = self.game_mode.get()
        delay = 4 if self.instant_move.get() else 20
        if mode == GameMode.AI_VS_AI.value:
            self.white_playing_bot_type = "op" if (self.ai_series_running and self.ai_series_stats['game_count'] % 2 == 1) else "main"
            self.board_orientation      = "white" if self.white_playing_bot_type == "main" else "black"
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
        bot_class = None
        bot_name  = None
        if mode == GameMode.HUMAN_VS_BOT.value:
            if self.turn != self.human_color:
                bot_class, bot_name = ChessBot, self.MAIN_AI_NAME
        elif mode == GameMode.AI_VS_AI.value:
            main_color          = "white" if self.white_playing_bot_type == "main" else "black"
            bot_class, bot_name = (ChessBot, self.MAIN_AI_NAME) if self.turn == main_color else (OpponentAI, self.OPPONENT_AI_NAME)
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
        board_state, turn_state, _ = self.full_history[self.history_pointer]
        self.board     = board_state.clone()
        self.turn      = turn_state
        self.game_over = False
        self.game_result = None

        self.position_counts.clear()
        for i in range(self.history_pointer + 1):
            board, turn, _ = self.full_history[i]
            h = board_hash(board, turn)
            self.position_counts[h] = self.position_counts.get(h, 0) + 1

        status, winner = get_game_state(self.board, self.turn, self.position_counts,
                                        self.history_pointer, self.MAX_GAME_MOVES)
        if status != "ongoing":
            self.game_over   = True
            self.game_result = (status, winner)

        self.update_ui_after_state_change()
        self._update_analysis_after_state_change()

    def update_game_info_label(self):
        mode = self.game_mode.get()
        if mode == GameMode.HUMAN_VS_BOT.value:
            text = f"Human vs {self.MAIN_AI_NAME}"
        elif mode == GameMode.AI_VS_AI.value:
            text = f"{self.MAIN_AI_NAME} vs {self.OPPONENT_AI_NAME}"
        else:
            text = "Human vs Human Analysis"
        self.game_info_label.config(text=text)

    # ------------------------------------------------------------------ board drawing
    def create_board_image(self, orientation):
        if self.square_size <= 0:
            return None
        img = tk.PhotoImage(width=COLS * self.square_size, height=ROWS * self.square_size)
        C1, C2 = "#D2B48C", "#8B5A2B"
        for r in range(ROWS):
            for c in range(COLS):
                color = C1 if (r + c) % 2 == 0 else C2
                x1, y1 = c * self.square_size, r * self.square_size
                img.put(color, to=(x1, y1, x1 + self.square_size, y1 + self.square_size))
        return img

    def draw_board(self):
        current_image = self.board_image_white if self.board_orientation == "white" else self.board_image_black
        if not current_image:
            return
        self.canvas.itemconfig(self.board_image_id, image=current_image)
        self.canvas.delete("highlight", "piece", "check_highlight", "border_highlight")

        mode = self.game_mode.get()
        warn = (mode == GameMode.HUMAN_VS_BOT.value and self.board_orientation != self.human_color) or \
               (mode != GameMode.HUMAN_VS_BOT.value and self.board_orientation == "black")
        if warn:
            w, h = COLS * self.square_size, ROWS * self.square_size
            self.canvas.create_rectangle(2, 2, w - 2, h - 2,
                                         outline=self.COLORS['warning'], width=4, tags="border_highlight")

        for r_move, c_move in getattr(self, 'valid_moves_for_highlight', []):
            x1, y1   = self.board_to_canvas(r_move, c_move)
            r_dot    = self.square_size // 5
            cx, cy   = x1 + self.square_size // 2, y1 + self.square_size // 2
            self.canvas.create_oval(cx - r_dot, cy - r_dot, cx + r_dot, cy + r_dot,
                                    fill="#1E90FF", outline="", tags="highlight")

        for r in range(ROWS):
            for c in range(COLS):
                if self.board.grid[r][c]:
                    self.draw_piece_with_check(r, c)
        self._position_side_labels()

    def _position_side_labels(self):
        if not hasattr(self, "canvas"):
            return
        desired_label_width = max(96, int(self.square_size * 1.9))
        self.board_row_frame.update_idletasks()
        canvas_x   = self.canvas.winfo_x()
        canvas_y   = self.canvas.winfo_y()
        board_h    = ROWS * self.square_size
        available  = max(1, canvas_x - 10)
        label_w    = min(desired_label_width, available)

        if label_w < 20:
            self.top_bot_label.place_forget()
            self.bottom_bot_label.place_forget()
            return

        for lbl in (self.top_bot_label, self.bottom_bot_label):
            lbl.config(wraplength=max(1, label_w - 6), anchor="center", justify=tk.CENTER)

        strip_right = max(3, canvas_x - 8)
        left_x      = 2 + max(0, (strip_right - 2 - label_w) // 2)
        self.top_bot_label   .place(in_=self.board_row_frame, x=left_x, y=canvas_y + 4,          width=label_w, anchor="nw")
        self.bottom_bot_label.place(in_=self.board_row_frame, x=left_x, y=canvas_y + board_h - 4, width=label_w, anchor="sw")

    def draw_piece_with_check(self, r, c):
        piece = self.board.grid[r][c]
        if isinstance(piece, King):
            is_lost    = (self.game_over and self.game_result
                          and self.game_result[0] == "checkmate"
                          and self.game_result[1] != piece.color)
            is_checked = is_in_check(self.board, piece.color)
            color      = "darkred" if is_lost else ("red" if is_checked else None)
            if color:
                x1, y1 = self.board_to_canvas(r, c)
                self.canvas.create_rectangle(x1, y1, x1 + self.square_size, y1 + self.square_size,
                                             outline=color, width=4, tags="check_highlight")
        if (r, c) != self.drag_start:
            self.draw_piece_at_canvas_coords(piece, r, c)

    def draw_piece_at_canvas_coords(self, piece, r, c):
        x, y      = self.board_to_canvas(r, c)
        cx, cy    = x + self.square_size // 2, y + self.square_size // 2 + 2
        font_size = int(self.square_size * 0.67)
        font      = ("Arial Unicode MS", font_size)
        sym       = piece.symbol()
        self.canvas.create_text(cx + 1, cy + 1, text=sym, font=font, fill="#888888", tags="piece")
        fill = "#000000" if piece.color == "black" else "#FFFFFF"
        self.canvas.create_text(cx, cy, text=sym, font=font, fill=fill, tags="piece")

    def draw_eval_bar(self, eval_score, depth=None):
        score = eval_score / 100.0
        w, h  = self.eval_bar_canvas.winfo_width(), self.eval_bar_canvas.winfo_height()
        if w <= 1 or h <= 1:
            return
        if w != self.last_eval_bar_w or h != self.last_eval_bar_h:
            self.eval_bar_canvas.delete("gradient")
            for x_px in range(w):
                intensity = int(255 * x_px / (w - 1))
                color     = f"#{intensity:02x}{intensity:02x}{intensity:02x}"
                self.eval_bar_canvas.create_line(x_px, 0, x_px, h, fill=color, tags="gradient")
            self.last_eval_bar_w, self.last_eval_bar_h = w, h

        self.eval_bar_canvas.delete("marker")
        marker_norm = max(-1.0, min(1.0, math.tanh(score / 10.0)))
        marker_x    = int(((marker_norm + 1) / 2.0) * w)
        self.eval_bar_canvas.create_line(marker_x, 0, marker_x, h, fill="#FF0000", width=3, tags="marker")
        self.eval_bar_canvas.create_line(w // 2,   0, w // 2,   h, fill="#00FF00", width=2, tags="marker")

        depth_sfx = f" (D{depth})" if depth is not None else ""
        if abs(score) < 0.05:
            self.eval_score_label.config(text=f"Even{depth_sfx}")
        else:
            self.eval_score_label.config(text=f"{'+' if score > 0 else ''}{score:.2f}{depth_sfx}")

    # ------------------------------------------------------------------ comm queue / PV
    def process_comm_queue(self):
        try:
            while not self.comm_queue.empty():
                message = self.comm_queue.get_nowait()
                kind    = message[0]
                if kind == 'log':
                    print(message[1])
                    if self.auto_save_stats_var.get() and self.game_mode.get() == GameMode.AI_VS_AI.value:
                        m = re.search(r'> (.*?) \(D(\d+|TB)\).*?Time=([0-9.]+)s', message[1])
                        if m:
                            bot_n = m.group(1).strip()
                            depth = m.group(2)
                            t_val = float(m.group(3))
                            for cat in (bot_n, 'Global'):
                                self.depth_stats.setdefault(cat, {}).setdefault(depth, []).append(t_val)
                elif kind == 'eval':
                    self.last_eval_score, self.last_eval_depth = message[1], message[2]
                    self.draw_eval_bar(self.last_eval_score, self.last_eval_depth)
                elif kind == 'pv':
                    self.last_pv_message = message
                    self._render_pv()
                elif kind == 'move':
                    self._execute_ai_move(message[1])
        except Exception:
            pass
        finally:
            self.master.after(100, self.process_comm_queue)

    def _render_pv(self):
        """Render (or hide) the PV line. Called on new PV data and on toggle."""
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
            score_disp = f"+M{(1000000 - score + 1) // 2}"
        elif score < -990000:
            score_disp = f"-M{(score + 1000000 + 1) // 2}"
        else:
            score_disp = f"{score / 100:+.2f}"

        self.pv_text.config(state=tk.NORMAL)
        self.pv_text.delete(1.0, tk.END)
        self.pv_text.insert(tk.END, f"[{score_disp}] (D{depth}):\n")

        for i, (san, _) in enumerate(zip(pv_san_list, pv_raw)):
            tag = f"pv_move_{i}"
            self.pv_text.insert(tk.END, self._format_san_display(san) + " ", tag)
            self.pv_text.tag_bind(tag, "<Enter>", lambda e, idx=i, t=tag: self.on_pv_hover_enter(e, idx, t))
            self.pv_text.tag_bind(tag, "<Leave>", lambda e, t=tag:        self.on_pv_hover_leave(e, t))

        self.pv_text.config(state=tk.DISABLED)

    # ------------------------------------------------------------------ AI process
    def _start_ai_process(self, bot_class, bot_name, search_depth):
        if self.ai_process and self.ai_process.is_alive():
            return
        self.ai_cancellation_event.clear()
        
        # --- TIME PASS-DOWN ---
        time_left = self.white_time if self.turn == 'white' else self.black_time
        inc = self.increment
        # ----------------------

        args = (self.board.clone(), self.turn, self.position_counts.copy(),
                self.comm_queue, self.ai_cancellation_event,
                bot_class, bot_name, search_depth, self.history_pointer, self.game_mode.get(),
                time_left, inc)
        self.ai_process      = mp.Process(target=run_ai_process, args=args, daemon=True)
        self.ai_process.name = bot_name
        self.ai_process.start()
        if bot_name != self.ANALYSIS_AI_NAME:
            self.set_interactivity(False)
        self.update_bot_labels()

    def _stop_ai_process(self):
        if self.ai_process and self.ai_process.is_alive():
            self.ai_cancellation_event.set()
            self.ai_process.join(timeout=0.1)
            if self.ai_process.is_alive():
                self.ai_process.terminate()
            self.ai_process = None
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
        if (self.analysis_mode_var.get()
                and self.game_mode.get() == GameMode.HUMAN_VS_HUMAN.value
                and not self.game_over):
            fullmove = (self.history_pointer + 1) // 2 + 1
            print(f"\n--- Analysis: Move {fullmove}, Ply {self.history_pointer}, {self.turn.capitalize()} to move ---")
            self.master.after(50, lambda: self._start_ai_process(ChessBot, self.ANALYSIS_AI_NAME, 99))

    # ------------------------------------------------------------------ misc helpers
    def _start_game_if_needed(self):
        if not self.game_started:
            self.game_started = True

    def _update_time_control_label(self):
        t = int(self.time_control_seconds.get())
        m = t // 60
        s = t % 60
        self.time_control_label.config(text=f"Time Control: {m:02d}:{s:02d}")

    def render_clocks(self):
        self.clock_frame.pack(after=self.turn_label, fill=tk.X, pady=(5, 5))

        def fmt(t):
            if t < 0: t = 0
            m = int(t) // 60
            s = int(t) % 60
            ms = int((t - int(t)) * 10)
            return f"{m:02d}:{s:02d}.{ms}"

        self.white_clock_lbl.config(text=f"W: {fmt(self.white_time)}")
        self.black_clock_lbl.config(text=f"B: {fmt(self.black_time)}")

        # Highlight active clock
        if self.turn == 'white' and not self.game_over:
            self.white_clock_lbl.config(bg=self.COLORS['accent'], fg=self.COLORS['text_light'])
            self.black_clock_lbl.config(bg=self.COLORS['bg_medium'], fg=self.COLORS['text_dark'])
        elif self.turn == 'black' and not self.game_over:
            self.white_clock_lbl.config(bg=self.COLORS['bg_light'], fg=self.COLORS['text_dark'])
            self.black_clock_lbl.config(bg=self.COLORS['accent'], fg=self.COLORS['text_light'])
        else:
            self.white_clock_lbl.config(bg=self.COLORS['bg_light'], fg=self.COLORS['text_light'])
            self.black_clock_lbl.config(bg=self.COLORS['bg_medium'], fg=self.COLORS['text_light'])

    def _tick_clock(self):
        if not self.clock_running or self.game_over:
            return

        now = time.time()
        elapsed = now - self.last_clock_tick
        self.last_clock_tick = now

        if self.turn == 'white':
            self.white_time -= elapsed
            if self.white_time <= 0:
                self.white_time = 0
                self.handle_timeout('white')
        else:
            self.black_time -= elapsed
            if self.black_time <= 0:
                self.black_time = 0
                self.handle_timeout('black')

        self.render_clocks()

        if not self.game_over:
            self.master.after(50, self._tick_clock)

    def handle_timeout(self, color):
        self.game_over = True
        self.clock_running = False
        winner = 'black' if color == 'white' else 'white'
        self.game_result = ('timeout', winner)
        self.update_turn_label()
        self._log_game_over()
        self._stop_ai_process()
        if self.game_mode.get() == GameMode.AI_VS_AI.value and self.ai_series_running:
            self.process_ai_series_result()

    def _log_game_over(self):
        if self.game_result:
            print(f"Game Over! Result: {self.game_result[0]}")

    def update_turn_label(self):
        if self.game_result:
            self.turn_label.config(text=f"GAME OVER: {self.game_result[0].upper()}")
        else:
            self.turn_label.config(text=f"TURN: {self.turn.upper()}")

    def update_bot_labels(self):
        mode = self.game_mode.get()
        if mode == GameMode.AI_VS_AI.value:
            white_label = self.MAIN_AI_NAME     if self.white_playing_bot_type == "main" else self.OPPONENT_AI_NAME
            black_label = self.OPPONENT_AI_NAME if self.white_playing_bot_type == "main" else self.MAIN_AI_NAME
        elif mode == GameMode.HUMAN_VS_BOT.value:
            white_label = "Human"       if self.human_color == "white" else self.MAIN_AI_NAME
            black_label = "Human"       if self.human_color == "black" else self.MAIN_AI_NAME
        else:
            white_label, black_label = "White", "Black"

        if self.turn == "white":
            white_label += "\n(to move)"
        else:
            black_label += "\n(to move)"

        bottom = white_label if self.board_orientation == 'white' else black_label
        top    = black_label if self.board_orientation == 'white' else white_label
        self.bottom_bot_label.config(text=bottom)
        self.top_bot_label   .config(text=top)
        self._position_side_labels()

    def set_interactivity(self, is_interactive):
        if is_interactive:
            self.canvas.bind("<Button-1>",        self.on_drag_start)
            self.canvas.bind("<B1-Motion>",       self.on_drag_motion)
            self.canvas.bind("<ButtonRelease-1>", self.on_drag_end)
        else:
            self.canvas.unbind("<Button-1>")
            self.canvas.unbind("<B1-Motion>")
            self.canvas.unbind("<ButtonRelease-1>")

    def is_ai_thinking(self):
        return bool(self.ai_process and self.ai_process.is_alive())

    def switch_turn(self):
        if not self.game_over:
            self.turn = "black" if self.turn == "white" else "white"

    def board_to_canvas(self, r, c):
        if self.board_orientation == "black":
            return (COLS - 1 - c) * self.square_size, (ROWS - 1 - r) * self.square_size
        return c * self.square_size, r * self.square_size

    def canvas_to_board(self, x, y):
        if self.board_orientation == "black":
            c = (COLS - 1) - (x // self.square_size)
            r = (ROWS - 1) - (y // self.square_size)
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
            for btn in (self.start_button, self.undo_button, self.redo_button, self.end_button):
                btn.config(state=tk.DISABLED)
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
        _, winner_color = self.game_result
        if winner_color:
            main_color = 'white' if self.white_playing_bot_type == 'main' else 'black'
            if winner_color == main_color:
                self.ai_series_stats['my_ai_wins'] += 1
            else:
                self.ai_series_stats['op_ai_wins'] += 1
        else:
            self.ai_series_stats['draws'] += 1

        self.update_scoreboard()
        if self.auto_save_stats_var.get():
            self.save_depth_stats_to_file()

        if self.ai_series_running and self.ai_series_stats['game_count'] < 100:
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
            print("\n--- Generating new 2-ply opening sequence for game pair ---")
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

        for move in self.current_opening_sequence:
            child = self.board.clone()
            child.make_move(move[0], move[1])
            print(f"Applying opening move: {format_move_san(self.board, child, move)}")
            self.board.make_move(move[0], move[1])
            self.execute_move_and_check_state(self.turn, move)

    def update_scoreboard(self):
        if self.game_mode.get() == GameMode.AI_VS_AI.value and self.ai_series_running:
            s = self.ai_series_stats
            text = (f"{self.MAIN_AI_NAME} vs {self.OPPONENT_AI_NAME} (Game {s['game_count'] + 1}/100)\n"
                    f"  {self.MAIN_AI_NAME} Wins: {s['my_ai_wins']}\n"
                    f"  {self.OPPONENT_AI_NAME} Wins: {s['op_ai_wins']}\n"
                    f"  Draws: {s['draws']}")
            self.scoreboard_label.config(text=text)
        else:
            self.scoreboard_label.config(text="")

    def save_depth_stats_to_file(self):
        if not self.depth_stats or not self.depth_stats.get('Global'):
            return
        filename = "AI_Series_Depth_Averages.txt"
        def sort_key(k): return int(k) if k.isdigit() else 999
        try:
            with open(filename, "w") as f:
                s = self.ai_series_stats
                f.write("=== AI vs OP Series Depth Stats ===\n")
                f.write(f"Games Completed: {s['game_count']}\n")
                f.write(f"Score: {self.MAIN_AI_NAME} {s['my_ai_wins']} - {s['op_ai_wins']} {self.OPPONENT_AI_NAME} (Draws: {s['draws']})\n\n")
                for category in ['Global', self.MAIN_AI_NAME, self.OPPONENT_AI_NAME]:
                    data = self.depth_stats.get(category)
                    if not data:
                        continue
                    f.write(f"--- {category} Averages ---\n")
                    for depth in sorted(data, key=sort_key):
                        times = data[depth]
                        f.write(f"  Depth {depth:<3} | Avg: {sum(times)/len(times):.3f}s | Max: {max(times):.3f}s | Samples: {len(times)}\n")
                    f.write("\n")
        except Exception as e:
            print(f"Failed to save stats to file: {e}")

    # ------------------------------------------------------------------ PV hover mini-board
    def on_pv_hover_enter(self, event, move_idx, tag):
        self.pv_text.tag_config(tag, background=self.COLORS['accent'], foreground=self.COLORS['text_light'])
        if getattr(self, 'pv_tooltip', None):
            self.pv_tooltip.destroy()

        self.pv_tooltip = tk.Toplevel(self.master)
        self.pv_tooltip.wm_overrideredirect(True)
        self.pv_tooltip.wm_geometry(f"+{event.x_root + 15}+{event.y_root - ROWS * 25 - 20}")

        self.tt_sq_size = 25
        self.tt_canvas  = tk.Canvas(
            self.pv_tooltip,
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
                    self.tt_canvas.create_text(x1 + sq // 2 + 1, y1 + sq // 2 + 2, text=sym, font=font, fill="#888888")
                    self.tt_canvas.create_text(x1 + sq // 2,     y1 + sq // 2 + 1, text=sym, font=font,
                                               fill="#000" if piece.color == "black" else "#FFF")


if __name__ == "__main__":
    mp.freeze_support()
    root = tk.Tk()
    app  = EnhancedChessApp(root)
    root.mainloop()
