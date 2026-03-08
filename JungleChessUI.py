# JungleChessUI.py (v12.21 - Save games in AI vs OP series to file)

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
    HUMAN_VS_BOT = "bot"
    HUMAN_VS_HUMAN = "human"
    AI_VS_AI = "ai_vs_ai"

def run_ai_process(board, color, position_counts, comm_queue, cancellation_event, bot_class, bot_name, search_depth, ply_count, game_mode):
    bot = bot_class(
        board, color, position_counts, comm_queue, cancellation_event,
        bot_name, ply_count, game_mode
    )
        
    bot.search_depth = search_depth
    if search_depth == 99:
        bot.ponder_indefinitely()
    else:
        bot.make_move()

class EnhancedChessApp:
    MAIN_AI_NAME = "AI Bot"
    OPPONENT_AI_NAME = "OP Bot"
    ANALYSIS_AI_NAME = "Analysis"
    slidermaxvalue = 12
    MAX_GAME_MOVES = 200 
    
    def __init__(self, master):
        self.master = master
        self.master.title("Jungle Chess")
        random.seed()
        
        self.comm_queue = mp.Queue()
        self.ai_process = None
        self.ai_cancellation_event = mp.Event()

        self.board = Board()
        
        self.turn = "white"
        self.selected = None
        self.valid_moves =[]
        self.game_over = False
        self.game_result = None
        self.dragging = False
        self.drag_piece_ghost = None
        self.drag_start = None

        self.full_history =[]
        self.history_pointer = -1
        self.position_counts = {}

        self.current_opening_move = None
        self.square_size = 75
        self.base_sidebar_width = 280

        self.game_mode = tk.StringVar(value=GameMode.HUMAN_VS_BOT.value)
        self.analysis_mode_var = tk.BooleanVar(value=True)
        self.ai_series_running = False
        self.ai_series_stats = {'game_count': 0, 'my_ai_wins': 0, 'op_ai_wins': 0, 'draws': 0}
        self.depth_stats = {}
        self.auto_save_stats_var = tk.BooleanVar(value=True)
        
        self.white_playing_bot_type = "main"
        self.human_color = "white"
        self.board_orientation = "white"
        self.last_move_timestamp = None
        self.game_started = False 

        self.last_eval_score = 0.0
        self.last_eval_depth = None
        self.last_eval_bar_w = 0
        self.last_eval_bar_h = 0

        self.COLORS = self.setup_styles()
        self.master.configure(bg=self.COLORS['bg_dark'])
        self.build_ui()
        
        self.master.bind("<Key>", self.handle_key_press)
        
        self.process_comm_queue()
        self.reset_game()

    def build_ui(self):
        screen_w, screen_h = self.master.winfo_screenwidth(), self.master.winfo_screenheight()
        self.master.geometry(f"{screen_w}x{screen_h}+0+0"); self.master.state('zoomed')
        
        self.main_frame = ttk.Frame(self.master, style='Left.TFrame')
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # --- LEFT PANEL ---
        self.left_panel = ttk.Frame(self.main_frame, style='Left.TFrame')
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10)); self.left_panel.pack_propagate(False)
        
        self.title_label = ttk.Label(self.left_panel, text="JUNGLE CHESS", style='Header.TLabel', font=('Helvetica', 22, 'bold'))
        self.title_label.pack(pady=(0,15))
        self._build_control_widgets(self.left_panel)
        
        # --- CENTER PANEL ---
        self.center_panel = ttk.Frame(self.main_frame, style='Right.TFrame')
        self.center_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.board_column = ttk.Frame(self.center_panel, style='Right.TFrame')
        self.board_column.pack(expand=True, fill=tk.BOTH)

        self.eval_frame = ttk.Frame(self.board_column, style='Right.TFrame', width=COLS * self.square_size, height=58)
        self.eval_frame.pack(side=tk.TOP, anchor=tk.CENTER, pady=(6, 5))
        self.eval_frame.pack_propagate(False)
        self.eval_score_label = ttk.Label(self.eval_frame, text="Even", style='Status.TLabel', anchor="center")
        self.eval_score_label.pack(side=tk.TOP, pady=(0, 4))
        self.eval_bar_canvas = tk.Canvas(
            self.eval_frame, width=COLS * self.square_size, height=20,
            bg=self.COLORS['bg_light'], highlightthickness=1, highlightbackground=self.COLORS['text_dark']
        )
        self.eval_bar_canvas.pack(side=tk.TOP, anchor=tk.CENTER)
        self.eval_bar_canvas.bind("<Configure>", self.redraw_eval_bar_on_resize)
        
        self.board_row_frame = ttk.Frame(self.board_column, style='Right.TFrame')
        self.board_row_frame.pack(expand=True, fill=tk.BOTH)

        self.canvas_frame = ttk.Frame(self.board_row_frame, style='Canvas.TFrame')
        self.canvas_frame.pack(expand=True, fill=tk.BOTH)
        self.canvas = tk.Canvas(self.canvas_frame, width=COLS * self.square_size, height=ROWS * self.square_size, bg=self.COLORS['bg_medium'], highlightthickness=0)
        self.board_image_white = self.create_board_image("white")
        self.board_image_black = self.create_board_image("black")
        self.board_image_id = self.canvas.create_image(0, 0, anchor='nw', tags="board")
        self.canvas.pack(expand=True)

        self.top_bot_label = ttk.Label(
            self.board_row_frame, text="", font=("Helvetica", 11, "bold"),
            background=self.COLORS['bg_medium'], foreground=self.COLORS['text_light'],
            anchor="center", justify=tk.CENTER
        )
        self.bottom_bot_label = ttk.Label(
            self.board_row_frame, text="", font=("Helvetica", 11, "bold"),
            background=self.COLORS['bg_medium'], foreground=self.COLORS['text_light'],
            anchor="center", justify=tk.CENTER
        )

        self.navigation_frame = ttk.Frame(self.center_panel, style='Right.TFrame')
        self.navigation_frame.pack(fill=tk.X, pady=(5, 10))
        self.start_button = ttk.Button(self.navigation_frame, text="«", command=self.go_to_start, style='Nav.TButton', state=tk.DISABLED)
        self.undo_button = ttk.Button(self.navigation_frame, text="‹", command=self.undo_move, style='Nav.TButton', state=tk.DISABLED)
        self.redo_button = ttk.Button(self.navigation_frame, text="›", command=self.redo_move, style='Nav.TButton', state=tk.DISABLED)
        self.end_button = ttk.Button(self.navigation_frame, text="»", command=self.go_to_end, style='Nav.TButton', state=tk.DISABLED)
        self.navigation_frame.columnconfigure(0, weight=1); self.navigation_frame.columnconfigure(5, weight=1)
        self.start_button.grid(row=0, column=1, padx=5); self.undo_button.grid(row=0, column=2, padx=5)
        self.redo_button.grid(row=0, column=3, padx=5); self.end_button.grid(row=0, column=4, padx=5)

        # --- RIGHT PANEL ---
        self.right_panel = ttk.Frame(self.main_frame, style='Left.TFrame')
        self.right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0)); self.right_panel.pack_propagate(False)
        self._build_right_sidebar_widgets(self.right_panel)

        self.main_frame.bind("<Configure>", self.handle_main_resize)
        self.center_panel.bind("<Configure>", self.handle_board_resize)
        
    def _build_control_widgets(self, parent_frame):
        self.game_mode_frame = ttk.Frame(parent_frame, style='Left.TFrame')
        self.game_mode_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(self.game_mode_frame, text="GAME MODE", style='Header.TLabel').pack(anchor=tk.W)
        for mode in GameMode:
            text = mode.name.replace("_", " ").title()
            ttk.Radiobutton(self.game_mode_frame, text=text, variable=self.game_mode, value=mode.value, command=self.on_mode_changed, style='Custom.TRadiobutton').pack(anchor=tk.W, pady=(2,0))
        
        self.controls_frame = ttk.Frame(parent_frame, style='Left.TFrame')
        self.controls_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(self.controls_frame, text="NEW GAME", command=self.reset_game, style='Control.TButton').pack(fill=tk.X, pady=3)
        ttk.Button(self.controls_frame, text="SWAP SIDES", command=self.swap_sides, style='Control.TButton').pack(fill=tk.X, pady=3)
        
        self.flip_view_btn = ttk.Button(self.controls_frame, text="FLIP VIEW", command=self.toggle_board_view, style='Control.TButton')
        self.flip_view_btn.pack(fill=tk.X, pady=3)
        
        ttk.Button(self.controls_frame, text="AI vs OP Series", command=self.start_ai_series, style='Control.TButton').pack(fill=tk.X, pady=3)
        ttk.Checkbutton(self.controls_frame, text="Auto-save Depth Stats", variable=self.auto_save_stats_var, style='Custom.TCheckbutton').pack(anchor=tk.W, pady=(0, 5))
        
        ttk.Label(self.controls_frame, text="Bot Depth:", style='SmallHeader.TLabel').pack(anchor=tk.W, pady=(10,0))
        self.bot_depth_slider = tk.Scale(self.controls_frame, from_=1, to=self.slidermaxvalue, orient=tk.HORIZONTAL, bg=self.COLORS['bg_dark'], fg=self.COLORS['text_light'], highlightthickness=0, relief='flat')
        self.bot_depth_slider.set(ChessBot.search_depth); self.bot_depth_slider.pack(fill=tk.X, pady=(0,5))
        self.instant_move = tk.BooleanVar(value=False); ttk.Checkbutton(self.controls_frame, text="Instant Moves", variable=self.instant_move, style='Custom.TCheckbutton').pack(anchor=tk.W, pady=(2,2))
        self.analysis_checkbox = ttk.Checkbutton(self.controls_frame, text="Analysis Mode (H-vs-H)", variable=self.analysis_mode_var, style='Custom.TCheckbutton', command=self._update_analysis_after_state_change)
        self.analysis_checkbox.pack(anchor=tk.W, pady=(2,2))

    def _build_right_sidebar_widgets(self, parent_frame):
        self.info_frame = ttk.Frame(parent_frame, style='Left.TFrame')
        self.info_frame.pack(fill=tk.X, pady=(0, 5))
        self.game_info_label = ttk.Label(self.info_frame, text="Match Info", style='Header.TLabel')
        self.game_info_label.pack(anchor=tk.W)
        self.turn_label = ttk.Label(self.info_frame, text="WHITE'S TURN", style='Status.TLabel')
        self.turn_label.pack(fill=tk.X, pady=(5,10))

        ttk.Label(parent_frame, text="Move History", style='SmallHeader.TLabel').pack(anchor=tk.W)
        self.tree_frame = ttk.Frame(parent_frame)
        self.tree_frame.pack(fill=tk.BOTH, expand=True, pady=(2, 10))
        
        self.moves_tree = ttk.Treeview(self.tree_frame, columns=('White', 'Black'), show='headings', selectmode='browse')
        self.moves_tree.heading('White', text='White'); self.moves_tree.heading('Black', text='Black')
        self.moves_tree.column('White', width=100, anchor=tk.CENTER); self.moves_tree.column('Black', width=100, anchor=tk.CENTER)
        
        scrollbar = ttk.Scrollbar(self.tree_frame, orient=tk.VERTICAL, command=self.moves_tree.yview)
        self.moves_tree.configure(yscrollcommand=scrollbar.set)
        self.moves_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.moves_tree.bind('<ButtonRelease-1>', self.on_move_selected)

        self.scoreboard_label = ttk.Label(parent_frame, text="", font=("Helvetica", 11), justify=tk.LEFT, background=self.COLORS['bg_dark'], foreground=self.COLORS['text_light'])
        self.scoreboard_label.pack(fill=tk.X, pady=(0, 10))

        # Replaced redundant logic with helper method
        self.fen_entry = self._create_import_export_widget(parent_frame, "FEN String:", self.load_fen_from_entry, self.copy_fen_to_clipboard)
        self.pgn_entry = self._create_import_export_widget(parent_frame, "PGN Record:", self.load_pgn_from_entry, self.copy_pgn_to_clipboard)

    def _create_import_export_widget(self, parent_frame, label_text, load_cmd, copy_cmd):
        frame = ttk.Frame(parent_frame, style='Left.TFrame')
        frame.pack(fill=tk.X, pady=(5, 5))
        ttk.Label(frame, text=label_text, style='SmallHeader.TLabel').pack(anchor=tk.W)
        entry = ttk.Entry(frame, font=('Courier', 10), style='TEntry')
        entry.pack(fill=tk.X, pady=(2, 4))
        btn_frame = ttk.Frame(frame, style='Left.TFrame')
        btn_frame.pack(fill=tk.X)
        button_prefix = label_text.split()[0]
        ttk.Button(btn_frame, text=f"Load {button_prefix}", command=load_cmd, style='Control.TButton').pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ttk.Button(btn_frame, text=f"Copy {button_prefix}", command=copy_cmd, style='Control.TButton').pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))
        return entry

    def setup_styles(self):
        style = ttk.Style(); style.theme_use('clam')
        C = {'bg_dark': '#1a1a2e', 'bg_medium': '#16213e', 'bg_light': '#0f3460', 'accent': '#e94560', 'text_light': '#ffffff', 'text_dark': '#a2a2a2', 'warning': '#FF8C00'}
        
        style.configure('.', background=C['bg_dark'], foreground=C['text_light'])
        style.configure('TFrame', background=C['bg_dark'])
        style.configure('Left.TFrame', background=C['bg_dark'])
        style.configure('Right.TFrame', background=C['bg_medium'])
        style.configure('Canvas.TFrame', background=C['bg_medium'])
        
        style.configure('Header.TLabel', background=C['bg_dark'], foreground=C['text_light'], font=('Helvetica', 14, 'bold'), padding=(0, 5))
        style.configure('SmallHeader.TLabel', background=C['bg_dark'], foreground=C['text_light'], font=('Helvetica', 12, 'bold'), padding=(0, 1))
        style.configure('Status.TLabel', background=C['bg_light'], foreground=C['text_light'], font=('Helvetica', 14, 'bold'), padding=(6, 4), relief='solid', borderwidth=1)
        
        style.configure('Control.TButton', background=C['accent'], foreground=C['text_light'], font=('Helvetica', 11, 'bold'), padding=(8, 6), borderwidth=0)
        style.map('Control.TButton', background=[('active', C['accent']), ('pressed', '#d13550')])
        
        style.configure('Flipped.TButton', background=C['warning'], foreground=C['text_light'], font=('Helvetica', 11, 'bold'), padding=(8, 6), borderwidth=0)
        style.map('Flipped.TButton', background=[('active', C['warning']), ('pressed', '#E07B00')])

        style.configure('Nav.TButton', background=C['bg_light'], foreground=C['text_light'], font=('Helvetica', 16, 'bold'), padding=(10, 5), borderwidth=0)
        style.map('Nav.TButton', background=[('active', C['bg_light']), ('pressed', C['bg_medium'])], foreground=[('disabled', C['text_dark'])])
        
        style.configure('Custom.TRadiobutton', background=C['bg_dark'], foreground=C['text_light'], font=('Helvetica', 11))
        style.map('Custom.TRadiobutton', background=[('active', C['bg_dark'])], indicatorcolor=[('selected', C['accent'])])
        style.configure('Custom.TCheckbutton', background=C['bg_dark'], foreground=C['text_light'], font=('Helvetica', 11))
        style.map('Custom.TCheckbutton', background=[('active', C['bg_dark'])], indicatorcolor=[('selected', C['accent'])])

        style.configure('Treeview', font=('Courier', 11), rowheight=25, background=C['bg_medium'], foreground=C['text_light'], fieldbackground=C['bg_medium'], borderwidth=0)
        style.configure('Treeview.Heading', font=('Helvetica', 11, 'bold'), background=C['bg_light'], foreground=C['text_light'], borderwidth=0)
        style.map('Treeview', background=[('selected', C['accent'])], foreground=[('selected', C['text_light'])])
        
        style.configure('TEntry', fieldbackground='#FFFFFF', foreground='#000000', insertcolor='#000000')
        return C

    def handle_main_resize(self, event):
        new_sidebar_width = max(240, int(event.width * 0.20))
        if new_sidebar_width == self.left_panel.winfo_width(): return
        self.left_panel.config(width=new_sidebar_width)
        self.right_panel.config(width=new_sidebar_width + 20)

    def handle_board_resize(self, event):
        eval_h = self.eval_frame.winfo_height() if self.eval_frame.winfo_height() > 1 else self.eval_frame.winfo_reqheight()
        nav_h = self.navigation_frame.winfo_height() if self.navigation_frame.winfo_height() > 1 else self.navigation_frame.winfo_reqheight()
        reserved_height = eval_h + nav_h + 35

        view_width = event.width - 40
        view_height = event.height - reserved_height
        if view_width <= 1 or view_height <= 1:
            return

        new_square_size = min(view_width // COLS, view_height // ROWS)
        board_pixel_width = COLS * self.square_size
        self.eval_frame.config(width=board_pixel_width)
        self.eval_bar_canvas.config(width=board_pixel_width)
        if new_square_size != self.square_size and new_square_size > 0:
            self.square_size = new_square_size
            board_pixel_width = COLS * self.square_size
            self.canvas.config(width=board_pixel_width, height=ROWS * self.square_size)
            self.eval_frame.config(width=board_pixel_width)
            self.eval_bar_canvas.config(width=board_pixel_width)
            self.board_image_white = self.create_board_image("white")
            self.board_image_black = self.create_board_image("black")
            self.draw_board()    
        self._position_side_labels()

    def handle_key_press(self, event):
        if self.is_ai_thinking() and self.ai_process.name != self.ANALYSIS_AI_NAME: return
        if event.keysym == 'Left': self.undo_move()
        elif event.keysym == 'Right': self.redo_move()
        elif event.keysym == 'Home': self.go_to_start()
        elif event.keysym == 'End': self.go_to_end()

    def redraw_eval_bar_on_resize(self, event):
        self.draw_eval_bar(self.last_eval_score, self.last_eval_depth)

    # --- FLIP VIEW / SWAP SIDES / MODE SWITCH ---
    def _update_flip_view_button_style(self):
        show_warning = False
        if self.game_mode.get() == GameMode.HUMAN_VS_BOT.value:
            if self.board_orientation != self.human_color:
                show_warning = True
        else:
            if self.board_orientation == "black":
                show_warning = True
                
        if show_warning:
            self.flip_view_btn.configure(style='Flipped.TButton')
        else:
            self.flip_view_btn.configure(style='Control.TButton')

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
            self.human_color = "black" if self.human_color == "white" else "white"
            self.board_orientation = self.human_color
            
            self._update_flip_view_button_style()
            self.update_ui_after_state_change()
            
            if not self.game_over and self.turn != self.human_color:
                print(f"Swapped sides. AI ({self.turn}) taking over...")
                delay = 4 if self.instant_move.get() else 20
                self.master.after(delay, self._make_game_ai_move)

    def _reset_game_state_vars(self):
        self.full_history =[(self.board.clone(), self.turn, None)]
        self.history_pointer = 0
        self.position_counts = {board_hash(self.board, self.turn): 1}
        self.game_over = False
        self.game_result = None
        self.last_eval_score = 0.0
        self.last_eval_depth = None
        self.draw_eval_bar(0)

    # --- PGN & FEN LOGIC ---
    def get_current_fen(self):
        fen = ""
        for r in range(ROWS):
            empty = 0
            for c in range(COLS):
                piece = self.board.grid[r][c]
                if piece is None: empty += 1
                else:
                    if empty > 0: fen += str(empty); empty = 0
                    symbol = "P" if isinstance(piece, Pawn) else "N" if isinstance(piece, Knight) else "B" if isinstance(piece, Bishop) else "R" if isinstance(piece, Rook) else "Q" if isinstance(piece, Queen) else "K"
                    if piece.color == "black": symbol = symbol.lower()
                    fen += symbol
            if empty > 0: fen += str(empty)
            if r < ROWS - 1: fen += "/"
        turn_char = 'w' if self.turn == 'white' else 'b'
        fen += f" {turn_char} - - 0 1"
        return fen

    def copy_fen_to_clipboard(self):
        fen = self.get_current_fen()
        self.fen_entry.delete(0, tk.END); self.fen_entry.insert(0, fen)
        self.master.clipboard_clear(); self.master.clipboard_append(fen)

    def load_fen_from_entry(self):
        fen = self.fen_entry.get().strip()
        if not fen: return
        parts = fen.split()
        board_part, turn_part = parts[0], parts[1] if len(parts) > 1 else 'w'
        
        self._stop_ai_process()
        self.board = Board(setup=False) 
        r, c = 0, 0
        for char in board_part:
            if char == '/': r += 1; c = 0
            elif char.isdigit(): c += int(char)
            else:
                color, char_lower = "white" if char.isupper() else "black", char.lower()
                piece_class = {'p':Pawn, 'n':Knight, 'b':Bishop, 'r':Rook, 'q':Queen, 'k':King}.get(char_lower)
                if piece_class: self.board.add_piece(piece_class(color), r, c)
                c += 1
                
        self.turn = "white" if turn_part.lower() == 'w' else "black"
        self.game_started = True
        self._reset_game_state_vars()

        status, winner = get_game_state(self.board, self.turn, self.position_counts, self.history_pointer, self.MAX_GAME_MOVES)
        if status != "ongoing":
            self.game_over, self.game_result = True, (status, winner)
        
        self.board_orientation = self.human_color
        self._update_flip_view_button_style()
        
        self.update_ui_after_state_change()
        self._update_analysis_after_state_change()
        
        if not self.game_over and self.game_mode.get() == GameMode.HUMAN_VS_BOT.value and self.turn != self.human_color:
             self.master.after(20, self._make_game_ai_move)

    def get_current_pgn(self):
        moves =[]
        for i in range(1, len(self.full_history)):
            board_before = self.full_history[i-1][0]
            board_after = self.full_history[i][0]
            move = self.full_history[i][2]
            if move:
                moves.append(format_move_san(board_before, board_after, move))
                
        pgn = ""
        move_num = 1
        start_turn = self.full_history[0][1]
        
        if start_turn == 'black' and moves:
            pgn += f"{move_num}... {moves[0]} "
            moves = moves[1:]
            move_num += 1
            
        for i in range(0, len(moves), 2):
            w = moves[i]
            if i + 1 < len(moves):
                b = moves[i+1]
                pgn += f"{move_num}. {w}, {b} "
            else:
                pgn += f"{move_num}. {w} "
            move_num += 1
            
        if self.game_result:
            res = self.game_result[1]
            if res == 'white': pgn += "1-0"
            elif res == 'black': pgn += "0-1"
            else: pgn += "1/2-1/2"
        else: 
            pgn += "*"
        return pgn.strip()

    def copy_pgn_to_clipboard(self):
        pgn = self.get_current_pgn()
        self.pgn_entry.delete(0, tk.END); self.pgn_entry.insert(0, pgn)
        self.master.clipboard_clear(); self.master.clipboard_append(pgn)

    def load_pgn_from_entry(self):
        pgn_text = self.pgn_entry.get().strip()
        if not pgn_text: return
        self.reset_game()
        import re
        
        # Clean text
        for res in["1-0", "0-1", "1/2-1/2", "*"]:
            pgn_text = pgn_text.replace(res, "")
            
        pgn_text = re.sub(r'\d+\.+', '', pgn_text)
        pgn_text = pgn_text.replace(',', ' ')
        
        # Smart sequential matching generator
        while pgn_text.strip():
            pgn_text = pgn_text.strip()
            legal_moves = get_all_legal_moves(self.board, self.turn)
            matched_move = None
            matched_san = ""
            
            san_map = {}
            for m in legal_moves:
                child = self.board.clone()
                child.make_move(m[0], m[1])
                san = format_move_san(self.board, child, m)
                san_map[san] = m
                
            # Sort keys by descending length so complex moves match before prefix slices
            for san in sorted(san_map.keys(), key=len, reverse=True):
                if pgn_text.startswith(san):
                    if len(pgn_text) == len(san) or pgn_text[len(san)].isspace():
                        matched_move = san_map[san]
                        matched_san = san
                        break
                        
            if matched_move:
                self.board.make_move(matched_move[0], matched_move[1])
                self.execute_move_and_check_state(self.turn, matched_move)
                pgn_text = pgn_text[len(matched_san):]
                if self.game_over: break
            else:
                messagebox.showwarning("PGN Error", f"Could not parse next move from: {pgn_text[:20]}...")
                break

    def update_moves_list(self):
        for item in self.moves_tree.get_children(): self.moves_tree.delete(item)
        formatted_moves =[]
        for i in range(1, len(self.full_history)):
            board_before = self.full_history[i-1][0]
            board_after = self.full_history[i][0]
            move = self.full_history[i][2]
            if move:
                formatted_moves.append(format_move_san(board_before, board_after, move))
                
        start_turn = self.full_history[0][1]
        
        pairs = []
        if start_turn == 'black' and formatted_moves:
            pairs.append(["...", formatted_moves[0]])
            formatted_moves = formatted_moves[1:]
            
        for i in range(0, len(formatted_moves), 2):
            w = formatted_moves[i]
            b = formatted_moves[i+1] if i+1 < len(formatted_moves) else ""
            pairs.append([w, b])
            
        for i, pair in enumerate(pairs):
            self.moves_tree.insert('', 'end', iid=str(i), text=str(i+1), values=(pair[0], pair[1]))
            
        if self.history_pointer > 0:
            moves_count = self.history_pointer - 1 
            row = (moves_count + 1) // 2 if start_turn == 'black' else moves_count // 2
            if str(row) in self.moves_tree.get_children():
                self.moves_tree.selection_set(str(row))
                self.moves_tree.see(str(row))
        else: self.moves_tree.selection_set()

    def on_move_selected(self, event):
        selected_items = self.moves_tree.selection()
        if not selected_items: return
        index = int(selected_items[0])
        start_turn = self.full_history[0][1]
        pointer_target = (index * 2) + 1 if start_turn == 'black' else (index * 2) + 2
        pointer_target = min(pointer_target, len(self.full_history) - 1)
        self._navigate_history(pointer_target)

    # --- CORE GAMEPLAY ---
    def execute_move_and_check_state(self, player_who_moved, move):
        self.switch_turn()
        if self.history_pointer < len(self.full_history) - 1:
            self.full_history = self.full_history[:self.history_pointer + 1]
            self.position_counts.clear()
            for board, turn, _ in self.full_history:
                self.position_counts[board_hash(board, turn)] = self.position_counts.get(board_hash(board, turn), 0) + 1
        
        self.full_history.append((self.board.clone(), self.turn, move))
        self.history_pointer += 1
        key = board_hash(self.board, self.turn)
        self.position_counts[key] = self.position_counts.get(key, 0) + 1
        
        # GameLogic now automatically returns 'checkmate' if there are no legal moves (Stalemate = Loss)
        status, winner = get_game_state(self.board, self.turn, self.position_counts, self.history_pointer, self.MAX_GAME_MOVES)
        
        if status != "ongoing": 
            self.game_over, self.game_result = True, (status, winner)
        
        self.update_ui_after_state_change()
        if self.game_over:
            self._log_game_over(); self._stop_ai_process()
            if self.game_mode.get() == GameMode.AI_VS_AI.value and self.ai_series_running:
                self.process_ai_series_result()

    def _execute_ai_move(self, the_move):
        if self.game_over: return
        if the_move:
            self.board.make_move(the_move[0], the_move[1])
            self.execute_move_and_check_state(self.turn, the_move)
            delay = 4 if self.instant_move.get() else 20
            if not self.game_over and self.game_mode.get() == GameMode.AI_VS_AI.value:
                self.master.after(delay, self._make_game_ai_move)
        else: print("AI reported no valid move.")
        
        self._stop_ai_process()
        self.update_bot_labels(); self.set_interactivity(True)

    def on_drag_end(self, event):
        is_analysis = self.is_ai_thinking() and self.ai_process.name == self.ANALYSIS_AI_NAME
        if self.is_ai_thinking() and not is_analysis:
            self.valid_moves =[]; self.draw_board(); return
        if not self.dragging:
            self.valid_moves =[]; self.draw_board(); return

        self.dragging = False; self.canvas.delete("drag_ghost")
        row, col = self.canvas_to_board(event.x, event.y)
        if row == -1 or not self.drag_start:
            self.update_ui_after_state_change(); self.set_interactivity(True); return

        start_pos, end_pos = self.drag_start, (row, col)
        move_to_check = (start_pos, end_pos)

        if move_to_check in self.valid_moves:
            self.board.make_move(start_pos, end_pos)
            self.execute_move_and_check_state(self.turn, move_to_check)
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
        if self.game_mode.get() != GameMode.AI_VS_AI.value: self.ai_series_running = False
        self._stop_ai_process()
        self.board = Board()
        self.turn = "white"
        self.game_started = False
        self.last_move_timestamp = time.time()
        self.selected, self.valid_moves = None,[]
        
        self._reset_game_state_vars()
        self.fen_entry.delete(0, tk.END); self.pgn_entry.delete(0, tk.END)
        self._start_game_if_needed()
        
        mode = self.game_mode.get()
        if mode == GameMode.AI_VS_AI.value:
            self.white_playing_bot_type = "op" if self.ai_series_running and self.ai_series_stats['game_count'] % 2 == 1 else "main"
            self.board_orientation = "white" if self.white_playing_bot_type == "main" else "black"
            if self.ai_series_running: self.apply_series_opening_move()
            if not self.game_over: self.master.after(4 if self.instant_move.get() else 20, self._make_game_ai_move)
        elif mode == GameMode.HUMAN_VS_BOT.value:
            self.board_orientation = self.human_color
            if self.turn != self.human_color: self.master.after(4 if self.instant_move.get() else 20, self._make_game_ai_move)
        else: self.board_orientation = "white"
        self.update_ui_after_state_change()
        self._update_analysis_after_state_change()

    def _make_game_ai_move(self):
        if self.game_over: return
        print(f"\n--- Turn {self.history_pointer + 1} ({self.turn.capitalize()}) ---")
        self.last_move_timestamp = time.time()
        mode = self.game_mode.get()
        bot_class, bot_name = None, None
        if mode == GameMode.HUMAN_VS_BOT.value:
            if self.turn != self.human_color: bot_class, bot_name = ChessBot, self.MAIN_AI_NAME
        elif mode == GameMode.AI_VS_AI.value:
            main_ai_color = "white" if self.white_playing_bot_type == "main" else "black"
            bot_class, bot_name = (ChessBot, self.MAIN_AI_NAME) if self.turn == main_ai_color else (OpponentAI, self.OPPONENT_AI_NAME)
        if bot_class: self._start_ai_process(bot_class, bot_name, self.bot_depth_slider.get())

    def update_ui_after_state_change(self):
        self.selected, self.valid_moves, self.valid_moves_for_highlight = None, [],[]
        self.update_turn_label()
        self.update_game_info_label()
        self.update_bot_labels()
        self.update_moves_list()
        self.draw_board()
        self.update_navigation_buttons()

    def _navigate_history(self, target_index):
        if self.game_mode.get() == GameMode.AI_VS_AI.value: return
        new_index = max(0, min(target_index, len(self.full_history) - 1))
        if new_index != self.history_pointer:
            self.history_pointer = new_index
            self._load_state_from_history()

    def _load_state_from_history(self):
        self._stop_ai_process()
        board_state, turn_state, _ = self.full_history[self.history_pointer]
        self.board = board_state.clone(); self.turn = turn_state
        self.game_over, self.game_result = False, None
        
        self.position_counts.clear()
        for i in range(self.history_pointer + 1):
            board, turn, _ = self.full_history[i]
            h = board_hash(board, turn)
            self.position_counts[h] = self.position_counts.get(h, 0) + 1

        status, winner = get_game_state(self.board, self.turn, self.position_counts, self.history_pointer, self.MAX_GAME_MOVES)
        if status != "ongoing": self.game_over, self.game_result = True, (status, winner)
            
        self.update_ui_after_state_change()
        self._update_analysis_after_state_change()

    def update_game_info_label(self):
        mode = self.game_mode.get()
        if mode == GameMode.HUMAN_VS_BOT.value: text = f"Human vs {self.MAIN_AI_NAME}"
        elif mode == GameMode.AI_VS_AI.value: text = f"{self.MAIN_AI_NAME} vs {self.OPPONENT_AI_NAME}"
        else: text = "Human vs Human Analysis"
        self.game_info_label.config(text=text)

    def create_board_image(self, orientation):
        if self.square_size <= 0: return None
        img = tk.PhotoImage(width=COLS*self.square_size, height=ROWS*self.square_size)
        BOARD_COLOR_1, BOARD_COLOR_2 = "#D2B48C", "#8B5A2B" 
        for r_draw in range(ROWS):
            for c_draw in range(COLS):
                color = BOARD_COLOR_1 if (r_draw + c_draw) % 2 == 0 else BOARD_COLOR_2
                x1, y1 = c_draw * self.square_size, r_draw * self.square_size
                img.put(color, to=(x1, y1, x1 + self.square_size, y1 + self.square_size))
        return img

    def draw_board(self):
        current_image = self.board_image_white if self.board_orientation == "white" else self.board_image_black
        if not current_image: return
        self.canvas.itemconfig(self.board_image_id, image=current_image)
        self.canvas.delete("highlight", "piece", "check_highlight", "border_highlight")
        
        draw_border = False
        mode = self.game_mode.get()
        if mode == GameMode.HUMAN_VS_BOT.value:
            if self.board_orientation != self.human_color:
                draw_border = True
        else:
            if self.board_orientation == "black":
                draw_border = True

        if draw_border:
            w, h = COLS * self.square_size, ROWS * self.square_size
            self.canvas.create_rectangle(2, 2, w-2, h-2, outline=self.COLORS['warning'], width=4, tags="border_highlight")

        for r_move, c_move in getattr(self, 'valid_moves_for_highlight',[]):
            x1, y1 = self.board_to_canvas(r_move, c_move)
            radius = self.square_size // 5
            center_x, center_y = x1 + self.square_size // 2, y1 + self.square_size // 2
            self.canvas.create_oval(center_x - radius, center_y - radius, center_x + radius, center_y + radius, fill="#1E90FF", outline="", tags="highlight")
        for r in range(ROWS):
            for c in range(COLS):
                if self.board.grid[r][c]: self.draw_piece_with_check(r, c)
        self._position_side_labels()

    def _position_side_labels(self):
        if not hasattr(self, "canvas"):
            return
        desired_label_width = max(96, int(self.square_size * 1.9))

        self.board_row_frame.update_idletasks()
        canvas_x = self.canvas.winfo_x()
        canvas_y = self.canvas.winfo_y()
        board_h = ROWS * self.square_size
        available_left = max(1, canvas_x - 10)
        label_width = min(desired_label_width, available_left)

        if label_width < 20:
            self.top_bot_label.place_forget()
            self.bottom_bot_label.place_forget()
            return

        self.top_bot_label.config(wraplength=max(1, label_width - 6), anchor="center", justify=tk.CENTER)
        self.bottom_bot_label.config(wraplength=max(1, label_width - 6), anchor="center", justify=tk.CENTER)
        strip_left = 2
        strip_right = max(strip_left + 1, canvas_x - 8)
        strip_width = max(1, strip_right - strip_left)
        left_x = strip_left + max(0, (strip_width - label_width) // 2)

        self.top_bot_label.place(in_=self.board_row_frame, x=left_x, y=canvas_y + 4, width=label_width, anchor="nw")
        self.bottom_bot_label.place(in_=self.board_row_frame, x=left_x, y=canvas_y + board_h - 4, width=label_width, anchor="sw")

    def draw_piece_with_check(self, r, c):
        piece = self.board.grid[r][c]
        if isinstance(piece, King):
            # Check if this King has lost (Checkmate OR Trap)
            is_lost = (self.game_over and self.game_result and 
                       self.game_result[0] == "checkmate" and 
                       self.game_result[1] != piece.color)
            
            # Check if currently in danger
            is_checked = is_in_check(self.board, piece.color)

            color = None
            if is_lost:
                color = "darkred" # Final blow / Trapped
            elif is_checked:
                color = "red"     # Warning

            if color:
                x1, y1 = self.board_to_canvas(r, c)
                self.canvas.create_rectangle(x1, y1, x1 + self.square_size, y1 + self.square_size, outline=color, width=4, tags="check_highlight")

        if (r, c) != self.drag_start: self.draw_piece_at_canvas_coords(piece, r, c)

    def draw_piece_at_canvas_coords(self, piece, r, c):
        x, y = self.board_to_canvas(r, c)
        x_center, y_center = x + self.square_size // 2, y + self.square_size // 2 + 2
        font_size = int(self.square_size * 0.67)
        symbol, font = piece.symbol(), ("Arial Unicode MS", font_size)
        self.canvas.create_text(x_center + 1, y_center + 1, text=symbol, font=font, fill="#888888", tags="piece")
        fill_color = "#000000" if piece.color == "black" else "#FFFFFF"
        self.canvas.create_text(x_center, y_center, text=symbol, font=font, fill=fill_color, tags="piece")
        
    def draw_eval_bar(self, eval_score, depth=None):
        score = eval_score / 100.0
        w, h = self.eval_bar_canvas.winfo_width(), self.eval_bar_canvas.winfo_height()
        if w <= 1 or h <= 1: return
        if w != self.last_eval_bar_w or h != self.last_eval_bar_h:
            self.eval_bar_canvas.delete("gradient")
            for x_pixel in range(w):
                intensity = int(255 * (x_pixel / float(w - 1)))
                color = f"#{intensity:02x}{intensity:02x}{intensity:02x}"
                self.eval_bar_canvas.create_line(x_pixel, 0, x_pixel, h, fill=color, tags="gradient")
            self.last_eval_bar_w, self.last_eval_bar_h = w, h
        self.eval_bar_canvas.delete("marker")
        marker_score = max(-1.0, min(1.0, math.tanh(score / 10.0)))
        marker_x = int(((marker_score + 1) / 2.0) * w)
        self.eval_bar_canvas.create_line(marker_x, 0, marker_x, h, fill="#FF0000", width=3, tags="marker")
        self.eval_bar_canvas.create_line(w // 2, 0, w // 2, h, fill="#00FF00", width=2, tags="marker") 
        depth_suffix = f" (D{depth})" if depth is not None else ""
        eval_text = f"{'+' if score > 0 else ''}{score:.2f}{depth_suffix}"
        if abs(score) < 0.05:
            eval_text = f"Even{depth_suffix}"
        self.eval_score_label.config(text=eval_text)

    def process_comm_queue(self):
        try:
            while not self.comm_queue.empty():
                message = self.comm_queue.get_nowait()
                if message[0] == 'log':
                    print(message[1])
                    if self.auto_save_stats_var.get() and self.game_mode.get() == GameMode.AI_VS_AI.value:
                        match = re.search(r'> (.*?) \(D(\d+|TB)\).*?Time=([0-9.]+)s', message[1])
                        if match:
                            bot_name = match.group(1).strip()
                            depth = match.group(2)
                            time_val = float(match.group(3))
                            
                            if bot_name not in self.depth_stats: self.depth_stats[bot_name] = {}
                            if depth not in self.depth_stats[bot_name]: self.depth_stats[bot_name][depth] = []
                            self.depth_stats[bot_name][depth].append(time_val)
                            
                            if 'Global' not in self.depth_stats: self.depth_stats['Global'] = {}
                            if depth not in self.depth_stats['Global']: self.depth_stats['Global'][depth] = []
                            self.depth_stats['Global'][depth].append(time_val)
                elif message[0] == 'eval': 
                    self.last_eval_score, self.last_eval_depth = message[1], message[2]
                    self.draw_eval_bar(self.last_eval_score, self.last_eval_depth)
                elif message[0] == 'move': 
                    self._execute_ai_move(message[1])
        except Exception: pass
        finally: self.master.after(100, self.process_comm_queue)

    def save_depth_stats_to_file(self):
        if not self.depth_stats or not self.depth_stats.get('Global'):
            return
            
        filename = "AI_Series_Depth_Averages.txt"
        try:
            with open(filename, "w") as f:
                f.write("=== AI vs OP Series Depth Stats ===\n")
                f.write(f"Games Completed: {self.ai_series_stats['game_count']}\n")
                f.write(f"Score: {self.MAIN_AI_NAME} {self.ai_series_stats['my_ai_wins']} - {self.ai_series_stats['op_ai_wins']} {self.OPPONENT_AI_NAME} (Draws: {self.ai_series_stats['draws']})\n\n")
                
                def sort_key(k): return int(k) if k.isdigit() else 999
                
                for category in ['Global', self.MAIN_AI_NAME, self.OPPONENT_AI_NAME]:
                    if category not in self.depth_stats or not self.depth_stats[category]:
                        continue
                        
                    f.write(f"--- {category} Averages ---\n")
                    for depth in sorted(self.depth_stats[category].keys(), key=sort_key):
                        times = self.depth_stats[category][depth]
                        avg_time = sum(times) / len(times)
                        max_time = max(times)
                        f.write(f"  Depth {depth:<3} | Avg: {avg_time:.3f}s | Max: {max_time:.3f}s | Samples: {len(times)}\n")
                    f.write("\n")
        except Exception as e:
            print(f"Failed to save stats to file: {e}")

    def _start_ai_process(self, bot_class, bot_name, search_depth):
        if self.ai_process and self.ai_process.is_alive(): return
        self.ai_cancellation_event.clear()
        args = (self.board.clone(), self.turn, self.position_counts.copy(), self.comm_queue, self.ai_cancellation_event, bot_class, bot_name, search_depth, self.history_pointer, self.game_mode.get())
        self.ai_process = mp.Process(target=run_ai_process, args=args, daemon=True)
        self.ai_process.name = bot_name; self.ai_process.start()
        if bot_name != self.ANALYSIS_AI_NAME: self.set_interactivity(False)
        self.update_bot_labels()

    def _stop_ai_process(self):
        if self.ai_process and self.ai_process.is_alive():
            self.ai_cancellation_event.set(); self.ai_process.join(timeout=0.1) 
            if self.ai_process.is_alive(): self.ai_process.terminate()
            self.ai_process = None
        while not self.comm_queue.empty():
            try: self.comm_queue.get_nowait()
            except Exception: break
        if self.game_mode.get() == GameMode.HUMAN_VS_HUMAN.value and not self.analysis_mode_var.get():
            self.last_eval_score, self.last_eval_depth = 0.0, None
            self.draw_eval_bar(0); self.eval_score_label.config(text="Even")
        self.set_interactivity(True); self.update_bot_labels()

    def _update_analysis_after_state_change(self):
        self._stop_ai_process()
        if self.analysis_mode_var.get() and self.game_mode.get() == GameMode.HUMAN_VS_HUMAN.value and not self.game_over:
            # Add a readable boundary so consecutive analysis blocks are easier to scan.
            # history_pointer is the current ply index from the initial position (ply 0).
            fullmove = (self.history_pointer + 1) // 2 + 1
            print(f"\n--- Analysis: Move {fullmove}, Ply {self.history_pointer}, {self.turn.capitalize()} to move ---")
            self.master.after(50, lambda: self._start_ai_process(ChessBot, self.ANALYSIS_AI_NAME, 99))

    def _start_game_if_needed(self):
        if not self.game_started: self.game_started = True

    def _log_game_over(self):
        if self.game_result: print(f"Game Over! Result: {self.game_result[0]}")

    def update_turn_label(self):
        msg = f"TURN: {self.turn.upper()}"
        if self.game_result: msg = f"GAME OVER: {self.game_result[0].upper()}"
        self.turn_label.config(text=msg)

    def update_bot_labels(self):
        mode = self.game_mode.get()
        if mode == GameMode.AI_VS_AI.value:
            white_label = self.MAIN_AI_NAME if self.white_playing_bot_type == "main" else self.OPPONENT_AI_NAME
            black_label = self.OPPONENT_AI_NAME if self.white_playing_bot_type == "main" else self.MAIN_AI_NAME
        elif mode == GameMode.HUMAN_VS_BOT.value:
            white_label = "Human" if self.human_color == "white" else self.MAIN_AI_NAME
            black_label = "Human" if self.human_color == "black" else self.MAIN_AI_NAME
        else:
            white_label = "White"
            black_label = "Black"

        if self.turn == "white":
            white_label += "\n(to move)"
        else:
            black_label += "\n(to move)"

        self.bottom_bot_label.config(text=white_label if self.board_orientation == 'white' else black_label)
        self.top_bot_label.config(text=black_label if self.board_orientation == 'white' else white_label)
        self._position_side_labels()

    def set_interactivity(self, is_interactive):
        if is_interactive:
            self.canvas.bind("<Button-1>", self.on_drag_start); self.canvas.bind("<B1-Motion>", self.on_drag_motion); self.canvas.bind("<ButtonRelease-1>", self.on_drag_end)
        else:
            self.canvas.unbind("<Button-1>"); self.canvas.unbind("<B1-Motion>"); self.canvas.unbind("<ButtonRelease-1>")

    def is_ai_thinking(self): return self.ai_process and self.ai_process.is_alive()

    def switch_turn(self):
        if not self.game_over: self.turn = "black" if self.turn == "white" else "white"

    def board_to_canvas(self, r, c):
        x = (COLS - 1 - c) * self.square_size if self.board_orientation == "black" else c * self.square_size
        y = (ROWS - 1 - r) * self.square_size if self.board_orientation == "black" else r * self.square_size
        return x, y

    def canvas_to_board(self, x, y):
        c = (COLS - 1) - (x // self.square_size) if self.board_orientation == "black" else x // self.square_size
        r = (ROWS - 1) - (y // self.square_size) if self.board_orientation == "black" else y // self.square_size
        return (r, c) if 0 <= r < ROWS and 0 <= c < COLS else (-1, -1)

    def on_drag_start(self, event):
        if self.game_over or (self.is_ai_thinking() and self.ai_process.name != self.ANALYSIS_AI_NAME): return
        r, c = self.canvas_to_board(event.x, event.y)
        if r != -1 and self.board.grid[r][c] and self.board.grid[r][c].color == self.turn:
            if self.game_mode.get() != GameMode.HUMAN_VS_BOT.value or self.turn == self.human_color:
                self.selected, self.dragging, self.drag_start = (r, c), True, (r, c)
                self.valid_moves = get_all_legal_moves(self.board, self.turn)
                self.valid_moves_for_highlight =[end for start, end in self.valid_moves if start == self.selected]
                self.drag_piece_ghost = self.canvas.create_text(event.x, event.y, text=self.board.grid[r][c].symbol(), font=("Arial Unicode MS", int(self.square_size * 0.7)), fill=self.turn, tags="drag_ghost")
                self.draw_board(); self.canvas.tag_raise("drag_ghost")
    
    def on_drag_motion(self, event):
        if self.dragging: self.canvas.coords(self.drag_piece_ghost, event.x, event.y)

    def undo_move(self): self._navigate_history(self.history_pointer - 1)
    def redo_move(self): self._navigate_history(self.history_pointer + 1)
    def go_to_start(self): self._navigate_history(0)
    def go_to_end(self): self._navigate_history(len(self.full_history) - 1)
    
    def update_navigation_buttons(self):
        if self.game_mode.get() == GameMode.AI_VS_AI.value:
            for btn in[self.start_button, self.undo_button, self.redo_button, self.end_button]: btn.config(state=tk.DISABLED)
            return
        can_back = self.history_pointer > 0; can_fwd = self.history_pointer < len(self.full_history) - 1
        self.start_button.config(state=tk.NORMAL if can_back else tk.DISABLED); self.undo_button.config(state=tk.NORMAL if can_back else tk.DISABLED)
        self.redo_button.config(state=tk.NORMAL if can_fwd else tk.DISABLED); self.end_button.config(state=tk.NORMAL if can_fwd else tk.DISABLED)

    # --- AI SERIES LOGIC ---
    def process_ai_series_result(self):
        self.ai_series_stats['game_count'] += 1
        _, winner_color = self.game_result
        if winner_color:
            main_ai_color = 'white' if self.white_playing_bot_type == 'main' else 'black'
            if winner_color == main_ai_color: self.ai_series_stats['my_ai_wins'] += 1
            else: self.ai_series_stats['op_ai_wins'] += 1
        else: self.ai_series_stats['draws'] += 1
        
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
        self.ai_series_stats = {'game_count': 0, 'my_ai_wins': 0, 'op_ai_wins': 0, 'draws': 0}
        self.depth_stats = {}
        self.ai_series_running = True
        self.current_opening_move = None
        self.reset_game()
        
    def apply_series_opening_move(self):
        if self.ai_series_stats['game_count'] % 2 == 0:
            print("\n--- Generating new opening for game pair ---")
            moves = get_all_legal_moves(self.board, "white")
            self.current_opening_move = random.choice(moves) if moves else None
        
        if self.current_opening_move:
            child = self.board.clone()
            child.make_move(self.current_opening_move[0], self.current_opening_move[1])
            san = format_move_san(self.board, child, self.current_opening_move)
            print(f"Applying opening move: {san}")
            
            self.board.make_move(self.current_opening_move[0], self.current_opening_move[1])
            self.execute_move_and_check_state(self.turn, self.current_opening_move)

    def update_scoreboard(self):
        if self.game_mode.get() == GameMode.AI_VS_AI.value and self.ai_series_running:
            stats = self.ai_series_stats
            game_display_count = stats['game_count'] + 1
            score_text = (f"{self.MAIN_AI_NAME} vs {self.OPPONENT_AI_NAME} (Game {game_display_count}/100)\n"
                          f"  {self.MAIN_AI_NAME} Wins: {stats['my_ai_wins']}\n"
                          f"  {self.OPPONENT_AI_NAME} Wins: {stats['op_ai_wins']}\n"
                          f"  Draws: {stats['draws']}")
            self.scoreboard_label.config(text=score_text)
        else:
            self.scoreboard_label.config(text="")

if __name__ == "__main__":
    mp.freeze_support()
    root = tk.Tk()
    app = EnhancedChessApp(root)
    root.mainloop()