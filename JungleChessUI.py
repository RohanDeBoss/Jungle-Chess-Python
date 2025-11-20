# JungleChessUI.py (v9.2 - Optimized Eval Bar Rendering)
import tkinter as tk
from tkinter import ttk
import math
import random
import time
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
    if bot_class == ChessBot:
        bot = bot_class(board, color, position_counts, comm_queue, cancellation_event, bot_name, ply_count, game_mode)
    else:
        bot = bot_class(board, color, position_counts, comm_queue, cancellation_event, bot_name)
        
    bot.search_depth = search_depth
    if search_depth == 99:
        bot.ponder_indefinitely()
    else:
        bot.make_move()

class EnhancedChessApp:
    MAIN_AI_NAME = "AI Bot"
    OPPONENT_AI_NAME = "OP Bot"
    ANALYSIS_AI_NAME = "Analysis"
    slidermaxvalue = 10
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
        self.valid_moves = []
        self.game_over = False
        self.game_result = None
        self.dragging = False
        self.drag_piece_ghost = None
        self.drag_start = None

        self.full_history = []
        self.history_pointer = -1
        self.position_counts = {}

        self.current_opening_move = None
        self.square_size = 75
        self.base_sidebar_width = 265

        self.game_mode = tk.StringVar(value=GameMode.HUMAN_VS_BOT.value)
        self.analysis_mode_var = tk.BooleanVar(value=True)
        self.ai_series_running = False
        self.ai_series_stats = {'game_count': 0, 'my_ai_wins': 0, 'op_ai_wins': 0, 'draws': 0}
        
        self.white_playing_bot_type = "main"
        self.human_color = "white"
        self.board_orientation = "white"
        self.last_move_timestamp = None
        self.game_started = False 

        self.last_eval_score = 0.0
        self.last_eval_depth = None
        
        # Optimization: Track eval bar dimensions to avoid needless redraws
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
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.left_panel = ttk.Frame(self.main_frame, style='Left.TFrame')
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10); self.left_panel.pack_propagate(False)
        
        self.top_controls_frame = ttk.Frame(self.left_panel, style='Left.TFrame')
        self.top_controls_frame.pack(side=tk.TOP, fill=tk.X, expand=False)
        self.bottom_status_frame = ttk.Frame(self.left_panel, style='Left.TFrame')
        self.bottom_status_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(1, 0))
        
        self.title_label = ttk.Label(self.top_controls_frame, text="JUNGLE CHESS", style='Header.TLabel', font=('Helvetica', 24, 'bold'))
        self.title_label.pack(pady=(0,10))
        
        self._build_control_widgets(self.top_controls_frame)
        self._build_status_widgets(self.bottom_status_frame)
        
        self.right_panel = ttk.Frame(self.main_frame, style='Right.TFrame')
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 10), pady=10)
        
        self.canvas_container = ttk.Frame(self.right_panel, style='Canvas.TFrame')
        self.canvas_container.pack(expand=True, fill=tk.BOTH)
        self.canvas_container.grid_rowconfigure(0, weight=1); self.canvas_container.grid_columnconfigure(0, weight=1)
        
        self.canvas_frame = ttk.Frame(self.canvas_container, style='Canvas.TFrame')
        
        self.canvas = tk.Canvas(self.canvas_frame, width=COLS * self.square_size, height=ROWS * self.square_size, bg=self.COLORS['bg_light'], highlightthickness=0)
        
        self.board_image_white = self.create_board_image("white")
        self.board_image_black = self.create_board_image("black")
        self.board_image_id = self.canvas.create_image(0, 0, anchor='nw', tags="board")
        self.canvas.pack()

        self.navigation_frame = ttk.Frame(self.right_panel, style='Right.TFrame')
        self.navigation_frame.pack(fill=tk.X, pady=(10, 0))
        self.start_button = ttk.Button(self.navigation_frame, text="«", command=self.go_to_start, style='Nav.TButton', state=tk.DISABLED)
        self.undo_button = ttk.Button(self.navigation_frame, text="‹", command=self.undo_move, style='Nav.TButton', state=tk.DISABLED)
        self.redo_button = ttk.Button(self.navigation_frame, text="›", command=self.redo_move, style='Nav.TButton', state=tk.DISABLED)
        self.end_button = ttk.Button(self.navigation_frame, text="»", command=self.go_to_end, style='Nav.TButton', state=tk.DISABLED)
        self.navigation_frame.columnconfigure(0, weight=1)
        self.navigation_frame.columnconfigure(5, weight=1)
        self.start_button.grid(row=0, column=1, padx=5); self.undo_button.grid(row=0, column=2, padx=5)
        self.redo_button.grid(row=0, column=3, padx=5); self.end_button.grid(row=0, column=4, padx=5)

        self._build_scoreboard_and_labels()
        
        self.main_frame.bind("<Configure>", self.handle_main_resize)
        self.right_panel.bind("<Configure>", self.handle_board_resize)
        
    def handle_key_press(self, event):
        if self.is_ai_thinking() and self.ai_process.name != self.ANALYSIS_AI_NAME: return
        if event.keysym == 'Left': self.undo_move()
        elif event.keysym == 'Right': self.redo_move()
        elif event.keysym == 'Home': self.go_to_start()
        elif event.keysym == 'End': self.go_to_end()

    def handle_main_resize(self, event):
        new_sidebar_width = max(self.base_sidebar_width, int(event.width * 0.22))
        if new_sidebar_width == self.left_panel.winfo_width(): return
        self.left_panel.config(width=new_sidebar_width)
        raw_scaling_factor = new_sidebar_width / self.base_sidebar_width
        dampened_scaling_factor = 1.0 + (raw_scaling_factor - 1.0) * 0.5
        title_font_size = max(20, int(24 * dampened_scaling_factor))
        header_font_size = max(12, int(14 * dampened_scaling_factor))
        small_header_font_size = max(11, int(13 * dampened_scaling_factor))
        control_font_size = max(10, int(11 * dampened_scaling_factor))
        status_font_size = max(12, int(14 * dampened_scaling_factor))
        self.title_label.config(font=('Helvetica', title_font_size, 'bold'))
        style = ttk.Style()
        style.configure('Header.TLabel', font=('Helvetica', header_font_size, 'bold'))
        style.configure('SmallHeader.TLabel', font=('Helvetica', small_header_font_size, 'bold'))
        style.configure('Control.TButton', font=('Helvetica', control_font_size, 'bold'))
        style.configure('Custom.TRadiobutton', font=('Helvetica', control_font_size))
        style.configure('Custom.TCheckbutton', font=('Helvetica', control_font_size))
        style.configure('Status.TLabel', font=('Helvetica', status_font_size, 'bold'))
        button_padding_y = max(6, int(7 * dampened_scaling_factor))
        style.configure('Control.TButton', padding=(10, button_padding_y))
        new_eval_bar_height = max(26, int(26 * dampened_scaling_factor))
        self.eval_bar_canvas.config(height=new_eval_bar_height)
            
    def _build_control_widgets(self, parent_frame):
        self.game_mode_frame = ttk.Frame(parent_frame, style='Left.TFrame')
        self.game_mode_frame.pack(fill=tk.X, pady=(0, 2))
        ttk.Label(self.game_mode_frame, text="GAME MODE", style='Header.TLabel').pack(anchor=tk.W)
        for mode in GameMode:
            text = mode.name.replace("_", " ").title()
            ttk.Radiobutton(self.game_mode_frame, text=text, variable=self.game_mode, value=mode.value, command=self.reset_game, style='Custom.TRadiobutton').pack(anchor=tk.W, pady=(2,0))
        
        self.controls_frame = ttk.Frame(parent_frame, style='Left.TFrame')
        self.controls_frame.pack(fill=tk.X, pady=3)
        ttk.Button(self.controls_frame, text="NEW GAME", command=self.reset_game, style='Control.TButton').pack(fill=tk.X, pady=3)
        ttk.Button(self.controls_frame, text="SWAP SIDES", command=self.swap_sides, style='Control.TButton').pack(fill=tk.X, pady=3)
        ttk.Button(self.controls_frame, text="AI vs OP Series", command=self.start_ai_series, style='Control.TButton').pack(fill=tk.X, pady=3)
        ttk.Button(self.controls_frame, text="QUIT", command=self.master.quit, style='Control.TButton').pack(fill=tk.X, pady=3)

        ttk.Label(self.controls_frame, text="Bot Depth:", style='SmallHeader.TLabel').pack(anchor=tk.W, pady=(2,0))
        self.bot_depth_slider = tk.Scale(self.controls_frame, from_=1, to=self.slidermaxvalue, orient=tk.HORIZONTAL, bg=self.COLORS['bg_dark'], fg=self.COLORS['text_light'], highlightthickness=0, relief='flat')
        self.bot_depth_slider.set(3); self.bot_depth_slider.pack(fill=tk.X, pady=(0,2))
        self.instant_move = tk.BooleanVar(value=False); ttk.Checkbutton(self.controls_frame, text="Instant Moves", variable=self.instant_move, style='Custom.TCheckbutton').pack(anchor=tk.W, pady=(2,2))
        self.analysis_checkbox = ttk.Checkbutton(self.controls_frame, text="Analysis Mode (H-vs-H)", variable=self.analysis_mode_var, style='Custom.TCheckbutton', command=self._update_analysis_after_state_change)
        self.analysis_checkbox.pack(anchor=tk.W, pady=(2,2))

    def _build_status_widgets(self, parent_frame):
        self.turn_frame = ttk.Frame(parent_frame, style='Left.TFrame'); self.turn_frame.pack(fill=tk.X, pady=(8,0))
        self.turn_label = ttk.Label(self.turn_frame, text="WHITE'S TURN", style='Status.TLabel'); self.turn_label.pack(fill=tk.X)
        self.eval_frame = ttk.Frame(parent_frame, style='Left.TFrame'); self.eval_frame.pack(fill=tk.X, pady=(9, 4))
        self.eval_score_label = ttk.Label(self.eval_frame, text="Even", style='Status.TLabel', anchor="center"); self.eval_score_label.pack(pady=(6,4))
        self.eval_bar_canvas = tk.Canvas(self.eval_frame, height=26, bg=self.COLORS['bg_light'], highlightthickness=0); self.eval_bar_canvas.pack(fill=tk.X, expand=True)
        self.eval_bar_canvas.bind("<Configure>", self.redraw_eval_bar_on_resize)

    def redraw_eval_bar_on_resize(self, event):
        self.draw_eval_bar(self.last_eval_score, self.last_eval_depth)

    def _build_scoreboard_and_labels(self):
        self.scoreboard_frame = ttk.Frame(self.right_panel, style='Right.TFrame')
        self.scoreboard_label = ttk.Label(self.scoreboard_frame, text="", font=("Helvetica", 10), justify=tk.LEFT, background=self.COLORS['bg_medium'], foreground=self.COLORS['text_light'])
        self.scoreboard_label.pack()
        
        self.top_bot_label = ttk.Label(self.canvas_container, text="", font=("Helvetica", 12), background=self.COLORS['bg_medium'], foreground=self.COLORS['text_light'])
        self.bottom_bot_label = ttk.Label(self.canvas_container, text="", font=("Helvetica", 12), background=self.COLORS['bg_medium'], foreground=self.COLORS['text_light'])

        self.top_bot_label.pack(side=tk.TOP, pady=(5, 2))
        self.canvas_frame.pack(side=tk.TOP, expand=True)
        self.bottom_bot_label.pack(side=tk.TOP, pady=(2, 5))

    def handle_board_resize(self, event):
        view_width, view_height = event.width - 20, event.height - 120 
        if view_width <= 1 or view_height <= 1: return
        
        new_square_size = min(view_width // COLS, view_height // ROWS)
        if new_square_size != self.square_size and new_square_size > 0:
            self.square_size = new_square_size
            self.canvas.config(width=COLS * self.square_size, height=ROWS * self.square_size)
            self.board_image_white = self.create_board_image("white")
            self.board_image_black = self.create_board_image("black")
            self.draw_board()    

    def draw_piece_at_canvas_coords(self, piece, r, c):
        x, y = self.board_to_canvas(r, c)
        x_center, y_center = x + self.square_size // 2, y + self.square_size // 2 + 2
        font_size = int(self.square_size * 0.67)
        y_center -= int(font_size / 20)
        symbol, font = piece.symbol(), ("Arial Unicode MS", font_size)
        shadow_offset = max(1, self.square_size // (40 if piece.color == "black" else 25))
        self.canvas.create_text(x_center + shadow_offset, y_center + shadow_offset, text=symbol, font=font, fill="#444444", tags="piece")
        self.canvas.create_text(x_center, y_center, text=symbol, font=font, fill=piece.color, tags="piece")

    def process_comm_queue(self):
        try:
            while not self.comm_queue.empty():
                message = self.comm_queue.get_nowait()
                msg_type, *payload = message
                if msg_type == 'log': print(payload[0])
                elif msg_type == 'eval':
                    self.last_eval_score, self.last_eval_depth = payload[0], payload[1]
                    self.draw_eval_bar(self.last_eval_score, self.last_eval_depth)
                elif msg_type == 'move': self._execute_ai_move(payload[0])
        except Exception: pass
        finally: self.master.after(100, self.process_comm_queue)

    def _execute_ai_move(self, the_move):
        if self.game_over: return
        if the_move:
            self.board.make_move(the_move[0], the_move[1])
            self.execute_move_and_check_state(player_who_moved=self.turn)
            delay = 4 if self.instant_move.get() else 20
            if not self.game_over and self.game_mode.get() == GameMode.AI_VS_AI.value:
                self.master.after(delay, self._make_game_ai_move)
        else: print("AI reported no valid move was made or was cancelled.")
        
        self._stop_ai_process()
        self.update_bot_labels(); self.set_interactivity(True)
    
    def _start_ai_process(self, bot_class, bot_name, search_depth):
        if self.ai_process and self.ai_process.is_alive(): return
        self.ai_cancellation_event.clear()
        ply_count = self.history_pointer
        game_mode_str = self.game_mode.get()
        args = (self.board.clone(), self.turn, self.position_counts.copy(), self.comm_queue, self.ai_cancellation_event, bot_class, bot_name, search_depth, ply_count, game_mode_str)
        self.ai_process = mp.Process(target=run_ai_process, args=args, daemon=True)
        self.ai_process.name = bot_name; self.ai_process.start()
        if bot_name != self.ANALYSIS_AI_NAME: self.set_interactivity(False)
        self.update_bot_labels()

    def _stop_ai_process(self):
        if self.ai_process and self.ai_process.is_alive():
            self.ai_cancellation_event.set()
            self.ai_process.join(timeout=0.1) 
            if self.ai_process.is_alive(): self.ai_process.terminate()
            self.ai_process = None
        while not self.comm_queue.empty():
            try: self.comm_queue.get_nowait()
            except Exception: break
        if self.game_mode.get() == GameMode.HUMAN_VS_HUMAN.value and not self.analysis_mode_var.get():
            self.last_eval_score, self.last_eval_depth = 0.0, None
            self.draw_eval_bar(0); self.eval_score_label.config(text="Even")
        self.set_interactivity(True); self.update_bot_labels()

    def reset_game(self):
        if self.game_mode.get() != GameMode.AI_VS_AI.value:
            self.ai_series_running = False
        self._stop_ai_process()
        self.board = Board()
        self.turn = "white"
        self.game_started = False
        self.last_move_timestamp = time.time()
        self.selected, self.valid_moves, self.game_over, self.game_result = None, [], False, None
        self.full_history = [(self.board.clone(), self.turn)]
        self.history_pointer = 0
        self.position_counts = {board_hash(self.board, self.turn): 1}
        self.last_eval_score, self.last_eval_depth = 0.0, None
        self.draw_eval_bar(0)
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
            bot_class, bot_name = (ChessBot, self.MAIN_AI_NAME) if self.turn == self.board_orientation else (OpponentAI, self.OPPONENT_AI_NAME)
        if bot_class: self._start_ai_process(bot_class, bot_name, self.bot_depth_slider.get())
    
    def _update_analysis_after_state_change(self):
        self._stop_ai_process()
        should_run = (self.analysis_mode_var.get() and self.game_mode.get() == GameMode.HUMAN_VS_HUMAN.value and not self.game_over)
        if should_run: self.master.after(50, lambda: self._start_ai_process(ChessBot, self.ANALYSIS_AI_NAME, 99))

    def on_drag_end(self, event):
        is_analysis = self.is_ai_thinking() and self.ai_process.name == self.ANALYSIS_AI_NAME
        if self.is_ai_thinking() and not is_analysis:
            self.valid_moves = []; self.draw_board(); return
        if not self.dragging:
            self.valid_moves = []; self.draw_board(); return

        self.dragging = False; self.canvas.delete("drag_ghost")
        row, col = self.canvas_to_board(event.x, event.y)
        if row == -1 or not self.drag_start:
            self.update_ui_after_state_change(); self.set_interactivity(True); return

        start_pos, end_pos = self.drag_start, (row, col)
        move_to_check = (start_pos, end_pos)

        if move_to_check in self.valid_moves:
            move_time = time.time() - self.last_move_timestamp
            print(f"\n--- Turn {self.history_pointer + 1} ({self.turn.capitalize()}) ---")
            print(f"Human played: {format_move(move_to_check)}, Time={move_time:.2f}s")
            self.board.make_move(start_pos, end_pos)
            self.execute_move_and_check_state(player_who_moved=self.turn)
            if not self.game_over:
                mode = self.game_mode.get()
                if mode == GameMode.HUMAN_VS_BOT.value and self.turn != self.human_color:
                    self.master.after(4 if self.instant_move.get() else 20, self._make_game_ai_move)
                elif mode == GameMode.HUMAN_VS_HUMAN.value:
                    self._update_analysis_after_state_change()
        self.drag_start = None
        self.update_ui_after_state_change()
        self.set_interactivity(True)

    def execute_move_and_check_state(self, player_who_moved):
        self.switch_turn()
        if self.history_pointer < len(self.full_history) - 1:
            self.full_history = self.full_history[:self.history_pointer + 1]
            self.position_counts.clear()
            for board, turn in self.full_history:
                self.position_counts[board_hash(board, turn)] = self.position_counts.get(board_hash(board, turn), 0) + 1
        self.full_history.append((self.board.clone(), self.turn))
        self.history_pointer += 1
        key = board_hash(self.board, self.turn)
        self.position_counts[key] = self.position_counts.get(key, 0) + 1
        status, winner = get_game_state(self.board, self.turn, self.position_counts, self.history_pointer, self.MAX_GAME_MOVES)
        if status != "ongoing":
            self.game_over, self.game_result = True, (status, winner)
        self.update_ui_after_state_change()
        if self.game_over:
            self._log_game_over(); self._stop_ai_process()
            if self.game_mode.get() == GameMode.AI_VS_AI.value and self.ai_series_running:
                self.process_ai_series_result()

    def set_interactivity(self, is_interactive):
        if is_interactive:
            self.canvas.bind("<Button-1>", self.on_drag_start); self.canvas.bind("<B1-Motion>", self.on_drag_motion); self.canvas.bind("<ButtonRelease-1>", self.on_drag_end)
        else:
            self.canvas.unbind("<Button-1>"); self.canvas.unbind("<B1-Motion>"); self.canvas.unbind("<ButtonRelease-1>")

    def is_ai_thinking(self):
        return self.ai_process is not None and self.ai_process.is_alive()

    def update_bot_labels(self):
        is_thinking = self.is_ai_thinking()
        current_bot_name = self.ai_process.name if self.ai_process else ""
        is_thinking_for_move = is_thinking and (current_bot_name != self.ANALYSIS_AI_NAME)
        thinking_text = " (Thinking...)" if is_thinking_for_move else ""
        mode = self.game_mode.get()
        if mode == GameMode.HUMAN_VS_BOT.value:
            if self.human_color == "white": white_label, black_label = "Human", f"{self.MAIN_AI_NAME}{thinking_text}"
            else: white_label, black_label = f"{self.MAIN_AI_NAME}{thinking_text}", "Human"
        elif mode == GameMode.AI_VS_AI.value:
            if self.white_playing_bot_type == 'main': white_name, black_name = self.MAIN_AI_NAME, self.OPPONENT_AI_NAME
            else: white_name, black_name = self.OPPONENT_AI_NAME, self.MAIN_AI_NAME
            white_label = f"{white_name}{thinking_text if self.turn == 'white' else ''}"
            black_label = f"{black_name}{thinking_text if self.turn == 'black' else ''}"
        else: white_label, black_label = "Human (White)", "Human (Black)"
        if self.board_orientation == "white":
            self.bottom_bot_label.config(text=white_label); self.top_bot_label.config(text=black_label)
        else:
            self.bottom_bot_label.config(text=black_label); self.top_bot_label.config(text=white_label)

    def on_drag_start(self, event):
        is_analysis = self.is_ai_thinking() and self.ai_process.name == self.ANALYSIS_AI_NAME
        if self.game_over or (self.is_ai_thinking() and not is_analysis): return
        r, c = self.canvas_to_board(event.x, event.y)
        if r != -1:
            piece = self.board.grid[r][c]
            is_human_turn = (self.game_mode.get() != GameMode.HUMAN_VS_BOT.value or self.turn == self.human_color)
            if piece and piece.color == self.turn and is_human_turn:
                self.last_move_timestamp = time.time()
                self.selected, self.dragging, self.drag_start = (r, c), True, (r, c)
                self.valid_moves = get_all_legal_moves(self.board, self.turn)
                self.valid_moves_for_highlight = [end for start, end in self.valid_moves if start == self.selected]
                font_size = int(self.square_size * 0.7)
                self.drag_piece_ghost = self.canvas.create_text(event.x, event.y, text=piece.symbol(), font=("Arial Unicode MS", font_size), fill=piece.color, tags="drag_ghost")
                self.draw_board(); self.canvas.tag_raise("drag_ghost")
    
    def on_drag_motion(self, event):
        if self.dragging: self.canvas.coords(self.drag_piece_ghost, event.x, event.y)

    def swap_sides(self):
        self._stop_ai_process()
        if self.game_mode.get() == GameMode.HUMAN_VS_BOT.value:
            self.human_color = "black" if self.human_color == "white" else "white"
        self.reset_game()

    def switch_turn(self):
        if not self.game_over: self.turn = "black" if self.turn == "white" else "white"

    def board_to_canvas(self, r, c):
        if self.square_size == 0: return 0, 0
        x = (COLS - 1 - c) * self.square_size if self.board_orientation == "black" else c * self.square_size
        y = (ROWS - 1 - r) * self.square_size if self.board_orientation == "black" else r * self.square_size
        return x, y

    def canvas_to_board(self, x, y):
        if self.square_size == 0: return -1, -1
        c = (COLS - 1) - (x // self.square_size) if self.board_orientation == "black" else x // self.square_size
        r = (ROWS - 1) - (y // self.square_size) if self.board_orientation == "black" else y // self.square_size
        return (r, c) if 0 <= r < ROWS and 0 <= c < COLS else (-1, -1)

    # --- FREE OPTIMIZATION: CACHED EVAL BAR RENDERING ---
    def draw_eval_bar(self, eval_score, depth=None):
        score = eval_score / 100.0
        w, h = self.eval_bar_canvas.winfo_width(), self.eval_bar_canvas.winfo_height()
        if w <= 1 or h <= 1: return
        
        # Only redraw the expensive gradient if the canvas size changed
        if w != self.last_eval_bar_w or h != self.last_eval_bar_h:
            self.eval_bar_canvas.delete("gradient")
            for x_pixel in range(w):
                intensity = int(255 * (x_pixel / float(w - 1)))
                color = f"#{intensity:02x}{intensity:02x}{intensity:02x}"
                self.eval_bar_canvas.create_line(x_pixel, 0, x_pixel, h, fill=color, tags="gradient")
            self.last_eval_bar_w = w
            self.last_eval_bar_h = h
            
        # Always update the marker (cheap)
        self.eval_bar_canvas.delete("marker")
        
        # --- CHANGE IS HERE ---
        # Changed divisor from 20 to 10 for better sensitivity
        marker_score = max(-1.0, min(1.0, math.tanh(score / 10.0)))
        # ----------------------
        
        marker_x = int(((marker_score + 1) / 2.0) * w)
        self.eval_bar_canvas.create_line(marker_x, 0, marker_x, h, fill=self.COLORS['accent'], width=3, tags="marker")
        self.eval_bar_canvas.create_line(w // 2, 0, w // 2, h, fill="#666666", width=1, tags="marker") # Center line
        
        if depth: eval_text = f"{'+' if score > 0 else ''}{score:.2f} (D{depth})"
        elif abs(score) < 0.05: eval_text = "Even"
        else: eval_text = f"{'+' if score > 0 else ''}{score:.2f}"
        self.eval_score_label.config(text=eval_text)

    def setup_styles(self):
        style = ttk.Style(); style.theme_use('clam')
        C = {'bg_dark': '#1a1a2e', 'bg_medium': '#16213e', 'bg_light': '#0f3460', 'accent': '#e94560', 'text_light': '#ffffff', 'text_dark': '#a2a2a2'}
        style.configure('.', background=C['bg_dark'], foreground=C['text_light'])
        style.configure('TFrame', background=C['bg_dark'])
        style.configure('Left.TFrame', background=C['bg_dark']); style.configure('Right.TFrame', background=C['bg_medium']); style.configure('Canvas.TFrame', background=C['bg_medium'])
        style.configure('Header.TLabel', background=C['bg_dark'], foreground=C['text_light'], font=('Helvetica', 14, 'bold'), padding=(0, 10))
        style.configure('SmallHeader.TLabel', background=C['bg_dark'], foreground=C['text_light'], font=('Helvetica', 13, 'bold'), padding=(0, 1))
        style.configure('Status.TLabel', background=C['bg_light'], foreground=C['text_light'], font=('Helvetica', 14, 'bold'), padding=(11, 4), relief='flat')
        style.configure('Control.TButton', background=C['accent'], foreground=C['text_light'], font=('Helvetica', 11, 'bold'), padding=(10, 8), borderwidth=0, relief='flat')
        style.map('Control.TButton', background=[('active', C['accent']), ('pressed', '#d13550')])
        style.configure('Nav.TButton', background=C['bg_light'], foreground=C['text_light'], font=('Helvetica', 16, 'bold'), padding=(10, 5), borderwidth=0, relief='flat')
        style.map('Nav.TButton', background=[('active', C['bg_light']), ('pressed', C['bg_medium'])], foreground=[('disabled', C['text_dark'])])
        style.configure('Custom.TRadiobutton', background=C['bg_dark'], foreground=C['text_light'], font=('Helvetica', 11), padding=(5, 8))
        style.map('Custom.TRadiobutton', background=[('active', C['bg_dark'])], indicatorcolor=[('selected', C['accent'])], foreground=[('active', C['accent'])])
        style.configure('Custom.TCheckbutton', background=C['bg_dark'], foreground=C['text_light'], font=('Helvetica', 11), padding=(5, 8))
        style.map('Custom.TCheckbutton', background=[('active', C['bg_dark'])], indicatorcolor=[('selected', C['accent'])], foreground=[('active', C['accent'])])
        return C

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
        self.canvas.delete("highlight", "piece", "check_highlight")
        moves_to_highlight = getattr(self, 'valid_moves_for_highlight', [])
        for r_move, c_move in moves_to_highlight:
            x1, y1 = self.board_to_canvas(r_move, c_move)
            radius = self.square_size // 5
            center_x, center_y = x1 + self.square_size // 2, y1 + self.square_size // 2
            self.canvas.create_oval(center_x - radius, center_y - radius, center_x + radius, center_y + radius, fill="#1E90FF", outline="", tags="highlight")
        for r in range(ROWS):
            for c in range(COLS):
                if self.board.grid[r][c]: self.draw_piece_with_check(r, c)

    def draw_piece_with_check(self, r, c):
        piece = self.board.grid[r][c]
        if isinstance(piece, King) and is_in_check(self.board, piece.color):
            color = "darkred" if self.game_over and self.game_result and self.game_result[0] == "checkmate" else "red"
            x1, y1 = self.board_to_canvas(r, c)
            self.canvas.create_rectangle(x1, y1, x1 + self.square_size, y1 + self.square_size, outline=color, width=4, tags="check_highlight")
        if (r, c) != self.drag_start: self.draw_piece_at_canvas_coords(piece, r, c)

    def _start_game_if_needed(self):
        if self.game_started: return
        self.game_started = True
        mode = self.game_mode.get()
        if mode == GameMode.HUMAN_VS_BOT.value: mode_name = f"Human ({self.human_color.capitalize()}) vs. AI"
        elif mode == GameMode.HUMAN_VS_HUMAN.value: mode_name = "Human vs. Human"
        elif mode == GameMode.AI_VS_AI.value:
            if self.white_playing_bot_type == 'main': mode_name = f"{self.MAIN_AI_NAME} (W) vs. {self.OPPONENT_AI_NAME} (B)"
            else: mode_name = f"{self.OPPONENT_AI_NAME} (W) vs. {self.MAIN_AI_NAME} (B)"
        else: mode_name = mode.replace("_", " ").title()
        print("\n" + "="*60 + f"\nNEW GAME: {mode_name}\n" + "="*60)
        self._update_analysis_after_state_change()
    
    def _log_game_over(self):
        if not self.game_result: return
        result_type, winner = self.game_result
        if result_type == "checkmate": message = f"Checkmate! {winner.capitalize()} wins!"
        elif result_type == "stalemate": message = "Stalemate! It's a draw."
        elif result_type == "repetition": message = "Draw by Repetition!"
        elif result_type == "move_limit": message = f"Draw by {self.MAX_GAME_MOVES} move limit!"
        elif result_type == "insufficient_material": message = "Draw by Insufficient Material!"
        else: message = "Game Over"
        print("\n" + "-"*25 + " GAME OVER " + "-"*24 + f"\nResult: {message}\n" + "-" * 60)

    def update_turn_label(self):
        message = f"TURN: {self.turn.upper()}"
        if self.game_result:
            result_type, winner = self.game_result
            if result_type == "checkmate": message = f"CHECKMATE! {winner.upper()} WINS!"
            elif result_type == "stalemate": message = "STALEMATE! IT'S A DRAW."
            else: message = "GAME OVER: DRAW"
        self.turn_label.config(text=message)
        
    def process_ai_series_result(self):
        self.ai_series_stats['game_count'] += 1
        _, winner_color = self.game_result
        if winner_color:
            main_ai_color = 'white' if self.white_playing_bot_type == 'main' else 'black'
            if winner_color == main_ai_color: self.ai_series_stats['my_ai_wins'] += 1
            else: self.ai_series_stats['op_ai_wins'] += 1
        else: self.ai_series_stats['draws'] += 1
        self.update_scoreboard()
        if self.ai_series_running and self.ai_series_stats['game_count'] < 100:
            self.master.after(1000, self.reset_game)
        else:
            self.ai_series_running = False
            self.turn_label.config(text="AI SERIES COMPLETE!")

    def start_ai_series(self):
        self._stop_ai_process()
        self.game_mode.set(GameMode.AI_VS_AI.value)
        self.ai_series_stats = {'game_count': 0, 'my_ai_wins': 0, 'op_ai_wins': 0, 'draws': 0}
        self.ai_series_running = True
        self.current_opening_move = None
        self.reset_game()
        
    def apply_series_opening_move(self):
        if self.ai_series_stats['game_count'] % 2 == 0:
            print("\n--- Generating new opening for game pair ---")
            moves = get_all_legal_moves(self.board, "white")
            self.current_opening_move = random.choice(moves) if moves else None
        
        if self.current_opening_move:
            print(f"Applying opening move: {format_move(self.current_opening_move)}")
            self.board.make_move(self.current_opening_move[0], self.current_opening_move[1])
            self.execute_move_and_check_state(player_who_moved=self.turn)

    def update_scoreboard(self):
        if self.game_mode.get() == GameMode.AI_VS_AI.value and self.ai_series_running:
            stats = self.ai_series_stats
            game_display_count = stats['game_count'] + 1
            score_text = (f"{self.MAIN_AI_NAME} vs {self.OPPONENT_AI_NAME} (Game {game_display_count}/100)\n"
                          f"  {self.MAIN_AI_NAME} Wins: {stats['my_ai_wins']}\n"
                          f"  {self.OPPONENT_AI_NAME} Wins: {stats['op_ai_wins']}\n"
                          f"  Draws: {stats['draws']}")
            self.scoreboard_label.config(text=score_text)
            self.scoreboard_frame.place(relx=1.0, rely=0.0, anchor='ne', x=-15, y=15)
        else:
            self.scoreboard_frame.place_forget()

    def undo_move(self):
        if self.game_mode.get() == GameMode.AI_VS_AI.value or self.history_pointer <= 0: return
        self.history_pointer -= 1
        self._load_state_from_history()

    def redo_move(self):
        if self.game_mode.get() == GameMode.AI_VS_AI.value or self.history_pointer >= len(self.full_history) - 1: return
        self.history_pointer += 1
        self._load_state_from_history()

    def go_to_start(self):
        if self.game_mode.get() == GameMode.AI_VS_AI.value or self.history_pointer <= 0: return
        self.history_pointer = 0
        self._load_state_from_history()
    
    def go_to_end(self):
        if self.game_mode.get() == GameMode.AI_VS_AI.value or self.history_pointer >= len(self.full_history) - 1: return
        self.history_pointer = len(self.full_history) - 1
        self._load_state_from_history()
        
    def _load_state_from_history(self):
        self._stop_ai_process()
        board_state, turn_state = self.full_history[self.history_pointer]
        self.board = board_state.clone()
        self.turn = turn_state
        self.game_over, self.game_result = False, None
        
        self.position_counts.clear()
        for i in range(self.history_pointer + 1):
            board, turn = self.full_history[i]
            h = board_hash(board, turn)
            self.position_counts[h] = self.position_counts.get(h, 0) + 1

        status, winner = get_game_state(self.board, self.turn, self.position_counts, self.history_pointer, self.MAX_GAME_MOVES)
        if status != "ongoing":
            self.game_over, self.game_result = True, (status, winner)
            
        self.update_ui_after_state_change()
        self._update_analysis_after_state_change()

    def update_navigation_buttons(self):
        if self.game_mode.get() == GameMode.AI_VS_AI.value:
            state = tk.DISABLED
            self.start_button.config(state=state); self.undo_button.config(state=state)
            self.redo_button.config(state=state); self.end_button.config(state=state)
            return

        can_go_back = self.history_pointer > 0
        can_go_forward = self.history_pointer < len(self.full_history) - 1
        
        self.start_button.config(state=tk.NORMAL if can_go_back else tk.DISABLED)
        self.undo_button.config(state=tk.NORMAL if can_go_back else tk.DISABLED)
        self.redo_button.config(state=tk.NORMAL if can_go_forward else tk.DISABLED)
        self.end_button.config(state=tk.NORMAL if can_go_forward else tk.DISABLED)

    def update_ui_after_state_change(self):
        self.selected, self.valid_moves, self.valid_moves_for_highlight = None, [], []
        self.update_turn_label()
        self.draw_board()
        self.update_navigation_buttons()

if __name__ == "__main__":
    mp.freeze_support()
    root = tk.Tk()
    app = EnhancedChessApp(root)
    root.mainloop()