# JungleChessUI.py (v3.8 - Analysis Mode)

import tkinter as tk
from tkinter import ttk
import math
import random
import time
from GameLogic import *
from AI import ChessBot, board_hash
from OpponentAI import OpponentAI
from enum import Enum
import threading
import queue

# --- UI Constants ---
SQUARE_SIZE = 75

class GameMode(Enum):
    HUMAN_VS_BOT = "bot"
    HUMAN_VS_HUMAN = "human"
    AI_VS_AI = "ai_vs_ai"

class EnhancedChessApp:
    MAIN_AI_NAME = "AI Bot"
    OPPONENT_AI_NAME = "OP Bot"
    
    def __init__(self, master):
        self.master = master
        self.master.title("Jungle Chess")
        
        self.log_queue = queue.Queue()
        self.ai_move_queue = queue.Queue()
        self.ai_is_thinking = False
        self.cancellation_event = threading.Event()
        self.ai_search_start_time = None
        
        ### --- ANALYSIS MODE --- ###
        # Dedicated thread and event for analysis to avoid conflict with game AI
        self.analysis_mode_var = tk.BooleanVar(value=False)
        self.analysis_cancellation_event = threading.Event()
        self.analysis_bot = None
        self.analysis_thread = None
        ### --- END ANALYSIS MODE --- ###

        self.board = create_initial_board()
        self.turn = "white"
        self.selected, self.valid_moves, self.game_over = None, [], False
        self.game_result, self.dragging, self.drag_piece_ghost, self.drag_start = None, False, None, None
        self.position_history, self.position_counts = [], {}

        self.game_mode = tk.StringVar(value=GameMode.HUMAN_VS_BOT.value)
        self.ai_series_running = False
        self.ai_series_stats = {'game_count': 0, 'my_ai_wins': 0, 'op_ai_wins': 0, 'draws': 0}
        self.ai_white_bot, self.ai_black_bot = None, None
        self.white_playing_bot = "main"
        self.current_opening_move = None
        self.human_color = "white"
        self.board_orientation = "white"
        self.bot = None
        self.last_move_timestamp = None
        self.game_started = False 

        self.COLORS = self.setup_styles()
        self.master.configure(bg=self.COLORS['bg_dark'])
        self.build_ui()
        
        self.process_log_queue()
        self.process_ai_queue()
        self.reset_game()

    def build_ui(self):
        screen_w, screen_h = self.master.winfo_screenwidth(), self.master.winfo_screenheight()
        self.master.geometry(f"{screen_w}x{screen_h}+0+0"); self.master.state('zoomed')
        self.main_frame = ttk.Frame(self.master, style='Left.TFrame')
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.left_panel = ttk.Frame(self.main_frame, width=250, style='Left.TFrame')
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=15, pady=(0,15)); self.left_panel.pack_propagate(False)
        ttk.Label(self.left_panel, text="JUNGLE CHESS", style='Header.TLabel', font=('Helvetica', 24, 'bold')).pack(pady=(0,10))
        self._build_control_widgets()
        self.right_panel = ttk.Frame(self.main_frame, style='Right.TFrame')
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=15, pady=15)
        self.canvas_container = ttk.Frame(self.right_panel, style='Canvas.TFrame')
        self.canvas_container.pack(expand=True, fill=tk.BOTH)
        self.canvas_container.grid_rowconfigure(0, weight=1); self.canvas_container.grid_columnconfigure(0, weight=1)
        self.canvas_frame = ttk.Frame(self.canvas_container, style='Canvas.TFrame'); self.canvas_frame.grid(row=0, column=0)
        self.canvas = tk.Canvas(self.canvas_frame, width=COLS*SQUARE_SIZE, height=ROWS*SQUARE_SIZE, bg=self.COLORS['bg_light'], highlightthickness=0)
        self.board_image_white = self.create_board_image("white"); self.board_image_black = self.create_board_image("black")
        self.board_image_id = self.canvas.create_image(0, 0, image=self.board_image_white, anchor='nw', tags="board"); self.canvas.pack()
        self._build_scoreboard_and_labels()
        self.draw_board()

    def _build_control_widgets(self):
        game_mode_frame = ttk.Frame(self.left_panel, style='Left.TFrame')
        game_mode_frame.pack(fill=tk.X, pady=(0,9))
        ttk.Label(game_mode_frame, text="GAME MODE", style='Header.TLabel').pack(anchor=tk.W)
        for mode in GameMode:
            text = mode.name.replace("_", " ").title()
            ttk.Radiobutton(game_mode_frame, text=text, variable=self.game_mode, value=mode.value, command=self.reset_game, style='Custom.TRadiobutton').pack(anchor=tk.W, pady=(5,0))
        controls_frame = ttk.Frame(self.left_panel, style='Left.TFrame')
        controls_frame.pack(fill=tk.X, pady=4)
        ttk.Button(controls_frame, text="NEW GAME", command=self.reset_game, style='Control.TButton').pack(fill=tk.X, pady=3)
        ttk.Button(controls_frame, text="SWAP SIDES", command=self.swap_sides, style='Control.TButton').pack(fill=tk.X, pady=3)
        ttk.Button(controls_frame, text="AI vs OP Series", command=self.start_ai_series, style='Control.TButton').pack(fill=tk.X, pady=3)
        ttk.Button(controls_frame, text="QUIT", command=self.master.quit, style='Control.TButton').pack(fill=tk.X, pady=3)
        ttk.Label(controls_frame, text="Bot Depth:", style='Header.TLabel').pack(anchor=tk.W, pady=(9,0))
        self.bot_depth_slider = tk.Scale(controls_frame, from_=1, to=6, orient=tk.HORIZONTAL, command=self.update_bot_depth, bg=self.COLORS['bg_dark'], fg=self.COLORS['text_light'], highlightthickness=0, relief='flat')
        self.bot_depth_slider.set(getattr(ChessBot, 'search_depth', 3)); self.bot_depth_slider.pack(fill=tk.X, pady=(0,3))
        self.instant_move = tk.BooleanVar(value=False)
        ttk.Checkbutton(controls_frame, text="Instant Moves", variable=self.instant_move, style='Custom.TCheckbutton').pack(anchor=tk.W, pady=(3,3))
        ### --- ANALYSIS MODE --- ###
        ttk.Checkbutton(controls_frame, text="Analysis Mode (H-vs-H)", variable=self.analysis_mode_var, style='Custom.TCheckbutton', command=self.toggle_analysis_mode).pack(anchor=tk.W, pady=(3,3))
        ### --- END ANALYSIS MODE --- ###
        opening_frame = ttk.Frame(self.left_panel, style='Left.TFrame')
        opening_frame.pack(fill=tk.X, pady=(15, 0))
        self.force_opening_var = tk.BooleanVar(value=False)
        self.turn_frame = ttk.Frame(self.left_panel, style='Left.TFrame'); self.turn_frame.pack(fill=tk.X, pady=(9,0))
        self.turn_label = ttk.Label(self.turn_frame, text="WHITE'S TURN", style='Status.TLabel'); self.turn_label.pack(fill=tk.X)
        self.eval_frame = ttk.Frame(self.left_panel, style='Left.TFrame'); self.eval_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=0, pady=(5,5))
        self.eval_score_label = ttk.Label(self.eval_frame, text="Even", style='Status.TLabel', anchor="center"); self.eval_score_label.pack(pady=(7,5))
        self.eval_bar_canvas = tk.Canvas(self.eval_frame, height=26, bg=self.COLORS['bg_light'], highlightthickness=0); self.eval_bar_canvas.pack(fill=tk.X, expand=True)
        self.eval_bar_canvas.bind("<Configure>", lambda event: self.draw_eval_bar(0))

    def _build_scoreboard_and_labels(self):
        self.scoreboard_frame = ttk.Frame(self.right_panel, style='Right.TFrame')
        self.scoreboard_label = ttk.Label(self.scoreboard_frame, text="", font=("Helvetica", 10), justify=tk.LEFT, background=self.COLORS['bg_medium'], foreground=self.COLORS['text_light']); self.scoreboard_label.pack()
        self.top_bot_label = ttk.Label(self.right_panel, text="", font=("Helvetica", 12), background=self.COLORS['bg_medium'], foreground=self.COLORS['text_light']); self.top_bot_label.place(relx=0.5, rely=0.02, anchor='n')
        self.bottom_bot_label = ttk.Label(self.right_panel, text="", font=("Helvetica", 12), background=self.COLORS['bg_medium'], foreground=self.COLORS['text_light']); self.bottom_bot_label.place(relx=0.5, rely=0.98, anchor='s')

    ### --- ANALYSIS MODE --- ###
    def toggle_analysis_mode(self):
        """Called when the analysis checkbox is clicked."""
        if self.analysis_mode_var.get():
            self.start_analysis_if_needed()
        else:
            self.stop_analysis()

    def stop_analysis(self):
        """Stops any running analysis thread and resets the eval bar."""
        if self.analysis_thread and self.analysis_thread.is_alive():
            # print("UI: Sending cancellation signal to analysis thread...")
            self.analysis_cancellation_event.set()
        self.analysis_bot = None
        self.analysis_thread = None
        # Reset the evaluation display to neutral
        self.draw_eval_bar(0)
        self.eval_score_label.config(text="Even")

    def start_analysis_if_needed(self):
        """Starts the background analysis if conditions are met."""
        self.stop_analysis() # Always stop any previous analysis first
        
        # Conditions to start analysis: checkbox is on, mode is Human vs Human, and game is not over.
        if not self.analysis_mode_var.get() or self.game_mode.get() != GameMode.HUMAN_VS_HUMAN.value or self.game_over:
            return

        print("UI: Starting background analysis...")
        self.analysis_cancellation_event.clear()
        
        # Create a new bot instance for analysis. It needs its own board clone to not interfere.
        # It also gets its own cancellation event.
        self.analysis_bot = ChessBot(self.board.clone(), self.turn, self, self.analysis_cancellation_event, "Analysis Bot")
        self.analysis_bot.search_depth = 99 # Effectively unlimited depth for pondering
        
        # Run the pondering in a separate, daemonic thread
        self.analysis_thread = threading.Thread(target=self.analysis_bot.ponder_indefinitely, daemon=True)
        self.analysis_thread.start()
    ### --- END ANALYSIS MODE --- ###

    def _start_game_if_needed(self):
        if not self.game_started:
            self.game_started = True
            print("\n" + "="*60)
            mode = self.game_mode.get()
            if mode == GameMode.HUMAN_VS_BOT.value:
                bot_color = "black" if self.human_color == "white" else "white"
                print(f"NEW GAME: Human vs. Bot (AI is {bot_color.capitalize()})")
            elif mode == GameMode.AI_VS_AI.value:
                print("NEW GAME: AI vs. AI")
            else: # Human vs Human
                print("NEW GAME: Human vs. Human")
            print("="*60)
            self.last_move_timestamp = time.time()

    def process_log_queue(self):
        try:
            while not self.log_queue.empty():
                message = self.log_queue.get_nowait()
                print(message)
        except queue.Empty:
            pass
        finally:
            self.master.after(100, self.process_log_queue)

    def process_ai_queue(self):
        try:
            move_success = self.ai_move_queue.get_nowait()
            if self.ai_search_start_time:
                self.ai_search_start_time = None
            self.ai_is_thinking = False
            self.update_bot_labels()
            if move_success:
                self.execute_move_and_check_state()
                if not self.game_over and self.game_mode.get() == GameMode.AI_VS_AI.value:
                    delay = 20 if self.instant_move.get() else 500
                    self.master.after(delay, self.make_ai_move)
            else:
                print("AI reported no valid move was made (or was cancelled).")
                self.set_interactivity()
        except queue.Empty: pass
        finally: self.master.after(100, self.process_ai_queue)

    def _ai_worker(self, bot_instance):
        success = bot_instance.make_move()
        if not self.cancellation_event.is_set(): self.ai_move_queue.put(success)

    def _start_ai_thread(self, bot_instance):
        if self.ai_is_thinking or self.game_over: return
        self.cancellation_event.clear(); self.ai_is_thinking = True; self.ai_search_start_time = time.time()
        self.update_bot_labels(); self.set_interactivity()
        thread = threading.Thread(target=self._ai_worker, args=(bot_instance,), daemon=True); thread.start()
        
    def _interrupt_ai_search(self):
        if self.ai_is_thinking:
            print("UI: Sending cancellation signal to AI..."); self.cancellation_event.set()
            self.ai_is_thinking = False; self.update_bot_labels()

    def reset_game(self):
        self._interrupt_ai_search()
        ### --- ANALYSIS MODE --- ###
        self.stop_analysis() # Stop any analysis when game is reset
        ### --- END ANALYSIS MODE --- ###
        
        self.board = create_initial_board()
        self.turn = "white"
        self.game_started = False
        self.last_move_timestamp = None
        
        self.selected, self.valid_moves, self.game_over, self.game_result = None, [], False, None
        self.dragging, self.drag_piece_ghost, self.drag_start = False, None, None
        self.position_history, self.position_counts = [], {}
        initial_key = board_hash(self.board, self.turn)
        self.position_history.append(initial_key); self.position_counts[initial_key] = 1
        current_mode = self.game_mode.get(); delay = 20 if self.instant_move.get() else 500
        
        if current_mode == GameMode.AI_VS_AI.value:
            self.white_playing_bot = "op" if self.ai_series_running and self.ai_series_stats['game_count'] % 2 == 1 else "main"
            self.board_orientation = "white" if self.white_playing_bot == "main" else "black"
            
            if self.white_playing_bot == "main":
                white_class, white_name = ChessBot, self.MAIN_AI_NAME
                black_class, black_name = OpponentAI, self.OPPONENT_AI_NAME
            else:
                white_class, white_name = OpponentAI, self.OPPONENT_AI_NAME
                black_class, black_name = ChessBot, self.MAIN_AI_NAME
            
            self.ai_white_bot = white_class(self.board, "white", self, self.cancellation_event, white_name)
            self.ai_black_bot = black_class(self.board, "black", self, self.cancellation_event, black_name)
            
            self.update_bot_depth(self.bot_depth_slider.get())
            if self.ai_series_running and not self.force_opening_var.get(): self.apply_series_opening_move()
            if not self.game_over: self.master.after(delay, self.make_ai_move)

        elif current_mode == GameMode.HUMAN_VS_BOT.value:
            bot_color = "black" if self.human_color == "white" else "white"
            self.bot = ChessBot(self.board, bot_color, self, self.cancellation_event, "AI Bot")
            self.update_bot_depth(self.bot_depth_slider.get())
            if self.turn != self.human_color and not self.game_over: self.master.after(delay, self.make_bot_move)
        
        self.update_turn_label(); self.draw_board(); self.set_interactivity(); self.update_bot_labels(); self.update_scoreboard()
        
        ### --- ANALYSIS MODE --- ###
        # Start analysis for the new game if conditions are right
        self.start_analysis_if_needed()
        ### --- END ANALYSIS MODE --- ###

    def on_drag_end(self, event):
        if not self.dragging: self.valid_moves = []; self.draw_board(); return
        self.dragging = False; self.canvas.delete("drag_ghost")
        row, col = self.canvas_to_board(event.x, event.y)
        if row == -1 or not self.drag_start:
            self.drag_start, self.selected, self.valid_moves = None, None, []; self.draw_board(); self.set_interactivity(); return
        start_pos, end_pos = self.drag_start, (row, col)
        if end_pos in self.valid_moves:
            self._start_game_if_needed()
            self.board.make_move(start_pos, end_pos)
            self.execute_move_and_check_state()
            if not self.game_over and self.game_mode.get() == GameMode.HUMAN_VS_BOT.value and self.turn != self.human_color:
                delay = 20 if self.instant_move.get() else 500
                self.master.after(delay, self.make_bot_move)
        self.drag_start, self.selected, self.valid_moves = None, None, []
        self.draw_board(); self.set_interactivity()

    def make_bot_move(self):
        self._start_game_if_needed()
        if not self.game_over and self.bot: self._start_ai_thread(self.bot)

    def make_ai_move(self):
        self._start_game_if_needed()
        if self.game_mode.get() == GameMode.AI_VS_AI.value and not self.game_over:
            current_bot = self.ai_white_bot if self.turn == "white" else self.ai_black_bot
            if current_bot: self._start_ai_thread(current_bot)

    def set_interactivity(self):
        is_interactive = self.game_mode.get() != GameMode.AI_VS_AI.value and not self.ai_is_thinking
        if is_interactive:
            self.canvas.bind("<Button-1>", self.on_drag_start); self.canvas.bind("<B1-Motion>", self.on_drag_motion); self.canvas.bind("<ButtonRelease-1>", self.on_drag_end)
        else:
            self.canvas.unbind("<Button-1>"); self.canvas.unbind("<B1-Motion>"); self.canvas.unbind("<ButtonRelease-1>")

    def update_bot_depth(self, value):
        new_depth = int(value)
        ChessBot.search_depth = new_depth; OpponentAI.search_depth = new_depth
        for bot in [self.bot, self.ai_white_bot, self.ai_black_bot]:
            if bot: bot.search_depth = new_depth

    def switch_turn(self): self.turn = "black" if self.turn == "white" else "white"

    def swap_sides(self):
        self._interrupt_ai_search()
        if self.game_mode.get() == GameMode.HUMAN_VS_BOT.value:
            self.human_color = "black" if self.human_color == "white" else "white"
        self.board_orientation = "black" if self.board_orientation == "white" else "white"
        self.reset_game()

    def board_to_canvas(self, r, c):
        x = (COLS - 1 - c) * SQUARE_SIZE if self.board_orientation == "black" else c * SQUARE_SIZE
        y = (ROWS - 1 - r) * SQUARE_SIZE if self.board_orientation == "black" else r * SQUARE_SIZE
        return x, y

    def canvas_to_board(self, x, y):
        c = (COLS - 1) - (x // SQUARE_SIZE) if self.board_orientation == "black" else (x // SQUARE_SIZE)
        r = (ROWS - 1) - (y // SQUARE_SIZE) if self.board_orientation == "black" else (y // SQUARE_SIZE)
        return (r, c) if 0 <= r < ROWS and 0 <= c < COLS else (-1, -1)

    ### --- ANALYSIS MODE --- ###
    def draw_eval_bar(self, eval_score, depth=None):
        """Draws the evaluation bar, optionally with the search depth."""
        score = eval_score / 100.0
        w, h = self.eval_bar_canvas.winfo_width(), self.eval_bar_canvas.winfo_height()
        self.eval_bar_canvas.delete("all")
        if w <= 1 or h <= 1: 
            self.eval_score_label.config(text="Eval: ...")
            return
            
        marker_score = max(-1.0, min(1.0, math.tanh(score / 20.0)))
        marker_x = int(((marker_score + 1) / 2.0) * w)
        
        for x_pixel in range(w):
            intensity = int(255 * (x_pixel / float(w - 1 if w > 1 else 1)))
            self.eval_bar_canvas.create_line(x_pixel, 0, x_pixel, h, fill=f"#{intensity:02x}{intensity:02x}{intensity:02x}")
        
        self.eval_bar_canvas.create_line(marker_x, 0, marker_x, h, fill=self.COLORS['accent'], width=3)
        self.eval_bar_canvas.create_line(w//2, 0, w//2, h, fill="#666666", width=1)
        
        # Update label text
        if depth:
            eval_text = f"{'+' if score > 0 else ''}{score:.2f} (D{depth})"
        else:
            eval_text = "Even" if abs(score) < 0.05 else f"{'+' if score > 0 else ''}{score:.2f}"
        
        self.eval_score_label.config(text=eval_text)
    ### --- END ANALYSIS MODE --- ###

    def setup_styles(self):
        style = ttk.Style(); style.theme_use('clam')
        C = {'bg_dark':'#1a1a2e','bg_medium':'#16213e','bg_light':'#0f3460','accent':'#e94560','text_light':'#ffffff','text_dark':'#a2a2a2'}
        style.configure('.', background=C['bg_dark'], foreground=C['text_light'])
        style.configure('TFrame', background=C['bg_dark']); style.configure('Left.TFrame', background=C['bg_dark']); style.configure('Right.TFrame', background=C['bg_medium']); style.configure('Canvas.TFrame', background=C['bg_medium'])
        style.configure('Header.TLabel', background=C['bg_dark'], foreground=C['text_light'], font=('Helvetica',14,'bold'), padding=(0,10))
        style.configure('Status.TLabel', background=C['bg_light'], foreground=C['text_light'], font=('Helvetica',16,'bold'), padding=(11,4), relief='flat')
        style.configure('Control.TButton', background=C['accent'], foreground=C['text_light'], font=('Helvetica',11,'bold'), padding=(10,8), borderwidth=0, relief='flat')
        style.map('Control.TButton', background=[('active',C['accent']), ('pressed','#d13550')])
        style.configure('Custom.TRadiobutton', background=C['bg_dark'], foreground=C['text_light'], font=('Helvetica',11), padding=(5,8))
        style.map('Custom.TRadiobutton', background=[('active',C['bg_dark'])], indicatorcolor=[('selected',C['accent'])], foreground=[('active',C['accent'])])
        style.configure('Custom.TCheckbutton', background=C['bg_dark'], foreground=C['text_light'], font=('Helvetica',11), padding=(5,8))
        style.map('Custom.TCheckbutton', background=[('active',C['bg_dark'])], indicatorcolor=[('selected',C['accent'])], foreground=[('active',C['accent'])])
        return C

    def create_board_image(self, orientation):
        img = tk.PhotoImage(width=COLS*SQUARE_SIZE, height=ROWS*SQUARE_SIZE)
        for r_draw in range(ROWS):
            for c_draw in range(COLS):
                color = BOARD_COLOR_1 if (r_draw+c_draw)%2==0 else BOARD_COLOR_2
                x1, y1 = c_draw*SQUARE_SIZE, r_draw*SQUARE_SIZE
                img.put(color, to=(x1, y1, x1+SQUARE_SIZE, y1+SQUARE_SIZE))
        return img

    def draw_board(self):
        self.canvas.itemconfig(self.board_image_id, image=self.board_image_white if self.board_orientation == "white" else self.board_image_black)
        self.canvas.delete("highlight", "piece", "check_highlight")
        for r_move, c_move in self.valid_moves:
            x1, y1 = self.board_to_canvas(r_move, c_move)
            center_x, center_y = x1 + SQUARE_SIZE//2, y1 + SQUARE_SIZE//2
            self.canvas.create_oval(center_x-10, center_y-10, center_x+10, center_y+10, fill="#1E90FF", outline="", tags="highlight")
        for r in range(ROWS):
            for c in range(COLS):
                piece = self.board.grid[r][c]
                if piece:
                    if isinstance(piece, King) and is_in_check(self.board, piece.color):
                        is_mate = self.game_over and self.game_result and self.game_result[0] == "checkmate"
                        color = "darkred" if is_mate else "red"
                        x1, y1 = self.board_to_canvas(r, c); x2, y2 = x1+SQUARE_SIZE, y1+SQUARE_SIZE
                        self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=4, tags="check_highlight")
                    if (r, c) != self.drag_start: self.draw_piece_at_canvas_coords(piece, r, c)

    def draw_piece_at_canvas_coords(self, piece, r, c):
        x, y = self.board_to_canvas(r, c); x_center, y_center = x + SQUARE_SIZE//2, y + SQUARE_SIZE//2
        symbol, font, shadow_font = piece.symbol(), ("Arial Unicode MS", 48), ("Arial", 48)
        if piece.color == "white":
            self.canvas.create_text(x_center+2, y_center+2, text=symbol, font=shadow_font, fill="#444444", tags="piece")
            self.canvas.create_text(x_center, y_center, text=symbol, font=font, fill="white", tags="piece")
        else: self.canvas.create_text(x_center, y_center, text=symbol, font=font, fill="black", tags="piece")

    def on_drag_start(self, event):
        if self.game_over: return
        r, c = self.canvas_to_board(event.x, event.y)
        if r != -1:
            piece = self.board.grid[r][c]
            if piece and piece.color == self.turn and (self.game_mode.get() != GameMode.HUMAN_VS_BOT.value or self.turn == self.human_color):
                self.selected, self.dragging, self.drag_start = (r, c), True, (r, c)
                all_legal_moves = get_all_legal_moves(self.board, self.turn)
                self.valid_moves = [end for start, end in all_legal_moves if start == self.selected]
                self.canvas.delete("drag_ghost")
                font, color = ("Arial Unicode MS", 50), "white" if piece.color == "white" else "black"
                self.drag_piece_ghost = self.canvas.create_text(event.x, event.y, text=piece.symbol(), font=font, fill=color, tags="drag_ghost")
                self.draw_board()
                self.canvas.tag_raise("drag_ghost")

    def on_drag_motion(self, event):
        if self.dragging: self.canvas.coords(self.drag_piece_ghost, event.x, event.y)

    def execute_move_and_check_state(self):
        if self.last_move_timestamp:
            move_duration = time.time() - self.last_move_timestamp
            print(f"\n--- Turn {len(self.position_history)} ({self.turn.capitalize()}) | Time: {move_duration:.2f}s ---")
        else:
            print(f"\n--- Turn {len(self.position_history)} ({self.turn.capitalize()}) ---")

        if self.game_over: return

        player_who_moved = self.turn
        if not self.game_over: self.switch_turn()
        
        key = board_hash(self.board, self.turn)
        self.position_history.append(key)
        self.position_counts[key] = self.position_counts.get(key, 0) + 1
        
        status, winner = check_game_over(self.board, player_who_moved)
        if status: self.game_over, self.game_result = True, (status, winner)
        elif self.position_counts.get(key, 0) >= 3: self.game_over, self.game_result = True, ("repetition", None)
        
        self.update_turn_label()
        self.draw_board()
        self.set_interactivity()
        self.update_bot_labels()
        self.last_move_timestamp = time.time()
        
        ### --- ANALYSIS MODE --- ###
        if self.game_over:
            self.stop_analysis() # Stop analysis on game over
        else:
            # After a move, restart analysis for the new position
            self.start_analysis_if_needed()
        ### --- END ANALYSIS MODE --- ###

        if self.game_over and self.game_mode.get() == GameMode.AI_VS_AI.value: self.process_ai_series_result()
        
    def update_turn_label(self):
        if self.game_over and self.game_result:
            r_type, winner = self.game_result
            if r_type == "repetition": msg = "Draw by three-fold repetition!"
            elif r_type == "stalemate": msg = "Stalemate! It's a draw."
            elif r_type == "checkmate": 
                msg = f"Checkmate! {winner.capitalize()} wins!"
            else: msg = "Game Over"
            
            print("-" * 60)
            print(f"GAME OVER: {msg}")
            print("-" * 60)

            self.turn_label.config(text=msg)
        else: self.turn_label.config(text=f"Turn: {self.turn.capitalize()}")
        
    def process_ai_series_result(self):
        if not self.game_result: return
        self.ai_series_stats['game_count'] += 1
        _, winner_color = self.game_result
        if winner_color:
            winning_ai = self.white_playing_bot if winner_color == "white" else ("op" if self.white_playing_bot == "main" else "main")
            if winning_ai == "main": self.ai_series_stats['my_ai_wins'] += 1
            else: self.ai_series_stats['op_ai_wins'] += 1
        else: self.ai_series_stats['draws'] += 1
        self.update_scoreboard()
        if self.ai_series_running and self.ai_series_stats['game_count'] < 100: self.master.after(1000, self.reset_game)
        else: self.turn_label.config(text="AI Series Complete!"); self.ai_series_running = False

    def start_ai_series(self):
        self._interrupt_ai_search(); random.seed()
        self.game_mode.set(GameMode.AI_VS_AI.value)
        self.ai_series_stats = {'game_count': 0, 'my_ai_wins': 0, 'op_ai_wins': 0, 'draws': 0}
        self.ai_series_running = True; self.current_opening_move = None
        self.reset_game()

    def update_scoreboard(self):
        if self.game_mode.get() == GameMode.AI_VS_AI.value and self.ai_series_running:
            stats = self.ai_series_stats
            score_text = (f"{self.MAIN_AI_NAME} vs {self.OPPONENT_AI_NAME} (Game {stats['game_count']+1}/100)\n"
                          f"  {self.MAIN_AI_NAME} Wins: {stats['my_ai_wins']}\n"
                          f"  {self.OPPONENT_AI_NAME} Wins: {stats['op_ai_wins']}\n"
                          f"  Draws: {stats['draws']}")
            self.scoreboard_label.config(text=score_text)
            self.scoreboard_frame.place(relx=1.0, rely=0.0, anchor='ne', x=-15, y=15)
        else: self.scoreboard_frame.place_forget()

    def update_bot_labels(self):
        thinking = " (Thinking...)" if self.ai_is_thinking else ""
        mode = self.game_mode.get()
        
        if mode == GameMode.HUMAN_VS_BOT.value:
            white_l, black_l = ("Human", f"Bot{thinking}") if self.human_color == "white" else (f"Bot{thinking}", "Human")
        
        elif mode == GameMode.AI_VS_AI.value:
            white_name = self.ai_white_bot.bot_name
            black_name = self.ai_black_bot.bot_name
            white_l = f"{white_name}{thinking if self.turn == 'white' else ''}"
            black_l = f"{black_name}{thinking if self.turn == 'black' else ''}"
            
        else: # Human vs Human
            # The analysis bot does not change the "Human" label
            white_l, black_l = "Human (White)", "Human (Black)"

        if self.board_orientation == "white":
            self.bottom_bot_label.config(text=white_l)
            self.top_bot_label.config(text=black_l)
        else:
            self.bottom_bot_label.config(text=black_l)
            self.top_bot_label.config(text=white_l)

    def apply_series_opening_move(self):
        if self.ai_series_stats['game_count'] % 2 == 0:
            print("--- Generating new opening for game pair ---")
            moves = get_all_legal_moves(self.board, "white")
            self.current_opening_move = random.choice(moves) if moves else None
        if self.current_opening_move:
            start, end = self.current_opening_move
            print(f"Applying opening move: {start} -> {end}")
            self.board.make_move(start, end); self.switch_turn()

if __name__ == "__main__":
    root = tk.Tk(); app = EnhancedChessApp(root); root.mainloop()