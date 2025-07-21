import tkinter as tk
from tkinter import ttk
import math
import random
# MODIFIED: Import board_hash from AI, not generate_position_key from GameLogic
from GameLogic import *
from AI import ChessBot, board_hash 
from OpponentAI import OpponentAI
from enum import Enum
import threading
import queue

class GameMode(Enum):
    HUMAN_VS_BOT = "bot"
    HUMAN_VS_HUMAN = "human"
    AI_VS_AI = "ai_vs_ai"

class EnhancedChessApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Enhanced Chess")
        
        self.ai_move_queue = queue.Queue()
        self.ai_is_thinking = False
        self.cancellation_event = threading.Event()

        self.COLORS = self.setup_styles()
        self.master.configure(bg=self.COLORS['bg_dark'])
        self.position_history = []
        self.position_counts = {}
        self.game_result = None
        self.ai_series_running = False
        self.board_orientation = "white"
        self.game_mode = tk.StringVar(value=GameMode.HUMAN_VS_BOT.value)

        self.ai_series_stats = {'game_count': 0, 'my_ai_wins': 0, 'op_ai_wins': 0, 'draws': 0}
        self.ai_white_bot = None
        self.ai_black_bot = None
        self.white_playing_bot = "main"

        screen_w = self.master.winfo_screenwidth()
        screen_h = self.master.winfo_screenheight()
        self.master.geometry(f"{screen_w}x{screen_h}+0+0")
        self.master.state('zoomed')
        self.fullscreen = True

        self.main_frame = ttk.Frame(master, style='Left.TFrame')
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        self.left_panel = ttk.Frame(self.main_frame, width=250, style='Left.TFrame')
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=15, pady=(0,15))
        self.left_panel.pack_propagate(False)

        ttk.Label(self.left_panel, text="JUNGLE CHESS", style='Header.TLabel', font=('Helvetica', 24, 'bold')).pack(pady=(0,10))

        game_mode_frame = ttk.Frame(self.left_panel, style='Left.TFrame')
        game_mode_frame.pack(fill=tk.X, pady=(0,9))
        ttk.Label(game_mode_frame, text="GAME MODE", style='Header.TLabel').pack(anchor=tk.W)
        ttk.Radiobutton(game_mode_frame, text="Human vs Bot", variable=self.game_mode, value=GameMode.HUMAN_VS_BOT.value, command=self.reset_game, style='Custom.TRadiobutton').pack(anchor=tk.W, pady=(5,3))
        ttk.Radiobutton(game_mode_frame, text="Human vs Human", variable=self.game_mode, value=GameMode.HUMAN_VS_HUMAN.value, command=self.reset_game, style='Custom.TRadiobutton').pack(anchor=tk.W)
        ttk.Radiobutton(game_mode_frame, text="AI vs AI", variable=self.game_mode, value=GameMode.AI_VS_AI.value, command=self.reset_game, style='Custom.TRadiobutton').pack(anchor=tk.W)

        controls_frame = ttk.Frame(self.left_panel, style='Left.TFrame')
        controls_frame.pack(fill=tk.X, pady=4)
        ttk.Button(controls_frame, text="NEW GAME", command=self.reset_game, style='Control.TButton').pack(fill=tk.X, pady=3)
        ttk.Button(controls_frame, text="SWAP SIDES", command=self.swap_sides, style='Control.TButton').pack(fill=tk.X, pady=3)
        ttk.Button(controls_frame, text="AI vs OP start", command=self.start_ai_series, style='Control.TButton').pack(fill=tk.X, pady=3)
        ttk.Button(controls_frame, text="QUIT", command=self.master.quit, style='Control.TButton').pack(fill=tk.X, pady=3)

        ttk.Label(controls_frame, text="Bot Depth:", style='Header.TLabel').pack(anchor=tk.W, pady=(9,0))
        self.bot_depth_slider = tk.Scale(controls_frame, from_=1, to=6, orient=tk.HORIZONTAL, command=self.update_bot_depth, bg=self.COLORS['bg_dark'], fg=self.COLORS['text_light'], highlightthickness=0)
        initial_bot_depth = getattr(ChessBot, 'search_depth', 3)
        self.bot_depth_slider.set(initial_bot_depth)
        self.bot_depth_slider.pack(fill=tk.X, pady=(0,3))

        self.instant_move = tk.BooleanVar(value=False)
        ttk.Checkbutton(controls_frame, text="Instant Move", variable=self.instant_move, style='Custom.TCheckbutton').pack(anchor=tk.W, pady=(3,3))

        self.turn_frame = ttk.Frame(self.left_panel, style='Left.TFrame'); self.turn_frame.pack(fill=tk.X, pady=(9,0))
        self.turn_label = ttk.Label(self.turn_frame, text="WHITE'S TURN", style='Status.TLabel'); self.turn_label.pack(fill=tk.X)

        self.eval_frame = ttk.Frame(self.left_panel, style='Left.TFrame'); self.eval_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=0, pady=(5,5))
        self.eval_score_label = ttk.Label(self.eval_frame, text="Even", style='Status.TLabel', anchor="center"); self.eval_score_label.pack(pady=(7,5))
        self.eval_bar_canvas = tk.Canvas(self.eval_frame, height=26, bg=self.COLORS['bg_light'], highlightthickness=0); self.eval_bar_canvas.pack(fill=tk.X, expand=True)
        self.eval_bar_canvas.bind("<Configure>", lambda event: self.draw_eval_bar(0))

        self.right_panel = ttk.Frame(self.main_frame, style='Right.TFrame'); self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=15, pady=15)
        self.canvas_container = ttk.Frame(self.right_panel, style='Canvas.TFrame'); self.canvas_container.pack(expand=True, fill=tk.BOTH)
        self.canvas_container.grid_rowconfigure(0, weight=1); self.canvas_container.grid_columnconfigure(0, weight=1)
        self.canvas_frame = ttk.Frame(self.canvas_container, style='Canvas.TFrame'); self.canvas_frame.grid(row=0, column=0)
        
        self.canvas = tk.Canvas(self.canvas_frame, width=COLS * SQUARE_SIZE, height=ROWS * SQUARE_SIZE, bg=self.COLORS['bg_light'], highlightthickness=0)
        self.board_image_white = self.create_board_image("white"); self.board_image_black = self.create_board_image("black")
        self.board_image_id = self.canvas.create_image(0, 0, image=self.board_image_white, anchor='nw', tags="board"); self.canvas.pack()

        self.scoreboard_frame = ttk.Frame(self.right_panel, style='Right.TFrame')
        self.scoreboard_label = ttk.Label(self.scoreboard_frame, text="", font=("Helvetica", 10), background=self.COLORS['bg_medium'], foreground=self.COLORS['text_light']); self.scoreboard_label.pack()

        self.top_bot_label = ttk.Label(self.right_panel, text="", font=("Helvetica", 12), background=self.COLORS['bg_medium'], foreground=self.COLORS['text_light']); self.top_bot_label.place(relx=0.5, rely=0.02, anchor='n')
        self.bottom_bot_label = ttk.Label(self.right_panel, text="", font=("Helvetica", 12), background=self.COLORS['bg_medium'], foreground=self.COLORS['text_light']); self.bottom_bot_label.place(relx=0.5, rely=0.98, anchor='s')

        self.human_color = "white"
        self.board = create_initial_board()
        self.turn = "white"
        self.selected, self.valid_moves, self.game_over = None, [], False
        self.dragging, self.drag_piece, self.drag_start = False, None, None 
        
        self.bot = None

        self.process_ai_queue()
        self.reset_game()
        self.draw_board()
    
    # ... (no changes to process_ai_queue, _ai_worker, _start_ai_thread, _interrupt_ai_search, on_drag_end, make_bot_move, make_ai_move, reset_game) ...
    def process_ai_queue(self):
        try:
            move_success = self.ai_move_queue.get_nowait()
            self.ai_is_thinking = False
            self.update_bot_labels()

            if move_success:
                self.draw_board()
                self.execute_move_and_check_state()
                if not self.game_over and self.game_mode.get() == GameMode.AI_VS_AI.value:
                    delay = 20 if self.instant_move.get() else 500
                    self.master.after(delay, self.make_ai_move)
            else:
                print("AI reported no valid move was made (or was cancelled).")
        except queue.Empty:
            pass
        finally:
            self.master.after(100, self.process_ai_queue)

    def _ai_worker(self, bot_instance):
        success = bot_instance.make_move()
        if not self.cancellation_event.is_set():
            self.ai_move_queue.put(success)

    def _start_ai_thread(self, bot_instance):
        if self.ai_is_thinking or self.game_over:
            return
        self.cancellation_event.clear()
        self.ai_is_thinking = True
        self.update_bot_labels()
        thread = threading.Thread(target=self._ai_worker, args=(bot_instance,), daemon=True)
        thread.start()
        
    def _interrupt_ai_search(self):
        if self.ai_is_thinking:
            print("UI: Sending cancellation signal to AI...")
            self.cancellation_event.set()
            self.ai_is_thinking = False
            self.update_bot_labels()

    def on_drag_end(self, event):
        if not self.dragging and not self.selected:
            self.valid_moves = []
            self.draw_board()
            return

        row, col = self.canvas_to_board(event.x, event.y)
        if row == -1:
            self.dragging = False; self.drag_piece = None; self.drag_start = None
            self.selected = None; self.valid_moves = []; self.draw_board()
            self.set_interactivity()
            return

        start_pos = self.selected if self.selected else self.drag_start
        if not start_pos:
            self.dragging = False; self.drag_piece = None; self.drag_start = None; self.selected = None; self.valid_moves = []; self.draw_board(); return

        end_pos = (row, col)
        if end_pos in self.valid_moves:
            moving_piece = self.board[start_pos[0]][start_pos[1]]
            self.board = moving_piece.move(self.board, start_pos, end_pos)
            check_evaporation(self.board)
            self.execute_move_and_check_state()
            
            if not self.game_over and self.game_mode.get() == GameMode.HUMAN_VS_BOT.value and self.turn != self.human_color:
                delay = 20 if self.instant_move.get() else 500
                self.master.after(delay, self.make_bot_move)
        
        self.dragging = False; self.drag_piece = None; self.drag_start = None; self.selected = None; self.valid_moves = []
        self.draw_board()
        self.set_interactivity()

    def make_bot_move(self):
        if self.game_over or not self.bot or self.ai_is_thinking: 
            return
        self._start_ai_thread(self.bot)

    def make_ai_move(self):
        if self.game_mode.get() != GameMode.AI_VS_AI.value or self.game_over or self.ai_is_thinking:
            return
        current_bot = self.ai_white_bot if self.turn == "white" else self.ai_black_bot
        if not current_bot:
            return
        self._start_ai_thread(current_bot)

    def reset_game(self):
        self._interrupt_ai_search()

        self.board = create_initial_board()
        self.turn = "white" 
        self.selected = None; self.valid_moves = []; self.game_over = False
        self.game_result = None; self.dragging = False; self.drag_piece = None; self.drag_start = None
        
        self.position_history = [] 
        self.position_counts = {}  
        initial_key = self.get_position_key()
        self.position_history.append(initial_key)
        self.position_counts[initial_key] = 1

        current_mode = self.game_mode.get()
        delay = 20 if self.instant_move.get() else 500

        if current_mode == GameMode.AI_VS_AI.value:
            if self.white_playing_bot == "main":
                self.ai_white_bot = ChessBot(self.board, "white", self, self.cancellation_event)
                self.ai_black_bot = OpponentAI(self.board, "black", self, self.cancellation_event)
            else: 
                self.ai_white_bot = OpponentAI(self.board, "white", self, self.cancellation_event)
                self.ai_black_bot = ChessBot(self.board, "black", self, self.cancellation_event)
            
            self.update_bot_depth(self.bot_depth_slider.get())
            if self.ai_series_running:
                 self.randomize_white_opening()
            
            self.turn_label.config(text=f"Turn: {self.turn.capitalize()} (AI vs AI)")
            if not self.game_over:
                 self.master.after(delay, self.make_ai_move)

        elif current_mode == GameMode.HUMAN_VS_BOT.value:
            bot_color = "black" if self.human_color == "white" else "white"
            self.bot = ChessBot(self.board, bot_color, self, self.cancellation_event)
            self.update_bot_depth(self.bot_depth_slider.get())
            self.turn_label.config(text=f"Turn: {self.turn.capitalize()}")
            if self.turn != self.human_color and not self.game_over: 
                self.master.after(delay, self.make_bot_move)
        else: # Human vs Human
            self.turn_label.config(text=f"Turn: {self.turn.capitalize()}")

        self.draw_board()
        self.set_interactivity()
        self.update_bot_labels() 
        self.update_scoreboard()
        
    def set_interactivity(self):
        if self.game_mode.get() == GameMode.AI_VS_AI.value or self.ai_is_thinking:
            self.canvas.unbind("<Button-1>")
            self.canvas.unbind("<B1-Motion>")
            self.canvas.unbind("<ButtonRelease-1>")
        else:
            self.canvas.bind("<Button-1>", self.on_drag_start)
            self.canvas.bind("<B1-Motion>", self.on_drag_motion)
            self.canvas.bind("<ButtonRelease-1>", self.on_drag_end)

    def update_bot_depth(self, value):
        new_depth = int(value)
        ChessBot.search_depth = new_depth
        # Check if bots exist and have the attribute before setting, preventing errors
        if self.bot and hasattr(self.bot, 'search_depth'): 
            self.bot.search_depth = new_depth
        if self.ai_white_bot and hasattr(self.ai_white_bot, 'search_depth'):
            self.ai_white_bot.search_depth = new_depth
        if self.ai_black_bot and hasattr(self.ai_black_bot, 'search_depth'):
            self.ai_black_bot.search_depth = new_depth

    def get_position_key(self):
        ### MODIFIED: This now correctly calls the fast, global hashing function.
        return board_hash(self.board, self.turn)

    def swap_sides(self):
        self._interrupt_ai_search()
        current_mode = self.game_mode.get()
        if current_mode == GameMode.HUMAN_VS_BOT.value:
            self.human_color = "black" if self.human_color == "white" else "white"
            self.board_orientation = self.human_color
        elif current_mode == GameMode.AI_VS_AI.value:
            self.white_playing_bot = "op" if self.white_playing_bot == "main" else "main"
            self.board_orientation = "white" if self.white_playing_bot == "main" else "black"
        elif current_mode == GameMode.HUMAN_VS_HUMAN.value:
            self.board_orientation = "black" if self.board_orientation == "white" else "white"
        self.reset_game()

    # ... (rest of UI file is unchanged)
    def board_to_canvas(self, r, c):
        if self.board_orientation == "white":
            x1, y1 = c * SQUARE_SIZE, r * SQUARE_SIZE
        else:
            x1, y1 = (COLS - 1 - c) * SQUARE_SIZE, (ROWS - 1 - r) * SQUARE_SIZE
        return x1, y1

    def canvas_to_board(self, x, y):
        if self.board_orientation == "white":
            row, col = y // SQUARE_SIZE, x // SQUARE_SIZE
        else:
            row, col = (ROWS - 1) - (y // SQUARE_SIZE), (COLS - 1) - (x // SQUARE_SIZE)
        return (row, col) if 0 <= row < ROWS and 0 <= col < COLS else (-1, -1)

    def draw_eval_bar(self, eval_score_from_ai):
        pawn_equivalent_score = eval_score_from_ai / 100.0
        self.eval_bar_canvas.delete("all")
        bar_width = self.eval_bar_canvas.winfo_width()
        bar_height = self.eval_bar_canvas.winfo_height()

        if bar_width <= 1 or bar_height <= 1:
            self.eval_score_label.config(text="Eval: ...", font=("Helvetica", 10))
            return

        pawn_scaling_for_tanh = 20.0 
        normalized_marker_score = max(-1.0, min(1.0, math.tanh(pawn_equivalent_score / pawn_scaling_for_tanh)))
        marker_x = int(((normalized_marker_score + 1) / 2.0) * bar_width)
        
        for x_pixel in range(bar_width):
            ratio = x_pixel / float(bar_width - 1 if bar_width > 1 else 1)
            intensity = int(255 * ratio)
            color = f"#{intensity:02x}{intensity:02x}{intensity:02x}"
            self.eval_bar_canvas.create_line(x_pixel, 0, x_pixel, bar_height, fill=color, width=1)

        accent_color = self.COLORS.get('accent', '#e94560')
        self.eval_bar_canvas.create_line(marker_x, 0, marker_x, bar_height, fill=accent_color, width=3)
        mid_x = bar_width // 2
        self.eval_bar_canvas.create_line(mid_x, 0, mid_x, bar_height, fill="#666666", width=1)

        if abs(pawn_equivalent_score) < 0.05: 
            self.eval_score_label.config(text="Even", font=("Helvetica", 10))
        else:
            prefix = "+" if pawn_equivalent_score > 0 else ""
            self.eval_score_label.config(text=f"{prefix}{pawn_equivalent_score:.2f}", font=("Helvetica", 10))

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        COLORS = {
            'bg_dark': '#1a1a2e', 'bg_medium': '#16213e', 'bg_light': '#0f3460',
            'accent': '#e94560', 'text_light': '#ffffff', 'text_dark': '#a2a2a2'
        }
        style.configure('Left.TFrame', background=COLORS['bg_dark'])
        style.configure('Right.TFrame', background=COLORS['bg_medium'])
        style.configure('Canvas.TFrame', background=COLORS['bg_medium'])
        style.configure('Header.TLabel', background=COLORS['bg_dark'], foreground=COLORS['text_light'], font=('Helvetica', 14, 'bold'), padding=(0, 10))
        style.configure('Status.TLabel', background=COLORS['bg_light'], foreground=COLORS['text_light'], font=('Helvetica', 16, 'bold'), padding=(11, 4), relief='flat', borderwidth=0)
        style.configure('Control.TButton', background=COLORS['accent'], foreground=COLORS['text_light'], font=('Helvetica', 11, 'bold'), padding=(10, 8), borderwidth=0, relief='flat')
        style.map('Control.TButton', background=[('active', COLORS['accent']), ('pressed', '#d13550')], relief=[('pressed', 'flat'), ('!pressed', 'flat')])
        style.configure('Custom.TRadiobutton', background=COLORS['bg_dark'], foreground=COLORS['text_light'], font=('Helvetica', 11), padding=(5, 8))
        style.map('Custom.TRadiobutton', background=[('active', COLORS['bg_dark'])], indicatorcolor=[('selected', COLORS['accent'])], foreground=[('active', COLORS['accent'])])
        style.configure('Custom.TCheckbutton', background=COLORS['bg_dark'], foreground=COLORS['text_light'], font=('Helvetica', 11), padding=(5, 8))
        style.map('Custom.TCheckbutton', background=[('active', COLORS['bg_dark'])], indicatorcolor=[('selected', COLORS['accent'])], foreground=[('active', COLORS['accent'])])
        return COLORS

    def create_board_image(self, orientation):
        board_image = tk.PhotoImage(width=COLS * SQUARE_SIZE, height=ROWS * SQUARE_SIZE)
        for r_logic in range(ROWS):
            for c_logic in range(COLS):
                is_light_square = (r_logic + c_logic) % 2 == 0 
                color = BOARD_COLOR_1 if is_light_square else BOARD_COLOR_2
                x1, y1 = self.board_to_canvas(r_logic, c_logic) if orientation == "black" else (c_logic * SQUARE_SIZE, r_logic * SQUARE_SIZE)
                x2, y2 = x1 + SQUARE_SIZE, y1 + SQUARE_SIZE
                board_image.put(color, to=(x1, y1, x2, y2))
        return board_image

    def draw_board(self):
        self.canvas.itemconfig(self.board_image_id, image=self.board_image_white if self.board_orientation == "white" else self.board_image_black)
        self.canvas.delete("highlight", "piece", "drag", "check_highlight")
        for (r, c) in self.valid_moves:
            x1, y1 = self.board_to_canvas(r, c)
            oval_radius = SQUARE_SIZE * 0.15 
            center_x, center_y = x1 + SQUARE_SIZE // 2, y1 + SQUARE_SIZE // 2
            self.canvas.create_oval(center_x - oval_radius, center_y - oval_radius, center_x + oval_radius, center_y + oval_radius, fill="#1E90FF", outline="", tags="highlight")
        for r in range(ROWS):
            for c in range(COLS):
                piece = self.board[r][c]
                if piece and isinstance(piece, King) and is_in_check(self.board, piece.color):
                    highlight_color = "darkred" if not has_legal_moves(self.board, piece.color) else "red"
                    x1, y1 = self.board_to_canvas(r, c)
                    x2, y2 = x1 + SQUARE_SIZE, y1 + SQUARE_SIZE
                    self.canvas.create_rectangle(x1, y1, x2, y2, outline=highlight_color, width=3, tags="check_highlight")
                if piece and (r, c) != self.drag_start:
                    self.draw_piece_at_canvas_coords(piece, r, c)
        if self.dragging and self.drag_piece and self.drag_start:
            piece_obj = self.board[self.drag_start[0]][self.drag_start[1]]
            if piece_obj:
                font_to_use = ("Arial Unicode MS", 36)
                drag_text_color = "white" if piece_obj.color == "white" else "black"
                self.canvas.create_text(self.drag_piece[0], self.drag_piece[1], text=piece_obj.symbol(), font=font_to_use, fill=drag_text_color, tags="drag")

    def draw_piece_at_canvas_coords(self, piece, r_board, c_board):
        x_cvs, y_cvs = self.board_to_canvas(r_board, c_board)
        x_center, y_center = x_cvs + SQUARE_SIZE // 2, y_cvs + SQUARE_SIZE // 2
        symbol = piece.symbol()
        font_to_use = ("Arial Unicode MS", 39)
        font_shadow = ("Arial", 39)
        if piece.color == "white":
            self.canvas.create_text(x_center + 2, y_center + 2, text=symbol, font=font_shadow, fill="#444444", tags="piece")
            self.canvas.create_text(x_center, y_center, text=symbol, font=font_to_use, fill="white", tags="piece")
        else:
            self.canvas.create_text(x_center, y_center, text=symbol, font=font_to_use, fill="black", tags="piece")

    def on_drag_start(self, event):
        if self.game_over: return
        r, c = self.canvas_to_board(event.x, event.y)
        if r != -1:
            piece = self.board[r][c]
            if piece and piece.color == self.turn:
                self.selected = (r, c)
                self.dragging = True
                self.drag_start = (r, c)
                self.drag_piece = (event.x, event.y)
                self.valid_moves = [m for m in piece.get_valid_moves(self.board, (r, c)) if validate_move(self.board, self.turn, (r,c), m)]
                self.draw_board()

    def on_drag_motion(self, event):
        if self.dragging:
            self.drag_piece = (event.x, event.y)
            self.draw_board()

    def execute_move_and_check_state(self):
        if self.game_over: return
        
        key = self.get_position_key()
        self.position_history.append(key)
        self.position_counts[key] = self.position_counts.get(key, 0) + 1
        
        if self.position_counts.get(key, 0) >= 3:
            self.game_over = True
            self.game_result = ("repetition", None)
        else:
            status, winner = check_game_over(self.board, self.turn)
            if status:
                self.game_over = True
                self.game_result = (status, winner)
        
        if self.game_over:
            result_type, winner_color = self.game_result
            msg = "Game Over"
            if result_type == "repetition":
                msg = "Draw by three-fold repetition!"
            elif result_type == "stalemate":
                msg = "Stalemate! It's a draw."
            elif result_type == "checkmate":
                msg = f"Checkmate! {winner_color.capitalize()} wins!"
            elif result_type == "king_capture":
                msg = f"{winner_color.capitalize()} wins by king capture!"
            
            self.turn_label.config(text=msg)
            
            if self.game_mode.get() == GameMode.AI_VS_AI.value:
                self.process_ai_series_result()
        else:
            self.turn = "black" if self.turn == "white" else "white"
            self.turn_label.config(text=f"Turn: {self.turn.capitalize()}")

        self.set_interactivity()
        self.update_bot_labels()

    def process_ai_series_result(self):
        if not self.game_result: return
        self.ai_series_stats['game_count'] += 1
        result_type, winner_color = self.game_result
        if result_type in ["repetition", "stalemate"]:
            self.ai_series_stats['draws'] += 1
        else:
            winning_ai = self.white_playing_bot if winner_color == "white" else ("op" if self.white_playing_bot == "main" else "main")
            if winning_ai == "main": self.ai_series_stats['my_ai_wins'] += 1
            else: self.ai_series_stats['op_ai_wins'] += 1
        self.update_scoreboard()
        if self.ai_series_stats['game_count'] < 100: 
            self.master.after(1000, self.reset_game) 
        else:
            self.turn_label.config(text="AI Series: 100 Games Complete!")
            self.ai_series_running = False

    def start_ai_series(self):
        self._interrupt_ai_search()
        random.seed()
        self.game_mode.set(GameMode.AI_VS_AI.value)
        self.ai_series_stats = {'game_count': 0, 'my_ai_wins': 0, 'op_ai_wins': 0, 'draws': 0}
        self.ai_series_running = True 
        self.white_playing_bot = "main"  
        self.board_orientation = "white" 
        self.reset_game() 

    def update_scoreboard(self):
        if self.game_mode.get() == GameMode.AI_VS_AI.value and self.ai_series_running:
            stats = self.ai_series_stats
            self.scoreboard_label.config(text=f"AI vs OP Score (Game {stats['game_count']}/100):\n  Your AI: {stats['my_ai_wins']}\n  Opponent AI: {stats['op_ai_wins']}\n  Draws: {stats['draws']}")
            if not self.scoreboard_frame.winfo_ismapped():
                 self.scoreboard_frame.place(relx=1.0, rely=0.0, anchor='ne', x=-15, y=15)
        else:
            if self.scoreboard_frame.winfo_ismapped():
                 self.scoreboard_frame.place_forget()

    def update_bot_labels(self):
        top_text, bottom_text = "", ""
        current_mode = self.game_mode.get()
        thinking_text = " (Thinking...)" if self.ai_is_thinking else ""

        if current_mode == GameMode.HUMAN_VS_BOT.value:
            human_actual_color = self.human_color
            bot_actual_color = "black" if human_actual_color == "white" else "white"
            bot_label = f"Bot ({bot_actual_color.capitalize()})" + (thinking_text if self.turn == bot_actual_color else "")
            
            if self.board_orientation == human_actual_color:
                bottom_text = f"Human ({human_actual_color.capitalize()})"
                top_text = bot_label
            else:
                bottom_text = bot_label
                top_text = f"Human ({human_actual_color.capitalize()})"
        elif current_mode == GameMode.AI_VS_AI.value:
            main_ai_name, op_ai_name = "MyAIBot", "OpponentAI"
            white_player_name = main_ai_name if self.white_playing_bot == "main" else op_ai_name
            black_player_name = op_ai_name if self.white_playing_bot == "main" else main_ai_name
            
            white_text = f"{white_player_name} (White)" + (thinking_text if self.turn == 'white' else "")
            black_text = f"{black_player_name} (Black)" + (thinking_text if self.turn == 'black' else "")

            if self.board_orientation == "white": 
                bottom_text = white_text
                top_text = black_text
            else: 
                bottom_text = black_text
                top_text = white_text
        elif current_mode == GameMode.HUMAN_VS_HUMAN.value:
            if self.board_orientation == "white": bottom_text, top_text = "White", "Black"
            else: bottom_text, top_text = "Black", "White"
        self.top_bot_label.config(text=top_text)
        self.bottom_bot_label.config(text=bottom_text)

    def randomize_white_opening(self):
        if self.turn != "white": return
        moves = []
        for r in range(ROWS):
            for c in range(COLS):
                piece = self.board[r][c]
                if piece and piece.color == "white":
                    for move in piece.get_valid_moves(self.board, (r, c)):
                        if validate_move(self.board, "white", (r,c), move):
                            moves.append(((r,c), move))
        if moves:
            start, end = random.choice(moves)
            piece = self.board[start[0]][start[1]]
            self.board = piece.move(self.board, start, end)
            check_evaporation(self.board)
            self.turn = "black" 
            key = self.get_position_key()
            self.position_history.append(key)
            self.position_counts[key] = 1

def main_app():
    root = tk.Tk()
    app = EnhancedChessApp(root)
    root.mainloop()

if __name__ == "__main__":
    main_app()