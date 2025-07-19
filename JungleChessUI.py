import tkinter as tk
from tkinter import ttk
import math
import random  # Used for randomizing white's opening move
from GameLogic import *
from AI import ChessBot
from OpponentAI import OpponentAI  # Ensure OpponentAI is implemented
from enum import Enum

class GameMode(Enum):
    HUMAN_VS_BOT = "bot"
    HUMAN_VS_HUMAN = "human"
    AI_VS_AI = "ai_vs_ai"

class EnhancedChessApp:
    def __init__(self, master): # Corrected from init to __init__
        """Initialize the main application, including UI components, game state, and bot settings."""
        self.master = master
        self.master.title("Enhanced Chess")
        self.COLORS = self.setup_styles()
        self.master.configure(bg=self.COLORS['bg_dark'])
        self.position_history = []
        self.position_counts = {}
        self.game_result = None
        self.ai_series_running = False
        self.board_orientation = "white"  # Which side is at the bottom
        self.game_mode = tk.StringVar(value=GameMode.HUMAN_VS_BOT.value)

        # Initialize AI vs AI series counters and settings
        self.ai_series_stats = {
            'game_count': 0,
            'my_ai_wins': 0,
            'op_ai_wins': 0,
            'draws': 0
        }
        self.ai_series_running = False
        self.ai_white_bot = None  # Bot playing white in AI vs AI mode
        self.ai_black_bot = None  # Bot playing black in AI vs AI mode
        # FIX: Initialize self.white_playing_bot for robustness
        self.white_playing_bot = "main"  # 'main' bot (your AI) defaults to white

        # Set up main window size and full-screen mode
        screen_w = self.master.winfo_screenwidth()
        screen_h = self.master.winfo_screenheight()
        self.master.geometry(f"{screen_w}x{screen_h}+0+0")
        self.master.state('zoomed')
        self.fullscreen = True

        # Main frame (container for sidebar and game board)
        self.main_frame = ttk.Frame(master, style='Left.TFrame')
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # Left Panel: Sidebar with game mode selection, controls, and bot settings
        self.left_panel = ttk.Frame(self.main_frame, width=250, style='Left.TFrame')
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=15, pady=(0,15))
        self.left_panel.pack_propagate(False)

        # Sidebar header
        ttk.Label(self.left_panel, text="JUNGLE CHESS", style='Header.TLabel',
                  font=('Helvetica', 24, 'bold')).pack(pady=(0,10))

        # Game mode selection
        game_mode_frame = ttk.Frame(self.left_panel, style='Left.TFrame')
        game_mode_frame.pack(fill=tk.X, pady=(0,9))
        ttk.Label(game_mode_frame, text="GAME MODE", style='Header.TLabel').pack(anchor=tk.W)
        ttk.Radiobutton(game_mode_frame, text="Human vs Bot", variable=self.game_mode,
                value=GameMode.HUMAN_VS_BOT.value, command=self.reset_game,
                style='Custom.TRadiobutton').pack(anchor=tk.W, pady=(5,3))
        ttk.Radiobutton(game_mode_frame, text="Human vs Human", variable=self.game_mode,
                        value=GameMode.HUMAN_VS_HUMAN.value, command=self.reset_game,
                        style='Custom.TRadiobutton').pack(anchor=tk.W)
        # AI_VS_AI Radiobutton - Assuming it exists or will be added if not implicitly managed by "AI vs OP start"
        ttk.Radiobutton(game_mode_frame, text="AI vs AI", variable=self.game_mode,
                        value=GameMode.AI_VS_AI.value, command=self.reset_game, # Or specific command
                        style='Custom.TRadiobutton').pack(anchor=tk.W)


        # Controls frame with buttons
        controls_frame = ttk.Frame(self.left_panel, style='Left.TFrame')
        controls_frame.pack(fill=tk.X, pady=4)
        ttk.Button(controls_frame, text="NEW GAME", command=self.reset_game,
                   style='Control.TButton').pack(fill=tk.X, pady=3)
        ttk.Button(controls_frame, text="SWAP SIDES", command=self.swap_sides,
                   style='Control.TButton').pack(fill=tk.X, pady=3)
        ttk.Button(controls_frame, text="AI vs OP start", command=self.start_ai_series,
                   style='Control.TButton').pack(fill=tk.X, pady=3)
        ttk.Button(controls_frame, text="QUIT", command=self.master.quit,
                   style='Control.TButton').pack(fill=tk.X, pady=3)

        # Bot settings: depth slider
        ttk.Label(controls_frame, text="Bot Depth:", style='Header.TLabel').pack(anchor=tk.W, pady=(9,0))
        self.bot_depth_slider = tk.Scale(controls_frame, from_=1, to=6, orient=tk.HORIZONTAL,
                                         command=self.update_bot_depth,
                                         bg=self.COLORS['bg_dark'], fg=self.COLORS['text_light'],
                                         highlightthickness=0)
        # Check if ChessBot.search_depth exists, otherwise default
        initial_bot_depth = getattr(ChessBot, 'search_depth', 3) # Default to 3 if not found
        self.bot_depth_slider.set(initial_bot_depth)
        if not hasattr(ChessBot, 'search_depth'): # If it didn't exist, set it now
            ChessBot.search_depth = initial_bot_depth

        self.bot_depth_slider.pack(fill=tk.X, pady=(0,3))


        # Instant Move checkbox
        self.instant_move = tk.BooleanVar(value=False)
        # Ensure 'Custom.TCheckbutton' style is defined or use 'Custom.TRadiobutton' if that's intended style
        ttk.Checkbutton(controls_frame, text="Instant Move", variable=self.instant_move,
                        style='Custom.TCheckbutton').pack(anchor=tk.W, pady=(3,3))


        # Turn display
        self.turn_frame = ttk.Frame(self.left_panel, style='Left.TFrame')
        self.turn_frame.pack(fill=tk.X, pady=(9,0))
        self.turn_label = ttk.Label(self.turn_frame, text="WHITE'S TURN", style='Status.TLabel')
        self.turn_label.pack(fill=tk.X)

        # Evaluation frame
        self.eval_frame = ttk.Frame(self.left_panel, style='Left.TFrame')
        self.eval_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=0, pady=(5,5))
        self.eval_score_label = ttk.Label(self.eval_frame, text="Even", style='Status.TLabel', anchor="center")
        self.eval_score_label.pack(pady=(7,5))
        self.eval_bar_canvas = tk.Canvas(self.eval_frame, height=26,
                                         bg=self.COLORS['bg_light'], highlightthickness=0)
        self.eval_bar_canvas.pack(fill=tk.X, expand=True)
        self.eval_bar_canvas.bind("<Configure>", lambda event: self.draw_eval_bar(0))
        self.draw_eval_bar(0)

        # Right Panel: Game board display
        self.right_panel = ttk.Frame(self.main_frame, style='Right.TFrame')
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=15, pady=15)
        self.canvas_container = ttk.Frame(self.right_panel, style='Canvas.TFrame')
        self.canvas_container.pack(expand=True, fill=tk.BOTH)
        self.canvas_container.grid_rowconfigure(0, weight=1)
        self.canvas_container.grid_columnconfigure(0, weight=1)
        self.canvas_frame = ttk.Frame(self.canvas_container, style='Canvas.TFrame')
        self.canvas_frame.grid(row=0, column=0)
        
        # Assume COLS, ROWS, SQUARE_SIZE are defined (e.g., from GameLogic or constants file)
        # For placeholder:
        global COLS, ROWS, SQUARE_SIZE, BOARD_COLOR_1, BOARD_COLOR_2
        COLS, ROWS = 8, 8
        SQUARE_SIZE = 60 
        BOARD_COLOR_1, BOARD_COLOR_2 = "#DDB88C", "#A66D4F"


        self.canvas = tk.Canvas(self.canvas_frame,
                                width=COLS * SQUARE_SIZE,
                                height=ROWS * SQUARE_SIZE,
                                bg=self.COLORS['bg_light'],
                                highlightthickness=0)
        self.board_image_white = self.create_board_image("white")
        self.board_image_black = self.create_board_image("black")
        self.board_image_id = self.canvas.create_image(0, 0, image=self.board_image_white, 
                                                    anchor='nw', tags="board")
        self.canvas.pack()

        # Scoreboard for AI vs AI series
        self.scoreboard_frame = ttk.Frame(self.right_panel, style='Right.TFrame')
        # self.scoreboard_frame.place(relx=1.0, rely=0.0, anchor='ne', x=-15, y=15) # Initial placement handled by update_scoreboard
        self.scoreboard_label = ttk.Label(self.scoreboard_frame,
                                          text="AI vs OP Score:\nWhite: 0\nBlack: 0\nDraws: 0\nGames: 0/100",
                                          font=("Helvetica", 10),
                                          background=self.COLORS['bg_medium'],
                                          foreground=self.COLORS['text_light'])
        self.scoreboard_label.pack()

        # Bot labels
        self.top_bot_label = ttk.Label(self.right_panel, text="", font=("Helvetica", 12),
                                       background=self.COLORS['bg_medium'],
                                       foreground=self.COLORS['text_light'])
        self.top_bot_label.place(relx=0.5, rely=0.02, anchor='n')
        self.bottom_bot_label = ttk.Label(self.right_panel, text="", font=("Helvetica", 12),
                                          background=self.COLORS['bg_medium'],
                                          foreground=self.COLORS['text_light'])
        self.bottom_bot_label.place(relx=0.5, rely=0.98, anchor='s')

        # Initialize game state
        self.human_color = "white"
        self.board = create_initial_board()
        self.turn = "white"
        self.selected = None 
        self.valid_moves = []
        self.game_over = False
        self.dragging = False
        self.drag_piece = None 
        self.drag_start = None 
        bot_color = "black" if self.human_color == "white" else "white"
        self.bot = ChessBot(self.board, bot_color, self) 

        # Set initial interactivity and draw board
        self.set_interactivity()
        self.update_bot_labels() # Call this to set initial labels
        self.update_scoreboard() # Call this to set initial scoreboard visibility
        self.draw_board()
    
    # ... (rest of the methods from EnhancedChessApp, e.g. set_interactivity, update_bot_depth, etc.) ...
    # Ensure all methods are correctly indented within the class

    def set_interactivity(self):
        """Bind or unbind mouse events based on the current game mode."""
        if self.game_mode.get() == GameMode.AI_VS_AI.value:
            self.canvas.unbind("<Button-1>")
            self.canvas.unbind("<B1-Motion>")
            self.canvas.unbind("<ButtonRelease-1>")
        else:
            self.canvas.bind("<Button-1>", self.on_drag_start)
            self.canvas.bind("<B1-Motion>", self.on_drag_motion)
            self.canvas.bind("<ButtonRelease-1>", self.on_drag_end)

    def update_bot_depth(self, value):
        """Update the search depth for all bot instances."""
        new_depth = int(value)
        ChessBot.search_depth = new_depth
        if self.bot: # For Human vs Bot
             self.bot.search_depth = new_depth
        if self.ai_white_bot: # For AI vs AI
            self.ai_white_bot.search_depth = new_depth
        if self.ai_black_bot: # For AI vs AI
            self.ai_black_bot.search_depth = new_depth


    def get_position_key(self):
        """Generate a unique key for the current board position."""
        return generate_position_key(self.board, self.turn)

    def swap_sides(self):
        """Swap sides based on the current game mode and reset the game."""
        current_mode = self.game_mode.get()
        if current_mode == GameMode.HUMAN_VS_BOT.value:
            self.human_color = "black" if self.human_color == "white" else "white"
            self.board_orientation = self.human_color
        elif current_mode == GameMode.AI_VS_AI.value:
            self.white_playing_bot = "op" if self.white_playing_bot == "main" else "main"
            self.board_orientation = "white" if self.white_playing_bot == "main" else "black"
            # self.update_bot_labels() # update_bot_labels is called in reset_game
        elif current_mode == GameMode.HUMAN_VS_HUMAN.value:
            self.board_orientation = "black" if self.board_orientation == "white" else "white"
        self.reset_game()

    def board_to_canvas(self, r, c):
        """Convert board coordinates to canvas coordinates."""
        if self.board_orientation == "white":
            x1 = c * SQUARE_SIZE
            y1 = r * SQUARE_SIZE
        else:
            x1 = (COLS - 1 - c) * SQUARE_SIZE
            y1 = (ROWS - 1 - r) * SQUARE_SIZE
        return x1, y1

    def canvas_to_board(self, x, y):
        """Convert canvas coordinates to board coordinates."""
        if self.board_orientation == "white":
            row = y // SQUARE_SIZE
            col = x // SQUARE_SIZE
        else:
            row = (ROWS - 1) - (y // SQUARE_SIZE)
            col = (COLS - 1) - (x // SQUARE_SIZE)
        # Ensure coordinates are within board limits before returning
        if 0 <= row < ROWS and 0 <= col < COLS:
            return row, col
        return -1, -1 # Indicate off-board


    def draw_eval_bar(self, eval_score_from_ai):
        """Draw evaluation bar to show board advantage."""
        pawn_equivalent_score = eval_score_from_ai / 100.0
        self.eval_bar_canvas.delete("all")
        bar_width = self.eval_bar_canvas.winfo_width()
        bar_height = self.eval_bar_canvas.winfo_height()

        if bar_width <= 1 or bar_height <= 1:
            self.eval_score_label.config(text="Eval: ...", font=("Helvetica", 10))
            return

        pawn_scaling_for_tanh = 20.0 
        normalized_marker_score = math.tanh(pawn_equivalent_score / pawn_scaling_for_tanh)
        normalized_marker_score = max(min(normalized_marker_score, 1.0), -1.0) 

        for x_pixel in range(bar_width):
            ratio = x_pixel / float(bar_width - 1 if bar_width > 1 else 1)
            intensity = int(255 * ratio)
            color = f"#{intensity:02x}{intensity:02x}{intensity:02x}"
            self.eval_bar_canvas.create_line(x_pixel, 0, x_pixel, bar_height, fill=color, width=1)

        marker_x = int(((normalized_marker_score + 1) / 2.0) * bar_width)
        accent_color = self.COLORS.get('accent', '#e94560')
        marker_line_width = 3 
        self.eval_bar_canvas.create_line(marker_x, 0, marker_x, bar_height,
                                         fill=accent_color, width=marker_line_width)
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
        style.configure('Header.TLabel', background=COLORS['bg_dark'], foreground=COLORS['text_light'],
                        font=('Helvetica', 14, 'bold'), padding=(0, 10))
        style.configure('Status.TLabel', background=COLORS['bg_light'], foreground=COLORS['text_light'],
                        font=('Helvetica', 16, 'bold'), padding=(11, 4), relief='flat', borderwidth=0)
        style.configure('Control.TButton', background=COLORS['accent'], foreground=COLORS['text_light'],
                        font=('Helvetica', 11, 'bold'), padding=(10, 8), borderwidth=0, relief='flat')
        style.map('Control.TButton', background=[('active', COLORS['accent']), ('pressed', '#d13550')],
                  relief=[('pressed', 'flat'), ('!pressed', 'flat')])
        style.configure('Custom.TRadiobutton', background=COLORS['bg_dark'], foreground=COLORS['text_light'],
                        font=('Helvetica', 11), padding=(5, 8))
        style.map('Custom.TRadiobutton', background=[('active', COLORS['bg_dark'])],
                  indicatorcolor=[('selected', COLORS['accent'])], foreground=[('active', COLORS['accent'])])
        style.configure('Custom.TCheckbutton', background=COLORS['bg_dark'], foreground=COLORS['text_light'],
                        font=('Helvetica', 11), padding=(5, 8))
        style.map('Custom.TCheckbutton', background=[('active', COLORS['bg_dark'])],
                  indicatorcolor=[('selected', COLORS['accent'])], foreground=[('active', COLORS['accent'])])
        return COLORS

    def create_board_image(self, orientation):
        board_image = tk.PhotoImage(width=COLS * SQUARE_SIZE, height=ROWS * SQUARE_SIZE)
        for r_logic in range(ROWS):
            for c_logic in range(COLS):
                is_light_square = (r_logic + c_logic) % 2 == 0 
                color = BOARD_COLOR_1 if is_light_square else BOARD_COLOR_2
                if orientation == "white":
                    x1_canvas, y1_canvas = c_logic * SQUARE_SIZE, r_logic * SQUARE_SIZE
                else:
                    x1_canvas = (COLS - 1 - c_logic) * SQUARE_SIZE
                    y1_canvas = (ROWS - 1 - r_logic) * SQUARE_SIZE
                x2_canvas, y2_canvas = x1_canvas + SQUARE_SIZE, y1_canvas + SQUARE_SIZE
                board_image.put(color, (x1_canvas, y1_canvas, x2_canvas, y2_canvas))
        return board_image

    def draw_board(self):
        if self.board_orientation == "white":
            self.canvas.itemconfig(self.board_image_id, image=self.board_image_white)
        else:
            self.canvas.itemconfig(self.board_image_id, image=self.board_image_black)
        self.canvas.delete("highlight", "piece", "drag", "check_highlight")
        for (r, c) in self.valid_moves:
            x1, y1 = self.board_to_canvas(r, c)
            oval_radius = SQUARE_SIZE * 0.15 
            center_x, center_y = x1 + SQUARE_SIZE // 2, y1 + SQUARE_SIZE // 2
            self.canvas.create_oval(center_x - oval_radius, center_y - oval_radius,
                                    center_x + oval_radius, center_y + oval_radius,
                                    fill="#1E90FF", outline="", tags="highlight")
        for r in range(ROWS):
            for c in range(COLS):
                piece = self.board[r][c]
                if piece and isinstance(piece, King):
                    if is_in_check(self.board, piece.color):
                        highlight_color = "darkred" if not has_legal_moves(self.board, piece.color) else "red"
                        x1, y1 = self.board_to_canvas(r, c)
                        x2, y2 = x1 + SQUARE_SIZE, y1 + SQUARE_SIZE
                        self.canvas.create_rectangle(x1, y1, x2, y2, outline=highlight_color, 
                                                    width=3, tags="check_highlight")
        for r in range(ROWS):
            for c in range(COLS):
                piece = self.board[r][c]
                if piece is not None and (r, c) != self.drag_start:
                    self.draw_piece_at_canvas_coords(piece, r, c) # Refactored piece drawing
        if self.dragging and self.drag_piece and self.drag_start is not None:
            piece_obj = self.board[self.drag_start[0]][self.drag_start[1]]
            if piece_obj is not None:
                font_to_use = ("Arial Unicode MS", 36)
                drag_text_color = "white" if piece_obj.color == "white" else "black"
                self.canvas.create_text(self.drag_piece[0], self.drag_piece[1],
                                        text=piece_obj.symbol(), font=font_to_use, 
                                        fill=drag_text_color, tags="drag")
        board_width, board_height = COLS * SQUARE_SIZE, ROWS * SQUARE_SIZE
        self.canvas.create_rectangle(0, 0, board_width, board_height,
                                    outline=self.COLORS['accent'], width=4, tags="border")

    def draw_piece_at_canvas_coords(self, piece, r_board, c_board): # Helper for drawing piece
        x_cvs, y_cvs = self.board_to_canvas(r_board, c_board)
        x_center, y_center = x_cvs + SQUARE_SIZE // 2, y_cvs + SQUARE_SIZE // 2
        symbol = piece.symbol()
        font_to_use = ("Arial Unicode MS", 39)
        font_shadow = ("Arial", 39) # Matched size for consistency
        if piece.color == "white":
            shadow_offset = 2
            shadow_color = "#444444" # Darker shadow for better contrast
            self.canvas.create_text(x_center + shadow_offset, y_center + shadow_offset,
                                    text=symbol, font=font_shadow, fill=shadow_color, tags="piece")
            self.canvas.create_text(x_center, y_center, text=symbol, font=font_to_use,
                                    fill="white", tags="piece")
        else: # Black piece
            self.canvas.create_text(x_center, y_center, text=symbol, font=font_to_use,
                                    fill="black", tags="piece")

    def draw_piece(self, r, c): # draw_piece is now simpler by using the helper
        piece = self.board[r][c]
        if piece is not None and (r,c) != self.drag_start:
            self.draw_piece_at_canvas_coords(piece, r, c)

    def on_drag_start(self, event):
        if self.game_over: return
        is_human_turn = False
        current_mode = self.game_mode.get()
        if current_mode == GameMode.HUMAN_VS_HUMAN.value: is_human_turn = True
        elif current_mode == GameMode.HUMAN_VS_BOT.value and self.turn == self.human_color: is_human_turn = True
        if not is_human_turn: return

        row, col = self.canvas_to_board(event.x, event.y)
        if row == -1 : return # Clicked outside board

        piece = self.board[row][col]
        if piece is not None and piece.color == self.turn:
            self.selected = (row, col)
            self.dragging = True
            self.drag_start = (row, col)
            self.drag_piece = (event.x, event.y)
            self.valid_moves = [
                m for m in piece.get_valid_moves(self.board, (row, col))
                if validate_move(self.board, self.turn, (row,col), m)
            ]
            self.draw_board()
        else:
            self.selected = None
            self.valid_moves = []
            self.draw_board() # Redraw to clear old highlights

    def on_drag_motion(self, event):
        if self.dragging:
            self.drag_piece = (event.x, event.y)
            self.draw_board()

    def execute_move_and_check_state(self):
        current_key = self.get_position_key()
        self.position_history.append(current_key)
        self.position_counts[current_key] = self.position_counts.get(current_key, 0) + 1
        
        if self.position_counts[current_key] >= 3:
            self.game_over = True
            self.game_result = ("repetition", None)
            self.turn_label.config(text="Draw by three-fold repetition!")
        else:
            game_over_status = check_game_over(self.board, self.turn)
            result, winner = (None, None)
            if isinstance(game_over_status, tuple): result, winner = game_over_status
            elif isinstance(game_over_status, str): result, winner = "king_capture", game_over_status
            
            if result:
                self.game_over = True
                self.game_result = (result, winner)
                if result == "checkmate": self.turn_label.config(text=f"Checkmate! {winner.capitalize()} wins!")
                elif result == "stalemate": self.turn_label.config(text="Stalemate! It's a draw.")
                elif result == "king_capture": self.turn_label.config(text=f"{winner.capitalize()} wins by king capture!")
            else:
                self.turn = "black" if self.turn == "white" else "white"
                self.turn_label.config(text=f"Turn: {self.turn.capitalize()}")
        self.set_interactivity()


    def on_drag_end(self, event):
        if not self.dragging and not self.selected:
            self.valid_moves = []
            self.draw_board()
            return

        row, col = self.canvas_to_board(event.x, event.y)
        if row == -1: # Released outside board
            self.dragging = False
            self.drag_piece = None
            self.drag_start = None
            self.selected = None
            self.valid_moves = []
            self.draw_board()
            self.set_interactivity()
            return

        start_pos = self.selected if self.selected else self.drag_start
        if not start_pos: # Should not happen if logic is correct
            self.dragging = False; self.drag_piece = None; self.drag_start = None; self.selected = None; self.valid_moves = []; self.draw_board(); return

        end_pos = (row, col)
        if end_pos in self.valid_moves: # validate_move was already checked
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
        if self.game_over or not self.bot: return
        if self.bot.make_move():
            self.draw_board()
            self.execute_move_and_check_state()
        else:
            print(f"Bot ({self.bot.color}) could not make a move.")
            # If bot has no moves, game should be over (checkmate/stalemate)
            # and execute_move_and_check_state should have caught it on previous turn.
            # If not, this might indicate an issue. For now, assume game over logic handles it.


    def process_ai_series_result(self):
        if not self.game_result: return

        self.ai_series_stats['game_count'] += 1
        result_type, winner_color = self.game_result

        if result_type == "repetition":
            self.turn_label.config(text="Draw, 3-fold repetition!")
            self.ai_series_stats['draws'] += 1
        elif result_type == "stalemate":
            self.turn_label.config(text="Stalemate! It's a draw.")
            self.ai_series_stats['draws'] += 1
        elif result_type in ["checkmate", "king_capture"]:
            winning_ai_type = None
            if winner_color == "white": winning_ai_type = self.white_playing_bot
            else: winning_ai_type = "op" if self.white_playing_bot == "main" else "main"

            if winning_ai_type == "main":
                self.ai_series_stats['my_ai_wins'] += 1
                self.turn_label.config(text=f"{result_type.capitalize()}! Your AI ({winner_color}) wins!")
            else: 
                self.ai_series_stats['op_ai_wins'] += 1
                self.turn_label.config(text=f"{result_type.capitalize()}! Opponent AI ({winner_color}) wins!")
        
        self.update_scoreboard()
        
        # Prepare for next game in series
        self.white_playing_bot = "op" if self.white_playing_bot == "main" else "main"
        self.board_orientation = "white" if self.white_playing_bot == "main" else "black" 
        # self.update_bot_labels() # Called in reset_game

        if self.ai_series_stats['game_count'] < 100: 
            self.master.after(1000, self.reset_game) 
        else:
            self.turn_label.config(text="AI Series: 100 Games Complete!")
            self.ai_series_running = False

    def make_ai_move(self):
        if self.game_mode.get() != GameMode.AI_VS_AI.value or self.game_over:
            return

        current_bot = self.ai_white_bot if self.turn == "white" else self.ai_black_bot
        if not current_bot:
            print(f"Error: No bot assigned for {self.turn} in AI vs AI mode.")
            return

        move_made = current_bot.make_move() 
        self.draw_board()

        if move_made:
            current_key = self.get_position_key()
            self.position_history.append(current_key)
            self.position_counts[current_key] = self.position_counts.get(current_key, 0) + 1

            if self.position_counts[current_key] >= 3:
                self.game_over = True
                self.game_result = ("repetition", None)
            else:
                game_over_status = check_game_over(self.board, self.turn)
                outcome, winner = (None, None)
                if isinstance(game_over_status, tuple): outcome, winner = game_over_status
                elif isinstance(game_over_status, str): outcome, winner = "king_capture", game_over_status

                if outcome:
                    self.game_over = True
                    self.game_result = (outcome, winner)
                else: 
                    self.turn = "black" if self.turn == "white" else "white"
                    self.turn_label.config(text=f"Turn: {self.turn.capitalize()} (AI vs AI)")
                    delay = 20 if self.instant_move.get() else 500
                    self.master.after(delay, self.make_ai_move)
        else: 
            print(f"AI Bot ({current_bot.color}) could not make a move in AI vs AI.")
            # Check game over if bot couldn't move (likely stalemate/checkmate)
            game_over_status = check_game_over(self.board, self.turn) 
            outcome, winner = (None, None)
            if isinstance(game_over_status, tuple): outcome, winner = game_over_status
            elif isinstance(game_over_status, str): outcome, winner = "king_capture", game_over_status
            if outcome and not self.game_over : # If game wasn't already over by other means
                 self.game_over = True
                 self.game_result = (outcome, winner)


        # FIX: Remove one of the duplicate calls. Only one call to process_ai_series_result is needed.
        if self.game_over:
            self.process_ai_series_result()


    def start_ai_series(self):
        """Start AI vs AI series."""
        # FIX: Re-seed random for new series
        random.seed()
        self.game_mode.set(GameMode.AI_VS_AI.value)
        self.ai_series_stats = {'game_count': 0, 'my_ai_wins': 0, 'op_ai_wins': 0, 'draws': 0}
        self.ai_series_running = True 
        self.white_playing_bot = "main"  
        self.board_orientation = "white" 
        # self.update_scoreboard() # Called in reset_game
        # self.update_bot_labels() # Called in reset_game
        self.reset_game() 


    def update_scoreboard(self):
        """Update AI vs AI scoreboard."""
        if self.game_mode.get() == GameMode.AI_VS_AI.value and self.ai_series_running:
            self.scoreboard_label.config(
                text=f"AI vs OP Score (Game {self.ai_series_stats['game_count']}/100):\n"
                     f"  Your AI: {self.ai_series_stats['my_ai_wins']}\n"
                     f"  Opponent AI: {self.ai_series_stats['op_ai_wins']}\n"
                     f"  Draws: {self.ai_series_stats['draws']}"
            )
            if not self.scoreboard_frame.winfo_ismapped(): # Place if not already visible
                 self.scoreboard_frame.place(relx=1.0, rely=0.0, anchor='ne', x=-15, y=15)
        else:
            if self.scoreboard_frame.winfo_ismapped(): # Hide if visible
                 self.scoreboard_frame.place_forget()


    def update_bot_labels(self):
        """Update bot labels for AI vs AI mode."""
        top_text, bottom_text = "", ""
        current_mode = self.game_mode.get()

        if current_mode == GameMode.HUMAN_VS_BOT.value:
            human_actual_color = self.human_color
            bot_actual_color = "black" if human_actual_color == "white" else "white"
            if self.board_orientation == human_actual_color: # Human at bottom
                bottom_text = f"Human ({human_actual_color.capitalize()})"
                top_text = f"Bot ({bot_actual_color.capitalize()})"
            else: # Bot at bottom (human is black and board orientation is black)
                bottom_text = f"Bot ({bot_actual_color.capitalize()})"
                top_text = f"Human ({human_actual_color.capitalize()})"

        elif current_mode == GameMode.AI_VS_AI.value:
            main_ai_name, op_ai_name = "MyAIBot", "OpponentAI"
            white_player_name = main_ai_name if self.white_playing_bot == "main" else op_ai_name
            black_player_name = op_ai_name if self.white_playing_bot == "main" else main_ai_name

            if self.board_orientation == "white": 
                bottom_text = f"{white_player_name} (White)"
                top_text = f"{black_player_name} (Black)"
            else: 
                bottom_text = f"{black_player_name} (Black)"
                top_text = f"{white_player_name} (White)"
        
        elif current_mode == GameMode.HUMAN_VS_HUMAN.value:
            if self.board_orientation == "white": bottom_text, top_text = "White", "Black"
            else: bottom_text, top_text = "Black", "White"

        self.top_bot_label.config(text=top_text)
        self.bottom_bot_label.config(text=bottom_text)


    def randomize_white_opening(self):
        """Make a random opening move for white in AI vs AI mode."""
        if self.turn != "white": return
        
        possible_moves = []
        for r in range(ROWS):
            for c in range(COLS):
                piece = self.board[r][c]
                if piece and piece.color == "white":
                    piece_moves = piece.get_valid_moves(self.board, (r, c))
                    for move_end_pos in piece_moves:
                        if validate_move(self.board, "white", (r,c), move_end_pos):
                            possible_moves.append(((r, c), move_end_pos))
        
        if possible_moves:
            start, end = random.choice(possible_moves)
            moving_piece = self.board[start[0]][start[1]]
            self.board = moving_piece.move(self.board, start, end)
            check_evaporation(self.board)
            
            self.turn = "black" 
            
            # Key for position black will see (after white's random move)
            current_key = self.get_position_key() # Key for board state with black to move
            self.position_history.append(current_key) 
            self.position_counts[current_key] = self.position_counts.get(current_key, 0) + 1

            print(f"Random opening move by white: {start} to {end}")
            # self.draw_board() # draw_board is called in reset_game after this
        else:
            print("Warning: No valid random opening moves found for White.")


    def reset_game(self):
        """Reset game state for a new game."""
        self.board = create_initial_board()
        self.turn = "white" 
        self.selected = None; self.valid_moves = []; self.game_over = False
        self.game_result = None; self.dragging = False; self.drag_piece = None; self.drag_start = None
        
        self.position_history = [] 
        self.position_counts = {}  
        # Initial position key: board is set, turn is white
        initial_key = self.get_position_key()
        self.position_history.append(initial_key)
        self.position_counts[initial_key] = 1

        current_mode = self.game_mode.get()

        if current_mode == GameMode.AI_VS_AI.value:
            # self.white_playing_bot should have been set by start_ai_series or process_ai_series_result
            if self.white_playing_bot == "main":
                self.ai_white_bot = ChessBot(self.board, "white", self)
                self.ai_black_bot = OpponentAI(self.board, "black", self)
            else: 
                self.ai_white_bot = OpponentAI(self.board, "white", self)
                self.ai_black_bot = ChessBot(self.board, "black", self)
            
            self.update_bot_depth(self.bot_depth_slider.get())

            if self.ai_series_running: # If series is running, make random opening
                 self.randomize_white_opening() # This updates self.turn to black if successful
                                                # and adds post-random-move state to history
            
            self.turn_label.config(text=f"Turn: {self.turn.capitalize()} (AI vs AI)")
            delay = 20 if self.instant_move.get() else 500
            # Schedule first move if it's AI's turn (could be white if not randomize_white_opening, or black if it was)
            if not self.game_over : # Don't schedule if random opening ended game (highly unlikely)
                 self.master.after(delay, self.make_ai_move)


        elif current_mode == GameMode.HUMAN_VS_BOT.value:
            # Determine bot_color based on self.human_color which is set by swap_sides or default
            bot_color = "black" if self.human_color == "white" else "white"
            self.bot = ChessBot(self.board, bot_color, self)
            self.update_bot_depth(self.bot_depth_slider.get())
            self.turn_label.config(text=f"Turn: {self.turn.capitalize()}")
            if self.turn != self.human_color and not self.game_over: 
                delay = 20 if self.instant_move.get() else 500
                self.master.after(delay, self.make_bot_move)
        else: # Human vs Human
            self.turn_label.config(text=f"Turn: {self.turn.capitalize()}")

        self.draw_board()
        self.set_interactivity()
        self.update_bot_labels() 
        self.update_scoreboard()


    def main(self): 
        """Enter Tkinter main event loop."""
        self.master.mainloop()


def main_app():
    root = tk.Tk()
    app = EnhancedChessApp(root)
    app.main()

if __name__ == "__main__": # Corrected from name == "main"
    main_app()