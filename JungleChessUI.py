import tkinter as tk
from tkinter import ttk
import math
import random  # Used for randomizing white's opening move
from GameLogic import (create_initial_board, ROWS, COLS, SQUARE_SIZE, 
                       BOARD_COLOR_1, BOARD_COLOR_2, generate_position_key, 
                       King, is_in_check, has_legal_moves, validate_move, 
                       check_game_over, check_evaporation)
from AI import ChessBot
from OpponentAI import OpponentAI  # Ensure OpponentAI is implemented
from enum import Enum

class GameMode(Enum):
    HUMAN_VS_BOT = "bot"
    HUMAN_VS_HUMAN = "human"
    AI_VS_AI = "ai_vs_ai"

class EnhancedChessApp:
    def __init__(self, master):
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
        self.bot_depth_slider.set(ChessBot.search_depth)
        self.bot_depth_slider.pack(fill=tk.X, pady=(0,3))

        # Instant Move checkbox
        self.instant_move = tk.BooleanVar(value=False)
        ttk.Checkbutton(controls_frame, text="Instant Move", variable=self.instant_move,
                        style='Custom.TRadiobutton').pack(anchor=tk.W, pady=(3,3))

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
        self.eval_bar_visible = True

        # Right Panel: Game board display
        self.right_panel = ttk.Frame(self.main_frame, style='Right.TFrame')
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=15, pady=15)
        self.canvas_container = ttk.Frame(self.right_panel, style='Canvas.TFrame')
        self.canvas_container.pack(expand=True, fill=tk.BOTH)
        self.canvas_container.grid_rowconfigure(0, weight=1)
        self.canvas_container.grid_columnconfigure(0, weight=1)
        self.canvas_frame = ttk.Frame(self.canvas_container, style='Canvas.TFrame')
        self.canvas_frame.grid(row=0, column=0)
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
        self.scoreboard_frame.place(relx=1.0, rely=0.0, anchor='ne', x=-15, y=15)
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
        self.draw_board()

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
        self.bot.search_depth = new_depth
        if hasattr(self, 'white_bot'):
            self.white_bot.search_depth = new_depth
        if hasattr(self, 'black_bot'):
            self.black_bot.search_depth = new_depth

    def get_position_key(self):
        """Generate a unique key for the current board position."""
        return generate_position_key(self.board, self.turn)

    def swap_sides(self):
        """Swap sides based on the current game mode and reset the game."""
        if self.game_mode.get() == GameMode.HUMAN_VS_BOT.value:
            # Swap human's color and orient the board to human's side
            self.human_color = "black" if self.human_color == "white" else "white"
            self.board_orientation = self.human_color
        elif self.game_mode.get() == GameMode.AI_VS_AI.value:
            # Swap which bot plays white
            self.white_playing_bot = "op" if self.white_playing_bot == "main" else "main"
            # Set board orientation based on the new white-playing bot
            self.board_orientation = "white" if self.white_playing_bot == "main" else "black"
            # Update the labels to show the new bot assignments
            self.update_bot_labels()
        elif self.game_mode.get() == GameMode.HUMAN_VS_HUMAN.value:
            # Flip board orientation for visual purposes
            self.board_orientation = "black" if self.board_orientation == "white" else "white"
        # Reset the game with the new settings
        self.reset_game()

    # Coordinate conversion helpers
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
        return row, col

    
    def draw_eval_bar(self, eval_score):
        """Draw evaluation bar to show board advantage."""
        eval_score /= 100.0
        self.eval_bar_canvas.delete("all")
        bar_width = self.eval_bar_canvas.winfo_width() or 235
        bar_height = 30
        scaling = 23.4
        normalized_score = math.tanh(eval_score / scaling)
        normalized_score = max(min(normalized_score, 1.0), -1.0)

        for x in range(bar_width):
            ratio = x / float(bar_width)
            r = int(255 * ratio)
            g = int(255 * ratio)
            b = int(255 * ratio)
            color = f"#{r:02x}{g:02x}{b:02x}"
            self.eval_bar_canvas.create_line(x, 0, x, bar_height, fill=color)

        marker_x = int((normalized_score + 1) / 2 * bar_width)
        accent_color = self.COLORS.get('accent', '#e94560')
        marker_width = 1
        self.eval_bar_canvas.create_rectangle(marker_x - marker_width, 0,
                                              marker_x + marker_width, bar_height,
                                              fill=accent_color, outline=accent_color)
        mid_x = (bar_width // 2)
        self.eval_bar_canvas.create_line(mid_x, 0, mid_x, bar_height, fill="#666666", width=1)

        if abs(eval_score) < 0.2:
            self.eval_score_label.config(text="Even", font=("Helvetica", 10))
        else:
            display_score = abs(eval_score)
            if eval_score > 0:
                self.eval_score_label.config(text=f"+{display_score:.2f}", font=("Helvetica", 10))
            else:
                self.eval_score_label.config(text=f"-{display_score:.2f}", font=("Helvetica", 10))
        self.master.update_idletasks()

    def setup_styles(self):
        """Define custom styles for widgets."""
        style = ttk.Style()
        style.theme_use('clam')
        COLORS = {
            'bg_dark': '#1a1a2e',
            'bg_medium': '#16213e',
            'bg_light': '#0f3460',
            'accent': '#e94560',
            'text_light': '#ffffff',
            'text_dark': '#a2a2a2'
        }
        style.configure('Left.TFrame', background=COLORS['bg_dark'])
        style.configure('Right.TFrame', background=COLORS['bg_medium'])
        style.configure('Canvas.TFrame', background=COLORS['bg_medium'])
        style.configure('Header.TLabel',
                        background=COLORS['bg_dark'],
                        foreground=COLORS['text_light'],
                        font=('Helvetica', 14, 'bold'),
                        padding=(0, 10))
        style.configure('Status.TLabel',
                        background=COLORS['bg_light'],
                        foreground=COLORS['text_light'],
                        font=('Helvetica', 16, 'bold'),
                        padding=(11, 4),
                        relief='flat',
                        borderwidth=0)
        style.configure('Control.TButton',
                        background=COLORS['accent'],
                        foreground=COLORS['text_light'],
                        font=('Helvetica', 11, 'bold'),
                        padding=(10, 8),
                        borderwidth=0,
                        relief='flat')
        style.map('Control.TButton',
                  background=[('active', COLORS['accent']),
                              ('pressed', '#d13550')],
                  relief=[('pressed', 'flat'),
                          ('!pressed', 'flat')])
        style.configure('Custom.TRadiobutton',
                        background=COLORS['bg_dark'],
                        foreground=COLORS['text_light'],
                        font=('Helvetica', 11),
                        padding=(5, 8))
        style.map('Custom.TRadiobutton',
                  background=[('active', COLORS['bg_dark'])],
                  foreground=[('active', COLORS['accent'])])
        return COLORS

    def create_board_image(self, orientation):
        """Create a static board image for the given orientation."""
        board_image = tk.PhotoImage(width=COLS * SQUARE_SIZE, height=ROWS * SQUARE_SIZE)
        for r in range(ROWS):
            for c in range(COLS):
                # Determine square color based on standard chess coloring (a1 is dark)
                color = BOARD_COLOR_1 if (r + c) % 2 == 0 else BOARD_COLOR_2
                # Map coordinates based on orientation
                if orientation == "white":
                    x1 = c * SQUARE_SIZE
                    y1 = r * SQUARE_SIZE
                else:
                    x1 = (COLS - 1 - c) * SQUARE_SIZE
                    y1 = (ROWS - 1 - r) * SQUARE_SIZE
                x2 = x1 + SQUARE_SIZE
                y2 = y1 + SQUARE_SIZE
                board_image.put(color, (x1, y1, x2, y2))
        return board_image

    def draw_board(self):
        """Render the chess board using a pre-rendered image and update dynamic elements."""
        # Switch board image based on orientation
        if self.board_orientation == "white":
            self.canvas.itemconfig(self.board_image_id, image=self.board_image_white)
        else:
            self.canvas.itemconfig(self.board_image_id, image=self.board_image_black)
        
        # Delete only dynamic elements
        self.canvas.delete("highlight", "piece", "drag")
        
        # Draw valid move highlights
        for (r, c) in self.valid_moves:
            x1, y1 = self.board_to_canvas(r, c)
            x2, y2 = x1 + SQUARE_SIZE, y1 + SQUARE_SIZE
            self.canvas.create_oval(x1 + 19, y1 + 19, x2 - 19, y2 - 19,
                                    fill="#1E90FF", outline="#1E90FF", width=3, tags="highlight")
        
        # Draw check highlights
        for r in range(ROWS):
            for c in range(COLS):
                piece = self.board[r][c]
                if piece and isinstance(piece, King):
                    if is_in_check(self.board, piece.color):
                        highlight_color = "darkred" if not has_legal_moves(self.board, piece.color) else "red"
                        x1, y1 = self.board_to_canvas(r, c)
                        x2, y2 = x1 + SQUARE_SIZE, y1 + SQUARE_SIZE
                        self.canvas.create_rectangle(x1, y1, x2, y2, outline=highlight_color, 
                                                    width=3, tags="highlight")
        
        # Draw pieces (except the one being dragged)
        for r in range(ROWS):
            for c in range(COLS):
                piece = self.board[r][c]
                if piece is not None and (r, c) != self.drag_start:
                    x, y = self.board_to_canvas(r, c)
                    x_center = x + SQUARE_SIZE // 2
                    y_center = y + SQUARE_SIZE // 2
                    symbol = piece.symbol()
                    if piece.color == "white":
                        shadow_offset = 2
                        shadow_color = "#444444"
                        self.canvas.create_text(x_center + shadow_offset, y_center + shadow_offset,
                                                text=symbol, font=("Arial", 39),
                                                fill=shadow_color, tags="piece")
                        self.canvas.create_text(x_center, y_center,
                                                text=symbol, font=("Arial Unicode MS", 39),
                                                fill="white", tags="piece")
                    else:
                        self.canvas.create_text(x_center, y_center, text=symbol,
                                                font=("Arial", 39), fill="black", tags="piece")
        
        # Draw the dragged piece
        if self.dragging and self.drag_piece and self.drag_start is not None:
            piece = self.board[self.drag_start[0]][self.drag_start[1]]
            if piece is not None:
                self.canvas.create_text(self.drag_piece[0], self.drag_piece[1],
                                        text=piece.symbol(), font=("Arial", 36), tags="drag")
        
        # Draw board border
        board_width = COLS * SQUARE_SIZE
        board_height = ROWS * SQUARE_SIZE
        self.canvas.create_rectangle(0, 0, board_width, board_height,
                                    outline=self.COLORS['accent'], width=4, tags="border")

    def draw_piece(self, r, c):
        """Draw a single piece unless it's being dragged."""
        piece = self.board[r][c]
        if piece is not None and (r, c) != self.drag_start:
            x = c * SQUARE_SIZE + SQUARE_SIZE // 2
            y = r * SQUARE_SIZE + SQUARE_SIZE // 2
            symbol = piece.symbol()
            if piece.color == "white":
                shadow_offset = 2
                shadow_color = "#444444"
                self.canvas.create_text(x + shadow_offset, y + shadow_offset, text=symbol,
                                        font=("Arial", 39), fill=shadow_color, tags="piece")
                self.canvas.create_text(x, y, text=symbol,
                                        font=("Arial Unicode MS", 39), fill="white", tags="piece")
            else:
                self.canvas.create_text(x, y, text=symbol,
                                        font=("Arial", 39), fill="black", tags="piece")

    def on_drag_start(self, event):
        """Handle drag start if it's the player's turn."""
        if self.game_over:
            return
        row, col = self.canvas_to_board(event.x, event.y)
        piece = self.board[row][col]
        if piece is not None and piece.color == self.turn:
            self.dragging = True
            self.drag_start = (row, col)
            self.drag_piece = (event.x, event.y)
            self.valid_moves = piece.get_valid_moves(self.board, (row, col))
            self.draw_board()

    def on_drag_motion(self, event):
        """Update dragged piece position."""
        if self.dragging:
            self.drag_piece = (event.x, event.y)
            self.draw_board()

    def execute_move_and_check_state(self):
        """Update game state after a move and check for game-over conditions."""
        current_key = self.get_position_key()
        self.position_history.append(current_key)
        self.position_counts[current_key] = self.position_counts.get(current_key, 0) + 1
        
        if self.position_counts[current_key] >= 3:
            self.game_over = True
            self.game_result = ("repetition", None)
            self.turn_label.config(text="Draw by three-fold repetition!")
        else:
            result, winner = check_game_over(self.board)
            if result:
                self.game_over = True
                self.game_result = (result, winner)
                if result == "checkmate":
                    self.turn_label.config(text=f"Checkmate! {winner.capitalize()} wins!")
                elif result == "stalemate":
                    self.turn_label.config(text="Stalemate! It's a draw.")
                elif result == "king_capture":
                    self.turn_label.config(text=f"{winner.capitalize()} wins by king capture!")
            else:
                self.turn = "black" if self.turn == "white" else "white"
                self.turn_label.config(text=f"Turn: {self.turn.capitalize()}")

    def on_drag_end(self, event):
        """Handle drag end and make move if valid."""
        if not self.dragging:
            return
        row, col = self.canvas_to_board(event.x, event.y)
        end_pos = (row, col)
        if end_pos in self.valid_moves and validate_move(self.board, self.turn, self.drag_start, end_pos):
            moving_piece = self.board[self.drag_start[0]][self.drag_start[1]]
            self.board = moving_piece.move(self.board, self.drag_start, end_pos)
            check_evaporation(self.board)
            self.execute_move_and_check_state()
            if not self.game_over and self.game_mode.get() == "bot" and self.turn != self.human_color:
                delay = 20 if self.instant_move.get() else 500
                self.master.after(delay, self.make_bot_move)
        else:
            print("Illegal move!")
        self.dragging = False
        self.drag_piece = None
        self.drag_start = None
        self.valid_moves = []
        self.draw_board()

    def make_bot_move(self):
        """Execute bot move in Human vs Bot mode."""
        if self.game_over:
            return
        if self.bot.make_move():
            self.draw_board()
            self.execute_move_and_check_state()
        else:
            self.game_over = True
            self.game_result = ("no_legal_moves", self.human_color)
            self.turn_label.config(text=f"{self.human_color.capitalize()} wins!")

    def process_ai_series_result(self):
        """Process result of an AI vs AI game."""
        self.ai_series_stats['game_count'] += 1
        result, winner = self.game_result
        if result in ["stalemate", "repetition"]:
            self.ai_series_stats['draws'] += 1
            self.turn_label.config(text="Draw, 3-fold repetition!" if result == "repetition" else "Stalemate! It's a draw.")
        elif result in ["checkmate", "king_capture"]:
            winning_bot = "main" if (winner == "white" and self.white_playing_bot == "main") or \
                                (winner == "black" and self.white_playing_bot == "op") else "op"
            if winning_bot == "main":
                self.ai_series_stats['my_ai_wins'] += 1
                self.turn_label.config(text=f"{'Checkmate' if result == 'checkmate' else 'King capture'}! Your AI wins!")
            else:
                self.ai_series_stats['op_ai_wins'] += 1
                self.turn_label.config(text=f"{'Checkmate' if result == 'checkmate' else 'King capture'}! Opponent AI wins!")
        self.update_scoreboard()
        
        self.white_playing_bot = "op" if self.white_playing_bot == "main" else "main"
        self.board_orientation = "black" if self.white_playing_bot == "main" else "white"
        self.update_bot_labels()
        
        if self.ai_series_stats['game_count'] < 100:
            self.master.after(1000, self.reset_game)
        else:
            self.turn_label.config(text="100 Games Complete!")

    def make_ai_move(self):
        """Execute AI move in AI vs AI or Bot mode."""
        if self.game_mode.get() != "ai_vs_ai":
            return
                
        if self.game_over:
            if self.game_mode.get() == "ai_vs_ai":
                self.process_ai_series_result()
            return

        if self.game_mode.get() == "ai_vs_ai":
            current_bot = self.white_bot if self.turn == "white" else self.black_bot
        elif self.game_mode.get() == "bot":
            current_bot = self.bot
        else:
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
                self.turn_label.config(text="Draw by three-fold repetition!")
                if self.game_mode.get() == "ai_vs_ai":
                    self.process_ai_series_result()
            else:
                outcome, winner = check_game_over(self.board)
                if outcome:
                    self.game_over = True
                    self.game_result = (outcome, winner)
                    if self.game_mode.get() == "ai_vs_ai":
                        self.process_ai_series_result()
                    else:
                        if outcome == "checkmate":
                            self.turn_label.config(text=f"Checkmate! {winner.capitalize()} wins!")
                        elif outcome == "stalemate":
                            self.turn_label.config(text="Stalemate! It's a draw.")
                        elif outcome == "king_capture":
                            self.turn_label.config(text=f"{winner.capitalize()} wins by king capture!")
                else:
                    self.turn = "black" if self.turn == "white" else "white"
                    self.turn_label.config(text=f"Turn: {self.turn.capitalize()}")
                    self.master.after(500, self.make_ai_move)
        else:
            print(f"{self.turn} bot did not make a move. Switching turn.")
            self.turn = "black" if self.turn == "white" else "white"
            self.master.after(500, self.make_ai_move)

    def start_ai_series(self):
        """Start AI vs AI series."""
        self.game_mode.set(GameMode.AI_VS_AI.value)
        self.ai_series_stats = {'game_count': 0, 'my_ai_wins': 0, 'op_ai_wins': 0, 'draws': 0}
        self.white_playing_bot = "main"  # Start with main bot as white
        self.board_orientation = "white"  # Initial board orientation
        self.update_scoreboard()
        self.update_bot_labels()
        self.reset_game()

    def update_scoreboard(self):
        """Update AI vs AI scoreboard."""
        self.scoreboard_label.config(
            text=f"AI vs OP Score:\nYour AI: {self.ai_series_stats['my_ai_wins']}\n"
                f"Opponent AI: {self.ai_series_stats['op_ai_wins']}\nDraws: {self.ai_series_stats['draws']}\n"
                f"Games: {self.ai_series_stats['game_count']}/100"
        )

    def update_bot_labels(self):
        """Update bot labels for AI vs AI mode."""
        if self.board_orientation == "white":
            self.bottom_bot_label.config(text="MyAIBot (White)")
            self.top_bot_label.config(text="OpponentAI (Black)")
        else:
            self.bottom_bot_label.config(text="MyAIBot (Black)")
            self.top_bot_label.config(text="OpponentAI (White)")

    def randomize_white_opening(self):
        """Make a random opening move for white in AI vs AI mode."""
        if self.turn != "white":
            return
        possible_moves = []
        for r in range(ROWS):
            for c in range(COLS):
                piece = self.board[r][c]
                if piece and piece.color == "white":
                    moves = piece.get_valid_moves(self.board, (r, c))
                    for move in moves:
                        possible_moves.append(((r, c), move))
        if possible_moves:
            start, end = random.choice(possible_moves)
            if validate_move(self.board, self.turn, start, end):
                moving_piece = self.board[start[0]][start[1]]
                self.board = moving_piece.move(self.board, start, end)
                self.turn = "black"
                self.position_history.append(self.get_position_key())
                print(f"Random opening move by white: {start} to {end}")
                self.draw_board()

    def reset_game(self):
        """Reset game state for a new game."""
        self.board = create_initial_board()
        self.turn = "white"
        self.position_history = []
        self.position_counts = {}
        self.game_result = None

        if self.game_mode.get() == "ai_vs_ai":
            self.randomize_white_opening()
        self.position_history.append(self.get_position_key())

        self.selected = None
        self.valid_moves = []
        self.game_over = False
        self.dragging = False
        self.drag_piece = None
        self.drag_start = None

        if self.game_mode.get() == "ai_vs_ai":
            if self.white_playing_bot == "main":
                self.white_bot = ChessBot(self.board, "white", self)
                self.black_bot = OpponentAI(self.board, "black", self)
            else:
                self.white_bot = OpponentAI(self.board, "white", self)
                self.black_bot = ChessBot(self.board, "black", self)
            self.board_orientation = "white" if self.white_playing_bot == "main" else "black"
            self.turn_label.config(text="AI vs OP: Starting...")
            self.master.after(500, self.make_ai_move)
        elif self.game_mode.get() == "bot":
            bot_color = "black" if self.human_color == "white" else "white"
            self.bot = ChessBot(self.board, bot_color, self)
            self.turn_label.config(text=f"Turn: {self.turn.capitalize()}")
            if self.turn != self.human_color:
                self.master.after(500, self.make_bot_move)
        else:
            self.turn_label.config(text=f"Turn: {self.turn.capitalize()}")
        self.draw_board()
        self.set_interactivity()

    def main(self):
        """Enter Tkinter main event loop."""
        self.master.mainloop()

def main():
    root = tk.Tk()
    app = EnhancedChessApp(root)
    app.main()

if __name__ == "__main__":
    main()