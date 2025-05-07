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
                        style='Custom.TRadiobutton').pack(anchor=tk.W, pady=(3,3)) # Style should be TCheckbutton if available

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
        # self.eval_bar_visible = True # This variable is not used

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
        self.selected = None # Renamed from selected_piece_pos in full refactor for clarity
        self.valid_moves = []
        self.game_over = False
        self.dragging = False
        self.drag_piece = None # Stores canvas xy of mouse during drag
        self.drag_start = None # Stores board (r,c) of drag start
        bot_color = "black" if self.human_color == "white" else "white"
        self.bot = ChessBot(self.board, bot_color, self) # This is for Human vs Bot

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
            # In AI vs AI, swap which bot (main or op) plays white
            self.white_playing_bot = "op" if self.white_playing_bot == "main" else "main" # Assuming you have self.white_playing_bot
            self.board_orientation = "white" if self.white_playing_bot == "main" else "black"
            self.update_bot_labels()
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
        return row, col

    
    def draw_eval_bar(self, eval_score_from_ai):
        """Draw evaluation bar to show board advantage."""
        
        # Assume eval_score_from_ai is in centipawns (e.g., 100 = +1 pawn)
        pawn_equivalent_score = eval_score_from_ai / 100.0

        self.eval_bar_canvas.delete("all")
        bar_width = self.eval_bar_canvas.winfo_width()
        bar_height = self.eval_bar_canvas.winfo_height()

        if bar_width <= 1 or bar_height <= 1:
            self.eval_score_label.config(text="Eval: ...", font=("Helvetica", 10))
            return

        # This scaling factor (20 from your original) is now applied to pawn_equivalent_score.
        # It means the bar would be near full if the advantage is +/- 20 pawns.
        # if you want the bar to be more sensitive to smaller pawn advantages.

        pawn_scaling_for_tanh = 20.0 # Adjust this for sensitivity
        
        normalized_marker_score = math.tanh(pawn_equivalent_score / pawn_scaling_for_tanh)
        normalized_marker_score = max(min(normalized_marker_score, 1.0), -1.0) # Clamp

        # Draw the grayscale gradient background
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

        # Update text label using the pawn_equivalent_score
        if abs(pawn_equivalent_score) < 0.05: 
            self.eval_score_label.config(text="Even", font=("Helvetica", 10))
        else:
            prefix = "+" if pawn_equivalent_score > 0 else ""
            self.eval_score_label.config(text=f"{prefix}{pawn_equivalent_score:.2f}", font=("Helvetica", 10))

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
                  indicatorcolor=[('selected', COLORS['accent'])], # Highlight selected radio
                  foreground=[('active', COLORS['accent'])])
        
        # Add style for TCheckbutton if you used Custom.TRadiobutton for it
        style.configure('Custom.TCheckbutton',
                        background=COLORS['bg_dark'],
                        foreground=COLORS['text_light'],
                        font=('Helvetica', 11),
                        padding=(5, 8))
        style.map('Custom.TCheckbutton',
                  background=[('active', COLORS['bg_dark'])],
                  indicatorcolor=[('selected', COLORS['accent'])],
                  foreground=[('active', COLORS['accent'])])
        return COLORS

    def create_board_image(self, orientation):
        """Create a static board image for the given orientation."""
        board_image = tk.PhotoImage(width=COLS * SQUARE_SIZE, height=ROWS * SQUARE_SIZE)
        for r_logic in range(ROWS):
            for c_logic in range(COLS):
                # Standard chess coloring: a1 (row 7, col 0 if 0-indexed from top-left) is dark.
                is_light_square = (r_logic + c_logic) % 2 == 0 
                color = BOARD_COLOR_1 if is_light_square else BOARD_COLOR_2


                if orientation == "white": # White at bottom, logical (0,0) is top-left
                    x1_canvas, y1_canvas = c_logic * SQUARE_SIZE, r_logic * SQUARE_SIZE
                else: # Black at bottom, logical (0,0) is effectively drawn at bottom-right of canvas
                    x1_canvas = (COLS - 1 - c_logic) * SQUARE_SIZE
                    y1_canvas = (ROWS - 1 - r_logic) * SQUARE_SIZE
                
                x2_canvas = x1_canvas + SQUARE_SIZE
                y2_canvas = y1_canvas + SQUARE_SIZE
                board_image.put(color, (x1_canvas, y1_canvas, x2_canvas, y2_canvas))
        return board_image

    def draw_board(self):
        """Render the chess board using a pre-rendered image and update dynamic elements."""
        if self.board_orientation == "white":
            self.canvas.itemconfig(self.board_image_id, image=self.board_image_white)
        else:
            self.canvas.itemconfig(self.board_image_id, image=self.board_image_black)
        
        self.canvas.delete("highlight", "piece", "drag", "check_highlight") # Add check_highlight
        
        for (r, c) in self.valid_moves:
            x1, y1 = self.board_to_canvas(r, c)
            x2, y2 = x1 + SQUARE_SIZE, y1 + SQUARE_SIZE
            # Draw a small circle/dot for valid moves
            oval_radius = SQUARE_SIZE * 0.15 
            center_x, center_y = x1 + SQUARE_SIZE // 2, y1 + SQUARE_SIZE // 2
            self.canvas.create_oval(center_x - oval_radius, center_y - oval_radius,
                                    center_x + oval_radius, center_y + oval_radius,
                                    fill="#1E90FF", outline="", tags="highlight")
        
        for r in range(ROWS):
            for c in range(COLS):
                piece = self.board[r][c]
                if piece and isinstance(piece, King): # Use from GameLogic
                    if is_in_check(self.board, piece.color): # Use from GameLogic
                        highlight_color = "darkred" if not has_legal_moves(self.board, piece.color) else "red" # Use from GameLogic
                        x1, y1 = self.board_to_canvas(r, c)
                        x2, y2 = x1 + SQUARE_SIZE, y1 + SQUARE_SIZE
                        self.canvas.create_rectangle(x1, y1, x2, y2, outline=highlight_color, 
                                                    width=3, tags="check_highlight") # Use different tag
        
        for r in range(ROWS):
            for c in range(COLS):
                piece = self.board[r][c]
                if piece is not None and (r, c) != self.drag_start:
                    x_cvs, y_cvs = self.board_to_canvas(r, c)
                    x_center = x_cvs + SQUARE_SIZE // 2
                    y_center = y_cvs + SQUARE_SIZE // 2
                    symbol = piece.symbol()
                    font_to_use = ("Arial Unicode MS", 39) # For better unicode symbols
                    font_shadow = ("Arial", 39)

                    if piece.color == "white":
                        shadow_offset = 2
                        shadow_color = "#444444"
                        self.canvas.create_text(x_center + shadow_offset, y_center + shadow_offset,
                                                text=symbol, font=font_shadow,
                                                fill=shadow_color, tags="piece")
                        self.canvas.create_text(x_center, y_center,
                                                text=symbol, font=font_to_use,
                                                fill="white", tags="piece")
                    else:
                        self.canvas.create_text(x_center, y_center, text=symbol,
                                                font=font_to_use, fill="black", tags="piece")
        
        if self.dragging and self.drag_piece and self.drag_start is not None:
            piece_obj = self.board[self.drag_start[0]][self.drag_start[1]] # Get piece object
            if piece_obj is not None:
                font_to_use = ("Arial Unicode MS", 36) # Slightly smaller for drag maybe
                drag_text_color = "white" if piece_obj.color == "white" else "black"
                self.canvas.create_text(self.drag_piece[0], self.drag_piece[1],
                                        text=piece_obj.symbol(), font=font_to_use, 
                                        fill=drag_text_color, tags="drag")
        
        board_width = COLS * SQUARE_SIZE
        board_height = ROWS * SQUARE_SIZE
        self.canvas.create_rectangle(0, 0, board_width, board_height,
                                    outline=self.COLORS['accent'], width=4, tags="border")

    # draw_piece method is largely redundant if draw_board handles everything,
    # but kept for potential future single-piece updates.
    def draw_piece(self, r, c):
        """Draw a single piece unless it's being dragged."""
        piece = self.board[r][c]
        if piece is not None and (r, c) != self.drag_start:
            x_cvs, y_cvs = self.board_to_canvas(r,c) # Use board_to_canvas
            x_center = x_cvs + SQUARE_SIZE // 2
            y_center = y_cvs + SQUARE_SIZE // 2
            symbol = piece.symbol()
            font_to_use = ("Arial Unicode MS", 39)
            font_shadow = ("Arial", 39)
            if piece.color == "white":
                shadow_offset = 2
                shadow_color = "#444444"
                self.canvas.create_text(x_center + shadow_offset, y_center + shadow_offset, text=symbol,
                                        font=font_shadow, fill=shadow_color, tags="piece")
                self.canvas.create_text(x_center, y_center, text=symbol,
                                        font=font_to_use, fill="white", tags="piece")
            else:
                self.canvas.create_text(x_center, y_center, text=symbol,
                                        font=font_to_use, fill="black", tags="piece")


    def on_drag_start(self, event):
        """Handle drag start if it's the player's turn."""
        if self.game_over:
            return
        
        # Determine if it's human's turn for the current mode
        is_human_turn = False
        current_mode = self.game_mode.get()
        if current_mode == GameMode.HUMAN_VS_HUMAN.value:
            is_human_turn = True
        elif current_mode == GameMode.HUMAN_VS_BOT.value and self.turn == self.human_color:
            is_human_turn = True

        if not is_human_turn:
            return

        row, col = self.canvas_to_board(event.x, event.y)
        if not (0 <= row < ROWS and 0 <= col < COLS): return # Click outside board

        piece = self.board[row][col]
        if piece is not None and piece.color == self.turn:
            self.selected = (row, col) # Store selected piece position
            self.dragging = True
            self.drag_start = (row, col)
            self.drag_piece = (event.x, event.y) # Mouse coords
            # Filter valid moves using validate_move
            self.valid_moves = [
                m for m in piece.get_valid_moves(self.board, (row, col))
                if validate_move(self.board, self.turn, (row,col), m)
            ]
            self.draw_board()
        else: # Clicked on empty square or opponent's piece
            self.selected = None
            self.valid_moves = []
            # self.draw_board() # Redraw to clear old highlights if any

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
            # --- MINIMAL CHANGE APPLIED HERE ---
            game_over_status = check_game_over(self.board, self.turn) # Pass self.turn
            result = None
            winner = None

            if isinstance(game_over_status, tuple):
                # This handles ("checkmate", color), ("stalemate", None),
                # and the new ("king_capture", color)
                result, winner = game_over_status
            elif isinstance(game_over_status, str): 
                # This handles the old GameLogic format where king capture returns only winner color string
                result = "king_capture" 
                winner = game_over_status
            # --- END OF MINIMAL CHANGE ---
            
            if result:
                self.game_over = True
                self.game_result = (result, winner)
                if result == "checkmate":
                    self.turn_label.config(text=f"Checkmate! {winner.capitalize()} wins!")
                elif result == "stalemate":
                    self.turn_label.config(text="Stalemate! It's a draw.")
                elif result == "king_capture":
                    self.turn_label.config(text=f"{winner.capitalize()} wins by king capture!")
            else: # No game over, switch turn
                self.turn = "black" if self.turn == "white" else "white"
                self.turn_label.config(text=f"Turn: {self.turn.capitalize()}")
        
        self.set_interactivity() # Update interactivity based on new turn/game over state


    def on_drag_end(self, event):
        """Handle drag end and make move if valid."""
        if not self.dragging and not self.selected: # If not dragging and no piece selected (e.g. clicked empty square)
            self.valid_moves = []
            self.draw_board()
            return

        row, col = self.canvas_to_board(event.x, event.y)
        
        # Handle click-to-move if a piece was selected but not dragged far
        # Or handle drag-and-drop
        if self.selected:
            start_pos = self.selected 
        elif self.drag_start: # Should be true if dragging
            start_pos = self.drag_start
        else: # Should not happen if logic is correct
            self.dragging = False
            self.drag_piece = None
            self.drag_start = None
            self.selected = None
            self.valid_moves = []
            self.draw_board()
            return

        end_pos = (row, col)

        if 0 <= row < ROWS and 0 <= col < COLS and end_pos in self.valid_moves:
            # validate_move was already checked when generating self.valid_moves
            moving_piece = self.board[start_pos[0]][start_pos[1]]
            self.board = moving_piece.move(self.board, start_pos, end_pos)
            check_evaporation(self.board)
            self.execute_move_and_check_state()
            
            if not self.game_over and self.game_mode.get() == GameMode.HUMAN_VS_BOT.value and self.turn != self.human_color:
                delay = 20 if self.instant_move.get() else 500
                self.master.after(delay, self.make_bot_move)
        else:
            # print("Illegal move or click outside valid moves!") # Optional debug
            pass # Just redraw without making a move

        self.dragging = False
        self.drag_piece = None
        self.drag_start = None
        self.selected = None # Clear selection after any attempt
        self.valid_moves = []
        self.draw_board()
        self.set_interactivity() # Re-evaluate who can interact

    def make_bot_move(self):
        """Execute bot move in Human vs Bot mode."""
        # Exit if game over or bot not configured.
        if self.game_over or not self.bot:
            return
        # Bot attempts move; make_move updates the board state on success.
        if self.bot.make_move():
            self.draw_board()
            # Update UI, check game end, switch turn.
            self.execute_move_and_check_state()
        else: # Bot has no legal moves available.
            print(f"Bot ({self.bot.color}) could not make a move.")
            # Note: If bot has no moves and it's not checkmate/stalemate,
            # there might be an issue with move generation or validation.
            # Normal stalemate/checkmate results should be caught by execute_move_and_check_state.


    def process_ai_series_result(self):
        """Process result of an AI vs AI game."""
        if not self.game_result: return # Should not happen if called after game_over

        self.ai_series_stats['game_count'] += 1
        result_type, winner_color = self.game_result

        # Update turn_label with game result for AI vs AI
        if result_type == "repetition":
            self.turn_label.config(text="Draw, 3-fold repetition!")
            self.ai_series_stats['draws'] += 1
        elif result_type == "stalemate":
            self.turn_label.config(text="Stalemate! It's a draw.")
            self.ai_series_stats['draws'] += 1
        elif result_type in ["checkmate", "king_capture"]:
            # Determine which AI won based on self.white_playing_bot
            # (Assuming self.white_playing_bot is 'main' or 'op')
            winning_ai_type = None
            if winner_color == "white":
                winning_ai_type = self.white_playing_bot
            else: # black won
                winning_ai_type = "op" if self.white_playing_bot == "main" else "main"

            if winning_ai_type == "main":
                self.ai_series_stats['my_ai_wins'] += 1
                self.turn_label.config(text=f"{result_type.capitalize()}! Your AI ({winner_color}) wins!")
            else: # 'op' AI won
                self.ai_series_stats['op_ai_wins'] += 1
                self.turn_label.config(text=f"{result_type.capitalize()}! Opponent AI ({winner_color}) wins!")
        
        self.update_scoreboard()
        
        # Swap sides for the next game in the series
        self.white_playing_bot = "op" if self.white_playing_bot == "main" else "main"
        # Board orientation might depend on which AI is white, or a fixed perspective
        # For simplicity, let's assume board_orientation tracks main AI's perspective for AIvAI
        self.board_orientation = "white" if self.white_playing_bot == "main" else "black" 
        self.update_bot_labels()
        
        if self.ai_series_stats['game_count'] < 100: # Max games
            self.master.after(1000, self.reset_game) # Delay before next game
        else:
            self.turn_label.config(text="AI Series: 100 Games Complete!")
            self.ai_series_running = False # Stop the series

    def make_ai_move(self):
        """Execute AI move in AI vs AI mode."""
        # This method is for AI vs AI specifically. Human vs Bot uses make_bot_move.
        if self.game_mode.get() != GameMode.AI_VS_AI.value or self.game_over:
            return

        current_bot = self.ai_white_bot if self.turn == "white" else self.ai_black_bot
        if not current_bot:
            print(f"Error: No bot assigned for {self.turn} in AI vs AI mode.")
            return

        # Bot attempts move; make_move updates the board state on success.
        move_made = current_bot.make_move() 
        self.draw_board()

        if move_made:
            # Standard post-move processing
            current_key = self.get_position_key()
            self.position_history.append(current_key)
            self.position_counts[current_key] = self.position_counts.get(current_key, 0) + 1

            if self.position_counts[current_key] >= 3:
                self.game_over = True
                self.game_result = ("repetition", None)
            else:
                # Check game over state
                game_over_status = check_game_over(self.board, self.turn) # Pass self.turn
                outcome = None
                winner = None

                if isinstance(game_over_status, tuple):
                    outcome, winner = game_over_status
                elif isinstance(game_over_status, str): 
                    outcome = "king_capture"
                    winner = game_over_status

                if outcome:
                    self.game_over = True
                    self.game_result = (outcome, winner)
                else: # No game over, switch turn and continue
                    self.turn = "black" if self.turn == "white" else "white"
                    self.turn_label.config(text=f"Turn: {self.turn.capitalize()} (AI vs AI)") # Simple label update
                    # Schedule next AI move
                    delay = 20 if self.instant_move.get() else 500
                    self.master.after(delay, self.make_ai_move)
        else: # Bot could not make a move
            print(f"AI Bot ({current_bot.color}) could not make a move in AI vs AI.")
            # It's good practice to check game over state here as well
            game_over_status = check_game_over(self.board, self.turn) # Pass self.turn
            # This typically indicates checkmate or stalemate for the bot's side.
            # The game over state is checked and handled by the logic that follows this block.
            # Ensure check_game_over is correctly identifying these scenarios.

        if self.game_over:
            self.process_ai_series_result() # This handles results for AI vs AI series games

        if self.game_over:
            self.process_ai_series_result() # This handles results for AI vs AI series games


    def start_ai_series(self):
        """Start AI vs AI series."""
        self.game_mode.set(GameMode.AI_VS_AI.value)
        self.ai_series_stats = {'game_count': 0, 'my_ai_wins': 0, 'op_ai_wins': 0, 'draws': 0}
        self.ai_series_running = True # Indicate series is active
        self.white_playing_bot = "main"  # 'main' bot (your AI) starts as white
        self.board_orientation = "white" # Orient board for main AI as white
        self.update_scoreboard()
        self.update_bot_labels()
        self.reset_game() # This will configure bots and start the first game


    def update_scoreboard(self):
        """Update AI vs AI scoreboard."""
        # Only show scoreboard if in AI vs AI mode AND series is running
        if self.game_mode.get() == GameMode.AI_VS_AI.value and self.ai_series_running:
            self.scoreboard_label.config(
                text=f"AI vs OP Score (Game {self.ai_series_stats['game_count']}/100):\n"
                     f"  Your AI: {self.ai_series_stats['my_ai_wins']}\n"
                     f"  Opponent AI: {self.ai_series_stats['op_ai_wins']}\n"
                     f"  Draws: {self.ai_series_stats['draws']}"
            )
            self.scoreboard_frame.place(relx=1.0, rely=0.0, anchor='ne', x=-15, y=15) # Ensure it's visible
        else:
            self.scoreboard_frame.place_forget() # Hide it


    def update_bot_labels(self):
        """Update bot labels for AI vs AI mode."""
        # This logic needs to know which AI is playing which color *for the current game*
        # and how the board is oriented.
        top_text = ""
        bottom_text = ""

        current_mode = self.game_mode.get()

        if current_mode == GameMode.HUMAN_VS_BOT.value:
            if self.board_orientation == "white": # Human plays white, white at bottom
                bottom_text = f"Human ({self.human_color.capitalize()})"
                top_text = f"Bot ({('black' if self.human_color == 'white' else 'white').capitalize()})"
            else: # Human plays black, black at bottom
                bottom_text = f"Human ({self.human_color.capitalize()})"
                top_text = f"Bot ({('white' if self.human_color == 'black' else 'black').capitalize()})"

        elif current_mode == GameMode.AI_VS_AI.value:
            # self.white_playing_bot determines which AI *type* is white ('main' or 'op')
            main_ai_name = "MyAIBot"
            op_ai_name = "OpponentAI"
            
            white_player_for_this_game = main_ai_name if self.white_playing_bot == "main" else op_ai_name
            black_player_for_this_game = op_ai_name if self.white_playing_bot == "main" else main_ai_name

            if self.board_orientation == "white": # White pieces at bottom
                bottom_text = f"{white_player_for_this_game} (White)"
                top_text = f"{black_player_for_this_game} (Black)"
            else: # Black pieces at bottom
                bottom_text = f"{black_player_for_this_game} (Black)"
                top_text = f"{white_player_for_this_game} (White)"
        
        elif current_mode == GameMode.HUMAN_VS_HUMAN.value:
            if self.board_orientation == "white":
                bottom_text = "White"
                top_text = "Black"
            else:
                bottom_text = "Black"
                top_text = "White"


        self.top_bot_label.config(text=top_text)
        self.bottom_bot_label.config(text=bottom_text)


    def randomize_white_opening(self):
        """Make a random opening move for white in AI vs AI mode."""
        if self.turn != "white": # Only if it's white's turn to make opening
            return
        
        possible_moves = []
        for r in range(ROWS):
            for c in range(COLS):
                piece = self.board[r][c]
                if piece and piece.color == "white":
                    # Get valid moves and ensure they are truly legal
                    piece_moves = piece.get_valid_moves(self.board, (r, c))
                    for move_end_pos in piece_moves:
                        if validate_move(self.board, "white", (r,c), move_end_pos):
                            possible_moves.append(((r, c), move_end_pos))
        
        if possible_moves:
            start, end = random.choice(possible_moves)
            moving_piece = self.board[start[0]][start[1]]
            self.board = moving_piece.move(self.board, start, end)
            check_evaporation(self.board) # Check after move
            
            # This move is made, now it's black's turn
            self.turn = "black" 
            
            # Update history for the state black will see
            self.position_history.append(self.get_position_key()) 
            self.position_counts[self.get_position_key()] = self.position_counts.get(self.get_position_key(), 0) + 1

            print(f"Random opening move by white: {start} to {end}")
            self.draw_board() # Redraw board after random move
        else:
            print("Warning: No valid random opening moves found for White.")


    def reset_game(self):
        """Reset game state for a new game."""
        self.board = create_initial_board()
        self.turn = "white" # White always starts (random opening handles AIvAI first move)
        self.selected = None
        self.valid_moves = []
        self.game_over = False
        self.game_result = None
        self.dragging = False
        self.drag_piece = None
        self.drag_start = None
        
        self.position_history = [] # Clear history
        self.position_counts = {}  # Clear counts
        # Add initial position to history AFTER board is set and turn is white
        self.position_history.append(self.get_position_key())
        self.position_counts[self.get_position_key()] = 1


        current_mode = self.game_mode.get()

        if current_mode == GameMode.AI_VS_AI.value:
            # Configure AI bots for AI vs AI mode based on self.white_playing_bot
            if self.white_playing_bot == "main":
                self.ai_white_bot = ChessBot(self.board, "white", self)
                self.ai_black_bot = OpponentAI(self.board, "black", self)
            else: # Opponent AI plays white
                self.ai_white_bot = OpponentAI(self.board, "white", self)
                self.ai_black_bot = ChessBot(self.board, "black", self)
            
            self.update_bot_depth(self.bot_depth_slider.get()) # Ensure new bots get current depth

            if self.ai_series_running: # Only randomize if series is running
                 self.randomize_white_opening() # This sets turn to black if successful
            
            self.turn_label.config(text=f"Turn: {self.turn.capitalize()} (AI vs AI)")
            # Schedule the first AI move (could be white or black after random opening)
            delay = 20 if self.instant_move.get() else 500
            self.master.after(delay, self.make_ai_move)

        elif current_mode == GameMode.HUMAN_VS_BOT.value:
            bot_color = "black" if self.human_color == "white" else "white"
            self.bot = ChessBot(self.board, bot_color, self)
            self.update_bot_depth(self.bot_depth_slider.get()) # Ensure bot gets current depth
            self.turn_label.config(text=f"Turn: {self.turn.capitalize()}")
            if self.turn != self.human_color: # If it's bot's turn to start
                delay = 20 if self.instant_move.get() else 500
                self.master.after(delay, self.make_bot_move)
        else: # Human vs Human
            self.turn_label.config(text=f"Turn: {self.turn.capitalize()}")

        self.draw_board()
        self.set_interactivity()
        self.update_bot_labels() # Update labels based on new config
        self.update_scoreboard() # Show/hide scoreboard


    def main(self): # Renamed from run for consistency if you have app.run() elsewhere
        """Enter Tkinter main event loop."""
        self.master.mainloop()

def main_app(): # Renamed from main to avoid conflict if GameLogic.py also has main()
    root = tk.Tk()
    app = EnhancedChessApp(root)
    app.main() # Call the instance's main loop method

if __name__ == "__main__":
    main_app()