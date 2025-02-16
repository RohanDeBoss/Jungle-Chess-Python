import tkinter as tk
from tkinter import ttk
import time
import math
from GameLogic import create_initial_board, ROWS, COLS, SQUARE_SIZE, BOARD_COLOR_1, BOARD_COLOR_2, generate_position_key, King, is_in_check, has_legal_moves, validate_move, check_game_over, check_evaporation
from AI import ChessBot


class EnhancedChessApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Enhanced Chess")
        self.COLORS = self.setup_styles()
        self.master.configure(bg=self.COLORS['bg_dark'])
        self.position_history = []  # Track positions for threefold repetition

        # Window setup
        screen_w = self.master.winfo_screenwidth()
        screen_h = self.master.winfo_screenheight()
        self.master.geometry(f"{screen_w}x{screen_h}+0+0")
        self.master.state('zoomed')
        self.fullscreen = True
        self.master.bind("<Configure>", self.on_configure)
        
        # Main frame setup
        self.main_frame = ttk.Frame(master, style='Left.TFrame')
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Left panel (sidebar)
        self.left_panel = ttk.Frame(self.main_frame, width=250, style='Left.TFrame')
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=15, pady=(0,15))
        self.left_panel.pack_propagate(False)
        
        # Header label
        ttk.Label(self.left_panel, text="JUNGLE CHESS", style='Header.TLabel',
                font=('Helvetica', 24, 'bold')).pack(pady=(0,10))
        
        # Game mode frame setup
        self.game_mode = tk.StringVar(value="bot")
        game_mode_frame = ttk.Frame(self.left_panel, style='Left.TFrame')
        game_mode_frame.pack(fill=tk.X, pady=(0,9))
        ttk.Label(game_mode_frame, text="GAME MODE", style='Header.TLabel').pack(anchor=tk.W)
        ttk.Radiobutton(game_mode_frame, text="Human vs Bot", variable=self.game_mode,
                        value="bot", command=self.reset_game, style='Custom.TRadiobutton').pack(anchor=tk.W, pady=(5,3))
        ttk.Radiobutton(game_mode_frame, text="Human vs Human", variable=self.game_mode,
                        value="human", command=self.reset_game, style='Custom.TRadiobutton').pack(anchor=tk.W)
        
       # Controls frame setup
        controls_frame = ttk.Frame(self.left_panel, style='Left.TFrame')
        controls_frame.pack(fill=tk.X, pady=4)
        ttk.Button(controls_frame, text="NEW GAME", command=self.reset_game,
                   style='Control.TButton').pack(fill=tk.X, pady=5)
        # Remove the BOT SETTINGS button
        ttk.Button(controls_frame, text="SWAP SIDES", command=self.swap_sides,
                   style='Control.TButton').pack(fill=tk.X, pady=5)
        ttk.Button(controls_frame, text="QUIT", command=self.master.quit,
                   style='Control.TButton').pack(fill=tk.X, pady=5)
        # Inline Bot settings with a Bot Depth slider
        ttk.Label(controls_frame, text="Bot Depth:", style='Header.TLabel').pack(anchor=tk.W, pady=(10,0))
        self.bot_depth_slider = tk.Scale(controls_frame, from_=1, to=6, orient=tk.HORIZONTAL,
                                         command=self.update_bot_depth,
                                         bg=self.COLORS['bg_dark'], fg=self.COLORS['text_light'],
                                         highlightthickness=0)
        self.bot_depth_slider.set(ChessBot.search_depth)
        self.bot_depth_slider.pack(fill=tk.X, pady=(0,4))
        
        # Add an Instant Move checkmark
        self.instant_move = tk.BooleanVar(value=False)
        ttk.Checkbutton(controls_frame, text="Instant Move", variable=self.instant_move,
                        style='Custom.TRadiobutton').pack(anchor=tk.W, pady=(3,3))
        
        # Turn display frame
        self.turn_frame = ttk.Frame(self.left_panel, style='Left.TFrame')
        self.turn_frame.pack(fill=tk.X, pady=(9,0))
        self.turn_label = ttk.Label(self.turn_frame, text="WHITE'S TURN", style='Status.TLabel')
        self.turn_label.pack(fill=tk.X)
        
        # Updated Evaluation frame setup
        self.eval_frame = ttk.Frame(self.left_panel, style='Left.TFrame')
        # Remove horizontal padding so it matches the other elements
        self.eval_frame.pack(side=tk.TOP, fill=tk.X, padx=0, pady=(5,5))

        self.eval_score_label = ttk.Label(self.eval_frame, text="Even", style='Status.TLabel', anchor="center")
        # Pack without extra fill or vertical padding
        self.eval_score_label.pack(pady=(7,5))

        # The eval canvas remains the same:
        self.eval_bar_canvas = tk.Canvas(self.eval_frame, height=26,
                                        bg=self.COLORS['bg_light'], highlightthickness=0)
        self.eval_bar_canvas.pack(side=tk.BOTTOM, fill=tk.X, expand=True)
        self.eval_bar_canvas.bind("<Configure>", lambda event: self.draw_eval_bar(0))
        self.draw_eval_bar(0)
        self.eval_bar_visible = True
        # Right panel (main game board)
        self.right_panel = ttk.Frame(self.main_frame, style='Right.TFrame')
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=15, pady=15)
        self.canvas_container = ttk.Frame(self.right_panel, style='Canvas.TFrame')
        self.canvas_container.pack(expand=True)
        self.canvas_container.grid_rowconfigure(0, weight=1)
        self.canvas_container.grid_columnconfigure(0, weight=1)
        self.canvas_frame = ttk.Frame(self.canvas_container, style='Canvas.TFrame')
        self.canvas_frame.grid(row=0, column=0)
        # In the __init__ method of EnhancedChessApp, change the canvas creation:
        self.canvas = tk.Canvas(self.canvas_frame,
                                width=COLS * SQUARE_SIZE,
                                height=ROWS * SQUARE_SIZE,
                                bg=self.COLORS['bg_light'],
                                highlightthickness=0)  # Remove built-in highlight border
        self.canvas.pack()

        # Initial game state initialization
        self.human_color = "white"
        self.board = create_initial_board()
        self.turn = "white"
        self.selected = None
        self.valid_moves = []
        self.game_over = False
        self.dragging = False
        self.drag_piece = None
        self.drag_start = None
        bot_color = "black"  # Opponent always plays opposite
        self.bot = ChessBot(self.board, bot_color, self)
        
        # Bind canvas events and initial drawing
        self.canvas.bind("<Button-1>", self.on_drag_start)
        self.canvas.bind("<B1-Motion>", self.on_drag_motion)
        self.canvas.bind("<ButtonRelease-1>", self.on_drag_end)
        self.draw_board()

    def update_bot_depth(self, value):
        new_depth = int(value)
        ChessBot.search_depth = new_depth
        self.bot.search_depth = new_depth
    
    # Update EnhancedChessApp's get_position_key method
    def get_position_key(self):
        return generate_position_key(self.board, self.turn)

    def swap_sides(self):
        # Swap which color the human plays. After swapping, the board is redrawn from the new perspective.
        self.human_color = "black" if self.human_color == "white" else "white"
        # Force human to play the new color; bot gets the opposite.
        bot_color = "black" if self.human_color == "white" else "white"
        self.bot = ChessBot(self.board, bot_color, self)
        self.turn = self.human_color  # Let human start.
        self.turn_label.config(text=f"Turn: {self.human_color.capitalize()}")
        self.draw_board()

    # Coordinate conversion helpers:
    def board_to_canvas(self, r, c):
        # For a 180° rotated view when playing as black.
        if self.human_color == "white":
            x1 = c * SQUARE_SIZE
            y1 = r * SQUARE_SIZE
        else:
            x1 = (COLS - 1 - c) * SQUARE_SIZE
            y1 = (ROWS - 1 - r) * SQUARE_SIZE
        return x1, y1

    def canvas_to_board(self, x, y):
        # Convert canvas coordinates to board indices based on current perspective.
        if self.human_color == "white":
            row = y // SQUARE_SIZE
            col = x // SQUARE_SIZE
        else:
            row = (ROWS - 1) - (y // SQUARE_SIZE)
            col = (COLS - 1) - (x // SQUARE_SIZE)
        return row, col

    def on_configure(self, event):
        pass

    def open_settings(self):
        settings_win = tk.Toplevel(self.master)
        settings_win.title("Bot Settings")
        win_width, win_height = 300, 150
        screen_width = settings_win.winfo_screenwidth()
        screen_height = settings_win.winfo_screenheight()
        x = (screen_width - win_width) // 2
        y = (screen_height - win_height) // 2
        settings_win.geometry(f"{win_width}x{win_height}+{x}+{y}")
        ttk.Label(settings_win, text="Bot Search Depth:", font=('Helvetica', 12)).pack(pady=(20, 5))
        depth_var = tk.IntVar(value=ChessBot.search_depth)
        spin = ttk.Spinbox(settings_win, from_=1, to=6, textvariable=depth_var, width=5)
        spin.pack(pady=(0, 20))
        eval_bar_var = tk.BooleanVar(value=self.eval_bar_visible)
        ttk.Checkbutton(settings_win, text="Show Evaluation Bar", variable=eval_bar_var).pack(pady=(0, 10))
        
        def apply_settings():
            new_depth = depth_var.get()
            ChessBot.search_depth = new_depth
            if hasattr(self, 'bot'):
                self.bot.search_depth = new_depth
            self.eval_bar_visible = eval_bar_var.get()
            if self.eval_bar_visible:
                self.eval_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=15, pady=15)
            else:
                self.eval_frame.pack_forget()
            settings_win.destroy()
        
        ttk.Button(settings_win, text="Apply", command=apply_settings).pack()

    def draw_eval_bar(self, eval_score):
        eval_score /= 100.0
        self.eval_bar_canvas.delete("all")
        bar_width = self.eval_bar_canvas.winfo_width() or 235
        bar_height = 30
        max_eval = 10.0
        # Picking a scaling factor such that tanh(62/scaling) is nearly 1.
        # For example, tanh(62/23.4) ~ 0.99.
        scaling = 23.4
        normalized_score = math.tanh(eval_score / scaling)
        # Clamp to the interval [-1, 1]
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
        marker_width = 1  # Reduced marker outline thickness
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
        # Reduced padding makes the label box slimmer
        style.configure('Status.TLabel',
                        background=COLORS['bg_light'],
                        foreground=COLORS['text_light'],
                        font=('Helvetica', 16, 'bold'),
                        padding=(11, 4),   # Reduced from (18, 10)
                        relief='flat',
                        borderwidth=0)
        # Adjusted button style: reduced padding for a skinnier red button look.
        style.configure('Control.TButton',
                        background=COLORS['accent'],
                        foreground=COLORS['text_light'],
                        font=('Helvetica', 11, 'bold'),
                        padding=(10, 8),   # Reduced padding from (15, 12)
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

    def toggle_fullscreen(self, event=None):
        self.fullscreen = not self.fullscreen
        self.master.attributes('-fullscreen', self.fullscreen)
        if not self.fullscreen:
            self.master.geometry("800x600")

    def draw_board(self):
        self.canvas.delete("all")
        # Draw squares
        for r in range(ROWS):
            for c in range(COLS):
                x1, y1 = self.board_to_canvas(r, c)
                x2, y2 = x1 + SQUARE_SIZE, y1 + SQUARE_SIZE
                color = BOARD_COLOR_1 if (r + c) % 2 == 0 else BOARD_COLOR_2
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")
                # highlight valid moves if needed
                if (r, c) in self.valid_moves:
                    self.canvas.create_oval(x1 + 19, y1 + 19, x2 - 19, y2 - 19,
                                            fill="#1E90FF", outline="#1E90FF", width=3)

        # Highlight kings if in check or checkmated
        for r in range(ROWS):
            for c in range(COLS):
                piece = self.board[r][c]
                if piece and isinstance(piece, King):
                    if is_in_check(self.board, piece.color):
                        highlight_color = "darkred" if not has_legal_moves(self.board, piece.color) else "red"
                        x1, y1 = self.board_to_canvas(r, c)
                        x2, y2 = x1 + SQUARE_SIZE, y1 + SQUARE_SIZE
                        self.canvas.create_rectangle(x1, y1, x2, y2, outline=highlight_color, width=3)

        # Draw pieces (existing code)
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

        # Draw dragging piece if any
        if self.dragging and self.drag_piece and self.drag_start is not None:
            piece = self.board[self.drag_start[0]][self.drag_start[1]]
            if piece is not None:
                self.canvas.create_text(self.drag_piece[0], self.drag_piece[1],
                                        text=piece.symbol(), font=("Arial", 36), tags="drag")
        
        # Draw an even border around the chess board
        board_width = COLS * SQUARE_SIZE
        board_height = ROWS * SQUARE_SIZE
        self.canvas.create_rectangle(0, 0, board_width, board_height, outline=self.COLORS['accent'], width=4)

    def draw_piece(self, r, c):
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
                                        font=("Arial Unicode MS", 39),
                                        fill="white", tags="piece")
            else:
                self.canvas.create_text(x, y, text=symbol,
                                        font=("Arial", 39),
                                        fill="black", tags="piece")

    def on_drag_start(self, event):
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
        if self.dragging:
            self.drag_piece = (event.x, event.y)
            self.draw_board()

    def on_drag_end(self, event):
        if not self.dragging:
            return

        row, col = self.canvas_to_board(event.x, event.y)
        end_pos = (row, col)

        if end_pos in self.valid_moves:
            if validate_move(self.board, self.turn, self.drag_start, end_pos):
                moving_piece = self.board[self.drag_start[0]][self.drag_start[1]]
                self.board = moving_piece.move(self.board, self.drag_start, end_pos)
                check_evaporation(self.board)
                
                # Redraw board and force an update so the capture animation completes
                self.draw_board()
                self.master.update_idletasks()  # Ensure canvas refresh

                # Check for game over conditions
                result, winner = check_game_over(self.board)
                if result == "checkmate":
                    self.game_over = True
                    self.turn_label.config(text=f"Checkmate! {winner.capitalize()} wins!")
                elif result == "stalemate":
                    self.game_over = True
                    self.turn_label.config(text="Stalemate! It's a draw.")
                else:
                    # Switch turns and record position history
                    self.turn = "black" if self.turn == "white" else "white"
                    self.position_history.append(self.get_position_key())
                    
                    # Use a minimal delay (20ms) to let board animation finalize if instant move is on
                    if self.game_mode.get() == "bot" and self.turn != self.human_color:
                        delay = 20 if self.instant_move.get() else 500
                        self.master.after(delay, self.make_bot_move)
            else:
                print("Illegal move!")
        # Reset dragging state
        self.dragging = False
        self.drag_piece = None
        self.drag_start = None
        self.valid_moves = []
        self.draw_board()

    def make_bot_move(self):
        if self.game_over:
            return

        # Start timing for debug output
        start_time = time.time()

        # Make the bot's move
        if self.bot.make_move():
            # Update the board display immediately
            self.draw_board()

            # Check for game over conditions
            result, winner = check_game_over(self.board)
            if result == "checkmate":
                self.game_over = True
                self.turn_label.config(text=f"Checkmate! {winner.capitalize()} wins!")
            elif result == "stalemate":
                self.game_over = True
                self.turn_label.config(text="Stalemate! It's a draw.")
            else:
                # Switch turns back to the human player
                self.turn = self.human_color
                # Update position history after switching turns
                self.position_history.append(self.get_position_key())

        else:
            # If the bot cannot make a move, the human wins
            self.game_over = True
            self.turn_label.config(text=f"{self.human_color.capitalize()} wins!")    
    
    def swap_sides(self):
        self.human_color = "black" if self.human_color == "white" else "white"
        self.reset_game()

    def reset_game(self):
        self.board = create_initial_board()
        # Always start with white's turn.
        self.turn = "white"  
        self.selected = None
        self.valid_moves = []
        self.game_over = False
        self.dragging = False
        self.drag_piece = None
        self.drag_start = None
        # Bot takes the color that is not chosen by the human.
        bot_color = "black" if self.human_color == "white" else "white"
        self.bot = ChessBot(self.board, bot_color, self)
        self.turn_label.config(text=f"Turn: {self.turn.capitalize()}")
        self.draw_board()
        # If playing in bot mode and it's not the human's turn, let the bot move.
        if self.game_mode.get() == "bot" and self.turn != self.human_color:
            self.master.after(500, self.make_bot_move)

def main():
    root = tk.Tk()
    app = EnhancedChessApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()