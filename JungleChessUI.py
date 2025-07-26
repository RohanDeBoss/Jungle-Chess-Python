# JungleChessUI.py (v7.1 - Final Logging Polish)
# - Relocated the "Stopping process..." log message to make the console output
#   for analysis restarts more linear and intuitive.
# - The core game logic is now stable and correct across all modes.

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

# --- UI Constants ---
SQUARE_SIZE = 75

class GameMode(Enum):
    HUMAN_VS_BOT = "bot"
    HUMAN_VS_HUMAN = "human"
    AI_VS_AI = "ai_vs_ai"

# --- MULTIPROCESSING WORKER FUNCTION ---
def run_ai_process(board, color, position_counts, comm_queue, cancellation_event, bot_class, bot_name, search_depth):
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
    
    def __init__(self, master):
        self.master = master
        self.master.title("Jungle Chess")

        random.seed()
        
        self.comm_queue = mp.Queue()
        self.ai_process = None
        self.ai_cancellation_event = mp.Event()
        self.ai_search_start_time = None

        self.board = create_initial_board()
        self.turn = "white"
        self.selected = None
        self.valid_moves = []
        self.game_over = False
        self.game_result = None
        self.dragging = False
        self.drag_piece_ghost = None
        self.drag_start = None
        self.position_history = []
        self.position_counts = {}
        self.current_opening_move = None

        self.game_mode = tk.StringVar(value=GameMode.HUMAN_VS_BOT.value)
        self.analysis_mode_var = tk.BooleanVar(value=True) # Default to true on startup
        self.ai_series_running = False
        self.ai_series_stats = {'game_count': 0, 'my_ai_wins': 0, 'op_ai_wins': 0, 'draws': 0}
        
        self.white_playing_bot_type = "main"
        self.human_color = "white"
        self.board_orientation = "white"
        self.last_move_timestamp = None
        self.game_started = False 

        self.COLORS = self.setup_styles()
        self.master.configure(bg=self.COLORS['bg_dark'])
        self.build_ui()
        
        self.process_comm_queue()
        
        # Initialize board state without calling reset_game
        self.board = create_initial_board()
        self.turn = "white"
        self.last_move_timestamp = time.time()
        self.selected, self.valid_moves, self.game_over, self.game_result = None, [], False, None
        self.position_history = [board_hash(self.board, self.turn)]
        self.position_counts = {self.position_history[0]: 1}
        self.draw_eval_bar(0)
        self.update_turn_label()
        self.draw_board()
        self.set_interactivity(True)
        self.update_bot_labels()

    def build_ui(self):
        screen_w = self.master.winfo_screenwidth(); screen_h = self.master.winfo_screenheight()
        self.master.geometry(f"{screen_w}x{screen_h}+0+0"); self.master.state('zoomed')
        self.main_frame = ttk.Frame(self.master, style='Left.TFrame')
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.left_panel = ttk.Frame(self.main_frame, width=280, style='Left.TFrame')
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=15, pady=(0,15)); self.left_panel.pack_propagate(False)
        top_controls_frame = ttk.Frame(self.left_panel, style='Left.TFrame')
        top_controls_frame.pack(side=tk.TOP, fill=tk.X, expand=False)
        bottom_status_frame = ttk.Frame(self.left_panel, style='Left.TFrame')
        bottom_status_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(1, 0))
        ttk.Label(top_controls_frame, text="JUNGLE CHESS", style='Header.TLabel', font=('Helvetica', 24, 'bold')).pack(pady=(0,10))
        self._build_control_widgets(top_controls_frame)
        self._build_status_widgets(bottom_status_frame)
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

    def _build_control_widgets(self, parent_frame):
        game_mode_frame = ttk.Frame(parent_frame, style='Left.TFrame')
        game_mode_frame.pack(fill=tk.X, pady=(0,8))
        ttk.Label(game_mode_frame, text="GAME MODE", style='Header.TLabel').pack(anchor=tk.W)
        for mode in GameMode:
            text = mode.name.replace("_", " ").title()
            ttk.Radiobutton(game_mode_frame, text=text, variable=self.game_mode, value=mode.value, command=self.reset_game, style='Custom.TRadiobutton').pack(anchor=tk.W, pady=(2,0))
        controls_frame = ttk.Frame(parent_frame, style='Left.TFrame')
        controls_frame.pack(fill=tk.X, pady=3)
        ttk.Button(controls_frame, text="NEW GAME", command=self.reset_game, style='Control.TButton').pack(fill=tk.X, pady=3)
        ttk.Button(controls_frame, text="SWAP SIDES", command=self.swap_sides, style='Control.TButton').pack(fill=tk.X, pady=3)
        ttk.Button(controls_frame, text="AI vs OP Series", command=self.start_ai_series, style='Control.TButton').pack(fill=tk.X, pady=3)
        ttk.Button(controls_frame, text="QUIT", command=self.master.quit, style='Control.TButton').pack(fill=tk.X, pady=3)
        ttk.Label(controls_frame, text="Bot Depth:", style='SmallHeader.TLabel').pack(anchor=tk.W, pady=(8,0))
        self.bot_depth_slider = tk.Scale(controls_frame, from_=1, to=6, orient=tk.HORIZONTAL, bg=self.COLORS['bg_dark'], fg=self.COLORS['text_light'], highlightthickness=0, relief='flat')
        self.bot_depth_slider.set(3); self.bot_depth_slider.pack(fill=tk.X, pady=(0,2))
        self.instant_move = tk.BooleanVar(value=False); ttk.Checkbutton(controls_frame, text="Instant Moves", variable=self.instant_move, style='Custom.TCheckbutton').pack(anchor=tk.W, pady=(2,2))
        self.analysis_checkbox = ttk.Checkbutton(controls_frame, text="Analysis Mode (H-vs-H)", variable=self.analysis_mode_var, style='Custom.TCheckbutton', command=self.toggle_analysis_mode)
        self.analysis_checkbox.pack(anchor=tk.W, pady=(2,2))

    def _build_status_widgets(self, parent_frame):
        self.turn_frame = ttk.Frame(parent_frame, style='Left.TFrame'); self.turn_frame.pack(fill=tk.X, pady=(8,0))
        self.turn_label = ttk.Label(self.turn_frame, text="WHITE'S TURN", style='Status.TLabel'); self.turn_label.pack(fill=tk.X)
        self.eval_frame = ttk.Frame(parent_frame, style='Left.TFrame'); self.eval_frame.pack(fill=tk.X, pady=(9, 4))
        self.eval_score_label = ttk.Label(self.eval_frame, text="Even", style='Status.TLabel', anchor="center"); self.eval_score_label.pack(pady=(6,4))
        self.eval_bar_canvas = tk.Canvas(self.eval_frame, height=26, bg=self.COLORS['bg_light'], highlightthickness=0); self.eval_bar_canvas.pack(fill=tk.X, expand=True)
        self.eval_bar_canvas.bind("<Configure>", lambda event: self.draw_eval_bar(0))

    def _build_scoreboard_and_labels(self):
        self.scoreboard_frame = ttk.Frame(self.right_panel, style='Right.TFrame')
        self.scoreboard_label = ttk.Label(self.scoreboard_frame, text="", font=("Helvetica", 10), justify=tk.LEFT, background=self.COLORS['bg_medium'], foreground=self.COLORS['text_light']); self.scoreboard_label.pack()
        self.top_bot_label = ttk.Label(self.right_panel, text="", font=("Helvetica", 12), background=self.COLORS['bg_medium'], foreground=self.COLORS['text_light']); self.top_bot_label.place(relx=0.5, rely=0.02, anchor='n')
        self.bottom_bot_label = ttk.Label(self.right_panel, text="", font=("Helvetica", 12), background=self.COLORS['bg_medium'], foreground=self.COLORS['text_light']); self.bottom_bot_label.place(relx=0.5, rely=0.98, anchor='s')

    def toggle_analysis_mode(self):
        if self.game_mode.get() == GameMode.HUMAN_VS_HUMAN.value:
            if self.analysis_mode_var.get():
                self._restart_analysis_process()
            else:
                self._stop_ai_process()

    def process_comm_queue(self):
        try:
            while not self.comm_queue.empty():
                message = self.comm_queue.get_nowait()
                msg_type, *payload = message
                if msg_type == 'log': print(payload[0])
                elif msg_type == 'eval': self.draw_eval_bar(payload[0], payload[1])
                elif msg_type == 'move': self._execute_ai_move(payload[0])
        except Exception: pass
        finally: self.master.after(100, self.process_comm_queue)

    def _execute_ai_move(self, the_move):
        if self.game_over: return

        if the_move:
            # The AI now prints its own timing logs, so we remove the one here.
            self.board.make_move(the_move[0], the_move[1])
            self.execute_move_and_check_state()
            if not self.game_over and self.game_mode.get() == GameMode.AI_VS_AI.value:
                self.master.after(20, self._make_game_ai_move)
        else:
            print("AI reported no valid move was made or was cancelled.")
        
        self.update_bot_labels(); self.set_interactivity(True)

    def _start_ai_process(self, bot_class, bot_name, search_depth):
        if self.ai_process and self.ai_process.is_alive(): return
        
        self.ai_cancellation_event.clear()
        args = (self.board.clone(), self.turn, self.position_counts.copy(), self.comm_queue, self.ai_cancellation_event, bot_class, bot_name, search_depth)
        self.ai_process = mp.Process(target=run_ai_process, args=args, daemon=True)
        self.ai_process.name = bot_name 
        self.ai_process.start()
        
        if bot_name != self.ANALYSIS_AI_NAME: self.set_interactivity(False)
        self.update_bot_labels()

    def _stop_ai_process(self):
        if self.ai_process and self.ai_process.is_alive():
            self.ai_cancellation_event.set()
            self.ai_process.join(timeout=0.5)
            if self.ai_process.is_alive(): self.ai_process.terminate()
            self.ai_process = None
        while not self.comm_queue.empty():
            try: self.comm_queue.get_nowait()
            except Exception: break
        self.set_interactivity(True); self.update_bot_labels()
        if self.game_mode.get() == GameMode.HUMAN_VS_HUMAN.value and not self.analysis_mode_var.get():
             self.draw_eval_bar(0); self.eval_score_label.config(text="Even")

    def reset_game(self):
        if self.game_mode.get() != GameMode.AI_VS_AI.value: self.ai_series_running = False

        self._stop_ai_process()
        self.board = create_initial_board()
        self.turn = "white"
        self.game_started = False  # Reset game_started before showing the new game message
        self._start_game_if_needed()
        self.last_move_timestamp = time.time()
        self.selected, self.valid_moves, self.game_over, self.game_result = None, [], False, None
        self.position_history = [board_hash(self.board, self.turn)]; self.position_counts = {self.position_history[0]: 1}
        self.draw_eval_bar(0)

        mode = self.game_mode.get()
        if mode == GameMode.AI_VS_AI.value:
            self.white_playing_bot_type = "op" if self.ai_series_running and self.ai_series_stats['game_count'] % 2 == 1 else "main"
            self.board_orientation = "white" if self.white_playing_bot_type == "main" else "black"
            if self.ai_series_running: self.apply_series_opening_move()
            if not self.game_over: self.master.after(20, self._make_game_ai_move)
        elif mode == GameMode.HUMAN_VS_BOT.value:
            self.board_orientation = self.human_color
            if self.turn != self.human_color: self.master.after(20, self._make_game_ai_move)
        else:
            self.board_orientation = "white"
        
        self.update_turn_label(); self.draw_board(); self.set_interactivity(True)
        self.update_bot_labels(); self.update_scoreboard()
        self._start_analysis_if_needed()

    def _make_game_ai_move(self):
        if self.game_over: return
        self._start_game_if_needed()
        self.ai_search_start_time = time.time()

        mode = self.game_mode.get()
        bot_class, bot_name = None, None
        if mode == GameMode.HUMAN_VS_BOT.value:
            if self.turn != self.human_color: bot_class, bot_name = ChessBot, self.MAIN_AI_NAME
        elif mode == GameMode.AI_VS_AI.value:
            if self.turn == 'white':
                bot_class, bot_name = (ChessBot, self.MAIN_AI_NAME) if self.white_playing_bot_type == 'main' else (OpponentAI, self.OPPONENT_AI_NAME)
            else:
                bot_class, bot_name = (OpponentAI, self.OPPONENT_AI_NAME) if self.white_playing_bot_type == 'main' else (ChessBot, self.MAIN_AI_NAME)
        if bot_class:
            self._start_ai_process(bot_class, bot_name, self.bot_depth_slider.get())

    def _start_analysis_if_needed(self):
        if self.analysis_mode_var.get() and self.game_mode.get() == GameMode.HUMAN_VS_HUMAN.value and not self.game_over:
            self._restart_analysis_process()
    
    def _restart_analysis_process(self):
        """Safely stops the old analysis process and schedules the new one."""
        self._stop_ai_process()
        self.master.after(20, lambda: self._start_ai_process(ChessBot, self.ANALYSIS_AI_NAME, 99))

    def on_drag_end(self, event):
        is_analysis_process_running = self.is_ai_thinking() and self.ai_process.name == self.ANALYSIS_AI_NAME
        if self.is_ai_thinking() and not is_analysis_process_running:
             self.valid_moves = []; self.draw_board(); return
        if not self.dragging:
            self.valid_moves = []; self.draw_board(); return

        self.dragging = False; self.canvas.delete("drag_ghost")
        row, col = self.canvas_to_board(event.x, event.y)
        
        if row == -1 or not self.drag_start:
            self.drag_start, self.selected, self.valid_moves = None, None, []; self.draw_board(); self.set_interactivity(True); return

        start_pos, end_pos = self.drag_start, (row, col)

        if end_pos in self.valid_moves:
            self._start_game_if_needed()
            
            turn_number = len(self.position_history)
            print(f"\n--- Turn {turn_number} ({self.turn.capitalize()}) ---")
            
            # The time is now calculated and printed inside execute_move_and_check_state
            
            self.board.make_move(start_pos, end_pos)
            self.execute_move_and_check_state()

            if not self.game_over:
                mode = self.game_mode.get()
                if mode == GameMode.HUMAN_VS_BOT.value and self.turn != self.human_color:
                    self.master.after(20, self._make_game_ai_move)
                elif mode == GameMode.HUMAN_VS_HUMAN.value:
                    self._start_analysis_if_needed()
                
        self.drag_start, self.selected, self.valid_moves = None, None, []; self.draw_board(); self.set_interactivity(True)

    def execute_move_and_check_state(self):
        self.last_move_timestamp = time.time()
        player_who_moved = self.turn
        if not self.game_over: self.switch_turn()
        key = board_hash(self.board, self.turn)
        self.position_history.append(key); self.position_counts[key] = self.position_counts.get(key, 0) + 1
        status, winner = check_game_over(self.board, player_who_moved)
        if status:
            self.game_over, self.game_result = True, (status, winner)
        elif self.position_counts.get(key, 0) >= 3:
            self.game_over, self.game_result = True, ("repetition", None)
        self.update_turn_label(); self.draw_board()
        if self.game_over:
            self._stop_ai_process()
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
        current_bot_name = self.ai_process.name if self.ai_process else None
        is_thinking_for_move = is_thinking and (current_bot_name != self.ANALYSIS_AI_NAME)
        thinking_text = " (Thinking...)" if is_thinking_for_move else ""
        
        mode = self.game_mode.get()
        if mode == GameMode.HUMAN_VS_BOT.value:
            if self.human_color == "white":
                white_label, black_label = "Human", f"{self.MAIN_AI_NAME}{thinking_text}"
            else:
                white_label, black_label = f"{self.MAIN_AI_NAME}{thinking_text}", "Human"
        elif mode == GameMode.AI_VS_AI.value:
            if self.white_playing_bot_type == 'main':
                white_name, black_name = self.MAIN_AI_NAME, self.OPPONENT_AI_NAME
            else:
                white_name, black_name = self.OPPONENT_AI_NAME, self.MAIN_AI_NAME
            white_label = f"{white_name}{thinking_text if self.turn == 'white' else ''}"
            black_label = f"{black_name}{thinking_text if self.turn == 'black' else ''}"
        else:
            white_label, black_label = "Human (White)", "Human (Black)"

        if self.board_orientation == "white":
            self.bottom_bot_label.config(text=white_label); self.top_bot_label.config(text=black_label)
        else:
            self.bottom_bot_label.config(text=black_label); self.top_bot_label.config(text=white_label)

    def on_drag_start(self, event):
        is_analysis_process_running = self.is_ai_thinking() and self.ai_process.name == self.ANALYSIS_AI_NAME
        if self.game_over or (self.is_ai_thinking() and not is_analysis_process_running):
            return
            
        r, c = self.canvas_to_board(event.x, event.y)
        if r != -1:
            piece = self.board.grid[r][c]
            is_human_turn = (self.game_mode.get() != GameMode.HUMAN_VS_BOT.value or self.turn == self.human_color)
            if piece and piece.color == self.turn and is_human_turn:
                self.selected = (r, c); self.dragging = True; self.drag_start = (r, c)
                self.valid_moves = [end for start, end in get_all_legal_moves(self.board, self.turn) if start == self.selected]
                piece_color = "white" if piece.color == "white" else "black"
                self.drag_piece_ghost = self.canvas.create_text(event.x, event.y, text=piece.symbol(), font=("Arial Unicode MS", 50), fill=piece_color, tags="drag_ghost")
                self.draw_board(); self.canvas.tag_raise("drag_ghost")
    
    def on_drag_motion(self, event):
        if self.dragging:
            self.canvas.coords(self.drag_piece_ghost, event.x, event.y)

    def swap_sides(self):
        self._stop_ai_process()
        if self.game_mode.get() == GameMode.HUMAN_VS_BOT.value:
            self.human_color = "black" if self.human_color == "white" else "white"
        self.reset_game()

    def switch_turn(self):
        self.turn = "black" if self.turn == "white" else "white"

    def board_to_canvas(self, r, c):
        if self.board_orientation == "black": x, y = (COLS-1-c)*SQUARE_SIZE, (ROWS-1-r)*SQUARE_SIZE
        else: x, y = c*SQUARE_SIZE, r*SQUARE_SIZE
        return x, y

    def canvas_to_board(self, x, y):
        if self.board_orientation == "black": c, r = (COLS-1)-(x//SQUARE_SIZE), (ROWS-1)-(y//SQUARE_SIZE)
        else: c, r = x//SQUARE_SIZE, y//SQUARE_SIZE
        if 0 <= r < ROWS and 0 <= c < COLS: return (r, c)
        else: return (-1, -1)

    def draw_eval_bar(self, eval_score, depth=None):
        score = eval_score/100.0; w = self.eval_bar_canvas.winfo_width(); h = self.eval_bar_canvas.winfo_height()
        self.eval_bar_canvas.delete("all")
        if w<=1 or h<=1: return
        for x_pixel in range(w):
            intensity = int(255*(x_pixel/float(w-1))); color = f"#{intensity:02x}{intensity:02x}{intensity:02x}"
            self.eval_bar_canvas.create_line(x_pixel,0,x_pixel,h,fill=color)
        marker_score = max(-1.0,min(1.0,math.tanh(score/20.0))); marker_x = int(((marker_score+1)/2.0)*w)
        self.eval_bar_canvas.create_line(marker_x,0,marker_x,h,fill=self.COLORS['accent'],width=3)
        self.eval_bar_canvas.create_line(w//2,0,w//2,h,fill="#666666",width=1)
        if depth: 
            eval_text = f"{'+' if score>0 else ''}{score:.2f} (D{depth})"
            # Also print eval to console for analysis and AI modes
            if (self.analysis_mode_var.get() and self.game_mode.get() == GameMode.HUMAN_VS_HUMAN.value) or \
               self.game_mode.get() == GameMode.AI_VS_AI.value:
                print(f"  > Eval: {'+' if score>0 else ''}{score:.2f}")
        elif abs(score)<0.05: eval_text = "Even"
        else: eval_text = f"{'+' if score>0 else ''}{score:.2f}"
        self.eval_score_label.config(text=eval_text)

    def setup_styles(self):
        style = ttk.Style(); style.theme_use('clam')
        C = {'bg_dark':'#1a1a2e','bg_medium':'#16213e','bg_light':'#0f3460','accent':'#e94560','text_light':'#ffffff','text_dark':'#a2a2a2'}
        style.configure('.', background=C['bg_dark'], foreground=C['text_light']); style.configure('TFrame', background=C['bg_dark'])
        style.configure('Left.TFrame', background=C['bg_dark']); style.configure('Right.TFrame', background=C['bg_medium']); style.configure('Canvas.TFrame', background=C['bg_medium'])
        style.configure('Header.TLabel', background=C['bg_dark'], foreground=C['text_light'], font=('Helvetica',14,'bold'), padding=(0,10))
        style.configure('SmallHeader.TLabel', background=C['bg_dark'], foreground=C['text_light'], font=('Helvetica',13,'bold'), padding=(0,1))
        style.configure('Status.TLabel', background=C['bg_light'], foreground=C['text_light'], font=('Helvetica',14,'bold'), padding=(11,4), relief='flat')
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
                x1,y1 = c_draw*SQUARE_SIZE, r_draw*SQUARE_SIZE; img.put(color, to=(x1,y1,x1+SQUARE_SIZE,y1+SQUARE_SIZE))
        return img

    def draw_board(self):
        self.canvas.itemconfig(self.board_image_id, image=self.board_image_white if self.board_orientation == "white" else self.board_image_black)
        self.canvas.delete("highlight", "piece", "check_highlight")
        for r_move, c_move in self.valid_moves:
            x1, y1 = self.board_to_canvas(r_move, c_move); center_x, center_y = x1 + SQUARE_SIZE//2, y1 + SQUARE_SIZE//2
            self.canvas.create_oval(center_x-10, center_y-10, center_x+10, center_y+10, fill="#1E90FF", outline="", tags="highlight")
        for r in range(ROWS):
            for c in range(COLS):
                if self.board.grid[r][c]: self.draw_piece_with_check(r, c)

    def draw_piece_with_check(self, r, c):
        piece = self.board.grid[r][c]
        if isinstance(piece, King) and is_in_check(self.board, piece.color):
            color = "darkred" if self.game_over and self.game_result and self.game_result[0] == "checkmate" else "red"
            x1, y1 = self.board_to_canvas(r, c); self.canvas.create_rectangle(x1, y1, x1+SQUARE_SIZE, y1+SQUARE_SIZE, outline=color, width=4, tags="check_highlight")
        if (r, c) != self.drag_start: self.draw_piece_at_canvas_coords(piece, r, c)

    def draw_piece_at_canvas_coords(self, piece, r, c):
        x, y = self.board_to_canvas(r, c); x_center, y_center = x + SQUARE_SIZE//2, y + SQUARE_SIZE//2
        symbol, font, shadow_font = piece.symbol(), ("Arial Unicode MS", 48), ("Arial", 48)
        if piece.color == "white":
            self.canvas.create_text(x_center+2, y_center+2, text=symbol, font=shadow_font, fill="#444444", tags="piece")
            self.canvas.create_text(x_center, y_center, text=symbol, font=font, fill="white", tags="piece")
        else:
            self.canvas.create_text(x_center, y_center, text=symbol, font=font, fill="black", tags="piece")

    def _start_game_if_needed(self):
        if not self.game_started:
            self.game_started = True
            mode_name = self.game_mode.get()
            if mode_name == GameMode.HUMAN_VS_HUMAN.value:
                mode_display = "Human vs Human"
            elif mode_name == GameMode.HUMAN_VS_BOT.value:
                mode_display = "Human vs Bot"
            elif mode_name == GameMode.AI_VS_AI.value:
                mode_display = "AI vs AI"
                
            print("\n" + "="*60 + f"\nNEW GAME: {mode_display}\n" + "="*60)
            print("\n--- Turn 0 (Starting Position) ---")
            
            # Start analysis right away to get initial eval
            if (self.analysis_mode_var.get() and mode_name == GameMode.HUMAN_VS_HUMAN.value) or \
               mode_name == GameMode.AI_VS_AI.value:
                self._restart_analysis_process()

    def update_turn_label(self):
        message = f"Turn: {self.turn.capitalize()}"
        if self.game_result:
            result_type, winner = self.game_result
            if result_type == "checkmate": message = f"Checkmate! {winner.capitalize()} wins!"
            elif result_type == "stalemate": message = "Stalemate! It's a draw."
            elif result_type == "repetition": message = "Draw by Repetition!"
        self.turn_label.config(text=message)
        
    def process_ai_series_result(self):
        self.ai_series_stats['game_count'] += 1
        result_type, winner_color = self.game_result
        if winner_color:
            main_ai_color_for_completed_game = 'white' if self.white_playing_bot_type == 'main' else 'black'
            if winner_color == main_ai_color_for_completed_game:
                self.ai_series_stats['my_ai_wins'] += 1
            else:
                self.ai_series_stats['op_ai_wins'] += 1
        else:
            self.ai_series_stats['draws'] += 1
        
        self.update_scoreboard()
        
        if self.ai_series_running and self.ai_series_stats['game_count'] < 100:
            self.master.after(1000, self.reset_game)
        else:
            self.ai_series_running = False
            self.turn_label.config(text="AI Series Complete!")

    def start_ai_series(self):
        self._stop_ai_process()
        self.game_mode.set(GameMode.AI_VS_AI.value)
        self.ai_series_stats = {'game_count': 0, 'my_ai_wins': 0, 'op_ai_wins': 0, 'draws': 0}
        self.ai_series_running = True
        self.current_opening_move = None
        self.reset_game()
        
    def apply_series_opening_move(self):
        if self.ai_series_stats['game_count'] % 2 == 0:
            print("--- Generating new opening for game pair ---")
            moves = get_all_legal_moves(self.board, "white")
            self.current_opening_move = random.choice(moves) if moves else None
        
        if self.current_opening_move:
            start, end = self.current_opening_move
            print(f"Applying opening move: {start} -> {end}")
            self.board.make_move(start, end)
            
            player_who_moved = self.turn
            self.switch_turn()
            
            key = board_hash(self.board, self.turn)
            self.position_history.append(key)
            self.position_counts[key] = self.position_counts.get(key, 0) + 1
            
            status, winner = check_game_over(self.board, player_who_moved)
            if status:
                self.game_over = True
                self.game_result = (status, winner)

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

if __name__ == "__main__":
    mp.freeze_support()
    root = tk.Tk()
    app = EnhancedChessApp(root)
    root.mainloop()