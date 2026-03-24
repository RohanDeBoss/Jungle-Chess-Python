#v1.0
import os
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from operator import itemgetter # <--- ADD THIS LINE

# --- CONFIGURATION ---
TB_DIR = "tablebases"
SQUARE_SIZE = 60
BOARD_COLOR_1 = "#D2B48C"
BOARD_COLOR_2 = "#8B5A2B"

PIECE_CHARS = {'King': 'K', 'Queen': 'Q', 'Rook': 'R', 'Bishop': 'B', 'Knight': 'N', 'Pawn': 'P'}
UNICODE_PIECES = {
    'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
    'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟'
}

class TBViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Jungle Chess Tablebase Explorer")
        self.root.geometry("900x650")
        self.root.configure(bg="#1a1a2e")

        self.fens = []
        self.current_fens = []
        self.categorized_files = {
            "3-Man": [],
            "4-Man Same-Side": [],
            "4-Man Cross": []
        }
        
        self.setup_styles()
        self.build_ui()
        self.load_file_list()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background="#1a1a2e")
        style.configure('TLabel', background="#1a1a2e", foreground="#ffffff", font=('Helvetica', 11))
        style.configure('Header.TLabel', font=('Helvetica', 14, 'bold'))
        style.configure('TButton', background="#e94560", foreground="#ffffff", font=('Helvetica', 11, 'bold'))
        style.map('TButton', background=[('active', '#d13550')])

    def build_ui(self):
        # Top Frame - Controls
        top_frame = ttk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        ttk.Label(top_frame, text="Category:").pack(side=tk.LEFT, padx=(0, 5))
        self.category_combo = ttk.Combobox(top_frame, state="readonly", width=20, font=('Helvetica', 11))
        self.category_combo.pack(side=tk.LEFT, padx=5)
        self.category_combo.bind('<<ComboboxSelected>>', self.on_select_category)

        ttk.Label(top_frame, text="Tablebase:").pack(side=tk.LEFT, padx=(10, 5))
        self.file_combo = ttk.Combobox(top_frame, state="readonly", width=30, font=('Helvetica', 11))
        self.file_combo.pack(side=tk.LEFT, padx=5)
        
        self.load_btn = ttk.Button(top_frame, text="Extract Longest Mates", command=self.load_tablebase)
        self.load_btn.pack(side=tk.LEFT, padx=10)

        # Main Content Frame
        content_frame = ttk.Frame(self.root)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left Side - List of FENs
        left_frame = ttk.Frame(content_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))

        ttk.Label(left_frame, text="Found Positions (Max 100):").pack(anchor=tk.W)
        
        self.listbox = tk.Listbox(left_frame, width=45, height=25, bg="#16213e", fg="#ffffff", 
                                  font=('Consolas', 10), selectbackground="#e94560", selectforeground="#ffffff")
        self.listbox.pack(side=tk.LEFT, fill=tk.Y)
        self.listbox.bind('<<ListboxSelect>>', self.on_select_fen)

        scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.config(yscrollcommand=scrollbar.set)

        # Right Side - Board Visualization
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(right_frame, width=8 * SQUARE_SIZE + 40, height=8 * SQUARE_SIZE + 40, 
                                bg="#1a1a2e", highlightthickness=0)
        self.canvas.pack(pady=10)
        
        self.info_label = ttk.Label(right_frame, text="Select a position to view", font=('Helvetica', 12, 'italic'))
        self.info_label.pack(pady=5)

        fen_frame = ttk.Frame(right_frame)
        fen_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(fen_frame, text="FEN:").pack(side=tk.LEFT)
        self.fen_entry = ttk.Entry(fen_frame, font=('Consolas', 11))
        self.fen_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        ttk.Button(fen_frame, text="Copy FEN", command=self.copy_fen).pack(side=tk.RIGHT)

        self.draw_empty_board()

    def load_file_list(self):
        if not os.path.exists(TB_DIR):
            os.makedirs(TB_DIR)
        
        files = [f for f in os.listdir(TB_DIR) if f.endswith(".bin")]
        if not files:
            messagebox.showinfo("No Tablebases", f"No .bin files found in '{TB_DIR}' directory.")
            return

        for f in sorted(files):
            parts = f[:-4].split('_')
            if len(parts) == 3 and parts[0] == 'K' and parts[2] == 'K':
                self.categorized_files["3-Man"].append(f)
            elif len(parts) == 4 and parts[0] == 'K' and parts[3] == 'K':
                self.categorized_files["4-Man Same-Side"].append(f)
            elif len(parts) == 5 and parts[0] == 'K' and parts[2] == 'vs' and parts[4] == 'K':
                self.categorized_files["4-Man Cross"].append(f)
        
        self.category_combo['values'] = list(self.categorized_files.keys())
        if self.category_combo['values']:
            self.category_combo.current(0)
            self.on_select_category(None)

    def on_select_category(self, event):
        category = self.category_combo.get()
        files = self.categorized_files.get(category, [])
        self.file_combo['values'] = files
        if files:
            self.file_combo.current(0)
        else:
            self.file_combo.set('')

    def load_tablebase(self):
        filename = self.file_combo.get()
        if not filename: 
            messagebox.showwarning("Warning", "No tablebase file selected.")
            return
        
        filepath = os.path.join(TB_DIR, filename)
        base = filename[:-4]
        parts = base.split('_')

        # Determine Tablebase Type and Pieces based on Filename
        tb_type = 0
        w_pieces = []
        b_pieces = []

        if len(parts) == 3:  # K_Piece_K
            tb_type = 3
            w_pieces = [parts[1]]
        elif len(parts) == 4:  # K_Piece_Piece_K
            tb_type = 4
            w_pieces = [parts[1], parts[2]]
        elif len(parts) == 5 and parts[2] == 'vs' and parts[4] == 'K':  # K_Piece_vs_Piece_K
            tb_type = 5
            w_pieces = [parts[1]]
            b_pieces = [parts[3]]
        else:
            messagebox.showerror("Error", f"Unrecognized filename format: {filename}")
            return

        try:
            # Instantly map the file into memory
            data = np.memmap(filepath, dtype=np.int16, mode='r')
            
            # Find the absolute max values
            max_w = np.max(data)
            min_b = np.min(data)
            
            self.current_fens = []
            self.listbox.delete(0, tk.END)

            # Extract White longest mates
            if max_w > 0:
                indices = np.where(data == max_w)[0][:100] # Cap at 100 to prevent UI lag
                for idx in indices:
                    fen = self.decode_to_fen(idx, tb_type, w_pieces, b_pieces)
                    if fen: self.current_fens.append((max_w, "White Wins", fen))

            # Extract Black longest mates
            if min_b < 0:
                indices = np.where(data == min_b)[0][:100]
                for idx in indices:
                    fen = self.decode_to_fen(idx, tb_type, w_pieces, b_pieces)
                    if fen: self.current_fens.append((abs(min_b), "Black Wins", fen))

            # --- BUG FIX STARTS HERE ---
            # Sort the combined list by DTM (the first element of the tuple), descending.
            self.current_fens.sort(key=itemgetter(0), reverse=True)
            # --- BUG FIX ENDS HERE ---

            if not self.current_fens:
                messagebox.showinfo("Result", "No decisive positions found in this tablebase.")
                return

            # Populate UI Listbox with the now-sorted data
            for i, (dtm, winner, fen) in enumerate(self.current_fens):
                turn = "W" if " w " in fen else "B"
                self.listbox.insert(tk.END, f"#{i+1} DTM {dtm} ({winner}) [{turn} to move]")
                
            self.listbox.selection_set(0)
            self.on_select_fen(None)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to parse tablebase:\n{e}")

    def decode_to_fen(self, flat, tb_type, w_pieces, b_pieces):
        """Reverses the generator's flat index calculation to get board coordinates."""
        placements = []
        
        if tb_type == 3:
            turn = flat % 2; rest = flat // 2
            bk = rest % 64; rest //= 64
            p1 = rest % 64; wk = rest // 64
            placements = [('K', wk), ('k', bk), (PIECE_CHARS[w_pieces[0]], p1)]
            
        elif tb_type == 4:
            turn = flat % 2; rest = flat // 2
            bk = rest % 64; rest //= 64
            p2 = rest % 64; rest //= 64
            p1 = rest % 64; wk = rest // 64
            placements = [('K', wk), ('k', bk), (PIECE_CHARS[w_pieces[0]], p1), (PIECE_CHARS[w_pieces[1]], p2)]
            
        elif tb_type == 5:
            turn = flat % 2; rest = flat // 2
            bp = rest % 64; rest //= 64
            bk = rest % 64; rest //= 64
            wp = rest % 64; wk = rest // 64
            placements = [('K', wk), ('k', bk), (PIECE_CHARS[w_pieces[0]], wp), (PIECE_CHARS[b_pieces[0]].lower(), bp)]

        # Filter overlapping pieces (Invalid states created by raw index limits)
        pos_set = set()
        for _, pos in placements:
            if pos in pos_set: return None
            pos_set.add(pos)

        # Build FEN
        board = [['' for _ in range(8)] for _ in range(8)]
        for char, pos in placements:
            r, c = pos // 8, pos % 8
            board[r][c] = char

        fen_rows = []
        for r in range(8):
            empty = 0
            row_str = ""
            for c in range(8):
                if board[r][c] == '':
                    empty += 1
                else:
                    if empty > 0:
                        row_str += str(empty)
                        empty = 0
                    row_str += board[r][c]
            if empty > 0:
                row_str += str(empty)
            fen_rows.append(row_str)

        turn_char = 'w' if turn == 0 else 'b'
        return "/".join(fen_rows) + f" {turn_char} - - 0 1"

    def on_select_fen(self, event):
        if not self.listbox.curselection(): return
        
        idx = self.listbox.curselection()[0]
        dtm, winner, fen = self.current_fens[idx]
        
        self.fen_entry.delete(0, tk.END)
        self.fen_entry.insert(0, fen)
        self.info_label.config(text=f"Longest Mate: {winner} in {dtm} plies ({(dtm + 1)//2} moves)")
        
        self.draw_board_from_fen(fen)

    def draw_empty_board(self):
        self.canvas.delete("all")
        # Draw coordinate margins
        for i in range(8):
            self.canvas.create_text(10, 30 + i * SQUARE_SIZE + SQUARE_SIZE // 2, text=str(8 - i), fill="#a2a2a2", font=('Helvetica', 10, 'bold'))
            self.canvas.create_text(30 + i * SQUARE_SIZE + SQUARE_SIZE // 2, 8 * SQUARE_SIZE + 30, text=chr(97 + i), fill="#a2a2a2", font=('Helvetica', 10, 'bold'))
            
        for r in range(8):
            for c in range(8):
                x1 = 20 + c * SQUARE_SIZE
                y1 = 20 + r * SQUARE_SIZE
                color = BOARD_COLOR_1 if (r + c) % 2 == 0 else BOARD_COLOR_2
                self.canvas.create_rectangle(x1, y1, x1 + SQUARE_SIZE, y1 + SQUARE_SIZE, fill=color, outline="")

    def draw_board_from_fen(self, fen):
        self.draw_empty_board()
        board_part = fen.split(" ")[0]
        
        r, c = 0, 0
        for char in board_part:
            if char == '/':
                r += 1
                c = 0
            elif char.isdigit():
                c += int(char)
            else:
                x = 20 + c * SQUARE_SIZE + SQUARE_SIZE // 2
                y = 20 + r * SQUARE_SIZE + SQUARE_SIZE // 2 + 2
                sym = UNICODE_PIECES.get(char, "?")
                
                # Drop shadow
                self.canvas.create_text(x + 1, y + 1, text=sym, font=("Arial Unicode MS", 40), fill="#444444")
                # Piece
                fill_color = "#ffffff" if char.isupper() else "#000000"
                self.canvas.create_text(x, y, text=sym, font=("Arial Unicode MS", 40), fill=fill_color)
                
                c += 1

    def copy_fen(self):
        fen = self.fen_entry.get()
        if fen:
            self.root.clipboard_clear()
            self.root.clipboard_append(fen)
            messagebox.showinfo("Copied", "FEN copied to clipboard!")

if __name__ == "__main__":
    root = tk.Tk()
    app = TBViewerApp(root)
    root.mainloop()