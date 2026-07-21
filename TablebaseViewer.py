# TablebaseViewer.py (v1.4 - Four independent DTM/turn panels)

import os
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from GameLogic import Board, King, Queen, Rook, Bishop, Knight, Pawn, is_in_check

# --- CONFIGURATION ---
TB_DIR = "TBs"
TB_SUFFIX = "_tb16.bin"
SQUARE_SIZE = 60
BOARD_COLOR_1 = "#D2B48C"
BOARD_COLOR_2 = "#8B5A2B"
CATEGORY_ORDER = (
    "3-Man",
    "4-Man Same-Side",
    "4-Man Cross",
    "5-Man Same-Side",
    "5-Man Cross",
)

PIECE_CLASSES = {'K': King, 'Q': Queen, 'R': Rook, 'B': Bishop, 'N': Knight, 'P': Pawn}
PIECE_CHARS = {'King': 'K', 'Queen': 'Q', 'Rook': 'R', 'Bishop': 'B', 'Knight': 'N', 'Pawn': 'P'}
UNICODE_PIECES = {
    'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
    'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟'
}

# Bucket definitions: key -> (display title, winner_label, turn_value_for_decode)
BUCKET_ORDER = ("ww_wtm", "ww_btm", "bw_wtm", "bw_btm")
BUCKET_TITLES = {
    "ww_wtm": "White Wins  •  White to Move",
    "ww_btm": "White Wins  •  Black to Move",
    "bw_wtm": "Black Wins  •  White to Move",
    "bw_btm": "Black Wins  •  Black to Move",
}
QUARTER_CAP = 25  # 4 buckets x 25 = 100 total, one full quota per winner/turn combo

# --- SML COMPRESSION ARRAYS ---
PAWN_WK_SQUARES = [r*8+c for r in range(8) for c in range(4)]
NON_PAWN_WK_SQUARES = []
for c in range(4):
    for r in range(c + 1):
        NON_PAWN_WK_SQUARES.append(r * 8 + c)


def parse_tablebase_filename(filename):
    stem = filename[:-len(TB_SUFFIX)] if filename.endswith(TB_SUFFIX) else filename
    parts = stem.split('_')
    if len(parts) < 3 or parts[0] != 'K' or parts[-1] != 'K':
        return None

    if 'vs' in parts:
        vs_idx = parts.index('vs')
        white_pieces = parts[1:vs_idx]
        black_pieces = parts[vs_idx + 1:-1]
        if len(white_pieces) == 1 and len(black_pieces) == 1:
            category = "4-Man Cross"
        elif len(white_pieces) == 2 and len(black_pieces) == 1:
            category = "5-Man Cross"
        else:
            return None
    else:
        white_pieces = parts[1:-1]
        black_pieces = []
        if len(white_pieces) == 1:
            category = "3-Man"
        elif len(white_pieces) == 2:
            category = "4-Man Same-Side"
        elif len(white_pieces) == 3:
            category = "5-Man Same-Side"
        else:
            return None

    return {
        "filename": filename,
        "category": category,
        "white_pieces": white_pieces,
        "black_pieces": black_pieces,
    }

class TBViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Jungle Chess Tablebase Explorer")
        self.root.geometry("1280x780")
        self.root.minsize(1100, 680)
        self.root.configure(bg="#1a1a2e")

        self.fens = []
        self.bucket_fens = {key: [] for key in BUCKET_ORDER}   # key -> [(dtm, fen), ...]
        self.bucket_listboxes = {}                              # key -> Listbox
        self.categorized_files = {category: [] for category in CATEGORY_ORDER}

        self.setup_styles()
        self.build_ui()
        self.load_file_list()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background="#1a1a2e")
        style.configure('TLabel', background="#1a1a2e", foreground="#ffffff", font=('Helvetica', 11))
        style.configure('Header.TLabel', font=('Helvetica', 14, 'bold'))
        style.configure('BucketTitle.TLabel', font=('Helvetica', 10, 'bold'), foreground="#e94560")
        style.configure('TButton', background="#e94560", foreground="#ffffff", font=('Helvetica', 11, 'bold'))
        style.map('TButton', background=[('active', '#d13550')])
        style.configure('Bucket.TFrame', background="#16213e", relief='flat')
        style.configure('BucketHeader.TFrame', background="#0f1729")

    def build_ui(self):
        top_frame = ttk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=14, pady=(14, 8))

        ttk.Label(top_frame, text="Category:").pack(side=tk.LEFT, padx=(0, 5))
        self.category_combo = ttk.Combobox(top_frame, state="readonly", width=20, font=('Helvetica', 11))
        self.category_combo.pack(side=tk.LEFT, padx=5)
        self.category_combo.bind('<<ComboboxSelected>>', self.on_select_category)

        ttk.Label(top_frame, text="Tablebase:").pack(side=tk.LEFT, padx=(10, 5))
        self.file_combo = ttk.Combobox(top_frame, state="readonly", width=30, font=('Helvetica', 11))
        self.file_combo.pack(side=tk.LEFT, padx=5)

        self.load_btn = ttk.Button(top_frame, text="Extract Verified Longest Mates", command=self.load_tablebase)
        self.load_btn.pack(side=tk.LEFT, padx=10)

        content_frame = ttk.Frame(self.root)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 14))

        # --- LEFT: 2x2 grid of bucket panels ---
        left_frame = ttk.Frame(content_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 16))

        grid_frame = ttk.Frame(left_frame)
        grid_frame.pack(fill=tk.BOTH, expand=True)

        for i, key in enumerate(BUCKET_ORDER):
            row, col = divmod(i, 2)
            self._build_bucket_panel(grid_frame, key, row, col)

        grid_frame.grid_columnconfigure(0, weight=1)
        grid_frame.grid_columnconfigure(1, weight=1)
        grid_frame.grid_rowconfigure(0, weight=1)
        grid_frame.grid_rowconfigure(1, weight=1)

        # --- RIGHT: board + info + FEN ---
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

    def _build_bucket_panel(self, parent, key, row, col):
        panel = ttk.Frame(parent, style='Bucket.TFrame')
        panel.grid(row=row, column=col, sticky="nsew", padx=6, pady=6)

        header = ttk.Frame(panel, style='BucketHeader.TFrame')
        header.pack(fill=tk.X)
        ttk.Label(header, text=BUCKET_TITLES[key], style='BucketTitle.TLabel',
                  background="#0f1729").pack(anchor=tk.W, padx=8, pady=6)

        body = ttk.Frame(panel, style='Bucket.TFrame')
        body.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        listbox = tk.Listbox(body, width=32, height=11, bg="#16213e", fg="#ffffff",
                             font=('Consolas', 9), selectbackground="#e94560", selectforeground="#ffffff",
                             borderwidth=0, highlightthickness=0, activestyle='none')
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        listbox.bind('<<ListboxSelect>>', lambda e, k=key: self.on_select_fen(k))

        scrollbar = ttk.Scrollbar(body, orient=tk.VERTICAL, command=listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        listbox.config(yscrollcommand=scrollbar.set)

        self.bucket_listboxes[key] = listbox

    def load_file_list(self):
        if not os.path.exists(TB_DIR): os.makedirs(TB_DIR)
        files = [f for f in os.listdir(TB_DIR) if f.endswith(TB_SUFFIX)]
        if not files: return

        for f in sorted(files):
            metadata = parse_tablebase_filename(f)
            if metadata is None:
                continue
            self.categorized_files[metadata["category"]].append(f)

        available_categories = [category for category in CATEGORY_ORDER if self.categorized_files[category]]
        self.category_combo['values'] = available_categories
        if self.category_combo['values']:
            self.category_combo.current(0)
            self.on_select_category(None)

    def on_select_category(self, event):
        files = self.categorized_files.get(self.category_combo.get(), [])
        self.file_combo['values'] = files
        if files: self.file_combo.current(0)
        else: self.file_combo.set('')

    def _is_position_legal(self, placements, turn):
        """Verifies no overlapping pieces and that the passive player is not in check."""
        board = Board(setup=False)
        pos_set = set()

        for char, pos in placements:
            if pos in pos_set: return False # Pieces overlapping
            pos_set.add(pos)

            r, c = pos // 8, pos % 8
            color = 'white' if char.isupper() else 'black'
            piece_class = PIECE_CLASSES[char.upper()]
            board.add_piece(piece_class(color), r, c)

        passive_color = 'black' if turn == 0 else 'white'
        if is_in_check(board, passive_color):
            return False # Passive player is in check (Illegal starting state)

        return True

    def load_tablebase(self):
        filename = self.file_combo.get()
        if not filename: return

        filepath = os.path.join(TB_DIR, filename)
        metadata = parse_tablebase_filename(filename)
        if metadata is None:
            messagebox.showerror("Error", f"Could not parse tablebase name:\n{filename}")
            return
        has_pawn = "Pawn" in filename

        try:
            data = np.memmap(filepath, dtype=np.int16, mode='r')
            data_flat = data.reshape(-1)
            abs_data = np.abs(data_flat)

            n = len(data_flat)
            turn0 = np.zeros(n, dtype=bool)   # True where flat index is even -> White to move
            turn0[0::2] = True
            turn1 = ~turn0

            decisive_mask = abs_data > 0
            white_wins_mask = decisive_mask & (((data_flat > 0) & turn0) | ((data_flat < 0) & turn1))
            black_wins_mask = decisive_mask & ~white_wins_mask & decisive_mask

            for key in self.bucket_fens:
                self.bucket_fens[key] = []
                self.bucket_listboxes[key].delete(0, tk.END)

            bucket_masks = {
                "ww_wtm": white_wins_mask & turn0,
                "ww_btm": white_wins_mask & turn1,
                "bw_wtm": black_wins_mask & turn0,
                "bw_btm": black_wins_mask & turn1,
            }

            any_found = False

            for key in BUCKET_ORDER:
                mask = bucket_masks[key]
                side_indices = np.where(mask)[0]
                if len(side_indices) == 0:
                    continue
                order = np.argsort(-abs_data[side_indices])
                ranked_indices = side_indices[order]

                bucket_count = 0
                listbox = self.bucket_listboxes[key]
                for idx in ranked_indices:
                    if bucket_count >= QUARTER_CAP:
                        break
                    idx = int(idx)
                    placements, turn = self.decode_placements(idx, metadata, has_pawn)
                    if not placements: continue

                    if not self._is_position_legal(placements, turn):
                        continue

                    fen = self.build_fen(placements, turn)
                    dtm_val = int(abs_data[idx])
                    self.bucket_fens[key].append((dtm_val, fen))
                    listbox.insert(tk.END, f"#{bucket_count+1}  DTM {dtm_val}")
                    bucket_count += 1
                    any_found = True

            if not any_found:
                messagebox.showinfo("Result", "No decisive, legal positions found.")
                return

            # Auto-select the first available entry across buckets, in display order
            for key in BUCKET_ORDER:
                if self.bucket_fens[key]:
                    self.bucket_listboxes[key].selection_set(0)
                    self.on_select_fen(key)
                    break

        except Exception as e:
            messagebox.showerror("Error", f"Failed to parse tablebase:\n{e}")

    def decode_placements(self, flat, metadata, has_pawn):
        turn = flat % 2
        rest = flat // 2
        wk_squares = PAWN_WK_SQUARES if has_pawn else NON_PAWN_WK_SQUARES
        white_pieces = metadata["white_pieces"]
        black_pieces = metadata["black_pieces"]
        category = metadata["category"]

        if category == "3-Man":
            bk = rest % 64
            p1 = (rest // 64) % 64
            wk_idx = rest // 4096
            wk = wk_squares[wk_idx]
            return [('K', wk), ('k', bk), (PIECE_CHARS[white_pieces[0]], p1)], turn

        elif category == "4-Man Same-Side":
            bk = rest % 64
            p2 = (rest // 64) % 64
            p1 = (rest // 4096) % 64
            wk_idx = rest // 262144
            wk = wk_squares[wk_idx]
            return [('K', wk), ('k', bk), (PIECE_CHARS[white_pieces[0]], p1), (PIECE_CHARS[white_pieces[1]], p2)], turn

        elif category == "4-Man Cross":
            bp = rest % 64
            bk = (rest // 64) % 64
            wp = (rest // 4096) % 64
            wk_idx = rest // 262144
            wk = wk_squares[wk_idx]
            return [('K', wk), ('k', bk), (PIECE_CHARS[white_pieces[0]], wp), (PIECE_CHARS[black_pieces[0]].lower(), bp)], turn

        elif category == "5-Man Same-Side":
            bk = rest % 64
            p3 = (rest // 64) % 64
            p2 = (rest // 4096) % 64
            p1 = (rest // 262144) % 64
            wk_idx = rest // 16777216
            wk = wk_squares[wk_idx]
            return [
                ('K', wk),
                ('k', bk),
                (PIECE_CHARS[white_pieces[0]], p1),
                (PIECE_CHARS[white_pieces[1]], p2),
                (PIECE_CHARS[white_pieces[2]], p3),
            ], turn

        elif category == "5-Man Cross":
            bp = rest % 64
            bk = (rest // 64) % 64
            wp2 = (rest // 4096) % 64
            wp1 = (rest // 262144) % 64
            wk_idx = rest // 16777216
            wk = wk_squares[wk_idx]
            return [
                ('K', wk),
                ('k', bk),
                (PIECE_CHARS[white_pieces[0]], wp1),
                (PIECE_CHARS[white_pieces[1]], wp2),
                (PIECE_CHARS[black_pieces[0]].lower(), bp),
            ], turn

        return None, None

    def build_fen(self, placements, turn):
        board = [['' for _ in range(8)] for _ in range(8)]
        for char, pos in placements:
            board[pos // 8][pos % 8] = char

        fen_rows = []
        for r in range(8):
            empty = 0
            row_str = ""
            for c in range(8):
                if board[r][c] == '': empty += 1
                else:
                    if empty > 0: row_str += str(empty); empty = 0
                    row_str += board[r][c]
            if empty > 0: row_str += str(empty)
            fen_rows.append(row_str)

        return "/".join(fen_rows) + f" {'w' if turn == 0 else 'b'} - - 0 1"

    def on_select_fen(self, key):
        listbox = self.bucket_listboxes[key]
        if not listbox.curselection(): return

        # Deselect the other three panels so it's always clear which one is active
        for other_key, other_box in self.bucket_listboxes.items():
            if other_key != key:
                other_box.selection_clear(0, tk.END)

        idx = listbox.curselection()[0]
        dtm, fen = self.bucket_fens[key][idx]
        self.fen_entry.delete(0, tk.END)
        self.fen_entry.insert(0, fen)
        self.info_label.config(
            text=f"{BUCKET_TITLES[key]}  —  Verified Longest Mate: {dtm} plies ({(dtm + 1)//2} moves)"
        )
        self.draw_board_from_fen(fen)

    def draw_empty_board(self):
        self.canvas.delete("all")
        for i in range(8):
            self.canvas.create_text(10, 30 + i * SQUARE_SIZE + SQUARE_SIZE // 2, text=str(8 - i), fill="#a2a2a2", font=('Helvetica', 10, 'bold'))
            self.canvas.create_text(30 + i * SQUARE_SIZE + SQUARE_SIZE // 2, 8 * SQUARE_SIZE + 30, text=chr(97 + i), fill="#a2a2a2", font=('Helvetica', 10, 'bold'))
        for r in range(8):
            for c in range(8):
                x1, y1 = 20 + c * SQUARE_SIZE, 20 + r * SQUARE_SIZE
                color = BOARD_COLOR_1 if (r + c) % 2 == 0 else BOARD_COLOR_2
                self.canvas.create_rectangle(x1, y1, x1 + SQUARE_SIZE, y1 + SQUARE_SIZE, fill=color, outline="")

    def draw_board_from_fen(self, fen):
        self.draw_empty_board()
        r, c = 0, 0
        for char in fen.split(" ")[0]:
            if char == '/': r += 1; c = 0
            elif char.isdigit(): c += int(char)
            else:
                x = 20 + c * SQUARE_SIZE + SQUARE_SIZE // 2
                y = 20 + r * SQUARE_SIZE + SQUARE_SIZE // 2 + 2
                sym = UNICODE_PIECES.get(char, "?")
                self.canvas.create_text(x + 1, y + 1, text=sym, font=("Arial Unicode MS", 40), fill="#444444")
                self.canvas.create_text(x, y, text=sym, font=("Arial Unicode MS", 40), fill="#ffffff" if char.isupper() else "#000000")
                c += 1

    def copy_fen(self):
        fen = self.fen_entry.get()
        if fen:
            self.root.clipboard_clear()
            self.root.clipboard_append(fen)
            self.root.update()  # force an eager clipboard write so it survives app close
            messagebox.showinfo("Copied", "FEN copied to clipboard!")

if __name__ == "__main__":
    root = tk.Tk()
    app = TBViewerApp(root)
    root.mainloop()