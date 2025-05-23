# game_logic.py

# -----------------------------
# Global Constants
# -----------------------------
ROWS, COLS = 8, 8
SQUARE_SIZE = 65  # <--- MAKE SURE THIS LINE EXISTS AND IS NOT COMMENTED OUT
BOARD_COLOR_1 = "#D2B48C"
BOARD_COLOR_2 = "#8B5A2B"
# SQUARE_SIZE, BOARD_COLOR_1, etc. are UI constants.

DIRECTIONS = {
    'king': ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)),
    'queen': ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)),
    'rook': ((0, 1), (0, -1), (1, 0), (-1, 0)),
    'bishop': ((-1, -1), (-1, 1), (1, -1), (1, 1)),
    'knight': ((2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2))
}

ADJACENT_DIRS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),          (0, 1),
    (1, -1),  (1, 0), (1, 1)
)

# -----------------------------
# Piece Base Class and Subclasses
# -----------------------------
class Piece:
    def __init__(self, color):
        self.color = color
        self.has_moved = False
        # Store opponent_color for convenience
        self.opponent_color = "black" if color == "white" else "white"


    def clone(self):
        new_piece = self.__class__(self.color)
        new_piece.has_moved = self.has_moved
        return new_piece

    # save_state and restore_state are not used by the core logic provided,
    # but can be kept if your AI or other parts use them.
    def save_state(self):
        return {'has_moved': self.has_moved}

    def restore_state(self, state):
        self.has_moved = state['has_moved']

    def symbol(self):
        return "?" # Should be overridden

    def get_valid_moves(self, board, pos):
        return [] # Should be overridden

    def move(self, board, start, end):
        # Base move: relocate piece, handle simple capture by overwriting.
        # Special effects (explosion, promotion, etc.) are in subclass overrides.
        # The piece at board[end[0]][end[1]] is the piece being captured.
        # It's effectively removed when 'self' is placed at 'end'.
        board[end[0]][end[1]] = self
        board[start[0]][start[1]] = None
        self.has_moved = True
        return board

    def _is_valid_square(self, r, c): # Helper for boundary checks
        return 0 <= r < ROWS and 0 <= c < COLS

class King(Piece):
    def symbol(self): return "♔" if self.color == "white" else "♚"

    def get_valid_moves(self, board, pos):
        moves = []
        r_start, c_start = pos # Use original variable names
        for dr, dc in DIRECTIONS['king']: # Use original DIRECTIONS
            for step in (1, 2):
                new_r, new_c = r_start + dr * step, c_start + dc * step
                if self._is_valid_square(new_r, new_c):
                    if step == 2:
                        inter_r, inter_c = r_start + dr, c_start + dc
                        if board[inter_r][inter_c] is not None:
                            break
                    target = board[new_r][new_c]
                    if target is None or target.color != self.color: # Use original self.color
                        moves.append((new_r, new_c))
                    if target is not None:
                        break
                else: # Off board
                    break
        return moves

class Queen(Piece):
    def symbol(self): return "♕" if self.color == "white" else "♛"

    def get_valid_moves(self, board, pos):
        moves = []
        r_start, c_start = pos # Original variable names
        for dr, dc in DIRECTIONS['queen']: # Original DIRECTIONS
            r, c = r_start, c_start # Use r,c for iteration as in original
            while True:
                r += dr
                c += dc
                if not self._is_valid_square(r, c):
                    break
                target = board[r][c]
                if target is None:
                    moves.append((r, c))
                else:
                    if target.color != self.color: # Original self.color
                        moves.append((r, c))
                    break
        return moves

    def move(self, board, start, end):
        # If the queen captures an enemy piece, trigger explosion
        piece_at_destination = board[end[0]][end[1]]
        is_capture = piece_at_destination and piece_at_destination.color != self.color

        if is_capture:
            board[end[0]][end[1]] = None # Remove the target piece
            for dr_adj, dc_adj in ADJACENT_DIRS: # Original ADJACENT_DIRS
                adj_r, adj_c = end[0] + dr_adj, end[1] + dc_adj # Use original adj_r, adj_c style
                if self._is_valid_square(adj_r, adj_c):
                    piece_on_adj_square = board[adj_r][adj_c]
                    if piece_on_adj_square and piece_on_adj_square.color != self.color:
                        board[adj_r][adj_c] = None
            board[start[0]][start[1]] = None # Remove the queen from her original square
            # board[end[0]][end[1]] = None # Queen "dies", so destination is empty
            self.has_moved = True # Still counts as moved
        else:
            # Normal move (no capture, no explosion)
            super().move(board, start, end) # Call base class move
        return board

class Rook(Piece):
    def symbol(self): return "♖" if self.color == "white" else "♜"

    def get_valid_moves(self, board, pos):
        moves = []
        r_start, c_start = pos # Original variable names
        for dr, dc in DIRECTIONS['rook']: # Original DIRECTIONS
            r, c = r_start, c_start # Use r,c for iteration
            enemy_encountered = False # Original flag name
            while True:
                r += dr
                c += dc
                if not self._is_valid_square(r,c):
                    break
                target = board[r][c]
                if not enemy_encountered:
                    if target is None:
                        moves.append((r, c))
                    else:
                        if target.color != self.color: # Original self.color
                            moves.append((r, c))
                            enemy_encountered = True
                        else: # Own piece
                            break
                else: # Enemy already encountered on this path
                    if target is None or target.color != self.color: # Original self.color
                        moves.append((r, c))
                    else: # Blocked by own piece or second enemy if rule changes
                        break
        return moves

    def move(self, board, start, end):
        # Rook path clearing logic
        # Determine direction delta d = (dr, dc)
        if start[0] == end[0]: # Horizontal move
            d = (0, 1 if end[1] > start[1] else -1)
        else: # Vertical move
            d = (1 if end[0] > start[0] else -1, 0)
        
        # Check and clear path
        current_r, current_c = start[0] + d[0], start[1] + d[1]
        path_contained_enemy = False
        path_to_clear = []

        while (current_r, current_c) != end:
            if not self._is_valid_square(current_r, current_c): break # Safety
            piece_on_path = board[current_r][current_c]
            if piece_on_path and piece_on_path.color != self.color:
                path_contained_enemy = True
            path_to_clear.append((current_r, current_c)) # Store squares on path
            current_r += d[0]
            current_c += d[1]
        
        if path_contained_enemy: # If any enemy was on the path
            for r_path, c_path in path_to_clear:
                piece_to_clear = board[r_path][c_path]
                if piece_to_clear and piece_to_clear.color != self.color: # Clear all enemies
                    board[r_path][c_path] = None
        
        super().move(board, start, end) # Perform the actual move of the Rook
        return board

class Bishop(Piece):
    def symbol(self): return "♗" if self.color == "white" else "♝"

    def get_valid_moves(self, board, pos):
        # Original logic: combine zigzag and diagonal, remove duplicates using set
        return list(set(get_zigzag_moves(board, pos, self.color) + get_diagonal_moves(board, pos, self.color)))

    # move is inherited from Piece base class

class Knight(Piece):
    def symbol(self): return "♘" if self.color == "white" else "♞"

    def get_valid_moves(self, board, pos):
        # Using original list comprehension structure
        r_start, c_start = pos
        moves = []
        for dr, dc in DIRECTIONS['knight']:
            nr, nc = r_start + dr, c_start + dc
            if self._is_valid_square(nr, nc):
                piece_at_target = board[nr][nc] # Explicitly get piece
                if not piece_at_target or piece_at_target.color != self.color:
                    moves.append((nr,nc))
        return moves
        # Original with walrus operator (Python 3.8+):
        # return [
        #     (pos[0] + dr, pos[1] + dc)
        #     for dr, dc in DIRECTIONS['knight']
        #     if 0 <= (pos[0] + dr) < ROWS
        #     and 0 <= (pos[1] + dc) < COLS
        #     and (not (piece := board[pos[0]+dr][pos[1]+dc])
        #     or piece.color != self.color)
        # ]


    def move(self, board, start, end):
        super().move(board, start, end)
        self.evaporate(board, end) # Call original evaporate method name
        return board

    def evaporate(self, board, pos): # Original method name
        enemy_knights_evaporated = [] # Original variable name
        r_knight, c_knight = pos
        for dr, dc in DIRECTIONS['knight']:
            r_adj, c_adj = r_knight + dr, c_knight + dc # Original r,c for adjacent
            if self._is_valid_square(r_adj, c_adj):
                piece_on_adj = board[r_adj][c_adj] # Original piece variable name
                if piece_on_adj and piece_on_adj.color != self.color:
                    if isinstance(piece_on_adj, Knight):
                        enemy_knights_evaporated.append((r_adj, c_adj)) # Original list name
                    board[r_adj][c_adj] = None
        if enemy_knights_evaporated: # Original check
            board[r_knight][c_knight] = None # Knight evaporates itself

class Pawn(Piece):
    def symbol(self): return "♙" if self.color == "white" else "♟"

    def get_valid_moves(self, board, pos):
        moves = []
        r_start, c_start = pos # Original variable names
        direction = -1 if self.color == "white" else 1
        starting_row = 6 if self.color == "white" else 1 # Original variable name

        for steps in [1, 2]:
            if steps == 2 and r_start != starting_row:
                continue

            new_r, new_c = r_start + (direction * steps), c_start # Original new_r, new_c
            if self._is_valid_square(new_r, new_c):
                piece_at_fwd = board[new_r][new_c]
                if piece_at_fwd is not None and piece_at_fwd.color != self.color: # Forward capture
                    moves.append((new_r, new_c))
                if piece_at_fwd is None: # Normal forward move
                    moves.append((new_r, new_c))
                if piece_at_fwd is not None: # Blocked by any piece
                    break
            else: # Off board
                break

        for dc_offset in [-1, 1]: # Original dc for sideways
            new_c_side = c_start + dc_offset # Original new_c for side capture
            if self._is_valid_square(r_start, new_c_side): # Check square validity
                piece_at_side = board[r_start][new_c_side]
                if piece_at_side is not None and piece_at_side.color != self.color:
                    moves.append((r_start, new_c_side))
        return moves

    def move(self, board, start, end):
        super().move(board, start, end) # Call base move
        promotion_rank = 0 if self.color == "white" else (ROWS - 1)
        if end[0] == promotion_rank:
            board[end[0]][end[1]] = Queen(self.color)
        return board

# -----------------------------
# Helper Functions
# -----------------------------
def get_zigzag_moves(board, pos, color): # Parameter 'color' is piece_color
    moves = set()
    r_start, c_start = pos # Original r,c for start
    # Original direction_pairs
    direction_pairs = (
        ((-1, 1), (-1, -1)), ((-1, -1), (-1, 1)),
        ((1, 1), (1, -1)), ((1, -1), (1, 1)),
        ((-1, 1), (1, 1)), ((1, 1), (-1, 1)),
        ((-1, -1), (1, -1)), ((1, -1), (-1, -1))
    )
    for d1, d2 in direction_pairs:
        cr, cc, cd = r_start, c_start, d1 # Original cr, cc, cd
        while True:
            cr += cd[0]
            cc += cd[1]
            if not (0 <= cr < ROWS and 0 <= cc < COLS): # Original boundary check
                break
            piece_on_path = board[cr][cc] # Original piece variable name
            if piece_on_path:
                if piece_on_path.color != color: # Original color check
                    moves.add((cr, cc))
                break
            moves.add((cr, cc))
            cd = d2 if cd == d1 else d1 # Original logic for alternating
    return list(moves)

def get_diagonal_moves(board, pos, color): # Parameter 'color' is piece_color
    moves = []
    r_start, c_start = pos # Original r,c for start
    for dr, dc in DIRECTIONS['bishop']: # Original DIRECTIONS
        r, c = r_start, c_start # Original r,c for iteration
        while True:
            r += dr
            c += dc
            if not (0 <= r < ROWS and 0 <= c < COLS): # Original boundary check
                break
            piece_on_path = board[r][c] # Original piece variable name
            if piece_on_path:
                if piece_on_path.color != color: # Original color check
                    moves.append((r, c))
                break
            moves.append((r, c))
    return moves

def copy_board(board):
    return [[p.clone() if p else None for p in row] for row in board]

# -----------------------------
# Game Logic
# -----------------------------
def create_initial_board():
    board = [[None for _ in range(COLS)] for _ in range(ROWS)]
    # Using original direct assignment
    board[0][0] = Rook("black"); board[0][1] = Knight("black"); board[0][2] = Bishop("black")
    board[0][3] = Queen("black"); board[0][4] = King("black"); board[0][5] = Bishop("black")
    board[0][6] = Knight("black"); board[0][7] = Rook("black")
    for i in range(COLS): board[1][i] = Pawn("black")
    
    board[7][0] = Rook("white"); board[7][1] = Knight("white"); board[7][2] = Bishop("white")
    board[7][3] = Queen("white"); board[7][4] = King("white"); board[7][5] = Bishop("white")
    board[7][6] = Knight("white"); board[7][7] = Rook("white")
    for i in range(COLS): board[6][i] = Pawn("white")
    return board

def _find_king_pos(board, color): # Internal helper
    for r_idx in range(ROWS):
        for c_idx in range(COLS):
            piece = board[r_idx][c_idx]
            if isinstance(piece, King) and piece.color == color:
                return (r_idx, c_idx)
    return None

def has_legal_moves(board, color):
    for r in range(ROWS):
        for c in range(COLS):
            piece = board[r][c]
            if piece and piece.color == color:
                # Get potential moves and validate each one
                for move_end_pos in piece.get_valid_moves(board, (r, c)):
                    if validate_move(board, color, (r, c), move_end_pos):
                        return True
    return False

def check_game_over(board, turn_color_who_just_moved): # Parameter name changed for clarity
    """Check if game is over. Returns (reason_str/None, winner_color/None)."""
    # 1. King Capture (highest priority)
    if not _find_king_pos(board, "white"):
        return "king_capture", "black" # Black wins if white king missing
    if not _find_king_pos(board, "black"):
        return "king_capture", "white" # White wins if black king missing

    # 2. Checkmate or Stalemate for the *next* player
    next_player_color = "black" if turn_color_who_just_moved == "white" else "white"
    
    if not has_legal_moves(board, next_player_color):
        if is_in_check(board, next_player_color):
            # Checkmate: next player is in check and has no legal moves.
            # The player who just moved (turn_color_who_just_moved) is the winner.
            return "checkmate", turn_color_who_just_moved
        else:
            # Stalemate: next player not in check, but no legal moves.
            return "stalemate", None
    
    return None, None # Game not over

def check_evaporation(board):
    """Applies Knight evaporation. Original global check behavior."""
    # This iterates all knights. If a knight evaporates, another knight might then
    # evaporate due to the first one vanishing. This simple loop doesn't handle chains.
    # For true chaining, a more complex loop or recursive calls would be needed.
    # The current implementation matches the original: each Knight object on board checks once.
    knights_on_board = []
    for r in range(ROWS):
        for c in range(COLS):
            piece = board[r][c]
            if isinstance(piece, Knight):
                knights_on_board.append(((r,c), piece)) # Need piece for its evaporate method

    for pos, knight_instance in knights_on_board:
        # Ensure knight is still at 'pos' (could have been evaporated by an earlier knight in the list)
        if board[pos[0]][pos[1]] is knight_instance:
             knight_instance.evaporate(board, pos)

# manhattan_distance is not used by other functions here, can be removed if not used elsewhere.
# def manhattan_distance(pos1, pos2):
#     return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def is_in_check(board, color): # 'color' is the king's color
    king_pos = _find_king_pos(board, color)
    if not king_pos:
        # This implies king was captured or removed, which check_game_over handles.
        # For is_in_check purposes, if king is gone, not in check by this definition.
        return False 

    enemy_color = 'black' if color == 'white' else 'white'
    for r_attacker in range(ROWS):
        for c_attacker in range(COLS):
            piece = board[r_attacker][c_attacker]
            if piece and piece.color == enemy_color:
                # Standard attack: Can the enemy piece move to the king's square?
                if king_pos in piece.get_valid_moves(board, (r_attacker, c_attacker)):
                    return True
                
                # Variant-specific threat checks (Queen explosion, Knight evaporation)
                # These checks determine if the king_pos becomes "attacked" due to these mechanics.
                # Queen's explosion threat
                if isinstance(piece, Queen):
                    for queen_move_dest in piece.get_valid_moves(board, (r_attacker, c_attacker)):
                        # If queen captures at queen_move_dest, would king_pos be hit by explosion?
                        # Queen must capture for explosion. Target of capture must be opponent of queen.
                        target_of_capture = board[queen_move_dest[0]][queen_move_dest[1]]
                        if target_of_capture and target_of_capture.color == color: # Queen captures piece of king's color
                            if max(abs(queen_move_dest[0] - king_pos[0]), abs(queen_move_dest[1] - king_pos[1])) == 1:
                                return True # King is adjacent to an explosion caused by queen capturing king's color piece

                # Knight's evaporation threat
                if isinstance(piece, Knight):
                    for knight_move_dest in piece.get_valid_moves(board, (r_attacker, c_attacker)):
                        # If enemy knight moves to knight_move_dest, would king_pos be evaporated?
                        for dr_evap, dc_evap in DIRECTIONS['knight']:
                            evap_r, evap_c = knight_move_dest[0] + dr_evap, knight_move_dest[1] + dc_evap
                            if (evap_r, evap_c) == king_pos:
                                return True
    return False

# The following three functions are specific threat types from your original code.
# They are used by validate_move.
def is_in_explosion_threat(board, color): # 'color' is the potential victim's color
    king_pos = _find_king_pos(board, color)
    if not king_pos: return False

    enemy_color = 'black' if color == 'white' else 'white'
    for r_queen in range(ROWS):
        for c_queen in range(COLS):
            piece = board[r_queen][c_queen]
            if isinstance(piece, Queen) and piece.color == enemy_color:
                for queen_move_dest in piece.get_valid_moves(board, (r_queen, c_queen)):
                    # Queen must capture a piece of 'color' to be an explosion threat to 'color's king
                    target_of_capture = board[queen_move_dest[0]][queen_move_dest[1]]
                    if target_of_capture and target_of_capture.color == color:
                        if max(abs(queen_move_dest[0] - king_pos[0]), abs(queen_move_dest[1] - king_pos[1])) == 1:
                            return True
    return False

def is_king_attacked_by_knight(board, color, king_pos_to_check): # 'color' is king's color
    # Checks if king_pos_to_check is *currently* under L-shape attack by an enemy knight
    enemy_color = 'black' if color == 'white' else 'white'
    for r_knight in range(ROWS):
        for c_knight in range(COLS):
            piece = board[r_knight][c_knight]
            if isinstance(piece, Knight) and piece.color == enemy_color:
                for dr, dc in DIRECTIONS['knight']:
                    if (r_knight + dr, c_knight + dc) == king_pos_to_check:
                        return True
    return False

def is_king_in_knight_evaporation_danger(board, color): # 'color' is king's color
    king_pos = _find_king_pos(board, color)
    if not king_pos: return False

    enemy_color = 'black' if color == 'white' else 'white'
    for r_enemy_knight in range(ROWS):
        for c_enemy_knight in range(COLS):
            piece = board[r_enemy_knight][c_enemy_knight]
            if isinstance(piece, Knight) and piece.color == enemy_color:
                # For each move the enemy knight can make
                for knight_move_dest in piece.get_valid_moves(board, (r_enemy_knight, c_enemy_knight)):
                    # If knight moves to knight_move_dest, would king_pos be evaporated?
                    for dr_evap, dc_evap in DIRECTIONS['knight']:
                        evap_r, evap_c = knight_move_dest[0] + dr_evap, knight_move_dest[1] + dc_evap
                        if (evap_r, evap_c) == king_pos:
                            return True
    return False


def validate_move(board, color, start, end): # 'color' is the moving piece's color
    # 1. Check basic validity: piece exists, belongs to player.
    piece_to_move = board[start[0]][start[1]]
    if not piece_to_move or piece_to_move.color != color:
        return False

    # 2. Check if 'end' is in the piece's generated pseudo-legal moves.
    #    get_valid_moves should ensure 'end' is empty or has an opponent.
    if end not in piece_to_move.get_valid_moves(board, start):
        return False

    # 3. Simulate the move on a temporary board.
    simulated_board = copy_board(board)
    piece_on_sim_board = simulated_board[start[0]][start[1]] # Get the cloned piece
    
    # Execute the move, this will apply piece-specific effects (Queen explode, Pawn promote)
    # and Knight's own evaporate.
    piece_on_sim_board.move(simulated_board, start, end) 
    
    # After the piece's specific move, apply any global effects like evaporation from other knights
    # if that's the rule (original 'check_evaporation' was global).
    check_evaporation(simulated_board) # This will check all knights on sim_board

    # 4. Post-simulation checks:
    # 4a. Ensure the king of 'color' still exists (wasn't captured/evaporated by its own move's side effects)
    current_king_pos_sim = _find_king_pos(simulated_board, color)
    if not current_king_pos_sim:
        return False # King disappeared, illegal move

    # 4b. Check if the move leaves 'color's king in check.
    if is_in_check(simulated_board, color):
        return False

    # 4c. Original variant-specific threat checks on the simulated board state.
    # These check if the *resulting position* is immediately dangerous in a way not covered by standard check.
    if is_in_explosion_threat(simulated_board, color): # Checks if king of 'color' is now threatened by explosion
        return False
    if is_king_in_knight_evaporation_danger(simulated_board, color): # Checks if king of 'color' is now in evap danger
        return False
    
    # This checks if a King made the move AND landed on a square directly attacked by an enemy knight.
    # is_in_check should cover this, but kept for explicitness matching original.
    if isinstance(piece_to_move, King) and is_king_attacked_by_knight(simulated_board, color, end):
        return False

    return True


def generate_position_key(board, turn):
    key_parts = []
    for row in board:
        for piece in row:
            if piece:
                key_parts.append(piece.symbol())
                key_parts.append('1' if piece.has_moved else '0')
            else:
                key_parts.append('..')
    key_parts.append(turn[0]) # 'w' or 'b'
    return ''.join(key_parts)

# Original is_stalemate function (not directly used if check_game_over is comprehensive)
def is_stalemate(board, color): # 'color' is the player whose turn it is
    if is_in_check(board, color):
        return False # If in check and no legal moves, it's checkmate
    if not has_legal_moves(board, color):
        return True # Not in check, no legal moves
    return False