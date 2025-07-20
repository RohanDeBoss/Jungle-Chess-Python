# OpponentAI.py

import threading
from AI import ChessBot, SearchCancelledException
from GameLogic import *

# OpponentAI is now a subclass of ChessBot.
# This means it automatically gets all the methods like negamax, qsearch, etc.,
# including all the cancellation logic you already implemented in ChessBot.
class OpponentAI(ChessBot):
    
    # We only need to provide a new __init__ method to accept the
    # cancellation_event and pass it up to the parent ChessBot class.
    def __init__(self, board, color, app, cancellation_event):
        
        # The 'super().__init__()' call runs the __init__ method of the parent class (ChessBot).
        # This correctly sets up the opponent bot just like the main bot.
        super().__init__(board, color, app, cancellation_event)
        
        print(f"OpponentAI ({self.color}) initialized.")

    # That's it! You don't need to redefine make_move, negamax, or anything else.
    # It inherits everything perfectly from ChessBot.
    # If you later want the OpponentAI to have a different personality
    # (e.g., different piece values or a more aggressive evaluation),
    # you can override just the 'evaluate_board' method here. For now, it's an exact clone.