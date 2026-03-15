This is a chess variant of mind called Jungle Chess. The rules are the same as normal chess except for the following changes:

Objective:
The aim of the game is to checkmate the opponent's king, just like in standard chess. You do not need to "evaporate" or "explode" the king to win, just put the opponent in a position where no matter what they do, they will lose on the next turn.
If your opponent has no legal moves on their turn, you win immediately. In that sense, Jungle Chess behaves like a king-capture variant that ends one move before the king would actually be taken.

Piece Changes:
1. Queen:
   o Moves normally (any number of squares in any direction: horizontally, vertically, or diagonally).
   o Explodes when capturing a piece: When the queen takes an enemy piece, it explodes, removing all enemy pieces in the 8 surrounding squares. The queen itself is also removed in the explosion.
   o The queen does not explode if it is captured by the opponent.

2. Rook:
   o Moves normally (any number of squares horizontally or vertically).
   o Capturing pierces through pieces:
     In one turn, the rook can move to any square in its normal movement (file or rank) to any square regardless of enemy pieces in the way, this means that it can take any amount of pieces in a row or column at once, stopping when it wants smashing through everything in its path. It can only be stopped by friendly pieces.

3. Bishop:
   o The bishop has two movement options:
     - Normal Diagonal Movement: Slides any number of squares diagonally, like a standard chess bishop. It cannot jump over pieces.
     - Zig-Zag Movement: Moves in a zig-zag pattern, alternating between two diagonal directions. For example:
       * Forward Zig-Zag: Moves 1 square diagonally forward-right, then 1 square diagonally forward-left, and repeats this pattern until it reaches the edge of the board or hits a piece. (Can do the same but starting the zig zag going left). (The same pattern works going backwards)
       * Sideways Zig-Zag: Moves 1 square diagonally forward-right, then 1 square diagonally backward-right, and repeats. (Can do the same but starting the zig zag going down-right). (Or the same pattern to the left).
   o The bishop cannot jump over pieces during its zig-zag movement. If it hits a piece, it stops and can capture that piece if it's an enemy.
   o The zig-zag movement can be used in any direction: forward, backward, or sideways.

4. Knight:
   o Moves normally (in an L-shape: 2 squares in one direction and 1 square perpendicular).
   o Evaporates all squares it could move to: At all times, the knight evaporates (removes) all pieces on the squares it could move to except its own pieces.
   o If two knights try to evaporate each other, both knights are removed, but the other pieces affected by their evaporation are still removed first.

5. Pawn:
   o Moves forward one square at a time (or two squares on its first move).
   o Captures forward or sideways (but not diagonally). Can capture forward two squares on their first move if there's an enemy piece there, and there is not blocking piece directly in front of it.
   o Pawns promote to a queen on the last rank (non optional).
   o Does not have en-passant.

6. King:
   o Can move two squares in any direction (instead of one).
   o The king can move through attacked squares during a 2-square move. Only the square it lands on must be safe. This is intentional and makes kings harder to checkmate.
   o The king is still subject to check and checkmate, and the game is won by checkmating the opponent's king.
   o No castling.

Extra Rules:
* Knight Evaporation Interaction: If two knights try to evaporate each other, both knights are removed, but the other pieces affected by their evaporation are still removed first, since the passive knight evaporation happens last.
* Explosion Rules: When the queen explodes after capturing a piece, the explosion affects all 8 surrounding squares, removing any enemy pieces in those squares. The queen itself is also removed.

Notation & Check Rules:
* Use standard chess notation for moves (e.g., Bxc3 for a Bishop capturing on c3) with commas between moves and + for check and # for checkmate.
* Exact SAN/FIDE-perfect notation details do not matter in Jungle Chess. If a move is clear and readable, it is acceptable. Extra disambiguation is fine and notation should not be treated as a gameplay rule.
* Definition of Check: If a move threatens to take the king on the next move, then it is check. This includes all special attacks:
  - Explosive Check: The King is standing within the potential blast radius of an allied piece that is under attack by an enemy Queen. (e.g., If the Queen attacks a pawn on f2, a King on e2 is in check because Qxf2 would explode and kill the King).
  - Evaporation Check: The King is standing on a square threatened by an enemy Knight.
  - Railgun Check: The King is anywhere in the line of sight of an enemy Rook, regardless of any intervening enemy pieces (since the Rook pierces).
* For knight evaporation captures, rook piercings, and queen explosions, list all the squares affected inside parentheses after the move. Capture list ordered from lowest number first, then lowest letter second for determinism.
  - Example: Nc6 (xe4 xg4)
  - Self-Targeting Note: Explicitly add the square the piece landed on to the list of casualties if it destroyed itself. Example: Ne5 (xc6 xf7 xe5)
  - Rook example: Rxh1 (xh2)
  - Queen example: Qxf7 (xg7 xg8 xf7)
* If the side to move has no legal moves, that is an immediate win for the other player, even if you want to think of the position as a trap rather than traditional checkmate.
* Game example:
  1. e4, Nf6 (xe4)
  2. Bxf7+, Kxf7
  3. Nc3, Ne4 (xd2 xf2)

Thats all! And have fun playing Jungle chess!
