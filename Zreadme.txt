JUNGLE CHESS - OFFICIAL RULES

OBJECTIVE:
The aim of the game is to checkmate the opponent's king, just like in standard chess. You do not need to "evaporate" or "explode" the king to win, just put the opponent in a position where no matter what they do, they will lose on the next turn.
This means that, unlike normal chess, a 'stalemate' position is actually a win.
If your opponent has no legal moves on their turn, you win immediately. In that sense, Jungle Chess behaves like a king-capture variant but ends one move before the king would actually be taken, so the king is not actually captured.


PIECE CHANGES:

Queen:
  - Moves normally (any number of squares in any direction: horizontally, vertically, or diagonally).
  - Explodes when capturing a piece: When the queen takes an enemy piece, it explodes, removing all enemy pieces in the 8 surrounding squares. The queen itself is also removed in the explosion.
  - The queen does not explode if it is captured by the opponent.

Rook:
  - Moves normally (any number of squares horizontally or vertically).
  - Capturing pierces through pieces: In one turn, the rook can move to any square in its normal movement (file or rank) to any square regardless of enemy pieces in the way. This means that it can take any amount of pieces in a row or column at once, stopping when it wants smashing through everything in its path. It can only be stopped by friendly pieces.

Bishop:
  - The bishop has two movement options:
    1. Normal Diagonal Movement: Slides any number of squares diagonally, like a standard chess bishop. It cannot jump over pieces.
    2. Zig-Zag Movement: Moves in a zig-zag pattern, alternating between two diagonal directions.
  - The bishop cannot jump over pieces during its zig-zag movement. If it hits a piece, it stops and can capture that piece if it's an enemy.
  - The zig-zag movement can be used in any direction: forward, backward, or sideways. For example:
    * Forward Zig-Zag: Moves 1 square diagonally forward-right, then 1 square diagonally forward-left, and repeats this pattern.
    * Sideways Zig-Zag: Moves 1 square diagonally forward-right, then 1 square diagonally backward-right, and repeats.

Knight:
  - Moves normally (in an L-shape: 2 squares in one direction and 1 square perpendicular), but can only jump to empty squares.
  - Evaporates all squares it could move to: At all times, the knight evaporates (removes) all pieces on the squares it could move to except its own pieces. Because evaporation is a passive aura, the knight "captures" pieces simply by landing near them, rather than landing on them.
  - If two knights try to evaporate each other, both knights are removed, but the other pieces affected by their evaporation are still removed first.

Pawn:
  - Moves forward one square at a time (or two squares from the starting rank unless there is a piece directly in front of it).
  - Captures forward (by stepping onto the enemy piece) or sideways (but not diagonally). Can capture forward two squares on its first move if there is an enemy piece there, and there is no blocking piece directly in front of it.
  - Pawns promote to a queen on the last rank (non-optional).
  - Does not have en-passant.

King:
  - Slides one or two squares horizontally, vertically, or diagonally unless a piece is in the way.
  - The king can move through attacked squares during a 2-square move. Only the square it lands on must be safe. This is intentional and makes kings harder to checkmate.
  - The king is still subject to check and checkmate, and the game is won by checkmating the opponent's king.
  - No castling.


NOTATION AND CHECK RULES:

- Use standard chess notation for moves (e.g., Bxc3 for a Bishop capturing on c3) with commas between moves and + for check and # for checkmate.
- Exact SAN details do not matter. If a move is clear and readable, it is acceptable. Notation is not a gameplay rule.
- Definition of Check: If a move threatens to take the king on the next move, then it is check. This includes all special attacks:
  * Explosive Check: The King is standing within the potential blast radius of an allied piece that is under attack by an enemy Queen. (e.g., If the Queen attacks a pawn on f2, a King on e2 is in check because Qxf2 would explode and kill the King).
  * Evaporation Check: The King is standing on a square threatened by an enemy Knight (meaning the Knight can jump to an empty square whose evaporation aura would then kill the King).
  * Railgun Check: The King is anywhere in the line of sight of an enemy Rook, regardless of any intervening enemy pieces (since the Rook pierces).

- For knight evaporation captures, rook piercings, and queen explosions, list all the squares affected inside parentheses after the move. Casualties are sorted deterministically: lowest number (rank) first, then lowest letter (file) second.
  * Example: Nc6 (xe4 xg4)
  * Self-Targeting Note: Explicitly add the square the piece landed on to the list of casualties if it destroyed itself. Example: Ne5 (xe5 xc6 xf7)
  * Rook example: Rxh1 (xh2)
  * Queen example: Qxf7 (xf7 xg7 xh7 xg8)

- If the side to move has no legal moves, that is an immediate win for the other player.


GAME EXAMPLE:

1. e4, Nf6 (xe4)
2. Bxf7+, Kxf7
3. Nc3, Ne4 (xd2 xf2)


Have fun playing Jungle Chess!