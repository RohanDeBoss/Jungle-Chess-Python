# Jungle Chess — Rules

Jungle Chess is a chess variant where four of the six piece types have been
given volatile, area-of-effect abilities. This page covers the rules you
need to play. (Looking for engine/development notes instead? See
`ENGINE_NOTES.md`.)

---

## 1. Objective

Checkmate the opponent's king, as in standard chess. You do **not** need to
explode, evaporate, or pierce the enemy king to win — you win by putting
your opponent in a position where they have **no legal move** on their
turn. The game always ends exactly one ply before a king would actually be
destroyed, so a king is never literally removed from the board during valid
play.

---

## 2. Terminal Condition — No Legal Moves

Having no legal moves is **always** an immediate loss for the side to move,
regardless of whether that side is currently in check.

- No legal moves **while in check** → checkmate.
- No legal moves **while not in check** → this is extremely rare in
  practice (see [Section 5](#5-why-stalemate-almost-never-happens)) but is
  still a loss. Jungle Chess has no stalemate-as-draw rule.

If your opponent has no legal moves, you win immediately — functionally
identical to a king-capture variant, except the king is never actually
taken off the board.

- **Quiet Checkmates & King-Delivered Mates:** Because a terminal loss is
  defined strictly as "0 legal moves," a checkmate can be delivered by a
  quiet move (including a 2-square King step) that cuts off the opponent's
  final escape square.

---

## 3. Piece Changes

### Queen
- Moves normally: any number of squares, any direction.
- **Explodes on capture.** When the queen captures a piece, it explodes,
  removing all enemy pieces on the 8 squares surrounding the capture
  square. The queen itself is also removed in the blast.
- The queen does **not** explode if it is the one being captured.

### Rook
- Moves normally: any number of squares horizontally or vertically.
- **Piercing capture.** On a single move, a rook can travel to any square on
  its rank or file regardless of enemy pieces in the way, destroying every
  enemy piece it passes through. It is stopped only by a friendly piece.

### Bishop
- Two movement modes:
  1. **Normal diagonal** — slides any number of squares diagonally, cannot
     jump over pieces.
  2. **Zig-zag** — alternates between two diagonal directions each step
     (forward, backward, or sideways zig-zags are all legal). Like normal
     diagonal movement, it cannot jump over pieces — it stops (and may
     capture) at the first piece it meets along the zig-zag path.

### Knight
- Moves in the normal L-shape, but **can only land on empty squares.**
- **Passive evaporation.** At all times, a knight evaporates (removes)
  every enemy piece sitting on a square it could jump to — this happens
  simply by the knight existing there, not by landing on the target. This
  is why a knight can only land on empty squares: anything else would
  already have been evaporated.
- If two enemy knights are mutually within evaporation range and one of
  them moves so as to trigger the exchange, **both knights are removed**,
  but any other pieces caught in either evaporation zone are removed
  first.

### Pawn
- Moves forward one square (two from its starting rank, unless blocked).
- Captures by stepping forward onto an enemy piece, or sideways onto an
  adjacent enemy piece (never diagonally). May advance two squares onto an
  enemy piece on its first move if the intervening square is empty.
- Promotes to a queen on the last rank — mandatory, no choice of piece.
- No en passant.

### King
- Slides one **or two** squares horizontally, vertically, or diagonally,
  blocked by any piece in its path.
- On a 2-square move, only the **landing square** must be safe — the king
  is allowed to pass through an attacked square. This is intentional; it
  makes the king meaningfully harder to checkmate than in standard chess.
- Still subject to check and checkmate.
- No castling.

---

## 4. Check, Notation, and Casualty Lists

**Definition of check:** a position is check if the side to move's king
would be destroyed by *any* legal move available to the opponent on their
next turn, including all of the special attacks below.

- **Explosive check** — your king sits adjacent to a piece (yours or
  theirs) that an enemy queen could capture, because the resulting
  explosion would reach your king. Example: an enemy queen attacks a pawn
  on f2; a king on e2 is in check, because Qxf2 would explode and kill it.
- **Evaporation check** — your king sits on a square an enemy knight could
  jump to (even though the knight can only land on *empty* squares — it's
  the landing threat, not an actual capture, that creates the check).
- **Railgun check** — your king shares a rank or file with an enemy rook,
  at any distance, regardless of any pieces in between (because the rook
  pierces through everything but a friendly blocker).

Standard algebraic notation is used, with a comma separating each half-move
and `+`/`#` for check/checkmate. Exact SAN formatting isn't a gameplay rule
— any move that's clear and unambiguous is acceptable.

**Casualty lists.** For any move that pierces, evaporates, or explodes
multiple squares, list every affected square in parentheses after the
move, sorted by lowest rank first, then lowest file:

```
Nc6 (xe4 xg4)          — knight evaporation
Ne5 (xe5 xc6 xf7)      — knight evaporation that also kills itself (self-target listed explicitly)
Rxh1 (xh2)             — rook piercing
Qxf7 (xf7 xg7 xh7 xg8) — queen explosion (queen's own square is included)
```

**Example game:**
```
1. e4, Nf6 (xe4)
2. Bxf7+, Kxf7
3. Nc3, Ne4 (xd2 xf2)
```

---

## 5. Why Stalemate Almost Never Happens

Because no-legal-moves is always a loss, you might worry about accidentally
running yourself out of moves. In practice this is rarely a concern:

- **Pawns can't lock a position.** A pawn captures forward, so a blocked
  pawn chain can simply step onto the blocker — chains can never freeze the
  board the way they sometimes threaten to in standard chess.
- **The king phases through attacks.** A 2-square king move only needs the
  *landing* square to be safe, so boxing a king in without ever checking it
  requires an unusually large, deliberate cage of pieces.
- **The tablebase owns the endgame.** With 5 pieces or fewer on the board,
  positions are resolved exactly by a precomputed tablebase — no
  heuristics, no guesswork.

---

## 6. Reading the Evaluation Bar

The evaluation bar and score (shown above the board, and in the engine-line
panel) always represent **White's** advantage, no matter which side of the
board is shown at the bottom of your screen:

- A bar/score to the **right of center** (positive number) means **White**
  is doing better.
- A bar/score to the **left of center** (negative number) means **Black**
  is doing better.

This is deliberate and does **not** flip when you use "Flip View" or when
you're playing as Black — the bar always reads the same way a printed
newspaper diagram would, regardless of which side you're sitting on.