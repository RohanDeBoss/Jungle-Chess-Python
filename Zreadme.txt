# Jungle Chess — Official Rules & Engine Notes

Jungle Chess is a chess variant where four of the six piece types have been
given volatile, area-of-effect abilities. This document is the single source
of truth for both **players** (rules) and **AI assistants / engine
developers** (implementation invariants). If you are an AI reading this
before touching the codebase, read the whole thing — the "Engine Invariants"
section at the end exists specifically to stop you from re-deriving (and
getting wrong) an argument that has already been settled.

---

## 1. Objective

Checkmate the opponent's king, as in standard chess. You do **not** need to
explode, evaporate, or pierce the enemy king to win — you win by putting your
opponent in a position where they have **no legal move** on their turn. The
game always ends exactly one ply before a king would actually be destroyed,
so a king is never literally removed from the board during valid play. See
[Section 6](#6-engine-invariants-read-this-before-touching-the-code) for why
this is true by construction, not just by convention.

---

## 2. Terminal Condition — No Legal Moves

Having no legal moves is **always** an immediate loss for the side to move,
regardless of whether that side is currently in check.

- No legal moves **while in check** → checkmate.
- No legal moves **while not in check** → this is extremely rare in
  practice (see [Section 5](#5-why-stalemate-almost-never-happens)) but is
  still a loss. Jungle Chess has no stalemate-as-draw rule.

If your opponent has no legal moves, you win immediately. The game therefore
behaves like a king-capture variant that stops one ply early — functionally
identical in outcome, but the king is never actually taken off the board.

---

## 3. Piece Changes

### Queen
- Moves normally: any number of squares, any direction.
- **Explodes on capture.** When the queen captures a piece, it explodes,
  removing all enemy pieces on the 8 squares surrounding the capture square.
  The queen itself is also removed in the blast.
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
- **Passive evaporation.** At all times, a knight evaporates (removes) every
  enemy piece sitting on a square it could jump to — this happens simply by
  the knight existing there, not by landing on the target. This is why a
  knight can only land on empty squares: anything else would already have
  been evaporated.
- If two enemy knights are mutually within evaporation range and one of them
  moves so as to trigger the exchange, **both knights are removed**, but any
  other pieces caught in either evaporation zone are removed first.

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
- On a 2-square move, only the **landing square** must be safe — the king is
  allowed to pass through an attacked square. This is intentional; it makes
  the king meaningfully harder to checkmate than in standard chess.
- Still subject to check and checkmate.
- No castling.

---

## 4. Check, Notation, and Casualty Lists

**Definition of check:** a position is check if the side to move's king would
be destroyed by *any* legal move available to the opponent on their next
turn, including all of the special attacks below.

- **Explosive check** — your king sits adjacent to a piece (yours or theirs)
  that an enemy queen could capture, because the resulting explosion would
  reach your king. Example: an enemy queen attacks a pawn on f2; a king on e2
  is in check, because Qxf2 would explode and kill it.
- **Evaporation check** — your king sits on a square an enemy knight could
  jump to (even though the knight can only land on *empty* squares — it's
  the landing threat, not an actual capture, that creates the check).
- **Railgun check** — your king shares a rank or file with an enemy rook, at
  any distance, regardless of any pieces in between (because the rook
  pierces through everything but a friendly blocker).

Standard algebraic notation is used, with a comma separating each half-move
and `+`/`#` for check/checkmate. Exact SAN formatting isn't a gameplay rule —
any move that's clear and unambiguous is acceptable.

**Casualty lists.** For any move that pierces, evaporates, or explodes
multiple squares, list every affected square in parentheses after the move,
sorted by lowest rank first, then lowest file:

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

Because no-legal-moves is always a loss, it's tempting to worry about the
engine accidentally "missing" a stalemate at the search horizon. In practice
this concern doesn't apply:

- **Pawns can't lock a position.** A pawn captures forward, so a blocked pawn
  chain can simply step onto the blocker — chains can never freeze the board
  the way they sometimes threaten to in standard chess.
- **The king phases through attacks.** A 2-square king move only needs the
  *landing* square to be safe, so boxing a king in without ever checking it
  requires an unusually large, deliberate cage of pieces.
- **The tablebase owns the endgame.** Non-check stalemates are essentially an
  endgame-only phenomenon (a near-empty, fully paralyzed board), and the
  precomputed tablebase (5 pieces or fewer) resolves those positions exactly,
  with no heuristics needed.

---

## 6. Engine Invariants (read this before touching the code)

This section exists because this exact question — *"couldn't the search
actually capture a king, and does the tablebase need to account for
unreachable positions?"* — has come up before, including from AI assistants
reviewing this codebase, and it deserves a definitive answer so it doesn't
get re-litigated (badly) every time.

### 6.1 The core invariant

**On every position the engine ever searches, the side *not* to move is
never in check.**

This holds for three independent reasons, all of which must remain true:

1. **Legal self-play enforces it structurally.** `generate_legal_moves_generator`
   in `GameLogic.py` filters out any candidate move that leaves the mover's
   own king in check. A position reached by legal play can therefore never
   have the side-not-to-move in check — if it were, that side would have had
   no legal move available to escape it, and the game would already have
   ended one ply earlier as a loss.
2. **PGN replay reuses the same legal-move generator.** `load_pgn_from_entry`
   matches typed notation against `get_all_legal_moves` output, so replayed
   games inherit the same guarantee as live play.
3. **FEN loading is explicitly validated.** `load_fen_from_entry` rejects any
   position where the side not to move is in check:
   ```python
   passive_color = "black" if self.turn == "white" else "white"
   if is_in_check(self.board, passive_color):
       messagebox.showerror("Invalid FEN", "Illegal Position: the side not "
                             "to move is already in check/danger.")
       self.reset_game(schedule_ai=False)
       return
   ```
   Before this check existed, a hand-typed FEN was the *only* way to hand the
   engine a position that violated the invariant. With it in place, all three
   entry points (self-play, PGN, FEN) now agree.

### 6.2 Why the invariant makes king-capture code unreachable

`is_square_attacked` in `GameLogic.py` computes "check" by directly
projecting each volatile piece's *kill capability*, not just its normal move
squares:

- A queen's explosive threat is checked by testing whether it can already
  capture something adjacent to the king.
- A rook's railgun threat is checked by scanning through every piece on the
  king's rank/file, since piercing ignores blockers.
- A knight's evaporation threat is checked with a **second-order lookup** —
  for every empty square the knight could jump to, it also checks whether
  *that* square's jump-set includes the king. This is what correctly flags
  "the knight could move to an empty square and evaporate the king from
  there" as check *right now*, before the knight has moved anywhere.

Because of this, **any geometric arrangement capable of killing a king next
turn is already classified as check on the current turn.** There's no piece
type in this ruleset whose kill-capability is created by the same move that
executes the kill — the capability (sightline, shared file, jump-adjacency)
always pre-exists the move. Combined with the invariant in §6.1, this means:

> If it is White's turn, Black's king cannot be in a position where any
> White move would destroy it — because that would mean Black started their
> turn in check with no way to escape, which is a terminal loss that would
> have already ended the game.

**Practical consequence:** code paths in `AI.py`/`OPAI.py` that check "did my
move just remove the opponent's king" (`find_king_pos(...) is None`,
`not sim.find_king_pos(...)`, etc.) are checking for a condition that can
never actually occur during normal search on a reachable position. They are
not wrong to have — they're a cheap fast-path stand-in for the more
expensive "did I deliver checkmate" computation, and they're harmless to
leave in place — but removing them (as done in `AI.py` v118) is a safe,
verified simplification, not a behavioral change. `AI.py` v117/v118 and
`OPAI.py` are provably equivalent in every search decision on any position
the UI will ever hand them.

### 6.3 Where this *used* to be false, and why it isn't anymore

Before the FEN-legality check in §6.1 existed, a hand-crafted FEN could
violate the invariant (e.g. loading a position where Black's king is already
in an unescapable evaporation threat, but it's recorded as White's move).
That was the one gap where the "unreachable" argument didn't hold — a
custom-loaded, non-self-play position could theoretically make the
otherwise-dead king-capture code paths live for one move. That gap is now
closed at the UI layer. If FEN loading is ever refactored, **this check must
be preserved** or the invariant silently breaks again.

### 6.4 Tablebases are unaffected by any of this

The tablebase generator (`TablebaseGenerator.py`) enforces the same
own-king-first ordering independently, in every transition worker (3-man,
4-man same-side, 4-man cross, 5-man same-side, 5-man cross) — e.g.:
```python
if is_in_check(board, 'white'): board.unmake_move(record); continue   # legality first
...
if not bkp or not has_legal_moves(board, 'black'):
    immediate_win = True; ...                                        # mate check second
```
Tablebase files legitimately contain many positions that are *unreachable
from the starting position of a real game* — that's normal and expected for
any tablebase, chess or otherwise. What matters is that every stored
position is **legal in isolation** (side not to move isn't in check, kings
aren't overlapping/adjacent-illegally, pawns aren't on illegal ranks), which
the generator guarantees independently of anything the live UI does. The
tablebase was never at risk from the FEN-loading gap described in §6.3.

### 6.5 Performance-sensitive rules

- **Stalemate is a loss, so don't add expensive legal-move checks to
  `qsearch`.** The engine already starves the opponent naturally at the
  horizon via static eval; there's no correctness gap to patch. The one
  `has_legal_moves()` call that does exist in `qsearch` is safe to keep,
  specifically *because* `has_legal_moves` short-circuits on the first legal
  move found — it does not enumerate all moves, so it's effectively O(1) in
  any position that isn't a genuine dead end.
- **Keep pruning conservative.** A single AoE knight or queen move can swing
  material by 3000+ points. Standard-chess-tuned LMR/futility margins will
  blind the engine to these tactics. Don't tighten pruning margins without
  re-testing against `OPAI.py`.
- **Don't hand-roll 5-piece endgame heuristics.** The tablebase already
  solves these exactly; heuristic code would only risk disagreeing with it.

### 6.6 `OPAI.py` is a frozen baseline — do not edit its search/eval logic

`OPAI.py` exists solely as a stable comparison target for measuring whether
changes to `AI.py` are actual improvements (via the "AI vs OP Series" mode)
rather than illusory ones. Its search logic, evaluation function, and
pruning constants must **never** be back-ported from `AI.py`, even for
"harmless" cleanups — a frozen baseline is only useful if it stays bit-for-
bit stable across comparison runs.

The one category of exception is shared, non-heuristic infrastructure that
both files import or duplicate for boilerplate reasons (e.g. constructor
argument handling in a standalone dispatch function) — those may be aligned
for consistency **only if** doing so cannot change OPAI's actual move
selection or evaluation output. When in doubt, don't touch `OPAI.py`.

---

## 7. Quick Reference for AI Assistants

If you're reviewing or modifying this codebase, before proposing a change:

1. Read §6 in full. If your review is about to raise "the king could get
   captured" or "the tablebase might miss unreachable positions," it's
   already answered above — check whether the FEN-legality guard (§6.1,
   point 3) is intact before re-opening either question.
2. Never edit `OPAI.py`'s search or evaluation logic (§6.6).
3. Don't propose adding legal-move enumeration to `qsearch` — it's already
   there in the one place it's needed, and it's cheap (§6.5).
4. Don't propose tightening pruning margins without flagging that it needs
   an AI-vs-OP regression run — Jungle Chess's swing sizes break assumptions
   pruning margins are normally tuned against.