# Jungle Chess — Engine & Development Notes

This document is the single source of truth for **engine implementation
invariants**. It's aimed at developers and AI assistants working on the
codebase, not players — for the rules themselves, see `Zreadme.txt` (the
file shown in-app via the "ⓘ" button).

If you are an AI reading this before touching the codebase, read the whole
thing. This exists specifically to stop you from re-deriving (and getting
wrong) arguments that have already been settled — several of the points
below have been independently raised, and resolved, more than once.

---

## 1. Evaluation Sign Convention (read this before touching eval/UI code)

**The evaluation the UI displays is always White-relative ("absolute"),
never mover-relative, and it does not flip with board orientation.**

This is intentional, not a bug — it's literally the subject of the
`JungleChessUI.py` v17.2 changelog entry ("Fix rare visual eval flipping
bug"). The chain of responsibility:

- `AI.py`'s `ChessBot._report_eval(score, depth)` converts whatever score
  the search just produced (which is mover-relative internally) into
  White's perspective before it ever reaches the UI:
  ```python
  self.comm_queue.put(('eval', score if self.color == 'white' else -score, depth))
  ```
- `EnhancedChessApp.draw_eval_bar()` in `JungleChessUI.py` draws that value
  directly. It does **not** consult `self.board_orientation` or `self.turn`
  to decide whether to invert the bar.
- The same White-relative convention applies to the PV eval shown in the
  engine-lines panel (`ui_eval = report_score if self.color == 'white' else
  -report_score` in `AI.py`'s `make_move`/`ponder_indefinitely`).

**Do not "fix" this to flip with `board_orientation` or with whichever side
is on move.** That would reintroduce the exact bug v17.2 fixed. If a future
change wants a mover-relative bar as an *option*, it needs to be a distinct,
explicitly-labeled toggle — not a silent change to the default behavior,
since the default is depended on by `Section 6` below and by anyone reading
the bar as "White's chances," full stop.

Note the asymmetry with `TablebaseManager.probe()`, which independently
documents (see Section 6.4 below) that it also returns White-relative
scores — this is consistent with the UI/AI convention above, not a
coincidence.

---

## 2. Engine Invariants

This section exists because this exact question — *"couldn't the search
actually capture a king, and does the tablebase need to account for
unreachable positions?"* — has come up before, including from AI assistants
reviewing this codebase, and it deserves a definitive answer so it doesn't
get re-litigated (badly) every time.

### 2.1 The core invariant

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

### 2.2 Why the invariant makes king-capture code unreachable

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
always pre-exists the move. Combined with the invariant in §2.1, this means:

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

### 2.3 Where this *used* to be false, and why it isn't anymore

Before the FEN-legality check in §2.1 existed, a hand-crafted FEN could
violate the invariant (e.g. loading a position where Black's king is already
in an unescapable evaporation threat, but it's recorded as White's move).
That was the one gap where the "unreachable" argument didn't hold — a
custom-loaded, non-self-play position could theoretically make the
otherwise-dead king-capture code paths live for one move. That gap is now
closed at the UI layer. If FEN loading is ever refactored, **this check must
be preserved** or the invariant silently breaks again.

### 2.4 Tablebases are unaffected by any of this

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
tablebase was never at risk from the FEN-loading gap described in §2.3.

- **Canonical Storage and Ghost Mirrors:** The 16-bit tablebase files ONLY store 
  the canonical representation of a position (where the White King is reflected 
  into the bottom-left a1-d4 triangle). Symmetrical "mirror" indices are left 
  un-evaluated as `0`. Any probing logic or viewer MUST translate a position into 
  its canonical tuple before querying the array, otherwise it will falsely read 
  un-evaluated "ghost mirrors" as Draws (0).

- **Evaluation Perspective Contract:** Probing routines (`tb_manager.probe()`) return 
  absolute evaluations from White's perspective (positive = White winning, negative = 
  Black winning) — the same convention documented in Section 1 for the UI/AI eval
  reporting. Any reporting or search pipeline expecting mover-relative scores 
  must convert Black-to-move results accordingly before passing them to UI dispatchers.

### 2.5 Performance-sensitive rules

- **Stalemate is a loss, so don't add expensive legal-move checks to
  `qsearch`.** The engine already starves the opponent naturally at the
  horizon via static eval; there's no correctness gap to patch. The one
  `has_legal_moves()` call that does exist in `qsearch` is safe to keep,
  specifically *because* `has_legal_moves` short-circuits on the first legal
  move found — it does not enumerate all moves, so it's effectively O(1) in
  any position that isn't a genuine dead end.

- **Keep pruning conservative.** A single AoE knight or queen move can swing
  material by 3000+ points. Standard-chess-tuned LMR/futility margins will
  blind the engine to these tactics. Additionally, because a quiet King move 
  can close a mating net and end the game, **Late Move Reduction (LMR) must 
  be artificially suppressed in the endgame** (e.g., when `total_pieces <= 6`). 
  Don't tighten pruning margins without re-testing against `OPAI.py`.

- **Don't hand-roll 5-piece endgame heuristics.** The tablebase already
  solves these exactly; heuristic code would only risk disagreeing with it.

- **Use Tuple Iteration over Range Indexing.** In `GameLogic.py`, raycasts use 
  precomputed tuples of tuples (e.g., `for ray in RAYS_ORTHOGONAL[sq]:`). Do NOT 
  replace this with `for i in range(4): ray = RAYS_ORTHOGONAL[sq][i]`. CPython 
  iterates over tuples at native C-speed; forcing integer assignment and double 
  index lookups in the hottest loop of the engine causes a measurable KNPS drop.

### 2.6 `OPAI.py` is a frozen baseline — its *behavior* is frozen, not its source layout

`OPAI.py` exists solely as a stable comparison target for measuring whether
changes to `AI.py` are actual improvements (via the "AI vs OP Series" mode)
rather than illusory ones. What must stay frozen is its **output**: move
selection, evaluation scores, and search behavior on any given position must
never change as a side effect of a refactor. This means:

- **Functional changes are forbidden.** Never back-port `AI.py`'s search
  logic, evaluation function, or pruning constants into `OPAI.py`, even for
  "harmless" cleanups — a frozen baseline is only useful if its move choices
  stay bit-for-bit stable across comparison runs.
- **Non-functional changes are fine — including in `OPAI.py`.** Moving code
  verbatim to eliminate duplication (e.g. extracting SAN formatting, PV
  reconstruction, or tablebase move selection into a shared helper in
  `EngineRuntime.py` and calling it from both files) is explicitly allowed
  and encouraged, *provided* the moved code is unmodified and produces
  identical output to what it replaced. If a change couldn't be caught by
  running `OPAI.py` against itself before/after and diffing every move of
  every game, it's non-functional and fine. If it could, it isn't.
- **Tunable strategy is never shared, even between two files that currently
  agree.** Time-allocation formulas (`_search_time_budget` and its `TIME_*`
  constants), pruning margins, and similar tunables live independently in
  each bot's class body — never in `EngineRuntime.py` — specifically so that
  retuning one bot later can't silently retune the other by way of a shared
  function. `AI.py` and `OPAI.py` currently duplicate this formula with
  identical values; that's expected, not a sign it should be merged.

Shared backend plumbing lives in `EngineRuntime.py`. It owns Zobrist
hashing, FEN/opening-book helpers, worker dispatch, tablebase enable/disable
plumbing, time-check *mask* budgeting (not the time-allocation formula —
see above), bot lifecycle updates such as new-game cache resets, and pure
reporting/PV/tablebase-move plumbing (`format_bot_move`, `get_pv_data`,
`get_best_tablebase_move_with_eval`, `report_root_tb_solution`,
`get_root_tb_eval_relative`) along with the shared `TTEntry`/`TT_FLAG_*`
transposition-table record format. Each worker still holds exactly one
separate bot instance, so `AI.py` and `OpponentAI.py` do not share a
transposition table, eval cache, history table, cancellation event, or
tablebase manager — only the code that operates on them is shared, never the
data itself.

### 2.7 Search-side twofold repetition policy

While the UI and actual game rules strictly require a 3-fold repetition to 
declare a draw, the engine's internal search (`AI.py`) intentionally uses a 
stricter **2-fold repetition policy**. Inside `negamax` and `qsearch`, if a 
position has occurred even *once* before (either in the real game history or 
the current hypothetical search path), it is instantly scored as a draw. 
This is a vital search heuristic: it forces winning bots to seek progress 
rather than shuffling, helps losing bots find swindles, and prunes massive 
subtrees a full ply early. **Do not "fix" the search to wait for a 3rd occurrence.**

---

## 3. Quick Reference for AI Assistants

If you're reviewing or modifying this codebase, before proposing a change:

1. Read Section 2 in full. If your review is about to raise "the king could
   get captured" or "the tablebase might miss unreachable positions," it's
   already answered above — check whether the FEN-legality guard (§2.1,
   point 3) is intact before re-opening either question.
2. Read Section 1 before touching anything eval/UI-related. If your review
   is about to raise "the eval bar doesn't flip when Black is on move / when
   the board is flipped," that's the documented, intentional behavior, not
   a bug.
3. Never edit `OPAI.py`'s search or evaluation logic (§2.6).
4. Don't propose adding legal-move enumeration to `qsearch` — it's already
   there in the one place it's needed, and it's cheap (§2.5).
5. Don't propose tightening pruning margins without flagging that it needs
   an AI-vs-OP regression run — Jungle Chess's swing sizes break assumptions
   pruning margins are normally tuned against. Tests shown to lose elo: LMP.