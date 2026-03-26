# TablebaseManager.py (v7.0 - 5-Man Ready)

import os
import numpy as np
from GameLogic import King, Board
from itertools import combinations, combinations_with_replacement

def _flip(pos):
    return (7 - pos[0], pos[1])

SYMMETRY_MAP = [[0]*8 for _ in range(64)]
for r in range(8):
    for c in range(8):
        sq = r * 8 + c
        SYMMETRY_MAP[sq][0] = r * 8 + c
        SYMMETRY_MAP[sq][1] = r * 8 + (7 - c)
        SYMMETRY_MAP[sq][2] = (7 - r) * 8 + c
        SYMMETRY_MAP[sq][3] = (7 - r) * 8 + (7 - c)
        SYMMETRY_MAP[sq][4] = c * 8 + r
        SYMMETRY_MAP[sq][5] = c * 8 + (7 - r)
        SYMMETRY_MAP[sq][6] = (7 - c) * 8 + r
        SYMMETRY_MAP[sq][7] = (7 - c) * 8 + (7 - r)

PAWN_VALID_T = [0 if (sq % 8) <= 3 else 1 for sq in range(64)]
PAWN_WK_SQUARES = [r*8+c for r in range(8) for c in range(4)]
PAWN_WK_IDX = {sq: i for i, sq in enumerate(PAWN_WK_SQUARES)}

NON_PAWN_WK_SQUARES = []
for c in range(4):
    for r in range(c + 1):
        NON_PAWN_WK_SQUARES.append(r * 8 + c)
NON_PAWN_WK_IDX = {sq: i for i, sq in enumerate(NON_PAWN_WK_SQUARES)}

NON_PAWN_VALID_TS = [[] for _ in range(64)]
for sq in range(64):
    for t in range(8):
        if SYMMETRY_MAP[sq][t] in NON_PAWN_WK_IDX:
            NON_PAWN_VALID_TS[sq].append(t)


class TablebaseManager:
    def __init__(self):
        self.tables = {}
        self.tb_dir = "tablebases"
        self.pre_load_all()

    def pre_load_all(self):
        if not os.path.exists(self.tb_dir): return

        pieces = ['Queen', 'Rook', 'Knight', 'Bishop', 'Pawn']

        # 3-Man
        for p in pieces:
            self.load_table(f"K_{p}_K_sml")

        # 4-Man same-side
        for p1, p2 in combinations_with_replacement(pieces, 2):
            names = sorted([p1, p2])
            self.load_table(f"K_{names[0]}_{names[1]}_K_sml")

        # 4-Man cross
        for i in range(len(pieces)):
            for j in range(i, len(pieces)):
                self.load_table(f"K_{pieces[i]}_vs_{pieces[j]}_K_sml")

        # 5-Man same-side
        for p1, p2, p3 in combinations_with_replacement(pieces, 3):
            names = sorted([p1, p2, p3])
            self.load_table(f"K_{names[0]}_{names[1]}_{names[2]}_K_sml")

        # 5-Man cross (2 white vs 1 black)
        for p1, p2 in combinations_with_replacement(pieces, 2):
            names_w = sorted([p1, p2])
            for p3 in pieces:
                self.load_table(f"K_{names_w[0]}_{names_w[1]}_vs_{p3}_K_sml")

    def load_table(self, name):
        if name in self.tables: return True
        filename = os.path.join(self.tb_dir, f"{name}.bin")
        if os.path.exists(filename):
            try:
                has_pawn = "Pawn" in name
                wk_size = 32 if has_pawn else 10

                parts = name.split('_')
                # num_pieces = number of non-king pieces total
                # parts layout: K [pieces...] (vs [pieces...]) K sml
                # subtract 3 for the two K tokens and sml, minus 1 more if vs present
                num_pieces = len(parts) - 3
                if 'vs' in parts:
                    num_pieces -= 1

                # Shape: [wk_size, 64, 64, ...(num_pieces+1 times total for pieces+bk), 2]
                # FIX: was (num_pieces - 1), correct is (num_pieces + 1)
                shape = tuple([wk_size] + [64] * (num_pieces + 1) + [2])
                self.tables[name] = np.memmap(filename, dtype=np.int16, mode='r', shape=shape)
                return True
            except Exception as e:
                print(f"[TablebaseManager] Failed to memmap {name}: {e}")
        return False

    def _tb_score_to_ai_score(self, tb_val, is_win_for_white):
        if tb_val == 0: return 0
        score = 1000000 - abs(int(tb_val))
        return score if is_win_for_white else -score

    @staticmethod
    def _canonical_tuple_3(wk, p1, bk, turn, has_pawn):
        if has_pawn:
            t = PAWN_VALID_T[wk]
            return (PAWN_WK_IDX[SYMMETRY_MAP[wk][t]], SYMMETRY_MAP[p1][t], SYMMETRY_MAP[bk][t], turn)
        else:
            best = None
            for t in NON_PAWN_VALID_TS[wk]:
                m = (NON_PAWN_WK_IDX[SYMMETRY_MAP[wk][t]], SYMMETRY_MAP[p1][t], SYMMETRY_MAP[bk][t])
                if best is None or m < best: best = m
            return (best[0], best[1], best[2], turn)

    @staticmethod
    def _canonical_tuple_4(wk, p1, p2, bk, turn, has_pawn, same_piece):
        if has_pawn:
            t = PAWN_VALID_T[wk]
            m_p1, m_p2 = SYMMETRY_MAP[p1][t], SYMMETRY_MAP[p2][t]
            if same_piece and m_p1 > m_p2: m_p1, m_p2 = m_p2, m_p1
            return (PAWN_WK_IDX[SYMMETRY_MAP[wk][t]], m_p1, m_p2, SYMMETRY_MAP[bk][t], turn)
        else:
            best = None
            for t in NON_PAWN_VALID_TS[wk]:
                m_p1, m_p2 = SYMMETRY_MAP[p1][t], SYMMETRY_MAP[p2][t]
                if same_piece and m_p1 > m_p2: m_p1, m_p2 = m_p2, m_p1
                m = (NON_PAWN_WK_IDX[SYMMETRY_MAP[wk][t]], m_p1, m_p2, SYMMETRY_MAP[bk][t])
                if best is None or m < best: best = m
            return (best[0], best[1], best[2], best[3], turn)

    @staticmethod
    def _canonical_tuple_4vs(wk, wp, bk, bp, turn, has_pawn):
        if has_pawn:
            t = PAWN_VALID_T[wk]
            return (PAWN_WK_IDX[SYMMETRY_MAP[wk][t]], SYMMETRY_MAP[wp][t], SYMMETRY_MAP[bk][t], SYMMETRY_MAP[bp][t], turn)
        else:
            best = None
            for t in NON_PAWN_VALID_TS[wk]:
                m = (NON_PAWN_WK_IDX[SYMMETRY_MAP[wk][t]], SYMMETRY_MAP[wp][t], SYMMETRY_MAP[bk][t], SYMMETRY_MAP[bp][t])
                if best is None or m < best: best = m
            return (best[0], best[1], best[2], best[3], turn)

    @staticmethod
    def _canonical_tuple_5(wk, p1, p2, p3, bk, turn, has_pawn):
        # NOTE: p1/p2/p3 must already be in canonical type order (matching the generator).
        # Within same-type groups, pieces are sorted by square.
        # This method mirrors _canonical_tuple_4 extended to 3 pieces.
        # Update the same_piece logic when the 5-man generator is written.
        if has_pawn:
            t = PAWN_VALID_T[wk]
            m_p = sorted([SYMMETRY_MAP[p1][t], SYMMETRY_MAP[p2][t], SYMMETRY_MAP[p3][t]])
            return (PAWN_WK_IDX[SYMMETRY_MAP[wk][t]], m_p[0], m_p[1], m_p[2], SYMMETRY_MAP[bk][t], turn)
        else:
            best = None
            for t in NON_PAWN_VALID_TS[wk]:
                m_p = sorted([SYMMETRY_MAP[p1][t], SYMMETRY_MAP[p2][t], SYMMETRY_MAP[p3][t]])
                m = (NON_PAWN_WK_IDX[SYMMETRY_MAP[wk][t]], m_p[0], m_p[1], m_p[2], SYMMETRY_MAP[bk][t])
                if best is None or m < best: best = m
            return (best[0], best[1], best[2], best[3], best[4], turn)

    @staticmethod
    def _canonical_tuple_5vs(wk, wp1, wp2, bk, bp, turn, has_pawn):
        # NOTE: wp1/wp2 must already be in canonical type order (matching the generator).
        # Update when the 5-man generator is written.
        if has_pawn:
            t = PAWN_VALID_T[wk]
            m_wp = sorted([SYMMETRY_MAP[wp1][t], SYMMETRY_MAP[wp2][t]])
            return (PAWN_WK_IDX[SYMMETRY_MAP[wk][t]], m_wp[0], m_wp[1], SYMMETRY_MAP[bk][t], SYMMETRY_MAP[bp][t], turn)
        else:
            best = None
            for t in NON_PAWN_VALID_TS[wk]:
                m_wp = sorted([SYMMETRY_MAP[wp1][t], SYMMETRY_MAP[wp2][t]])
                m = (NON_PAWN_WK_IDX[SYMMETRY_MAP[wk][t]], m_wp[0], m_wp[1], SYMMETRY_MAP[bk][t], SYMMETRY_MAP[bp][t])
                if best is None or m < best: best = m
            return (best[0], best[1], best[2], best[3], best[4], turn)

    def probe(self, board, turn_to_move):
        w_objs = [p for p in board.white_pieces if not isinstance(p, King)]
        b_objs = [p for p in board.black_pieces if not isinstance(p, King)]
        if not board.white_king_pos or not board.black_king_pos: return None

        wk = board.white_king_pos[0] * 8 + board.white_king_pos[1]
        bk = board.black_king_pos[0] * 8 + board.black_king_pos[1]
        t_idx = 0 if turn_to_move == 'white' else 1
        w_cnt, b_cnt = len(w_objs), len(b_objs)

        # --- 3-MAN ---
        if w_cnt == 1 and b_cnt == 0:
            p = w_objs[0]
            tb = f"K_{type(p).__name__}_K_sml"
            if tb in self.tables:
                p_sq = p.pos[0] * 8 + p.pos[1]
                idx = self._canonical_tuple_3(wk, p_sq, bk, t_idx, type(p).__name__ == "Pawn")
                val = int(self.tables[tb][idx])
                is_win = (t_idx == 0 and val > 0) or (t_idx == 1 and val < 0)
                return self._tb_score_to_ai_score(val, is_win)

        elif b_cnt == 1 and w_cnt == 0:
            p = b_objs[0]
            tb = f"K_{type(p).__name__}_K_sml"
            if tb in self.tables:
                bk_f = _flip(board.black_king_pos)
                p_f  = _flip(p.pos)
                wk_f = _flip(board.white_king_pos)
                bk_sq, p_sq, wk_sq = bk_f[0]*8+bk_f[1], p_f[0]*8+p_f[1], wk_f[0]*8+wk_f[1]
                idx = self._canonical_tuple_3(bk_sq, p_sq, wk_sq, 1 - t_idx, type(p).__name__ == "Pawn")
                val = int(self.tables[tb][idx])
                b_wins = ((1 - t_idx) == 0 and val > 0) or ((1 - t_idx) == 1 and val < 0)
                return self._tb_score_to_ai_score(val, not b_wins)

        # --- 4-MAN SAME-SIDE ---
        elif w_cnt == 2 and b_cnt == 0:
            n1, n2 = type(w_objs[0]).__name__, type(w_objs[1]).__name__
            p1_sq = w_objs[0].pos[0]*8 + w_objs[0].pos[1]
            p2_sq = w_objs[1].pos[0]*8 + w_objs[1].pos[1]
            if n1 > n2: n1, n2, p1_sq, p2_sq = n2, n1, p2_sq, p1_sq
            tb = f"K_{n1}_{n2}_K_sml"
            if tb in self.tables:
                idx = self._canonical_tuple_4(wk, p1_sq, p2_sq, bk, t_idx, "Pawn" in tb, n1 == n2)
                val = int(self.tables[tb][idx])
                is_win = (t_idx == 0 and val > 0) or (t_idx == 1 and val < 0)
                return self._tb_score_to_ai_score(val, is_win)

        elif b_cnt == 2 and w_cnt == 0:
            n1, n2 = type(b_objs[0]).__name__, type(b_objs[1]).__name__
            p1_f = _flip(b_objs[0].pos); p2_f = _flip(b_objs[1].pos)
            p1_sq, p2_sq = p1_f[0]*8+p1_f[1], p2_f[0]*8+p2_f[1]
            if n1 > n2: n1, n2, p1_sq, p2_sq = n2, n1, p2_sq, p1_sq
            tb = f"K_{n1}_{n2}_K_sml"
            if tb in self.tables:
                bk_f = _flip(board.black_king_pos); wk_f = _flip(board.white_king_pos)
                bk_sq, wk_sq = bk_f[0]*8+bk_f[1], wk_f[0]*8+wk_f[1]
                idx = self._canonical_tuple_4(bk_sq, p1_sq, p2_sq, wk_sq, 1 - t_idx, "Pawn" in tb, n1 == n2)
                val = int(self.tables[tb][idx])
                b_wins = ((1 - t_idx) == 0 and val > 0) or ((1 - t_idx) == 1 and val < 0)
                return self._tb_score_to_ai_score(val, not b_wins)

        # --- 4-MAN CROSS ---
        elif w_cnt == 1 and b_cnt == 1:
            wn, bn = type(w_objs[0]).__name__, type(b_objs[0]).__name__
            wp_sq = w_objs[0].pos[0]*8 + w_objs[0].pos[1]
            bp_sq = b_objs[0].pos[0]*8 + b_objs[0].pos[1]
            if wn <= bn:
                tb = f"K_{wn}_vs_{bn}_K_sml"
                if tb in self.tables:
                    idx = self._canonical_tuple_4vs(wk, wp_sq, bk, bp_sq, t_idx, "Pawn" in tb)
                    val = int(self.tables[tb][idx])
                    is_win = (t_idx == 0 and val > 0) or (t_idx == 1 and val < 0)
                    return self._tb_score_to_ai_score(val, is_win)
            else:
                tb = f"K_{bn}_vs_{wn}_K_sml"
                if tb in self.tables:
                    bk_f = _flip(board.black_king_pos); bp_f = _flip(b_objs[0].pos)
                    wk_f = _flip(board.white_king_pos); wp_f = _flip(w_objs[0].pos)
                    bk_sq = bk_f[0]*8+bk_f[1]; bp_sq2 = bp_f[0]*8+bp_f[1]
                    wk_sq = wk_f[0]*8+wk_f[1]; wp_sq2 = wp_f[0]*8+wp_f[1]
                    idx = self._canonical_tuple_4vs(bk_sq, bp_sq2, wk_sq, wp_sq2, 1 - t_idx, "Pawn" in tb)
                    val = int(self.tables[tb][idx])
                    b_wins = ((1 - t_idx) == 0 and val > 0) or ((1 - t_idx) == 1 and val < 0)
                    return self._tb_score_to_ai_score(val, not b_wins)

        # --- 5-MAN SAME-SIDE ---
        elif w_cnt == 3 and b_cnt == 0:
            names = sorted([type(p).__name__ for p in w_objs])
            tb = f"K_{names[0]}_{names[1]}_{names[2]}_K_sml"
            if tb in self.tables:
                # Sort pieces by (name, square) to match generator canonical ordering
                pairs = sorted(zip([type(p).__name__ for p in w_objs],
                                   [p.pos[0]*8+p.pos[1] for p in w_objs]))
                p_sqs = [sq for _, sq in pairs]
                idx = self._canonical_tuple_5(wk, p_sqs[0], p_sqs[1], p_sqs[2], bk, t_idx, "Pawn" in tb)
                val = int(self.tables[tb][idx])
                is_win = (t_idx == 0 and val > 0) or (t_idx == 1 and val < 0)
                return self._tb_score_to_ai_score(val, is_win)

        elif b_cnt == 3 and w_cnt == 0:
            names = sorted([type(p).__name__ for p in b_objs])
            tb = f"K_{names[0]}_{names[1]}_{names[2]}_K_sml"
            if tb in self.tables:
                pairs = sorted(zip([type(p).__name__ for p in b_objs],
                                   [_flip(p.pos)[0]*8+_flip(p.pos)[1] for p in b_objs]))
                p_sqs = [sq for _, sq in pairs]
                bk_f = _flip(board.black_king_pos); wk_f = _flip(board.white_king_pos)
                bk_sq, wk_sq = bk_f[0]*8+bk_f[1], wk_f[0]*8+wk_f[1]
                idx = self._canonical_tuple_5(bk_sq, p_sqs[0], p_sqs[1], p_sqs[2], wk_sq, 1 - t_idx, "Pawn" in tb)
                val = int(self.tables[tb][idx])
                b_wins = ((1 - t_idx) == 0 and val > 0) or ((1 - t_idx) == 1 and val < 0)
                return self._tb_score_to_ai_score(val, not b_wins)

        # --- 5-MAN CROSS (2 white vs 1 black) ---
        elif w_cnt == 2 and b_cnt == 1:
            w_names = sorted([type(p).__name__ for p in w_objs])
            b_name  = type(b_objs[0]).__name__
            tb = f"K_{w_names[0]}_{w_names[1]}_vs_{b_name}_K_sml"
            if tb in self.tables:
                # Sort white pieces by (name, square) to match generator canonical ordering
                pairs = sorted(zip([type(p).__name__ for p in w_objs],
                                   [p.pos[0]*8+p.pos[1] for p in w_objs]))
                w_sqs = [sq for _, sq in pairs]
                bp_sq = b_objs[0].pos[0]*8 + b_objs[0].pos[1]
                idx = self._canonical_tuple_5vs(wk, w_sqs[0], w_sqs[1], bk, bp_sq, t_idx, "Pawn" in tb)
                val = int(self.tables[tb][idx])
                is_win = (t_idx == 0 and val > 0) or (t_idx == 1 and val < 0)
                return self._tb_score_to_ai_score(val, is_win)
            # Note: 1 white vs 2 black requires flip — omitted until those tables are generated

        return None