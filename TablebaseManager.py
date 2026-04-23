# TablebaseManager.py (v9.0 - Unified 16-bit tablebase loading)

import os
import numpy as np
from GameLogic import King, Board
from itertools import combinations, combinations_with_replacement

TB_SUFFIX = "_tb16.bin"
TB_DTYPE = np.int16

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

# Canonical order must match _PIECE_CANONICAL_ORDER in TablebaseGenerator
_PIECE_NAME_ORDER = {"Bishop": 0, "Knight": 1, "Pawn": 2, "Queen": 3, "Rook": 4}


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
            self.load_table(f"K_{p}_K")

        # 4-Man same-side
        for p1, p2 in combinations_with_replacement(pieces, 2):
            names = sorted([p1, p2])
            self.load_table(f"K_{names[0]}_{names[1]}_K")

        # 4-Man cross — build alphabetically-sorted keys and avoid duplicates.
        # The generator writes files with alphabetically-ordered piece names,
        # so we must match that ordering when attempting to load tables.
        seen_vs = set()
        for p1 in pieces:
            for p2 in pieces:
                names = sorted([p1, p2])
                key = f"K_{names[0]}_vs_{names[1]}_K"
                if key not in seen_vs:
                    seen_vs.add(key)
                    self.load_table(key)

        # 5-Man same-side
        for p1, p2, p3 in combinations_with_replacement(pieces, 3):
            names = sorted([p1, p2, p3])
            self.load_table(f"K_{names[0]}_{names[1]}_{names[2]}_K")

        # 5-Man cross (2 white vs 1 black)
        for p1, p2 in combinations_with_replacement(pieces, 2):
            names_w = sorted([p1, p2])
            for p3 in pieces:
                self.load_table(f"K_{names_w[0]}_{names_w[1]}_vs_{p3}_K")

    def load_table(self, base_name):
        # We strip suffixes just in case callers include one.
        base_name = base_name.replace("_tb16", "").replace("_xsml", "").replace("_sml16", "")
        
        if base_name in self.tables: return True
        
        filename = os.path.join(self.tb_dir, f"{base_name}{TB_SUFFIX}")
        if not os.path.exists(filename):
            return False
            
        try:
            has_pawn = "Pawn" in base_name
            wk_size = 32 if has_pawn else 10

            parts = base_name.split('_')
            num_pieces = len(parts) - 2 # "K_Pawn_K" -> 3 parts -> 1 piece
            if 'vs' in parts:
                num_pieces -= 1

            shape = tuple([wk_size] + [64] * (num_pieces + 1) + [2])
            
            self.tables[base_name] = np.memmap(filename, dtype=TB_DTYPE, mode='r', shape=shape)
            return True
        except Exception as e:
            print(f"[TablebaseManager] Failed to memmap {base_name}: {e}")
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
    def _canonical_tuple_5(wk, p1, p2, p3, bk, turn, has_pawn, p1n, p2n, p3n):
        """
        FIX (v7.1): Now takes piece name strings p1n/p2n/p3n and uses a 3-element bubble
        sort that ONLY swaps pieces of the EXACT SAME TYPE. This matches the generator's
        _canonical_flat_5 exactly. v7.0 sorted all three squares unconditionally, which
        broke lookups for any table where the three pieces are not all the same type.
        """
        if has_pawn:
            t = PAWN_VALID_T[wk]
            m1, m2, m3 = SYMMETRY_MAP[p1][t], SYMMETRY_MAP[p2][t], SYMMETRY_MAP[p3][t]
            if p1n == p2n and m1 > m2: m1, m2 = m2, m1
            if p2n == p3n and m2 > m3: m2, m3 = m3, m2
            if p1n == p2n and m1 > m2: m1, m2 = m2, m1
            return (PAWN_WK_IDX[SYMMETRY_MAP[wk][t]], m1, m2, m3, SYMMETRY_MAP[bk][t], turn)
        else:
            best = None
            for t in NON_PAWN_VALID_TS[wk]:
                m1, m2, m3 = SYMMETRY_MAP[p1][t], SYMMETRY_MAP[p2][t], SYMMETRY_MAP[p3][t]
                if p1n == p2n and m1 > m2: m1, m2 = m2, m1
                if p2n == p3n and m2 > m3: m2, m3 = m3, m2
                if p1n == p2n and m1 > m2: m1, m2 = m2, m1
                m = (NON_PAWN_WK_IDX[SYMMETRY_MAP[wk][t]], m1, m2, m3, SYMMETRY_MAP[bk][t])
                if best is None or m < best: best = m
            return (best[0], best[1], best[2], best[3], best[4], turn)

    @staticmethod
    def _canonical_tuple_5vs(wk, wp1, wp2, bk, bp, turn, has_pawn, wp1n, wp2n):
        """
        FIX (v7.1): Now takes white piece name strings wp1n/wp2n and only swaps same-type
        white pieces, matching _canonical_flat_5vs in the generator exactly.
        v7.0 sorted both white piece squares unconditionally, which broke lookups when the
        two white pieces are different types (e.g. K+Queen+Rook vs K+Knight).
        """
        if has_pawn:
            t = PAWN_VALID_T[wk]
            m1, m2 = SYMMETRY_MAP[wp1][t], SYMMETRY_MAP[wp2][t]
            if wp1n == wp2n and m1 > m2: m1, m2 = m2, m1
            return (PAWN_WK_IDX[SYMMETRY_MAP[wk][t]], m1, m2, SYMMETRY_MAP[bk][t], SYMMETRY_MAP[bp][t], turn)
        else:
            best = None
            for t in NON_PAWN_VALID_TS[wk]:
                m1, m2 = SYMMETRY_MAP[wp1][t], SYMMETRY_MAP[wp2][t]
                if wp1n == wp2n and m1 > m2: m1, m2 = m2, m1
                m = (NON_PAWN_WK_IDX[SYMMETRY_MAP[wk][t]], m1, m2, SYMMETRY_MAP[bk][t], SYMMETRY_MAP[bp][t])
                if best is None or m < best: best = m
            return (best[0], best[1], best[2], best[3], best[4], turn)

    def probe(self, board, turn_to_move):
        w_objs = [p for p in board.white_pieces if not isinstance(p, King)]
        b_objs = [p for p in board.black_pieces if not isinstance(p, King)]
        if not board.white_king_pos or not board.black_king_pos: return None

        # Same-color bishop blindspot fix
        for objs in (w_objs, b_objs):
            bishops = [p for p in objs if type(p).__name__ == "Bishop"]
            if len(bishops) >= 2:
                for i in range(len(bishops)):
                    for j in range(i + 1, len(bishops)):
                        sq1, sq2 = bishops[i].pos, bishops[j].pos
                        if ((sq1[0] + sq1[1]) % 2) == ((sq2[0] + sq2[1]) % 2):
                            return None 

        wk = board.white_king_pos[0] * 8 + board.white_king_pos[1]
        bk = board.black_king_pos[0] * 8 + board.black_king_pos[1]
        t_idx = 0 if turn_to_move == 'white' else 1
        w_cnt, b_cnt = len(w_objs), len(b_objs)

        # --- 3-MAN ---
        if w_cnt == 1 and b_cnt == 0:
            p = w_objs[0]
            tb = f"K_{type(p).__name__}_K"
            if tb in self.tables:
                p_sq = p.pos[0] * 8 + p.pos[1]
                idx = self._canonical_tuple_3(wk, p_sq, bk, t_idx, type(p).__name__ == "Pawn")
                val = int(self.tables[tb][idx])
                is_win = (t_idx == 0 and val > 0) or (t_idx == 1 and val < 0)
                return self._tb_score_to_ai_score(val, is_win)

        elif b_cnt == 1 and w_cnt == 0:
            p = b_objs[0]
            tb = f"K_{type(p).__name__}_K"
            if tb in self.tables:
                bk_f, p_f, wk_f = _flip(board.black_king_pos), _flip(p.pos), _flip(board.white_king_pos)
                bk_sq, p_sq, wk_sq = bk_f[0]*8+bk_f[1], p_f[0]*8+p_f[1], wk_f[0]*8+wk_f[1]
                idx = self._canonical_tuple_3(bk_sq, p_sq, wk_sq, 1 - t_idx, type(p).__name__ == "Pawn")
                val = int(self.tables[tb][idx])
                b_wins = ((1 - t_idx) == 0 and val > 0) or ((1 - t_idx) == 1 and val < 0)
                return self._tb_score_to_ai_score(val, not b_wins)

        # --- 4-MAN SAME-SIDE ---
        elif w_cnt == 2 and b_cnt == 0:
            pairs = sorted([(type(p).__name__, p.pos[0]*8+p.pos[1]) for p in w_objs],
                           key=lambda x: (_PIECE_NAME_ORDER.get(x[0], 99), x[1]))
            (n1, p1_sq), (n2, p2_sq) = pairs[0], pairs[1]
            tb = f"K_{n1}_{n2}_K"
            if tb in self.tables:
                idx = self._canonical_tuple_4(wk, p1_sq, p2_sq, bk, t_idx, "Pawn" in tb, n1 == n2)
                val = int(self.tables[tb][idx])
                is_win = (t_idx == 0 and val > 0) or (t_idx == 1 and val < 0)
                return self._tb_score_to_ai_score(val, is_win)

        elif b_cnt == 2 and w_cnt == 0:
            pairs = sorted([(type(p).__name__, _flip(p.pos)[0]*8+_flip(p.pos)[1]) for p in b_objs],
                           key=lambda x: (_PIECE_NAME_ORDER.get(x[0], 99), x[1]))
            (n1, p1_sq), (n2, p2_sq) = pairs[0], pairs[1]
            tb = f"K_{n1}_{n2}_K"
            if tb in self.tables:
                bk_f, wk_f = _flip(board.black_king_pos), _flip(board.white_king_pos)
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
            
            if _PIECE_NAME_ORDER[wn] <= _PIECE_NAME_ORDER[bn]:
                tb = f"K_{wn}_vs_{bn}_K"
                if tb in self.tables:
                    idx = self._canonical_tuple_4vs(wk, wp_sq, bk, bp_sq, t_idx, "Pawn" in tb)
                    val = int(self.tables[tb][idx])
                    is_win = (t_idx == 0 and val > 0) or (t_idx == 1 and val < 0)
                    return self._tb_score_to_ai_score(val, is_win)
            else:
                tb = f"K_{bn}_vs_{wn}_K"
                if tb in self.tables:
                    bk_f, bp_f = _flip(board.black_king_pos), _flip(b_objs[0].pos)
                    wk_f, wp_f = _flip(board.white_king_pos), _flip(w_objs[0].pos)
                    bk_sq, bp_sq2 = bk_f[0]*8+bk_f[1], bp_f[0]*8+bp_f[1]
                    wk_sq, wp_sq2 = wk_f[0]*8+wk_f[1], wp_f[0]*8+wp_f[1]
                    
                    idx = self._canonical_tuple_4vs(bk_sq, bp_sq2, wk_sq, wp_sq2, 1 - t_idx, "Pawn" in tb)
                    val = int(self.tables[tb][idx])
                    b_wins = ((1 - t_idx) == 0 and val > 0) or ((1 - t_idx) == 1 and val < 0)
                    return self._tb_score_to_ai_score(val, not b_wins)

        # --- 5-MAN SAME-SIDE ---
        elif w_cnt == 3 and b_cnt == 0:
            pairs = sorted([(type(p).__name__, p.pos[0]*8+p.pos[1]) for p in w_objs],
                           key=lambda x: (_PIECE_NAME_ORDER.get(x[0], 99), x[1]))
            (n1, p1_sq), (n2, p2_sq), (n3, p3_sq) = pairs[0], pairs[1], pairs[2]
            tb = f"K_{n1}_{n2}_{n3}_K"
            if tb in self.tables:
                idx = self._canonical_tuple_5(wk, p1_sq, p2_sq, p3_sq, bk, t_idx, "Pawn" in tb, n1, n2, n3)
                val = int(self.tables[tb][idx])
                is_win = (t_idx == 0 and val > 0) or (t_idx == 1 and val < 0)
                return self._tb_score_to_ai_score(val, is_win)

        elif b_cnt == 3 and w_cnt == 0:
            pairs = sorted([(type(p).__name__, _flip(p.pos)[0]*8+_flip(p.pos)[1]) for p in b_objs],
                           key=lambda x: (_PIECE_NAME_ORDER.get(x[0], 99), x[1]))
            (n1, p1_sq), (n2, p2_sq), (n3, p3_sq) = pairs[0], pairs[1], pairs[2]
            tb = f"K_{n1}_{n2}_{n3}_K"
            if tb in self.tables:
                bk_f, wk_f = _flip(board.black_king_pos), _flip(board.white_king_pos)
                bk_sq, wk_sq = bk_f[0]*8+bk_f[1], wk_f[0]*8+wk_f[1]
                idx = self._canonical_tuple_5(bk_sq, p1_sq, p2_sq, p3_sq, wk_sq, 1 - t_idx, "Pawn" in tb, n1, n2, n3)
                val = int(self.tables[tb][idx])
                b_wins = ((1 - t_idx) == 0 and val > 0) or ((1 - t_idx) == 1 and val < 0)
                return self._tb_score_to_ai_score(val, not b_wins)
            
        # --- 5-MAN CROSS (2 white vs 1 black) ---
        elif w_cnt == 2 and b_cnt == 1:
            w_pairs = sorted([(type(p).__name__, p.pos[0]*8+p.pos[1]) for p in w_objs],
                             key=lambda x: (_PIECE_NAME_ORDER.get(x[0], 99), x[1]))
            (wn1, wp1_sq), (wn2, wp2_sq) = w_pairs[0], w_pairs[1]
            bn = type(b_objs[0]).__name__
            bp_sq = b_objs[0].pos[0]*8 + b_objs[0].pos[1]
            
            tb = f"K_{wn1}_{wn2}_vs_{bn}_K"
            if tb in self.tables:
                idx = self._canonical_tuple_5vs(wk, wp1_sq, wp2_sq, bk, bp_sq, t_idx, "Pawn" in tb, wn1, wn2)
                val = int(self.tables[tb][idx])
                is_win = (t_idx == 0 and val > 0) or (t_idx == 1 and val < 0)
                return self._tb_score_to_ai_score(val, is_win)
            
        # --- 5-MAN CROSS (1 white vs 2 black) ---
        elif w_cnt == 1 and b_cnt == 2:
            b_pairs = sorted([(type(p).__name__, _flip(p.pos)[0]*8+_flip(p.pos)[1]) for p in b_objs],
                             key=lambda x: (_PIECE_NAME_ORDER.get(x[0], 99), x[1]))
            (bn1, bp1_sq), (bn2, bp2_sq) = b_pairs[0], b_pairs[1]
            wn = type(w_objs[0]).__name__
            wp_sq = _flip(w_objs[0].pos)[0]*8 + _flip(w_objs[0].pos)[1]
            
            tb = f"K_{bn1}_{bn2}_vs_{wn}_K"
            if tb in self.tables:
                bk_f, wk_f = _flip(board.black_king_pos), _flip(board.white_king_pos)
                idx = self._canonical_tuple_5vs(bk_f[0]*8+bk_f[1], bp1_sq, bp2_sq, wk_f[0]*8+wk_f[1], wp_sq, 1 - t_idx, "Pawn" in tb, bn1, bn2)
                val = int(self.tables[tb][idx])
                b_wins = ((1 - t_idx) == 0 and val > 0) or ((1 - t_idx) == 1 and val < 0)
                return self._tb_score_to_ai_score(val, not b_wins)
                
        return None
