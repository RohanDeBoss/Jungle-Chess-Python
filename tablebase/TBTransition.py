import os, itertools
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from numba import njit
from GameLogic import (KNIGHT_ATTACKS_FROM, RAYS, BISHOP_ZIGZAG_RAYS, RAY_BETWEEN, KING_PATHS,
                       ADJACENT_SQUARES_MAP, PAWN_DIRECT, PAWN_DOUBLE, PAWN_FORWARD_SQUARES,
                       PAWN_SIDEWAYS_SQUARES, KING_ATTACK_CHECK, KNIGHT_EVAP_SQUARES)

I16 = np.int16
MAXM = 192          # >= max pseudo-legal moves for K+3 pieces
MAXK = 16           # black has a bare king -> <= 16 moves
ABSENT = -(1 << 40)
COLORS = ('white', 'black')
Z_BY_NAME = {"Pawn": 0, "Knight": 1, "Bishop": 2, "Rook": 3, "Queen": 4}
CR = np.array([2, 1, 0, 4, 3], np.int64)          # z -> canonical rank (B,N,P,Q,R)
RANK_NAMES = ["Bishop", "Knight", "Pawn", "Queen", "Rook"]

def _s(rc): return rc[0] * 8 + rc[1]

# ---------------- static tables (built from GameLogic so rules can't drift) ----------------
KN_N = np.zeros(64, I16);            KN_SQ = np.zeros((64, 8), I16)
EV_KIND = np.zeros((64, 64), I16);   EV_N = np.zeros((64, 64), I16); EV_SQ = np.zeros((64, 64, 4), I16)
RAY_N = np.zeros((64, 8), I16);      RAY_SQ = np.zeros((64, 8, 7), I16)
ZZ_N = np.zeros((64, 8), I16);       ZZ_SQ = np.zeros((64, 8, 7), I16)
BT_N = np.zeros((64, 64), I16);      BT_SQ = np.zeros((64, 64, 6), I16)
KP_N = np.zeros(64, I16);            KP_LEN = np.zeros((64, 8), I16); KP_SQ = np.zeros((64, 8, 2), I16)
ADJ_N = np.zeros(64, I16);           ADJ_SQ = np.zeros((64, 8), I16)
PD_N = np.zeros((2, 64), I16);       PD_SQ = np.zeros((2, 64, 3), I16); PDB = np.full((2, 64, 2), -1, I16)
PF_N = np.zeros((2, 64), I16);       PF_SQ = np.zeros((2, 64, 2), I16)
PS_N = np.zeros(64, I16);            PS_SQ = np.zeros((64, 2), I16)
KAC = np.full((64, 64), -1, I16)     # -1 none, -2 adjacent, >=0 square that must be empty

for s in range(64):
    r, c = divmod(s, 8)
    kn = KNIGHT_ATTACKS_FROM[(r, c)]; KN_N[s] = len(kn)
    for i, x in enumerate(kn): KN_SQ[s, i] = _s(x)
    for t in range(64):
        e = KNIGHT_EVAP_SQUARES[s][t]
        if e is True: EV_KIND[s, t] = 1
        elif e is not None:
            assert len(e) <= 4
            EV_KIND[s, t] = 2; EV_N[s, t] = len(e)
            for i, x in enumerate(e): EV_SQ[s, t, i] = _s(x)
        bt = RAY_BETWEEN[s][t]; BT_N[s, t] = len(bt)
        for i, x in enumerate(bt): BT_SQ[s, t, i] = _s(x)
        v = KING_ATTACK_CHECK[s][t]
        if v is True: KAC[s, t] = -2
        elif v: KAC[s, t] = _s(v)
    for d in range(8):
        RAY_N[s, d] = len(RAYS[s][d])
        for i, x in enumerate(RAYS[s][d]): RAY_SQ[s, d, i] = _s(x)
        ZZ_N[s, d] = len(BISHOP_ZIGZAG_RAYS[s][d])
        for i, x in enumerate(BISHOP_ZIGZAG_RAYS[s][d]): ZZ_SQ[s, d, i] = _s(x)
    KP_N[s] = len(KING_PATHS[s])
    for i, path in enumerate(KING_PATHS[s]):
        KP_LEN[s, i] = len(path)
        for k, x in enumerate(path): KP_SQ[s, i, k] = _s(x)
    adj = ADJACENT_SQUARES_MAP[(r, c)]; ADJ_N[s] = len(adj)
    for i, x in enumerate(adj): ADJ_SQ[s, i] = _s(x)
    PS_N[s] = len(PAWN_SIDEWAYS_SQUARES[s])
    for i, x in enumerate(PAWN_SIDEWAYS_SQUARES[s]): PS_SQ[s, i] = _s(x)
    for ci, cn in enumerate(COLORS):
        pd = PAWN_DIRECT[cn][s]; PD_N[ci, s] = len(pd)
        for i, x in enumerate(pd): PD_SQ[ci, s, i] = _s(x)
        dbl = PAWN_DOUBLE[cn][s]
        if dbl: PDB[ci, s, 0] = dbl[0] * 8 + dbl[1]; PDB[ci, s, 1] = dbl[2] * 8 + dbl[3]
        pf = PAWN_FORWARD_SQUARES[cn][s]; PF_N[ci, s] = len(pf)
        for i, x in enumerate(pf): PF_SQ[ci, s, i] = _s(x)

# symmetry / canonical tables (same formulas as TablebaseGenerator)
SYM = np.zeros((64, 8), I16)
for r in range(8):
    for c in range(8):
        s = r * 8 + c
        for k, (rr, cc) in enumerate([(r, c), (r, 7 - c), (7 - r, c), (7 - r, 7 - c),
                                      (c, r), (c, 7 - r), (7 - c, r), (7 - c, 7 - r)]):
            SYM[s, k] = rr * 8 + cc
PAWN_VALID_T = np.array([0 if (s % 8) <= 3 else 1 for s in range(64)], I16)
P_WK_SQ = np.array([r * 8 + c for r in range(8) for c in range(4)], I16)
P_WK_IDX = np.full(64, -1, I16)
for i, s in enumerate(P_WK_SQ): P_WK_IDX[s] = i
_np_sq = [r * 8 + c for c in range(4) for r in range(c + 1)]
NP_WK_SQ = np.array(_np_sq, I16)
NP_WK_IDX = np.full(64, -1, I16)
for i, s in enumerate(_np_sq): NP_WK_IDX[s] = i
NP_VT_N = np.zeros(64, I16); NP_VT = np.zeros((64, 8), I16)
for s in range(64):
    for t in range(8):
        if NP_WK_IDX[SYM[s, t]] >= 0:
            NP_VT[s, NP_VT_N[s]] = t; NP_VT_N[s] += 1

# ---------------- board primitives: int8[64], code = (z+1) + 8*color, 0 = empty ----------------
@njit(cache=True, nogil=True)
def king_sq(b, color):
    kc = color * 8 + 6
    for s in range(64):
        if b[s] == kc: return s
    return -1

@njit(cache=True, nogil=True)
def attacked(b, t, att):
    base = att * 8
    pawn = base + 1
    for i in range(PD_N[att, t]):
        if b[PD_SQ[att, t, i]] == pawn: return True
    d = PDB[att, t, 0]
    if d >= 0:
        if b[d] == 0 and b[PDB[att, t, 1]] == pawn: return True
    kp = -1; hasr = False; hasb = False; hasq = False
    for s in range(64):
        v = b[s]
        if v == 0 or (v >> 3) != att: continue
        z = v & 7
        if z == 2:
            ek = EV_KIND[s, t]
            if ek == 1: return True
            if ek == 2:
                for q in range(EV_N[s, t]):
                    if b[EV_SQ[s, t, q]] == 0: return True
        elif z == 6: kp = s
        elif z == 4: hasr = True
        elif z == 3: hasb = True
        elif z == 5: hasq = True
    if kp >= 0:
        kc = KAC[kp, t]
        if kc == -2: return True
        if kc >= 0 and b[kc] == 0: return True
    if hasr:
        for r in range(4):
            for i in range(RAY_N[t, r]):
                v = b[RAY_SQ[t, r, i]]
                if v != 0 and (v >> 3) == att:
                    if v == base + 4: return True
                    break                       # rook pierces enemies, stops at own side's pieces
    if hasb:
        for r in range(4, 8):
            for i in range(RAY_N[t, r]):
                v = b[RAY_SQ[t, r, i]]
                if v != 0:
                    if v == base + 3: return True
                    break
        for r in range(8):
            for i in range(ZZ_N[t, r]):
                v = b[ZZ_SQ[t, r, i]]
                if v != 0:
                    if v == base + 3: return True
                    break
    if hasq:
        tr = t >> 3; tc = t & 7
        for s in range(64):
            if b[s] == base + 5:
                for r in range(8):
                    for i in range(RAY_N[s, r]):
                        q = RAY_SQ[s, r, i]; v = b[q]
                        if v != 0:
                            if (v >> 3) != att:
                                dr = (q >> 3) - tr; dc = (q & 7) - tc
                                if -1 <= dr <= 1 and -1 <= dc <= 1: return True
                            break
    return False

@njit(cache=True, nogil=True)
def in_check(b, color):
    k = king_sq(b, color)
    if k < 0: return True
    return attacked(b, k, 1 - color)

@njit(cache=True, nogil=True)
def apply_move(b, frm, to, c):
    for i in range(64): c[i] = b[i]
    p = c[frm]; mc = p >> 3; z = (p & 7) - 1
    is_cap = c[to] != 0
    if z == 3:
        for k in range(BT_N[frm, to]):
            s = BT_SQ[frm, to, k]; v = c[s]
            if v != 0 and (v >> 3) != mc: c[s] = 0
    if is_cap: c[to] = 0
    c[frm] = 0; c[to] = p
    if z == 4 and is_cap:
        c[to] = 0
        for j in range(ADJ_N[to]):
            a = ADJ_SQ[to, j]; v = c[a]
            if v != 0 and (v >> 3) != mc: c[a] = 0
    elif z == 0 and (to >> 3) == (0 if mc == 0 else 7):
        c[to] = mc * 8 + 5
    if z == 1:
        for j in range(KN_N[to]):
            ek = KN_SQ[to, j]; v = c[ek]
            if v != 0 and (v >> 3) != mc and (v & 7) == 2:
                for i in range(KN_N[ek]):
                    q = KN_SQ[ek, i]; w = c[q]
                    if w != 0 and (w >> 3) == mc: c[q] = 0
        for j in range(KN_N[to]):
            q = KN_SQ[to, j]; w = c[q]
            if w != 0 and (w >> 3) != mc: c[q] = 0
    else:
        if c[to] != 0:
            for j in range(KN_N[to]):
                v = c[KN_SQ[to, j]]
                if v != 0 and (v & 7) == 2 and (v >> 3) != mc:
                    c[to] = 0
                    break

@njit(cache=True, nogil=True)
def gen_moves(b, color, mf, mt):
    n = 0
    for s in range(64):
        v = b[s]
        if v == 0 or (v >> 3) != color: continue
        z = (v & 7) - 1
        if z == 5:
            for i in range(KP_N[s]):
                s1 = KP_SQ[s, i, 0]; t1 = b[s1]
                if t1 == 0 or (t1 >> 3) != color:
                    mf[n] = s; mt[n] = s1; n += 1
                    if t1 == 0 and KP_LEN[s, i] == 2:
                        s2 = KP_SQ[s, i, 1]; t2 = b[s2]
                        if t2 == 0 or (t2 >> 3) != color:
                            mf[n] = s; mt[n] = s2; n += 1
        elif z == 4:
            for r in range(8):
                for i in range(RAY_N[s, r]):
                    q = RAY_SQ[s, r, i]; t = b[q]
                    if t == 0:
                        mf[n] = s; mt[n] = q; n += 1
                    else:
                        if (t >> 3) != color:
                            mf[n] = s; mt[n] = q; n += 1
                        break
        elif z == 3:
            for r in range(4):
                for i in range(RAY_N[s, r]):
                    q = RAY_SQ[s, r, i]; t = b[q]
                    if t != 0 and (t >> 3) == color: break
                    mf[n] = s; mt[n] = q; n += 1
        elif z == 2:
            seen = 0
            for r in range(4, 8):
                for i in range(RAY_N[s, r]):
                    q = RAY_SQ[s, r, i]; t = b[q]
                    if t != 0:
                        if (t >> 3) != color and ((seen >> q) & 1) == 0:
                            seen |= (1 << q); mf[n] = s; mt[n] = q; n += 1
                        break
                    if ((seen >> q) & 1) == 0:
                        seen |= (1 << q); mf[n] = s; mt[n] = q; n += 1
            for r in range(8):
                for i in range(ZZ_N[s, r]):
                    q = ZZ_SQ[s, r, i]; t = b[q]
                    if t != 0:
                        if (t >> 3) != color and ((seen >> q) & 1) == 0:
                            seen |= (1 << q); mf[n] = s; mt[n] = q; n += 1
                        break
                    if ((seen >> q) & 1) == 0:
                        seen |= (1 << q); mf[n] = s; mt[n] = q; n += 1
        elif z == 1:
            for i in range(KN_N[s]):
                q = KN_SQ[s, i]
                if b[q] == 0:
                    mf[n] = s; mt[n] = q; n += 1
        else:  # pawn
            nf = PF_N[color, s]
            if nf > 0:
                s1 = PF_SQ[color, s, 0]; t1 = b[s1]
                if t1 == 0 or (t1 >> 3) != color:
                    mf[n] = s; mt[n] = s1; n += 1
                    if t1 == 0 and nf == 2:
                        s2 = PF_SQ[color, s, 1]; t2 = b[s2]
                        if t2 == 0 or (t2 >> 3) != color:
                            mf[n] = s; mt[n] = s2; n += 1
            for i in range(PS_N[s]):
                q = PS_SQ[s, i]; t = b[q]
                if t != 0 and (t >> 3) != color:
                    mf[n] = s; mt[n] = q; n += 1
    return n

@njit(cache=True, nogil=True)
def has_legal(b, color, hf, ht, hc):
    n = gen_moves(b, color, hf, ht)
    for i in range(n):
        apply_move(b, hf[i], ht[i], hc)
        if not in_check(hc, color): return True
    return False

# ---------------- canonical flat (generic over n = 1..3 white pieces) ----------------
@njit(cache=True, nogil=True)
def canon_flat(n, wk, p0, p1, p2, bk, turn, has_pawn, same01, same12):
    best = -1
    nt = 1 if has_pawn else NP_VT_N[wk]
    for ti in range(nt):
        if has_pawn:
            t = PAWN_VALID_T[wk]; a = P_WK_IDX[SYM[wk, t]]
        else:
            t = NP_VT[wk, ti]; a = NP_WK_IDX[SYM[wk, t]]
        m0 = np.int64(SYM[p0, t]); m1 = np.int64(0); m2 = np.int64(0)
        if n >= 2: m1 = np.int64(SYM[p1, t])
        if n >= 3: m2 = np.int64(SYM[p2, t])
        if n >= 2 and same01 and m0 > m1: m0, m1 = m1, m0
        if n >= 3:
            if same12 and m1 > m2: m1, m2 = m2, m1
            if same01 and m0 > m1: m0, m1 = m1, m0
        e = np.int64(a) * 64 + m0
        if n >= 2: e = e * 64 + m1
        if n >= 3: e = e * 64 + m2
        e = e * 64 + SYM[bk, t]
        if best < 0 or e < best: best = e
    return best * 2 + turn

@njit(cache=True, nogil=True)
def rd(tb, is16, idx):
    if is16 == 0:
        v = np.int64(tb[idx])
        return v - 256 if v > 127 else v
    v = np.int64(tb[2 * idx]) | (np.int64(tb[2 * idx + 1]) << 8)
    return v - 65536 if v > 32767 else v

@njit(cache=True, nogil=True)
def lookup3(T3, has3, i16, z, wk, p, bk, turn):
    if has3[z] == 0: return ABSENT
    idx = canon_flat(1, wk, p, 0, 0, bk, turn, z == 0, False, False)
    return rd(T3[z], i16[z], idx)

@njit(cache=True, nogil=True)
def lookup4(T4, has4, i16, za, zb, wk, pa, pb, bk, turn):   # pieces pre-sorted by rank
    k = CR[za] * 5 + CR[zb]
    if has4[k] == 0: return ABSENT
    idx = canon_flat(2, wk, pa, pb, 0, bk, turn, za == 0 or zb == 0, za == zb, False)
    return rd(T4[k], i16[k], idx)

@njit(cache=True, nogil=True)
def gather_white(c, ws, wz):
    m = 0
    for s in range(64):
        v = c[s]
        if v != 0 and v < 8 and v != 6:
            ws[m] = s; wz[m] = v - 1; m += 1
    return m

@njit(cache=True, nogil=True)
def sort_white(ws, wz, m):
    for i in range(1, m):
        s = ws[i]; z = wz[i]; key = CR[z] * 64 + s
        k = i - 1
        while k >= 0 and CR[wz[k]] * 64 + ws[k] > key:
            ws[k + 1] = ws[k]; wz[k + 1] = wz[k]; k -= 1
        ws[k + 1] = s; wz[k + 1] = z

@njit(cache=True, nogil=True)
def child_flat(n, wks, ws, wz, bks, turn, has_pawn):
    return canon_flat(n, wks, ws[0], ws[1], ws[2], bks, turn, has_pawn, wz[0] == wz[1], wz[1] == wz[2])

@njit(cache=True, nogil=True)
def same_types(n, wz, gz):
    for i in range(n):
        if wz[i] != gz[i]: return False
    return True

# ---------------- per-state transition (mirrors _build_transition_worker{,_4,_5}) ----------------
@njit(cache=True, nogil=True)
def one_state(j, flat, n, has_pawn, gz, T3, has3, i16_3, T4, has4, i16_4,
              TP, tp_has, tp_i16, tp_key,
              o_valid, o_tab, o_deg, o_pmin, o_nch, o_ch, o_nkn, o_kv,
              b, c, mf, mt, hf, ht, hc, ws, wz):
    rest = flat >> 1
    bk = rest & 63; rest >>= 6
    p2 = 0; p1 = 0
    if n == 3:
        p2 = rest & 63; rest >>= 6
    if n >= 2:
        p1 = rest & 63; rest >>= 6
    p0 = rest & 63; rest >>= 6
    wk = np.int64(P_WK_SQ[rest]) if has_pawn else np.int64(NP_WK_SQ[rest])
    for i in range(64): b[i] = 0
    b[wk] = 6; b[bk] = 14
    b[p0] = gz[0] + 1
    if n >= 2: b[p1] = gz[1] + 1
    if n >= 3: b[p2] = gz[2] + 1
    turn = flat & 1
    if in_check(b, 1 if turn == 0 else 0):
        o_valid[j] = 0
        return
    o_valid[j] = 1
    nmv = gen_moves(b, turn, mf, mt)
    nc = 0; nk = 0

    if turn == 0:                                   # ---------- white to move
        imm = False; minwin = 0
        for mi in range(nmv):
            apply_move(b, mf[mi], mt[mi], c)
            if in_check(c, 0): continue
            bks = king_sq(c, 1)
            if bks < 0 or not has_legal(c, 1, hf, ht, hc):
                imm = True
                break
            wks = king_sq(c, 0)
            m = gather_white(c, ws, wz)
            sort_white(ws, wz, m)
            val = ABSENT
            if m == n:
                if same_types(n, wz, gz):
                    o_ch[j, nc] = child_flat(n, wks, ws, wz, bks, 1, has_pawn); nc += 1
                    continue
                if n == 1:
                    val = lookup3(T3, has3, i16_3, wz[0], wks, ws[0], bks, 1)
                elif n == 2:
                    if tp_has:
                        idx = canon_flat(2, wks, ws[0], ws[1], 0, bks, 1,
                                         wz[0] == 0 or wz[1] == 0, wz[0] == wz[1], False)
                        val = rd(TP, tp_i16, idx)
                else:
                    if tp_has and CR[wz[0]] == tp_key[0] and CR[wz[1]] == tp_key[1] and CR[wz[2]] == tp_key[2]:
                        idx = canon_flat(3, wks, ws[0], ws[1], ws[2], bks, 1,
                                         wz[0] == 0 or wz[1] == 0 or wz[2] == 0,
                                         wz[0] == wz[1], wz[1] == wz[2])
                        val = rd(TP, tp_i16, idx)
            elif m == 1:
                val = lookup3(T3, has3, i16_3, wz[0], wks, ws[0], bks, 1)
            elif m == 2:
                val = lookup4(T4, has4, i16_4, wz[0], wz[1], wks, ws[0], ws[1], bks, 1)
            if val != ABSENT and val < 0:
                w = -val + 1
                if minwin == 0 or w < minwin: minwin = w
        if imm:
            o_tab[j] = 1; o_deg[j] = 0; o_nch[j] = 0; o_pmin[j] = 0
        else:
            o_tab[j] = 0; o_deg[j] = nc; o_nch[j] = nc; o_pmin[j] = minwin
        o_nkn[j] = 0
        return

    legal = 0; esc = False                          # ---------- black to move
    for mi in range(nmv):
        apply_move(b, mf[mi], mt[mi], c)
        if in_check(c, 1): continue
        legal += 1
        wks = king_sq(c, 0)
        bks = king_sq(c, 1)
        if n <= 2:
            if wks < 0:
                esc = True; continue
            m = gather_white(c, ws, wz)
            if m == 0 or not has_legal(c, 0, hf, ht, hc):
                esc = True; continue
            sort_white(ws, wz, m)
            if n == 1:
                if has_pawn and wz[0] == 4:
                    esc = True; continue
                o_ch[j, nc] = child_flat(1, wks, ws, wz, bks, 0, has_pawn); nc += 1
            else:
                if m == 1:
                    val = lookup3(T3, has3, i16_3, wz[0], wks, ws[0], bks, 0)
                    if val != ABSENT:
                        if val <= 0: esc = True
                        else:
                            o_kv[j, nk] = val; nk += 1
                else:
                    o_ch[j, nc] = child_flat(2, wks, ws, wz, bks, 0, has_pawn); nc += 1
        else:
            if wks < 0:
                esc = True; continue
            m = gather_white(c, ws, wz)
            sort_white(ws, wz, m)
            if m == 3:
                if same_types(3, wz, gz):
                    o_ch[j, nc] = child_flat(3, wks, ws, wz, bks, 0, has_pawn); nc += 1
                else:
                    esc = True
            else:
                val = ABSENT
                if m == 2: val = lookup4(T4, has4, i16_4, wz[0], wz[1], wks, ws[0], ws[1], bks, 0)
                elif m == 1: val = lookup3(T3, has3, i16_3, wz[0], wks, ws[0], bks, 0)
                if val != ABSENT and val > 0:
                    o_kv[j, nk] = val; nk += 1
                else:
                    esc = True
    o_pmin[j] = 0
    if legal == 0:
        o_tab[j] = -1; o_deg[j] = 0; o_nch[j] = 0; o_nkn[j] = 0
    elif esc:
        o_tab[j] = 0; o_deg[j] = 65535; o_nch[j] = 0; o_nkn[j] = 0
    else:
        o_tab[j] = 0; o_deg[j] = nc + nk; o_nch[j] = nc; o_nkn[j] = nk

@njit(cache=True, nogil=True)
def process_chunk(flats, n, has_pawn, gz, T3, has3, i16_3, T4, has4, i16_4,
                  TP, tp_has, tp_i16, tp_key,
                  o_valid, o_tab, o_deg, o_pmin, o_nch, o_ch, o_nkn, o_kv):
    b = np.zeros(64, np.int8); c = np.zeros(64, np.int8); hc = np.zeros(64, np.int8)
    mf = np.zeros(MAXM, np.int16); mt = np.zeros(MAXM, np.int16)
    hf = np.zeros(MAXM, np.int16); ht = np.zeros(MAXM, np.int16)
    ws = np.zeros(4, np.int64); wz = np.zeros(4, np.int64)
    for j in range(flats.shape[0]):
        one_state(j, flats[j], n, has_pawn, gz, T3, has3, i16_3, T4, has4, i16_4,
                  TP, tp_has, tp_i16, tp_key,
                  o_valid, o_tab, o_deg, o_pmin, o_nch, o_ch, o_nkn, o_kv,
                  b, c, mf, mt, hf, ht, hc, ws, wz)

# ---------------- Python driver ----------------
def _open_raw(path, k):
    elems = (32 if "Pawn" in os.path.basename(path) else 10) * 64 ** k * 2
    size = os.path.getsize(path)
    if size not in (elems, 2 * elems):
        raise ValueError(f"{path}: unexpected size {size} (expected {elems} or {2*elems})")
    a = np.asarray(np.memmap(path, dtype=np.uint8, mode="r"))
    a.flags.writeable = False
    return a, int(size == 2 * elems)

def enum_valid(lo, hi, n, gz, has_pawn):
    flat = np.arange(lo, hi, dtype=np.int64)
    rest = flat >> 1
    bk = rest & 63; rest = rest >> 6
    ps = [None] * n
    for i in range(n - 1, -1, -1):
        ps[i] = rest & 63; rest = rest >> 6
    wk = (P_WK_SQ if has_pawn else NP_WK_SQ)[rest].astype(np.int64)
    sq = [wk] + ps + [bk]
    mask = np.ones(hi - lo, bool)
    for a in range(len(sq)):
        for c in range(a + 1, len(sq)):
            mask &= sq[a] != sq[c]
    for i in range(n):
        if gz[i] == 0: mask &= (ps[i] >= 8) & (ps[i] < 56)
    for i in range(n - 1):
        if gz[i] == gz[i + 1]:
            mask &= ps[i] <= ps[i + 1]
            if gz[i] == 2:
                mask &= ((ps[i] // 8 + ps[i] % 8) % 2) != ((ps[i + 1] // 8 + ps[i + 1] % 8) % 2)
    return flat[mask]

class SameSideKernel:
    def __init__(self, names, tb_dir, suffix, promo_path=None):
        self.names = list(names); self.n = len(names)
        self.gz = np.array([Z_BY_NAME[x] for x in names], np.int64)
        self.has_pawn = "Pawn" in names
        dummy = np.zeros(1, np.uint8); dummy.flags.writeable = False
        t3 = [dummy] * 5; self.has3 = np.zeros(5, np.int8); self.i16_3 = np.zeros(5, np.int8)
        for nm, z in Z_BY_NAME.items():
            p = os.path.join(tb_dir, f"K_{nm}_K{suffix}")
            if os.path.exists(p):
                t3[z], self.i16_3[z] = _open_raw(p, 2); self.has3[z] = 1
        t4 = [dummy] * 25; self.has4 = np.zeros(25, np.int8); self.i16_4 = np.zeros(25, np.int8)
        for ra in range(5):
            for rb in range(ra, 5):
                p = os.path.join(tb_dir, f"K_{RANK_NAMES[ra]}_{RANK_NAMES[rb]}_K{suffix}")
                if os.path.exists(p):
                    k = ra * 5 + rb
                    t4[k], self.i16_4[k] = _open_raw(p, 3); self.has4[k] = 1
        self.T3, self.T4 = tuple(t3), tuple(t4)
        self.TP, self.tp_has, self.tp_i16 = dummy, 0, 0
        self.tp_key = np.zeros(3, np.int64)
        if self.n == 3 and self.has_pawn:
            pn = list(names); pn[pn.index("Pawn")] = "Queen"
            pn.sort(key=lambda s: (CR[Z_BY_NAME[s]], s))
            promo_path = os.path.join(tb_dir, f"K_{pn[0]}_{pn[1]}_{pn[2]}_K{suffix}")
            self.tp_key = np.array([CR[Z_BY_NAME[x]] for x in pn], np.int64)
        if self.n >= 2 and self.has_pawn and promo_path and os.path.exists(promo_path):
            self.TP, self.tp_i16 = _open_raw(promo_path, self.n + 0 if False else (3 if self.n == 2 else 4))
            self.tp_has = 1

    def raw(self, flats):
        k = flats.size
        o = (np.zeros(k, np.uint8), np.zeros(k, np.int16), np.zeros(k, np.uint16),
             np.zeros(k, np.uint16), np.zeros(k, np.int16), np.empty((k, MAXM), np.uint32),
             np.zeros(k, np.int16), np.empty((k, MAXK), np.uint16))
        process_chunk(flats, self.n, self.has_pawn, self.gz, self.T3, self.has3, self.i16_3,
                      self.T4, self.has4, self.i16_4, self.TP, self.tp_has, self.tp_i16,
                      self.tp_key, *o)
        return o

    def run_range(self, rng):
        lo, hi = rng
        flats = enum_valid(lo, hi, self.n, self.gz, self.has_pawn)
        return hi - lo, _compact(flats, *self.raw(flats))

def _compact(flats, valid, tab, deg, pmin, nch, ch, nkn, kv):
    idx = np.flatnonzero(valid)
    f = flats[idx].astype(np.uint32)
    pm = pmin[idx]
    w = np.flatnonzero(pm)
    nc = nch[idx].astype(np.int64)
    rows, cols = np.nonzero(np.arange(MAXM)[None, :] < nc[:, None])
    child = ch[idx[rows], cols]
    par = rows.astype(np.uint32)
    is_w = (f[rows] & 1) == 0
    e_b2w = np.ascontiguousarray(np.stack([child[is_w], par[is_w]], axis=1))
    e_w2b = np.ascontiguousarray(np.stack([child[~is_w], par[~is_w]], axis=1))
    nk = nkn[idx].astype(np.int64)
    r2, c2 = np.nonzero(np.arange(MAXK)[None, :] < nk[:, None])
    return (f, tab[idx].copy(), deg[idx].copy(),
            w.astype(np.uint32), pm[w].astype(np.uint16),
            r2.astype(np.uint32), kv[idx[r2], c2].astype(np.uint16),
            e_b2w, e_w2b)

def ordered_map(fn, items, workers, window=None):
    """Ordered, bounded-memory parallel map over threads (kernel releases the GIL)."""
    window = window or workers * 3
    it = iter(items)
    with ThreadPoolExecutor(workers) as ex:
        pending = deque(ex.submit(fn, x) for x in itertools.islice(it, window))
        while pending:
            yield pending.popleft().result()
            nxt = next(it, None)
            if nxt is not None: pending.append(ex.submit(fn, nxt))