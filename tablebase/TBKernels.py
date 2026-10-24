# TBKernels.py (v1 - Numba retrograde BFS kernel for TablebaseGenerator Stage 3)
import numpy as np
from numba import njit

# st[] layout: 0=alloc cursor, 1=free-list head, 2=free count, 3=solved, 4=pending


@njit(cache=True)
def _push(pool_idx, pool_next, st, bhead, btail, bcnt, d, idx):
    if st[1] != -1:
        n = st[1]
        st[1] = pool_next[n]
        st[2] -= 1
    else:
        n = st[0]
        st[0] += 1
    pool_idx[n] = idx
    pool_next[n] = -1
    t = btail[d]
    if t == -1:
        bhead[d] = n
    else:
        pool_next[t] = n
    btail[d] = n
    bcnt[d] += 1


@njit(cache=True)
def _queue_value(local, pool_idx, pool_next, st, bhead, btail, bcnt, state, val, max_dtm):
    if val == 0:
        return False
    curr = np.int64(local[state])
    if curr != 0:
        # only a strictly better (smaller) WIN may overwrite an existing entry
        if not (val > 0 and curr > 0 and val < curr):
            return False
    d = val if val > 0 else -val
    if d > max_dtm:
        raise OverflowError("DTM overflow: value exceeds tablebase range (MAX_DTM)")
    local[state] = val
    _push(pool_idx, pool_next, st, bhead, btail, bcnt, d, state)
    if curr == 0:
        st[3] += 1
    st[4] += 1
    return True


@njit(cache=True)
def _grow(pool_idx, pool_next, need):
    cap = pool_idx.shape[0]
    new_cap = cap * 2
    if new_cap < cap + need + 1024:
        new_cap = cap + need + 1024
    if new_cap > 2147483000:
        raise RuntimeError("BFS queue pool exceeds int32 index range")
    ni = np.empty(new_cap, np.uint32)
    nn = np.empty(new_cap, np.int32)
    ni[:cap] = pool_idx
    nn[:cap] = pool_next
    return ni, nn


@njit(cache=True)
def _relax(edges, s, e, dtm, child_val, local, out_degree, max_child,
           pool_idx, pool_next, st, bhead, btail, bcnt, max_dtm):
    changed = 0
    for k in range(s, e):
        p = np.int64(edges[k])
        if child_val < 0:
            # child is a LOSS -> every parent is a WIN in dtm+1
            if _queue_value(local, pool_idx, pool_next, st, bhead, btail, bcnt,
                            p, dtm + 1, max_dtm):
                changed += 1
            continue
        # child is a WIN -> decrement parent's escape count
        if local[p] != 0:
            continue
        if out_degree[p] == 65535:          # has a draw escape
            continue
        if max_child[p] < dtm:
            max_child[p] = dtm
        out_degree[p] -= 1
        if out_degree[p] == 0:
            p_dtm = np.int64(max_child[p]) + 1
            if _queue_value(local, pool_idx, pool_next, st, bhead, btail, bcnt,
                            p, -p_dtm, max_dtm):
                changed += 1
    return changed


@njit(cache=True)
def bfs_kernel(unsolved, local, out_degree, max_child,
               b2w_head, b2w_edges, w2b_head, w2b_edges,
               wtm_idx, wtm_val, btm_idx, btm_val,
               max_dtm, init_cap, verbose):
    n = local.shape[0]
    nb = max_dtm + 2
    bhead = np.full(nb, -1, np.int64)
    btail = np.full(nb, -1, np.int64)
    bcnt = np.zeros(nb, np.int64)
    pool_idx = np.empty(init_cap, np.uint32)
    pool_next = np.empty(init_cap, np.int32)
    st = np.zeros(5, np.int64)
    st[1] = -1

    # --- seed: everything already solved by Stage 2 ---
    for i in range(n):
        v = np.int64(local[i])
        if v != 0:
            d = v if v > 0 else -v
            if d > max_dtm:
                raise OverflowError("DTM overflow while seeding")
            _push(pool_idx, pool_next, st, bhead, btail, bcnt, d, i)
            st[3] += 1
            st[4] += 1

    # --- seed: white-to-move promotion wins (non-full tables) ---
    for i in range(wtm_idx.shape[0]):
        idx = wtm_idx[i]
        if local[idx] == 0:
            _queue_value(local, pool_idx, pool_next, st, bhead, btail, bcnt,
                         idx, wtm_val[i], max_dtm)

    # --- seed: black-to-move promotion losses (non-full tables) ---
    for i in range(btm_idx.shape[0]):
        idx = btm_idx[i]
        val = btm_val[i]
        if local[idx] == 0 and out_degree[idx] != 65535:
            if val > max_child[idx]:
                max_child[idx] = val
            out_degree[idx] -= 1
            if out_degree[idx] == 0:
                _queue_value(local, pool_idx, pool_next, st, bhead, btail, bcnt,
                             idx, -(np.int64(max_child[idx]) + 1), max_dtm)

    if verbose:
        print("[Stage 3] BFS seeded. Solved:", st[3], "/", n)

    max_dtm_solved = 1 if st[3] > 0 else 0

    for dtm in range(1, max_dtm + 1):
        if bhead[dtm] == -1:
            continue
        max_dtm_solved = dtm
        node = bhead[dtm]
        st[4] -= bcnt[dtm]
        bhead[dtm] = -1
        btail[dtm] = -1
        bcnt[dtm] = 0
        changed = 0

        while node != -1:
            nxt = np.int64(pool_next[node])
            idx = np.int64(pool_idx[node])
            # release the node immediately (idx/nxt already copied)
            pool_next[node] = st[1]
            st[1] = node
            st[2] += 1
            node = nxt

            child_val = np.int64(local[idx])
            if child_val == 0:
                continue
            ad = child_val if child_val > 0 else -child_val
            if ad != dtm:
                continue                      # stale (improved) entry

            if (np.int64(unsolved[idx]) & 1) == 1:
                s = np.int64(b2w_head[idx])
                e = np.int64(b2w_head[idx + 1])
                if (pool_idx.shape[0] - st[0]) + st[2] < (e - s) + 2:
                    pool_idx, pool_next = _grow(pool_idx, pool_next, e - s + 2)
                changed += _relax(b2w_edges, s, e, dtm, child_val, local, out_degree,
                                  max_child, pool_idx, pool_next, st, bhead, btail,
                                  bcnt, max_dtm)
            else:
                s = np.int64(w2b_head[idx])
                e = np.int64(w2b_head[idx + 1])
                if (pool_idx.shape[0] - st[0]) + st[2] < (e - s) + 2:
                    pool_idx, pool_next = _grow(pool_idx, pool_next, e - s + 2)
                changed += _relax(w2b_edges, s, e, dtm, child_val, local, out_degree,
                                  max_child, pool_idx, pool_next, st, bhead, btail,
                                  bcnt, max_dtm)

        if dtm == max_dtm and st[4] > 0:
            raise OverflowError("DTM overflowed the supported tablebase range")

        if verbose:
            print("  [DTM", dtm, "] Solved:", st[3], "/", n,
                  "| Next Q:", bcnt[dtm + 1], "| Held:", st[4], "| New:", changed)

    return max_dtm_solved, st[3]