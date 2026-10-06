# planes_check.py -- geometric check (m = 2) of the predicted Hodge count on split quartic 4-folds
#   X = {F0(u0,v0) + F1(u1,v1) + F2(u2,v2) = 0} in P^5.
# (A consistency check for m = 2, where the Hodge conjecture was already known; not used in the proof.)
# Planes used (all are joins of a point with a line, or of three points; cf. Duque Franco - Villaflor Loyola,
# "Periods of join algebraic cycles", arXiv:2312.17222):
#   (a) joins of three points p_s in Z_s = {F_s = 0}:          span{(p0,0,0),(0,p1,0),(0,0,p2)}
#   (b) joins of a graph line of S_st with a point of Z_r:      {(u_s,v_s) = A(u_t,v_t)} + (point of Z_r),
#       F_s o A = -F_t.
# Intersection numbers on a smooth quartic 4-fold: P.P = c2(N) = 7, planes meeting in a line: -2,
# in one point: +1, disjoint: 0; h^2.P = 1, h^2.h^2 = 4.  (c(N_{P/X}) = (1+h)^3/(1+4h) = 1 - h + 7h^2.)
# Predictions (hodge_count.py model): F0=F1=F2 non-CM generic: 109, and every Hodge sector has a zero entry,
# so joins of NS(S_st) (spanned by lines when F_s = F_t up to sign, lines_check) with points should give all 109.
# Sector bookkeeping (hodge_count.py model): every plane of type (a)/(b) is a join p_r * (line on S_st), so its
# class lies in Q h^2 + (sector c_r = 0).  Predicted rank of <h^2, planes> = 1 + #Hodge classes in sectors with a 0:
#   F0=F1=F2 = Fg (non-CM): 1 + 27 + 3*(24 + 3) = 109 = all Hodge classes (model 109).
#   F0=F1=Fg, F2 unrelated: graph lines only for the pair (0,1): 1 + 27 + 24 + 3 = 55 (model Hodge total 103;
#     the other 48 live in sectors (0,1,3),(0,3,1),(1,0,3),(3,0,1) and need non-line curves on S_02, S_12).
#   Fermat u^4+v^4 (x3): 1 + 27 + 3*(24 + 6) = 118 (model 142; the remaining 24 sit in zero-free sectors
#     (1,1,2),(3,3,2) and perms, reached only by the non-split Fermat planes, not used here).
# Rank is computed (i) numerically (SVD gap) and (ii) exactly mod two large primes (a lower bound for rank over Q).
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import itertools, json, sys
import numpy as np
import mpmath as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # find lines_check.py next to this file
from lines_check import roots_binary, graph_matrices

def plane_rows_a(p0, p1, p2):
    return np.array([[complex(p0[0]), complex(p0[1]), 0, 0, 0, 0],
                     [0, 0, complex(p1[0]), complex(p1[1]), 0, 0],
                     [0, 0, 0, 0, complex(p2[0]), complex(p2[1])]])

def plane_rows_b(A, s, t, r, pr):
    # {(u_s,v_s) = A (u_t,v_t)} : vectors with block t = e_k, block s = A e_k ; plus point pr in block r
    rows = []
    for k in range(2):
        v = np.zeros(6, dtype=complex)
        v[2 * t + k] = 1
        v[2 * s] = complex(A[0, k]); v[2 * s + 1] = complex(A[1, k])
        rows.append(v)
    v = np.zeros(6, dtype=complex); v[2 * r] = complex(pr[0]); v[2 * r + 1] = complex(pr[1])
    rows.append(v)
    return np.array(rows)

def evalF(F, u, v):
    return sum(c * u ** i * v ** (4 - i) for i, c in enumerate(F))

def on_X(Fs, P):
    rng = np.random.default_rng(1)
    for _ in range(4):
        c = rng.normal(size=3) + 1j * rng.normal(size=3)
        x = c @ P
        val = sum(evalF([complex(a) for a in Fs[s]], x[2 * s], x[2 * s + 1]) for s in range(3))
        if abs(val) > 1e-8 * (1 + np.abs(x).max()) ** 4:
            return False
    return True

def rank_mod_p(G, p):
    M = [[int(round(x)) % p for x in row] for row in G]
    n = len(M); m = len(M[0]); r = 0
    for c in range(m):
        piv = next((i for i in range(r, n) if M[i][c]), None)
        if piv is None: continue
        M[r], M[piv] = M[piv], M[r]
        inv = pow(M[r][c], p - 2, p)
        M[r] = [x * inv % p for x in M[r]]
        for i in range(n):
            if i != r and M[i][c]:
                f = M[i][c]
                M[i] = [(a - f * b) % p for a, b in zip(M[i], M[r])]
        r += 1
    return r

def run(Fs, label, predicted):
    neg = lambda F: [-c for c in F]
    Z = [roots_binary(F) for F in Fs]
    planes = []
    for p0 in Z[0]:
        for p1 in Z[1]:
            for p2 in Z[2]:
                planes.append(plane_rows_a(p0, p1, p2))
    na = len(planes)
    for (s, t, r) in [(0, 1, 2), (0, 2, 1), (1, 2, 0)]:
        for A in graph_matrices(Fs[s], neg(Fs[t])):   # F_s(A y) = -F_t(y)
            for pr in Z[r]:
                planes.append(plane_rows_b(A, s, t, r, pr))
    # dedupe
    uniq = []
    for P in planes:
        Pn = P / np.linalg.norm(P, axis=1, keepdims=True)
        if not any(np.linalg.matrix_rank(np.vstack([Pn, Q]), tol=1e-8) == 3 for Q in uniq):
            uniq.append(Pn)
    assert all(on_X(Fs, P) for P in uniq), 'plane not on X'
    n = len(uniq)
    G = np.zeros((n + 1, n + 1))
    G[0, 0] = 4
    for i in range(n):
        G[0, i + 1] = G[i + 1, 0] = 1
        for j in range(i, n):
            if i == j:
                g = 7
            else:
                rk = np.linalg.matrix_rank(np.vstack([uniq[i], uniq[j]]), tol=1e-8)
                dim_int = 6 - rk  # = 3 + 3 - rk  (projective dim = dim_int - 1)
                g = {0: 0, 1: 1, 2: -2}[dim_int]
            G[i + 1, j + 1] = G[j + 1, i + 1] = g
    sv = np.linalg.svd(G, compute_uv=False)
    rank = int((sv > 1e-6 * sv.max()).sum())
    gap = (float(min(s for s in sv if s > 1e-6 * sv.max())), float(max([s for s in sv if s <= 1e-6 * sv.max()] + [0])))
    rp = [rank_mod_p(G, p) for p in (1000003, 998244353)]
    return dict(label=label, planes=n, planes_a=na, rank=rank, rank_mod_p=rp, predicted=predicted, sv_gap=gap)

if __name__ == '__main__':
    Fg = [1, 0, mp.mpf(7) / 3, 0, 1]
    Ff = [1, 0, 0, 0, 1]
    F1 = [2, -1, 3, 1, 5]
    res = []
    res.append(run([Fg, Fg, Fg], 'F0=F1=F2 generic non-CM (model Hdg = 109; all Hodge sectors contain a 0)', 109))
    res.append(run([Fg, Fg, F1], 'F0=F1 generic, F2 unrelated (model Hdg = 103; planes reach only sectors with a 0 via pair (0,1))', 55))
    res.append(run([Ff, Ff, Ff], 'Fermat split form, CONTROL (model Hdg = 142; (a)+(b) planes reach only sectors with a 0)', 118))
    lines = ['%s\n   planes %d (point-joins %d), rank<h^2, planes> = %d (exact mod p: %s), predicted %s, sv gap %.2e / %.2e'
             % (r['label'], r['planes'], r['planes_a'], r['rank'], r['rank_mod_p'], r['predicted'], r['sv_gap'][0], r['sv_gap'][1]) for r in res]
    open('planes_check.out', 'w').write('\n'.join(lines) + '\n')
    json.dump(res, open('planes_check.json', 'w'), indent=1)
    print('\n'.join(lines))
