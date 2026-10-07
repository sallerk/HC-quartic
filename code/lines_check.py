# lines_check.py -- geometric spot check (m = 1; a consistency check, not used in the proof):
# rank of the lattice spanned by H and explicit lines on
#   S = {F(u,v) - F(z,w) = 0}  (P = Q case, F generic: predicted rho = 18 + rank Hom(E_F,E_F) = 19)
#   S = {F0(u,v) - F1(z,w) = 0} with unrelated F0, F1 (predicted rho = 18; lines give only the join part 1+9)
#   Fermat quartic in split form (u^4+v^4 - (z^4+w^4)) (predicted 20; 48 lines known to span NS)
# Lines: join lines {u = a v, z = b w} (a root of F0, b root of F1); graph lines {(u,v) = A(z,w)} with F0 o A = F1.
# Intersection numbers on a K3: L.L = -2, L.L' = 1 if the lines meet, else 0; H.L = 1, H.H = 4.
# The rank of the integer Gram matrix is computed numerically and exactly over Q (python-flint).
import itertools, json
import numpy as np
import flint
import mpmath as mp
mp.mp.dps = 40

def roots_binary(F):
    # F = [c0..c4] for sum c_i x^i (x = u/v); returns list of points (u,v) on P^1
    cs = [mp.mpf(c) for c in F]
    pts = []
    if cs[4] == 0:
        pts.append((mp.mpf(1), mp.mpf(0)))
        poly = cs[:4]
    else:
        poly = cs
    deg = len(poly) - 1
    while deg > 0 and poly[deg] == 0: deg -= 1
    rts = mp.polyroots(list(reversed(poly[:deg + 1])), maxsteps=200, extraprec=200)
    for r in rts: pts.append((r, mp.mpf(1)))
    return pts

def evalF(F, u, v):
    return sum(c * u ** i * v ** (4 - i) for i, c in enumerate(F))

def graph_matrices(F0, F1):
    """All A in GL_2(C) with F0(A(z,w)) = F1(z,w): A maps the roots of F1 to the roots of F0 (as a Mobius map)
    preserving cross-ratio; for each bijection of roots that is induced by a Mobius map, solve for A up to the
    scalar with c^4 = ratio."""
    R0 = roots_binary(F0); R1 = roots_binary(F1)
    mats = []
    for perm in itertools.permutations(range(4)):
        # Mobius M with M(R1[k]) = R0[perm[k]] for k = 0,1,2; check k = 3
        def mob_from3(src, dst):
            # 2x2 matrix sending src points to dst points (projective), via standard construction
            def to_std(p):  # matrix sending p0->0, p1->inf, p2->1  (points as (x,y))
                (a0, b0), (a1, b1), (a2, b2) = p
                M = mp.matrix([[b0, -a0], [b1, -a1]])  # rows: linear forms vanishing at p0, p1
                # scale rows so that p2 -> (1:1)
                r0 = M[0, 0] * a2 + M[0, 1] * b2; r1 = M[1, 0] * a2 + M[1, 1] * b2
                return mp.matrix([[M[0, 0] / r0, M[0, 1] / r0], [M[1, 0] / r1, M[1, 1] / r1]])
            S = to_std(src); Tm = to_std(dst)
            return mp.inverse(Tm) * S
        M = mob_from3([R1[0], R1[1], R1[2]], [R0[perm[0]], R0[perm[1]], R0[perm[2]]])
        x, y = R1[3]
        img = M * mp.matrix([[x], [y]])
        tx, ty = R0[perm[3]]
        if abs(img[0] * ty - img[1] * tx) > mp.mpf(10) ** (-25) * (abs(img[0]) + abs(img[1])):
            continue
        # scale so that F0(M(z,w)) = F1(z,w): compare at a random point
        z0, w0 = mp.mpf('0.37'), mp.mpf('1.21')
        v = M * mp.matrix([[z0], [w0]])
        lam = evalF(F1, z0, w0) / evalF(F0, v[0], v[1])
        c = lam ** (mp.mpf(1) / 4)
        for k in range(4):
            mats.append(M * (c * mp.mpc(0, 1) ** k))
    return mats

def line_basis_join(a, b):
    # {u = a_u/a_v...}: point a = (au, av) on P^1 for (u,v), b = (bz, bw) for (z,w): line spanned by
    # (au, av, 0, 0) and (0, 0, bz, bw)
    return [(a[0], a[1], 0, 0), (0, 0, b[0], b[1])]

def line_basis_graph(A):
    # {(u,v) = A(z,w)}: spanned by (A e1, e1) and (A e2, e2)
    return [(A[0, 0], A[1, 0], 1, 0), (A[0, 1], A[1, 1], 0, 1)]

def meet(L1, L2):
    M = np.array([[complex(x) for x in r] for r in [L1[0], L1[1], L2[0], L2[1]]])
    M = M / np.linalg.norm(M, axis=1, keepdims=True)
    return np.linalg.matrix_rank(M, tol=1e-9) <= 3

def same(L1, L2):
    M = np.array([[complex(x) for x in r] for r in [L1[0], L1[1], L2[0], L2[1]]])
    return np.linalg.matrix_rank(M, tol=1e-9) <= 2

def check_on_S(F0, F1, L):
    # verify that the line lies on F0(u,v) - F1(z,w) = 0 at a few points
    for t in (mp.mpf('0.3'), mp.mpf('1.7'), mp.mpf('-2.2')):
        p = [L[0][i] + t * L[1][i] for i in range(4)]
        val = evalF(F0, p[0], p[1]) - evalF(F1, p[2], p[3])
        if abs(val) > mp.mpf(10) ** (-20) * (1 + max(abs(x) for x in p)) ** 4:
            return False
    return True

def gram_rank(F0, F1, label):
    R0 = roots_binary(F0); R1 = roots_binary(F1)
    lines = [line_basis_join(a, b) for a in R0 for b in R1]
    njoin = len(lines)
    for A in graph_matrices(F0, F1):
        L = line_basis_graph(A)
        if not any(same(L, L2) for L2 in lines):
            lines.append(L)
    assert all(check_on_S(F0, F1, L) for L in lines), 'a line is not on S'
    n = len(lines)
    G = np.zeros((n + 1, n + 1))
    G[0, 0] = 4
    for i in range(n):
        G[0, i + 1] = G[i + 1, 0] = 1
        for j in range(n):
            G[i + 1, j + 1] = -2 if i == j else (1 if meet(lines[i], lines[j]) else 0)
    rk = np.linalg.matrix_rank(G)
    rk_q = int(flint.fmpz_mat([[int(round(x)) for x in row] for row in G]).rank())
    sv = np.linalg.svd(G, compute_uv=False)
    return dict(label=label, join_lines=njoin, graph_lines=n - njoin, rank=int(rk), rank_exact_Q=rk_q,
                smallest_nonzero_sv=float(min(s for s in sv if s > 1e-8)), largest_zero_sv=float(max([s for s in sv if s <= 1e-8] + [0])))

if __name__ == '__main__':
    out = []
    # F generic in normal form u^4 + a u^2 v^2 + v^4 with a = 7/3; S_PQ: F(u,v) - F(z,w)
    Fg = [1, 0, mp.mpf(7) / 3, 0, 1]
    out.append(gram_rank(Fg, Fg, 'P=Q generic (F = u^4 + 7/3 u^2v^2 + v^4): predicted rho = 19'))
    F1 = [2, -1, 3, 1, 5]
    out.append(gram_rank(Fg, F1, 'F0, F1 unrelated: predicted rho = 18, lines give 1 + 9 = 10'))
    Ff = [1, 0, 0, 0, 1]
    out.append(gram_rank(Ff, Ff, 'Fermat u^4+v^4 = z^4+w^4: predicted rho = 20'))
    Fj = [0, 1, 0, 0, 1]  # u^4 + u v^3 (j = 0)
    out.append(gram_rank(Fj, Fj, 'P=Q with j=0 (u^4+uv^3): predicted rho = 20'))
    lines = ['%s\n   join lines %d, graph lines %d, rank<H,lines> = %d (exact over Q: %d; sv gap %.2e / %.2e)' %
             (r['label'], r['join_lines'], r['graph_lines'], r['rank'], r['rank_exact_Q'], r['smallest_nonzero_sv'], r['largest_zero_sv']) for r in out]
    open('lines_check.out', 'w').write('\n'.join(lines) + '\n')
    json.dump(out, open('lines_check.json', 'w'), indent=1)
    print('\n'.join(lines))
