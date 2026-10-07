# exact_incidence.py -- exact intersection pattern of the lines on S = {F(u,v) = F(z,w)} and of the planes on
#   X(F,F,F) = {F(u0,v0) + F(u1,v1) + F(u2,v2) = 0},  F = u^4 + (7/3) u^2 v^2 + v^4,
# used in Section 7 of the note ("Explicit cycles").  lines_check.py and planes_check.py decide which pairs meet
# from floating-point ranks of the linear spans; here the same decisions are made exactly, the two are compared
# pair by pair, and the Gram matrices are formed from the exact data and their rank is computed exactly over Q.
#
# The linear spaces are direct sums over the coordinate blocks V_s = C^2 (coordinates (u_s, v_s)):
#   join spaces   <q_a>_0 + <q_b>_1 (+ <q_c>_2),          q_a = (r_a, 1) for the four roots r_a of F(t,1),
#   graph spaces  {x_s = A x_t} (+ <q_c>_r),               A in GL_2(C) with F o A = F (lines) or F o A = -F (planes).
# Writing pi_A for the permutation of the roots induced by A (A q_b is a multiple of q_{pi_A(b)}), the dimension of
# the intersection of two such linear subspaces is (by the block structure):
#   join/join     number of blocks in which the roots agree;
#   join/graph    [pi_A(b_t) = b_s] + [b_r = c]                     (b = roots of the join space);
#   graph/graph   same pair of blocks: dim ker(A - A') + [c = c'];
#                 different pairs, sharing the block k:  [pi_B(c') = pi_B'(c)], where B (resp. B') is the map of the
#                 first (resp. second) space from its non-shared block to block k, i.e. A or A^{-1}.
# These rules need only pi_A and dim ker(A - A'), which are computed exactly with SymPy: a quantity is declared zero
# only if its minimal polynomial over Q is x, and non-zero if a 50-digit evaluation exceeds 1e-30 in absolute value.
# The rules are then compared with the floating-point ranks of the spans (all pairs), and the Gram matrices
#   lines:  h^2 = 4, h.L = 1, L^2 = -2, L.L' = 1 if the lines meet, else 0;
#   planes: h^4 = 4, h^2.P = 1, P^2 = 7, P.P' = 1 (one point), -2 (a line), 0 (disjoint)
# are formed from the exact data; their rank is computed exactly over Q (python-flint).
# Writes exact_incidence.out (= what is printed) and exact_incidence.json.
import itertools, json, sys
import numpy as np
import sympy as sp
import flint

X = sp.Symbol('X')
I = sp.I


def is_zero(e):
    v = sp.N(e, 50)
    if abs(complex(v)) > 1e-30:
        return False
    mp_ = sp.minimal_polynomial(e, X)
    assert mp_ == X, ('small but not provably zero', e, v, mp_)
    return True


u, v = sp.symbols('u v')
Fpoly = u ** 4 + sp.Rational(7, 3) * u ** 2 * v ** 2 + v ** 4

# roots of F(t,1) = t^4 + 7/3 t^2 + 1: t^2 = (-7 +- sqrt 13)/6 (both negative), i.e. t = +-beta, +-1/beta
beta = I * sp.sqrt((7 - sp.sqrt(13)) / 6)
roots = [beta, -beta, 1 / beta, -1 / beta]
for r in roots:
    assert is_zero(sp.expand(Fpoly.subs({u: r, v: 1})))
assert all(not is_zero(roots[i] - roots[j]) for i in range(4) for j in range(i + 1, 4))
Q = [sp.Matrix([r, 1]) for r in roots]

# matrices with F o A = F: lambda * M, lambda in mu_4, M in {1, diag(1,-1), swap, [[0,1],[-1,0]]};
# with F o A = -F: eps * (those), eps = exp(i pi/4), eps^4 = -1.
Ms = [sp.Matrix([[1, 0], [0, 1]]), sp.Matrix([[1, 0], [0, -1]]), sp.Matrix([[0, 1], [1, 0]]),
      sp.Matrix([[0, 1], [-1, 0]])]
lams = [1, I, -1, -I]
eps = (1 + I) / sp.sqrt(2)
STAB = [lam * M for lam in lams for M in Ms]
ANTI = [eps * A for A in STAB]


def F_of(A):
    x = A * sp.Matrix([u, v])
    return sp.expand(Fpoly.subs({u: x[0], v: x[1]}, simultaneous=True))


def poly_is_zero(p):
    P = sp.Poly(p, u, v)
    return all(is_zero(c) for c in P.coeffs()) if P.coeffs() else True


for A in STAB:
    assert poly_is_zero(F_of(A) - Fpoly)
for A in ANTI:
    assert poly_is_zero(F_of(A) + Fpoly)
# the 16 matrices of each kind are distinct
for L in (STAB, ANTI):
    for i, j in itertools.combinations(range(16), 2):
        assert not all(is_zero(x) for x in (L[i] - L[j]))


def perm_of(A):
    out = []
    for b in range(4):
        w = A * Q[b]
        hits = [a for a in range(4) if is_zero(sp.expand(w[0] * Q[a][1] - w[1] * Q[a][0]))]
        assert len(hits) == 1, (A, b, hits)
        out.append(hits[0])
    return tuple(out)


def kerdim(A, B):
    D = A - B
    if all(is_zero(x) for x in D):
        return 2
    return 1 if is_zero(sp.expand(D.det())) else 0


PI_STAB = [perm_of(A) for A in STAB]
PI_ANTI = [perm_of(A) for A in ANTI]
PI_ANTI_INV = [tuple(sorted(range(4), key=lambda b: p[b])) for p in PI_ANTI]   # permutation of A^{-1}
KER_STAB = [[kerdim(A, B) for B in STAB] for A in STAB]
KER_ANTI = [[kerdim(A, B) for B in ANTI] for A in ANTI]

# ---------------------------------------------------------------- lines on S = {F(u,v) = F(z,w)}: blocks 0 = (u,v), 1 = (z,w)
lines = [('join', a, b) for a in range(4) for b in range(4)] + [('graph', k) for k in range(16)]


def dim_lines(L1, L2):
    if L1[0] == 'join' and L2[0] == 'join':
        return int(L1[1] == L2[1]) + int(L1[2] == L2[2])
    if L1[0] == 'graph' and L2[0] == 'graph':
        return KER_STAB[L1[1]][L2[1]]
    J, G = (L1, L2) if L1[0] == 'join' else (L2, L1)
    return int(PI_STAB[G[1]][J[2]] == J[1])          # graph {x_0 = A x_1} meets <q_a> + <q_b> iff pi_A(b) = a


# ---------------------------------------------------------------- planes on X(F,F,F); pairs (s,t) with s < t, r the third block
PAIRS = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]
planes = [('join', (a0, a1, a2)) for a0 in range(4) for a1 in range(4) for a2 in range(4)]
planes += [('graph', st, k, c) for st in range(3) for k in range(16) for c in range(4)]


def to_block(P, frm, to):
    # the matrix of the graph space P = {x_s = A x_t} from block frm to block to ({frm, to} = {s, t}): A or A^{-1}
    s, t, r = PAIRS[P[1]]
    return PI_ANTI[P[2]] if (frm, to) == (t, s) else PI_ANTI_INV[P[2]]


def dim_planes(P1, P2):
    if P1[0] == 'join' and P2[0] == 'join':
        return sum(int(x == y) for x, y in zip(P1[1], P2[1]))
    if P1[0] == 'graph' and P2[0] == 'graph':
        if P1[1] == P2[1]:
            return KER_ANTI[P1[2]][P2[2]] + int(P1[3] == P2[3])
        s1, t1, r1 = PAIRS[P1[1]]; s2, t2, r2 = PAIRS[P2[1]]
        k = ({s1, t1} & {s2, t2}).pop()
        # P1 = {x_k = B1 x_r2, x_r1 in <q_c1>} and P2 = {x_k = B2 x_r1, x_r2 in <q_c2>}; a common non-zero vector
        # exists iff B1 q_c2 and B2 q_c1 are proportional, and then the intersection is one-dimensional.
        pi1 = to_block(P1, r2, k)        # B1: block r2 -> block k
        pi2 = to_block(P2, r1, k)        # B2: block r1 -> block k
        return int(pi1[P2[3]] == pi2[P1[3]])
    J, G = (P1, P2) if P1[0] == 'join' else (P2, P1)
    s, t, r = PAIRS[G[1]]
    b = J[1]
    return int(PI_ANTI[G[2]][b[t]] == b[s]) + int(b[r] == G[3])


# ---------------------------------------------------------------- floating-point comparison
def num(e):
    return complex(sp.N(e, 30))


Qn = [np.array([num(x) for x in q]) for q in Q]
STABn = [np.array([[num(x) for x in A.row(i)] for i in range(2)]) for A in STAB]
ANTIn = [np.array([[num(x) for x in A.row(i)] for i in range(2)]) for A in ANTI]


def span_lines(L):
    if L[0] == 'join':
        return np.array([np.r_[Qn[L[1]], 0, 0], np.r_[0, 0, Qn[L[2]]]])
    A = STABn[L[1]]
    return np.array([np.r_[A[:, 0], 1, 0], np.r_[A[:, 1], 0, 1]])


def span_planes(P):
    rows = []
    if P[0] == 'join':
        for s, a in enumerate(P[1]):
            x = np.zeros(6, complex); x[2 * s:2 * s + 2] = Qn[a]; rows.append(x)
        return np.array(rows)
    s, t, r = PAIRS[P[1]]
    A = ANTIn[P[2]]
    for k in range(2):
        x = np.zeros(6, complex); x[2 * t + k] = 1; x[2 * s:2 * s + 2] = A[:, k]; rows.append(x)
    x = np.zeros(6, complex); x[2 * r:2 * r + 2] = Qn[P[3]]; rows.append(x)
    return np.array(rows)


def compare(objs, span, dimfun):
    worst_nonzero, worst_zero, disagree = np.inf, 0.0, []
    S = [span(o) for o in objs]
    S = [x / np.linalg.norm(x, axis=1, keepdims=True) for x in S]
    k = S[0].shape[0]
    for i, j in itertools.combinations(range(len(objs)), 2):
        sv = np.linalg.svd(np.vstack([S[i], S[j]]), compute_uv=False)
        d_exact = dimfun(objs[i], objs[j])
        rank_exact = 2 * k - d_exact
        if rank_exact > 0:
            worst_nonzero = min(worst_nonzero, sv[rank_exact - 1])
        if rank_exact < len(sv):
            worst_zero = max(worst_zero, sv[rank_exact])
        rk_num = int((sv > 1e-8).sum())
        if 2 * k - rk_num != d_exact:
            disagree.append((i, j, d_exact, 2 * k - rk_num))
    return disagree, worst_nonzero, worst_zero


def gram(objs, dimfun, self_int, top, deg, values):
    n = len(objs)
    G = [[0] * (n + 1) for _ in range(n + 1)]
    G[0][0] = deg
    for i in range(n):
        G[0][i + 1] = G[i + 1][0] = top
        for j in range(n):
            if i == j:
                G[i + 1][j + 1] = self_int
            else:
                d = dimfun(objs[i], objs[j])
                assert d in values, (objs[i], objs[j], d)
                G[i + 1][j + 1] = values[d]
    return G


out, txt = {}, []
for name, objs, dimfun, span, selfint, values, full in [
        ('lines on S', lines, dim_lines, span_lines, -2, {0: 0, 1: 1}, 2),
        ('planes on X(F,F,F)', planes, dim_planes, span_planes, 7, {0: 0, 1: 1, 2: -2}, 3)]:
    dims = [dimfun(a, b) for a, b in itertools.combinations(objs, 2)]
    assert full not in dims, 'two labels give the same linear space'
    dis, wnz, wz = compare(objs, span, dimfun)
    G = gram(objs, dimfun, selfint, 1, 4, values)
    rk = int(flint.fmpz_mat(G).rank())
    hist = {d: dims.count(d) for d in sorted(set(dims))}
    out[name] = dict(count=len(objs), pairs_by_intersection_dim=hist, float_disagreements=len(dis),
                     smallest_sv_that_must_be_nonzero=float(wnz), largest_sv_that_must_be_zero=float(wz),
                     gram_rank_exact_Q=rk)
    txt.append('%s: %d distinct linear spaces; pairs by dimension of the linear intersection %s' % (name, len(objs), hist))
    txt.append('   exact pattern vs floating-point ranks of the spans: %d disagreements '
               '(smallest singular value that must be non-zero %.2e, largest that must be zero %.2e)'
               % (len(dis), wnz, wz))
    txt.append('   rank of the Gram matrix of h and these classes, exactly over Q: %d' % rk)
txt.append('permutations of the roots: F o A = F: %s; F o A = -F: %s' % (sorted(set(PI_STAB)), sorted(set(PI_ANTI))))
open('exact_incidence.out', 'w').write('\n'.join(txt) + '\n')
json.dump(out, open('exact_incidence.json', 'w'), indent=1)
print('\n'.join(txt))
