# jac_check.py -- check Jac(C_F) ~ E_F x E_i^2 for C_F: w^4 + F(u,v) = 0 (d = 4),
# and the eigenspace Hodge types for d = 4, 5, 6, by exact point counts over finite fields.
# (A numerical consistency check at finitely many primes; the proof of the isogeny is in the note.)
#
# Checks (all exact integer / Gaussian-integer / p-adic arithmetic):
#  (A) d = 4: L_C(T) (from #C(F_{p^k}), k = 1,2,3) is divisible by L_{E_F}(T), E_F: y^2 = -F(u,v);
#      the quotient L_P(T) (degree 4) satisfies L_P^{(N)} = (L_{E_i}^{(N)})^2 over F_{p^N} for some N | 24,
#      E_i: y^2 = x^3 - x.  By Tate's isogeny theorem this means that the reductions mod p of P and E_i^2
#      are isogenous over F_{p^N}.
#  (B) eigenspace types: for p = 1 mod d, Frobenius on the chi^k-eigenspace W_k of H^1(C_F), chi of order d.
#      det(Frob | W_k) in Z[zeta_d]; its valuations at the primes above p must equal the Hodge numbers
#      h^{1,0}(W_k) at the corresponding embeddings (Newton = Hodge for a CM-type determinant):
#      d=4: (2,0),(1,1),(0,2); d=5: {3,2,1,0}; d=6: {4,3,2,1,0} (as k varies, read at one fixed prime).
#  (C) the holomorphic-differential count: x^a w^b dx / w^{d-1}, a+b <= d-3, eigenvalue zeta^{b+1}.
# One CPU core, pure Python.
import json, math, random, itertools, sys
from fractions import Fraction

# ---------------------------------------------------------------- finite fields GF(p^k) via Zech logs
def poly_mulmod(a, b, mod, p):
    k = len(mod) - 1
    res = [0] * (2 * k - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                res[i + j] = (res[i + j] + x * y) % p
    for i in range(len(res) - 1, k - 1, -1):
        c = res[i]
        if c:
            for j in range(k + 1):
                res[i - k + j] = (res[i - k + j] - c * mod[j]) % p
    return res[:k]

import functools
@functools.lru_cache(maxsize=None)
def build_field(p, k):
    q = p ** k
    if k == 1:
        # primitive root
        fac = factorize(p - 1)
        for g in range(2, p):
            if all(pow(g, (p - 1) // r, p) != 1 for r in fac):
                break
        if p == 2: g = 1
        exp = [0] * (q - 1); log = {}
        x = 1
        for e in range(q - 1):
            exp[e] = x; log[x] = e; x = x * g % p
        def add(a, b): return (a + b) % p
        return q, exp, log, add, lambda c: c % p
    for coeffs in itertools.product(range(p), repeat=k):
        mod = list(coeffs) + [1]  # monic, constant term first
        if mod[0] == 0: continue
        # order of x
        x = [0] * k; x[1 if k > 1 else 0] = 1
        cur = [1] + [0] * (k - 1)
        seen_one = None
        ok = True
        for e in range(1, q - 1):
            cur = poly_mulmod(cur, x, mod, p)
            if cur == [1] + [0] * (k - 1):
                ok = False; break
        if not ok: continue
        cur = poly_mulmod(cur, x, mod, p)
        if cur != [1] + [0] * (k - 1): continue
        # x is primitive
        def enc(v): return sum(c * p ** i for i, c in enumerate(v))
        exp = [0] * (q - 1); log = {}
        cur = [1] + [0] * (k - 1)
        for e in range(q - 1):
            code = enc(cur); exp[e] = code; log[code] = e
            cur = poly_mulmod(cur, x, mod, p)
        def add(a, b):
            r = 0; m = 1
            for i in range(k):
                r += ((a % p + b % p) % p) * m
                a //= p; b //= p; m *= p
            return r
        return q, exp, log, add, lambda c: c % p
    raise RuntimeError('no primitive polynomial')

def factorize(n):
    fs = set(); d = 2
    while d * d <= n:
        while n % d == 0:
            fs.add(d); n //= d
        d += 1
    if n > 1: fs.add(n)
    return fs

def values_on_P1(F, p, k):
    """Return (q, exp, log, list of discrete logs (or None for 0) of -F(u,v) for (u:v) in P^1(F_q))."""
    q, exp, log, add, emb = build_field(p, k)
    d = len(F) - 1  # F = [c_0..c_d], F(u,v) = sum c_i u^i v^(d-i)
    cl = [None if emb(c) == 0 else log[emb(c)] for c in F]
    half = (q - 1) // 2 if q % 2 else 0
    out = []
    # affine points (x:1), x in F_q: F(x,1) = sum c_i x^i
    for xe in [None] + list(range(q - 1)):
        val = 0
        for i, c in enumerate(cl):
            if c is None: continue
            if i == 0:
                term = exp[c]
            else:
                if xe is None: continue
                term = exp[(c + i * xe) % (q - 1)]
            val = add(val, term)
        out.append(None if val == 0 else (log[val] + half) % (q - 1))  # log of -F
    # point (1:0): F(1,0) = c_d
    c = cl[d]
    out.append(None if c is None else (c + half) % (q - 1))
    return q, out

def count_curve(F, p, k, d):
    """#{(u:v:w) : w^d + F(u,v) = 0} over F_{p^k}, plus #E_F: y^2 = -F(u,v) (weighted P(1,1,2))."""
    q, vals = values_on_P1(F, p, k)
    gd = math.gcd(d, q - 1)
    NC = 0; NE = 0
    for e in vals:
        if e is None:
            NC += 1; NE += 1
        else:
            if e % gd == 0: NC += gd
            NE += 2 if e % 2 == 0 else 0
    return NC, NE

def Lpoly_from_counts(Ns, p, g):
    # s_k = p^k + 1 - N_k = sum alpha^k ; L(T) = prod (1 - alpha T)
    s = [None] + [p ** k + 1 - Ns[k - 1] for k in range(1, g + 1)]
    c = [Fraction(1)] + [Fraction(0)] * (2 * g)
    for k in range(1, g + 1):
        acc = Fraction(0)
        for i in range(1, k + 1):
            acc += s[i] * c[k - i]
        c[k] = -acc / k
    for k in range(g + 1, 2 * g + 1):
        c[k] = c[2 * g - k] * Fraction(p) ** (k - g)
    assert all(x.denominator == 1 for x in c)
    return [int(x) for x in c]

def polydiv(a, b):
    a = [Fraction(x) for x in a]; out = [Fraction(0)] * (len(a) - len(b) + 1)
    for i in range(len(out)):
        coef = a[i] / b[0]
        out[i] = coef
        for j, y in enumerate(b):
            a[i + j] -= coef * y
    rem = a[len(out):]
    return [int(x) if x.denominator == 1 else x for x in out], all(x == 0 for x in rem)

def polymul(a, b):
    r = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            r[i + j] += x * y
    return r

def power_sums(L, kmax):
    # L = [1, c1, ..., cn]; roots alpha with L(T) = prod(1 - alpha T)
    n = len(L) - 1; s = [0] * (kmax + 1)
    for k in range(1, kmax + 1):
        acc = -k * L[k] if k <= n else 0
        for i in range(1, min(k - 1, n) + 1):
            acc -= L[i] * s[k - i]
        s[k] = acc
    return s

def frobN(L, N):
    n = len(L) - 1
    s = power_sums(L, n * N)
    t = [None] + [s[j * N] for j in range(1, n + 1)]
    c = [Fraction(1)] + [Fraction(0)] * n
    for k in range(1, n + 1):
        acc = sum(t[i] * c[k - i] for i in range(1, k + 1))
        c[k] = -acc / k
    return [int(x) for x in c]

def a_Ei(p):
    # E_i : y^2 = x^3 - x
    chi = lambda a: 0 if a % p == 0 else (1 if pow(a % p, (p - 1) // 2, p) == 1 else -1)
    return -sum(chi(x ** 3 - x) for x in range(p))

# ---------------------------------------------------------------- eigenspace determinants
def char_sums(F, p, k, d):
    """S_j(q) = sum_{(u:v)} chi_q^j(-F(u,v)), j = 1..d-1, q = p^k = 1 mod d, with chi_q = chi_p o Norm,
    chi_p(g1^e) = zeta_d^e for a fixed generator g1 of F_p^*.  Returned as count vectors over exponents of zeta_d."""
    q, vals = values_on_P1(F, p, k)
    assert (q - 1) % d == 0
    qq, exp, log, add, emb = build_field(p, k)
    p1, exp1, log1, add1, emb1 = build_field(p, 1)
    h = exp[(q - 1) // (p - 1)]          # norm of the generator of F_q^*, an element of F_p (constant poly)
    assert 0 < h < p
    Ln = log1[h]                          # chi_q(g^e) = chi_p(h^e) = zeta^(e * Ln)
    sums = {}
    for j in range(1, d):
        cnt = [0] * d
        for e in vals:
            if e is None: continue
            cnt[(j * e * Ln) % d] += 1
        sums[j] = cnt
    return q, sums

def roots_of_unity_mod(p, d, K):
    # primitive d-th roots of unity in Z/p^K (Hensel lift)
    mod = p ** K
    rs = [r for r in range(1, p) if pow(r, d, p) == 1 and all(pow(r, d // l, p) != 1 for l in factorize(d))]
    out = []
    for r in rs:
        x = r
        for _ in range(K + 2):  # Newton: x <- x - (x^d - 1)/(d x^{d-1})
            x = (x - (pow(x, d, mod) - 1) * pow(d * pow(x, d - 1, mod), -1, mod)) % mod
        out.append(x)
    return out

def vp(n, p):
    if n == 0: return 99
    v = 0
    while n % p == 0:
        n //= p; v += 1
    return v

def eigen_dets(F, p, d, dimW):
    """Exact det(Frob | W_j) as a polynomial in zeta (coefficient list mod d-th cyclotomic not reduced),
    computed p-adically: for each embedding zeta -> r (r a primitive d-th root of unity in Z_p),
    trace t^(n) = -S_j(p^n) evaluated at r; det from Newton identities. Returns valuations."""
    K = 12; mod = p ** K
    rs = roots_of_unity_mod(p, d, K)
    traces = {}  # n -> {j: cnt}
    for n in range(1, dimW + 1):
        q, sums = char_sums(F, p, n, d)
        traces[n] = sums
    res = {}
    for j in range(1, d):
        row = []
        for r in rs:
            # evaluate t^(n) = -sum_e cnt[e] r^e  (mod p^K)
            t = [None] + [(-sum(c * pow(r, e, mod) for e, c in enumerate(traces[n][j]))) % mod for n in range(1, dimW + 1)]
            # elementary symmetric e_dim via Newton (needs division by n! -- p > dimW so invertible)
            e = [1] + [0] * dimW
            for kk in range(1, dimW + 1):
                acc = 0
                for i in range(1, kk + 1):
                    acc += (-1) ** (i - 1) * e[kk - i] * t[i]
                e[kk] = acc * pow(kk, -1, mod) % mod
            row.append(vp(e[dimW] % mod, p))
        res[j] = row
    return rs, res

def holo_diff_counts(d):
    cnt = {}
    for a in range(d - 2):
        for b in range(d - 2 - a):
            # x^a w^b dx / w^{d-1}: sigma(w) = zeta w acts by zeta^{b-(d-1)} = zeta^{b+1}
            k = (b + 1) % d
            cnt[k] = cnt.get(k, 0) + 1
    return {k: cnt.get(k, 0) for k in range(1, d)}

def disc_nonzero_mod(F, p):
    """Binary form F = sum c_i u^i v^(d-i) has d distinct roots on P^1 over F_p-bar."""
    import sympy
    x = sympy.symbols('x')
    d = len(F) - 1
    cs = [c % p for c in F]
    if cs[d] == 0 and cs[d - 1] == 0:
        return False  # multiple root at infinity
    f = sympy.Poly(sum(c * x ** i for i, c in enumerate(cs)), x, modulus=p)
    if f.is_zero:
        return False
    g = sympy.gcd(f, f.diff(x))
    return g.degree() == 0

def main():
    out = {'A_jacobian_d4': [], 'B_eigen_types': [], 'C_holo': {}}
    lines = []
    random.seed(20261005)
    # F given as [c0..c4], F(u,v) = sum c_i u^i v^(4-i)
    named = {
        'u^4-v^4':       [-1, 0, 0, 0, 1],
        'u^4+v^4':       [1, 0, 0, 0, 1],
        'u^3v-uv^3':     [0, -1, 0, 1, 0],
        'u^4+uv^3 (j=0)': [0, 1, 0, 0, 1],
        'u^4+u^2v^2+v^4': [1, 0, 1, 0, 1],
        'u^4-3u^2v^2+v^4 (a=-3)': [1, 0, -3, 0, 1],
        'v(u^3+4u^2v+2uv^2) (j=8000)': [0, 2, 4, 1, 0],
    }
    for i in range(4):
        while True:
            Fr = [random.randint(-9, 9) for _ in range(5)]
            if Fr[4] != 0 and Fr[0] != 0:
                break
        named['random%d' % (i + 1)] = Fr
    for k in [3, 4, 5, 6]:
        out['C_holo'][k] = holo_diff_counts(k)
    lines.append('(C) holomorphic differentials x^a w^b dx/w^(d-1) on w^d = -F(x): eigenvalue zeta^(b+1) -> count')
    for k in [3, 4, 5, 6]:
        lines.append('   d=%d: %s' % (k, out['C_holo'][k]))

    primes1 = [5, 13, 17, 29, 37]
    primes3 = [3, 7, 11, 19, 23]
    lines.append('(A) d=4: L_C = L_{E_F} * L_P and L_P^(N) == (L_{E_i}^(N))^2 for N in {1,2,3,4,6,8,12,24}')
    for name, F in named.items():
        for p in primes1 + primes3:
            if not disc_nonzero_mod(F, p):
                continue
            Ns = []; NEs = []
            for k in (1, 2, 3):
                NC, NE = count_curve(F, p, k, 4)
                Ns.append(NC); NEs.append(NE)
            LC = Lpoly_from_counts(Ns, p, 3)
            aE = p + 1 - NEs[0]
            LE = [1, -aE, p]
            # consistency of E_F over F_{p^2}: N_E(p^2) = p^2 + 1 - (aE^2 - 2p)
            okE2 = (NEs[1] == p * p + 1 - (aE * aE - 2 * p))
            LP, exact = polydiv(LC, LE)
            ai = a_Ei(p)
            Li = [1, -ai, p]
            goodN = []
            if exact:
                for N in (1, 2, 3, 4, 6, 8, 12, 24):
                    if frobN(LP, N) == polymul(frobN(Li, N), frobN(Li, N)):
                        goodN.append(N)
            rec = dict(F=name, coeffs=F, p=p, N_C=Ns, L_C=LC, a_EF=aE, E_F_p2_consistent=okE2,
                       L_P=LP if exact else None, divisible=exact, a_Ei=ai, N_ok=goodN)
            out['A_jacobian_d4'].append(rec)
            lines.append('   %-28s p=%-3d #C=%s  L_P=%s  divisible=%s  isogenous-to-E_i^2 over F_{p^N} for N in %s'
                         % (name, p, Ns, LP if exact else '-', exact, goodN))
    # (B) eigenspace determinant valuations
    lines.append('(B) valuations of det(Frob|W_j) at the primes above p (one column per prime), j = 1..d-1')
    tests = [
        (4, [1, 0, 0, 0, 1], 'u^4+v^4', [5, 13]),
        (4, named['random1'], 'random1', [13, 17]),
        (4, named['u^4+uv^3 (j=0)'], 'u^4+uv^3', [13]),
        (5, [1, 2, -3, 1, 0, 1], 'quintic A', [11, 31]),
        (5, [2, 0, 1, -1, 3, 1], 'quintic B', [11]),
        (6, [1, -1, 2, 0, 3, 1, 1], 'sextic A', [7, 13]),
    ]
    for d, F, name, ps in tests:
        dimW = d - 2
        for p in ps:
            if not disc_nonzero_mod(F, p):
                lines.append('   skip %s p=%d (bad reduction)' % (name, p)); continue
            rs, res = eigen_dets(F, p, d, dimW)
            rec = dict(d=d, F=name, coeffs=F, p=p, dimW=dimW, valuations=res)
            out['B_eigen_types'].append(rec)
            lines.append('   d=%d %-10s p=%-3d dimW=%d  ' % (d, name, p, dimW) +
                         '  '.join('j=%d:%s' % (j, res[j]) for j in res))
    json.dump(out, open('jac_check.json', 'w'), indent=1, default=str)
    open('jac_check.out', 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))

if __name__ == '__main__':
    main()
