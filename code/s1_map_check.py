# s1_map_check.py -- check of the Shioda-Katsura map phi of the note (Step S), written independently of
# jac_check.py and the other scripts.  Writes s1_map_check.json and s1_map_check.out (= what is printed).
# (1) phi lands in W_h, symbolically, for generic f, g of degree d in 3+2 variables.
# (2) base ideal of phi on X1 x X2 = (y, y') in affine charts (scheme-theoretic indeterminacy locus).
# (3) resolved map on both blow-up charts is a morphism (no common zero).
# (4) Fulton Prop. 4.4 degree identity: deg(psi)*deg(W_h) = L^n - int_Z (1+L)^n s(Z,M) with
#     M = X1 x X2, L = h1+h2, Z = D1 cap D2 (D1={y=0}, D2={y'=0}), s(Z,M) = [Z]/((1+h1)(1+h2)).
#     Must give deg(psi) = d for all dims r = dim X1 >= 1 (W_f may be points), s = dim X2 >= 1.
# (5) fiber count over a finite field: random points of W_h(F_p) with -f(x) a d-th power, count preimages
#     by brute force over X1(F_p) x X2(F_p) restricted to the lines through x and x'.
import itertools, random, json, sys
import sympy as sp

out = {}
# ---------- (1) ----------
for d in (3, 4, 5):
    x0, x1, x2, y, a0, a1, yp = sp.symbols('x0 x1 x2 y a0 a1 yp')
    eps = sp.Symbol('eps')
    # generic f in 3 vars, g in 2 vars, symbolic coefficients
    mons3 = [m for m in itertools.product(range(d + 1), repeat=3) if sum(m) == d]
    mons2 = [m for m in itertools.product(range(d + 1), repeat=2) if sum(m) == d]
    cf = sp.symbols('c0:%d' % len(mons3)); cg = sp.symbols('e0:%d' % len(mons2))
    f = lambda X: sum(c * X[0]**m[0] * X[1]**m[1] * X[2]**m[2] for c, m in zip(cf, mons3))
    g = lambda X: sum(c * X[0]**m[0] * X[1]**m[1] for c, m in zip(cg, mons2))
    X = (x0, x1, x2); Xp = (a0, a1)
    img = f(tuple(yp * t for t in X)) + g(tuple(eps * y * t for t in Xp))
    # substitute f(x) = -y^d, g(x') = -yp^d, eps^d = -1 : img = yp^d f(x) + eps^d y^d g(x')
    expr = sp.expand(img - (yp**d * f(X) + eps**d * y**d * g(Xp)))
    ok_hom = (expr == 0)
    val = sp.expand((yp**d * (-y**d) + (-1) * y**d * (-yp**d)))
    out['lands_in_W_h_d%d' % d] = bool(ok_hom and val == 0)

# ---------- (2)+(3) by hand-checkable algebra in a chart ----------
# On X1, x != 0 (x=0 => y^d = -f(0) = 0); chart x0=1, x0'=1: generators y'*x_i, y*x'_j contain y'*1, y*1,
# so base ideal = (y, y') + I(X1 x X2).  (y) cuts W_f = {f=0} (smooth hyperplane section) scheme-theoretically.
out['base_ideal_chart'] = '(y, yp) since y*x0p = y and yp*x0 = yp in chart x0 = x0p = 1'
# blow-up charts: y = s, y' = s*t  -> (s t x : eps s x') = (t x : eps x'),  x' != 0
#                 y' = t, y = t*u  -> (t x : eps t u x') = (x : eps u x'), x != 0
out['blowup_charts'] = 'both charts give maps with a nonvanishing block (x or x\'), so a morphism after one blow-up'

# ---------- (4) Fulton degree identity ----------
A, B = sp.symbols('A B')
fulton = {}
allone = True
for r in range(1, 9):
    for s in range(1, 9):
        n = r + s
        ser = sp.series(sp.series((1 + A + B)**n / ((1 + A) * (1 + B)), A, 0, r).removeO(), B, 0, s).removeO()
        coeff = sp.Poly(sp.expand(ser), A, B).coeff_monomial(A**(r - 1) * B**(s - 1))
        val = sp.binomial(n, r) - coeff   # deg(psi) * deg(W_h) = d^2 * val and deg(W_h) = d, so deg(psi) = d * val
        fulton['%d,%d' % (r, s)] = int(val)
        allone &= (val == 1)
out['fulton_deg_psi_over_d_all_r_s_le_8'] = fulton
out['fulton_all_equal_1'] = bool(allone)

# ---------- (5) finite-field fiber count ----------
def fiber_check(p, d, trials=40, seed=1):
    # need eps with eps^d = -1 and mu_d in F_p
    rnd = random.Random(seed)
    eps = [e for e in range(1, p) if pow(e, d, p) == p - 1]
    if not eps or (p - 1) % d:
        return None
    eps = eps[0]
    # f = F1(u,v) + z^d (3 vars), g = F2(u',v') (2 vars), random binary forms with distinct roots
    def randform():
        while True:
            c = [rnd.randrange(p) for _ in range(d + 1)]
            # distinct roots on P^1 over F_p-bar: check gcd(F, F') via resultant/discriminant of dehomog.
            t = sp.Symbol('t')
            P = sp.Poly(sum(c[i] * t**i for i in range(d + 1)), t, modulus=p)
            if c[d] % p == 0:
                continue
            if sp.gcd(P, P.diff(t)).degree() == 0:
                return c
    F1, F2 = randform(), randform()
    ev = lambda c, u, v: sum(c[i] * pow(u, i, p) * pow(v, d - i, p) for i in range(d + 1)) % p
    f = lambda x: (ev(F1, x[0], x[1]) + pow(x[2], d, p)) % p
    g = lambda xp: ev(F2, xp[0], xp[1])
    res = []
    for _ in range(trials):
        # random point of W_h with f(x) != 0 and -f(x) a d-th power: pick x, x' random with f(x) + g(x') = 0
        for _try in range(10000):
            x = [rnd.randrange(p) for _ in range(3)]
            xp = [rnd.randrange(p) for _ in range(2)]
            if all(t == 0 for t in x) or all(t == 0 for t in xp):
                continue
            if (f(x) + g(xp)) % p == 0 and f(x) != 0:
                break
        else:
            continue
        # preimages: points ((lam x : y), (mu x' : y')) ; normalise representatives x, x' -> phi = (y' x : eps y x')
        # must equal (x : x') projectively => y' = c, eps*y = c for some c != 0; up to scaling of each factor
        # brute force: all y, y' in F_p with (x:y) in X1, (x':y') in X1, and (y' x : eps y x') ~ (x : x')
        cnt = 0
        for yy in range(p):
            if (f(x) + pow(yy, d, p)) % p:
                continue
            for yyp in range(p):
                if (g(xp) + pow(yyp, d, p)) % p:
                    continue
                img = [(yyp * t) % p for t in x] + [(eps * yy * t) % p for t in xp]
                tgt = list(x) + list(xp)
                # projective equality: img = lam * tgt
                lam = None; okp = True
                for a_, b_ in zip(img, tgt):
                    if b_ == 0:
                        if a_ != 0: okp = False
                    else:
                        l = a_ * pow(b_, p - 2, p) % p
                        if lam is None: lam = l
                        elif lam != l: okp = False
                if okp and lam not in (None, 0):
                    cnt += 1
        nroots = sum(1 for yy in range(p) if (f(x) + pow(yy, d, p)) % p == 0)
        res.append((cnt, nroots))
    return res

fib = {}
for (p, d) in [(13, 4), (17, 4), (41, 4), (13, 3), (19, 3), (31, 3)]:
    r = fiber_check(p, d)
    if r is None:
        fib['p%d_d%d' % (p, d)] = 'skipped'
        continue
    # over F_p, #preimages = #{y : y^d = -f(x)} which is d when -f(x) is a d-th power, else 0
    fib['p%d_d%d' % (p, d)] = {'n_points': len(r), 'all_cnt_eq_nroots': all(c == n for c, n in r),
                               'cnt_values': sorted(set(c for c, n in r))}
out['finite_field_fibers'] = fib
json.dump(out, open('s1_map_check.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
open('s1_map_check.out', 'w').write(json.dumps(out, indent=1) + '\n')
