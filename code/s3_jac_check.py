# s3_jac_check.py -- numerical check of Jac(C_F) ~ E_F x E_i^2 (d = 4) and C_F ~ E_w (d = 3), written from
# scratch independently of jac_check.py.  Writes s3_jac_check.json and s3_jac_check.out (= what is printed).
# d = 4: C_F : w^4 + F(u,v) = 0 (smooth plane quartic, genus 3), E_F : y^2 = -F(u,v), E_i : y^2 = x^3 - x.
#   Claim: Jac(C_F) ~ E_F x E_i^2 over C.  Over F_p this predicts:
#     (a) P_{E_F}(X) divides the Frobenius char. poly P_C(X) exactly (E_F is a quotient, defined over F_p);
#     (b) the cofactor Q(X) (degree 4) is a twist of P_{E_i}(X)^2: for some N, Q^{(N)} = (P_{E_i}^{(N)})^2,
#         where R^{(N)} has the N-th powers of the roots of R.
#   Controls: Q vs E_w^2 (y^2 = x^3 + 1, CM by Z[w]) and vs a non-CM curve y^2 = x^3 + x + 3.
# d = 3: C_F : w^3 + F(u,v) = 0 is genus 1; claim C_F = E_w (j = 0): P_C^{(N)} = P_{E_w}^{(N)} for some N | 6.
# Point counts by brute force over F_{p^k}, k = 1,2,3, with a self-contained GF(p^k) implementation
# (primitive polynomial + log tables).
import numpy as np, json, random, math
from fractions import Fraction

def prime_factors(n):
    fs, q = set(), 2
    while q * q <= n:
        while n % q == 0:
            fs.add(q); n //= q
        q += 1
    if n > 1: fs.add(n)
    return fs

def polymulmod(a, b, m, p):
    # a, b: coeff lists low->high, deg < k; m: monic modulus coeffs low->high (len k+1)
    k = len(m) - 1
    res = [0] * (2 * k - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                res[i + j] = (res[i + j] + x * y) % p
    for deg in range(len(res) - 1, k - 1, -1):
        c = res[deg]
        if c:
            for t in range(k + 1):
                res[deg - k + t] = (res[deg - k + t] - c * m[t]) % p
    return (res + [0] * k)[:k]

def polypowmod(a, e, m, p):
    k = len(m) - 1
    r = [1] + [0] * (k - 1)
    while e:
        if e & 1: r = polymulmod(r, a, m, p)
        a = polymulmod(a, a, m, p); e >>= 1
    return r

class GF:
    def __init__(self, p, k):
        self.p, self.k, self.q = p, k, p ** k
        q = self.q
        one = [1] + [0] * (k - 1)
        if k == 1:
            g = next(g for g in range(2, p) if all(pow(g, (p - 1) // r, p) != 1 for r in prime_factors(p - 1)))
            gen = [g]; m = [(-g) % p, 1]   # X - g, so "X" = g
        else:
            m = None
            for coeffs in __import__('itertools').product(range(p), repeat=k):
                mm = list(coeffs) + [1]
                if mm[0] == 0: continue
                X = [0, 1] + [0] * (k - 2)
                if polypowmod(X, q - 1, mm, p) != one: continue
                if all(polypowmod(X, (q - 1) // r, mm, p) != one for r in prime_factors(q - 1)):
                    m = mm; break
            gen = [0, 1] + [0] * (k - 2)
        self.m = m
        # exp table as digit arrays
        exp = np.zeros((q - 1, k), dtype=np.int64)
        cur = one
        for i in range(q - 1):
            exp[i] = cur
            cur = polymulmod(cur, gen, m, p) if k > 1 else [(cur[0] * gen[0]) % p]
        assert cur == one
        self.exp = exp
        self.pw = p ** np.arange(k)
        enc = exp @ self.pw
        self.log = np.full(q, -1, dtype=np.int64)
        self.log[enc] = np.arange(q - 1)
        assert (self.log[1:] >= 0).all()
        # all elements as digit arrays: index 0 = zero, index 1+i = gen^i
        self.elems = np.vstack([np.zeros((1, k), dtype=np.int64), exp])
        self.elog = np.concatenate([[-1], np.arange(q - 1)])

    def eval_binary(self, coeffs):
        """value of F(t,1) = sum c_i t^i for all t in F_q (in self.elems order), plus F(1,0) = c_top; returns logs (-1 for 0)."""
        p, k, q = self.p, self.k, self.q
        acc = np.zeros((q, k), dtype=np.int64)
        for i, c in enumerate(coeffs):
            c %= p
            if c == 0: continue
            if i == 0:
                term = np.zeros((q, k), dtype=np.int64); term[:, 0] = c
            else:
                lg = self.elog.copy()
                term = np.zeros((q, k), dtype=np.int64)
                nz = lg >= 0
                term[nz] = self.exp[(lg[nz] * i) % (q - 1)]
                term = (term * c) % p
            acc = (acc + term) % p
        enc = acc @ self.pw
        logs = np.where(enc == 0, -1, self.log[enc])
        ctop = coeffs[-1] % p
        top = -1 if ctop == 0 else int(self.log[ctop])   # ctop in F_p subset F_q, encoding = ctop
        return logs, top

def count_power_curve(Fq, coeffs, e, neg=True):
    """#points of {w^e = -F(u,v)} (or +F) over P^1(F_q): sum over (u:v) of #{w: w^e = a}."""
    q = Fq.q
    logs, top = Fq.eval_binary([(-c if neg else c) for c in coeffs])
    g = math.gcd(e, q - 1)
    def cnt(lg):
        return np.where(lg < 0, 1, np.where(lg % g == 0, g, 0))
    return int(cnt(logs).sum() + cnt(np.array([top])).sum())

def squarefree_P1(coeffs, p):
    # binary form sum c_i u^i v^(d-i): distinct roots on P^1 over F_p-bar
    import sympy as sp
    t = sp.Symbol('t')
    c = [x % p for x in coeffs]
    d = len(c) - 1
    P = sp.Poly(list(reversed(c)), t, modulus=p)
    if P.is_zero: return False
    if P.degree() < d - 1: return False
    return sp.gcd(P, P.diff(t)).degree() == 0

def charpoly_from_counts(p, N):
    # genus g = len(N); N[k-1] = #C(F_{p^k}); returns integer coeff list (high->low) of X^{2g} ... p^g
    g = len(N)
    s = [p ** (k + 1) + 1 - N[k] for k in range(g)]   # power sums of Frobenius roots
    e = [1]
    for k in range(1, g + 1):
        val = sum(((-1) ** (i - 1)) * e[k - i] * s[i - 1] for i in range(1, k + 1))
        assert val % k == 0
        e.append(val // k)
    co = [((-1) ** k) * e[k] for k in range(g + 1)]          # X^{2g} - e1 X^{2g-1} + ...
    full = co + [co[g - j] * p ** (j) for j in range(1, g + 1)]   # functional equation
    # full[g + j] = coefficient of X^{g-j} = p^j * coefficient of X^{g+j}
    return full

def polydiv(a, b):
    a = [Fraction(x) for x in a]; out = []
    while len(a) >= len(b):
        c = a[0] / b[0]; out.append(c)
        for i in range(len(b)): a[i] -= c * b[i]
        a.pop(0)
    return out, a

def powersums(poly, M):
    # poly monic high->low; power sums S_0..S_M of roots via Newton's identities
    n = len(poly) - 1
    e = [1] + [((-1) ** k) * poly[k] for k in range(1, n + 1)]
    S = [n]
    for m in range(1, M + 1):
        val = sum(((-1) ** (i - 1)) * e[i] * S[m - i] for i in range(1, min(m - 1, n) + 1))
        if m <= n:
            val += ((-1) ** (m - 1)) * m * e[m]
        S.append(val)
    return S

def poly_of_powers(poly, N):
    n = len(poly) - 1
    S = powersums(poly, n * N)
    s = [S[j * N] for j in range(1, n + 1)]
    e = [1]
    for k in range(1, n + 1):
        val = sum(((-1) ** (i - 1)) * e[k - i] * s[i - 1] for i in range(1, k + 1))
        assert val % k == 0
        e.append(val // k)
    return [((-1) ** k) * e[k] for k in range(n + 1)]

def polymul(a, b):
    r = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            r[i + j] += x * y
    return r

def ell_charpoly(p, a2, a4, a6):
    # y^2 = x^3 + a2 x^2 + a4 x + a6 over F_p
    n = 1
    for x in range(p):
        r = (x ** 3 + a2 * x * x + a4 * x + a6) % p
        n += 1 if r == 0 else (2 if pow(r, (p - 1) // 2, p) == 1 else 0)
    return [1, -(p + 1 - n), p]

NS = [1, 2, 3, 4, 6, 8, 12, 24]
def min_twist_N(Q, target):
    if len(Q) != len(target):
        return None
    for N in NS:
        if poly_of_powers(Q, N) == poly_of_powers(target, N):
            return N
    return None

# self-test of the Newton machinery: roots {2,3} -> powers N=2 -> {4,9}
assert poly_of_powers([1, -5, 6], 2) == [1, -13, 36]
assert powersums([1, -5, 6], 3) == [2, 5, 13, 35]

rnd = random.Random(20261005)
forms4 = {'u4+v4': [1, 0, 0, 0, 1], 'u4-v4': [-1, 0, 0, 0, 1], 'u3v-uv3': [0, -1, 0, 1, 0],
          'u4+uv3': [0, 1, 0, 0, 1], 'u4+2u2v2+3v4': [3, 0, 2, 0, 1],
          'v(u3+u2v+5uv2+7v3)': [7, 5, 1, 1, 0]}
for j in range(6):
    forms4['rand%d' % j] = [rnd.randint(-9, 9) for _ in range(5)]
forms3 = {'u3+v3': [1, 0, 0, 1], 'u3+uv2+v3': [1, 1, 0, 1], 'v(u2+uv+3v2)': [3, 1, 1, 0]}
for j in range(3):
    forms3['rand%d' % j] = [rnd.randint(-9, 9) for _ in range(4)]

primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
fields = {}
def field(p, k):
    if (p, k) not in fields: fields[(p, k)] = GF(p, k)
    return fields[(p, k)]

results = {'d4': [], 'd3': []}
summary = {'d4_pairs': 0, 'd4_EF_divides': 0, 'd4_Q_twist_of_Ei2': 0, 'd4_ctrl_Ew2_match': 0, 'd4_ctrl_nonCM2_match': 0,
           'd4_Nmin_hist': {}, 'd3_pairs': 0, 'd3_twist_of_Ew': 0, 'd3_ctrl_Ei_match': 0}
for p in primes:
    Pi = ell_charpoly(p, 0, -1, 0)      # E_i : y^2 = x^3 - x
    Pw = ell_charpoly(p, 0, 0, 1)       # E_w : y^2 = x^3 + 1   (p != 3 needed for good reduction)
    Pn = ell_charpoly(p, 0, 1, 3)       # non-CM control y^2 = x^3 + x + 3 (disc = -16(4+243) = -16*247 = -16*13*19)
    for name, F in forms4.items():
        if p == 2 or not squarefree_P1(F, p): continue
        N = [count_power_curve(field(p, k), F, 4) for k in (1, 2, 3)]
        PC = charpoly_from_counts(p, N)
        NE = count_power_curve(field(p, 1), F, 2)
        PE = [1, -(p + 1 - NE), p]
        Qf, rem = polydiv(PC, PE)
        divides = all(r == 0 for r in rem)
        rec = {'p': p, 'F': name, 'N1..3': N, 'PC': PC, 'PE_F': PE, 'divides': divides}
        summary['d4_pairs'] += 1
        if divides:
            summary['d4_EF_divides'] += 1
            Q = [int(x) for x in Qf]
            Pi2 = polymul(Pi, Pi)
            nmin = min_twist_N(Q, Pi2)
            rec['Q'] = Q; rec['Nmin_Ei2'] = nmin
            if nmin is not None:
                summary['d4_Q_twist_of_Ei2'] += 1
                summary['d4_Nmin_hist'][str(nmin)] = summary['d4_Nmin_hist'].get(str(nmin), 0) + 1
            if p != 3:
                if min_twist_N(Q, polymul(Pw, Pw)) is not None: summary['d4_ctrl_Ew2_match'] += 1
            if p not in (13, 19):
                if min_twist_N(Q, polymul(Pn, Pn)) is not None: summary['d4_ctrl_nonCM2_match'] += 1
        results['d4'].append(rec)
    if p == 3: continue
    for name, F in forms3.items():
        if not squarefree_P1(F, p): continue
        N = [count_power_curve(field(p, 1), F, 3)]
        PC = charpoly_from_counts(p, N)
        nmin = min_twist_N(PC, Pw)
        summary['d3_pairs'] += 1
        if nmin is not None: summary['d3_twist_of_Ew'] += 1
        if min_twist_N(PC, Pi) is not None: summary['d3_ctrl_Ei_match'] += 1
        results['d3'].append({'p': p, 'F': name, 'N1': N[0], 'PC': PC, 'Nmin_Ew': nmin})

# control counts are only meaningful where the control curve is ordinary and not CM by Q(i):
summary['note'] = ('ctrl matches for E_w^2 are expected exactly when p = 3 mod 4 AND p = 2 mod 3 (both '
                   'supersingular: all Weil numbers are sqrt(p)*roots of unity); count them separately below')
summary['d4_ctrl_Ew2_expected_supersingular_pairs'] = sum(1 for r in results['d4'] if r['divides'] and r['p'] % 4 == 3 and r['p'] % 3 == 2)
json.dump({'summary': summary, 'results': results}, open('s3_jac_check.json', 'w'), indent=1)
bad = [r for r in results['d4'] if not r['divides'] or r.get('Nmin_Ei2') is None]
text = (json.dumps(summary, indent=1) + '\n' + 'd4 failures: ' + str(bad) + '\n'
        + 'd3 failures: ' + str([r for r in results['d3'] if r['Nmin_Ew'] is None]) + '\n')
print(text, end='')
open('s3_jac_check.out', 'w').write(text)
