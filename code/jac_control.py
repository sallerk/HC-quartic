# jac_control.py -- negative control for jac_check.py: the test "L_P^(N) == (L_E^(N))^2 for some N | 24"
# must FAIL when E_i is replaced by an elliptic curve E' that is not geometrically isogenous to E_i
# (checked at primes p = 1 mod 4 where E' has good ordinary reduction; E': y^2 = x^3 + x + 3 has discriminant
# -16*247 = -16*13*19, so p = 13 is skipped).  Also: the quotient L_C / L_{E_F} must FAIL to be
# divisible when E_F is replaced by a wrong curve (here: E_F of a different random form).
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # find jac_check.py next to this file
from jac_check import count_curve, Lpoly_from_counts, polydiv, frobN, polymul, a_Ei, disc_nonzero_mod

def a_curve(p, A, B):
    chi = lambda a: 0 if a % p == 0 else (1 if pow(a % p, (p - 1) // 2, p) == 1 else -1)
    return -sum(chi(x ** 3 + A * x + B) for x in range(p))

lines = []; res = []
F = [3, -2, 5, 1, 7]          # a random quartic
G = [1, 4, -1, 0, 2]          # another one, for the wrong-E_F control
for p in [13, 17, 29, 37, 41]:
    if not (disc_nonzero_mod(F, p) and disc_nonzero_mod(G, p)):
        continue
    if (4 * 1 ** 3 + 27 * 3 ** 2) % p == 0:
        lines.append("skip p=%d (bad reduction of E': 4A^3 + 27B^2 = 247 = 13*19)" % p); continue
    if a_curve(p, 1, 3) % p == 0:
        lines.append("skip p=%d (E' is supersingular)" % p); continue
    Ns = [count_curve(F, p, k, 4)[0] for k in (1, 2, 3)]
    LC = Lpoly_from_counts(Ns, p, 3)
    aE = p + 1 - count_curve(F, p, 1, 4)[1]
    LP, ok = polydiv(LC, [1, -aE, p])
    ai = a_Ei(p); ap = a_curve(p, 1, 3)   # E': y^2 = x^3 + x + 3
    good_i = [N for N in (1, 2, 3, 4, 6, 8, 12, 24) if frobN(LP, N) == polymul(frobN([1, -ai, p], N), frobN([1, -ai, p], N))]
    good_p = [N for N in (1, 2, 3, 4, 6, 8, 12, 24) if frobN(LP, N) == polymul(frobN([1, -ap, p], N), frobN([1, -ap, p], N))]
    aG = p + 1 - count_curve(G, p, 1, 4)[1]
    _, okG = polydiv(LC, [1, -aG, p])
    line = 'p=%d  a(E_i)=%d a(E\')=%d (disc %d)  E_i^2 match N=%s   E\'^2 match N=%s   L_C divisible by wrong L_{E_G}: %s' % (
        p, ai, ap, ap * ap - 4 * p, good_i, good_p, okG)
    lines.append(line); res.append(dict(p=p, a_Ei=ai, a_Eprime=ap, match_Ei=good_i, match_Eprime=good_p, wrong_EF_divides=okG,
                                        a_EF=aE, a_EG=aG))
# The wrong-E_F control is uninformative at a prime where E_F and E_G have the same trace of Frobenius
# (then L_{E_G} = L_{E_F} and the divisibility is automatic).  Report every such prime explicitly.
for r in res:
    if r['wrong_EF_divides']:
        lines.append('note: at p=%d, a_p(E_F)=%d and a_p(E_G)=%d: %s' % (
            r['p'], r['a_EF'], r['a_EG'],
            'equal traces, so L_{E_G} = L_{E_F} at this prime and the divisibility is automatic'
            if r['a_EF'] == r['a_EG'] else 'UNEXPLAINED'))
open('jac_control.out', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
