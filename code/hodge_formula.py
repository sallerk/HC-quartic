# hodge_formula.py -- evaluates the closed formula of Proposition 6.1 of the note for dim Hdg^m(X),
#   X = X(F_0,...,F_m) = {F_0(u_0,v_0) + ... + F_m(u_m,v_m) = 0} in P^(2m+1),
# and compares it with the Delsarte character counts of hodge_count.py (Section 7 of the note).
# Written from the statement of the proposition, independently of hodge_count.py.
#
# dim Hdg^m(X) = 1 + sum_chi 3^n0 2^(n1+n3) binom(|K_i|, |K_i|/2 + (n3-n1)/4) prod_k nu(K_k),
# chi in (Z/4)^(m+1) with sum 0; n_c = #{s : chi_s = c}; K = {s : chi_s = 2}; K_i = {s in K : E_s isogenous
# to E_i (j = 1728)}; K \ K_i = K_1 u ... u K_t by isogeny class; nu = 0 for odd |K_k|, otherwise
# binom(2e, e) (CM) or the Catalan number binom(2e, e)/(e+1) (no CM), e = |K_k|/2.
# Input per block: (isogeny class label, has CM); the label 'i' is the class of E_i.
import itertools, json
from fractions import Fraction
from math import comb


def binom(a, b):
    if b.denominator != 1:
        return 0
    b = int(b)
    return comb(a, b) if 0 <= b <= a else 0


def hodge_dim(blocks):
    m = len(blocks) - 1
    total, hmm, prim = 1, 0, 0
    for chi in itertools.product(range(4), repeat=m + 1):
        if sum(chi) % 4:
            continue
        n0, n1, n3 = chi.count(0), chi.count(1), chi.count(3)
        K = [s for s in range(m + 1) if chi[s] == 2]
        mult = 3 ** n0 * 2 ** (n1 + n3)
        prim += mult * 2 ** len(K)                                  # dim V(chi)
        hmm += mult * binom(len(K), Fraction(len(K), 2) + Fraction(n3 - n1, 4))   # dim of its (m,m)-part
        Ki = [s for s in K if blocks[s][0] == 'i']
        N = binom(len(Ki), Fraction(len(Ki), 2) + Fraction(n3 - n1, 4))
        classes = {}
        for s in K:
            if blocks[s][0] != 'i':
                classes.setdefault(blocks[s][0], []).append(s)
        for ss in classes.values():
            if len(ss) % 2:
                N = 0
                break
            e = len(ss) // 2
            N *= comb(2 * e, e) if blocks[ss[0]][1] else comb(2 * e, e) // (e + 1)
        total += mult * N
    return dict(hodge=total, hmm_prim=hmm, dim_prim=prim)


if __name__ == '__main__':
    F = ('i', True)    # u^4 + v^4: E_F has j = 1728
    J = ('r', True)    # u^4 + u v^3: E_J has j = 0 (CM by Z[(1+sqrt(-3))/2]), not isogenous to E_i
    G = ('g', False)   # u^4 + (7/3) u^2 v^2 + v^4: j(E_G) = 61918288/1521, no CM
    H = ('h', False)   # a second curve without CM, not isogenous to E_G
    delsarte = {'FF': 20, 'FJ': 18, 'JJ': 20, 'FFF': 142, 'FFJ': 122, 'FJJ': 114, 'JJJ': 118,
                'FFFF': 1108, 'FFFJ': 928, 'FFJJ': 826, 'FJJJ': 784, 'JJJJ': 820}   # hodge_count.py, note Sec. 7
    out, lines, ok = {}, [], True
    lines.append('Proposition 6.1 vs the Delsarte counts of hodge_count.py (F = u^4+v^4, J = u^4+uv^3)')
    for name, want in delsarte.items():
        m = len(name) - 1
        r = hodge_dim([F if c == 'F' else J for c in name])
        good = r['hodge'] == want and r['dim_prim'] == (9 ** (m + 1) + 3) // 4
        ok &= good
        out[name] = dict(r, delsarte=want, agree=good)
        lines.append('   %-5s m=%d  formula %5d  Delsarte %5d  h^{m,m}_prim %5d  dim H_prim %5d  %s'
                     % (name, m, r['hodge'], want, r['hmm_prim'], r['dim_prim'], 'agree' if good else 'DISAGREE'))
    lines.append('Blocks without CM (G = u^4+(7/3)u^2v^2+v^4; H another curve without CM, not isogenous to E_G)')
    for name, bl in [('GG', [G, G]), ('GH', [G, H]), ('GGG', [G, G, G]), ('GGH', [G, G, H]), ('GGGG', [G] * 4)]:
        r = hodge_dim(bl)
        out[name] = r
        lines.append('   %-5s formula %5d  h^{m,m}_prim %5d' % (name, r['hodge'], r['hmm_prim']))
    lines.append('ALL TWELVE AGREE' if ok else 'SOME DISAGREE')
    open('hodge_formula.out', 'w').write('\n'.join(lines) + '\n')
    json.dump(out, open('hodge_formula.json', 'w'), indent=1)
    print('\n'.join(lines))
