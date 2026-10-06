# hodge_count.py -- Hodge classes on split quartics X = {F_0(u_0,v_0)+...+F_m(u_m,v_m)=0}.
# (Consistency checks only; not used in the proof of the theorem.)
#
# MODEL (the decomposition of H^{2m}(X) coming from the Shioda-Katsura construction and the isogeny
# Jac(w^4 + F = 0) ~ E_F x E_i^2; see README.txt): H^{2m}_prim(X) = sum over sectors c in (Z/4)^{m+1},
# sum c = 0, of V_c = (x)_s P_s(c_s) (x) L_c, with
#   P_s(0): 3-dim Tate (type (1,1)),   P_s(1), P_s(3): 2-dim, CM by Q(i) (pieces of H^1(E_i^2)),
#   P_s(2) = H^1(E_{F_s}),             L_c: 1-dim CM by Q(i) (Fermat piece).
# Jacobian-ring degrees: e_s = c_s + 2 (c_s = 0,1,3) and e_s in {0,4} (c_s = 2); (m,m) iff sum e_s = 2m+2.
# Hodge classes = invariants of Hg(prod_s E_{F_s} x E_i) = U(1)_{Q(i)} x prod_K U(1)_K x prod_kappa SL_2
# (Imai / Murty, cited in Gordon's survey Sec. 3).  Model input: the isogeny/CM type of each E_{F_s}.
#
# INDEPENDENT CHECK (no elliptic curves, no Shioda-Katsura): for Delsarte blocks
#   'F' = u^4 + v^4 (E_F = E_i),  'J' = u^4 + u v^3 (E_F = E_omega, j = 0),
# compute the Jacobian ring with the diagonal symmetry group G (multiplicity one is checked), the
# Galois action on characters, and count characters all of whose Galois conjugates are of type (m,m).
#
# PRODUCED CLASSES: inside the model, the span of products of pairings (divisor classes on products of two
# elliptic factors: weight-0 pairs for the tori, symplectic pairings for SL_2) -- numerical rank via numpy.
import itertools, json, math, sys
from fractions import Fraction
import numpy as np
import sympy

# ------------------------------------------------------------------ model count
def catalan(n):
    return math.comb(2 * n, n) // (n + 1)

def model_count(types, return_sectors=False):
    """types[s] = ('i',) for E_s ~ E_i, ('cm', K) for CM by K != Q(i) (class label K),
    ('gen', label) for non-CM isogeny class 'label'."""
    m1 = len(types); m = m1 - 1
    total = 0; hmm = 0; sectors = {}
    for c in itertools.product(range(4), repeat=m1):
        if sum(c) % 4: continue
        zeros = c.count(0); ones = c.count(1); threes = c.count(3)
        S2 = [s for s in range(m1) if c[s] == 2]
        # (m,m)-part dimension (for the h^{m,m} cross-check)
        cnt_mm = 0
        for signs in itertools.product((1, -1), repeat=len(S2)):
            # sigma=+1 <-> e_s = 0 (delta=-2), sigma=-1 <-> e_s = 4 (delta=+2); (m,m): sum delta = 0
            delta = ones - threes + sum(-2 * sg for sg in signs)
            if delta == 0: cnt_mm += 1
        hmm += 3 ** zeros * 2 ** (ones + threes) * cnt_mm
        # Hodge classes
        gen = {}; cms = []
        for s in S2:
            t = types[s]
            if t[0] == 'gen': gen[t[1]] = gen.get(t[1], 0) + 1
            else: cms.append(s)
        fg = 1
        for k in gen.values():
            fg *= catalan(k // 2) if k % 2 == 0 else 0
        if fg == 0: continue
        valid = 0
        for signs in itertools.product((1, -1), repeat=len(cms)):
            sig = dict(zip(cms, signs))
            # U(1)_i weight: -(ones - threes)/2 + sum_{s in S2, E_s ~ E_i} sigma_s ; must vanish
            wi = Fraction(-(ones - threes), 2) + sum(sig[s] for s in cms if types[s][0] == 'i')
            if wi != 0: continue
            ok = True
            Ks = set(types[s][1] for s in cms if types[s][0] == 'cm')
            for K in Ks:
                if sum(sig[s] for s in cms if types[s][0] == 'cm' and types[s][1] == K) != 0:
                    ok = False; break
            if ok: valid += 1
        d = 3 ** zeros * 2 ** (ones + threes) * fg * valid
        if d:
            sectors[c] = d; total += d
    res = dict(hodge=1 + total, prim=total, hmm_prim=hmm)
    if return_sectors: res['sectors'] = sectors
    return res

# ------------------------------------------------------------------ produced classes (model, explicit vectors)
def produced_rank(types):
    """Rank of the span of products of pairings inside each sector, summed (+1 for h^m).
    Basis of V_c: for each factor a weight label; build all 'perfect pairings' compatible with groups."""
    m1 = len(types); total = 0
    for c in itertools.product(range(4), repeat=m1):
        if sum(c) % 4: continue
        zeros = c.count(0); ones = c.count(1); threes = c.count(3)
        S2 = [s for s in range(m1) if c[s] == 2]
        mult = 3 ** zeros * 2 ** (ones + threes)  # Tate and P-factor multiplicities (each basis vector of these
        # factors is a weight vector of fixed U(1)_i weight; pairing structure below is per weight pattern)
        # U(1)_i-charged 'slots': ones (+1 each), threes (-1 each), L_c (weight -(ones-threes)/2 - ... see model):
        # we realise L_c as a single slot of U(1)_i weight wL chosen so that the pure Q(i) part balances:
        # total U(1)_i weight of a basis vector = (#1 - #3)(+1/2 each, via delta) ... implemented as in model_count
        gen = {}; cms = []
        for s in S2:
            t = types[s]
            if t[0] == 'gen': gen.setdefault(t[1], []).append(s)
            else: cms.append(s)
        # SL_2 part: span of products of symplectic pairings in (C^2)^{(x)k}: numerical rank
        fg = 1
        for lab, ss in gen.items():
            k = len(ss)
            if k % 2: fg = 0; break
            vecs = []
            for match in perfect_matchings(list(range(k))):
                v = np.zeros(2 ** k)
                for idx in itertools.product((0, 1), repeat=k):
                    val = 1.0
                    for (a, b) in match:
                        # omega(e_x, e_y) = det: (0,1)->1, (1,0)->-1
                        if idx[a] == idx[b]: val = 0.0; break
                        val *= 1.0 if (idx[a], idx[b]) == (0, 1) else -1.0
                    if val:
                        v[int(''.join(map(str, idx)), 2)] = val
                vecs.append(v)
            fg *= np.linalg.matrix_rank(np.array(vecs)) if vecs else 1
        if fg == 0: continue
        # torus part: weight-zero monomials are products of +/- pairs (always matchable); count them explicitly
        # as vectors (standard basis) -> rank = number of balanced sign patterns
        cnt = 0
        for signs in itertools.product((1, -1), repeat=len(cms)):
            sig = dict(zip(cms, signs))
            wi2 = -(ones - threes) + 2 * sum(sig[s] for s in cms if types[s][0] == 'i')
            if wi2 != 0: continue
            Ks = set(types[s][1] for s in cms if types[s][0] == 'cm')
            if all(sum(sig[s] for s in cms if types[s][0] == 'cm' and types[s][1] == K) == 0 for K in Ks):
                # explicit matching exists: list the charged slots and pair + with -
                cnt += 1
        total += mult * fg * cnt
    return 1 + total

def perfect_matchings(lst):
    if not lst:
        yield []
        return
    a = lst[0]
    for i in range(1, len(lst)):
        b = lst[i]
        rest = lst[1:i] + lst[i + 1:]
        for mm in perfect_matchings(rest):
            yield [(a, b)] + mm

# ------------------------------------------------------------------ Delsarte character count
BLOCKS = {
    'F': [(4, 0), (0, 4)],        # u^4 + v^4      (E_F ~ E_i)
    'J': [(4, 0), (1, 3)],        # u^4 + u v^3    (E_F ~ E_omega)
}

def jacobian_basis(block):
    """Monomial basis (i,j) of C[u,v]/(dF/du, dF/dv), F = sum of the monomials in block (coefficients 1)."""
    u, v = sympy.symbols('u v')
    F = sum(u ** i * v ** j for (i, j) in block)
    Fu, Fv = sympy.diff(F, u), sympy.diff(F, v)
    basis = []
    for deg in range(0, 5):
        mons = [(i, deg - i) for i in range(deg, -1, -1)]
        # ideal part in degree deg: multiples of Fu, Fv by monomials of degree deg-3
        gens = []
        if deg >= 3:
            for a in range(deg - 2):
                mon = u ** a * v ** (deg - 3 - a)
                gens += [sympy.expand(mon * Fu), sympy.expand(mon * Fv)]
        def vec(expr):
            P = sympy.Poly(expr, u, v) if expr != 0 else None
            return [P.coeff_monomial(u ** i * v ** j) if P is not None else 0 for (i, j) in mons]
        Imat = sympy.Matrix([vec(g) for g in gens]) if gens else sympy.zeros(0, len(mons))
        rI = Imat.rank() if gens else 0
        cur = Imat
        chosen = []
        for mono in mons:
            row = sympy.Matrix([[1 if mm == mono else 0 for mm in mons]])
            test = row if cur.shape[0] == 0 else cur.col_join(row)
            if test.rank() > (cur.rank() if cur.shape[0] else 0):
                chosen.append(mono); cur = test
        basis += chosen
    return basis

def block_group(block, D):
    return [(a, b) for a in range(D) for b in range(D) if all((i * a + j * b) % D == 0 for (i, j) in block)]

def delsarte_count(blocks):
    D = 12 if 'J' in blocks else 4
    m1 = len(blocks); m = m1 - 1
    info = []
    for bname in blocks:
        blk = BLOCKS[bname]
        B = jacobian_basis(blk)
        assert len(B) == 9, (bname, B)
        G = block_group(blk, D)
        chars = {}
        for (i, j) in B:
            chi = tuple(((i + 1) * a + (j + 1) * b) % D for (a, b) in G)
            assert chi not in chars, ('multiplicity >1', bname, (i, j), chars[chi])
            chars[chi] = (i, j)
        info.append((B, G, chars))
    units = [t for t in range(1, D) if math.gcd(t, D) == 1]
    hodge = 0; hmm = 0; total = 0
    for e in itertools.product(*[inf[0] for inf in info]):
        total += 1
        deg = sum(i + j for (i, j) in e)
        if deg != 2 * m + 2: continue
        hmm += 1
        ok = True
        for t in units:
            dd = 0
            for s in range(m1):
                B, G, chars = info[s]
                (i, j) = e[s]
                chi = tuple((t * (((i + 1) * a + (j + 1) * b) % D)) % D for (a, b) in G)
                assert chi in chars, 'Galois conjugate character not realised'
                (i2, j2) = chars[chi]
                dd += i2 + j2
            if dd != 2 * m + 2:
                ok = False; break
        if ok: hodge += 1
    return dict(hodge=1 + hodge, prim=hodge, hmm_prim=hmm, dimJ=total, D=D)

def fermat_shioda(n_vars):
    """Shioda's criterion for the Fermat quartic in n_vars variables (dimension n_vars-2, even):
    alpha in {1,2,3}^N, sum = 0 mod 4; Hodge iff sum <t alpha_i/4> = N/2 for t = 1, 3."""
    N = n_vars; cnt = 0; hmm = 0
    for a in itertools.product((1, 2, 3), repeat=N):
        if sum(a) % 4: continue
        if sum(a) == 2 * N:  # |alpha| = N/2  <->  type (m,m)
            hmm += 1
            if sum((3 * x) % 4 for x in a) == 2 * N:
                cnt += 1
    return dict(hodge=1 + cnt, prim=cnt, hmm_prim=hmm)

def R_formula(d, m):
    return sum(math.factorial(m + 1) // (math.factorial(k) ** 2 * math.factorial(m + 1 - 2 * k)) *
               (d - 2) ** (2 * k) * (d - 1) ** (m + 1 - 2 * k) for k in range(0, (m + 1) // 2 + 1))

def main():
    out = {}; lines = []
    T = {'F': ('i',), 'J': ('cm', 'omega')}
    # (1) model vs Delsarte character count
    lines.append('(1) MODEL (elliptic-curve Hodge-group invariants) vs DELSARTE character count (independent)')
    cases = []
    for m in (1, 2, 3):
        for combo in itertools.combinations_with_replacement('FJ', m + 1):
            cases.append(''.join(combo))
    for name in cases:
        mod = model_count([T[b] for b in name])
        dels = delsarte_count(list(name))
        prod = produced_rank([T[b] for b in name])
        agree = (mod['hodge'] == dels['hodge'] == prod) and (mod['hmm_prim'] == dels['hmm_prim'])
        out['delsarte_' + name] = dict(model=mod, delsarte=dels, produced=prod, agree=agree)
        lines.append('   %-5s m=%d  model Hdg=%-5d delsarte Hdg=%-5d produced=%-5d  h^{m,m}_prim model/jac=%d/%d  dimJ(all degrees)=%d  %s'
                     % (name, len(name) - 1, mod['hodge'], dels['hodge'], prod, mod['hmm_prim'], dels['hmm_prim'],
                        dels['dimJ'], 'AGREE' if agree else 'DISAGREE'))
    # (2) Fermat via Shioda's criterion
    lines.append('(2) Fermat quartic X^{2m}_4 via Shioda criterion (all variables Fermat) vs model FF..F')
    for m in (1, 2, 3):
        fs = fermat_shioda(2 * m + 2)
        mod = model_count([('i',)] * (m + 1))
        out['fermat_m%d' % m] = dict(shioda=fs, model=mod)
        lines.append('   m=%d  Shioda Hdg=%d (h^{m,m}_prim=%d)   model Hdg=%d' % (m, fs['hodge'], fs['hmm_prim'], mod['hodge']))
    # (3) non-Delsarte special and generic cases (model + produced; literature controls)
    lines.append('(3) other configurations (model; produced-classes rank; controls)')
    g = lambda lab: ('gen', lab)
    cfg = [
        ('generic m=1', [g('a'), g('b')], 'rho=18 (Shioda 1981 (5.11), r=8)'),
        ('P=Q non-CM m=1', [g('a'), g('a')], 'rho=19 (Mizukami, via Shioda 1981 Rem 5.3)'),
        ('E_i + generic m=1', [('i',), g('a')], 'rho=18'),
        ('CM sqrt-2 twice m=1', [('cm', 's2'), ('cm', 's2')], 'rho=20 (singular K3)'),
        ('generic m=2', [g('a'), g('b'), g('c')], 'R(4,2)+1 = %d' % (R_formula(4, 2) + 1)),
        ('generic m=3', [g('a'), g('b'), g('c'), g('d')], 'R(4,3)+1 = %d' % (R_formula(4, 3) + 1)),
        ('generic m=4', [g(x) for x in 'abcde'], 'R(4,4)+1 = %d' % (R_formula(4, 4) + 1)),
        ('F0=F1 non-CM, F2 gen m=2', [g('a'), g('a'), g('b')], ''),
        ('F0=F1=F2 non-CM m=2', [g('a'), g('a'), g('a')], ''),
        ('E_i, gen, gen m=2', [('i',), g('a'), g('b')], 'extra (2,1,1),(2,3,3) non-join classes'),
        ('F0=..=F3 non-CM m=3', [g('a')] * 4, 'SL_2 invariants with Catalan 2'),
        ('E_i x4 (Fermat 6-fold) m=3', [('i',)] * 4, ''),
        ('sqrt-2 CM x3 m=2', [('cm', 's2')] * 3, ''),
    ]
    for name, types, note in cfg:
        mod = model_count(types, return_sectors=True)
        prod = produced_rank(types)
        out['cfg_' + name] = dict(model={k: v for k, v in mod.items() if k != 'sectors'}, produced=prod, note=note,
                                  nonjoin_sectors={str(k): v for k, v in mod['sectors'].items()
                                                   if 0 not in k and 2 in k})
        # classes in sectors with no 0 entry that admit no pairing into (1,3)/(3,1)/(2,2): not DF-V joins
        def matchable(k):
            return k.count(1) == k.count(3) and k.count(2) % 2 == 0
        unm = sum(v for k, v in mod['sectors'].items() if 0 not in k and not matchable(k))
        note = (note + '; ' if note else '') + 'unpaired-sector classes=%d' % unm
        lines.append('   %-28s Hdg=%-6d produced=%-6d h^{m,m}_prim=%-6d  %s' % (name, mod['hodge'], prod, mod['hmm_prim'], note))
    # (4) sanity: Hodge numbers h^{m,m}_prim of quartic 2m-folds
    lines.append('(4) h^{m,m}_prim from sectors vs known: quartic surface 19 (h^{1,1}=20), quartic 4-fold 141 (h^{2,2}=142)')
    for m in (1, 2, 3):
        mod = model_count([g('a')] * 0 + [g(str(s)) for s in range(m + 1)])
        lines.append('   m=%d  h^{m,m}_prim (sector count) = %d' % (m, mod['hmm_prim']))
    json.dump(out, open('hodge_count.json', 'w'), indent=1, default=str)
    open('hodge_count.out', 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))

if __name__ == '__main__':
    main()
