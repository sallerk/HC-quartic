# s3_ctrl_explain.py -- explain every control coincidence in s3_jac_check: a control curve E matches a twist
# of E_i^2 (or E_w) at p exactly when Frob_p(E) lies in Q(i) (resp. Q(w)) up to roots of unity, i.e.
# 4p - a_p^2 = k^2 (resp. 3k^2), or a_p = 0 with the reference curve also supersingular.  If every coincidence
# is of this explained type the control is consistent; an unexplained match would mean the test is not
# discriminating.
# Run s3_jac_check.py first, in the same working directory (this script reads s3_jac_check.json from there).
# Writes s3_ctrl_explain.out (= what is printed).
import json, math, os, sys

class _Tee:
    def __init__(self, path):
        self.f = open(path, 'w'); self.s = sys.stdout
    def write(self, x):
        self.f.write(x); self.s.write(x)
    def flush(self):
        self.f.flush(); self.s.flush()
sys.stdout = _Tee('s3_ctrl_explain.out')

_here = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(_here, 's3_jac_check.py')).read().split("rnd = random.Random")[0])   # reuse helper functions only
data = json.load(open('s3_jac_check.json'))
def frob_field(p, a):
    D = 4 * p - a * a
    if a == 0: return 'supersingular'
    r = math.isqrt(D)
    if r * r == D: return 'Q(i)'
    if D % 3 == 0 and math.isqrt(D // 3) ** 2 == D // 3: return 'Q(w)'
    return 'other'
rows = []
for r in data['results']['d4']:
    p = r['p']
    if not r['divides']: continue
    Q = r['Q']
    Pw = ell_charpoly(p, 0, 0, 1); Pn = ell_charpoly(p, 0, 1, 3)
    mw = p != 3 and min_twist_N(Q, polymul(Pw, Pw)) is not None
    mn = p not in (13, 19) and min_twist_N(Q, polymul(Pn, Pn)) is not None
    if mw or mn:
        rows.append((p, r['F'], 'Ew2' if mw else '', 'nonCM2' if mn else '', 'p%4=' + str(p % 4),
                     'Ew:' + frob_field(p, -Pw[1]), 'ctrl:' + frob_field(p, -Pn[1])))
unexpl = []
for row in rows:
    p = row[0]
    ok = True
    if row[2]:   # E_w^2 match: need E_w Frobenius ~ E_i Frobenius: both supersingular (p%4==3 and p%3==2)
        ok &= (p % 4 == 3 and p % 3 == 2)
    if row[3]:
        ff = row[6].split(':')[1]
        ok &= (ff == 'Q(i)') or (ff == 'supersingular' and p % 4 == 3)
    if not ok: unexpl.append(row)
print('control matches (p, F, which, p mod 4, E_w frob field, nonCM-ctrl frob field):')
for row in rows: print('  ', row)
print('unexplained control matches:', unexpl)
for r in data['results']['d3']:
    p = r['p']; Pi = ell_charpoly(p, 0, -1, 0)
    if min_twist_N(r['PC'], Pi) is not None and not (p % 4 == 3 and p % 3 == 2):
        print('UNEXPLAINED d3 control match', r)
print('d3 control matches all at p = 11 mod 12 (both E_i, E_w supersingular):',
      all((r['p'] % 12 == 11) for r in data['results']['d3'] if min_twist_N(r['PC'], ell_charpoly(r['p'], 0, -1, 0)) is not None))
sys.stdout.flush()
