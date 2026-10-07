Supporting code and data for the note
"The Hodge conjecture for sums of binary quartics"
==================================================

Status: draft accompanying a draft note; not peer reviewed.

What the note proves (Theorem A / Corollary B there): the Hodge conjecture holds for every smooth
hypersurface X = {F_0(u_0,v_0) + ... + F_m(u_m,v_m) = 0} in P^(2m+1), F_s binary quartic forms with
distinct roots, and more generally for products of such hypersurfaces (also with extra terms z_j^4)
and curves whose Jacobians are isogenous to products of elliptic curves. It also gives a formula
for dim Hdg^m(X) (Proposition 6.1 there).

IMPORTANT: the proofs of Theorem A and Proposition 6.1 in the note do not use any of these
computations. The scripts below are consistency checks of the ingredients (at finitely many primes /
in finitely many cases) and illustrations in low dimension, where the Hodge conjecture was already
known. Statements of the note that do rest on computations are in Section 7: the values of
dim Hdg^m(X) listed there (hodge_count.py, hodge_formula.py; also quoted in the introduction), and,
under "Explicit cycles for m <= 2", that for F = u^4 + (7/3)u^2v^2 + v^4 the 32 lines on
{F(u,v) = F(z,w)} together with h, and the 256 planes on X(F,F,F) together with h^2, span the
Hodge classes. The latter rests on exact_incidence.py (exact intersection pattern, exact rank over
Q) and Proposition 6.1.

Licence: the code in code/ is under the MIT licence; the data in outputs/ and this text are under
CC BY 4.0 (see the LICENSE files of the repository). No third-party files are included.


1. Requirements
---------------
Python 3 with numpy, sympy, mpmath and python-flint (python-flint only for lines_check.py,
planes_check.py and exact_incidence.py). Tested with Python 3.12.9, numpy 2.1.3, sympy 1.13.1,
mpmath 1.3.0, python-flint 0.9.0. No Sage, no docker, no GPU. The scripts do not parallelize
themselves, but numpy's linear algebra may start several threads; planes_check.py sets the
BLAS/OpenMP thread variables to 1 itself, and for the other scripts one can set OMP_NUM_THREADS=1,
OPENBLAS_NUM_THREADS=1 and MKL_NUM_THREADS=1 in the environment (this was done for the runtimes
below). Runtimes measured on a desktop PC, one script at a time:
jac_control.py 91 s, jac_check.py 35 s, s1_map_check.py 9 s, planes_check.py 6 s,
s3_jac_check.py 4 s, the others about 1 s each.


2. How to run
-------------
Each script writes its result files (.out, and for most also .json) into the CURRENT working
directory. The reference results are in outputs/. To reproduce them, from the top folder
of this repository:

    mkdir rerun
    cd rerun
    python ../code/jac_check.py
    python ../code/jac_control.py
    python ../code/hodge_count.py
    python ../code/lines_check.py
    python ../code/planes_check.py
    python ../code/s1_map_check.py
    python ../code/s3_jac_check.py
    python ../code/s3_ctrl_explain.py     (needs s3_jac_check.json from the previous line,
                                           so run it in the same folder, after s3_jac_check.py)
    python ../code/hodge_formula.py
    python ../code/exact_incidence.py

and compare each produced file with the file of the same name in ../outputs/ (for example with
"diff" or "fc"). jac_control.py imports jac_check.py and planes_check.py imports lines_check.py
from code/; s3_ctrl_explain.py reuses the helper functions of s3_jac_check.py; hodge_formula.py and
exact_incidence.py import no other script.

Reproducibility: in a fresh folder, all 18 files produced by the commands above were
byte-for-byte identical to the files in outputs/ (rerun on 6 October 2026; Windows, Python 3.12.9,
versions as above). On another platform the files may differ in line endings (the reference files
have CRLF line endings); the floating-point singular values printed by lines_check.py,
planes_check.py ("sv gap ...") and exact_incidence.py may differ in the last digits with other
numpy/BLAS versions. All other content is exact integer, rational or modular arithmetic, or uses
fixed random seeds.


3. Files
--------
Names: E_F is the genus-one curve y^2 = -F(u,v), E_i the elliptic curve y^2 = x^3 - x
(j = 1728, CM by Z[i]), E_w the curve y^2 = x^3 + 1 (j = 0), C_F the plane quartic
w^4 + F(u,v) = 0 (genus 3). Binary forms are encoded as coefficient lists [c_0, ..., c_d] with
F(u,v) = sum_i c_i u^i v^(d-i).

code/s1_map_check.py  ->  outputs/s1_map_check.json, outputs/s1_map_check.out
  Supports: Section 3 of the note (Proposition 3.1: the Shioda-Katsura map).
  Checks: (1) the map ((x:y),(x':y')) -> (y'x : eps*y*x'), eps^d = -1, sends
  {f + y^d = 0} x {g + y'^d = 0} into {f + g = 0}, symbolically for generic f (3 variables) and
  g (2 variables) of degree d = 3, 4, 5; (2)-(3) the base locus and the two blow-up charts
  (recorded as text; the argument is in the note); (4) Fulton's degree formula for a rational
  map defined by a linear system (Intersection Theory, Prop. 4.4) gives degree d for the
  resolved map for all dimensions 1 <= dim X_1, dim X_2 <= 8; (5) over F_p, for 40 random points
  of {f + g = 0} for each (p, d) in {(17,4), (41,4), (13,3), (19,3), (31,3)}, brute-force count
  of preimages = number of y in F_p with y^d = -f(x) (0 or d). (p = 13, d = 4 is skipped: no
  eps in F_13 with eps^4 = -1.)
  Expected: "lands_in_W_h_d3/d4/d5": true; "fulton_all_equal_1": true; for every tested (p,d):
  "all_cnt_eq_nroots": true with "cnt_values" [0, d].

code/jac_check.py  ->  outputs/jac_check.out, outputs/jac_check.json
  Supports: Section 4 (Proposition 4.1: Jac(C_F) ~ E_F x E_i^2), as a numerical consistency check.
  Checks: (A) for 11 binary quartics F (u^4-v^4, u^4+v^4, u^3v-uv^3, u^4+uv^3, u^4+u^2v^2+v^4,
  u^4-3u^2v^2+v^4, v(u^3+4u^2v+2uv^2), and 4 random forms with seed 20261005) and the primes
  p in {3,5,7,11,13,17,19,23,29,37} at which F keeps 4 distinct roots (102 pairs): the
  L-polynomial of C_F over F_p (from point counts over F_p, F_p^2, F_p^3) is divisible by that
  of E_F, and the quotient L_P satisfies L_P^(N) = (L_{E_i}^(N))^2 over F_(p^N) for N = 24
  (N = list of all N in {1,2,3,4,6,8,12,24} that work). By Tate's isogeny theorem this says that
  the reductions mod p of the complementary abelian surface P and of E_i^2 are isogenous over
  F_(p^24). (B) p-adic valuations of det(Frobenius) on the eigenspaces of w -> zeta*w in H^1,
  compared with the Hodge numbers (d = 4: eigenspace dimensions (2,1,0) of H^{1,0}; the d = 5, 6
  lines are not used in the note). (C) the dimensions of the eigenspaces of w -> zeta*w on the
  holomorphic differentials x^a w^b dx/w^(d-1), a + b <= d-3, for d = 3,...,6.
  Expected: 102 lines "divisible=True", each with 24 in its list of N; (C) d=4: {1: 2, 2: 1, 3: 0}.
  Caveat: at p = 3 mod 4 both P and E_i^2 are supersingular and the comparison is weak; the
  informative primes are p = 1 mod 4.

code/jac_control.py  ->  outputs/jac_control.out
  Negative control for jac_check.py, for the random quartic F = [3,-2,5,1,7] at
  p in {17,29,37,41}: (i) the same test with E_i replaced by y^2 = x^3 + x + 3 (j = 6912/247,
  not an integer, so no CM) must fail; (ii) divisibility of L_{C_F} by L_{E_G} for the unrelated
  form G = [1,4,-1,0,2] should fail. The script also lists p = 13 and skips it: y^2 = x^3 + x + 3
  has discriminant -16*247 = -16*13*19, so it has bad reduction there. (An earlier version
  printed a meaningless row for p = 13; the note uses only 17, 29, 37 and 41.)
  Expected: a "skip p=13" line, then "E_i^2 match" non-empty and "E'^2 match N=[]" at the four
  primes 17, 29, 37, 41; divisibility by the wrong factor is False except at p = 41, where
  a_p(E_F) = a_p(E_G) = -10, so the two L-factors coincide and the divisibility is automatic (the
  script prints this explanation).

code/s3_jac_check.py  ->  outputs/s3_jac_check.json, outputs/s3_jac_check.out
  Supports: Section 4, as a second, independently written check (own finite-field arithmetic,
  own L-polynomial code). 12 binary quartics (6 named, 6 random with seed 20261005), primes 3 to
  43 with good reduction: 144 pairs; and 6 binary cubics (d = 3: C_G is a plane cubic with an
  automorphism of order 3, expected to be E_w up to twist), 68 pairs.
  Expected: "d4_pairs": 144, "d4_EF_divides": 144, "d4_Q_twist_of_Ei2": 144, "d3_pairs": 68,
  "d3_twist_of_Ew": 68, "d4 failures: []", "d3 failures: []". The control counts
  ("d4_ctrl_Ew2_match": 22, "d4_ctrl_nonCM2_match": 18, "d3_ctrl_Ei_match": 11) are explained by
  s3_ctrl_explain.py.

code/s3_ctrl_explain.py  ->  outputs/s3_ctrl_explain.out
  Explains each coincidence of the controls in s3_jac_check.py: a control curve can match only at
  primes where it is supersingular together with the reference curve, or where its Frobenius lies
  in Q(i).
  Expected: "unexplained control matches: []" and "d3 control matches all at p = 11 mod 12 (both
  E_i, E_w supersingular): True".

code/hodge_count.py  ->  outputs/hodge_count.out, outputs/hodge_count.json
  Supports: Section 7 of the note ("Hodge classes on some Delsarte members"), a check of
  Proposition 6.1. Not used in any proof.
  Part (1): for the 12 block patterns built from F = u^4 + v^4 (E_F = E_i) and J = u^4 + uv^3
  (E_J = E_w) with m = 1, 2, 3, the hypersurface is of Delsarte type and dim Hdg^m(X) (including
  the power of the hyperplane class) is computed by the character method (Jacobian ring, diagonal
  group, Galois action; the script asserts that every character occurs with multiplicity one).
  This is compared with a "model" count: an earlier implementation of the count that is now
  Proposition 6.1 of the note (iterating the Shioda-Katsura construction together with
  Jac(C_F) ~ E_F x E_i^2 and the Hodge group of a product of elliptic curves, after Imai; see
  Gordon's survey). The derivation is only sketched in the code comments; the proof is the one in
  the note, and hodge_formula.py evaluates the formula as stated there.
  Expected: AGREE in all 12 lines, with dim Hdg^m = FF 20, FJ 18, JJ 20; FFF 142, FFJ 122,
  FJJ 114, JJJ 118; FFFF 1108, FFFJ 928, FFJJ 826, FJJJ 784, JJJJ 820.
  Part (2): Fermat quartic in dimension 2, 4, 6 by Shioda's criterion: 20, 142, 1108.
  Parts (3)-(4): model counts for further configurations (not used in the note), compared where
  possible with the Picard numbers of the surfaces P(x,y) + Q(z,w) = 0 from Shioda,
  Ann. Sci. ENS 14 (1981), Section 5, and with the closed formula 1 + R(4,m) for distinct generic
  blocks; and h^{m,m}_prim = 19, 141, 1107 for quartics of dimension 2, 4, 6.

code/lines_check.py  ->  outputs/lines_check.out, outputs/lines_check.json
  Supports: Section 7 (explicit cycles for m = 1, where the Hodge conjecture is the Lefschetz
  (1,1) theorem). Rank of the Gram matrix of the hyperplane class and explicit lines (join lines
  {u = a v, z = b w} and graph lines {(u,v) = A(z,w)}) on quartic surfaces F_0(u,v) = F_1(z,w);
  the lines are computed numerically (mpmath, 40 digits) and checked to lie on the surface. Which
  lines meet is decided in floating point; the rank of the resulting integer Gram matrix is
  computed numerically and exactly over Q (python-flint).
  Expected (both ranks): F_0 = F_1 = u^4 + (7/3)u^2v^2 + v^4 (no CM: j(E_F) = 61918288/1521 is not
  an integer): 16 + 16 lines, rank 19; F_0 = u^4 + (7/3)u^2v^2 + v^4, F_1 = [2,-1,3,1,5]: 16 lines,
  rank 10; Fermat quartic surface: 48 lines, rank 20; F_0 = F_1 = u^4 + uv^3: 64 lines, rank 20.

code/planes_check.py  ->  outputs/planes_check.out, outputs/planes_check.json
  Supports: Section 7 (explicit cycles for m = 2, where the Hodge conjecture was already known).
  Planes on X = {F_0 + F_1 + F_2 = 0} in P^5 that are joins of three points or of a graph line
  with a point; Gram matrix of h^2 and the planes (P.P = 7, -2 for planes meeting in a line, 1 for
  planes meeting in a point, 0 for disjoint planes, h^2.P = 1, h^2.h^2 = 4). Which planes meet,
  and in what dimension, is decided from floating-point ranks of the spans; the rank of the Gram
  matrix is computed numerically, exactly modulo 1000003 and 998244353, and exactly over Q
  (python-flint).
  Expected: F_0 = F_1 = F_2 = u^4 + (7/3)u^2v^2 + v^4: 256 planes, rank 109 (all four ways);
  F_0 = F_1 = u^4 + (7/3)u^2v^2 + v^4, F_2 = [2,-1,3,1,5]: 128 planes, rank 55; Fermat quartic
  fourfold: 448 planes, rank 118 (the Fermat fourfold has dim Hdg^2 = 142; the planes used here
  are only of the two join types above).

code/hodge_formula.py  ->  outputs/hodge_formula.out, outputs/hodge_formula.json
  Supports: Section 7 ("Hodge classes on some Delsarte members"). Evaluates the closed formula of
  Proposition 6.1 for dim Hdg^m(X); written from the statement of the proposition, independently of
  hodge_count.py. Compares it with the Delsarte counts of hodge_count.py part (1) (copied into the
  script) for the 12 block patterns there, and prints the values for blocks without CM
  (G = u^4 + (7/3)u^2v^2 + v^4, H a second curve without CM, not isogenous to E_G).
  Expected: "agree" in all 12 lines, with the values listed under hodge_count.py; GG 19, GH 18,
  GGG 109, GGH 103, GGGG 714; last line "ALL TWELVE AGREE".

code/exact_incidence.py  ->  outputs/exact_incidence.out, outputs/exact_incidence.json
  Supports: Section 7 ("Explicit cycles for m <= 2"), for F = u^4 + (7/3)u^2v^2 + v^4: the 32 lines
  on S = {F(u,v) = F(z,w)} and the 256 planes on X(F,F,F) of lines_check.py / planes_check.py.
  The linear spaces are sums of lines in the coordinate blocks C^2 and graphs of matrices A with
  F o A = +-F, so the dimension of the intersection of two of them is determined by the roots of F
  they contain, the permutations of the roots induced by the matrices, and dim ker(A - A'). These
  data are computed exactly with SymPy (a quantity is declared zero only if its minimal polynomial
  over Q is x); the resulting intersection pattern is compared, for all pairs, with floating-point
  ranks of the spans; the Gram matrices are formed from the exact data and their rank is computed
  exactly over Q (python-flint).
  Expected: lines: pairs by dimension of the intersection {0: 336, 1: 160}, 0 disagreements,
  rank 19; planes: {0: 19584, 1: 11136, 2: 1920}, 0 disagreements, rank 109.
  With Proposition 6.1 (Picard number 19, dim Hdg^2(X(F,F,F)) = 109) this shows that the lines
  and h span NS(S) (x) Q and that the planes and h^2 span Hdg^2(X(F,F,F)).


4. What the checks do not show
------------------------------
- Point counts at finitely many primes cannot prove an isogeny over C; the proof of
  Jac(C_F) ~ E_F x E_i^2 is the argument in Section 4 of the note.
- The "model" counts in hodge_count.py and the values of hodge_formula.py are evaluations of the
  formula of Proposition 6.1, whose proof is in the note. They are compared with an independent
  method (the Delsarte character counts of hodge_count.py, parts (1)-(2)) only for blocks
  u^4 + v^4 and u^4 + uv^3, all with complex multiplication, and for m = 1 with the Picard numbers
  from Shioda (1981) quoted in part (3).
- Explicit cycles are computed in dimensions 2 and 4 only. For F = u^4 + (7/3)u^2v^2 + v^4 they
  span the Hodge classes (exact_incidence.py with Proposition 6.1). In the other cases of
  lines_check.py / planes_check.py they give lower bounds only, and which cycles meet is decided
  there in floating point.
