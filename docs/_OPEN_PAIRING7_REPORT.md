# Pairing existence at the threshold $k=7$, $g=3$

VERDICT: EXISTS

## Periodic equality constraints

Let a pairing be periodic under a full-rank lattice. If necessary, pass to
a sublattice whose quotient is injective on every set of diameter at most
6. Let the resulting quotient have $N$ cell orbits and $P$ pair orbits.
Discard pairs contained in no length-7 window. A remaining pair is aligned
with one axis and has axis-distance $\delta\in\{1,\ldots,6\}$. It is
contained in exactly $7-\delta\leq 6$ window orbits.

There are $3N$ window orbits. Counting incidences between pair orbits and
window orbits gives

\[
3N
\;\leq\; \sum_e (7-\delta_e)
\;\leq\; 6P
\;\leq\; 3N.
\]

The last inequality is $2P\leq N$, since the pairs form a matching. All
three inequalities are equalities. Therefore every cell is matched, every
pair is an adjacent axis pair, and every window contains exactly one pair.
Each axis has $N/6$ pair orbits.

On one axis line, let $x_t\in\{0,1\}$ indicate that the unit edge starting
at coordinate $t$ is a pair. Exact coverage says

\[
\sum_{i=0}^{5}x_{s+i}=1
\quad\hbox{for every }s.
\]

Subtracting the equations at $s$ and $s+1$ gives $x_s=x_{s+6}$.
Thus each line has one selected unit-edge phase modulo 6.
Obtaining exact coverage above uses periodicity; once exact coverage is
known, the recurrence is pointwise. The Folner-density argument alone does
not exclude zero-density defects.

## Construction

Use axial coordinates $(q,r)\in\mathbb Z^2$, with axes

\[
H=(1,0),\qquad V=(0,1),\qquad D=(1,-1).
\]

Let the period lattice be

\[
\Lambda=\langle(2,2),(0,6)\rangle
=\{(2m,2m+6n):m,n\in\mathbb Z\}.
\]

It has index 12. A fundamental domain is

\[
F=\{0,1\}\times\{0,1,2,3,4,5\}.
\]

Take every simultaneous \(\Lambda\)-translate of the following six pairs:

\[
\begin{array}{c|c}
\text{axis}&\text{pair representatives}\\ \hline
H&(0,0)-(1,0),\quad (0,1)-(1,1)\\
V&(0,3)-(0,4),\quad (1,3)-(1,4)\\
D&(1,2)-(2,1),\quad (1,5)-(2,4).
\end{array}
\]

## Matching proof

The homomorphism

\[
\phi(q,r)=(q\bmod 2,\ r-q\bmod 6)
\quad\text{from }\mathbb Z^2\text{ to }\mathbb Z_2\times\mathbb Z_6
\]

has kernel \(\Lambda\). The six pair representatives map to

\[
\begin{array}{c|c}
H&(0,0)-(1,5),\quad (0,1)-(1,0)\\
V&(0,3)-(0,4),\quad (1,2)-(1,3)\\
D&(1,1)-(0,5),\quad (1,4)-(0,2).
\end{array}
\]

Their twelve endpoints are the twelve distinct elements of
\(\mathbb Z_2\times\mathbb Z_6\). Hence every cell of the infinite lattice
is incident with exactly one lifted pair.

Equivalently, reducing endpoints into $F$ gives the following axis label
for each cell:

```text
        q=0 q=1
r=0      H   H
r=1      H   H
r=2      D   D
r=3      V   V
r=4      V   V
r=5      D   D
```

## Window-covering proof

The images under \(\phi\) of the three axis steps are

\[
(1,5),\qquad(0,1),\qquad(1,4).
\]

Each has order 6. Each axis therefore forms two six-cycles on the
12-element quotient, and the construction selects one edge on each of the
six cycles. More directly:

- horizontal lines have two \(\Lambda\)-orbits, represented by $r=0,1$,
  and their selected pairs repeat by $(6,0)$;
- vertical lines have two orbits, represented by $q=0,1$, and their
  selected pairs repeat by $(0,6)$;
- diagonal lines $q+r=\text{constant}$ have two orbits, represented by
  the odd-sum and even-sum diagonal pairs above, and their selected pairs
  repeat by $(6,-6)$.

Thus the pair starts on every axis line are one residue class modulo 6. A
length-7 window has six consecutive internal unit edges, so it contains
exactly one selected pair.

## Exhaustive search and verification

The offline script is `scripts/_pairing7_search.py`; it uses Python 3 and
no external packages.

On \(\mathbb Z^2/\Lambda\), each axis has two six-cycles. The search has six
line-cycle variables, each with six possible phases. It encodes a phase as
an exact-cover row containing one line-cycle constraint and the two
endpoint constraints. The instance has 36 rows and 18 columns: 6 line-cycle
columns and 12 cell columns.

The full raw space has

\[
6^6=46{,}656
\]

phase assignments. An MRV Algorithm X enumeration exhausted the instance
in 419 recursive states and found 120 labelled solutions. A separate direct
iteration over all 46,656 phase tuples also found 120. The construction
above was present in both enumerations.

The verification results were:

```text
pair orbits                                      6
quotient cell orbits checked                    12
cell-incidence histogram                   {1: 12}
quotient start/axis window orbits checked       36
contained-pair histogram                   {1: 36}
cells checked in [-30,30]^2                   3721
finite-patch cell-incidence histogram       {1: 3721}
finite-patch windows checked                  11163
finite-patch contained-pair histogram       {1: 11163}
```

The quotient checks are exhaustive, not samples. Translation by
\(\Lambda\) preserves both the matching and every axis direction, so every
cell is equivalent to one of the 12 checked cell orbits and every window is
equivalent to one of the $12\times3=36$ checked window orbits.

## Period-index minimality

The rigidity conclusions obtained after passing to a locally injective
sublattice are properties of the pairing itself. For any original period
lattice with quotient size $N$, let $o$ be the order of an axis step in the
quotient. Translation by $o$ steps preserves that line's edge indicator.
The indicator has least period 6, so $6\mid o$ for all three axes and hence
$6\mid N$. If $N<12$, then $N=6$. The quotient is abelian, so a quotient of
order 6 is cyclic, and its only elements of order 6 are $g$ and $-g$. The
difference of two such elements has order at most 3, but the third hex-grid
axis step is the difference of the first two. This is impossible. The
index-12 construction is therefore minimal among periodic constructions.

This threshold problem is of independent combinatorial interest. It does
not affect Hexo, which has $k=6$.
