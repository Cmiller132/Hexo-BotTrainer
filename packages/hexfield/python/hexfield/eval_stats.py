"""hexfield eval — the statistics layer (corrected, adversarially reviewed).

This module is the load-bearing inference core for the multi-stage hexfield
evaluation (docs/specs/hexfield_v2_synthesis.md eval track). It is *purely*
arithmetic: it turns game tallies into win-rate CIs, Elos, a converged
Bradley-Terry rating pool, and SPRT triage verdicts. It knows nothing about
games, GPUs, or the run; callers feed it counts and it returns labels and
intervals. It NEVER gates, promotes, or halts anything — the verdict is a
reported string.

Pure ``numpy`` (+ optional ``scipy`` for the BT optimiser, with a self-
contained Newton fallback). No engine, no torch, no CUDA. Safe to unit-test on
a CPU-only interpreter.

THE FIVE CORRECTED-DESIGN FIXES (these are the whole point; the naive versions
are wrong and were rejected in review):

1. BT MUST CONVERGE. The legacy ``scripts/_wf_r4_bt_rating.py`` does 500 steps
   of fixed-step gradient descent (step 0.002) and stops with ``max|grad|~0.30``
   — nowhere near a stationary point, so its Fisher-from-the-final-iterate
   covariance is meaningless. Here we run Newton (or ``scipy.optimize.minimize``)
   to a gradient tolerance and ``assert max|grad| < grad_tol`` BEFORE inverting
   the Hessian. See :func:`bradley_terry`.

2. PAIRING IS REAL CORRELATION. Paired games (same opening, swapped seats) are
   NOT independent Bernoulli trials. The unit of replication is the PAIR. Win
   rates get pair-level SEs (``N_pairs`` units, :func:`paired_winrate`), and the
   BT likelihood is fed EFFECTIVE counts (pairing deflates the effective sample
   size relative to ``2*N_pairs`` independent games), so the resulting CIs are
   not anti-conservative. See :func:`pentanomial_summary` and
   :func:`effective_counts`.

3. MULTIPLE COMPARISONS. There is exactly ONE pre-registered primary hypothesis
   per verdict: candidate ``L`` vs prior champion ``B``, tested via the
   BT difference CI ``r_L - r_B`` using the full covariance (the ``-2*Cov_LB``
   term — :func:`var_diff`). Every other opponent edge is DESCRIPTIVE: report
   its Wilson/Elo CI, attach NO significance verdict. If several edges must gate
   (they do not, by default), Bonferroni each at ``alpha/k``
   (:func:`bonferroni_alpha`).

4. HONEST SIGNIFICANCE. 128 games/epoch resolve ~100-120 Elo
   (single-epoch ``SE(r_L - r_B) ~= 40-55 Elo``). The ~15-20 Elo resolution is
   the MULTI-EPOCH ROLLING ASYMPTOTE of the persisted pool, not a per-epoch
   property. :func:`expected_se_elo` and :func:`games_for_se` make this explicit;
   do not overclaim a single epoch.

5. THE NOISY SEALBOT EDGE. SealBot's search depth varies under GPU load, so its
   edge is over-dispersed and must NOT enter difference inference at full weight.
   SealBot is used ONLY as the pinned zero-point of the rating scale; its edges
   are down-weighted by an over-dispersion factor (:func:`overdispersion_weight`,
   wired through ``bradley_terry(weights=...)``). Hexo has NO draws (binomial
   base); the pentanomial "trinomial-per-game" device is only a conservative
   bookkeeping for split pairs, never an invented draw outcome.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# logit (natural-log strength) -> Elo. Elo is defined so that a 10x odds ratio
# (one e-fold in log-strength is *not* 400 Elo; 400 Elo is one base-10 fold).
# r is in natural log units, so Elo = r * 400 / ln(10).
ELO_PER_LOGIT = 400.0 / math.log(10.0)
LOGIT_PER_ELO = math.log(10.0) / 400.0

# Default two-sided 95% normal quantile (z_{0.975}); matches the arena script.
Z95 = 1.959963984540054


# --------------------------------------------------------------------------- #
# 1. Win-rate confidence intervals.
# --------------------------------------------------------------------------- #
def wilson_ci(wins: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (no continuity corr.).

    The Wilson interval is the inversion of the score test and stays inside
    ``[0, 1]`` with good small-sample coverage, unlike the Wald interval. On an
    empty sample it returns the trivial ``(0, 1)``.

    NOTE on pairing: this treats the ``n`` games as independent Bernoulli
    trials. For PAIRED games (shared openings, swapped seats) the independent
    ``n`` is anti-conservative — use :func:`paired_winrate` /
    :func:`pentanomial_summary`, whose SE is computed on ``N_pairs`` units.
    """

    if n <= 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def los(wins: int, losses: int) -> float:
    """Likelihood-of-superiority: ``Phi((w - l) / sqrt(w + l))``.

    The standard chess-engine LOS — the probability that the candidate is truly
    stronger than the opponent given ``w`` wins and ``l`` losses, under the
    normal approximation to the binomial (Hexo has no draws, so ``w + l`` is the
    decided count). Returns 0.5 when no games are decided.
    """

    decided = wins + losses
    if decided <= 0:
        return 0.5
    return _phi((wins - losses) / math.sqrt(decided))


# --------------------------------------------------------------------------- #
# 2. Paired / pentanomial scoring (fix #2: pairing is correlation).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PairedResult:
    """Pair-level summary of a set of paired games (shared opening, swapped seats).

    A "pair" is two games on one opening with the candidate playing each seat
    once. Each pair scores in {0, 0.5, 1, 1.5, 2} candidate-points out of 2; the
    five-bucket count vector is the *pentanomial*. The mean per-pair score / 2 is
    the candidate win rate, but its SE is taken over PAIRS (the true unit of
    replication), which captures within-pair correlation that a per-game
    binomial SE ignores.

    n_pairs:    number of complete pairs.
    penta:      length-5 counts [LL, LD, DD(=split), DW, WW] in candidate terms,
                i.e. points {0, 0.5, 1, 1.5, 2}. (Hexo has no draws, so a single
                game is never 0.5; the 0.5/1.5 buckets only arise from a pair
                whose two games split, and the middle 1.0 bucket is a 1-1 split.
                We still carry the full five-bucket vector so the variance is
                computed from the realised point distribution, not assumed.)
    win_rate:   candidate win rate = mean pair score / 2, in [0, 1].
    se:         standard error of ``win_rate`` on N_pairs units.
    var_drift:  per-pair variance of the (points/2) score (the over-dispersion
                signal: 0 => every pair identical, 0.25 => max Bernoulli-like).
    """

    n_pairs: int
    penta: tuple[int, int, int, int, int]
    win_rate: float
    se: float
    var_drift: float

    def ci(self, z: float = Z95) -> tuple[float, float]:
        """Normal pair-level CI for the win rate, clipped to ``[0, 1]``."""

        if self.n_pairs <= 0:
            return (0.0, 1.0)
        lo = self.win_rate - z * self.se
        hi = self.win_rate + z * self.se
        return (max(0.0, lo), min(1.0, hi))


def pentanomial_summary(penta: "tuple[int, ...] | np.ndarray") -> PairedResult:
    """Summarise a pentanomial count vector into a pair-level win rate + SE.

    ``penta`` is the length-5 vector of pair counts for candidate points
    ``{0, 0.5, 1, 1.5, 2}`` (i.e. ``[LL, split-low, even, split-high, WW]``).
    The per-pair score (in [0, 1]) takes values ``{0, .25, .5, .75, 1}``; the
    win rate is their mean and the SE is ``sqrt(sample_var / n_pairs)`` — the
    pair is the unit, so within-pair correlation is already absorbed.

    This is the variance-correct alternative to treating ``2 * n_pairs`` games
    as independent: when the two games of a pair are positively correlated
    (the usual case with shared openings) the pentanomial concentrates mass in
    the 0 and 1 buckets, inflating ``var_drift`` and the SE relative to the
    naive binomial — exactly the anti-conservatism fix.
    """

    counts = np.asarray(penta, dtype=np.float64)
    if counts.shape != (5,):
        raise ValueError(f"pentanomial vector must have length 5, got {counts.shape}")
    if np.any(counts < 0):
        raise ValueError("pentanomial counts must be non-negative")
    n_pairs = int(round(float(counts.sum())))
    scores = np.array([0.0, 0.25, 0.5, 0.75, 1.0])  # candidate points / 2
    if n_pairs <= 0:
        return PairedResult(0, _as5(counts), float("nan"), float("nan"), float("nan"))
    weight = counts / counts.sum()
    mean = float((weight * scores).sum())
    # Population variance of the per-pair score over the realised distribution.
    var = float((weight * (scores - mean) ** 2).sum())
    if n_pairs > 1:
        # Sample-variance correction (Bessel) so the SE is unbiased for the
        # mean; with one pair the SE is undefined (reported as the population
        # sqrt(var/1), which is 0 for a single bucket — caller should widen).
        sample_var = var * n_pairs / (n_pairs - 1)
    else:
        sample_var = var
    se = math.sqrt(sample_var / n_pairs)
    return PairedResult(n_pairs, _as5(counts), mean, se, var)


def paired_winrate(pair_scores: "list[float] | np.ndarray") -> PairedResult:
    """Pair-level win rate + SE directly from per-pair scores in ``[0, 1]``.

    ``pair_scores`` is one value per complete pair = (candidate points in the
    pair) / 2, i.e. each entry is in ``{0, 0.25, 0.5, 0.75, 1}`` for Hexo (no
    draws) but any value in ``[0, 1]`` is accepted. Bins into the pentanomial
    and defers to :func:`pentanomial_summary`, so the SE is on ``N_pairs``
    units. This is the convenient entry point when the runner emits raw pair
    scores rather than a binned vector.
    """

    arr = np.asarray(pair_scores, dtype=np.float64)
    if arr.size and (np.any(arr < 0.0) or np.any(arr > 1.0)):
        raise ValueError("pair scores must lie in [0, 1]")
    penta = np.zeros(5, dtype=np.float64)
    # Nearest of the five canonical buckets (robust to fp dust); off-grid values
    # (shouldn't occur for Hexo) snap to the closest bucket for the count, but
    # the SE below is recomputed from the exact scores to stay faithful.
    canonical = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    for v in arr:
        penta[int(np.argmin(np.abs(canonical - v)))] += 1.0
    n_pairs = int(arr.size)
    if n_pairs <= 0:
        return PairedResult(0, _as5(penta), float("nan"), float("nan"), float("nan"))
    mean = float(arr.mean())
    var = float(arr.var())  # population variance of the realised scores
    sample_var = var * n_pairs / (n_pairs - 1) if n_pairs > 1 else var
    se = math.sqrt(sample_var / n_pairs)
    return PairedResult(n_pairs, _as5(penta), mean, se, var)


def effective_counts(result: PairedResult) -> tuple[float, float, float]:
    """Effective ``(wins, losses, n_eff)`` for feeding paired data to the BT fit.

    The BT likelihood (:func:`bradley_terry`) wants per-edge ``(wins, losses)``
    in *independent-Bernoulli* units. Paired games are over- (or under-)
    dispersed relative to that, so plugging in the raw ``2 * n_pairs`` games
    yields anti-conservative covariance. We deflate to an EFFECTIVE game count
    via the design effect ``deff = Var_paired / Var_binomial`` (the ratio of the
    realised pair-mean variance to the binomial variance of independent games),
    then split ``n_eff`` by the observed win rate.

    deff > 1 (positive within-pair correlation, the common case) shrinks
    ``n_eff`` below ``2 * n_pairs`` — fewer effective trials, wider CIs. deff < 1
    (anti-correlated pairs) is allowed but never inflates ``n_eff`` past the
    physical game count.
    """

    if result.n_pairs <= 0 or not math.isfinite(result.win_rate):
        return (0.0, 0.0, 0.0)
    n_games = 2.0 * result.n_pairs
    p = result.win_rate
    var_binom = p * (1.0 - p)  # per-game (independent) variance of a 0/1 score
    if var_binom <= 0.0:
        # Degenerate (all wins or all losses): no information about dispersion;
        # fall back to the physical game count (the BT prior keeps it finite).
        n_eff = n_games
    else:
        # var_drift is the per-pair variance of (points/2). For independent
        # games the per-pair mean variance would be var_binom / 2, so the design
        # effect on the pair mean is var_drift / (var_binom / 2). The effective
        # number of independent games is the physical count / deff.
        deff = (result.var_drift / (var_binom / 2.0)) if result.var_drift > 0 else 1.0
        deff = max(deff, 1e-6)
        n_eff = n_games / deff
        n_eff = min(n_eff, n_games)  # never claim more info than games played
    wins = p * n_eff
    losses = (1.0 - p) * n_eff
    return (wins, losses, n_eff)


# --------------------------------------------------------------------------- #
# 3. Elo from a win rate, with delta-method SE.
# --------------------------------------------------------------------------- #
def elo_from_winrate(p: float) -> float:
    """Elo difference implied by a win rate ``p`` (logistic / Bradley-Terry).

    ``elo = (400 / ln 10) * ln(p / (1 - p))``. Saturates at +-inf as ``p`` hits
    0 or 1 (an undefeated result is unbounded in Elo — callers should report the
    CI, not the point estimate, near the rails).
    """

    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")
    return ELO_PER_LOGIT * math.log(p / (1.0 - p))


def winrate_from_elo(elo: float) -> float:
    """Inverse of :func:`elo_from_winrate`: logistic of an Elo gap."""

    return 1.0 / (1.0 + math.exp(-elo * LOGIT_PER_ELO))


def elo_winrate_se(p: float, se_p: float) -> float:
    """Delta-method SE of the Elo estimate given win rate ``p`` and its SE.

    ``d/dp elo = (400 / ln 10) / (p (1 - p))``, so
    ``SE(elo) ~= (400 / ln 10) * SE(p) / (p (1 - p))``. Pass ``se_p`` from the
    PAIR-level SE (:func:`pentanomial_summary`) so the Elo CI inherits the
    pairing correction. Returns +inf at the rails (derivative blows up).
    """

    if not (0.0 < p < 1.0) or not math.isfinite(se_p):
        return float("inf")
    return ELO_PER_LOGIT * se_p / (p * (1.0 - p))


def elo_ci_from_winrate(p: float, se_p: float, z: float = Z95) -> tuple[float, float]:
    """Two-sided Elo CI from a win rate and its (pair-level) SE.

    Built on the win-rate scale then mapped through :func:`elo_from_winrate`
    (monotone), which keeps the interval sane near the rails where the
    delta-method Elo SE diverges. The point estimate uses ``p``; the endpoints
    use ``p +- z*se_p`` clipped to ``(0, 1)``.
    """

    lo_p = min(max(p - z * se_p, 0.0), 1.0)
    hi_p = min(max(p + z * se_p, 0.0), 1.0)
    return (elo_from_winrate(lo_p), elo_from_winrate(hi_p))


# --------------------------------------------------------------------------- #
# 4. Over-dispersion / SealBot down-weight hook (fix #5).
# --------------------------------------------------------------------------- #
def overdispersion_weight(observed_var: float, binomial_var: float) -> float:
    """Likelihood down-weight ``1 / deff`` for an over-dispersed edge.

    ``deff = observed_var / binomial_var`` is the design effect (>1 when the
    edge is noisier than an independent binomial — e.g. the SealBot edge, whose
    opponent depth drifts under GPU load). The BT likelihood contribution of
    such an edge is scaled by ``1 / max(deff, 1)`` so an over-dispersed edge
    cannot pretend to carry full Fisher information. An under-dispersed edge
    (deff < 1) is clamped to weight 1.0 — we never let an edge claim MORE
    information than its nominal game count.
    """

    if binomial_var <= 0.0 or observed_var <= 0.0:
        return 1.0
    deff = observed_var / binomial_var
    return 1.0 / deff if deff > 1.0 else 1.0


# --------------------------------------------------------------------------- #
# 5. Bradley-Terry rating pool — CONVERGED (fix #1), with full covariance.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BTEdge:
    """One head-to-head edge for the Bradley-Terry fit.

    a, b:    player labels (any hashable; conventionally strings).
    wins_a:  decided games won by ``a`` against ``b``.
    wins_b:  decided games won by ``b`` against ``a``.
    weight:  per-edge likelihood weight (default 1.0). Set < 1 to down-weight an
             over-dispersed edge (e.g. ``overdispersion_weight(...)`` for the
             SealBot edge), or use :func:`effective_counts` to pre-deflate paired
             games and leave ``weight`` at 1.0. Weighting scales BOTH the
             gradient and the Fisher information of the edge, so the covariance
             stays consistent with the down-weighted likelihood.
    """

    a: str
    b: str
    wins_a: float
    wins_b: float
    weight: float = 1.0


@dataclass(frozen=True)
class BTResult:
    """Converged Bradley-Terry fit, SealBot pinned at 0 Elo.

    players:    label -> index into the rating/covariance arrays.
    elo:        Elo rating per player (anchor pinned exactly at 0.0).
    cov:        full covariance matrix of the FREE (non-anchor) ratings, in
                Elo^2, embedded into the full ``(n, n)`` layout with the anchor
                row/col zeroed. Use :func:`var_diff` for ``Var(r_i - r_j)``.
    anchor:     the pinned label (zero-point of the scale).
    max_grad:   max|gradient| of the free params at the solution (asserted
                ``< grad_tol`` during the fit; reported for the manifest).
    iterations: optimiser iterations used.
    converged:  always True on return (the fit raises otherwise); kept for the
                record / serialisation.
    """

    players: dict[str, int]
    elo: np.ndarray
    cov: np.ndarray
    anchor: str
    max_grad: float
    iterations: int
    converged: bool

    def rating(self, label: str) -> float:
        return float(self.elo[self.players[label]])

    def se(self, label: str) -> float:
        i = self.players[label]
        return float(math.sqrt(max(self.cov[i, i], 0.0)))

    def elo_ci(self, label: str, z: float = Z95) -> tuple[float, float]:
        r = self.rating(label)
        s = self.se(label)
        return (r - z * s, r + z * s)

    def diff(self, i: str, j: str) -> float:
        """Elo difference ``r_i - r_j`` (point estimate)."""

        return self.rating(i) - self.rating(j)

    def var_diff(self, i: str, j: str) -> float:
        """``Var(r_i - r_j)`` using the FULL covariance (incl. ``-2 Cov_ij``)."""

        return var_diff(self.cov, self.players[i], self.players[j])

    def diff_ci(self, i: str, j: str, z: float = Z95) -> tuple[float, float]:
        """CI for ``r_i - r_j`` — the PRIMARY hypothesis test (fix #3).

        This is the only edge that carries a significance verdict: candidate
        ``i`` vs prior champion ``j``. The variance includes the shared-anchor
        positive covariance, which TIGHTENS the difference CI relative to adding
        the two marginal SEs in quadrature.
        """

        d = self.diff(i, j)
        s = math.sqrt(max(self.var_diff(i, j), 0.0))
        return (d - z * s, d + z * s)

    def se_diff(self, i: str, j: str) -> float:
        return float(math.sqrt(max(self.var_diff(i, j), 0.0)))


def var_diff(cov: np.ndarray, i: int, j: int) -> float:
    """``Var(r_i - r_j) = Cov_ii + Cov_jj - 2 Cov_ij`` from a covariance matrix.

    The cross term is the whole reason a *pooled* rating beats per-edge Elos:
    when ``i`` and ``j`` are measured against shared anchors, ``Cov_ij > 0`` and
    the difference variance shrinks below ``Cov_ii + Cov_jj``. Never compute a
    difference CI by adding marginal SEs in quadrature — that drops the
    ``-2 Cov_ij`` term and is anti-conservative for a comparison.
    """

    return float(cov[i, i] + cov[j, j] - 2.0 * cov[i, j])


def bradley_terry(
    edges: "list[BTEdge] | list[tuple]",
    *,
    anchor: str,
    prior_sd_elo: float = 1000.0,
    grad_tol: float = 1e-6,
    max_iter: int = 200,
    use_scipy: bool = True,
) -> BTResult:
    """Fit a Bradley-Terry rating pool, SealBot (``anchor``) pinned at 0 Elo.

    The model: ``P(a beats b) = sigma(r_a - r_b)`` with ``r`` in natural-log
    strength. We maximise the (weighted) binomial log-likelihood over the FREE
    ratings (every player except ``anchor``, which is fixed at 0) plus a weak
    Gaussian ridge prior ``r ~ N(0, prior_sd^2)`` that (a) keeps the fit
    identifiable / finite even with separated edges (an undefeated player has an
    unbounded MLE) and (b) makes the Hessian positive-definite so the covariance
    exists. The prior SD defaults to 1000 Elo — broad enough to be negligible
    against any real data, tight enough to regularise rails.

    CONVERGENCE (fix #1): we use Newton's method with a backtracking line search
    (or ``scipy.optimize.minimize(method="trust-ncg"/"Newton-CG")`` when
    available and ``use_scipy``), iterating until ``max|grad| < grad_tol``, then
    ``assert`` it. The covariance is the inverse of the Hessian of the negative
    log-posterior at that converged point — which is only meaningful BECAUSE we
    actually reached a stationary point (the legacy fixed-step GD did not).

    Returns ratings + covariance in Elo units (the internal logit-scale Hessian
    inverse is scaled by ``ELO_PER_LOGIT**2``). The anchor's row/col in ``cov``
    is zero (it is fixed, not estimated). Raises ``RuntimeError`` if the
    gradient tolerance is not met (never returns an unconverged fit).
    """

    norm_edges = _normalise_edges(edges)
    edge_labels = {lbl for e in norm_edges for lbl in (e.a, e.b)}
    if anchor not in edge_labels:
        # A pinned zero-point with no games is not a usable anchor: it would sit
        # at 0 Elo by fiat with no edge tying the scale to it, leaving the rest
        # of the pool free-floating. Require at least one anchor edge.
        raise ValueError(
            f"anchor {anchor!r} appears in no edge; the pinned zero-point must "
            f"play at least one game. Players with edges: {sorted(edge_labels)}"
        )
    labels = sorted(edge_labels)
    index = {lbl: k for k, lbl in enumerate(labels)}
    n = len(labels)
    a_idx = index[anchor]
    free = [k for k in range(n) if k != a_idx]  # estimated coordinates
    fpos = {k: p for p, k in enumerate(free)}   # full-index -> free-index
    m = len(free)

    # Precompute edge index arrays for a vectorised gradient/Hessian.
    ia = np.array([index[e.a] for e in norm_edges], dtype=np.int64)
    ib = np.array([index[e.b] for e in norm_edges], dtype=np.int64)
    wa = np.array([e.wins_a for e in norm_edges], dtype=np.float64)
    wb = np.array([e.wins_b for e in norm_edges], dtype=np.float64)
    we = np.array([e.weight for e in norm_edges], dtype=np.float64)
    nab = wa + wb
    prior_prec = 1.0 / (prior_sd_elo * LOGIT_PER_ELO) ** 2  # ridge in logit^2

    def full_r(theta: np.ndarray) -> np.ndarray:
        r = np.zeros(n)
        r[free] = theta
        return r  # anchor stays 0

    def neg_log_post(theta: np.ndarray) -> float:
        r = full_r(theta)
        d = r[ia] - r[ib]
        # -loglik via the numerically stable log-sigmoid; +ridge prior.
        ll = np.sum(we * (wa * _log_sigmoid(d) + wb * _log_sigmoid(-d)))
        prior = 0.5 * prior_prec * float(theta @ theta)
        return -ll + prior

    def grad_free(theta: np.ndarray) -> np.ndarray:
        r = full_r(theta)
        d = r[ia] - r[ib]
        p = _sigmoid(d)
        # dL/dr_a = weight * (wins_a - n_ab * p); chain to free coords.
        s = we * (wa - nab * p)
        g_full = np.zeros(n)
        np.add.at(g_full, ia, s)
        np.add.at(g_full, ib, -s)
        g = -g_full[free] + prior_prec * theta  # negative loglik + prior
        return g

    def hess_free(theta: np.ndarray) -> np.ndarray:
        r = full_r(theta)
        d = r[ia] - r[ib]
        p = _sigmoid(d)
        w_edge = we * nab * p * (1.0 - p)  # Fisher weight per edge
        H = np.zeros((n, n))
        # Each edge contributes w*[ [1,-1],[-1,1] ] over (a, b).
        np.add.at(H, (ia, ia), w_edge)
        np.add.at(H, (ib, ib), w_edge)
        np.add.at(H, (ia, ib), -w_edge)
        np.add.at(H, (ib, ia), -w_edge)
        Hf = H[np.ix_(free, free)]
        Hf = Hf + prior_prec * np.eye(m)  # prior curvature
        return Hf

    theta = np.zeros(m)

    used_scipy = False
    if use_scipy and m > 0:
        try:
            from scipy.optimize import minimize  # local import: optional dep

            res = minimize(
                neg_log_post, theta, jac=grad_free, hess=hess_free,
                method="trust-ncg",
                options={"gtol": grad_tol * 0.1, "maxiter": max_iter},
            )
            theta = np.asarray(res.x, dtype=np.float64)
            used_scipy = True
            iterations = int(getattr(res, "nit", 0))
        except Exception:
            used_scipy = False

    if not used_scipy:
        theta, iterations = _newton_solve(neg_log_post, grad_free, hess_free, theta, grad_tol, max_iter)

    # If scipy ran but missed tolerance, polish with Newton (belt and braces).
    if np.max(np.abs(grad_free(theta))) >= grad_tol:
        theta, extra = _newton_solve(neg_log_post, grad_free, hess_free, theta, grad_tol, max_iter)
        iterations += extra

    max_grad = float(np.max(np.abs(grad_free(theta)))) if m > 0 else 0.0
    # FIX #1 — the assert that the legacy script could not satisfy.
    assert max_grad < grad_tol, (
        f"Bradley-Terry did not converge: max|grad|={max_grad:.3e} "
        f">= grad_tol={grad_tol:.3e}. A non-stationary fit makes the "
        f"Hessian-inverse covariance meaningless (this is exactly the legacy "
        f"_wf_r4_bt_rating.py bug). Increase max_iter or check for degenerate edges."
    )

    # Covariance = inverse Hessian of the neg-log-posterior at the optimum,
    # scaled logit^2 -> Elo^2. Anchor row/col stay zero (fixed parameter).
    cov_full = np.zeros((n, n))
    if m > 0:
        Hf = hess_free(theta)
        cov_free = np.linalg.inv(Hf) * (ELO_PER_LOGIT ** 2)
        for p_i, k_i in enumerate(free):
            for p_j, k_j in enumerate(free):
                cov_full[k_i, k_j] = cov_free[p_i, p_j]

    r_full = full_r(theta)
    elo = r_full * ELO_PER_LOGIT
    return BTResult(
        players=index,
        elo=elo,
        cov=cov_full,
        anchor=anchor,
        max_grad=max_grad,
        iterations=iterations,
        converged=True,
    )


# --------------------------------------------------------------------------- #
# 6. SPRT — gross-regression triage (fix: honest about expected-N).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SPRTResult:
    """One SPRT step over a binomial (no-draw) win/loss tally.

    llr:       cumulative log-likelihood ratio of ``H1`` (elo1) vs ``H0`` (elo0).
    lower:     reject boundary ``ln(beta / (1 - alpha))`` (accept H0 below it).
    upper:     accept boundary ``ln((1 - beta) / alpha)`` (accept H1 above it).
    verdict:   "accept_h1" | "accept_h0" | "continue".
    """

    llr: float
    lower: float
    upper: float
    verdict: str


def sprt_llr(wins: int, losses: int, elo0: float, elo1: float) -> float:
    """Cumulative SPRT log-likelihood ratio for a no-draw (binomial) record.

    Per decided game the LLR increment is ``ln(p1/p0)`` for a win and
    ``ln((1-p1)/(1-p0))`` for a loss, where ``p_i = winrate_from_elo(elo_i)``.
    H0 is the null Elo gap ``elo0`` (typically a small negative = "gross
    regression"), H1 the alternative ``elo1`` (typically ~0 or slightly
    positive). This is the binomial SPRT; Hexo has no draws so there is no
    trinomial pentanomial-SPRT here (that device is only for paired CI variance,
    not for this triage).
    """

    p0 = winrate_from_elo(elo0)
    p1 = winrate_from_elo(elo1)
    p0 = min(max(p0, 1e-12), 1.0 - 1e-12)
    p1 = min(max(p1, 1e-12), 1.0 - 1e-12)
    return wins * math.log(p1 / p0) + losses * math.log((1.0 - p1) / (1.0 - p0))


def sprt_bounds(alpha: float, beta: float) -> tuple[float, float]:
    """Wald SPRT decision boundaries ``(lower, upper)``.

    ``upper = ln((1 - beta) / alpha)`` (cross above -> accept H1),
    ``lower = ln(beta / (1 - alpha))`` (cross below -> accept H0). With the
    usual ``alpha = beta`` these are symmetric about 0:
    ``upper = -lower = ln((1 - alpha) / alpha)``.
    """

    if not (0.0 < alpha < 1.0 and 0.0 < beta < 1.0):
        raise ValueError("alpha and beta must be in (0, 1)")
    upper = math.log((1.0 - beta) / alpha)
    lower = math.log(beta / (1.0 - alpha))
    return (lower, upper)


def sprt(
    wins: int,
    losses: int,
    *,
    elo0: float,
    elo1: float,
    alpha: float = 0.05,
    beta: float = 0.05,
) -> SPRTResult:
    """SPRT verdict for a no-draw record — a GROSS-REGRESSION TRIAGE only.

    HONESTY NOTE (do not advertise this as a calibrated 5%/5% test of a small
    edge): the SPRT's nominal ``alpha``/``beta`` hold only at the two simple
    hypotheses ``elo0`` and ``elo1``. For an effect sitting BETWEEN them (the
    indifference region — exactly where a borderline candidate lives) the
    expected sample size is large (order ~285 decided games for a [-X, +X]-Elo
    bracket), so any small per-stage game cap will MOSTLY return "continue"
    (escalate to the deep eval). Stage B uses this purely to catch a *gross*
    regression cheaply and to early-accept an obvious improvement; the calibrated
    measurement is Stage C/D (paired games + BT pool), never this.
    """

    lower, upper = sprt_bounds(alpha, beta)
    llr = sprt_llr(wins, losses, elo0, elo1)
    if llr >= upper:
        verdict = "accept_h1"
    elif llr <= lower:
        verdict = "accept_h0"
    else:
        verdict = "continue"
    return SPRTResult(llr=llr, lower=lower, upper=upper, verdict=verdict)


# --------------------------------------------------------------------------- #
# 7. Honest-resolution helpers (fix #4) and multiple-comparison helper (#3).
# --------------------------------------------------------------------------- #
def expected_se_elo(n_games: int, p: float = 0.5) -> float:
    """Approx single-block SE of a measured Elo difference from ``n_games``.

    Uses the binomial win-rate SE ``sqrt(p(1-p)/n)`` mapped through the
    delta-method Elo slope at ``p``. At ``p = 0.5`` and ``n = 128`` this is
    ~30.7 Elo for ONE win-rate of INDEPENDENT games. The honest single-epoch SE
    of the PRIMARY *difference* ``r_L - r_B`` is larger (~40-55 Elo): the games
    are PAIRED so the effective N is below 128 (design effect ~2 => ~64), and a
    difference of two ratings adds the ~``sqrt(2)`` of two SEs. Either way it is
    NOT the ~15-20 Elo that the *rolling, multi-epoch* pool asymptotes to. This
    function exists so callers/docs quote the honest single-epoch number rather
    than the multi-epoch asymptote.
    """

    if n_games <= 0:
        return float("inf")
    se_p = math.sqrt(p * (1.0 - p) / n_games)
    return elo_winrate_se(p, se_p)


def games_for_se(target_se_elo: float, p: float = 0.5) -> int:
    """Independent games needed for a target Elo SE (the multi-epoch budget).

    Inverts :func:`expected_se_elo`. E.g. reaching ~15 Elo SE near ``p = 0.5``
    needs on the order of 1000+ decided games — i.e. roughly ``ceil(1000/128)``
    epochs of the 128-game eval COMPOUNDING in the persisted pool, which is the
    only way the tight resolution is achieved. Makes fix #4 quantitative.
    """

    if target_se_elo <= 0.0:
        return 2**31 - 1
    slope = ELO_PER_LOGIT / (p * (1.0 - p))  # dElo/dp at p
    se_p = target_se_elo / slope
    if se_p <= 0.0:
        return 2**31 - 1
    return int(math.ceil(p * (1.0 - p) / (se_p * se_p)))


def bonferroni_alpha(alpha: float, k: int) -> float:
    """Per-edge alpha for ``k`` simultaneous gating comparisons (``alpha / k``).

    Only relevant if multiple edges must each gate (they do NOT by default — the
    primary verdict is a single BT difference test, fix #3). When a caller opts
    several edges into gating, split the family-wise ``alpha`` evenly so the
    overall false-positive rate stays bounded.
    """

    if k <= 0:
        raise ValueError("k must be >= 1")
    return alpha / k


# --------------------------------------------------------------------------- #
# Verdict label (PURE EVAL — never gates; just a reported string).
# --------------------------------------------------------------------------- #
def primary_verdict(
    diff_ci: tuple[float, float],
    *,
    regress_elo: float = 0.0,
) -> str:
    """Map a BT difference CI ``(lo, hi)`` for ``r_candidate - r_champion`` to a label.

    PURELY a reported label (the feature never gates):
      - "PROMOTE"      if the whole CI is above 0 (candidate provably stronger),
      - "REGRESS"      if the whole CI is below ``regress_elo`` (provably weaker),
      - "INCONCLUSIVE" otherwise (CI straddles the threshold — the common
        single-epoch outcome given fix #4's ~40-55 Elo SE).

    Promotion/gating consumers of this label MUST default OFF
    (``eval_gating_enabled = False``, ``eval_promotion_enabled = False``) and be
    wired to nothing that alters the training run.
    """

    lo, hi = diff_ci
    if lo > 0.0:
        return "PROMOTE"
    if hi < regress_elo:
        return "REGRESS"
    return "INCONCLUSIVE"


# --------------------------------------------------------------------------- #
# Internal numerics.
# --------------------------------------------------------------------------- #
def _phi(x: float) -> float:
    """Standard normal CDF via ``erf`` (no scipy needed)."""

    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _log_sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable ``log(sigmoid(x)) = -softplus(-x)``."""

    return -np.logaddexp(0.0, -x)


def _as5(counts: np.ndarray) -> tuple[int, int, int, int, int]:
    c = [int(round(v)) for v in counts.tolist()]
    return (c[0], c[1], c[2], c[3], c[4])


def _normalise_edges(edges: "list[BTEdge] | list[tuple]") -> list[BTEdge]:
    out: list[BTEdge] = []
    for e in edges:
        if isinstance(e, BTEdge):
            out.append(e)
        else:
            a, b, wa, wb = e[0], e[1], float(e[2]), float(e[3])
            weight = float(e[4]) if len(e) > 4 else 1.0
            out.append(BTEdge(a, b, wa, wb, weight))
    return out


def _newton_solve(
    f, grad, hess, theta0: np.ndarray, grad_tol: float, max_iter: int
) -> tuple[np.ndarray, int]:
    """Newton's method with backtracking line search to a gradient tolerance.

    Returns ``(theta, iterations)``. The Hessian is SPD here (Fisher info + ridge
    prior), so the Newton step is a descent direction; the Armijo backtrack
    guarantees monotone decrease of the neg-log-posterior and convergence to the
    unique minimiser. This is the convergence guarantee the legacy fixed-step GD
    lacked.
    """

    theta = np.array(theta0, dtype=np.float64)
    if theta.size == 0:
        return theta, 0
    for it in range(1, max_iter + 1):
        g = grad(theta)
        if np.max(np.abs(g)) < grad_tol:
            return theta, it - 1
        H = hess(theta)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        # Backtracking line search (Armijo) along the Newton direction -step.
        f0 = f(theta)
        t = 1.0
        gdotstep = float(g @ step)
        for _ls in range(40):
            cand = theta - t * step
            if f(cand) <= f0 - 1e-4 * t * gdotstep:
                break
            t *= 0.5
        theta = theta - t * step
    # Final tolerance check is the caller's assert; return best effort.
    return theta, max_iter
