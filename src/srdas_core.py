# %% [imports]
import math
import time
import warnings
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import pandas as pd
import networkx as nx
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=RuntimeWarning)
TRADING_DAYS = 252


# %% [estimators]
def standardize(X):
    """Column-standardize a T x N array (zero-variance columns left at 0)."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return (X - mu) / sd


def shrunk_corr(X):
    """Ledoit-Wolf shrunk correlation matrix of a T x N return array.

    Returns are standardized first, so Ledoit-Wolf shrinks the sample
    correlation toward the identity. Works when T < N (as on stress days).
    """
    Z = standardize(np.asarray(X, dtype=float))
    S = LedoitWolf().fit(Z).covariance_
    d = np.sqrt(np.clip(np.diag(S), 1e-12, None))
    C = S / np.outer(d, d)
    np.fill_diagonal(C, 1.0)
    return C


def nearest_psd_corr(C, eps=1e-6):
    """Project a symmetric matrix to a valid (PSD, unit-diagonal) correlation matrix.

    Eigenvalue clipping followed by rescaling to unit diagonal. Rescaling is a
    congruence transform, so positive semi-definiteness is preserved.
    """
    C = (C + C.T) / 2.0
    w, V = np.linalg.eigh(C)
    w = np.clip(w, eps, None)
    C2 = (V * w) @ V.T
    d = np.sqrt(np.diag(C2))
    C2 = C2 / np.outer(d, d)
    C2 = (C2 + C2.T) / 2.0
    np.fill_diagonal(C2, 1.0)
    return C2


def stress_mask(mkt, stress_q):
    """Boolean mask of 'stress days': market return at or below its stress_q quantile."""
    thr = np.quantile(mkt, stress_q)
    return mkt <= thr


def window_starts(T, sub_len, sub_step):
    """Start indices of sub-windows of length sub_len, anchored at the most recent day."""
    if sub_len >= T:
        return [0]
    starts = []
    s = T - sub_len
    while s >= 0:
        starts.append(s)
        s -= sub_step
    return starts[::-1]


def robust_stress_corr(R, mkt, stress_q=0.10, sub_len=504, sub_step=126,
                       agg_q=0.75, min_stress_days=20):
    """Stress-conditioned, temporally robust correlation matrix (the SR-DAS dependence estimate).

    1. Split the look-back into overlapping sub-windows.
    2. In each sub-window keep only stress days (market in its bottom stress_q tail)
       and estimate a Ledoit-Wolf shrunk correlation on those days.
    3. Aggregate pessimistically: for every pair take the agg_q quantile of its
       stress correlation across sub-windows (agg_q > 0.5 = assume the pair is
       as dependent as it has reliably been in bad times).
    4. Project to the nearest valid correlation matrix so log-det is well defined.

    R: T x N array of returns, mkt: length-T array of market returns.
    Returns (C, n_windows_used).
    """
    R = np.asarray(R, dtype=float)
    mkt = np.asarray(mkt, dtype=float)
    T = R.shape[0]
    mats = []
    for s in window_starts(T, sub_len, sub_step):
        e = min(T, s + sub_len)
        m = stress_mask(mkt[s:e], stress_q)
        if m.sum() < min_stress_days:
            continue
        mats.append(shrunk_corr(R[s:e][m]))
    if not mats:  # fall back to all-day correlation
        return shrunk_corr(R), 0
    stack = np.stack(mats)
    C = np.quantile(stack, agg_q, axis=0) if len(mats) > 1 else stack[0]
    return nearest_psd_corr(C), len(mats)


def lower_tail_dependence(R, u=0.10):
    """Empirical (non-parametric) lower tail-dependence coefficients.

    lambda_ij = P(U_i <= u, U_j <= u) / u, with U the column ranks scaled to (0,1).
    This is the copula-free estimator used in tail-dependence clustering work.
    """
    R = np.asarray(R, dtype=float)
    T = R.shape[0]
    U = (pd.DataFrame(R).rank(axis=0).values) / (T + 1.0)
    I = (U <= u).astype(float)
    lam = (I.T @ I) / T / u
    lam = np.clip(lam, 0.0, 1.0)
    np.fill_diagonal(lam, 1.0)
    return lam


# %% [quality]
def rank_normal(x):
    """Map a cross-section to standard-normal scores via ranks (robust to outliers)."""
    x = pd.Series(x)
    r = x.rank(method="average")
    return pd.Series(norm.ppf((r - 0.5) / len(x)), index=x.index)


def quality_scores(R_df, vol_window=252, mom_window=252, mom_skip=21):
    """Composite quality score z_i from price-only signals with out-of-sample evidence.

    - Low volatility: minus the standard deviation of the last vol_window daily returns.
    - 12-1 momentum: return from t-mom_window to t-mom_skip (skips the last month).
    Each signal is rank-normalized, averaged and re-normalized to a z-score.
    Uses only data up to the rebalance date (no look-ahead).
    """
    X = R_df.values
    vol = X[-vol_window:].std(axis=0, ddof=1)
    seg = X[-mom_window:-mom_skip] if mom_skip > 0 else X[-mom_window:]
    mom = np.prod(1.0 + seg, axis=0) - 1.0
    z = (rank_normal(-vol).values + rank_normal(mom).values) / 2.0
    z = (z - z.mean()) / (z.std(ddof=1) + 1e-12)
    return pd.Series(z, index=R_df.columns)


# %% [selection]
def affine_corr_kernel(C, a=0.5):
    """Similarity kernel K = (1-a) C + a 11^T.

    Maps correlation -1..1 to similarity 0..1 for a=0.5, so that NEGATIVE
    correlation counts as MORE diverse (a plain correlation kernel would
    penalize hedging pairs, since det([[1,r],[r,1]]) = 1-r^2).
    K is PSD because it is a sum of PSD matrices. One can show
    det(K_S) = (1-a)^k det(C_S) (1 + a/(1-a) * 1' C_S^{-1} 1),
    i.e. it rewards both low generalized correlation and a low-risk
    minimum-variance combination of the chosen assets.
    """
    return (1.0 - a) * C + a * np.ones_like(C)


def greedy_logdet(L, k):
    """Greedy MAP for log det(L_S) with |S| = k (fast incremental Cholesky).

    log det is submodular, so greedy has the classic (1 - 1/e) guarantee for
    the monotone case and is the standard DPP MAP heuristic otherwise.
    Returns the list of selected indices (in order of selection).
    """
    N = L.shape[0]
    k = min(k, N)
    cis = np.zeros((k, N))
    di2 = np.diag(L).astype(float).copy()
    sel = []
    for j in range(k):
        score = di2.copy()
        score[sel] = -np.inf
        i = int(np.argmax(score))
        if di2[i] <= 1e-12:
            break
        sel.append(i)
        ei = (L[i, :] - cis[:j, i] @ cis[:j, :]) / np.sqrt(di2[i])
        cis[j, :] = ei
        di2 = di2 - ei ** 2
    return sel


def logdet_select(C, z, k, theta=0.5, a=0.5):
    """SR-DAS selection rule on a given correlation estimate C and quality z.

    L = diag(q) K diag(q) with q = exp(theta * z), so that
    log det L_S = 2 theta sum_{i in S} z_i + log det K_S
    (theta trades quality against diversity; theta=0 is pure diversity).
    """
    K = affine_corr_kernel(C, a)
    q = np.exp(theta * np.asarray(z))
    L = K * np.outer(q, q)
    return greedy_logdet(L, k)


def cluster_pick(D, z, k, method="average"):
    """Cluster-then-pick baseline: hierarchical clustering on distance D into k
    clusters, then the highest-quality asset from each cluster."""
    D = np.asarray(D, dtype=float)
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    D = np.clip(D, 0.0, None)
    Z = linkage(squareform(D, checks=False), method=method)
    labels = fcluster(Z, t=k, criterion="maxclust")
    z = np.asarray(z)
    sel = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        sel.append(int(idx[np.argmax(z[idx])]))
    # fcluster can return fewer than k clusters on ties; top up by quality
    if len(sel) < k:
        rest = [i for i in np.argsort(-z) if i not in sel]
        sel += rest[: k - len(sel)]
    return sel[:k]


def corr_distance(C):
    return np.sqrt(np.clip(2.0 * (1.0 - C), 0.0, None))


def mst_periphery_select(C, k):
    """Pozzi, Di Matteo & Aste (2013)-style selection: build the Minimum Spanning
    Tree on correlation distance and pick the k most peripheral nodes.

    Hybrid centrality = rank(degree) + rank(betweenness) + rank(closeness)
    + rank(-eccentricity); lowest values are the most peripheral.
    (Simplified version of their hybrid measure, on the MST only.)
    """
    from scipy.sparse.csgraph import minimum_spanning_tree
    N = C.shape[0]
    D = corr_distance(C) + 1e-9          # tiny offset: zero distances would be dropped as non-edges
    np.fill_diagonal(D, 0.0)
    mst = minimum_spanning_tree(D).tocoo()
    T = nx.Graph()
    T.add_nodes_from(range(N))
    T.add_weighted_edges_from(zip(mst.row.tolist(), mst.col.tolist(), mst.data.tolist()))
    deg = pd.Series(dict(T.degree(weight=None)))
    btw = pd.Series(nx.betweenness_centrality(T, weight="weight"))
    clo = pd.Series(nx.closeness_centrality(T, distance="weight"))
    ecc = pd.Series(nx.eccentricity(T))
    score = deg.rank() + btw.rank() + clo.rank() + (-ecc).rank()
    return [int(i) for i in score.sort_values().index[:k]]


def max_drawdown(x):
    """Maximum drawdown (negative number) of a return array or matrix (per column)."""
    w = np.cumprod(1.0 + np.asarray(x), axis=0)
    peak = np.maximum.accumulate(w, axis=0)
    return (w / peak - 1.0).min(axis=0)


def pca_kmeans_select(R_df, k, threshold=0.5, per_cluster=2, seed=0):
    """Re-implementation of the PCA-KMeans stratified selection (Vo, Wang & Nguyen, 2026).

    Features: 3-month return, Sharpe, 60-day volatility (stability), max drawdown,
    all computed over the last year of the look-back. Min-max scaled, risk
    features flipped, PC1 loadings (absolute, normalized) used as weights;
    stocks with score > threshold are clustered with K-Means and the top
    `per_cluster` by score are taken from each cluster.
    """
    X = R_df.values[-TRADING_DAYS:]
    ret3m = np.prod(1.0 + X[-63:], axis=0) - 1.0
    sd = X.std(axis=0, ddof=1) + 1e-12
    sharpe = X.mean(axis=0) / sd * np.sqrt(TRADING_DAYS)
    vol60 = X[-60:].std(axis=0, ddof=1)
    mdd = -max_drawdown(X)  # positive size of drawdown
    F = np.column_stack([ret3m, sharpe, vol60, mdd])
    mn, mx = F.min(axis=0), F.max(axis=0)
    F = (F - mn) / np.where(mx - mn > 1e-12, mx - mn, 1.0)
    F[:, 2] = 1.0 - F[:, 2]
    F[:, 3] = 1.0 - F[:, 3]
    w = np.abs(PCA(n_components=1).fit(F).components_[0])
    w = w / w.sum()
    score = F @ w
    good = np.where(score > threshold)[0]
    n_clusters = int(math.ceil(k / per_cluster))
    sel = []
    if len(good) > n_clusters:
        lab = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(F[good])
        for c in range(n_clusters):
            idx = good[lab == c]
            sel += [int(i) for i in idx[np.argsort(-score[idx])][:per_cluster]]
    else:
        sel = [int(i) for i in good]
    if len(sel) < k:
        rest = [int(i) for i in np.argsort(-score) if i not in sel]
        sel += rest[: k - len(sel)]
    return sel[:k]


# %% [weights]
def hrp_weights(cov):
    """Hierarchical Risk Parity (Lopez de Prado, 2016) on a covariance matrix."""
    cov = np.asarray(cov)
    sd = np.sqrt(np.diag(cov))
    C = cov / np.outer(sd, sd)
    D = corr_distance(np.clip(C, -1, 1))
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="single")
    # quasi-diagonal order
    N = cov.shape[0]
    order = [int(Z[-1, 0]), int(Z[-1, 1])]
    while max(order) >= N:
        new = []
        for i in order:
            if i >= N:
                r = int(i - N)
                new += [int(Z[r, 0]), int(Z[r, 1])]
            else:
                new.append(i)
        order = new
    w = pd.Series(1.0, index=order)
    clusters = [order]
    while clusters:
        nxt = []
        for c in clusters:
            if len(c) <= 1:
                continue
            h = len(c) // 2
            c1, c2 = c[:h], c[h:]

            def cvar(ix):
                sub = cov[np.ix_(ix, ix)]
                iv = 1.0 / np.diag(sub)
                iv /= iv.sum()
                return float(iv @ sub @ iv)

            v1, v2 = cvar(c1), cvar(c2)
            alpha = 1.0 - v1 / (v1 + v2)
            w[c1] *= alpha
            w[c2] *= 1.0 - alpha
            nxt += [c1, c2]
        clusters = nxt
    w = w.sort_index().values
    return w / w.sum()


def minvar_weights(cov, max_w=1.0):
    """Long-only minimum-variance weights (fully invested), solved with SLSQP."""
    n = cov.shape[0]
    x0 = np.ones(n) / n
    res = minimize(lambda w: w @ cov @ w, x0, jac=lambda w: 2 * cov @ w,
                   bounds=[(0.0, max_w)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                   method="SLSQP", options={"maxiter": 500, "ftol": 1e-12})
    w = np.clip(res.x if res.success else x0, 0, None)
    return w / w.sum()


def compute_weights(scheme, R_sel):
    """Portfolio weights for the selected assets, estimated on the look-back only."""
    n = R_sel.shape[1]
    if scheme == "ew" or n == 1:
        return np.ones(n) / n
    X = R_sel[-TRADING_DAYS:]
    if scheme == "invvol":
        iv = 1.0 / (X.std(axis=0, ddof=1) + 1e-12)
        return iv / iv.sum()
    cov = LedoitWolf().fit(X).covariance_
    if scheme == "minvar":
        return minvar_weights(cov)
    if scheme == "hrp":
        return hrp_weights(cov)
    raise ValueError(scheme)


# %% [engine]
@dataclass
class Config:
    k: int = 10                  # number of assets to select
    lookback: int = 1260         # look-back in trading days (5 years)
    rebalance_every: int = 63    # trading days between rebalances (quarterly)
    stress_q: float = 0.10       # stress days = market in bottom 10% of the sub-window
    sub_len: int = 504           # sub-window length for robust aggregation (2 years)
    sub_step: int = 126          # step between sub-windows (6 months)
    agg_q: float = 0.75          # pessimistic quantile across sub-windows
    theta: float = 0.5           # quality-diversity trade-off
    kernel_a: float = 0.5        # affine kernel parameter
    tail_u: float = 0.10         # tail level for empirical tail dependence
    min_coverage: float = 0.95   # min share of non-missing days in the look-back
    cost_bps: float = 10.0       # one-way transaction cost (bps of traded value)
    test_start: str = "1990-01-01"
    test_end: str = None
    n_random: int = 20           # random-k replications


class Context:
    """Everything a selection method may use at one rebalance date (look-back data only).
    Expensive estimates are computed once and shared across methods."""

    def __init__(self, R_df, mkt, cfg, date=None):
        self.date = date
        self.R_df = R_df
        self.R = R_df.values
        self.mkt = np.asarray(mkt)
        self.cfg = cfg
        self.tickers = list(R_df.columns)

    @cached_property
    def z(self):
        return quality_scores(self.R_df).values

    @cached_property
    def C_plain(self):
        return shrunk_corr(self.R)

    @cached_property
    def C_robust(self):
        c = self.cfg
        return robust_stress_corr(self.R, self.mkt, c.stress_q, c.sub_len, c.sub_step, c.agg_q)[0]

    @cached_property
    def C_stress_single(self):
        c = self.cfg
        n = len(self.mkt)
        return robust_stress_corr(self.R, self.mkt, c.stress_q, n, n, 0.5)[0]

    @cached_property
    def C_plain_robust(self):
        """Robust aggregation across sub-windows but on ALL days (no stress conditioning)."""
        c = self.cfg
        return robust_stress_corr(self.R, self.mkt, 1.0, c.sub_len, c.sub_step, c.agg_q)[0]

    @cached_property
    def taildep(self):
        return lower_tail_dependence(self.R, self.cfg.tail_u)


# Each selection method: f(ctx) -> list of column indices.
def m_srdas(ctx, theta=None):
    th = ctx.cfg.theta if theta is None else theta
    return logdet_select(ctx.C_robust, ctx.z, ctx.cfg.k, th, ctx.cfg.kernel_a)


def m_srdas_noq(ctx):
    return logdet_select(ctx.C_robust, ctx.z, ctx.cfg.k, 0.0, ctx.cfg.kernel_a)


def m_das_plain(ctx):
    return logdet_select(ctx.C_plain, ctx.z, ctx.cfg.k, ctx.cfg.theta, ctx.cfg.kernel_a)


def m_das_plain_robust(ctx):
    return logdet_select(ctx.C_plain_robust, ctx.z, ctx.cfg.k, ctx.cfg.theta, ctx.cfg.kernel_a)


def m_das_stress_single(ctx):
    return logdet_select(ctx.C_stress_single, ctx.z, ctx.cfg.k, ctx.cfg.theta, ctx.cfg.kernel_a)


def m_stress_cluster(ctx):
    return cluster_pick(corr_distance(ctx.C_robust), ctx.z, ctx.cfg.k)


def m_corr_cluster(ctx):
    return cluster_pick(corr_distance(ctx.C_plain), ctx.z, ctx.cfg.k)


def m_taildep_cluster(ctx):
    return cluster_pick(1.0 - ctx.taildep, ctx.z, ctx.cfg.k)


def m_mst_periphery(ctx):
    return mst_periphery_select(ctx.C_plain, ctx.cfg.k)


def m_pca_kmeans(ctx):
    return pca_kmeans_select(ctx.R_df, ctx.cfg.k)


def m_topk_quality(ctx):
    return [int(i) for i in np.argsort(-ctx.z)[: ctx.cfg.k]]


def make_random(seed):
    def m_random(ctx):
        day = ctx.date.toordinal() if ctx.date is not None else 0
        rng = np.random.default_rng([seed, day])
        return [int(i) for i in rng.choice(len(ctx.tickers), size=min(ctx.cfg.k, len(ctx.tickers)), replace=False)]
    return m_random


def m_all(ctx):
    return list(range(len(ctx.tickers)))


@dataclass
class Strategy:
    name: str
    select: callable
    weights: str = "ew"
    group: str = ""


def default_strategies(cfg, weights="ew", include_random=True):
    """The full comparison set: proposed method, ablations, baselines, benchmarks."""
    S = [
        Strategy("SR-DAS", m_srdas, weights, "Proposed"),
        # ablations (each removes one component of SR-DAS)
        Strategy("Abl: Stress-Cluster (rule)", m_stress_cluster, weights, "Ablation"),
        Strategy("Abl: DAS all-days (stress)", m_das_plain_robust, weights, "Ablation"),
        Strategy("Abl: DAS stress 1-window (robust)", m_das_stress_single, weights, "Ablation"),
        Strategy("Abl: SR-DAS theta=0 (quality)", m_srdas_noq, weights, "Ablation"),
        Strategy("Abl: DAS plain corr (all)", m_das_plain, weights, "Ablation"),
        # literature baselines
        Strategy("TailDep-Cluster", m_taildep_cluster, weights, "Baseline"),
        Strategy("Corr-Cluster", m_corr_cluster, weights, "Baseline"),
        Strategy("MST-Periphery", m_mst_periphery, weights, "Baseline"),
        Strategy("PCA-KMeans (Vo et al.)", m_pca_kmeans, weights, "Baseline"),
        Strategy("Top-k Quality", m_topk_quality, weights, "Baseline"),
        # full-universe benchmarks (not selection methods)
        Strategy("EW-All (1/N)", m_all, "ew", "Benchmark"),
        Strategy("HRP-All", m_all, "hrp", "Benchmark"),
    ]
    if include_random:
        S += [Strategy(f"Random-k #{s}", make_random(s), weights, "Random") for s in range(cfg.n_random)]
    return S


@dataclass
class Result:
    name: str
    group: str
    returns: pd.Series                      # daily net portfolio returns
    gross_returns: pd.Series
    turnover: pd.Series                     # one-way turnover at each rebalance
    holdings: dict = field(default_factory=dict)   # rebalance date -> {ticker: weight}
    diag: pd.DataFrame = None               # per-period diagnostics


def drifted_weights(w, R_hold):
    """Buy-and-hold weights at the end of a holding period."""
    v = w * np.prod(1.0 + R_hold, axis=0)
    s = v.sum()
    return v / s if s > 0 else w


def avg_pairwise_corr(X):
    sd = X.std(axis=0, ddof=1)
    ok = sd > 1e-12
    if ok.sum() < 2 or X.shape[0] < 5:
        return np.nan
    C = np.corrcoef(X[:, ok], rowvar=False)
    return (C.sum() - np.trace(C)) / (C.shape[0] * (C.shape[0] - 1))


def period_diagnostics(R_hold, w):
    """Out-of-sample diversification of the held basket over one holding period."""
    n = R_hold.shape[1]
    if n < 2 or R_hold.shape[0] < 5:
        return np.nan, np.nan
    sd = R_hold.std(axis=0, ddof=1)
    ok = sd > 1e-12
    if ok.sum() < 2:
        return np.nan, np.nan
    C = np.corrcoef(R_hold[:, ok], rowvar=False)
    avg_corr = (C.sum() - np.trace(C)) / (C.shape[0] * (C.shape[0] - 1))
    port_sd = (R_hold @ w).std(ddof=1)
    dr = float(w @ sd / port_sd) if port_sd > 1e-12 else np.nan
    return avg_corr, dr


def run_backtest(returns, market, strategies, cfg, universe_fn=None, verbose=True, eval_stress_q=0.10):
    """Walk-forward backtest.

    At each rebalance date t (every cfg.rebalance_every days):
      * eligible assets = universe_fn(t) with >= min_coverage data in the look-back
        and a valid return on day t;
      * each strategy selects/weights using ONLY data in (t-lookback, t];
      * portfolios are held buy-and-hold from t+1 to the next rebalance.
    Missing returns while held (e.g. delisting) are treated as 0 (cash).
    Transaction cost = cost_bps * one-way turnover, charged on the first day.
    """
    returns = returns.sort_index()
    market = market.reindex(returns.index)
    dates = returns.index
    t0 = dates.searchsorted(pd.Timestamp(cfg.test_start))
    t0 = max(t0, cfg.lookback)
    t_end = len(dates) if cfg.test_end is None else dates.searchsorted(pd.Timestamp(cfg.test_end), side="right")
    rebal_idx = list(range(t0 - 1, t_end - 1, cfg.rebalance_every))
    # Evaluation-only threshold for "stress days" in the test period (never used for selection)
    eval_thr = market.iloc[t0:t_end].quantile(eval_stress_q)

    out = {s.name: dict(net=[], gross=[], turn={}, hold={}, diag=[], prev=None) for s in strategies}
    tic = time.time()
    for n_r, t in enumerate(rebal_idx):
        t_next = min(t + cfg.rebalance_every, t_end - 1)
        if t_next <= t:
            break
        win = returns.iloc[t - cfg.lookback + 1: t + 1]
        mkt_win = market.iloc[t - cfg.lookback + 1: t + 1]
        cols = list(win.columns) if universe_fn is None else [c for c in universe_fn(dates[t]) if c in win.columns]
        w_df = win[cols]
        cov_ok = w_df.notna().mean() >= cfg.min_coverage
        last_ok = w_df.iloc[-1].notna()
        elig = [c for c in cols if cov_ok[c] and last_ok[c]]
        mk_ok = mkt_win.notna().values
        R_df = w_df.loc[mk_ok, elig].fillna(0.0)
        ctx = Context(R_df, mkt_win.values[mk_ok], cfg, dates[t])
        hold = returns.iloc[t + 1: t_next + 1]
        hold_stress = (market.iloc[t + 1: t_next + 1] <= eval_thr).values
        for s in strategies:
            o = out[s.name]
            idx = s.select(ctx)
            tick = [elig[i] for i in idx]
            w = compute_weights(s.weights, R_df.values[:, idx])
            R_hold = hold[tick].fillna(0.0).values
            # buy-and-hold path
            growth = np.cumprod(1.0 + R_hold, axis=0) * w
            val = np.concatenate([[1.0], growth.sum(axis=1)])
            r_gross = val[1:] / val[:-1] - 1.0
            # turnover vs drifted previous portfolio
            new = pd.Series(w, index=tick)
            prev = o["prev"]
            if prev is None:
                turn = 1.0
            else:
                u = new.index.union(prev.index)
                turn = 0.5 * float(np.abs(new.reindex(u, fill_value=0) - prev.reindex(u, fill_value=0)).sum())
            r_net = r_gross.copy()
            r_net[0] -= turn * cfg.cost_bps / 1e4
            o["gross"].append(pd.Series(r_gross, index=hold.index))
            o["net"].append(pd.Series(r_net, index=hold.index))
            o["turn"][dates[t]] = turn
            o["hold"][dates[t]] = dict(zip(tick, np.round(w, 6)))
            o["prev"] = pd.Series(drifted_weights(w, R_hold), index=tick)
            ac, dr = period_diagnostics(R_hold, w)
            n_st = int(hold_stress.sum())
            ac_st = avg_pairwise_corr(R_hold[hold_stress]) if n_st >= 5 else np.nan
            o["diag"].append(dict(date=dates[t], n_eligible=len(elig), n_held=len(tick),
                                  avg_corr_oos=ac, div_ratio_oos=dr,
                                  stress_corr_oos=ac_st, n_stress_days=n_st))
        if verbose and (n_r % 10 == 0 or n_r == len(rebal_idx) - 1):
            print(f"  rebalance {n_r + 1}/{len(rebal_idx)}  {dates[t].date()}  eligible={len(elig)}  "
                  f"elapsed={time.time() - tic:.0f}s")
    results = {}
    for s in strategies:
        o = out[s.name]
        results[s.name] = Result(s.name, s.group, pd.concat(o["net"]), pd.concat(o["gross"]),
                                 pd.Series(o["turn"]), o["hold"], pd.DataFrame(o["diag"]).set_index("date"))
    return results


# %% [metrics]
def perf_metrics(r, rf=None):
    """Standard performance metrics of a daily return series."""
    r = r.dropna()
    rf_ = pd.Series(0.0, index=r.index) if rf is None else rf.reindex(r.index).fillna(0.0)
    ex = r - rf_
    yrs = len(r) / TRADING_DAYS
    wealth = (1 + r).cumprod()
    cagr = wealth.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else np.nan
    vol = r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe = ex.mean() / ex.std(ddof=1) * np.sqrt(TRADING_DAYS)
    dn = ex[ex < 0]
    sortino = ex.mean() * TRADING_DAYS / (np.sqrt((dn ** 2).sum() / len(ex)) * np.sqrt(TRADING_DAYS))
    mdd = (wealth / wealth.cummax() - 1).min()
    cvar = r[r <= r.quantile(0.05)].mean()
    return dict(CAGR=cagr, Vol=vol, Sharpe=sharpe, Sortino=sortino, MaxDD=mdd,
                Calmar=cagr / abs(mdd) if mdd < 0 else np.nan, CVaR5_daily=cvar)


def stress_stats(r, market, q=0.10):
    """Out-of-sample behaviour on market stress days (bottom q of the market in the window)."""
    df = pd.concat([r, market], axis=1, keys=["p", "m"], sort=True).dropna()
    st = df[df["m"] <= df["m"].quantile(q)]
    beta = np.cov(st["p"], st["m"])[0, 1] / st["m"].var() if len(st) > 5 else np.nan
    return dict(StressBeta=beta, StressRet_daily=st["p"].mean())


def summary_table(results, rf=None, start=None, end=None, collapse_random=True, market=None):
    """One row per strategy; Random-k replications are averaged into one row.

    If `market` is given, adds out-of-sample stress diagnostics: beta and mean return on
    market stress days, and the average pairwise correlation of the held basket on
    stress days (weighted by number of stress days per holding period)."""
    rows = {}
    for name, res in results.items():
        r = res.returns.loc[start:end]
        if len(r) < 20:
            continue
        m = perf_metrics(r, rf)
        yrs = len(r) / TRADING_DAYS
        tv = res.turnover.loc[start:end] if start or end else res.turnover
        m["Turnover/yr"] = tv.sum() / yrs if yrs > 0 else np.nan
        d = res.diag.loc[start:end] if res.diag is not None else None
        m["AvgCorr_OOS"] = d["avg_corr_oos"].mean() if d is not None else np.nan
        m["DivRatio_OOS"] = d["div_ratio_oos"].mean() if d is not None else np.nan
        if d is not None and "stress_corr_oos" in d:
            ok = d["stress_corr_oos"].notna()
            m["StressCorr_OOS"] = np.average(d.loc[ok, "stress_corr_oos"], weights=d.loc[ok, "n_stress_days"]) if ok.any() else np.nan
        if market is not None:
            m.update(stress_stats(r, market.loc[start:end]))
        m["Group"] = res.group
        rows[name] = m
    df = pd.DataFrame(rows).T
    if collapse_random:
        rnd = [n for n in df.index if n.startswith("Random-k #")]
        if rnd:
            num = df.loc[rnd].drop(columns="Group").astype(float).mean()
            num["Group"] = "Random"
            df = df.drop(index=rnd)
            df.loc[f"Random-k (mean of {len(rnd)})"] = num
    order = {"Proposed": 0, "Proposed-2": 0.5, "Ablation": 1, "Original": 1.5, "Baseline": 2, "Random": 3, "Benchmark": 4}
    df["_o"] = df["Group"].map(order)
    df = df.sort_values(["_o"], kind="stable").drop(columns="_o")
    cols = ["Group", "CAGR", "Vol", "Sharpe", "Sortino", "MaxDD", "Calmar", "CVaR5_daily",
            "Turnover/yr", "AvgCorr_OOS", "StressCorr_OOS", "DivRatio_OOS", "StressBeta", "StressRet_daily"]
    return df[[c for c in cols if c in df.columns]]


def format_table(df):
    fmt = {"CAGR": "{:.2%}", "Vol": "{:.2%}", "Sharpe": "{:.3f}", "Sortino": "{:.3f}",
           "MaxDD": "{:.2%}", "Calmar": "{:.3f}", "CVaR5_daily": "{:.2%}", "Turnover/yr": "{:.2f}",
           "AvgCorr_OOS": "{:.3f}", "DivRatio_OOS": "{:.3f}", "StressCorr_OOS": "{:.3f}",
           "StressBeta": "{:.3f}", "StressRet_daily": "{:.2%}"}
    out = df.copy()
    for c, f in fmt.items():
        if c in out:
            out[c] = out[c].map(lambda v: f.format(v) if pd.notna(v) else "")
    return out


def crisis_table(results, windows, market=None):
    """Cumulative return and max drawdown of each strategy inside crisis windows."""
    rows = []
    series = {n: r.returns for n, r in results.items() if not n.startswith("Random-k #")}
    rnd = [r.returns for n, r in results.items() if n.startswith("Random-k #")]
    if rnd:
        series[f"Random-k (mean of {len(rnd)})"] = pd.concat(rnd, axis=1, sort=True).mean(axis=1)
    if market is not None:
        series["Market"] = market
    for label, (a, b) in windows.items():
        for n, r in series.items():
            x = r.loc[a:b].dropna()
            if len(x) < 5:
                continue
            w = (1 + x).cumprod()
            rows.append(dict(Crisis=label, Strategy=n, Return=w.iloc[-1] - 1,
                             MaxDD=(w / w.cummax() - 1).min()))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.pivot(index="Strategy", columns="Crisis", values="MaxDD"), \
        df.pivot(index="Strategy", columns="Crisis", values="Return")


def _sharpe(x):
    s = x.std(axis=0, ddof=1)
    return x.mean(axis=0) / np.where(s > 0, s, np.nan)


def sharpe_test(a, b, rf=None, block=21, B=2000, seed=0):
    """Test H0: Sharpe(a) = Sharpe(b) for two daily return series.

    * Jobson-Korkie test with Memmel (2003) correction (assumes iid normal).
    * Circular block bootstrap (block length `block`) of the Sharpe difference,
      centred under H0 — robust to autocorrelation and fat tails, in the
      spirit of Ledoit & Wolf (2008).
    Returns annualized difference and both p-values.
    """
    df = pd.concat([a, b], axis=1, sort=True).dropna()
    if rf is not None:
        r = rf.reindex(df.index).fillna(0.0).values[:, None]
        X = df.values - r
    else:
        X = df.values
    T = len(X)
    sr = _sharpe(X)
    diff = (sr[0] - sr[1]) * np.sqrt(TRADING_DAYS)
    rho = np.corrcoef(X.T)[0, 1]
    var_jk = (2 * (1 - rho) + 0.5 * (sr[0] ** 2 + sr[1] ** 2 - 2 * sr[0] * sr[1] * rho ** 2)) / T
    p_jk = 2 * (1 - norm.cdf(abs(sr[0] - sr[1]) / np.sqrt(var_jk)))
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(T / block))
    d_obs = sr[0] - sr[1]
    boots = np.empty(B)
    for i in range(B):
        st = rng.integers(0, T, nb)
        idx = ((st[:, None] + np.arange(block)[None, :]) % T).ravel()[:T]
        s = _sharpe(X[idx])
        boots[i] = s[0] - s[1]
    p_bs = float(np.mean(np.abs(boots - d_obs) >= abs(d_obs)))
    return dict(SharpeDiff_ann=diff, p_JK_Memmel=p_jk, p_block_bootstrap=p_bs)


def sharpe_tests_vs(results, ref, rf=None, B=2000):
    rows = {}
    for n, r in results.items():
        if n == ref or n.startswith("Random-k #"):
            continue
        rows[n] = sharpe_test(results[ref].returns, r.returns, rf, B=B)
    return pd.DataFrame(rows).T


def selection_frequency(result):
    """How often each asset was held across rebalances."""
    cnt = {}
    for h in result.holdings.values():
        for t in h:
            cnt[t] = cnt.get(t, 0) + 1
    return pd.Series(cnt).sort_values(ascending=False) / max(1, len(result.holdings))


# %% [sensitivity]
def run_sensitivity(returns, market, base_cfg, grid, rf=None, universe_fn=None, weights="ew"):
    """One-at-a-time sensitivity of SR-DAS: vary each parameter in `grid` around base_cfg.
    grid example: {"theta": [0, 0.25, 0.5, 1, 2], "agg_q": [0.5, 0.75, 0.9]}"""
    from dataclasses import replace
    rows = []
    for param, values in grid.items():
        for v in values:
            cfg = replace(base_cfg, **{param: v})
            res = run_backtest(returns, market, [Strategy("SR-DAS", m_srdas, weights, "Proposed")],
                               cfg, universe_fn, verbose=False)
            m = summary_table(res, rf, market=market).iloc[0]
            rows.append(dict(param=param, value=v, **m.drop("Group").to_dict()))
            print(f"  {param}={v}: Sharpe={m['Sharpe']:.3f}  MaxDD={m['MaxDD']:.2%}")
    return pd.DataFrame(rows)


# %% [plots]
import matplotlib.pyplot as plt

PALETTE = {"Proposed": "#1f4e9c", "Proposed-2": "#2a9d8f", "Original": "#e9a23b", "Ablation": "#8fb0de", "Baseline": "#d1495b",
           "Random": "#9a9a9a", "Benchmark": "#2b2b2b"}


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)


def plot_wealth(results, names, title="Cumulative wealth (log scale, net of costs)"):
    fig, ax = plt.subplots(figsize=(11, 5))
    for n in names:
        r = results[n]
        w = (1 + r.returns).cumprod()
        lw = 2.4 if r.group == "Proposed" else 1.2
        ax.plot(w.index, w.values, lw=lw, label=n,
                color=PALETTE.get(r.group) if r.group in ("Proposed", "Benchmark") else None)
    ax.set_yscale("log"); ax.set_title(title, loc="left"); ax.legend(fontsize=8, ncol=2, frameon=False)
    _style(ax); plt.tight_layout(); plt.show()


def plot_drawdowns(results, names, title="Drawdowns"):
    fig, ax = plt.subplots(figsize=(11, 4))
    for n in names:
        r = results[n]
        w = (1 + r.returns).cumprod()
        ax.plot(w.index, (w / w.cummax() - 1).values, lw=2 if r.group == "Proposed" else 1, label=n)
    ax.set_title(title, loc="left"); ax.legend(fontsize=8, ncol=2, frameon=False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    _style(ax); plt.tight_layout(); plt.show()


def plot_metric_bars(table, metric, title=None, higher_is_better=True):
    t = table[metric].astype(float).sort_values(ascending=not higher_is_better)
    colors = [PALETTE.get(g, "#888") for g in table.loc[t.index, "Group"]]
    fig, ax = plt.subplots(figsize=(9, 0.35 * len(t) + 1))
    ax.barh(t.index[::-1], t.values[::-1], color=colors[::-1])
    ax.set_title(title or metric, loc="left"); _style(ax); ax.grid(axis="x", alpha=0.3)
    plt.tight_layout(); plt.show()


def plot_sensitivity(sens, metrics=("Sharpe", "MaxDD", "StressCorr_OOS")):
    params = sens["param"].unique()
    fig, axes = plt.subplots(len(metrics), len(params), figsize=(3.2 * len(params), 2.6 * len(metrics)), squeeze=False)
    for j, p in enumerate(params):
        d = sens[sens["param"] == p]
        for i, m in enumerate(metrics):
            ax = axes[i, j]
            ax.plot(d["value"].astype(str), d[m].astype(float), marker="o", color=PALETTE["Proposed"])
            if i == 0: ax.set_title(p)
            if j == 0: ax.set_ylabel(m)
            _style(ax)
    plt.tight_layout(); plt.show()
