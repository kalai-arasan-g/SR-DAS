# %% [revised]
from scipy.linalg import pinvh
from scipy.stats import wilcoxon

THETA_LOG = {}   # rebalance date -> balanced theta chosen by the primary method (for reporting)


def uniqueness_scores(K):
    """Diversity contribution of each asset: log of its conditional variance given all
    other assets under kernel K, i.e. log det(K) - log det(K without i) = -log (K^-1)_ii.
    These are the canonical per-asset diversity scores of the log-det objective."""
    Kinv = pinvh(K)
    return -np.log(np.clip(np.diag(Kinv), 1e-12, None))


def balanced_theta(K, z, beta=1.0):
    """Scale rule that fixes the quality/diversity imbalance of the original SR-DAS.

    The objective is  2*theta*sum(z_S) + log det K_S. theta is set at every rebalance so
    that the cross-sectional spread of the quality term (2*theta*z_i) equals the spread
    of the diversity scores (uniqueness_scores). beta = 1 gives both terms equal weight.
    This is a fixed rule — nothing is tuned on returns."""
    g = uniqueness_scores(K)
    return beta * np.std(g) / (2.0 * np.std(z) + 1e-12)


def make_balanced(corr_attr, log=False):
    """Balanced SR-DAS on a given dependence estimate (attribute name of Context)."""
    def m(ctx):
        C = getattr(ctx, corr_attr)
        K = affine_corr_kernel(C, ctx.cfg.kernel_a)
        th = balanced_theta(K, ctx.z)
        if log and ctx.date is not None:
            THETA_LOG[ctx.date] = th
        return logdet_select(C, ctx.z, ctx.cfg.k, th, ctx.cfg.kernel_a)
    return m


def make_screen(corr_attr, frac=0.5):
    """Screen-then-diversify: keep the top `frac` of assets by quality, then pure
    log-det diversity (theta = 0) among them on the given dependence estimate."""
    def m(ctx):
        C = getattr(ctx, corr_attr)
        n = len(ctx.z)
        keep = np.argsort(-ctx.z)[: max(ctx.cfg.k, int(np.ceil(frac * n)))]
        keep = np.sort(keep)
        sub = logdet_select(C[np.ix_(keep, keep)], ctx.z[keep], ctx.cfg.k, 0.0, ctx.cfg.kernel_a)
        return [int(keep[i]) for i in sub]
    return m


def make_theta(theta):
    """SR-DAS with a fixed theta (used for the trade-off frontier)."""
    def m(ctx):
        return logdet_select(ctx.C_robust, ctx.z, ctx.cfg.k, theta, ctx.cfg.kernel_a)
    return m


def v3_strategies(cfg, include_random=True, weights="ew"):
    """Pre-specified comparison set for the revised study."""
    S = [
        Strategy("B-SR-DAS (balanced)", make_balanced("C_robust", log=True), weights, "Proposed"),
        Strategy("S-SR-DAS (screen+diversify)", make_screen("C_robust"), weights, "Proposed-2"),
        # ablations of the PRIMARY method: one ingredient changed each
        Strategy("Abl: B-DAS all-days (stress)", make_balanced("C_plain_robust"), weights, "Ablation"),
        Strategy("Abl: B-DAS stress 1-window (robust)", make_balanced("C_stress_single"), weights, "Ablation"),
        Strategy("Abl: Stress-Cluster (rule)", m_stress_cluster, weights, "Ablation"),
        # the original specification, for reference
        Strategy("Original SR-DAS (theta=0.5)", m_srdas, weights, "Original"),
        Strategy("Original SR-DAS (theta=0)", m_srdas_noq, weights, "Original"),
        # literature baselines and benchmarks
        Strategy("TailDep-Cluster", m_taildep_cluster, weights, "Baseline"),
        Strategy("Corr-Cluster", m_corr_cluster, weights, "Baseline"),
        Strategy("MST-Periphery", m_mst_periphery, weights, "Baseline"),
        Strategy("PCA-KMeans (Vo et al.)", m_pca_kmeans, weights, "Baseline"),
        Strategy("Top-k Quality", m_topk_quality, weights, "Baseline"),
        Strategy("EW-All (1/N)", m_all, "ew", "Benchmark"),
        Strategy("HRP-All", m_all, "hrp", "Benchmark"),
    ]
    if include_random:
        S += [Strategy(f"Random-k #{s}", make_random(s), weights, "Random") for s in range(cfg.n_random)]
    return S


# ---------- statistical helpers for the pre-specified hypotheses ----------
def paired_period_test(res_a, res_b, col="stress_corr_oos", alternative="less"):
    """Paired test across holding periods of a per-period diagnostic (e.g. the held
    basket's stress-day correlation). Wilcoxon signed-rank on period-by-period
    differences a - b. alternative='less' tests whether a is systematically lower."""
    d = (res_a.diag[col] - res_b.diag[col]).dropna()
    if len(d) < 6 or np.allclose(d, 0):
        return dict(n_periods=len(d), mean_diff=float(d.mean()) if len(d) else np.nan,
                    share_a_lower=float((d < 0).mean()) if len(d) else np.nan, p_value=np.nan)
    p = wilcoxon(d, alternative=alternative).pvalue
    return dict(n_periods=len(d), mean_diff=float(d.mean()), share_a_lower=float((d < 0).mean()), p_value=float(p))


def ratio_ci(a, b, stat="vol", block=21, B=2000, seed=0, q=0.10, market=None):
    """Block-bootstrap 95% CI for a risk ratio of two daily return series:
    stat='vol' -> vol(a)/vol(b);  stat='cvar' -> CVaR5(a)/CVaR5(b)."""
    X = pd.concat([a, b], axis=1, sort=True).dropna().values
    T = len(X)

    def f(Y):
        if stat == "vol":
            s = Y.std(axis=0, ddof=1)
            return s[0] / s[1]
        lo = [Y[Y[:, j] <= np.quantile(Y[:, j], 0.05), j].mean() for j in range(2)]
        return lo[0] / lo[1]

    obs = f(X)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(T / block))
    bs = np.empty(B)
    for i in range(B):
        idx = ((rng.integers(0, T, nb)[:, None] + np.arange(block)[None, :]) % T).ravel()[:T]
        bs[i] = f(X[idx])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(ratio=float(obs), ci_low=float(lo), ci_high=float(hi))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (controls family-wise error)."""
    p = pd.Series(pvals, dtype=float)
    ok = p.dropna().sort_values()
    m = len(ok)
    adj = pd.Series(np.nan, index=p.index)
    running = 0.0
    for i, (k, v) in enumerate(ok.items()):
        running = max(running, min(1.0, (m - i) * v))
        adj[k] = running
    return adj


def sharpe_tests_holm(results, ref, rf=None, B=2000):
    t = sharpe_tests_vs(results, ref, rf, B=B)
    t["p_holm (bootstrap)"] = holm(t["p_block_bootstrap"])
    return t


def evaluate_hypotheses(res, primary, rf, B=2000, core_only=False):
    """Evaluate the pre-registered hypotheses for one method on one sample.
    H1-H4 apply to any method; H5-H7 (ablations) apply to the primary method only
    (core_only=True skips them). Returns: hypothesis, statistic, p-value / CI, verdict."""
    rows = []

    def add(h, desc, stat, detail, ok):
        rows.append(dict(Hypothesis=h, Description=desc, Statistic=stat, Detail=detail,
                         Verdict="SUPPORTED" if ok else "NOT supported"))

    P = res[primary]
    # H1: lower stress-day correlation of the held basket than Top-k Quality
    t = paired_period_test(P, res["Top-k Quality"])
    add("H1", "Stress-day basket correlation lower than Top-k Quality (paired, one-sided)",
        f"mean diff {t['mean_diff']:+.3f}", f"p = {t['p_value']:.4f}, lower in {t['share_a_lower']:.0%} of periods",
        t["p_value"] < 0.05)
    # H2: lower volatility than 1/N
    c = ratio_ci(P.returns, res["EW-All (1/N)"].returns, "vol", B=B)
    add("H2", "Volatility lower than EW-All (1/N)", f"ratio {c['ratio']:.3f}",
        f"95% CI [{c['ci_low']:.3f}, {c['ci_high']:.3f}]", c["ci_high"] < 1)
    # H3: lower volatility than pure diversity (theta = 0) -> the balance works
    c = ratio_ci(P.returns, res["Original SR-DAS (theta=0)"].returns, "vol", B=B)
    add("H3", "Volatility lower than pure-diversity SR-DAS (theta=0)", f"ratio {c['ratio']:.3f}",
        f"95% CI [{c['ci_low']:.3f}, {c['ci_high']:.3f}]", c["ci_high"] < 1)
    # H4: no significant Sharpe under-performance vs 1/N
    s = sharpe_test(P.returns, res["EW-All (1/N)"].returns, rf, B=B)
    add("H4", "Sharpe not significantly below EW-All (1/N) (two-sided, 5%)",
        f"diff {s['SharpeDiff_ann']:+.3f}", f"p = {s['p_block_bootstrap']:.4f}",
        not (s["SharpeDiff_ann"] < 0 and s["p_block_bootstrap"] < 0.05))
    if core_only:
        return pd.DataFrame(rows).set_index("Hypothesis")
    # H5: lower stress-day correlation than cluster-then-pick on the same matrix (RQ1)
    t = paired_period_test(P, res["Abl: Stress-Cluster (rule)"])
    add("H5", "Stress-day basket correlation lower than Stress-Cluster rule (RQ1, one-sided)",
        f"mean diff {t['mean_diff']:+.3f}", f"p = {t['p_value']:.4f}, lower in {t['share_a_lower']:.0%} of periods",
        t["p_value"] < 0.05)
    # H6: stress conditioning lowers realized stress correlation (RQ2)
    t = paired_period_test(P, res["Abl: B-DAS all-days (stress)"])
    add("H6", "Stress conditioning lowers stress-day basket correlation vs all-day correlation (RQ2)",
        f"mean diff {t['mean_diff']:+.3f}", f"p = {t['p_value']:.4f}, lower in {t['share_a_lower']:.0%} of periods",
        t["p_value"] < 0.05)
    # H7: robust aggregation lowers realized stress correlation (RQ3)
    t = paired_period_test(P, res["Abl: B-DAS stress 1-window (robust)"])
    add("H7", "Robust aggregation lowers stress-day basket correlation vs single window (RQ3)",
        f"mean diff {t['mean_diff']:+.3f}", f"p = {t['p_value']:.4f}, lower in {t['share_a_lower']:.0%} of periods",
        t["p_value"] < 0.05)
    return pd.DataFrame(rows).set_index("Hypothesis")


def run_frontier(returns, market, cfg, thetas, rf=None):
    """SR-DAS for a grid of fixed theta values: maps the quality-diversity trade-off."""
    S = [Strategy(f"theta={t:g}", make_theta(t), "ew", "Proposed") for t in thetas]
    r = run_backtest(returns, market, S, cfg, verbose=False)
    tab = summary_table(r, rf, market=market)
    tab["theta"] = thetas
    return tab


def plot_frontier(frontier, points, x="StressCorr_OOS", y="Vol", title=None):
    """Frontier of fixed-theta SR-DAS (line) with other strategies as labelled points."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    f = frontier.sort_values("theta")
    ax.plot(f[x].astype(float), f[y].astype(float), "-o", color=PALETTE["Proposed"], lw=1.5, ms=4,
            label="SR-DAS, fixed theta")
    for _, r in f.iterrows():
        ax.annotate(f"θ={r['theta']:g}", (float(r[x]), float(r[y])), fontsize=7, xytext=(4, 2),
                    textcoords="offset points", color=PALETTE["Proposed"])
    for name, r in points.iterrows():
        g = r["Group"]
        ax.scatter(float(r[x]), float(r[y]), s=60 if g in ("Proposed", "Proposed-2") else 30,
                   color=PALETTE.get(g, "#888"), zorder=3,
                   marker="*" if g in ("Proposed", "Proposed-2") else "o")
        ax.annotate(name, (float(r[x]), float(r[y])), fontsize=7, xytext=(4, -9), textcoords="offset points")
    ax.set_xlabel(f"{x} (lower = more diversified in stress)")
    ax.set_ylabel(f"{y} (lower = better)")
    if y in ("Vol",):
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title(title or f"Trade-off: {x} vs {y}", loc="left")
    _style(ax); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.show()
