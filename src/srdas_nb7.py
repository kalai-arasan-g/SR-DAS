# %% [nb7]
from scipy.stats import t as _tdist, wilcoxon as _wilc7


# ---------- crisis windows defined by a rule (for the new early sample) ----------
def drawdown_episodes(mkt, start, end, threshold=0.15, min_days=15):
    """Market drawdown episodes inside [start, end]: from a running peak to the lowest point
    reached before the market regains that peak, kept if the fall is at least `threshold`.
    Only the market index is used, so the windows do not depend on any strategy."""
    m = mkt.loc[start:end].dropna()
    w = (1 + m).cumprod()
    out, peak_val, peak_day = {}, w.iloc[0], w.index[0]
    trough_val, trough_day = peak_val, peak_day
    for d, v in w.items():
        if v >= peak_val:
            if trough_val / peak_val - 1 <= -threshold:
                out[peak_day, trough_day] = trough_val / peak_val - 1
            peak_val, peak_day, trough_val, trough_day = v, d, v, d
        elif v < trough_val:
            trough_val, trough_day = v, d
    if trough_val / peak_val - 1 <= -threshold:
        out[peak_day, trough_day] = trough_val / peak_val - 1
    eps = {}
    for (a, b), dd in out.items():
        if len(m.loc[a:b]) >= min_days:
            eps[f"{a.year}-{b.year} ({dd:.0%})" if a.year != b.year else f"{a.year} ({dd:.0%})"] = (str(a.date()), str(b.date()))
    return eps


# ---------- inference that allows for dependence across holding periods ----------
def nw_lag(n):
    """Newey-West (1994) automatic lag rule."""
    return int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))


def hac_mean_test(d, lag=None):
    """Mean of d with a Newey-West (Bartlett) standard error. Returns mean, se, 95% CI,
    one-sided p for mean < 0, the lag and the first-order autocorrelation of d."""
    d = np.asarray(d, float)
    n = len(d)
    lag = nw_lag(n) if lag is None else lag
    e = d - d.mean()
    s = e @ e / n
    for L in range(1, lag + 1):
        s += 2 * (1 - L / (lag + 1)) * (e[L:] @ e[:-L]) / n
    se = np.sqrt(max(s, 1e-18) / n)
    tval = d.mean() / se
    q = _tdist.ppf(0.975, n - 1)
    ar1 = np.corrcoef(d[1:], d[:-1])[0, 1] if n > 3 else np.nan
    return dict(mean=d.mean(), se=se, ci_low=d.mean() - q * se, ci_high=d.mean() + q * se,
                p_less=float(_tdist.cdf(tval, n - 1)), lag=lag, ar1=ar1)


def mbb_mean_p(d, block=4, B=5000, seed=0):
    """Moving-block bootstrap of holding periods (block = number of consecutive quarters).
    One-sided p-value for mean(d) < 0, from the bootstrap distribution centred at the
    observed mean, and a 95% percentile interval for the mean."""
    d = np.asarray(d, float)
    n = len(d)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(B, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(B, -1)[:, :n]
    bm = d[idx].mean(axis=1)
    obs = d.mean()
    p = float(np.mean(bm - obs <= obs))          # H0 mean = 0 against mean < 0
    lo, hi = np.percentile(bm, [2.5, 97.5])
    return p, lo, hi


def period_diffs(res_a, res_b, col="stress_corr_oos"):
    return (res_a.diag[col] - res_b.diag[col]).dropna()


def robust_period_table(res, primary, comparisons, col="stress_corr_oos"):
    """For each comparison: the pre-specified Wilcoxon test and three tests that allow for
    dependence across quarters (Newey-West t-test, moving-block bootstrap with 4 and 8 quarters)."""
    rows = {}
    for h, other in comparisons.items():
        if other not in res:
            continue
        d = period_diffs(res[primary], res[other], col)
        if len(d) < 8:
            continue
        hac = hac_mean_test(d.values)
        p4, lo4, hi4 = mbb_mean_p(d.values, 4)
        p8, _, _ = mbb_mean_p(d.values, 8)
        rows[f"{h}: vs {other}"] = {
            "Periods": len(d), "Mean diff": hac["mean"], "Lower in (share)": float((d < 0).mean()),
            "AR(1) of diff": hac["ar1"], "Wilcoxon p (pre-specified)": float(_wilc7(d, alternative="less").pvalue),
            "NW lag": hac["lag"], "NW 95% CI low": hac["ci_low"], "NW 95% CI high": hac["ci_high"],
            "NW p (one-sided)": hac["p_less"], "MBB-4q p": p4, "MBB-4q CI low": lo4, "MBB-4q CI high": hi4,
            "MBB-8q p": p8}
    return pd.DataFrame(rows).T


# ---------- sensitivity of the daily-return bootstrap to the block length ----------
def sharpe_diff_boot(a, b, rf, block=21, B=2000, seed=0):
    """Annualized Sharpe difference a - b, 95% percentile interval and two-sided p-value
    (circular block bootstrap, distribution centred at the observed difference)."""
    df = pd.concat([a, b], axis=1, sort=True).dropna()
    X = df.values - rf.reindex(df.index).fillna(0.0).values[:, None]
    T = len(X)
    sr = _sharpe(X)
    obs = sr[0] - sr[1]
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(T / block))
    bs = np.empty(B)
    for i in range(B):
        idx = ((rng.integers(0, T, nb)[:, None] + np.arange(block)[None, :]) % T).ravel()[:T]
        s = _sharpe(X[idx])
        bs[i] = s[0] - s[1]
    ann = np.sqrt(TRADING_DAYS)
    lo, hi = np.percentile(bs, [2.5, 97.5]) * ann
    return dict(diff=obs * ann, ci_low=lo, ci_high=hi, p=float(np.mean(np.abs(bs - obs) >= abs(obs))))


def block_sensitivity(res, primary, rf, blocks=(5, 10, 21, 42, 63, 126), B=2000,
                      pairs=(("H2", "vol", "EW-All (1/N)"), ("H3", "vol", "Original SR-DAS (theta=0)"),
                             ("H4", "sharpe", "EW-All (1/N)"), ("Sharpe vs top-k", "sharpe", "Top-k Quality"))):
    rows = []
    for blk in blocks:
        for h, kind, other in pairs:
            if other not in res:
                continue
            if kind == "vol":
                c = ratio_ci(res[primary].returns, res[other].returns, "vol", block=blk, B=B)
                rows.append(dict(Block=blk, Test=h, Against=other, Estimate=c["ratio"], **{"CI low": c["ci_low"],
                            "CI high": c["ci_high"], "p (two-sided)": np.nan, "Verdict": "yes" if c["ci_high"] < 1 else "no"}))
            else:
                s = sharpe_diff_boot(res[primary].returns, res[other].returns, rf, block=blk, B=B)
                ok = not (s["diff"] < 0 and s["p"] < 0.05)
                rows.append(dict(Block=blk, Test=h, Against=other, Estimate=s["diff"], **{"CI low": s["ci_low"],
                            "CI high": s["ci_high"], "p (two-sided)": s["p"], "Verdict": "yes" if ok else "no"}))
    return pd.DataFrame(rows)


def save_period_diag(res, name, cols=("stress_corr_oos", "avg_corr_oos", "n_stress_days", "n_eligible")):
    """Save per-holding-period diagnostics of every strategy (long format) for later checks."""
    frames = []
    for n, r in res.items():
        if n.startswith("Random-k #"):
            continue
        d = r.diag[[c for c in cols if c in r.diag.columns]].copy()
        d["strategy"] = n
        frames.append(d)
    out = pd.concat(frames).reset_index()
    out.to_csv(os.path.join(TAB_DIR, f"{name}.csv"), index=False)
    return out
