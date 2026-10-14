# %% [posthoc]
from scipy.stats import wilcoxon as _wilcoxon

ALL_CRISES = {
    "1973-74 bear": ("1973-01-11", "1974-10-03"),
    "1980 Q1": ("1980-02-13", "1980-03-27"),
    "1981-82 recession": ("1980-11-28", "1982-08-12"),
    "1987 crash": ("1987-08-25", "1987-12-04"),
    "Gulf war 1990": ("1990-07-16", "1990-10-11"),
    "LTCM 1998": ("1998-07-17", "1998-10-08"),
    "Dot-com 2000-02": ("2000-03-24", "2002-10-09"),
    "GFC 2007-09": ("2007-10-09", "2009-03-09"),
    "Euro debt 2011": ("2011-04-29", "2011-10-03"),
    "China/oil 2015-16": ("2015-05-21", "2016-02-11"),
    "Q4 2018": ("2018-09-20", "2018-12-24"),
    "COVID 2020": ("2020-02-19", "2020-03-23"),
    "Rates 2022": ("2022-01-03", "2022-10-12"),
    "Tariffs 2025": ("2025-02-19", "2025-04-08"),
}


# ---------- A. the truncation bias ----------
def simulate_truncation_bias(rhos=(0.3, 0.5, 0.7), tails=(0.05, 0.10, 0.20, 0.35, 0.5, 1.0),
                             N=20, T=200_000, seed=0):
    """One-factor model with CONSTANT pairwise correlation rho (no regime change).
    Measures the average pairwise correlation on days when the equal-weighted market
    is in its bottom `tail` fraction. Any gap to rho is pure selection (truncation) bias."""
    rng = np.random.default_rng(seed)
    m = rng.standard_normal(T)
    rows = []
    for rho in rhos:
        e = rng.standard_normal((T, N))
        R = np.sqrt(rho) * m[:, None] + np.sqrt(1 - rho) * e
        mk = R.mean(axis=1)
        for q in tails:
            sel = mk <= np.quantile(mk, q) if q < 1 else np.ones(T, bool)
            rows.append(dict(true_rho=rho, tail=q, measured=avg_pairwise_corr(R[sel])))
    return pd.DataFrame(rows)


def fig_bias(sim):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for (rho, d), col, mk in zip(sim.groupby("true_rho"), [C_AQUA, C_BLUE, C_VIOLET], ["s", "o", "D"]):
        ax.plot(d["tail"], d["measured"], "-", marker=mk, color=col, ms=5, label=f"true correlation {rho:.1f}")
        ax.axhline(rho, color=col, lw=0.8, ls=":")
    ax.set_xlabel("share of days kept (market return in its bottom tail)")
    ax.set_ylabel("measured average pairwise correlation")
    ax.set_xticks(sorted(sim["tail"].unique()))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("Selecting days on the market's own return biases correlation down", loc="left")
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig


# ---------- B. crisis-window correlations (no selection on daily returns) ----------
def forbes_rigobon(rho, delta):
    """Forbes & Rigobon (2002) heteroskedasticity adjustment: rho* = rho / sqrt(1 + delta (1 - rho^2)),
    delta = relative increase in market variance in the crisis window vs the calm window."""
    return rho / np.sqrt(1 + delta * (1 - rho ** 2))


def crisis_correlation_table(ind, mkt, crises, calm_days=252, gap=21):
    """For each crisis window: average pairwise industry correlation in the window (contiguous
    days, no selection on returns) vs the calm year before it, raw and Forbes-Rigobon adjusted."""
    rows = []
    for name, (a, b) in crises.items():
        x = ind.loc[a:b]
        i0 = ind.index.searchsorted(pd.Timestamp(a))
        pre = ind.iloc[max(0, i0 - gap - calm_days): max(0, i0 - gap)]
        if len(x) < 15 or len(pre) < 100:
            continue
        cols = x.columns[(x.notna().mean() > 0.95) & (pre.notna().mean() > 0.95)]
        Cc = x[cols].corr().values
        Cp = pre[cols].corr().values
        n = len(cols)
        off = ~np.eye(n, dtype=bool)
        v_crisis = mkt.loc[x.index].var()
        v_calm = mkt.loc[pre.index].var()
        delta = v_crisis / v_calm - 1
        adj = forbes_rigobon(Cc[off], delta) if delta > 0 else Cc[off]
        rows.append({"Crisis": name, "Days": len(x), "Calm-year corr": Cp[off].mean(),
                     "Crisis corr (raw)": Cc[off].mean(), "Crisis corr (FR-adjusted)": adj.mean(),
                     "Market var ratio": v_crisis / v_calm})
    return pd.DataFrame(rows).set_index("Crisis")


def fig_motivation_v2(ind, mkt, ctab, start="1965-01-01"):
    x, m = ind.loc[start:], mkt.loc[start:]
    pts = list(range(126, len(x), 21))
    roll = pd.Series([avg_corr_matrix(x.iloc[i - 126:i].dropna(axis=1, thresh=100)) for i in pts], index=x.index[pts])
    w = (1 + m).cumprod()
    dd = w / w.cummax() - 1
    fig = plt.figure(figsize=(10.5, 5.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.0, 1.15], height_ratios=[1, 0.7], hspace=0.38, wspace=0.42)
    a1 = fig.add_subplot(gs[0, 0]); a2 = fig.add_subplot(gs[1, 0], sharex=a1); a3 = fig.add_subplot(gs[:, 1])
    a1.plot(roll.index, roll.values, color=C_BLUE, lw=1.2)
    for _, (s, e) in ALL_CRISES.items():
        a1.axvspan(pd.Timestamp(s), pd.Timestamp(e), color=C_GRAY, alpha=0.15, lw=0)
    a1.set_title("(a) Rolling 6-month average pairwise correlation (shaded: crisis windows)", loc="left")
    a2.fill_between(dd.index, dd.values, 0, color=C_GRAY, alpha=0.6, lw=0)
    a2.set_title("(b) Market drawdown", loc="left")
    a2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    y = np.arange(len(ctab))
    for i, (_, r) in enumerate(ctab.iterrows()):
        a3.plot([r["Calm-year corr"], r["Crisis corr (raw)"]], [i, i], color=C_GRID, lw=2, zorder=1)
    a3.scatter(ctab["Calm-year corr"], y, color=C_AQUA, marker="s", s=30, label="calm year before", zorder=3)
    a3.scatter(ctab["Crisis corr (FR-adjusted)"], y, facecolor="white", edgecolor=C_ORANGE, marker="D", s=30,
               label="crisis, FR-adjusted", zorder=3)
    a3.scatter(ctab["Crisis corr (raw)"], y, color=C_ORANGE, marker="D", s=30, label="crisis window", zorder=4)
    a3.set_yticks(y); a3.set_yticklabels(ctab.index); a3.invert_yaxis()
    a3.set_xlabel("average pairwise correlation")
    a3.set_title("(c) Crisis windows vs calm year before", loc="left")
    a3.legend(loc="upper center", bbox_to_anchor=(0.45, -0.09), ncol=2, fontsize=7.5)
    return fig


# ---------- C. post-hoc H1: basket co-movement inside crisis windows ----------
def crisis_mask(index, crises):
    mask = pd.Series(False, index=index)
    for a, b in crises.values():
        mask.loc[a:b] = True
    return mask


def crisis_basket_corr(result, returns, cmask, min_days=10):
    """Average pairwise correlation of the held basket on crisis-window days, per holding period
    (time-defined crises: no selection on daily returns). Returns DataFrame indexed by rebalance date."""
    dates = sorted(result.holdings)
    idx = returns.index
    rows = []
    for j, d in enumerate(dates):
        end = dates[j + 1] if j + 1 < len(dates) else idx[-1]
        seg = returns.loc[(idx > d) & (idx <= end)]
        seg = seg[cmask.reindex(seg.index).fillna(False).values]
        if len(seg) < min_days:
            continue
        tick = list(result.holdings[d])
        X = seg[tick].fillna(0.0).values
        rows.append(dict(date=d, days=len(seg), corr=avg_pairwise_corr(X)))
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame(columns=["days", "corr"])


def posthoc_h1_table(res, returns, crises, ref="B-SR-DAS (balanced)", others=None):
    cmask = crisis_mask(returns.index, crises)
    per = {n: crisis_basket_corr(res[n], returns, cmask) for n in [ref] + list(others)}
    rows = {}
    for n, d in per.items():
        w = np.average(d["corr"], weights=d["days"]) if len(d) else np.nan
        row = {"Crisis-window basket corr (day-weighted)": w, "Holding periods": len(d)}
        if n != ref and len(d):
            j = per[ref].join(d, lsuffix="_ref", rsuffix="_oth", how="inner")
            diff = (j["corr_ref"] - j["corr_oth"]).dropna()
            if len(diff) >= 6 and not np.allclose(diff, 0):
                row.update({"Mean diff (ref - this)": diff.mean(), "Ref lower in": (diff < 0).mean(),
                            "p one-sided (ref lower)": _wilcoxon(diff, alternative="less").pvalue,
                            "p two-sided": _wilcoxon(diff).pvalue})
        rows[n] = row
    return pd.DataFrame(rows).T, per


# ---------- D. frontier with non-overlapping labels ----------
def _repel(ax, texts):
    try:
        from adjustText import adjust_text
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color=C_GRAY, lw=0.5),
                    expand=(1.15, 1.4), only_move={"text": "xy"})
    except Exception as e:
        print("  (label repel unavailable:", e, ")")


def fig_frontiers_v2(fronts, tabs):
    groups_color = {"Proposed": C_BLUE, "Proposed-2": C_VIOLET, "Original": C_ORANGE,
                    "Baseline": C_AQUA, "Benchmark": C_GRAY, "Random": C_GRAY}
    keep = ["B-SR-DAS (balanced)", "S-SR-DAS (screen+diversify)", "Top-k Quality", "TailDep-Cluster",
            "Corr-Cluster", "MST-Periphery", "PCA-KMeans (Vo et al.)", "EW-All (1/N)", "HRP-All"]
    short = {"B-SR-DAS (balanced)": "B-SR-DAS", "S-SR-DAS (screen+diversify)": "S-SR-DAS",
             "PCA-KMeans (Vo et al.)": "PCA-KMeans", "EW-All (1/N)": "1/N"}
    fig, axes = plt.subplots(len(fronts), 2, figsize=(10.5, 4.4 * len(fronts)), squeeze=False)
    for i, (label, front) in enumerate(fronts.items()):
        tab = tabs[label]
        for j, ycol in enumerate(["Vol", "MaxDD"]):
            ax = axes[i, j]
            sgn = 1 if ycol == "Vol" else -1
            f = front.sort_values("theta")
            ax.plot(f["StressCorr_OOS"].astype(float), sgn * f[ycol].astype(float), "-o", color=C_BLUE,
                    ms=3.5, lw=1.2, alpha=0.8)
            texts = []
            for _, r in f.iterrows():
                if r["theta"] in (0, 0.1, 0.5, 1.0):
                    texts.append(ax.text(float(r["StressCorr_OOS"]), sgn * float(r[ycol]), f"θ={r['theta']:g}",
                                         fontsize=6.5, color=C_INK2))
            for n in keep:
                if n not in tab.index:
                    continue
                r = tab.loc[n]
                xv, yv = float(r["StressCorr_OOS"]), sgn * float(r[ycol])
                prop = r["Group"] in ("Proposed", "Proposed-2")
                col = C_ORANGE if n == "Top-k Quality" else groups_color.get(r["Group"], C_GRAY)
                ax.scatter(xv, yv, s=90 if prop else 28, marker="*" if prop else "o", color=col,
                           edgecolor="white", linewidth=0.8, zorder=4)
                texts.append(ax.text(xv, yv, short.get(n, n), fontsize=7, color=C_INK))
            _repel(ax, texts)
            ax.set_xlabel("stress-day co-movement of held basket, out of sample (lower = better)")
            ax.set_ylabel("annualized volatility" if ycol == "Vol" else "maximum drawdown (size)")
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
            ax.set_title(f"({'abcd'[2 * i + j]}) {label}: co-movement vs {'volatility' if ycol == 'Vol' else 'drawdown'}",
                         loc="left")
    fig.text(0.01, -0.01, "Line: SR-DAS with fixed θ. Stars: proposed methods. Blue = B-SR-DAS and θ line, violet = S-SR-DAS, "
             "orange = Top-k Quality, aqua = other selection baselines, grey = full-universe benchmarks.",
             fontsize=7.5, color=C_INK2)
    plt.tight_layout()
    return fig
