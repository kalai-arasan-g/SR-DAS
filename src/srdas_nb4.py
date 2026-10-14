# %% [manuscript]
from dataclasses import replace as dc_replace
import matplotlib as mpl
from matplotlib.patches import FancyBboxPatch

# ---------- validated categorical palette (fixed order) + neutral ----------
C_BLUE, C_ORANGE, C_AQUA, C_VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
C_GRAY, C_INK, C_INK2, C_GRID = "#8a8985", "#0b0b0b", "#52514e", "#e4e3df"
# entity -> (color, linestyle); color follows the strategy, never its rank
SERIES_STYLE = {
    "B-SR-DAS (balanced)": (C_BLUE, "-"),
    "Top-k Quality": (C_ORANGE, "--"),
    "TailDep-Cluster": (C_AQUA, "-."),
    "S-SR-DAS (screen+diversify)": (C_VIOLET, ":"),
    "EW-All (1/N)": (C_GRAY, "-"),
}


def pub_style():
    """Journal-style matplotlib defaults: thin marks, recessive grid, readable type."""
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "font.size": 9, "axes.titlesize": 10,
        "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": C_INK2,
        "axes.labelcolor": C_INK, "xtick.color": C_INK2, "ytick.color": C_INK2,
        "axes.grid": True, "grid.color": C_GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.6,
        "legend.frameon": False, "font.family": "DejaVu Sans", "pdf.fonttype": 42,
    })


FIG_DIR = "outputs_manuscript/figures"
TAB_DIR = "outputs_manuscript/tables"


def save_fig(fig, name):
    """Save a figure as 300-dpi PNG and vector PDF."""
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"), bbox_inches="tight")
    plt.show()


def export_table(df, name, caption=None, float_fmt="{:.3f}"):
    """Save a table as CSV and LaTeX (booktabs), and register it for the Excel workbook."""
    os.makedirs(TAB_DIR, exist_ok=True)
    df.to_csv(os.path.join(TAB_DIR, f"{name}.csv"))
    try:
        tex = df.to_latex(float_format=lambda v: float_fmt.format(v), caption=caption,
                          label=f"tab:{name}", escape=True)
        with open(os.path.join(TAB_DIR, f"{name}.tex"), "w") as f:
            f.write(tex)
    except Exception as e:  # jinja2 missing etc.
        print("  (LaTeX export skipped:", e, ")")
    TABLES[name] = df
    return df


TABLES = {}


def write_workbook(path="outputs_manuscript/tables/all_tables.xlsx"):
    with pd.ExcelWriter(path) as xw:
        for name, df in TABLES.items():
            df.to_excel(xw, sheet_name=name[:31])
    print("Excel workbook:", path)


# ---------- analysis helpers ----------
def apply_costs(res, bps):
    """Re-price a backtest at a different one-way cost (bps) from its gross returns and
    turnover - identical to re-running the backtest with that cost."""
    out = {}
    for n, r in res.items():
        net = r.gross_returns.copy()
        idx = net.index
        for d, turn in r.turnover.items():
            i = idx.searchsorted(d, side="right")
            if i < len(idx):
                net.iloc[i] -= turn * bps / 1e4
        out[n] = dc_replace(r, returns=net)
    return out


def sharpe_ci(r, rf=None, block=21, B=2000, seed=0):
    """Annualized Sharpe ratio with a circular block-bootstrap 95% CI."""
    r = r.dropna()
    x = (r - rf.reindex(r.index).fillna(0.0)).values if rf is not None else r.values
    T = len(x)
    s = x.mean() / x.std(ddof=1) * np.sqrt(TRADING_DAYS)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(T / block))
    bs = np.empty(B)
    for i in range(B):
        idx = ((rng.integers(0, T, nb)[:, None] + np.arange(block)[None, :]) % T).ravel()[:T]
        y = x[idx]
        bs[i] = y.mean() / y.std(ddof=1) * np.sqrt(TRADING_DAYS)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return s, lo, hi


def sharpe_ci_table(res, names, rf, B=2000):
    rows = {}
    for n in names:
        s, lo, hi = sharpe_ci(res[n].returns, rf, B=B)
        rows[n] = dict(Sharpe=s, CI_low=lo, CI_high=hi)
    return pd.DataFrame(rows).T


def avg_corr_matrix(df):
    C = df.corr().values
    n = C.shape[0]
    return (C.sum() - n) / (n * (n - 1))


def descriptive_block(ind, mkt, start, end, label):
    x, m = ind.loc[start:end], mkt.loc[start:end]
    x = x.loc[:, x.notna().mean() > 0.95]
    st = m <= m.quantile(0.10)
    calm = m > m.quantile(0.50)
    ann_ret = (1 + x).prod() ** (TRADING_DAYS / len(x)) - 1
    ann_vol = x.std() * np.sqrt(TRADING_DAYS)
    return {
        "Sample": label, "Start": str(x.index[0].date()), "End": str(x.index[-1].date()),
        "Trading days": len(x), "Assets": x.shape[1],
        "Asset ann. return, median": ann_ret.median(), "Asset ann. vol, median": ann_vol.median(),
        "Asset ann. vol, min": ann_vol.min(), "Asset ann. vol, max": ann_vol.max(),
        "Market ann. return": (1 + m).prod() ** (TRADING_DAYS / len(m)) - 1,
        "Market ann. vol": m.std() * np.sqrt(TRADING_DAYS),
        "Avg pairwise corr, all days": avg_corr_matrix(x),
        "Avg pairwise corr, calm days": avg_corr_matrix(x[calm]),
        "Avg pairwise corr, stress days": avg_corr_matrix(x[st]),
    }


def robustness_rows(res, label, rf, B=1000):
    """Key metrics + H1-H4 verdicts of the primary method in one robustness configuration."""
    tab = summary_table(res, rf, market=MKT)
    hyp = evaluate_hypotheses(res, "B-SR-DAS (balanced)", rf, B=B, core_only=True)
    row = {"Configuration": label}
    for n, short in [("B-SR-DAS (balanced)", "B-SR-DAS"), ("Top-k Quality", "TopK"), ("EW-All (1/N)", "1/N")]:
        row[f"Sharpe {short}"] = tab.loc[n, "Sharpe"]
        row[f"Vol {short}"] = tab.loc[n, "Vol"]
        row[f"StressCorr {short}"] = tab.loc[n, "StressCorr_OOS"]
    row["MaxDD B-SR-DAS"] = tab.loc["B-SR-DAS (balanced)", "MaxDD"]
    for h in ["H1", "H2", "H3", "H4"]:
        row[h] = "yes" if hyp.loc[h, "Verdict"] == "SUPPORTED" else "no"
    return row


# ---------- figures ----------
def _label_end(ax, x, y, text, color):
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points", fontsize=7.5,
                color=C_INK, va="center")


def fig_framework():
    """Schematic of the B-SR-DAS pipeline."""
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.6); ax.axis("off")

    def box(x, y, w, h, title, body, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc="white", ec=color, lw=1.4))
        ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="top", fontsize=8.5, weight="bold", color=C_INK)
        ax.text(x + w / 2, y + h - 0.52, body, ha="center", va="top", fontsize=7.2, color=C_INK2, linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=C_INK2, lw=1))

    box(0.05, 2.0, 1.9, 1.45, "Look-back data", "5 years of daily\nreturns up to t\n(no look-ahead)", C_GRAY)
    box(2.35, 2.0, 2.3, 1.45, "Stress dependence", "market-stress days only,\noverlapping sub-windows,\npessimistic quantile,\nPSD projection → C", C_AQUA)
    box(2.35, 0.15, 2.3, 1.45, "Quality score z", "low volatility +\n12-1 momentum,\nrank-normalized", C_ORANGE)
    box(5.05, 1.1, 2.25, 1.45, "Balanced objective", "K = ½C + ½·11ᵀ\nmax 2θΣz + log det K_S\nθ: equal spread of quality\nand diversity terms", C_BLUE)
    box(7.7, 1.1, 2.2, 1.45, "Portfolio", "greedy log-det → k assets\nequal weights, held\none quarter, then\nroll forward", C_VIOLET)
    arrow(1.95, 2.72, 2.35, 2.72); arrow(1.95, 2.3, 2.35, 0.95)
    arrow(4.65, 2.6, 5.05, 2.05); arrow(4.65, 0.9, 5.05, 1.5); arrow(7.3, 1.82, 7.7, 1.82)
    ax.set_title("B-SR-DAS: stress-robust, balanced determinantal asset selection (walk-forward)", loc="left")
    return fig


def fig_motivation(ind, mkt, start="1965-01-01"):
    x, m = ind.loc[start:], mkt.loc[start:]
    pts = list(range(126, len(x), 21))
    roll = pd.Series([avg_corr_matrix(x.iloc[i - 126:i].dropna(axis=1, thresh=100)) for i in pts], index=x.index[pts])
    w = (1 + m).cumprod()
    dd = w / w.cummax() - 1
    fig = plt.figure(figsize=(10, 5.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.2, 1], height_ratios=[1, 0.7], hspace=0.35, wspace=0.25)
    a1 = fig.add_subplot(gs[0, 0]); a2 = fig.add_subplot(gs[1, 0], sharex=a1); a3 = fig.add_subplot(gs[:, 1])
    a1.plot(roll.index, roll.values, color=C_BLUE, lw=1.2)
    a1.set_title("(a) Rolling 6-month average pairwise correlation of the 49 industries", loc="left")
    a2.fill_between(dd.index, dd.values, 0, color=C_GRAY, alpha=0.6, lw=0)
    a2.set_title("(b) Market drawdown", loc="left")
    a2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    rows = []
    for a, b in [("1965", "1979"), ("1980", "1989"), ("1990", "2004"), ("2005", "2026")]:
        xx, mm = x.loc[a:b].dropna(axis=1, thresh=int(0.95 * len(x.loc[a:b]))), m.loc[a:b]
        rows.append((f"{a}–{b[-2:]}", avg_corr_matrix(xx[mm > mm.quantile(0.5)]),
                     avg_corr_matrix(xx), avg_corr_matrix(xx[mm <= mm.quantile(0.10)])))
    lab = [r[0] for r in rows]
    yy = np.arange(len(rows))
    for j, (name, col, mk) in enumerate([("calm days", C_AQUA, "s"), ("all days", C_GRAY, "o"),
                                          ("stress days", C_ORANGE, "D")]):
        a3.scatter([r[j + 1] for r in rows], yy, color=col, marker=mk, s=40, label=name, zorder=3)
    for i, r in enumerate(rows):
        a3.plot([r[1], r[3]], [i, i], color=C_GRID, lw=2, zorder=1)
    a3.set_yticks(yy); a3.set_yticklabels(lab); a3.invert_yaxis()
    a3.set_xlabel("average pairwise correlation")
    a3.set_title("(c) Correlation by market state", loc="left")
    a3.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=7.5, handletextpad=0.2, columnspacing=0.8)
    return fig, pd.DataFrame(rows, columns=["Period", "Calm days", "All days", "Stress days"]).set_index("Period")


def fig_frontiers(fronts, tabs):
    """2x2: rows = samples, cols = (vol, max drawdown) vs out-of-sample stress-day correlation."""
    groups_color = {"Proposed": C_BLUE, "Proposed-2": C_VIOLET, "Original": C_ORANGE,
                    "Baseline": C_AQUA, "Benchmark": C_GRAY, "Random": C_GRAY}
    keep = ["B-SR-DAS (balanced)", "S-SR-DAS (screen+diversify)", "Top-k Quality", "TailDep-Cluster",
            "Corr-Cluster", "MST-Periphery", "PCA-KMeans (Vo et al.)", "EW-All (1/N)", "HRP-All"]
    short = {"B-SR-DAS (balanced)": "B-SR-DAS", "S-SR-DAS (screen+diversify)": "S-SR-DAS",
             "PCA-KMeans (Vo et al.)": "PCA-KMeans", "EW-All (1/N)": "1/N"}
    fig, axes = plt.subplots(len(fronts), 2, figsize=(10, 4.2 * len(fronts)), squeeze=False)
    for i, (label, front) in enumerate(fronts.items()):
        tab = tabs[label]
        for j, ycol in enumerate(["Vol", "MaxDD"]):
            ax = axes[i, j]
            f = front.sort_values("theta")
            yv = f[ycol].astype(float) if ycol == "Vol" else -f[ycol].astype(float)
            ax.plot(f["StressCorr_OOS"].astype(float), yv, "-o", color=C_BLUE, ms=3.5, lw=1.2, alpha=0.8)
            for _, r in f.iterrows():
                if r["theta"] in (0, 0.1, 0.5, 1.0):
                    ax.annotate(f"θ={r['theta']:g}", (float(r["StressCorr_OOS"]), float(r[ycol]) if ycol == "Vol" else -float(r[ycol])),
                                fontsize=6.5, color=C_INK2, xytext=(3, 3), textcoords="offset points")
            for n in keep:
                if n not in tab.index:
                    continue
                r = tab.loc[n]
                yy = float(r[ycol]) if ycol == "Vol" else -float(r[ycol])
                prop = r["Group"] in ("Proposed", "Proposed-2")
                col = C_ORANGE if n == "Top-k Quality" else groups_color.get(r["Group"], C_GRAY)
                ax.scatter(float(r["StressCorr_OOS"]), yy, s=90 if prop else 28,
                           marker="*" if prop else "o", color=col,
                           edgecolor="white", linewidth=0.8, zorder=4)
                ax.annotate(short.get(n, n), (float(r["StressCorr_OOS"]), yy), fontsize=7, color=C_INK,
                            xytext=(4, -8), textcoords="offset points")
            ax.set_xlabel("out-of-sample stress-day correlation of held basket (lower = better)")
            ax.set_ylabel("annualized volatility" if ycol == "Vol" else "maximum drawdown (size)")
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
            ax.set_title(f"({'abcd'[2 * i + j]}) {label}: stress correlation vs {'volatility' if ycol == 'Vol' else 'drawdown'}",
                         loc="left")
    fig.text(0.01, -0.01, "Line: SR-DAS with fixed θ (quality weight). Stars: proposed methods. "
             "Colours: blue = B-SR-DAS and the θ line, violet = S-SR-DAS, orange = Top-k Quality, aqua = other selection baselines, grey = full-universe benchmarks.",
             fontsize=7.5, color=C_INK2)
    plt.tight_layout()
    return fig


def fig_wealth_drawdown(res, names, title):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [1.6, 1]})
    for n in names:
        col, ls = SERIES_STYLE.get(n, (C_GRAY, "-"))
        w = (1 + res[n].returns).cumprod()
        lw = 2.0 if n.startswith("B-SR-DAS") else 1.3
        a1.plot(w.index, w.values, color=col, ls=ls, lw=lw, label=n)
        a2.plot(w.index, (w / w.cummax() - 1).values, color=col, ls=ls, lw=lw * 0.8)
        _label_end(a1, w.index[-1], w.values[-1], f"{n.split(' (')[0]} {w.values[-1]:.1f}×", col)
    a1.set_yscale("log"); a1.set_ylabel("growth of 1 (log scale)")
    a1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    a1.yaxis.set_minor_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}" if f"{v:.1g}"[0] in "25" else ""))
    a1.set_title(f"(a) {title}: cumulative wealth, net of 10 bps costs", loc="left")
    a1.legend(loc="upper left", ncol=2)
    a2.set_ylabel("drawdown"); a2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    a2.set_title("(b) Drawdowns", loc="left")
    plt.tight_layout()
    return fig


def fig_theta(thetas):
    fig, ax = plt.subplots(figsize=(10, 2.8))
    for (label, th), col in zip(thetas.items(), [C_BLUE, C_VIOLET]):
        ax.plot(th.index, th.values, color=col, lw=1.3, label=f"balanced θ, {label}")
    ax.axhline(0.5, color=C_ORANGE, ls="--", lw=1.2, label="original fixed θ = 0.5")
    ax.set_ylabel("θ"); ax.legend(loc="upper right", ncol=3)
    ax.set_title("Quality weight θ chosen by the balance rule at each rebalance", loc="left")
    plt.tight_layout()
    return fig


def fig_h1(pairs):
    """Distribution across quarters of the stress-day basket correlation difference
    (B-SR-DAS minus Top-k Quality). Negative = B-SR-DAS more diversified in stress."""
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    rng = np.random.default_rng(0)
    for i, (label, d) in enumerate(pairs.items()):
        d = d.dropna()
        y = i + rng.uniform(-0.12, 0.12, len(d))
        ax.scatter(d.values, y, s=10, color=C_BLUE, alpha=0.55, lw=0)
        ax.plot([d.median()] * 2, [i - 0.25, i + 0.25], color=C_INK, lw=2)
        ax.annotate(f"median {d.median():+.2f}; below 0 in {(d < 0).mean():.0%} of quarters",
                    (d.median(), i + 0.3), fontsize=7.5, color=C_INK, ha="center")
    ax.axvline(0, color=C_INK2, lw=1, ls=":")
    ax.set_yticks(range(len(pairs))); ax.set_yticklabels(list(pairs))
    ax.set_xlabel("stress-day basket correlation: B-SR-DAS minus Top-k Quality (per quarter)")
    ax.set_title("H1: B-SR-DAS holds a basket that is less correlated on stress days", loc="left")
    ax.set_ylim(-0.6, len(pairs) - 0.3)
    plt.tight_layout()
    return fig


def fig_robustness(rob):
    """Dot plot of Sharpe and stress correlation of B-SR-DAS vs Top-k and 1/N per configuration."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.32 * len(rob) + 1.6), sharey=True)
    y = np.arange(len(rob))
    for ax, metric in zip(axes, ["Sharpe", "StressCorr"]):
        for short, lab, col, mk in [("B-SR-DAS", "B-SR-DAS", C_BLUE, "o"), ("TopK", "Top-k Quality", C_ORANGE, "s"),
                                    ("1/N", "1/N", C_GRAY, "D")]:
            ax.scatter(rob[f"{metric} {short}"].astype(float), y, color=col, marker=mk, s=30, label=lab,
                       edgecolor="white", linewidth=0.7, zorder=3)
        ax.set_title("Sharpe ratio" if metric == "Sharpe" else "Stress-day basket correlation (lower = better)", loc="left")
    axes[0].set_yticks(y); axes[0].set_yticklabels(rob["Configuration"]); axes[0].invert_yaxis()
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    return fig
