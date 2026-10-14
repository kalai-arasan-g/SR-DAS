# %% [membership]
import os
import io
import time
import requests

MEMBERSHIP_URL = ("https://raw.githubusercontent.com/fja05680/sp500/master/"
                  "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv")


def load_sp500_membership(cache_dir="data"):
    """Point-in-time S&P 500 membership (fja05680/sp500, based on Clenow's list + updates).

    Each row is a date on which the index changed, with the full member list on that
    date. Membership on any day = the most recent row on or before that day.
    Returns a Series: date -> list of tickers (exchange-style, e.g. 'BRK.B').
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "sp500_membership.csv")
    if not os.path.exists(path):
        print("Downloading S&P 500 membership history ...")
        r = requests.get(MEMBERSHIP_URL, timeout=120)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    s = df.set_index("date")["tickers"].map(lambda x: sorted(set(t.strip() for t in x.split(",") if t.strip())))
    return s.sort_index()


def yahoo_symbol(t):
    """Exchange ticker -> Yahoo symbol (share classes use '-', e.g. BRK.B -> BRK-B)."""
    return t.replace(".", "-")


def make_universe_fn(membership):
    """Return f(date) -> list of Yahoo symbols that were S&P 500 members on that date."""
    dates = membership.index

    def universe(date):
        i = dates.searchsorted(pd.Timestamp(date), side="right") - 1
        if i < 0:
            return []
        return [yahoo_symbol(t) for t in membership.iloc[i]]
    return universe


def tickers_ever(membership, start, end=None):
    """All tickers that were members at any time in [start, end]."""
    m = membership.loc[:end] if end else membership
    first = m.index.searchsorted(pd.Timestamp(start), side="right") - 1
    rows = m.iloc[max(first, 0):]
    return sorted({yahoo_symbol(t) for lst in rows for t in lst})


# %% [prices]
def _yf_download(tickers, start, end):
    """Thin wrapper around yfinance (kept separate so it can be swapped or mocked)."""
    import yfinance as yf
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, actions=False,
                     progress=False, threads=True, group_by="column")
    if df is None or len(df) == 0:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"]
    else:
        close = df[["Close"]].rename(columns={"Close": tickers[0]})
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    return close


def download_prices(tickers, start, end=None, cache_dir="data/prices", chunk=50, pause=1.5, retries=3):
    """Download adjusted close prices in chunks, caching each chunk to disk.

    Re-running skips chunks already on disk, so an interrupted download (rate limit,
    Colab disconnect) resumes where it stopped. Point cache_dir at Google Drive to
    keep the data between Colab sessions.
    """
    os.makedirs(cache_dir, exist_ok=True)
    tickers = sorted(set(tickers))
    frames = []
    chunks = [tickers[i:i + chunk] for i in range(0, len(tickers), chunk)]
    for ci, ch in enumerate(chunks):
        path = os.path.join(cache_dir, f"chunk_{ci:03d}_{len(tickers)}_{start}.pkl")
        if os.path.exists(path):
            frames.append(pd.read_pickle(path))
            continue
        for attempt in range(retries):
            try:
                px = _yf_download(ch, start, end)
                break
            except Exception as e:  # rate limit / network hiccup
                print(f"  chunk {ci}: attempt {attempt + 1} failed ({e}); retrying")
                time.sleep(pause * 5 * (attempt + 1))
                px = pd.DataFrame()
        px = px.reindex(columns=ch) if len(px) else pd.DataFrame(columns=ch)
        px.to_pickle(path)
        frames.append(px)
        print(f"  chunk {ci + 1}/{len(chunks)}: {px.notna().any().sum()}/{len(ch)} tickers with data")
        time.sleep(pause)
    prices = pd.concat(frames, axis=1).sort_index()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    return prices.loc[:, ~prices.columns.duplicated()]


# %% [cleaning]
def prices_to_returns(prices, min_share=0.5, spike=0.4, verbose=True):
    """Convert prices to clean daily simple returns.

    * returns computed per ticker without forward-filling across gaps;
    * days on which fewer than `min_share` of the tickers that are trading have
      data are dropped (holiday/feed artefacts);
    * one-day spike-and-reversal errors (|r_t| > spike followed by an opposite move
      that undoes it to within 5%) are set to NaN — a common Yahoo data error.
    Returns (returns, report) where report lists the corrections.
    """
    px = prices.sort_index()
    active = px.notna().sum(axis=1)
    keep = active >= min_share * active.rolling(20, min_periods=1).median()
    px = px[keep]
    rets = px.pct_change(fill_method=None)
    nxt = rets.shift(-1)
    spike_mask = (rets.abs() > spike) & (nxt.abs() > spike / 2) & (np.sign(rets) != np.sign(nxt)) & \
                 (((1 + rets) * (1 + nxt) - 1).abs() < 0.05)
    both = spike_mask | spike_mask.shift(1, fill_value=False)
    n_fix = int(both.values.sum())
    rets = rets.mask(both)
    extreme = rets.stack()
    extreme = extreme[extreme.abs() > 0.5]
    report = dict(days_dropped=int((~keep).sum()), spike_points_removed=n_fix,
                  remaining_moves_over_50pct=len(extreme))
    if verbose:
        print("Cleaning report:", report)
    return rets.iloc[1:], report, extreme


def coverage_audit(membership, returns, rebalance_dates, lookback, min_coverage):
    """For each rebalance date: index members, members with any Yahoo data,
    and members eligible for selection (enough history in the look-back)."""
    uni = make_universe_fn(membership)
    rows = []
    idx = returns.index
    for d in rebalance_dates:
        t = idx.searchsorted(d, side="right") - 1
        mem = uni(d)
        have = [m for m in mem if m in returns.columns]
        win = returns.iloc[max(0, t - lookback + 1): t + 1].reindex(columns=have)
        cov = win.notna().mean()
        elig = int(((cov >= min_coverage) & win.iloc[-1].notna()).sum()) if len(have) else 0
        rows.append(dict(date=d, members=len(mem), with_yahoo_data=int((cov > 0).sum()),
                         eligible=elig))
    df = pd.DataFrame(rows).set_index("date")
    df["coverage_%"] = 100 * df["eligible"] / df["members"]
    return df
