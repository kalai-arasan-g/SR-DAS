# %% [french_loader]
import io
import os
import zipfile
import requests

FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"


def _download_zip_text(filename, cache_dir="data"):
    """Download a Kenneth French zip file (cached locally) and return the CSV text."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, filename)
    if not os.path.exists(path):
        url = FRENCH_BASE + filename
        print("Downloading", url)
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (research script)"}, timeout=120)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
    with zipfile.ZipFile(path) as z:
        name = [n for n in z.namelist() if n.lower().endswith((".csv", ".txt"))][0]
        return z.read(name).decode("latin-1")


def parse_french_csv(text, block=0):
    """Parse a Kenneth French CSV file and return block `block` as a DataFrame.

    The files contain one or more tables (e.g. value-weighted, then equal-weighted
    returns). Each table starts with a header line beginning with ',' and continues
    with rows whose first field is a YYYYMMDD date. Values are in percent and
    missing values are coded -99.99 or -999; both are converted to NaN and
    returns are converted to decimals.
    """
    lines = text.splitlines()
    blocks, cur, header = [], None, None
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if line.startswith(",") and len(parts) > 2:
            header = parts[1:]
            cur = []
            blocks.append((header, cur))
            continue
        if cur is not None:
            if len(parts) >= 2 and parts[0].isdigit() and len(parts[0]) == 8:
                cur.append(parts[: len(header) + 1])
            elif cur:  # end of the current table
                cur = None
    header, rows = blocks[block]
    df = pd.DataFrame(rows, columns=["date"] + header)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.set_index("date").apply(pd.to_numeric, errors="coerce")
    df = df.mask(df <= -99.99)
    return df / 100.0


def load_french_industries(n=49, start="1985-01-01", cache_dir="data", weighting="value"):
    """Daily returns of the Fama-French n-industry portfolios (decimal)."""
    text = _download_zip_text(f"{n}_Industry_Portfolios_daily_CSV.zip", cache_dir)
    df = parse_french_csv(text, block=0 if weighting == "value" else 1)
    return df.loc[start:]


def load_french_factors(start="1985-01-01", cache_dir="data"):
    """Daily Fama-French factors: returns DataFrame with Mkt (total market return), Mkt-RF, RF."""
    text = _download_zip_text("F-F_Research_Data_Factors_daily_CSV.zip", cache_dir)
    f = parse_french_csv(text, block=0)
    f["Mkt"] = f["Mkt-RF"] + f["RF"]
    return f.loc[start:]
