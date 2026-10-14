# SR-DAS: Balancing Quality and Diversity in Determinantal Asset Selection

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23360778.svg)](https://doi.org/10.5281/zenodo.23360778)

Code, frozen specification and results for the manuscript

> Kalaiarasan G. *Balancing Quality and Diversity in Determinantal Asset Selection: A Pre-specified Test under Market Stress.* Submitted to *Quantitative Finance*.

The study asks whether a small basket of assets (k = 10) can be chosen to stay diversified when markets fall. It tests stress-robust determinantal asset selection (SR-DAS) and its balanced version (B-SR-DAS) against cluster-based, network-based and quality-based selection rules and the equally weighted portfolio, using the 49 Fama–French industry portfolios (1931–2026) and point-in-time S&P 500 constituents (2005–2026).

## Repository contents

| Folder | Contents |
|---|---|
| `notebooks/` | Seven self-contained Google Colab notebooks. Each downloads its own data and contains all the code it needs. |
| `specification/` | The frozen specification (`specification.json`, fingerprint `a1be5ab38d9e20b6`) and its addendum (`addendum_1.json`, fingerprint `27bc5ad38194a6e9`). |
| `src/` | The same library code as in the notebooks, as plain Python modules, for reading and reuse. |
| `results/` | The output tables (CSV) produced by each notebook in the runs reported in the paper. |
| `verify_fingerprints.py` | Recomputes the two fingerprints from the specification files. |

## Notebooks

| Notebook | Purpose | Role in the paper |
|---|---|---|
| `SRDAS_01_FF49_Industries` | First specification (fixed weight θ = 0.5) on the industries, 1990–2026 | Development sample; Appendix Table C2 |
| `SRDAS_02_SP500_Yahoo` | First specification on S&P 500 stocks, 2005–2026 | Development sample; Appendix Table C2 |
| `SRDAS_03_Revised_Holdout` | Frozen specification (B-SR-DAS, hypotheses H1–H7) run once on 1970–1989; also 1990–2026 | **First confirmatory sample** |
| `SRDAS_04_Manuscript_Evidence` | Tables and figures of the paper, robustness variants, S&P 500 replication and survivorship check | Sections 4–5, Appendix C |
| `SRDAS_05_PostHoc` | Truncation bias of stress-day correlation, crisis-window correlations, crisis-window basket correlation | Section 6 (added after the first confirmatory run) |
| `SRDAS_06_SP500_Descriptives` | Descriptive statistics of the stock sample | Table 1 |
| `SRDAS_07_Confirmatory_Robustness` | Frozen specification run once on 1931–1969 under the addendum; inference robust to serial dependence; block-length sensitivity; crisis windows for stocks | **Second confirmatory sample**; Appendix C |

Note on wording: the code comments call the frozen specification "pre-registered". Because it was fixed within our own workflow and not deposited in a public registry, the paper calls it **pre-specified**.

## How to reproduce

1. Open a notebook in Google Colab (File → Upload notebook, or open it from GitHub).
2. Runtime → Run all. Each notebook installs its packages, downloads its data and saves its outputs as a zip file.
3. The expected run time is given at the top of each notebook. The stock notebooks (2, 4, 6 and 7) spend most of their time on the Yahoo Finance download; with `SAVE_TO_DRIVE = True` the price cache is kept on Google Drive and reused.

Notebooks 3, 4, 5 and 7 stop with an error if the specification has been changed (fingerprint check). Notebook 7 also checks the addendum.

To check the fingerprints locally:

```bash
python verify_fingerprints.py
```

## Data sources

* Kenneth French Data Library: 49 and 12 industry portfolios (daily, value-weighted) and the daily factors (market return, risk-free rate). https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
* Point-in-time S&P 500 membership: https://github.com/fja05680/sp500
* Stock prices: Yahoo Finance, through the `yfinance` package.

The data are not redistributed here; the notebooks download them. Yahoo Finance data change over time (for example, delisted tickers disappear), so a rerun of the stock notebooks can differ slightly from the reported results. The industry results are exactly reproducible as long as the French library files do not change.

## Note on annualization in 1931–1969

The notebooks annualize returns, volatilities and Sharpe ratios with 252 trading days per year. Until 1952 the NYSE also traded on Saturdays, so the 1931–1969 sample has 10,545 trading days in 38.5 calendar years (273.9 per year). The paper reports that sample in calendar time: volatilities and Sharpe ratios in `results/nb07_confirmatory_robustness` are multiplied by √(273.9/252) = 1.0425, CAGR is converted as (1 + CAGR)^(273.9/252) − 1, and turnover per year is multiplied by 273.9/252. Drawdowns, stress betas, co-movement, volatility ratios and all p-values are unaffected. The later samples have about 252 trading days per year, so no conversion is needed there.

## Software

Python 3.13 with numpy 2.1.3, pandas 2.2.3, scipy 1.16.3, scikit-learn 1.9.1 and networkx 3.7 (Google Colab, October 2026). See `requirements.txt`.

## Citation

If you use this code, please cite the paper and the archived code: Kalaiarasan G (2026), *SR-DAS: code and frozen specification*, version 1.0.0, Zenodo, https://doi.org/10.5281/zenodo.23360778 (see also `CITATION.cff`).

## License

The code is released under the MIT License (see `LICENSE`).

## Contact

Kalaiarasan G, Amrita School of Artificial Intelligence, Amrita Vishwa Vidyapeetham, Coimbatore, India. kalai.forward@gmail.com
