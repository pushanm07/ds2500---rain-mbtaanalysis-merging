"""
DS 2500 Team Project - Goal 1 (Pushan): do rainy days have longer headways and
travel times than dry days?

Input : merged_daily.csv from merge_mbta_noaa.py
Output: results/ (CSV tables + PNG figures) and a printed summary.

Usage:
    python rain_analysis.py --data data/merged_daily.csv --start 2022-01-01

Design choices (state these in the report):
  * Rain day  = PRCP >= 0.10 in, no snow, TMAX > 35F.   Dry day = PRCP == 0 and no snow.
    Trace/light-rain days and snow days are excluded from the rain-vs-dry test.
  * Unit of analysis = one day.  The main t-test averages the four lines into one value
    per day so that observations are not counted four times.  Per-line tests then show
    which lines are most affected (Holm-corrected for 4 comparisons).
  * Welch's t-test (unequal variances) + Mann-Whitney U as a robustness check, with
    Cohen's d as the effect size and a 95% CI on the difference.
  * A regression with line / day-of-week / month / year controls, errors clustered by
    date, checks the result isn't just "rainy days happen to be weekdays / winter /
    a slow-zone year".
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

METRICS = {
    "hw_ratio_mean": "Headway / scheduled headway (1.0 = on schedule)",
    "hw_dev_mean":   "Headway minus scheduled headway (sec)",
    "pct_long_gaps": "Share of headways > 1.5x scheduled",
    "tt_ratio_mean": "Travel time / benchmark travel time",
    "tt_dev_mean":   "Travel time minus benchmark (sec)",
}
PRIMARY = ["hw_ratio_mean", "tt_ratio_mean"]       # headline metrics for the report


# ------------------------------------------------------------------ helpers
def welch(rain, dry):
    rain, dry = rain.dropna(), dry.dropna()
    if len(rain) < 3 or len(dry) < 3:
        return None
    t, p = stats.ttest_ind(rain, dry, equal_var=False)
    diff = rain.mean() - dry.mean()
    v1, v2 = rain.var(ddof=1) / len(rain), dry.var(ddof=1) / len(dry)
    se = np.sqrt(v1 + v2)
    df = (v1 + v2) ** 2 / (v1 ** 2 / (len(rain) - 1) + v2 ** 2 / (len(dry) - 1))
    ci = stats.t.ppf(0.975, df) * se
    pooled = np.sqrt(((len(rain) - 1) * rain.var(ddof=1) + (len(dry) - 1) * dry.var(ddof=1))
                     / (len(rain) + len(dry) - 2))
    u_p = stats.mannwhitneyu(rain, dry, alternative="two-sided").pvalue
    return dict(n_rain=len(rain), n_dry=len(dry), mean_rain=rain.mean(), mean_dry=dry.mean(),
                diff=diff, ci_low=diff - ci, ci_high=diff + ci, t=t, p_welch=p,
                p_mannwhitney=u_p, cohens_d=diff / pooled if pooled else np.nan)


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def clustered_ols(df, y, controls, cluster):
    """OLS of y on a rain dummy + categorical controls, SEs clustered by `cluster`."""
    d = df[[y, "rain", cluster] + controls].dropna()
    X = pd.get_dummies(d[["rain"] + controls], columns=controls, drop_first=True, dtype=float)
    X.insert(0, "const", 1.0)
    Xm, yv = X.to_numpy(float), d[y].to_numpy(float)
    XtX_inv = np.linalg.pinv(Xm.T @ Xm)
    beta = XtX_inv @ Xm.T @ yv
    u = yv - Xm @ beta
    groups = pd.factorize(d[cluster])[0]
    meat = np.zeros((Xm.shape[1],) * 2)
    for g in np.unique(groups):
        s = Xm[groups == g].T @ u[groups == g]
        meat += np.outer(s, s)
    G, n, k = len(np.unique(groups)), len(yv), Xm.shape[1]
    V = XtX_inv @ meat @ XtX_inv * (G / (G - 1)) * ((n - 1) / (n - k))
    i = list(X.columns).index("rain")
    se = np.sqrt(V[i, i])
    p = 2 * stats.t.sf(abs(beta[i] / se), G - 1)
    return dict(coef=beta[i], se=se, ci_low=beta[i] - 1.96 * se, ci_high=beta[i] + 1.96 * se,
                p=p, n=n, clusters=G)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/merged_daily.csv")
    ap.add_argument("--start", default="2022-01-01",
                    help="skip 2020-21 COVID-era service levels")
    ap.add_argument("--end", default=None)
    ap.add_argument("--weekdays-only", action="store_true",
                    help="restrict to Mon-Fri to match commuter service patterns")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = pd.read_csv(a.data, parse_dates=["date"])
    df = df[df["date"] >= a.start]
    if a.end:
        df = df[df["date"] <= a.end]
    if a.weekdays_only:
        df = df[~df["is_weekend"]]
    df = df[df["weather_group"].isin(["rain", "dry"])].copy()
    df["rain"] = (df["weather_group"] == "rain").astype(float)
    metrics = [m for m in METRICS if m in df.columns]

    print(f"Rain-vs-dry sample: {df['date'].nunique()} days "
          f"({df.loc[df.rain == 1, 'date'].nunique()} rain, {df.loc[df.rain == 0, 'date'].nunique()} dry), "
          f"{len(df)} line-days\n")

    # ---- 1. day-level Welch t-test (lines averaged per day) -------------------
    daily = df.groupby(["date", "rain"], as_index=False)[metrics].mean()
    rows = []
    for m in metrics:
        r = welch(daily.loc[daily.rain == 1, m], daily.loc[daily.rain == 0, m])
        if r:
            rows.append(dict(metric=m, **r))
    overall = pd.DataFrame(rows)
    overall.to_csv(f"{a.out}/rain_vs_dry_overall.csv", index=False)

    # ---- 2. per-line tests (which lines are hit hardest) ----------------------
    rows = []
    for m in metrics:
        sub = []
        for line, g in df.groupby("line"):
            r = welch(g.loc[g.rain == 1, m], g.loc[g.rain == 0, m])
            if r:
                sub.append(dict(metric=m, line=line, **r))
        if sub:
            adj = holm([s["p_welch"] for s in sub])
            for s, pa in zip(sub, adj):
                s["p_holm"] = pa
            rows += sub
    by_line = pd.DataFrame(rows)
    by_line.to_csv(f"{a.out}/rain_vs_dry_by_line.csv", index=False)

    # ---- 3. dose-response: does heavier rain mean worse service? ---------------
    # (uses all non-snow days, so reload the unfiltered frame)
    full = pd.read_csv(a.data, parse_dates=["date"])
    full = full[(full["date"] >= a.start) & (full["weather_group"] != "snow")
                & full["rain_intensity"].notna()]
    if a.end:
        full = full[full["date"] <= a.end]
    if a.weekdays_only:
        full = full[~full["is_weekend"]]
    full_daily = full.groupby(["date", "rain_intensity"], as_index=False)[metrics].mean()
    intensity = (full_daily.groupby("rain_intensity", observed=True)[metrics]
                 .agg(["mean", "count"]).round(4))
    intensity.to_csv(f"{a.out}/by_rain_intensity.csv")
    prcp_daily = full.groupby("date", as_index=False).agg(prcp=("prcp", "first"),
                                                          **{m: (m, "mean") for m in metrics})
    corr_rows = []
    for m in metrics:
        s = prcp_daily[["prcp", m]].dropna()
        if len(s) > 5:
            rho, p = stats.spearmanr(s["prcp"], s[m])
            corr_rows.append(dict(metric=m, spearman_rho=rho, p=p, n_days=len(s)))
    pd.DataFrame(corr_rows).to_csv(f"{a.out}/prcp_spearman.csv", index=False)

    # ---- 4. regression with controls -----------------------------------------
    reg_rows = []
    for m in metrics:
        res = clustered_ols(df, m, ["line", "dow", "month", "year"], cluster="date")
        reg_rows.append(dict(metric=m, **res))
    pd.DataFrame(reg_rows).to_csv(f"{a.out}/rain_regression_controls.csv", index=False)

    # ---- figures ---------------------------------------------------------------
    fig, axes = plt.subplots(1, len(PRIMARY), figsize=(5.5 * len(PRIMARY), 4.5))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, [p for p in PRIMARY if p in metrics]):
        ax.boxplot([daily.loc[daily.rain == 0, m].dropna(), daily.loc[daily.rain == 1, m].dropna()],
                   showmeans=True)
        ax.set_xticklabels(["Dry days", "Rain days"])
        ax.set_ylabel(METRICS[m]); ax.set_title(m)
    fig.suptitle("MBTA subway performance: rain vs. dry days (daily, all lines)")
    fig.tight_layout(); fig.savefig(f"{a.out}/fig_rain_vs_dry_boxplot.png", dpi=200); plt.close(fig)

    colors = {"Red": "#DA291C", "Orange": "#ED8B00", "Blue": "#003DA5", "Green": "#00843D"}
    for m in [p for p in PRIMARY if p in metrics]:
        s = by_line[by_line.metric == m]
        if s.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.bar(s["line"], s["diff"], color=[colors.get(l, "gray") for l in s["line"]],
               yerr=[s["diff"] - s["ci_low"], s["ci_high"] - s["diff"]], capsize=4)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel(f"Rain minus dry: {METRICS[m]}")
        ax.set_title(f"Effect of rain by line ({m}, 95% CI)")
        fig.tight_layout(); fig.savefig(f"{a.out}/fig_rain_effect_by_line_{m}.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    if "hw_ratio_mean" in metrics:
        ax.scatter(prcp_daily["prcp"], prcp_daily["hw_ratio_mean"], s=8, alpha=0.4)
        ax.set_xlabel("Daily precipitation (inches)"); ax.set_ylabel(METRICS["hw_ratio_mean"])
        ax.set_title("Headway ratio vs. precipitation (non-snow days)")
        fig.tight_layout(); fig.savefig(f"{a.out}/fig_headway_vs_prcp.png", dpi=200)
    plt.close(fig)

    # ---- printed summary ------------------------------------------------------
    pd.set_option("display.width", 160)
    cols = ["metric", "n_rain", "n_dry", "mean_rain", "mean_dry", "diff", "ci_low", "ci_high",
            "p_welch", "p_mannwhitney", "cohens_d"]
    print("=== Rain vs dry, day-level Welch t-test ===")
    print(overall[cols].round(4).to_string(index=False))
    print("\n=== By line (Holm-adjusted p) ===")
    print(by_line[["metric", "line", "diff", "ci_low", "ci_high", "p_welch", "p_holm", "cohens_d"]]
          .round(4).to_string(index=False))
    print("\n=== Rain effect after controls (line, weekday, month, year; clustered by date) ===")
    print(pd.DataFrame(reg_rows).round(4).to_string(index=False))
    print(f"\nTables and figures saved in {a.out}/")


if __name__ == "__main__":
    main()
