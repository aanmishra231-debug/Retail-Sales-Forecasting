"""
Retail sales forecasting pipeline
=================================
1. Load + clean time-based data (missing values, spikes)
2. Explore patterns (trend, weekly/yearly seasonality, promo effect)
3. Compare forecasting methods on a time-based hold-out (last 90 days)
4. Refit the best method and forecast the next 90 days with prediction intervals
5. Save charts + CSV results
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from retail_utils import calendar_features, fourier

warnings.filterwarnings("ignore")
HORIZON = 90
SELECTED_STORE = "Amsterdam"
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})
PAL = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

# ------------------------------------------------------------------ 1. LOAD + CLEAN
raw = pd.read_csv("data/sales_data.csv", parse_dates=["date"])
print(f"Loaded {len(raw):,} rows | {raw.store.nunique()} stores x {raw['product'].nunique()} products "
      f"| {raw.date.min().date()} -> {raw.date.max().date()}")

def clean_series(s: pd.Series):
    """Detect spikes (>3x centred rolling median & far above Poisson noise), then fill gaps
    using the same weekday of adjacent weeks, falling back to interpolation."""
    s = s.copy()
    med = s.rolling(29, center=True, min_periods=7).median()
    spikes = (s > 3 * med) & (s - med > 6 * np.sqrt(med.clip(lower=1)))
    n_spike, n_missing = int(spikes.sum()), int(s.isna().sum())
    s[spikes] = np.nan
    seasonal_fill = pd.concat([s.shift(7), s.shift(-7)], axis=1).mean(axis=1)
    s = s.fillna(seasonal_fill).interpolate(limit_direction="both")
    return s, n_spike, n_missing

series, cleaning_log = {}, []
for (store, prod), g in raw.groupby(["store", "product"]):
    g = g.set_index("date").asfreq("D")                      # enforce a complete daily index
    y, ns, nm = clean_series(g["units_sold"])
    series[(store, prod)] = y
    cleaning_log.append((store, prod, nm, ns))
cl = pd.DataFrame(cleaning_log, columns=["store", "product", "missing_filled", "spikes_fixed"])
print(f"Cleaning: filled {cl.missing_filled.sum()} missing values, repaired {cl.spikes_fixed.sum()} spikes")
clean_df = pd.concat([y.rename("units_sold").to_frame().assign(store=k[0], product=k[1])
                      for k, y in series.items()]).reset_index().rename(columns={"index": "date"})
clean_df.to_csv("data/sales_data_clean.csv", index=False)

# ------------------------------------------------------------------ 2. FEATURES
def design_matrix(dates, kind):
    cal = calendar_features(dates)
    if kind == "ridge":
        dow = pd.get_dummies(cal["dow"]).reindex(columns=range(7), fill_value=0).astype(float).values[:, 1:]
        t = ((cal.index - pd.Timestamp("2023-01-01")).days.values / 365.25)[:, None]
        return np.hstack([t, dow, fourier(cal["doy"], 8),
                          cal[["promo", "xmas", "newyear"]].values.astype(float)])
    return cal[["dow", "month", "doy", "promo", "xmas", "newyear"]].values.astype(float)

# ------------------------------------------------------------------ 3. MODELS
# Each model: f(y_train: Series, future_dates) -> np.ndarray of length len(future_dates)
def m_seasonal_naive7(y, fd):
    last = y.values[-7:]
    return np.array([last[i % 7] for i in range(len(fd))])

def m_moving_avg(y, fd):
    return np.full(len(fd), y.values[-28:].mean())

def m_seasonal_naive_year(y, fd):
    """Same weekday last year (364 days back) scaled by YoY growth."""
    v = y.values
    growth = v[-364:].sum() / v[-728:-364].sum()
    return np.array([v[len(v) - 364 + i] for i in range(len(fd))]) * growth

def m_holt_winters(y, fd, m=7):
    v = y.values.astype(float); n = len(v)
    def run(p, ret=False):
        a, b, g, phi = p
        lvl = v[:m].mean(); tr = (v[m:2*m].mean() - v[:m].mean()) / m
        seas = list(v[:m] - lvl); sse = 0.0
        for t in range(m, n):
            s_old = seas[t % m]
            fc = lvl + phi * tr + s_old
            sse += (v[t] - fc) ** 2
            new_lvl = a * (v[t] - s_old) + (1 - a) * (lvl + phi * tr)
            tr = b * (new_lvl - lvl) + (1 - b) * phi * tr
            seas[t % m] = g * (v[t] - new_lvl) + (1 - g) * s_old
            lvl = new_lvl
        return (lvl, tr, seas, phi) if ret else sse
    res = minimize(run, x0=[0.2, 0.01, 0.1, 0.95], method="L-BFGS-B",
                   bounds=[(0.01, 0.99), (0.0001, 0.3), (0.01, 0.99), (0.8, 0.999)])
    lvl, tr, seas, phi = run(res.x, ret=True)
    out = []
    for h in range(1, len(fd) + 1):
        damp = sum(phi ** i for i in range(1, h + 1))
        out.append(lvl + damp * tr + seas[(n + h - 1) % m])
    return np.maximum(out, 0)

def m_ridge(y, fd):
    X = design_matrix(y.index, "ridge")
    model = Ridge(alpha=1.0).fit(X, np.log1p(y.values))     # log target => multiplicative effects
    return np.expm1(model.predict(design_matrix(fd, "ridge")))

LAGS = (7, 14, 28)
def lag_feats(hist):
    h = np.asarray(hist)
    return [h[-7], h[-14], h[-28], h[-7:].mean(), h[-28:].mean()]

def m_gbm(y, fd):
    v, idx = y.values.astype(float), y.index
    Xc = design_matrix(idx, "gbm")
    X = np.array([np.r_[Xc[i], lag_feats(v[:i])] for i in range(28, len(v))])
    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=5,
                                          random_state=0).fit(X, np.log1p(v[28:]))
    Xf, hist, preds = design_matrix(fd, "gbm"), list(v), []
    for i in range(len(fd)):                                 # recursive multi-step forecast
        p = max(np.expm1(model.predict(np.r_[Xf[i], lag_feats(hist)].reshape(1, -1))[0]), 0)
        preds.append(p); hist.append(p)
    return np.array(preds)

MODELS = {
    "Seasonal naive (7d)":      m_seasonal_naive7,
    "Moving average (28d)":     m_moving_avg,
    "Seasonal naive (1y+growth)": m_seasonal_naive_year,
    "Holt-Winters (weekly)":    m_holt_winters,
    "Ridge regression (calendar)": m_ridge,
    "Gradient boosting (lags+calendar)": m_gbm,
}

def metrics(a, p):
    a, p = np.asarray(a), np.asarray(p)
    return dict(MAE=np.mean(np.abs(a - p)), RMSE=np.sqrt(np.mean((a - p) ** 2)),
                WAPE=np.abs(a - p).sum() / a.sum() * 100, Bias=(p - a).sum() / a.sum() * 100)

# ------------------------------------------------------------------ 4. BACKTEST (time-based hold-out)
rows, holdout_preds = [], {}
for key, y in series.items():
    train, test = y.iloc[:-HORIZON], y.iloc[-HORIZON:]
    for name, f in MODELS.items():
        pred = f(train, test.index)
        holdout_preds[(key, name)] = pred
        rows.append({"store": key[0], "product": key[1], "model": name, **metrics(test.values, pred)})
bt = pd.DataFrame(rows)
bt.to_csv("results/backtest_by_series.csv", index=False)
summary = (bt.groupby("model")[["MAE", "RMSE", "WAPE", "Bias"]].mean()
             .sort_values("WAPE").round(2))
summary["wins (series)"] = bt.loc[bt.groupby(["store", "product"])["WAPE"].idxmin(), "model"] \
                              .value_counts().reindex(summary.index).fillna(0).astype(int)
summary.to_csv("results/backtest_summary.csv")
print("\nBacktest on last", HORIZON, "days (mean over 12 series):\n", summary.to_string())
BEST = summary.index[0]
print("\nBest model:", BEST)

# ------------------------------------------------------------------ 5. FINAL FORECAST
future_idx = pd.date_range(raw.date.max() + pd.Timedelta(days=1), periods=HORIZON, freq="D")
fc_rows = []
for key, y in series.items():
    pred = MODELS[BEST](y, future_idx)
    test = y.iloc[-HORIZON:]
    resid = test.values - holdout_preds[(key, BEST)]
    lo_q, hi_q = np.quantile(resid, [0.10, 0.90])            # empirical 80% interval
    fc_rows.append(pd.DataFrame({"date": future_idx, "store": key[0], "product": key[1],
                                 "forecast_units": pred.round(1),
                                 "lower_80": np.maximum(pred + lo_q, 0).round(1),
                                 "upper_80": (pred + hi_q).round(1)}))
fc = pd.concat(fc_rows, ignore_index=True)
price = raw.groupby("product")["unit_price"].apply(lambda s: s[raw.loc[s.index, "promo"] == 0].iloc[-90:].mean())
fc["forecast_revenue_eur"] = (fc["forecast_units"] * fc["product"].map(price)).round(2)
fc.to_csv("results/forecast_next_90_days.csv", index=False)
print(f"\nForecast {future_idx[0].date()} -> {future_idx[-1].date()} saved.")

# ------------------------------------------------------------------ 6. CHARTS
products = sorted(raw["product"].unique())
pcol = dict(zip(products, PAL))

# 6.1 overview
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={"width_ratios": [2.2, 1]})
tot = clean_df.groupby("date")["units_sold"].sum()
ax[0].plot(tot.index, tot, color="#bbb", lw=0.7, label="Daily units")
ax[0].plot(tot.index, tot.rolling(28, center=True).mean(), color="#d62728", lw=2, label="28-day average")
ax[0].set(title="Total units sold – all stores & products", ylabel="Units / day"); ax[0].legend()
yr = clean_df.assign(year=clean_df.date.dt.year).pivot_table(index="year", columns="product",
                                                             values="units_sold", aggfunc="sum")
yr.plot.bar(ax=ax[1], color=[pcol[p] for p in yr.columns], width=0.8, rot=0)
ax[1].set(title="Yearly units by product", xlabel="", ylabel="Units"); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig("charts/01_overview_trend.png"); plt.close()

# 6.2 seasonality + promo
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
cd = clean_df.copy(); cd = cd.join(calendar_features(cd["date"]).reset_index(drop=True)[["dow", "month", "promo"]])
cd["yr"] = cd.date.dt.year
for p in products:
    d = cd[cd["product"] == p]
    dow = d.groupby("dow")["units_sold"].mean(); ax[0].plot(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], dow / dow.mean(), marker="o", color=pcol[p], label=p)
    mo = d.groupby("month")["units_sold"].mean(); ax[1].plot(mo.index, mo / mo.mean(), marker="o", color=pcol[p], label=p)
ax[0].set(title="Weekly pattern (index, 1 = average)"); ax[1].set(title="Yearly pattern (index, 1 = average)", xlabel="Month")
ax[1].set_xticks(range(1, 13)); ax[0].legend(fontsize=8)
# promo uplift within same month to avoid season confounding
up = []
for p in products:
    d = cd[(cd["product"] == p)]
    base = d.groupby(["yr", "month", "promo"])["units_sold"].mean().unstack()
    up.append(((base[1] / base[0]).dropna() - 1).mean() * 100)
ax[2].bar(products, up, color=[pcol[p] for p in products]); ax[2].tick_params(axis="x", rotation=15)
ax[2].set(title="Average promo uplift vs same-month non-promo days", ylabel="% uplift")
plt.tight_layout(); plt.savefig("charts/02_seasonality_promo.png"); plt.close()

# 6.3 holdout comparison for each product at selected store
fig, axes = plt.subplots(2, 2, figsize=(15, 8), sharex=True)
for ax_, p in zip(axes.ravel(), products):
    y = series[(SELECTED_STORE, p)]; test = y.iloc[-HORIZON:]
    ax_.plot(y.index[-150:-HORIZON + 1], y.iloc[-150:-HORIZON + 1], color="#999", lw=1, label="Train (history)")
    ax_.plot(test.index, test, color="black", lw=1.8, label="Actual")
    for (name, c) in zip(MODELS, PAL):
        ls = "-" if name == BEST else "--"
        ax_.plot(test.index, holdout_preds[((SELECTED_STORE, p), name)], ls, color=c, lw=1.6 if name == BEST else 1,
                 alpha=1 if name == BEST else 0.7, label=name)
    ax_.set_title(f"{SELECTED_STORE} – {p}")
axes[0, 0].legend(fontsize=7, ncol=2)
fig.suptitle(f"Hold-out forecasts: last {HORIZON} days (solid coloured = best model)", y=1.0)
plt.tight_layout(); plt.savefig("charts/03_holdout_forecasts.png"); plt.close()

# 6.4 model scores
fig, ax = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [1, 1.5]})
s = summary["WAPE"].sort_values(ascending=False)
ax[0].barh(s.index, s.values, color=["#2ca02c" if m == BEST else "#9aa" for m in s.index])
for i, v in enumerate(s.values): ax[0].text(v + 0.3, i, f"{v:.1f}%", va="center")
ax[0].set(title="Mean WAPE on hold-out (lower is better)", xlabel="WAPE %")
hm = bt.assign(series=bt.store + " | " + bt["product"]).pivot(index="series", columns="model", values="WAPE")[list(summary.index)]
im = ax[1].imshow(hm.values, cmap="YlOrRd", aspect="auto")
ax[1].set_xticks(range(len(hm.columns))); ax[1].set_xticklabels([c.replace(" (", "\n(") for c in hm.columns], fontsize=7)
ax[1].set_yticks(range(len(hm.index))); ax[1].set_yticklabels(hm.index, fontsize=8); ax[1].grid(False)
for i in range(hm.shape[0]):
    for j in range(hm.shape[1]): ax[1].text(j, i, f"{hm.values[i, j]:.0f}", ha="center", va="center", fontsize=7)
ax[1].set_title("WAPE % per store | product"); plt.colorbar(im, ax=ax[1], fraction=0.03)
plt.tight_layout(); plt.savefig("charts/04_model_comparison.png"); plt.close()

# 6.5 final forecast
fig, axes = plt.subplots(2, 2, figsize=(15, 8), sharex=True)
for ax_, p in zip(axes.ravel(), products):
    y = series[(SELECTED_STORE, p)]; f = fc[(fc.store == SELECTED_STORE) & (fc["product"] == p)]
    ax_.plot(y.index[-180:], y.iloc[-180:], color="#555", lw=1, label="History")
    ax_.plot(f.date, f.forecast_units, color=pcol[p], lw=2, label=f"Forecast ({BEST.split(' (')[0]})")
    ax_.fill_between(f.date, f.lower_80, f.upper_80, color=pcol[p], alpha=0.2, label="80% interval")
    ax_.set_title(f"{SELECTED_STORE} – {p}"); ax_.legend(fontsize=7, loc="upper left")
fig.suptitle(f"Forecast: {future_idx[0].date()} to {future_idx[-1].date()}", y=1.0)
plt.tight_layout(); plt.savefig("charts/05_future_forecast.png"); plt.close()

# 6.6 residual diagnostics for best model (all series, hold-out)
res = np.concatenate([series[k].iloc[-HORIZON:].values - holdout_preds[(k, BEST)] for k in series])
rel = []
for k in series:
    a = series[k].iloc[-HORIZON:].values; rel.append((a - holdout_preds[(k, BEST)]) / series[k].iloc[-HORIZON:].mean() * 100)
rel = np.concatenate(rel)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(rel, bins=40, color="#1f77b4", alpha=0.85); ax[0].axvline(0, color="k")
ax[0].set(title=f"Hold-out errors, {BEST}", xlabel="Error as % of series mean (actual − forecast)")
store_tot = {}
for st in sorted(raw.store.unique()):
    a = sum(series[(st, p)].iloc[-HORIZON:] for p in products)
    f_ = sum(pd.Series(holdout_preds[((st, p), BEST)], index=a.index) for p in products)
    ax[1].plot(a.index, a, lw=1.5, label=f"{st} actual"); ax[1].plot(a.index, f_, "--", lw=1.2, label=f"{st} forecast")
ax[1].set(title="Store-level totals: actual vs forecast (hold-out)"); ax[1].legend(fontsize=7, ncol=2)
plt.tight_layout(); plt.savefig("charts/06_residuals_store_totals.png"); plt.close()
print("Charts saved.")
