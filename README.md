# Retail Sales Forecasting Project

Goal: use historical retail sales to predict future sales for selected products / stores.

## Files
| File | Purpose |
|---|---|
| `generate_data.py` | Builds the dataset (`data/sales_data.csv`): 3 stores x 4 products x 3 years of daily sales (13,152 rows) with trend, weekly + yearly seasonality, promotions, holidays, noise, and injected data-quality issues |
| `retail_utils.py` | Calendar / promo rules + Fourier features (shared) |
| `forecast.py` | Full pipeline: clean -> explore -> compare models -> forecast -> charts |
| `data/sales_data.csv` | Raw data (47 missing values, 24 spikes) |
| `data/sales_data_clean.csv` | Cleaned data |
| `results/backtest_summary.csv`, `backtest_by_series.csv` | Model comparison |
| `results/forecast_next_90_days.csv` | Forecast 2026-01-01 .. 2026-03-31 with 80% intervals, for all 12 series |
| `charts/*.png` | 6 visualizations |

Run: `python generate_data.py && python forecast.py` (needs pandas, numpy, scipy, scikit-learn, matplotlib)

## Method
1. **Preparation** - enforce complete daily index, detect spikes (>3x rolling median), fill gaps/spikes from the same weekday in adjacent weeks.
2. **Patterns found** - upward trend; strong weekend effect; opposite yearly cycles (ice cream peaks Jul, jackets Dec, school supplies Aug-Sep, coffee flat); promos give ~+35% units.
3. **Models compared** (time-based hold-out = last 90 days, no shuffling):
   Seasonal naive (7d), Moving average (28d), Seasonal naive (1y + growth), Holt-Winters (weekly), Ridge regression (trend + weekday + Fourier yearly seasonality + promo/holiday, log target), Gradient boosting (calendar + lag features, recursive forecast).
4. **Metrics** - MAE, RMSE, WAPE (% error weighted by volume), Bias.

## Results (mean of 12 series, last 90 days)
| Model | MAE | RMSE | WAPE % | Bias % | Series won |
|---|---|---|---|---|---|
| **Ridge regression (calendar)** | 6.58 | 8.41 | **15.6** | -2.7 | 10 |
| Gradient boosting (lags+calendar) | 7.84 | 10.10 | 18.7 | -8.5 | 2 |
| Seasonal naive (1y+growth) | 9.81 | 12.29 | 22.8 | -1.4 | 0 |
| Holt-Winters (weekly) | 13.84 | 17.13 | 33.1 | 7.5 | 0 |
| Seasonal naive (7d) | 17.21 | 20.79 | 42.4 | 13.5 | 0 |
| Moving average (28d) | 20.55 | 23.66 | 52.9 | 24.7 | 0 |

Why: models without yearly seasonality (Holt-Winters weekly, 7d naive, moving average) fail on strongly seasonal products (ice cream, jackets). The regression model captures trend + weekday + yearly cycle + promo calendar explicitly.

## Caveats
- Data is **synthetic** - swap in your real CSV with columns `date, store, product, units_sold, unit_price, promo`.
- Promo calendar is assumed known in advance (it is a scheduled input, not a prediction).
- Part of the remaining error is irreducible daily noise; weekly aggregation would give much lower WAPE.
- Intervals come from empirical hold-out residuals (80%).
