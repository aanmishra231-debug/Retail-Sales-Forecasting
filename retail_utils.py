"""Shared helpers: calendar / promo rules used by both the data generator and the models."""
import numpy as np
import pandas as pd


def black_friday(year: int) -> pd.Timestamp:
    nov1 = pd.Timestamp(year, 11, 1)
    first_thu = nov1 + pd.Timedelta(days=(3 - nov1.weekday()) % 7)
    return first_thu + pd.Timedelta(days=21 + 1)  # day after 4th Thursday


def calendar_features(dates) -> pd.DataFrame:
    """Deterministic calendar info. Promo windows are *scheduled*, so they are
    known in advance for future dates (no leakage)."""
    d = pd.DatetimeIndex(dates)
    out = pd.DataFrame(index=d)
    out["dow"] = d.dayofweek
    out["month"] = d.month
    out["doy"] = d.dayofyear

    promo = np.zeros(len(d), dtype=int)
    for y in sorted(set(d.year)):
        bf = black_friday(y)
        windows = [
            (bf - pd.Timedelta(days=3), bf + pd.Timedelta(days=3)),   # Black Friday week
            (pd.Timestamp(y, 7, 1), pd.Timestamp(y, 7, 14)),          # Summer sale
            (pd.Timestamp(y, 3, 15), pd.Timestamp(y, 3, 21)),         # Spring sale
        ]
        for a, b in windows:
            promo |= ((d >= a) & (d <= b)).astype(int)
    out["promo"] = promo
    out["xmas"] = ((d.month == 12) & (d.day >= 20) & (d.day <= 26)).astype(int)
    out["newyear"] = ((d.month == 1) & (d.day == 1)).astype(int)
    return out


def fourier(doy, k_max: int) -> np.ndarray:
    doy = np.asarray(doy, dtype=float)
    cols = []
    for k in range(1, k_max + 1):
        cols.append(np.sin(2 * np.pi * k * doy / 365.25))
        cols.append(np.cos(2 * np.pi * k * doy / 365.25))
    return np.column_stack(cols)
