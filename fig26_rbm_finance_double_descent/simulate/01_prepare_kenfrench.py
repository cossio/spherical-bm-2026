# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "numpy>=2.4",
#     "pandas>=2.2",
#     "h5py>=3.10",
#     "requests>=2.32",
# ]
# ///
"""Download Ken French 49 Industry daily returns, preprocess, save HDF5.

Outputs:
  data/kenfrench49_daily.h5 — train_x, test_x, etc. (standardized returns)
  data/kenfrench49_binary.h5 — binarized returns (sign vs industry median)
"""

import io
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import requests

URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/49_Industry_Portfolios_daily_CSV.zip"
# Canonical value-weighted daily returns CSV inside the zip.

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)


def fetch_csv():
    cache = DATA / "49_industry.csv"
    if cache.exists():
        print(f"Using cached {cache}")
        return cache.read_text()
    print(f"Downloading {URL} ...")
    r = requests.get(URL, timeout=60)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        text = z.read(name).decode("latin-1")
    cache.write_text(text)
    return text


def parse_vw_daily(text):
    """Find first value-weighted daily table and return a DataFrame indexed by date."""
    lines = text.splitlines()
    # First data block begins after the first 'Value Weighted' header row
    start = None
    end = None
    header_idx = None
    for i, line in enumerate(lines):
        if header_idx is None and line.strip().split(",")[0].strip().isdigit() and len(line.split(",")) > 10:
            # First row that starts with a date
            header_idx = i - 1
            start = i
        if start is not None and line.strip() == "":
            end = i
            break
    # Header has industry names (preceded by 1-2 blank columns)
    hdr = lines[header_idx].split(",")
    cols = [c.strip() for c in hdr[1:]]
    data_text = "\n".join(lines[start:end])
    df = pd.read_csv(io.StringIO(data_text), header=None, names=["date"] + cols)
    df["date"] = df["date"].astype(str).str.strip()
    df = df[df["date"].str.match(r"^\d{8}$")].copy()
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.set_index("date").astype(float)
    # Ken French codes missing as -99.99 / -999
    df = df.replace([-99.99, -999.0], np.nan)
    # Percent returns → decimal
    df = df / 100.0
    return df


def main():
    text = fetch_csv()
    df = parse_vw_daily(text)
    print(f"Loaded {df.shape[0]} days × {df.shape[1]} industries "
          f"from {df.index.min():%Y-%m-%d} to {df.index.max():%Y-%m-%d}")

    # Drop days with any missing
    before = len(df)
    df = df.dropna(axis=0, how="any")
    print(f"Dropped {before - len(df)} days with any NaN → {len(df)} rows")

    # Subtract the cross-sectional mean per day? NO — that removes the market factor.
    # We WANT the market factor in — the whole point is that the covariance has a rank-1 spike.

    # Standardize each industry to unit variance (preserves correlations / the spike).
    X = df.values
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    Xstd = (X - mu) / sd

    # Winsorize at ±5σ to kill extreme outliers (Oct 1987, Mar 2020, etc.)
    Xstd = np.clip(Xstd, -5, 5)

    # Chronological 80/20 split
    n = Xstd.shape[0]
    n_tr = int(0.8 * n)
    X_tr = Xstd[:n_tr]
    X_te = Xstd[n_tr:]
    print(f"train: {X_tr.shape}  test: {X_te.shape}")

    # Spectrum of train covariance (BBP diagnostic)
    C = (X_tr.T @ X_tr) / X_tr.shape[0]
    w, _ = np.linalg.eigh(C)
    w = w[::-1]
    print(f"  top-5 λ(C_train): {w[:5]}")
    print(f"  λ1 / λ2 = {w[0]/w[1]:.2f}   λ1 / λ_bulk_med = {w[0]/np.median(w[5:40]):.2f}")

    # Binarize each industry's returns by its own train median (sign-like)
    med = np.median(X_tr, axis=0, keepdims=True)
    B_tr = (X_tr > med).astype(np.uint8)
    B_te = (X_te > med).astype(np.uint8)

    # Spectrum of binarized covariance
    Bc = B_tr.astype(np.float64) - B_tr.mean(axis=0, keepdims=True)
    Cb = (Bc.T @ Bc) / Bc.shape[0]
    wb, _ = np.linalg.eigh(Cb)
    wb = wb[::-1]
    print(f"  top-5 λ(C_train_binary): {wb[:5]}")
    print(f"  λ1 / λ2 (binary) = {wb[0]/wb[1]:.2f}   λ1 / bulk_med = {wb[0]/np.median(wb[5:40]):.2f}")

    # Continuous-values save
    out1 = DATA / "kenfrench49_daily.h5"
    with h5py.File(out1, "w") as h:
        h["train_x"] = X_tr.astype(np.float32)   # (N_days, 49)
        h["test_x"] = X_te.astype(np.float32)
        h["mu"] = mu.astype(np.float32)
        h["sd"] = sd.astype(np.float32)
        h["industries"] = np.array(df.columns.tolist(), dtype="S")
        h["dates_train"] = df.index[:n_tr].strftime("%Y%m%d").astype(int).to_numpy()
        h["dates_test"] = df.index[n_tr:].strftime("%Y%m%d").astype(int).to_numpy()
        h.attrs["N_vis"] = X_tr.shape[1]
        h.attrs["winsor"] = 5.0
    print(f"Wrote {out1} ({out1.stat().st_size} bytes)")

    # Binary save
    out2 = DATA / "kenfrench49_binary.h5"
    with h5py.File(out2, "w") as h:
        h["train_x"] = B_tr
        h["test_x"] = B_te
        h["threshold"] = med.astype(np.float32)
        h["industries"] = np.array(df.columns.tolist(), dtype="S")
        h.attrs["N_vis"] = X_tr.shape[1]
    print(f"Wrote {out2} ({out2.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
