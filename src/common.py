"""Shared helpers: time axis, random streams, file I/O, hashing, run lengths."""
import hashlib
import sys
import unicodedata

import numpy as np
import pandas as pd

import config as C

# Never crash on a console that cannot print a character (Windows).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass

HOUR = pd.Timedelta(hours=1)
PHASE_INDEX = {p: k for k, p in enumerate(C.PHASES)}
STREAM_SELECT, STREAM_STREET, STREAM_NOISE, STREAM_EVENTS = 1, 2, 3, 4


def hours():
    """The study's hourly time axis: 2018-01-01 00:00 to 2019-12-31 23:00 (17,520 hours)."""
    return pd.date_range(C.START, C.END, freq=HOUR)


def rng(stream):
    """Independent random stream for one purpose, derived from the single SEED."""
    return np.random.default_rng(np.random.SeedSequence([C.SEED, stream]))


def period_labels(idx):
    labels = np.full(len(idx), "", dtype=object)
    for name, p in C.PERIODS.items():
        inside = (idx >= pd.Timestamp(p["first"])) & (idx <= pd.Timestamp(p["last"]))
        labels[inside & np.isin(idx.month, p["months"])] = name
    return labels


def norm_text(value):
    s = "" if value is None else str(value)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().strip().lower()
    return "" if s in ("", "na", "nan", "none", "null") else s


def canonical_province(value):
    """Map a metadata province (name or INE code, any spelling) to a key of config.PROVINCES."""
    s = norm_text(value)
    if not s:
        return ""
    try:
        code = float(s)
        for key, spec in C.PROVINCES.items():
            if code in spec["codes"]:
                return key
        return s
    except ValueError:
        pass
    for key, spec in C.PROVINCES.items():
        if any(word in s for word in spec["words"]):
            return key
    return s


def province_label(key):
    return C.PROVINCES[key]["label"] if key in C.PROVINCES else (key or "unknown")


def longest_run(mask):
    """Length of the longest run of True values."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return 0
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return int((np.where(edges == -1)[0] - np.where(edges == 1)[0]).max())


def equal_runs(x, ok=None):
    """Runs of identical consecutive values (2 or more readings). Returns (lengths, start positions).
    If ok is given, only hours where ok is True can belong to a run (e.g. measured, not estimated)."""
    x = np.asarray(x)
    same = x[1:] == x[:-1]
    if ok is not None:
        ok = np.asarray(ok, dtype=bool)
        same = same & ok[1:] & ok[:-1]
    same = np.concatenate([[False], same])
    edges = np.diff(np.concatenate([[0], same.astype(np.int8), [0]]))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    return ends - starts + 1, starts - 1


def save_parquet(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="zstd")


def write_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n", encoding="utf-8")


def csv_text(df):
    return df.to_csv(index=False, lineterminator="\n")


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def content_hash(df):
    """Hash of the values (not of the file bytes), so it does not depend on library versions."""
    h = hashlib.sha256()
    for col in df.columns:
        h.update(str(col).encode())
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            h.update(s.to_numpy(dtype="datetime64[ns]").astype("int64").tobytes())
        elif pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
            arr = np.round(s.to_numpy(dtype="float64", na_value=np.nan), 6) + 0.0
            h.update(np.nan_to_num(arr, nan=-999999.0).tobytes())
        else:
            h.update("\x1f".join(s.astype(str).tolist()).encode())
    return h.hexdigest()[:16]


def to_long(home_ids, idx, kwh, volts, amps, est):
    """Homes x hours arrays -> one row per home and hour."""
    n_homes, n_hours = kwh.shape
    return pd.DataFrame({
        "home_id": np.repeat(np.asarray(home_ids, dtype=object), n_hours),
        "timestamp": np.tile(idx.to_numpy(), n_homes),
        "kwh": kwh.reshape(-1),
        "voltage_v": volts.reshape(-1),
        "current_a": amps.reshape(-1),
        "estimated": est.reshape(-1).astype("int8"),
    })


def load_source():
    """data/source_subset.parquet -> (home_ids, hours, kwh[home, hour], estimated[home, hour])."""
    path = C.DATA_DIR / "source_subset.parquet"
    if not path.exists():
        sys.exit(f"Missing {path}. Run step 1 first: python src/run_all.py --from-raw")
    df = pd.read_parquet(path)
    df = df.sort_values(["home_id", "timestamp"], kind="mergesort").reset_index(drop=True)
    idx = hours()
    home_ids = np.array(sorted(df["home_id"].astype(str).unique()), dtype=object)
    n_homes, n_hours = len(home_ids), len(idx)
    first = df["timestamp"].to_numpy(dtype="datetime64[ns]")[:n_hours]
    if len(df) != n_homes * n_hours or not np.array_equal(first, idx.to_numpy(dtype="datetime64[ns]")):
        sys.exit("data/source_subset.parquet does not hold one row per home and hour for 2018-2019.")
    kwh = df["kwh"].to_numpy(dtype="float64").reshape(n_homes, n_hours)
    est = df["estimated"].to_numpy(dtype="int8").reshape(n_homes, n_hours)
    return home_ids, idx, kwh, est
