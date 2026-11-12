"""Step 1 - clean the GoiEner data and choose the homes of our street.

Input : dataset/data/sources/goiener_download/imp-pre.tzst    12,149 CSV files (timestamp, kWh, imputed), hourly, before March 2020
        dataset/data/sources/goiener_download/metadata.csv    one row per GoiEner series (activity code, tariff, province, ...)
Output: dataset/data/sources/selected_homes_readings.parquet  original hourly readings of the chosen homes, 2018-2019
        dataset/data/sources/selected_homes.csv               one row per chosen home
        dataset/data/sources/homes_selection_log.csv          how many homes are left after each cleaning rule

Cleaning rules (thresholds in config.py), in this order:
  metadata  household (CNAE 98xx), no self-consumption (no solar), household tariff 2.x,
            series dates cover 2018-2019
  readings  complete hourly 2018-2019 (gaps up to 48 h are filled and flagged as estimated),
            few estimated hours, plausible yearly use, no sharp 2018 -> 2019 change,
            no unexplained zeros
Then the province with the most candidate homes is kept (one weather, one holiday calendar)
and N_HOMES clean homes are drawn at random with the fixed seed. The draw does not depend
on the order of the files inside the archive.
"""
import heapq
import io
import sys
import tarfile
from collections import Counter

import numpy as np
import pandas as pd

import config as C
from shared.helpers import (STREAM_SELECT, canonical_province, hours, longest_run, norm_text,
                            province_label, rng, save_parquet, write_csv)

IDX = hours()
T0, T1 = IDX[0], IDX[-1]
HOURS_2018 = int((IDX.year == 2018).sum())
IMPUTED = {"1": 1, "0": 0, "1.0": 1, "0.0": 0, "true": 1, "false": 0, "t": 1, "f": 0}
READING_RULES = [
    ("incomplete", f"complete hourly 2018-2019 (gaps up to {C.MAX_GAP_HOURS} h filled, flagged estimated)"),
    ("estimated", f"estimated hours at most {C.MAX_ESTIMATED_SHARE:.0%}"),
    ("annual_kwh", f"yearly use {C.ANNUAL_KWH_MIN}-{C.ANNUAL_KWH_MAX} kWh in 2018 and in 2019"),
    ("yoy", f"2019 vs 2018 change at most {C.MAX_YOY_CHANGE:.0%}"),
    ("zeros", f"no unexplained zeros (runs up to {C.MAX_ZERO_RUN_H} h, at most {C.MAX_ZERO_SHARE:.0%} of hours)"),
    ("flat", f"no flat stretches (at most {C.MAX_FLAT_READINGS} identical measured readings in a row)"),
]


def to_time(series):
    s = series.astype(str).str.strip().str.slice(0, 19).str.replace("T", " ", regex=False)
    return pd.to_datetime(s, format="ISO8601", errors="coerce")


def read_metadata():
    md = pd.read_csv(C.METADATA, dtype=str, keep_default_na=False)
    md.columns = [c.strip().lower() for c in md.columns]
    for col in ("user", "cnae", "self_consumption_type", "contracted_tariff", "province",
                "municipality", "zip_code", "p1", "start_date", "end_date"):
        if col not in md.columns:
            md[col] = ""
        md[col] = md[col].astype(str).str.strip()
    return md.drop_duplicates("user").reset_index(drop=True)


def metadata_filter(md, log):
    log.append(("metadata", "all series in metadata.csv", len(md)))
    md = md[md["cnae"].str.startswith("98")]
    log.append(("metadata", "household (CNAE 98xx)", len(md)))
    md = md[md["self_consumption_type"].map(norm_text).isin(["", "0", "00", "0.0"])]
    log.append(("metadata", "no self-consumption (no solar panels)", len(md)))
    tariff = md["contracted_tariff"].map(norm_text)
    md = md[((tariff == "") | tariff.str.startswith("2")).to_numpy()]
    log.append(("metadata", "household tariff (2.x, up to 15 kW)", len(md)))
    day = pd.Timedelta(days=1)
    start, end = to_time(md["start_date"]), to_time(md["end_date"])
    ok = (start.isna() | (start <= T0 + day)) & (end.isna() | (end >= T1 - day))
    md = md[ok.to_numpy()]
    log.append(("metadata", "series dates cover 2018-2019", len(md)))
    return md.reset_index(drop=True)


def parse_csv(raw):
    try:
        df = pd.read_csv(io.BytesIO(raw), dtype=str)
    except Exception:
        return None
    cols = [str(c).strip().lower() for c in df.columns]
    if "timestamp" in cols:
        df.columns = cols
    else:  # no header row: timestamp, kWh[, imputed]
        df = pd.read_csv(io.BytesIO(raw), dtype=str, header=None)
        names = ["timestamp", "kwh", "imputed"]
        df.columns = names[: df.shape[1]] + [f"extra{k}" for k in range(df.shape[1] - len(names))]
    if "kwh" not in df.columns:
        return None
    years = df["timestamp"].astype(str).str.strip().str.slice(0, 4)
    in_window = years.isin(["2018", "2019"])
    if in_window.any():  # ISO timestamps: parse only the two study years (much faster)
        df = df[in_window.to_numpy()]
    est = df["imputed"].str.strip().str.lower().map(IMPUTED).fillna(0) if "imputed" in df.columns else 0
    out = pd.DataFrame({"ts": to_time(df["timestamp"]),
                        "kwh": pd.to_numeric(df["kwh"], errors="coerce"),
                        "est": est})
    return out.dropna(subset=["ts"])


def evaluate(raw):
    """Apply the reading rules to one CSV. Returns (failed rule or None, result)."""
    df = parse_csv(raw)
    if df is None or df.empty:
        return "incomplete", None
    df = df[(df["ts"] >= T0) & (df["ts"] <= T1)]
    g = df.groupby("ts").agg(kwh=("kwh", "mean"), est=("est", "max")).reindex(IDX)
    gap = g["kwh"].isna().to_numpy()
    if gap.sum() > C.MAX_GAP_HOURS:
        return "incomplete", None
    kwh = g["kwh"].interpolate(limit_direction="both").to_numpy(dtype=float)
    if np.isnan(kwh).any():
        return "incomplete", None
    est = np.where(gap, 1, g["est"].fillna(1).to_numpy(dtype=float)).astype(np.int8)
    if est.mean() > C.MAX_ESTIMATED_SHARE:
        return "estimated", None
    y18, y19 = kwh[:HOURS_2018].sum(), kwh[HOURS_2018:].sum()
    if not (C.ANNUAL_KWH_MIN <= y18 <= C.ANNUAL_KWH_MAX and C.ANNUAL_KWH_MIN <= y19 <= C.ANNUAL_KWH_MAX):
        return "annual_kwh", None
    if abs(y19 / y18 - 1) > C.MAX_YOY_CHANGE:
        return "yoy", None
    zero = (kwh <= 0) & (est == 0)
    if longest_run(zero) > C.MAX_ZERO_RUN_H or zero.mean() > C.MAX_ZERO_SHARE:
        return "zeros", None
    flat = (kwh[1:] == kwh[:-1]) & (est[1:] == 0) & (est[:-1] == 0)
    if flat.any() and longest_run(flat) + 1 > C.MAX_FLAT_READINGS:
        return "flat", None
    return None, dict(kwh=np.round(kwh, 3), est=est, kwh_2018=round(float(y18), 1),
                      kwh_2019=round(float(y19), 1), estimated_share=round(float(est.mean()), 4))


def archive_members():
    """Yield (user id, tar, member) for every CSV in the GoiEner archive, streaming (no extraction to disk)."""
    import zstandard
    with open(C.ARCHIVE, "rb") as fh:
        reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(fh, read_size=1 << 20)
        with tarfile.open(fileobj=reader, mode="r|", bufsize=1 << 20) as tar:
            for member in tar:
                name = member.name.replace("\\", "/").rsplit("/", 1)[-1]
                if member.isfile() and name.lower().endswith(".csv"):
                    yield name[:-4], tar, member


def keep_best(heap, item, n):
    """Keep the n items with the smallest random priority (item[0] is minus the priority)."""
    if len(heap) < n:
        heapq.heappush(heap, item)
    elif item[0] > heap[0][0]:
        heapq.heapreplace(heap, item)


def main():
    for path in (C.ARCHIVE, C.METADATA):
        if not path.exists():
            sys.exit(f"Missing {path}\nDownload {path.name} from https://zenodo.org/records/7362094 "
                     "into the dataset/data/sources/goiener_download/ folder.")
    log = []
    raw_md = read_metadata()
    md = metadata_filter(raw_md, log)
    if md.empty:
        for stage, rule, left in log:
            print(f"    {left:>6}  {rule}")
        sys.exit("No home passes the metadata rules. Columns found in metadata.csv: " + ", ".join(raw_md.columns))
    md["province_key"] = md["province"].map(canonical_province)
    info = md.set_index("user")
    users = sorted(info.index)
    priority = dict(zip(users, rng(STREAM_SELECT).random(len(users))))
    counts = Counter(k for k in md["province_key"] if k)
    target = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] if counts else None
    print(f"  {len(users)} candidate homes after the metadata rules; "
          f"province with most candidates: {province_label(target)}")

    fails, clean = Counter(), []
    best_target, best_any = [], []
    n_files = n_found = 0
    print(f"  reading {C.ARCHIVE.name} (5 to 15 minutes; nothing is extracted to disk) ...", flush=True)
    for user, tar, member in archive_members():
        n_files += 1
        if n_files == 1:
            head = tar.extractfile(member).read()
            print("  first file in the archive:", member.name)
            for line in head.decode("utf-8", "replace").splitlines()[:3]:
                print("    " + line)
            raw = head
        else:
            raw = None
        if n_files % 1000 == 0:
            print(f"  {n_files} files read, {n_found} candidates checked, {len(clean)} clean so far", flush=True)
        if user not in priority:
            continue
        n_found += 1
        reason, res = evaluate(raw if raw is not None else tar.extractfile(member).read())
        if reason:
            fails[reason] += 1
            continue
        key = info.at[user, "province_key"]
        clean.append(dict(user=user, province_key=key))
        item = (-priority[user], user, res)
        keep_best(best_any, item, C.N_HOMES)
        if key == target:
            keep_best(best_target, item, C.N_HOMES)
    print(f"  done: {n_files} files read, {n_found} candidates checked, {len(clean)} clean homes")
    if n_files == 0:
        sys.exit("The archive contains no CSV file. Is dataset/data/sources/goiener_download/imp-pre.tzst "
                 "complete (791,622,311 bytes)?")
    if n_found == 0:
        sys.exit("No candidate home was found in the archive. Do metadata.csv and imp-pre.tzst come from the same record?")

    log.append(("readings", "series found in imp-pre.tzst", n_found))
    left = n_found
    for code, rule in READING_RULES:
        left -= fails[code]
        log.append(("readings", rule, left))
    in_target = sum(1 for c in clean if c["province_key"] == target)
    if target and len(best_target) >= C.N_HOMES:
        chosen = best_target
        log.append(("selection", f"in {province_label(target)}, the province with most candidates", in_target))
    else:
        chosen = best_any
        print(f"  WARNING: only {in_target} clean homes in {province_label(target)}; drawing from all provinces.")
        log.append(("selection", "all provinces kept (too few clean homes in one province)", len(clean)))
    log.append(("selection", f"random draw with seed {C.SEED}", len(chosen)))
    if len(chosen) < C.N_HOMES:
        print(f"  WARNING: only {len(chosen)} homes pass the rules (wanted {C.N_HOMES}). Relax the rules in config.py.")

    chosen = sorted(chosen, key=lambda item: item[1])
    home_ids = [f"H{k:03d}" for k in range(1, len(chosen) + 1)]
    homes, parts = [], []
    for hid, (_, user, res) in zip(home_ids, chosen):
        row = info.loc[user]
        homes.append(dict(
            home_id=hid, goiener_user=user, province_key=row["province_key"],
            province=province_label(row["province_key"]), municipality=row["municipality"],
            zip_code=norm_text(row["zip_code"]).replace(".0", ""), contracted_tariff=row["contracted_tariff"],
            contracted_kw=pd.to_numeric(row["p1"].replace(",", "."), errors="coerce"),
            kwh_2018=res["kwh_2018"], kwh_2019=res["kwh_2019"], estimated_share=res["estimated_share"]))
        parts.append(pd.DataFrame({"home_id": hid, "timestamp": IDX, "kwh": res["kwh"], "estimated": res["est"]}))

    save_parquet(pd.concat(parts, ignore_index=True), C.SELECTED_HOMES_READINGS)
    homes = pd.DataFrame(homes)
    write_csv(homes, C.SELECTED_HOMES)
    write_csv(pd.DataFrame(log, columns=["stage", "rule", "homes_left"]), C.HOMES_SELECTION_LOG)

    zips = homes["zip_code"].replace("", np.nan)
    print(f"  {len(homes)} homes from {homes['province'].mode().iat[0]}, "
          f"{zips.nunique()} postcodes ({zips.isna().sum()} homes have no postcode: towns under 50,000 people)")
    for stage, rule, left in log:
        print(f"    {left:>6}  {rule}")


if __name__ == "__main__":
    main()
