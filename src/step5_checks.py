"""Step 5 - the dataset checks (proposal section 4.5) and the data card.

Output: data/checks_report.txt   PASS / FAIL / INFO for every check, with the numbers
        data/DATA_CARD.md        what the dataset is, how it was made, its limits
"""
import platform
import sys

import numpy as np
import pandas as pd

import config as C
from common import PHASE_INDEX, content_hash, csv_text, equal_runs, longest_run, period_labels, write_text
from step3_simulate_normal import build_normal
from step4_inject_problems import build_final


class Report:
    def __init__(self):
        self.lines, self.results = [], []

    def add(self, tag, name, detail=""):
        self.results.append((tag, name))
        self.lines.append(f"[{tag}] {name}")
        self.lines += [f"       {d}" for d in str(detail).splitlines() if d.strip()]
        self.lines.append("")

    def check(self, name, ok, detail=""):
        self.add("PASS" if ok else "FAIL", name, detail)


def run_checks(normal, final):
    rep = Report()
    idx, est, wiring = normal["idx"], normal["est"], normal["wiring"]
    nout, fout, events = normal["out"], final["out"], final["events"]
    ph = wiring["phase"].map(PHASE_INDEX).to_numpy()
    n_homes, n_hours = est.shape
    labels = period_labels(idx)
    learn = labels == "learning"

    # 1. reproducibility
    files = {"normal/readings_normal.parquet": normal["readings"],
             "normal/transformer_normal.parquet": normal["transformer"],
             "readings_2018.parquet": final["readings_2018"],
             "readings_2019.parquet": final["readings_2019"],
             "transformer.parquet": final["transformer"]}
    bad, hashes = [], []
    for rel, df in files.items():
        h_file = content_hash(pd.read_parquet(C.DATA_DIR / rel))
        hashes.append(f"{rel:<36} {h_file}")
        if h_file != content_hash(df):
            bad.append(rel)
    for rel, df in {"wiring.csv": wiring, "answers.csv": final["answers"]}.items():
        if (C.DATA_DIR / rel).read_bytes().decode("utf-8").replace("\r\n", "\n") != csv_text(df):
            bad.append(rel)
    rep.check("Rebuilding from the same seed gives exactly the same dataset",
              not bad, f"seed {C.SEED}; content hashes:\n" + "\n".join(hashes)
              + (f"\nDIFFERENT: {', '.join(bad)}" if bad else ""))

    # 2. the learning year is clean
    n18 = normal["readings"][normal["readings"]["timestamp"].dt.year == 2018].reset_index(drop=True)
    tx18 = learn
    same18 = (content_hash(n18) == content_hash(final["readings_2018"])
              and np.array_equal(nout["tx_e"][:, tx18], fout["tx_e"][:, tx18]))
    rep.check("2018 (learning) is identical to the normal simulation: no problem injected", same18)

    # 3. all problems placed
    planned = sum(sum(spec["count"]) for spec in C.PROBLEMS.values())
    rep.check("Every planned problem was placed", not final["skipped"],
              f"{len(events)} of {planned} placed" + (f"; not placed: {', '.join(final['skipped'])}" if final["skipped"] else ""))

    # 4. placement rules
    issues = []
    for ev in events:
        s, e, r = ev["t0"], ev["t1"] + 1, ev["rows"]
        if est[r, s:e].any():
            issues.append(f"{ev['event_id']} touches an estimated hour")
        if idx[s].month != idx[e - 1].month:
            issues.append(f"{ev['event_id']} crosses a month boundary")
        if not (labels[s:e] == ev["split"]).all():
            issues.append(f"{ev['event_id']} leaves its period")
        if ev["held_out"] and ev["split"] != "final_test":
            issues.append(f"{ev['event_id']} held-out type outside the final test")
    for i, a in enumerate(events):
        for b in events[i + 1:]:
            close = a["t0"] <= b["t1"] + C.BUFFER_H and b["t0"] <= a["t1"] + C.BUFFER_H
            if not close:
                continue
            if set(a["rows"].tolist()) & set(b["rows"].tolist()):
                issues.append(f"{a['event_id']} and {b['event_id']} share a home without {C.BUFFER_H} free hours")
            spec_a, spec_b = C.PROBLEMS[a["type"]], C.PROBLEMS[b["type"]]
            if spec_a.get("balance") and spec_b.get("balance") and set(a["phases"]) & set(b["phases"]):
                issues.append(f"{a['event_id']} and {b['event_id']} both change the balance of one phase")
    rep.check("Placement rules: no estimated hours, one month, own period, no overlap",
              not issues, "\n".join(issues[:20]) if issues else f"{len(events)} problems verified")

    # 5. every problem changed the readings
    unchanged = []
    for ev in events:
        s, e, r = ev["t0"], ev["t1"] + 1, ev["rows"]
        changed = any(not np.array_equal(nout[k][r, s:e], fout[k][r, s:e], equal_nan=True) for k in ("kwh", "v", "i"))
        if not changed:
            unchanged.append(ev["event_id"])
    rep.check("Every injected problem changed the readings", not unchanged,
              f"unchanged: {', '.join(unchanged)}" if unchanged else f"{len(events)} of {len(events)} changed")

    # 6. street balance on clean data
    meters = np.stack([nout["kwh"][ph == p].sum(axis=0) for p in range(len(C.PHASES))])
    tx = nout["tx_e"][:, learn]
    gap_pct = 100 * (tx - meters[:, learn]) / np.where(tx > 0, tx, np.nan)
    rows = [f"phase {p}: mean {np.nanmean(gap_pct[k]):.2f}%, 5-95% of hours {np.nanpercentile(gap_pct[k], 5):.2f}% to "
            f"{np.nanpercentile(gap_pct[k], 95):.2f}%" for k, p in enumerate(C.PHASES)]
    yearly = 100 * nout["loss_kwh"][:, learn].sum() / tx.sum()
    mean_gap = float(np.nanmean(gap_pct))
    rep.check("Street balance on clean 2018 data: transformer = homes + cable losses + meter noise",
              0 < mean_gap < 5,
              f"cable losses over the year: {yearly:.2f}% of energy delivered\n" + "\n".join(rows)
              + "\nThis gap is the noise floor any theft detection must beat.")

    # 7. voltages
    on = nout["v"] > 0
    v = nout["v"][on]
    share = float(((v >= 207) & (v <= 253)).mean())
    far = wiring["distance_m"].to_numpy() == wiring["distance_m"].max()
    rep.check("Normal voltages stay inside 230 V +/- 10 % (EN 50160)", share >= 0.99,
              f"{share:.2%} of home-hours inside; range {v.min():.1f} to {v.max():.1f} V; "
              f"mean at the end of the cables {nout['v'][far].mean():.1f} V")

    # 8. honest repeats (sets the threshold of the stuck-meter rule)
    hour = idx.hour.to_numpy()[learn]
    lengths, night = [], []
    for h in range(n_homes):
        L, start = equal_runs(nout["kwh"][h, learn], est[h, learn] == 0)
        lengths.append(L)
        night.append(hour[start] < 6)
    L, night = np.concatenate(lengths), np.concatenate(night)
    table = ["run length   runs   per home per year   share starting 00:00-05:59"]
    for lo, hi in ((2, 2), (3, 3), (4, 6), (7, 12), (13, 24), (25, 10**9)):
        m = (L >= lo) & (L <= hi)
        name = f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10**9 else f"{lo}+")
        pct = f"{night[m].mean():.0%}" if m.any() else "-"
        table.append(f"{name:<12} {int(m.sum()):>6}   {m.sum() / n_homes:>17.1f}   {pct:>26}")
    table.append(f"longest energy-only repeat: {int(L.max()) if len(L) else 0} readings")
    whole, whole_longest = 0, 1
    for h in range(n_homes):
        e, v, a = nout["kwh"][h, learn], nout["v"][h, learn], nout["i"][h, learn]
        m = est[h, learn] == 0
        same = (e[1:] == e[:-1]) & (v[1:] == v[:-1]) & (a[1:] == a[:-1]) & m[1:] & m[:-1]
        whole += int(same.sum())
        if same.any():
            whole_longest = max(whole_longest, longest_run(same) + 1)
    table.append(f"whole reading (energy, voltage and current all identical to the previous hour): "
                 f"{whole} times in the year; longest run {whole_longest} readings")
    rep.add("INFO", "Honest meters repeating a value by coincidence (2018)",
            "\n".join(table)
            + "\nEnergy alone can stay flat for hours in the source data (probably estimates the distributor did not flag)."
            + "\nA working meter's voltage always moves, so the stuck-meter rule checks the whole reading, not energy alone.")

    # 9. zeros and estimated hours in the chosen homes
    zero = (normal["base"] <= 0) & (est == 0)
    base = normal["base"]
    flat = (base[:, 1:] == base[:, :-1]) & (est[:, 1:] == 0) & (est[:, :-1] == 0)
    longest_flat = max((longest_run(f) + 1 if f.any() else 1) for f in flat)
    rep.check("No unexplained zeros or flat stretches in the chosen homes",
              max(longest_run(z) for z in zero) <= C.MAX_ZERO_RUN_H and longest_flat <= C.MAX_FLAT_READINGS,
              f"{int(zero.sum())} measured zero hours in total; longest run {max(longest_run(z) for z in zero)} h\n"
              f"longest run of identical measured readings: {longest_flat} (limit {C.MAX_FLAT_READINGS})")
    rep.add("INFO", "Hours estimated by GoiEner (flag `estimated`)",
            f"{est.mean():.2%} of all home-hours; per home max {est.mean(axis=1).max():.2%}")

    # 10. file sizes
    sizes = {p.relative_to(C.DATA_DIR).as_posix(): p.stat().st_size / 1e6 for p in C.DATA_DIR.rglob("*") if p.is_file()}
    big = {k: v for k, v in sizes.items() if v > 50}
    rep.check("Every data file is under 50 MB (GitHub limit 100 MB)", not big,
              "\n".join(f"{k:<36} {v:6.1f} MB" for k, v in sorted(sizes.items())))
    return rep


def counts_table(answers):
    t = answers.groupby(["type", "period"]).size().unstack(fill_value=0)
    t = t.reindex(index=list(C.PROBLEMS), columns=["practice", "final_test"], fill_value=0)
    return t


def data_card(normal, final, rep):
    homes = pd.read_csv(C.DATA_DIR / "homes.csv", dtype=str, keep_default_na=False)
    log = pd.read_csv(C.DATA_DIR / "selection_log.csv")
    ctx = pd.read_csv(C.DATA_DIR / "context.csv")
    wiring, answers = normal["wiring"], final["answers"]
    zips = homes["zip_code"].replace("", np.nan)
    temp_ok = ctx["temperature_c"].notna().any()
    weather = (pd.read_csv(C.WEATHER_CACHE).iloc[0] if C.WEATHER_CACHE.exists() else None)
    t = counts_table(answers)
    n_hol = int(ctx.loc[ctx["is_holiday"] == 1, "timestamp"].str.slice(0, 10).nunique())
    files = [
        ("readings_2018.parquet", "learning year: one row per home and hour - kWh, voltage_v, current_a, estimated"),
        ("readings_2019.parquet", "practice + final test, same columns, with the injected problems"),
        ("transformer.parquet", "reference meter: energy and voltage per phase each hour, plus the street total"),
        ("wiring.csv", "cable, phase, position and distance of each home"),
        ("context.csv", "temperature, weekend, holiday and period of each hour"),
        ("answers.csv", "every injected problem: type, homes, start, end, size, expected verdict, recipe"),
        ("RECIPES.md", "what each problem type changes, sizes and counts"),
        ("normal/readings_normal.parquet", "the same street with no problem at all (simulated normal values)"),
        ("normal/transformer_normal.parquet", "the transformer with no problem at all"),
        ("source_subset.parquet", "original GoiEner readings of the 75 homes, 2018-2019 (rebuild without the 791 MB download)"),
        ("homes.csv", "the chosen homes: GoiEner id, province, postcode, contracted power, yearly use"),
        ("selection_log.csv", "how many homes each cleaning rule removed"),
        ("weather_openmeteo.csv", "raw hourly temperature download (UTC)"),
        ("checks_report.txt", "results of the checks below"),
    ]
    rows = []
    for name, what in files:
        p = C.DATA_DIR / name
        if p.exists():
            rows.append(f"| `{name}` | {p.stat().st_size / 1e6:.1f} MB | {what} |")
    period_rows = []
    month_names = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
    for name, p in C.PERIODS.items():
        n = int((answers["period"] == name).sum()) if name != "learning" else 0
        months = "all months" if len(p["months"]) == 12 else ", ".join(month_names[m - 1] for m in p["months"])
        period_rows.append(f"| {name} | {p['first'][:4]}: {months} | {n} |")
    lines = [
        "# Data card - EnergyMind street dataset (Phase 1)",
        "",
        f"A simulated low-voltage street built from the **real hourly consumption of {len(homes)} Spanish homes** "
        "(GoiEner smart meter data, 2018-2019). Each home has a meter that reports energy, voltage and current every hour; "
        "a reference meter at the transformer measures energy and voltage on each of the 3 phases. "
        f"In 2019, {len(answers)} problems are injected following written recipes, and every one of them is listed in `answers.csv`.",
        "",
        "Everything here is rebuilt by `python src/run_all.py` (fixed seed "
        f"{C.SEED}); every invented number is in `src/config.py`.",
        "",
        "## Source",
        "",
        "- **Energy:** GoiEner smart meters data, `imp-pre` subset (hourly kWh before March 2020, with an `imputed` flag). "
        "Quesada, Borges, Astigarraga & Merveille, Zenodo, doi:10.5281/zenodo.7362094; data paper: Scientific Data 11, 59 (2024), "
        "doi:10.1038/s41597-023-02846-0. Timestamps are Spanish wall-clock time; the authors filled the skipped spring hour and averaged the repeated autumn hour.",
        "- **Temperature:** " + (f"Open-Meteo historical weather archive (ERA5 reanalysis), hourly 2 m temperature at "
                                 f"{weather['latitude']}, {weather['longitude']} (capital of the province), CC BY 4.0."
                                 if weather is not None and temp_ok else "not available in this build (download failed); `temperature_c` is empty."),
        f"- **Holidays:** Spanish national and regional calendar from the `holidays` Python package ({n_hol} holidays in 2018-2019).",
        "",
        "## How the homes were chosen (cleaning)",
        "",
        "| Stage | Rule | Homes left |",
        "|---|---|---|",
        *[f"| {r.stage} | {r.rule} | {r.homes_left} |" for r in log.itertuples()],
        "",
        f"Result: {len(homes)} homes from **{homes['province'].mode().iat[0]}**, {zips.nunique()} postcodes "
        f"({int(zips.isna().sum())} homes have no postcode: GoiEner removes postcodes of towns under 50,000 people). "
        "Short gaps (at most 48 hours) were filled by linear interpolation and flagged as estimated.",
        "",
        "## The street",
        "",
        f"- {C.N_CABLES} cables leave the transformer; homes are spaced {C.SPACING_M:.0f} m apart "
        f"(farthest home {wiring['distance_m'].max():.0f} m away).",
        "- Phases rotate A, B, C along each cable, starting on a different phase per cable: "
        + ", ".join(f"{p} = {int((wiring['phase'] == p).sum())} homes" for p in C.PHASES) + ".",
        "- Homes are placed on cables at random (fixed seed). They were not neighbours in reality.",
        "",
        "## Simulated values (formulas)",
        "",
        "| Value | How it is computed |",
        "|---|---|",
        f"| Energy (kWh) | the real GoiEner reading of the home |",
        f"| Current (A) | energy as average power / ({C.NOMINAL_V:.0f} V x {C.POWER_FACTOR}), plus {C.METER_I_NOISE_REL_SD:.1%} meter noise |",
        f"| Transformer voltage (V) | {C.SOURCE_V:.0f} V minus up to {C.SOURCE_V_LOAD_DROP:.0f} V at the street's peak load, plus {C.TX_V_NOISE_SD} V noise |",
        f"| Home voltage (V) | transformer voltage minus the cable drop: R x sum over homes on the same cable and phase of I x min(their distance, this distance), R = {C.R_OHM_PER_M * 1000:.2f} ohm/km (phase + neutral); plus {C.METER_V_NOISE_SD} V noise |",
        "| Cable losses (kWh) | sum over cable segments of R x length x (current through the segment)^2 |",
        f"| Transformer energy (kWh) | per phase: true energy of its homes + cable losses, plus {C.TX_E_NOISE_REL_SD:.2%} reference-meter noise |",
        "",
        "## Periods",
        "",
        "| Period | Months | Problems injected |",
        "|---|---|---|",
        *period_rows,
        "",
        "## Injected problems",
        "",
        "| Type | Practice | Final test |",
        "|---|---|---|",
        *[f"| `{k}` | {int(t.at[k, 'practice'])} | {int(t.at[k, 'final_test'])} |" for k in t.index],
        "",
        "Recipes and rules: `RECIPES.md`. The answer key: `answers.csv`.",
        "",
        "## Files",
        "",
        "| File | Size | Content |",
        "|---|---|---|",
        *rows,
        "",
        "## Checks (full numbers in `checks_report.txt`)",
        "",
        *[f"- **{tag}** {name}" for tag, name in rep.results],
        "",
        "## Known limits",
        "",
        "- Voltage, current, cable losses and the transformer meter are simulated with the simple formulas above, not measured. "
        "Results on them show how a method behaves under these assumptions, not on a real network.",
        "- The smallest theft that can be detected depends on the assumed meter noise and cable losses; Phase 2 reports results at several noise levels.",
        "- The energy readings are real, but the homes were not neighbours: they come from one province and are assembled into one street.",
        "- Injected problems follow clean recipes; real faults are messier and can overlap.",
        "- One weather point for the whole province.",
        "",
        "## Licences and credit",
        "",
        "- Data derived from GoiEner smart meters data (University of Deusto and GoiEner), published here under CC BY-SA 4.0. "
        "Zenodo lists the source under CC BY 4.0 (its description also says CC-BY-SA); CC BY-SA 4.0 respects both.",
        "- Temperatures: Open-Meteo (open-meteo.com), CC BY 4.0, from Copernicus ERA5.",
        "- Code: MIT licence (see `LICENSE`).",
        "",
    ]
    return "\n".join(lines)


def main():
    print("  rebuilding steps 3 and 4 in memory to test reproducibility ...")
    normal, final = build_normal(), build_final()
    rep = run_checks(normal, final)
    t = counts_table(final["answers"])
    head = [
        "EnergyMind - Phase 1 dataset checks (proposal section 4.5)",
        "Generated by src/step5_checks.py",
        f"Python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}, seed {C.SEED}",
        "",
        f"Problems injected by type (practice / final test): "
        + ", ".join(f"{k} {int(t.at[k, 'practice'])}/{int(t.at[k, 'final_test'])}" for k in t.index),
        "",
    ]
    text = "\n".join(head + rep.lines)
    write_text(C.DATA_DIR / "checks_report.txt", text)
    write_text(C.DATA_DIR / "DATA_CARD.md", data_card(normal, final, rep))
    print("\n".join("    " + line for line in rep.lines if line.startswith("[")))
    failed = [name for tag, name in rep.results if tag == "FAIL"]
    if failed:
        print(f"  {len(failed)} check(s) FAILED - see data/checks_report.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()
