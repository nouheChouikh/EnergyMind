"""Plan the problems to inject: which type, where, when and how big. The answer key comes from here.

Placement rules (checked again in step 5):
  - only in 2019, alternating months: practice (Jan, Mar, May, Jul, Sep, Nov) and
    final test (Feb, Apr, Jun, Aug, Oct, Dec); 2018 stays clean
  - each problem fits inside one calendar month of its period
  - never on hours GoiEner marked as estimated
  - no two problems on the same home, with 24 free hours before and after
  - two problems that change the street energy balance never overlap on the same phase
  - sizes climb a ladder: subtle, medium, obvious
"""
import numpy as np
import pandas as pd

import config as C
from shared.helpers import PHASE_INDEX, STREAM_EVENTS, period_labels, rng

SPLITS = ("practice", "final_test")


def month_windows(idx, split):
    p = C.PERIODS[split]
    first, last = pd.Timestamp(p["first"]), pd.Timestamp(p["last"])
    out = []
    for month in pd.period_range(first, last, freq="M"):
        if month.month not in p["months"]:
            continue
        pos = np.where((idx >= max(month.start_time, first)) & (idx <= min(month.end_time, last)))[0]
        out.append((month.month, int(pos[0]), int(pos[-1])))
    return out


def plan_events(base_kwh, est, wiring, idx):
    r = rng(STREAM_EVENTS)
    n_homes, n_hours = base_kwh.shape
    ph = wiring["phase"].map(PHASE_INDEX).to_numpy()
    busy_home = np.zeros((n_homes, n_hours), dtype=bool)
    busy_phase = np.zeros((len(C.PHASES), n_hours), dtype=bool)
    hour = idx.hour.to_numpy()
    night = np.isin(hour, [2, 3, 4, 5]) & (period_labels(idx) == "learning")
    standby = np.maximum(np.quantile(base_kwh[:, night], 0.10, axis=1), 0.02)
    ctx = dict(r=r, base=base_kwh, est=est, ph=ph, idx=idx, hour=hour, standby=standby,
               busy_home=busy_home, busy_phase=busy_phase)

    events, skipped = [], []
    for split_no, split in enumerate(SPLITS):
        months = month_windows(idx, split)
        for kind, spec in C.PROBLEMS.items():
            for k in range(spec["count"][split_no]):
                ev = _place(ctx, kind, spec, k, split, months)
                if ev is None:
                    skipped.append(f"{split}/{kind}#{k + 1}")
                else:
                    events.append(ev)
    events.sort(key=lambda e: (e["t0"], e["type"]))
    for n, ev in enumerate(events, 1):
        ev["event_id"] = f"E{n:03d}"
    return events, skipped


def _place(ctx, kind, spec, k, split, months):
    r, base, est, ph = ctx["r"], ctx["base"], ctx["est"], ctx["ph"]
    n_homes, n_hours = base.shape
    level_no = k % 3
    allowed = [m for m in months if m[0] in spec.get("months", range(1, 13))]
    if not allowed:
        return None
    for _ in range(C.MAX_TRIES):
        _, m0, m1 = allowed[int(r.integers(len(allowed)))]
        if spec["ladder"] == "hours":
            dur = spec["hours"][level_no]
        else:
            dur = int(spec["hours"][int(r.integers(len(spec["hours"])))])
        if dur > m1 - m0 + 1:
            continue
        t0 = int(r.integers(m0, m1 - dur + 2))
        t1 = t0 + dur - 1
        if spec["scope"] == "street":
            rows, phases = np.arange(n_homes), list(range(len(C.PHASES)))
        elif spec["scope"] == "phase":
            p = int(r.integers(len(C.PHASES)))
            rows, phases = np.where(ph == p)[0], [p]
        else:
            h = int(r.integers(n_homes))
            rows, phases = np.array([h]), [int(ph[h])]
        b0, b1 = max(0, t0 - C.BUFFER_H), min(n_hours - 1, t1 + C.BUFFER_H)
        if ctx["busy_home"][rows, b0:b1 + 1].any():
            continue
        if spec.get("balance") and ctx["busy_phase"][phases, b0:b1 + 1].any():
            continue
        if est[rows, t0:t1 + 1].any():
            continue

        window = base[rows, t0:t1 + 1]
        level = C.LEVELS[level_no]
        size, unit, payload = dur, "hours", None
        if spec["ladder"] == "size":
            size, unit = spec["sizes"][level_no], spec["unit"]
        elif spec["ladder"] == "variant":
            level, size, unit = spec["variants"][level_no], None, ""
        if kind in ("theft", "over_reporting") and window.mean() < 0.05:
            continue
        if kind == "stuck_meter" and (t0 < 1 or est[rows, t0 - 1].any() or (window == base[rows, t0 - 1][:, None]).all()):
            continue
        if kind == "impossible_value" and level == "voltage_zero_with_load" and window.min() <= 0:
            continue
        if kind == "voltage_drift":
            size = float(size) * (1 if r.random() < 0.5 else -1)
        if kind == "holiday_absence":
            h = rows[0]
            if window.mean() < 2 * ctx["standby"][h]:
                continue
            payload = np.round(ctx["standby"][h] * np.clip(1 + r.normal(0, 0.05, dur), 0.5, 1.5), 3)
        if kind == "new_ac":
            days = ctx["idx"][t0:t1 + 1].normalize()
            uniq = days.unique()
            factor = dict(zip(uniq, r.uniform(0.4, 1.0, len(uniq))))
            shape = np.array([C.AC_PROFILE.get(int(hh), 0.0) for hh in ctx["hour"][t0:t1 + 1]])
            payload = np.round(size * shape * np.array([factor[d] for d in days]), 3)

        ctx["busy_home"][rows, b0:b1 + 1] = True
        if spec.get("balance"):
            ctx["busy_phase"][phases, b0:b1 + 1] = True
        return dict(type=kind, category=spec["category"], split=split, scope=spec["scope"],
                    rows=rows, phases=phases, t0=t0, t1=t1, hours=dur, level=level, size=size,
                    unit=unit, payload=payload, verdict=spec["verdict"],
                    fault=spec.get("fault", True), held_out=spec.get("held_out", False))
    return None


def recipe(ev, ids, standby_text=""):
    text = _recipe(ev, ids, standby_text)
    return text + " Held-out type: no rule is written for it." if ev["held_out"] else text


def _recipe(ev, ids, standby_text=""):
    home = ids[ev["rows"][0]]
    phase = C.PHASES[ev["phases"][0]]
    h, size, kind = ev["hours"], ev["size"], ev["type"]
    if kind == "outage_street":
        return f"Power cut for the whole street for {h} h: every meter and the transformer read 0."
    if kind == "missing_all":
        return f"No meter reading arrives for {h} h; the transformer keeps measuring."
    if kind == "outage_phase":
        return f"Phase {phase} loses power for {h} h: its homes and that transformer phase read 0."
    if kind in ("voltage_sag", "voltage_surge"):
        verb = "drops" if size < 0 else "rises"
        return f"Phase {phase} voltage {verb} by {abs(size)} V upstream for {h} h; its homes and the transformer move together."
    if kind == "missing_one":
        return f"Meter {home} sends nothing for {h} h; the transformer still sees its consumption."
    if kind == "theft":
        return f"Meter {home} records {100 * (1 - size):.0f}% of the real energy and current for {h} h; voltage and transformer untouched."
    if kind == "over_reporting":
        return f"Meter {home} records {100 * (1 + size):.0f}% of the real energy and current for {h} h."
    if kind == "voltage_drift":
        return f"Meter {home} voltage error grows from 0 to {size:+.0f} V over {h} h; energy and current stay correct."
    if kind == "stuck_meter":
        return f"Meter {home} repeats its last energy, voltage and current values for {h} h."
    if kind == "holiday_absence":
        return f"Not a fault: the family of {home} is away for {h / 24:.0f} days; real use falls to standby{standby_text}."
    if kind == "new_ac":
        return f"Not a fault: {home} runs a new air conditioner (up to {size} kW, 12:00-22:00) for {h / 24:.0f} days."
    if kind == "impossible_value":
        return {
            "spike": f"Meter {home} reports {C.SPIKE_KWH:.0f} kWh per hour for {h} h (impossible for a home).",
            "negative": f"Meter {home} reports negative energy for {h} h (the home has no solar panels).",
            "voltage_zero_with_load": f"Meter {home} reports 0 V while energy flows, for {h} h.",
        }[ev["level"]]
    return kind


def answers_frame(events, wiring, idx, standby=None):
    ids = wiring["home_id"].to_numpy()
    rows = []
    for ev in events:
        sb = ""
        if ev["type"] == "holiday_absence" and ev["payload"] is not None:
            sb = f" (about {float(np.mean(ev['payload'])):.2f} kWh per hour)"
        rows.append(dict(
            event_id=ev["event_id"], period=ev["split"], type=ev["type"], category=ev["category"],
            is_fault=int(ev["fault"]), held_out=int(ev["held_out"]), expected_verdict=ev["verdict"],
            scope=ev["scope"], phase="".join(C.PHASES[p] for p in ev["phases"]),
            affected_homes=";".join(ids[ev["rows"]]),
            start=idx[ev["t0"]], end=idx[ev["t1"]], hours=ev["hours"],
            level=ev["level"], size=ev["size"] if ev["size"] is not None else "", size_unit=ev["unit"],
            recipe=recipe(ev, ids, sb)))
    return pd.DataFrame(rows)
