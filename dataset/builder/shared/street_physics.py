"""The street: wiring, simple physics, measurement noise, and how each problem changes readings.

The physics is deliberately simple and fully stated (numbers in config.py):
  current of a home     I = P / (230 V x 0.95), P = the home's hourly energy as average power
  transformer voltage   V0 = 233 V - 3 V x (street load / street peak load)  (+ upstream sag or surge)
  voltage at a home     V = V0 - R x sum_j I_j x min(x_j, x_home)
                        (j = homes on the same cable and phase, x = distance from the transformer)
  cable losses          sum over cable segments of R x length x (current through the segment)^2
  transformer energy    per phase = sum of the true energy of its homes + cable losses
Home meters add small noise to voltage and current; energy is the real GoiEner reading.
The reference meter at the transformer adds its own small noise.
"""
import numpy as np
import pandas as pd

import config as C
from shared.helpers import PHASE_INDEX, STREAM_NOISE, STREAM_STREET, rng


def make_wiring(homes):
    """3 cables x 25 homes; phases rotate A, B, C along each cable (25 homes per phase)."""
    homes = homes.sort_values("home_id").reset_index(drop=True)
    n = len(homes)
    per_cable = int(np.ceil(n / C.N_CABLES))
    order = rng(STREAM_STREET).permutation(n)
    cable, pos = np.empty(n, dtype=int), np.empty(n, dtype=int)
    for k, row in enumerate(order):
        cable[row], pos[row] = k // per_cable + 1, k % per_cable + 1
    phase = [C.PHASES[(p - 1 + c - 1) % len(C.PHASES)] for c, p in zip(cable, pos)]
    return pd.DataFrame({
        "home_id": homes["home_id"], "cable": cable, "phase": phase, "position": pos,
        "distance_m": np.round(pos * C.SPACING_M, 1),
        "province": homes["province"], "zip_code": homes["zip_code"],
        "contracted_kw": homes["contracted_kw"], "goiener_user": homes["goiener_user"],
    })


def make_noise(n_homes, n_hours):
    """Measurement noise, drawn once, so normal and final datasets share it exactly."""
    r = rng(STREAM_NOISE)
    n_ph = len(C.PHASES)
    return dict(
        v=r.normal(0.0, C.METER_V_NOISE_SD, (n_homes, n_hours)),
        i=r.normal(0.0, C.METER_I_NOISE_REL_SD, (n_homes, n_hours)),
        tx_v=r.normal(0.0, C.TX_V_NOISE_SD, (n_ph, n_hours)),
        tx_e=r.normal(0.0, C.TX_E_NOISE_REL_SD, (n_ph, n_hours)),
    )


def simulate(base_kwh, wiring, noise, events=()):
    """Return meter and transformer readings (arrays) for the street, with the given problems."""
    n_homes, n_hours = base_kwh.shape
    n_ph = len(C.PHASES)
    ph = wiring["phase"].map(PHASE_INDEX).to_numpy()
    kwh = base_kwh.copy()

    # 1. real behaviour changes (family away, new air conditioner) change true consumption
    powered = np.ones((n_ph, n_hours), dtype=bool)
    v_shift = np.zeros((n_ph, n_hours))
    for ev in events:
        if ev["category"] == "behaviour":
            _apply_behaviour(kwh, ev)
        elif ev["category"] == "grid":
            _apply_grid(powered, v_shift, ev)
    home_on = powered[ph]
    kwh = np.where(home_on, kwh, 0.0)

    # 2. physics
    amps = kwh * 1000.0 / (C.NOMINAL_V * C.POWER_FACTOR)
    peak = np.quantile(base_kwh.sum(axis=0), 0.999)
    v0 = C.SOURCE_V - C.SOURCE_V_LOAD_DROP * np.clip(kwh.sum(axis=0) / peak, 0.0, 1.5)
    v0 = v0[None, :] + v_shift
    drop = np.zeros((n_homes, n_hours))
    loss = np.zeros((n_ph, n_hours))
    dist = wiring["distance_m"].to_numpy(dtype=float)
    cab = wiring["cable"].to_numpy()
    for c in np.unique(cab):
        for p in range(n_ph):
            rows = np.where((cab == c) & (ph == p))[0]
            if len(rows) == 0:
                continue
            x = dist[rows]
            drop[rows] = C.R_OHM_PER_M * (np.minimum.outer(x, x) @ amps[rows])
            order = np.argsort(x, kind="stable")
            seg = np.diff(np.concatenate([[0.0], x[order]]))
            through = np.cumsum(amps[rows][order][::-1], axis=0)[::-1]
            loss[p] += C.R_OHM_PER_M * (seg[:, None] * through**2).sum(axis=0) / 1000.0
    volts = v0[ph] - drop
    tx_true = np.stack([kwh[ph == p].sum(axis=0) for p in range(n_ph)]) + loss

    # 3. what the meters measure
    out = dict(
        true_kwh=kwh, loss_kwh=loss,
        kwh=kwh.copy(),
        v=np.where(home_on, volts + noise["v"], 0.0),
        i=np.where(home_on, amps * (1.0 + noise["i"]), 0.0),
        tx_e=tx_true * (1.0 + noise["tx_e"]),
        tx_v=np.where(powered, v0 + noise["tx_v"], 0.0),
    )
    # 4. meter and data-link problems change only what the home meters report
    for ev in events:
        if ev["category"] in ("meter", "comms"):
            _apply_meter(out, ev)
    for key, digits in (("kwh", 3), ("v", 1), ("i", 2), ("tx_e", 3), ("tx_v", 1)):
        out[key] = np.round(out[key], digits) + 0.0
    return out


def _apply_behaviour(kwh, ev):
    r, s, e = ev["rows"], ev["t0"], ev["t1"] + 1
    if ev["type"] == "holiday_absence":
        kwh[r, s:e] = ev["payload"][None, :]
    elif ev["type"] == "new_ac":
        kwh[r, s:e] = kwh[r, s:e] + ev["payload"][None, :]


def _apply_grid(powered, v_shift, ev):
    s, e = ev["t0"], ev["t1"] + 1
    for p in ev["phases"]:
        if ev["type"] in ("outage_street", "outage_phase"):
            powered[p, s:e] = False
        elif ev["type"] in ("voltage_sag", "voltage_surge"):
            v_shift[p, s:e] += ev["size"]


def _apply_meter(out, ev):
    r, s, e, kind = ev["rows"], ev["t0"], ev["t1"] + 1, ev["type"]
    if kind in ("missing_one", "missing_all"):
        for key in ("kwh", "v", "i"):
            out[key][r, s:e] = np.nan
    elif kind == "theft":
        for key in ("kwh", "i"):
            out[key][r, s:e] = out[key][r, s:e] * (1.0 - ev["size"])
    elif kind == "over_reporting":
        for key in ("kwh", "i"):
            out[key][r, s:e] = out[key][r, s:e] * (1.0 + ev["size"])
    elif kind == "voltage_drift":
        out["v"][r, s:e] = out["v"][r, s:e] + ev["size"] * np.linspace(0.0, 1.0, e - s)[None, :]
    elif kind == "impossible_value":
        if ev["level"] == "spike":
            out["kwh"][r, s:e] = C.SPIKE_KWH
        elif ev["level"] == "negative":
            out["kwh"][r, s:e] = -(np.abs(out["kwh"][r, s:e]) + 0.5)
        else:  # voltage_zero_with_load
            out["v"][r, s:e] = 0.0
    elif kind == "stuck_meter":
        for key in ("kwh", "v", "i"):
            out[key][r, s:e] = out[key][r, s - 1][:, None]


def transformer_frame(idx, out):
    data = {"timestamp": idx}
    for k, p in enumerate(C.PHASES):
        data[f"energy_kwh_{p}"] = out["tx_e"][k]
    data["energy_kwh_total"] = np.round(out["tx_e"].sum(axis=0), 3) + 0.0
    for k, p in enumerate(C.PHASES):
        data[f"voltage_v_{p}"] = out["tx_v"][k]
    return pd.DataFrame(data)
