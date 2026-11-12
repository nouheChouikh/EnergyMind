"""Step 3 - build the street and simulate its normal values (no problem anywhere).

Output: dataset/data/system_inputs/street_wiring.csv               cable, phase and position of each home
        dataset/data/grading/normal_meter_readings.parquet         every home, every hour of 2018-2019:
                                                                   kWh (real GoiEner), voltage and current (simulated)
        dataset/data/grading/normal_transformer_readings.parquet   reference meter: energy and voltage per phase + total
These normal values are the reference that step 4 changes when it injects problems.
"""
import numpy as np
import pandas as pd

import config as C
from shared.helpers import load_source, period_labels, save_parquet, to_long, write_csv
from shared.street_physics import make_noise, make_wiring, simulate, transformer_frame


def read_homes():
    homes = pd.read_csv(C.SELECTED_HOMES, dtype=str, keep_default_na=False)
    homes["contracted_kw"] = pd.to_numeric(homes["contracted_kw"], errors="coerce")
    return homes


def build_normal():
    home_ids, idx, kwh, est = load_source()
    wiring = make_wiring(read_homes())
    if list(wiring["home_id"]) != list(home_ids):
        raise SystemExit("selected_homes.csv and selected_homes_readings.parquet do not list the same homes. Rerun step 1.")
    out = simulate(kwh, wiring, make_noise(*kwh.shape))
    return dict(wiring=wiring, home_ids=home_ids, idx=idx, est=est, base=kwh, out=out,
                readings=to_long(home_ids, idx, out["kwh"], out["v"], out["i"], est),
                transformer=transformer_frame(idx, out))


def main():
    res = build_normal()
    write_csv(res["wiring"], C.STREET_WIRING)
    save_parquet(res["readings"], C.NORMAL_METER_READINGS)
    save_parquet(res["transformer"], C.NORMAL_TRANSFORMER_READINGS)

    out, wiring = res["out"], res["wiring"]
    learn = period_labels(res["idx"]) == "learning"
    meters = np.stack([out["kwh"][(wiring["phase"] == p).to_numpy()].sum(axis=0) for p in C.PHASES])
    gap = 100 * (out["tx_e"] - meters)[:, learn].sum() / out["tx_e"][:, learn].sum()
    v = out["v"][out["v"] > 0]
    print(f"  {len(wiring)} homes on {C.N_CABLES} cables; per phase: "
          + ", ".join(f"{p}={int((wiring['phase'] == p).sum())}" for p in C.PHASES))
    print(f"  home voltage {v.min():.1f} to {v.max():.1f} V; street balance gap on 2018 (losses + noise): {gap:.2f}%")


if __name__ == "__main__":
    main()
