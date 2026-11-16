"""Shared test data used by every test file (pytest loads this file automatically).

A small made-up street: 6 homes (2 per phase, on 2 cables) over 48 hours, a Friday then a Saturday.
Every value comes from a formula, so each test knows exactly what to expect.
It is an honest street, with no problem anywhere: each test plants the one problem it checks.
"""

import numpy as np  # number arrays
import pandas as pd  # tables
import pytest  # the test tool; @pytest.fixture marks a function that hands fresh test data to each test

from detector.preparation.tables import build_home_hours  # turns the four inputs into the home table

# The 48 hours: Friday 4 and Saturday 5 January 2019.
HOURS = pd.date_range("2019-01-04 00:00", periods=48, freq="h")

# The 6 homes: (home id, cable, phase, contracted power in kW).
HOMES = [
    ("H1", 1, "A", 3.3), ("H2", 2, "A", 4.4),
    ("H3", 1, "B", 5.5), ("H4", 2, "B", 3.3),
    ("H5", 1, "C", 4.4), ("H6", 2, "C", 5.5),
]


@pytest.fixture
def street_wiring():
    """One row per home: its cable, phase and contracted power."""
    return pd.DataFrame(HOMES, columns=["home_id", "cable", "phase", "contracted_kw"])


@pytest.fixture
def hourly_context():
    """One row per hour. Timestamps are text, as in the dataset's CSV file; Saturday is a weekend day."""
    return pd.DataFrame({
        "timestamp": HOURS.strftime("%Y-%m-%d %H:%M:%S"),
        "is_weekend": (HOURS.dayofweek == 5).astype(int),  # dayofweek: Monday = 0, Saturday = 5
        "is_holiday": 0,
    })


@pytest.fixture
def meter_readings():
    """One row per home and hour, with a realistic daily pattern: low around 6:00, high around 18:00.

    The daily swing makes hour-to-hour changes much bigger than the 1% cable losses, as on a real street;
    without it, a one-hour shift between the transformer and the meters couldn't be told apart from losses.
    A tiny rise every hour on top keeps every reading different, so no reading repeats by accident.
    """
    tables = []
    hour_of_day = HOURS.hour.to_numpy()  # 0 to 23
    for number, (home_id, _cable, _phase, _kw) in enumerate(HOMES):
        step = np.arange(len(HOURS))  # 0, 1, 2 ... 47
        # Daily swing between 0 and 0.3 kWh: sin(...) is -1 at 6:00 and +1 at 18:00.
        daily = 0.3 * (1 + np.sin(2 * np.pi * (hour_of_day - 12) / 24)) / 2
        kwh = 0.2 + 0.05 * number + daily + 0.001 * step  # each home and each hour has its own value
        tables.append(pd.DataFrame({
            "home_id": home_id,
            "timestamp": HOURS,
            "kwh": kwh.round(3),
            "voltage_v": (230.0 - 0.5 * number + 0.1 * (step % 10)).round(1),
            # Current from power: kWh in one hour = average kW, and I = P / (230 V x 0.95).
            "current_a": (kwh * 1000 / (230 * 0.95)).round(2),
            "estimated": 0,
        }))
    return pd.concat(tables, ignore_index=True)


@pytest.fixture
def transformer_readings(meter_readings, street_wiring):
    """One row per hour: each phase's energy = its homes' kWh + 1% cable losses, about 232 V on every phase.

    The +1% keeps the transformer's energy different from the homes' sum, so tests can tell the two apart.
    The voltage moves a little every hour, as a real one does, so an honest transformer reading never repeats.
    """
    # Each reading with its home's phase, then the kWh added up per hour and phase.
    joined = meter_readings.merge(street_wiring[["home_id", "phase"]], on="home_id")
    per_phase = joined.pivot_table(index="timestamp", columns="phase", values="kwh", aggfunc="sum")
    table = pd.DataFrame({"timestamp": per_phase.index})
    step = np.arange(len(table))  # 0, 1, 2 ... 47
    for phase in ["A", "B", "C"]:
        table[f"energy_kwh_{phase}"] = (per_phase[phase].to_numpy() * 1.01).round(3)
        table[f"voltage_v_{phase}"] = (232.0 + 0.1 * (step % 10)).round(1)  # 232.0, 232.1 ... 232.9, 232.0 ...
    return table


@pytest.fixture
def home_hours(meter_readings, street_wiring, hourly_context):
    """The home table of the street above (one row per home and hour), built by the real preparation step.

    The rules' tests start from it: they plant a problem in it, then cut it into the new hour and the memory.
    """
    return build_home_hours(meter_readings, street_wiring, hourly_context)
