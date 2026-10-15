"""EnergyMind - Phase 1 settings.

Every invented number of the simulation is in this file, with a one-line reason.
Change a value, run `python src/run_all.py`, and the whole dataset is rebuilt
from the same seed.
"""
from pathlib import Path

# ----------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "raw"                    # GoiEner download, never committed
DATA_DIR = ROOT / "data"
NORMAL_DIR = DATA_DIR / "normal"
FIG_DIR = ROOT / "figures"
NOTEBOOK = ROOT / "notebooks" / "01_exploration.ipynb"
ARCHIVE = RAW_DIR / "imp-pre.tzst"        # https://zenodo.org/records/7362094
METADATA = RAW_DIR / "metadata.csv"       # same Zenodo record
WEATHER_CACHE = DATA_DIR / "weather_openmeteo.csv"

# ----------------------------------------------------------------------------- randomness
SEED = 2026                               # one seed; each purpose draws from its own stream

# ----------------------------------------------------------------------------- time
# GoiEner hours are local wall-clock time. The dataset authors filled the missing
# spring daylight-saving hour and averaged the repeated autumn hour, so every hour
# exists exactly once.
START = "2018-01-01 00:00:00"
END = "2019-12-31 23:00:00"
PERIODS = {
    "learning": ("2018-01-01 00:00:00", "2018-12-31 23:00:00"),    # normal behaviour only
    "practice": ("2019-01-01 00:00:00", "2019-06-30 23:00:00"),    # problems injected, answers open
    "final_test": ("2019-07-01 00:00:00", "2019-12-31 23:00:00"),  # problems + held-out type
}

# ----------------------------------------------------------------------------- cleaning rules (step 1)
N_HOMES = 75
MAX_GAP_HOURS = 48             # absent hours in 2018-2019 we may fill (they are flagged as estimated)
MAX_ESTIMATED_SHARE = 0.05     # hours GoiEner marks as imputed
ANNUAL_KWH_MIN = 600           # below: likely an empty or second home
ANNUAL_KWH_MAX = 12000         # above: likely not a typical home
MAX_YOY_CHANGE = 0.35          # sharp 2018 -> 2019 change: new occupants or new equipment
MAX_ZERO_RUN_H = 0             # no measured zero hour at all: a lived-in home never uses exactly nothing
MAX_ZERO_SHARE = 0.0
MAX_FLAT_READINGS = 3          # identical measured readings in a row allowed; 4+ = flat stretch (probably an unflagged estimate)

# Province -> weather point (capital city) and regional holiday calendar ("holidays" package code)
PROVINCES = {
    "gipuzkoa": dict(words=("gipuzkoa", "guipuzcoa"), codes=(20,), lat=43.3183, lon=-1.9812, holidays="PV", label="Gipuzkoa"),
    "bizkaia": dict(words=("bizkaia", "vizcaya"), codes=(48,), lat=43.2630, lon=-2.9350, holidays="PV", label="Bizkaia"),
    "araba": dict(words=("araba", "alava"), codes=(1,), lat=42.8467, lon=-2.6727, holidays="PV", label="Araba/Alava"),
    "navarra": dict(words=("navarra", "nafarroa"), codes=(31,), lat=42.8125, lon=-1.6458, holidays="NC", label="Navarra"),
    "madrid": dict(words=("madrid",), codes=(28,), lat=40.4168, lon=-3.7038, holidays="MD", label="Madrid"),
    "cantabria": dict(words=("cantabria",), codes=(39,), lat=43.4623, lon=-3.8100, holidays="CB", label="Cantabria"),
    "la_rioja": dict(words=("rioja",), codes=(26,), lat=42.4627, lon=-2.4450, holidays="RI", label="La Rioja"),
    "barcelona": dict(words=("barcelona",), codes=(8,), lat=41.3874, lon=2.1686, holidays="CT", label="Barcelona"),
}
DEFAULT_PROVINCE = "gipuzkoa"  # GoiEner's home province, used if the metadata names none we know

# ----------------------------------------------------------------------------- street (step 3)
N_CABLES = 3
PHASES = ("A", "B", "C")
SPACING_M = 15.0               # metres between neighbouring connections on a cable
NOMINAL_V = 230.0
SOURCE_V = 233.0               # transformer output at no load (set a little high, as is usual)
SOURCE_V_LOAD_DROP = 3.0       # transformer output falls by up to 3 V at the street's peak load
R_OHM_PER_M = 0.00064          # phase + neutral loop of a 95 mm2 aluminium cable (2 x 0.32 ohm/km)
POWER_FACTOR = 0.95
METER_V_NOISE_SD = 0.3         # volts, home meter
METER_I_NOISE_REL_SD = 0.005   # 0.5 % of the reading, home meter
TX_V_NOISE_SD = 0.2            # volts, reference meter at the transformer
TX_E_NOISE_REL_SD = 0.0025     # reference energy meter, accuracy class 0.5S

# ----------------------------------------------------------------------------- injected problems (step 4)
LEVELS = ("subtle", "medium", "obvious")
BUFFER_H = 24                  # free hours kept before and after every problem on the same homes
MAX_TRIES = 5000               # random placements tried per problem before giving up
SPIKE_KWH = 40.0               # 40 kWh in one hour = 40 kW: impossible on a home contract (max 15 kW)
AC_PROFILE = {12: 0.5, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0,
              19: 1.0, 20: 1.0, 21: 1.0, 22: 0.5}   # share of the AC's power used at each hour

# count = (practice, final_test). ladder: what grows from subtle to obvious.
# balance=True: changes the street energy balance, so two of these never overlap on one phase.
# Order matters: problems covering many homes are placed first.
PROBLEMS = {
    "outage_street": dict(
        category="grid", scope="street", count=(3, 3), hours=(1, 3, 6), ladder="hours",
        verdict="outage_street",
        what="Power is cut for the whole street. Every meter and the transformer read 0 kWh, 0 V, 0 A."),
    "missing_all": dict(
        category="comms", scope="street", count=(3, 3), hours=(1, 4, 12), ladder="hours",
        verdict="missing_data_all_meters",
        what="The data link fails: no meter reading arrives (empty values). Power still flows, so the transformer keeps measuring."),
    "outage_phase": dict(
        category="grid", scope="phase", count=(4, 4), hours=(1, 3, 6), ladder="hours",
        verdict="outage_phase",
        what="One phase loses power. Its homes and that phase of the transformer read 0."),
    "voltage_sag": dict(
        category="grid", scope="phase", count=(3, 3), hours=(1, 4, 12), ladder="size",
        sizes=(-12, -25, -45), unit="V", verdict="voltage_sag_upstream",
        what="Voltage on one phase drops upstream. Every home on that phase and the transformer phase voltage fall together; energy is unchanged."),
    "voltage_surge": dict(
        category="grid", scope="phase", count=(3, 3), hours=(1, 4, 12), ladder="size",
        sizes=(8, 15, 28), unit="V", verdict="voltage_surge_upstream",
        what="Voltage on one phase rises upstream. Every home on that phase and the transformer phase voltage rise together; energy is unchanged."),
    "missing_one": dict(
        category="comms", scope="home", count=(6, 6), hours=(3, 12, 48), ladder="hours", balance=True,
        verdict="missing_data_one_meter",
        what="One meter sends nothing (empty values). The transformer still sees the home's consumption."),
    "theft": dict(
        category="meter", scope="home", count=(9, 9), hours=(72, 168, 336), ladder="size", balance=True,
        sizes=(0.2, 0.5, 0.8), unit="share of energy hidden", verdict="under_reporting",
        what="The meter is bypassed: it records only (1 - size) of the energy and current that really flow. Voltage is normal and the transformer sees the full consumption, so the street gap opens."),
    "over_reporting": dict(
        category="meter", scope="home", count=(0, 9), hours=(72, 168, 336), ladder="size", balance=True,
        sizes=(0.15, 0.4, 1.0), unit="share of energy added",
        held_out=True, verdict="any_alert_on_this_meter_before_it_ends",
        what="Held-out type, final test only, no rule written for it: a faulty meter records (1 + size) times the real energy and current. The transformer does not, so the street gap turns negative."),
    "voltage_drift": dict(
        category="meter", scope="home", count=(6, 6), hours=(168, 336, 504), ladder="size",
        sizes=(3, 6, 12), unit="V error at the end", verdict="meter_voltage_drift",
        what="The meter's voltage sensor drifts: its error grows steadily from 0 to the size (sign drawn at random). Energy and current stay correct."),
    "impossible_value": dict(
        category="meter", scope="home", count=(6, 6), hours=(1, 2, 3), ladder="variant", balance=True,
        variants=("spike", "negative", "voltage_zero_with_load"), verdict="impossible_value",
        what="A physically impossible reading: 40 kWh in one hour (spike), negative energy on a home without solar (negative), or 0 V while energy flows (voltage_zero_with_load)."),
    "stuck_meter": dict(
        category="meter", scope="home", count=(6, 6), hours=(6, 24, 72), ladder="hours", balance=True,
        verdict="stuck_meter",
        what="The meter keeps repeating its last energy, voltage and current values. The transformer sees the real consumption."),
    "holiday_absence": dict(
        category="behaviour", scope="home", count=(6, 6), hours=(120, 240, 456), ladder="hours", fault=False,
        verdict="normal_real_change",
        what="Not a fault: the family is away. Real consumption falls to the home's standby level, so meter and transformer agree."),
    "new_ac": dict(
        category="behaviour", scope="home", count=(2, 4), hours=(168, 336, 504), ladder="size", fault=False,
        sizes=(0.4, 0.8, 1.5), unit="kW added in the afternoon", months=(6, 7, 8, 9),
        verdict="normal_real_change",
        what="Not a fault: a new air conditioner adds up to the size in kW between 12:00 and 22:00 (June to September). Meter and transformer agree."),
}
