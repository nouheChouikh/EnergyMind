"""Tests for detector/preparation/: the two tables (tables.py) and the input checks (input_checks.py).

Each test receives the made-up street from conftest.py as arguments (meter_readings, street_wiring,
hourly_context, transformer_readings), runs the code being tested, and checks one behaviour.
"""

import pandas as pd  # tables
import pytest  # the test tool; pytest.approx compares decimal numbers, pytest.raises expects an error

from detector.data_formats import (  # the expected column lists
    HOME_HOURS_COLUMNS,
    HOURLY_CONTEXT_COLUMNS,
    METER_READINGS_COLUMNS,
    PHASE_HOURS_COLUMNS,
    STREET_WIRING_COLUMNS,
    TRANSFORMER_READINGS_COLUMNS,
)
from detector.preparation.input_checks import (  # the code being tested
    check_required_columns,
    check_same_hour,
    check_units,
)
from detector.preparation.tables import build_home_hours, build_phase_hours  # the code being tested

# Two hours of the made-up street that several tests look at.
FRIDAY_10 = pd.Timestamp("2019-01-04 10:00")  # Friday 4 January 2019, 10:00: a weekday
SATURDAY_10 = pd.Timestamp("2019-01-05 10:00")  # Saturday 5 January 2019, 10:00: a weekend day


def row(table, **match):
    """Returns the one row of the table matching every given column value, e.g. row(t, home_id="H1").

    Fails the test if no row or several rows match, so a test never checks the wrong row by accident.
    """
    selected = table
    for column, value in match.items():
        selected = selected[selected[column] == value]  # keep only the rows where this column has this value
    assert len(selected) == 1
    return selected.iloc[0]


# ============================================================================= the home table
def test_home_table_has_one_row_per_home_and_hour(meter_readings, street_wiring, hourly_context):
    """The home table is a complete grid: every home of the wiring, at every hour of the context."""
    # Run.
    homes = build_home_hours(meter_readings, street_wiring, hourly_context)
    # Check: 6 homes x 48 hours, and exactly the columns listed in data_formats.py, in that order.
    assert len(homes) == 6 * 48
    assert list(homes.columns) == HOME_HOURS_COLUMNS


def test_reading_that_never_arrived_becomes_an_empty_row(meter_readings, street_wiring, hourly_context):
    """A reading absent from the input must not vanish: it becomes a row with empty values, so it counts as missing."""
    # Prepare: delete H3's reading at Friday 10:00, as if the meter never sent it.
    is_h3_friday_10 = (meter_readings["home_id"] == "H3") & (meter_readings["timestamp"] == FRIDAY_10)
    readings_without_it = meter_readings[~is_h3_friday_10]
    # Run.
    homes = build_home_hours(readings_without_it, street_wiring, hourly_context)
    # Check: the row is still there, with kWh, voltage and current empty.
    gap = row(homes, home_id="H3", timestamp=FRIDAY_10)
    assert gap[["kwh", "voltage_v", "current_a"]].isna().all()
    # Check: it is marked as not estimated, because the utility didn't fill it in: it is simply missing.
    assert gap["estimated"] == 0


def test_friday_is_a_weekday_and_saturday_a_weekend_or_holiday(meter_readings, street_wiring, hourly_context):
    """Each row gets its type of day from the context's weekend flag."""
    # Run.
    homes = build_home_hours(meter_readings, street_wiring, hourly_context)
    # Check: Friday is a weekday, Saturday is a weekend day.
    assert row(homes, home_id="H1", timestamp=FRIDAY_10)["day_type"] == "weekday"
    assert row(homes, home_id="H1", timestamp=SATURDAY_10)["day_type"] == "weekend_or_holiday"


def test_holiday_on_a_friday_counts_as_weekend_or_holiday(meter_readings, street_wiring, hourly_context):
    """Holidays join weekends (too few holidays in 4 weeks for a usual of their own)."""
    # Prepare: make every hour a public holiday, Friday included.
    hourly_context["is_holiday"] = 1
    # Run.
    homes = build_home_hours(meter_readings, street_wiring, hourly_context)
    # Check: Friday, now a holiday, is grouped with weekends.
    assert row(homes, home_id="H1", timestamp=FRIDAY_10)["day_type"] == "weekend_or_holiday"


def test_text_timestamps_in_the_context_still_join(meter_readings, street_wiring, hourly_context):
    """The context's timestamps are text, as in the CSV file; the readings' are real dates. They must still match."""
    # Run (the made-up context already holds its timestamps as text).
    homes = build_home_hours(meter_readings, street_wiring, hourly_context)
    # Check: every reading found its hour. If text and dates didn't match, the kWh column would be empty.
    assert homes["kwh"].notna().all()


def test_period_column_never_reaches_the_home_table(meter_readings, street_wiring, hourly_context):
    """period tells practice from final test: the detector must never see it, even if a file contains it."""
    # Prepare: add a period column to the context, as in the dataset's file.
    hourly_context["period"] = "practice"
    # Run.
    homes = build_home_hours(meter_readings, street_wiring, hourly_context)
    # Check: the column was left behind.
    assert "period" not in homes.columns


# ============================================================================= the phase table
def test_phase_table_adds_up_each_phase(meter_readings, street_wiring, hourly_context, transformer_readings):
    """Each phase row holds the transformer's measurement and the sum of its homes' readings, side by side."""
    # Run: the home table first, then the phase table built from it.
    homes = build_home_hours(meter_readings, street_wiring, hourly_context)
    phases = build_phase_hours(homes, transformer_readings)
    # Check: 3 phases x 48 hours, and exactly the columns listed in data_formats.py.
    assert len(phases) == 3 * 48
    assert list(phases.columns) == PHASE_HOURS_COLUMNS
    # Check phase A at Friday 10:00. Phase A holds homes H1 and H2.
    phase_a = row(phases, phase="A", timestamp=FRIDAY_10)
    h1_kwh = row(meter_readings, home_id="H1", timestamp=FRIDAY_10)["kwh"]
    h2_kwh = row(meter_readings, home_id="H2", timestamp=FRIDAY_10)["kwh"]
    transformer = row(transformer_readings, timestamp=FRIDAY_10)
    assert phase_a["homes_kwh"] == pytest.approx(h1_kwh + h2_kwh)  # the homes' side: H1 + H2
    assert phase_a["nb_homes_reporting"] == 2  # both sent a reading
    assert phase_a["nb_homes_on_phase"] == 2  # two homes hang on phase A
    assert phase_a["transformer_kwh"] == transformer["energy_kwh_A"]  # the transformer's side
    assert phase_a["transformer_voltage_v"] == transformer["voltage_v_A"]


def test_phase_table_carries_each_hours_day_type(meter_readings, street_wiring, hourly_context, transformer_readings):
    """Each phase row carries its hour's type of day: the transformer voltage's usual compares the same kind of day."""
    # Run.
    homes = build_home_hours(meter_readings, street_wiring, hourly_context)
    phases = build_phase_hours(homes, transformer_readings)
    # Check: on Friday all three phase rows say weekday; on Saturday all three say weekend_or_holiday.
    assert set(phases.loc[phases["timestamp"] == FRIDAY_10, "day_type"]) == {"weekday"}
    assert set(phases.loc[phases["timestamp"] == SATURDAY_10, "day_type"]) == {"weekend_or_holiday"}


def test_phase_table_counts_only_the_homes_that_reported(meter_readings, street_wiring, hourly_context,
                                                         transformer_readings):
    """A missing meter lowers the count of homes that reported, so later steps know the phase's sum is incomplete."""
    # Prepare: empty H1's reading at Friday 10:00 (H1 is on phase A).
    is_h1_friday_10 = (meter_readings["home_id"] == "H1") & (meter_readings["timestamp"] == FRIDAY_10)
    meter_readings.loc[is_h1_friday_10, ["kwh", "voltage_v", "current_a"]] = float("nan")
    # Run.
    homes = build_home_hours(meter_readings, street_wiring, hourly_context)
    phase_a = row(build_phase_hours(homes, transformer_readings), phase="A", timestamp=FRIDAY_10)
    # Check: the sum and the count hold only H2; the phase still has 2 homes.
    h2_kwh = row(meter_readings, home_id="H2", timestamp=FRIDAY_10)["kwh"]
    assert phase_a["homes_kwh"] == pytest.approx(h2_kwh)
    assert phase_a["nb_homes_reporting"] == 1
    assert phase_a["nb_homes_on_phase"] == 2


def test_transformer_hour_that_never_arrived_leaves_its_phases_empty(meter_readings, street_wiring,
                                                                     hourly_context, transformer_readings):
    """An hour absent from the transformer's readings keeps its phase rows, with empty values, so a rule can catch it."""
    # Prepare: delete the transformer's reading at Friday 10:00, as if it never arrived.
    transformer_without_it = transformer_readings[transformer_readings["timestamp"] != FRIDAY_10]
    # Run.
    homes = build_home_hours(meter_readings, street_wiring, hourly_context)
    phases = build_phase_hours(homes, transformer_without_it)
    # Check: Friday 10:00 still has its 3 phase rows, and their transformer values are empty.
    friday_10 = phases[phases["timestamp"] == FRIDAY_10]
    assert len(friday_10) == 3
    assert friday_10[["transformer_kwh", "transformer_voltage_v"]].isna().all().all()


# ============================================================================= the input checks
def test_missing_column_is_named_in_the_error(meter_readings):
    """A file without a needed column is stopped, and the message says which column is missing."""
    # Prepare: meter readings without their voltage column.
    readings_without_voltage = meter_readings.drop(columns="voltage_v")
    # Run and check: an error is raised, and its message names voltage_v.
    with pytest.raises(ValueError, match="voltage_v"):
        check_required_columns(readings_without_voltage, METER_READINGS_COLUMNS, "meter readings")


def test_kwh_given_in_wh_is_refused(meter_readings, street_wiring, hourly_context):
    """Readings sent in Wh instead of kWh (1000 times too big) are stopped."""
    # Prepare: multiply every kWh by 1000, as if the file held Wh.
    meter_readings["kwh"] = meter_readings["kwh"] * 1000
    homes = build_home_hours(meter_readings, street_wiring, hourly_context)
    # Run and check: an error is raised, and its message mentions kWh.
    with pytest.raises(ValueError, match="kWh"):
        check_units(homes)


def test_voltage_given_in_kilovolts_is_refused(meter_readings, street_wiring, hourly_context):
    """Voltages sent in kilovolts instead of volts (1000 times too small) are stopped."""
    # Prepare: divide every voltage by 1000, as if the file held kilovolts.
    meter_readings["voltage_v"] = meter_readings["voltage_v"] / 1000
    homes = build_home_hours(meter_readings, street_wiring, hourly_context)
    # Run and check: an error is raised, and its message mentions volts.
    with pytest.raises(ValueError, match="volts"):
        check_units(homes)


def test_transformer_shifted_by_one_hour_is_refused(meter_readings, street_wiring, hourly_context,
                                                    transformer_readings):
    """If the transformer and the meters don't label the same hour, every hour would look like missing energy."""
    # Prepare: label each transformer reading one hour later than it really is.
    transformer_readings["timestamp"] = transformer_readings["timestamp"] + pd.Timedelta(hours=1)
    homes = build_home_hours(meter_readings, street_wiring, hourly_context)
    phases = build_phase_hours(homes, transformer_readings)
    # Run and check: an error is raised, and its message mentions the hour.
    with pytest.raises(ValueError, match="hour"):
        check_same_hour(phases)


def test_correct_inputs_pass_every_check(meter_readings, street_wiring, hourly_context, transformer_readings):
    """Good data must never be stopped: every check passes on the made-up street as it is."""
    # Run: every check, on the untouched inputs. Any error raised here fails the test.
    check_required_columns(meter_readings, METER_READINGS_COLUMNS, "meter readings")
    check_required_columns(transformer_readings, TRANSFORMER_READINGS_COLUMNS, "transformer readings")
    check_required_columns(street_wiring, STREET_WIRING_COLUMNS, "street wiring")
    check_required_columns(hourly_context, HOURLY_CONTEXT_COLUMNS, "hourly context")
    homes = build_home_hours(meter_readings, street_wiring, hourly_context)
    check_units(homes)
    check_same_hour(build_phase_hours(homes, transformer_readings))
