# EnergyMind MVP: build plan

Status: agreed design, 12 November 2026; revised 23 November 2026 (five features, trust on two levels) and 27 November 2026 (a first usual also counts unchecked hours). Build from this file.

**The core:** rules handle definitions; one regression handles losses; five simple features (the values the forest learns from, one set per home and hour); an Isolation Forest learns what normal looks like; SHAP says why a row was flagged; a short diagnosis names it.

---

## Step 1. Prepare the data

- **Join the sources:** attach each meter reading to its home's phase and cable (wiring) and to the hour's day type (context): weekday, or weekend-or-holiday (see step 4).
- **Build a phase table, one row per phase per hour:** transformer energy on the phase, the sum of its meters' kWh, how many meters reported, and the hour's type of day (so the transformer voltage's usual compares the same kind of day).
- **Check once:** the units (kWh, volts), and that the transformer lines up with the sum of the meters in the **same** hour, not shifted by one.

*Why:* everything else depends on these joins. A one-hour shift would make every hour look like missing energy.

## Step 2. Rules for definitions

Every rule judges the **new hour** only, from the new hour's readings plus the memory the operator app sends (the two stuck checks read the 3 previous hours from it). A decision is made once and never revisited.

| Rule | Fires when | Verdict |
|---|---|---|
| Missing | the reading's kWh, voltage or current is empty | `missing_data_one_meter` |
| Impossible | a measured reading with negative kWh, kWh above 0 at 0 V, or kWh above 1.5 × the contracted power (honest 2018 maximum: 1.29 ×) | `impossible_value` |
| Stuck | the whole reading (kWh, V, I) identical to the 3 previous hours: 4 identical readings (honest maximum in 2018: 3). Estimated readings and readings at 0 V never count | `stuck_meter` |
| Outage | a transformer phase reads 0 kWh and 0 V, **and** nearly all (90%) of its homes show no voltage (0 V, or nothing sent) | `outage_phase`, or `outage_street` when all 3 phases are out |
| All meters missing | nearly all (90%) of the street's meters sent nothing, and the transformer doesn't read 0 kWh and 0 V on every phase (that would be an outage) | `missing_data_all_meters` |
| Transformer problem | the transformer's reading on a phase is empty, negative, shows energy at 0 V, reads 0 kWh and 0 V while nearly all (90%) of its homes show voltage, or is identical for 4 hours (readings at 0 V excluded) | `reference_meter_problem`, one per broken phase (the dataset never injects it) |

- **The transformer and the homes must agree.** A transformer phase at 0 kWh and 0 V is an outage only if its homes have no voltage either; if they have voltage, the transformer's meter is wrong. When the homes disagree among themselves (10% to 90% show voltage), no rule fires: the evidence contradicts itself, so the forest and an operator look at it.
- **The order, so one problem gives one finding** (`rules/findings.py`): transformer problems, then outages, then all meters missing, then the meter rules (missing, impossible, stuck) for the homes no street finding covers. An outage covers the homes on its phases; all meters missing covers every home (readings that slip through a failing collection system can't be trusted). Each home gets at most one verdict; a meter frozen on an impossible value is reported as impossible.
- A reading caught by a rule, or covered by a street finding, is not scored by the forest, and it's marked **untrusted** (see step 4).
- Estimated readings are kept and used in history, but not scored, and they're never impossible or stuck.

*Why:* these are definitions, so a rule is exact. A model would only make them less reliable.

## Step 3. The losses regression

- **What it predicts:** the normal gap on a phase, meaning transformer energy − sum of its meters (energy lost in the cables).
- **Formula:** expected losses = *a* + *b* × (transformer phase energy)², fitted as a linear regression on the energy squared. When the transformer sent no energy, the expected losses are 0 (no energy sent, nothing lost); an empty or negative transformer reading is broken and has no expected losses.
- **Fitted on 2018,** separately for each phase, because each phase carries different loads (2018 average losses, each hour's percentage averaged over the year: 0.76% on A, 0.59% on B, 0.52% on C. As the year's losses divided by the year's energy, where busy hours weigh more because they lose a bigger share: 0.90%, 0.70% and 0.66%).
- **Only on clean hours:** the transformer measured energy, every home on the phase sent its kWh, and no rule finding touches that phase at that hour (a home's or the transformer's finding touches its phase; a street finding touches all three). One broken reading can outweigh weeks of real losses, and a least-squares line is pulled hardest by the biggest errors.
- **Check:** *b* must come out positive, and every phase needs at least one usable hour.

*Why:* in the unexplained energy (energy sent − energy the homes recorded − expected losses), losses are the only part that isn't measured. They grow with the square of the load, so a fixed allowance can't fit both quiet and busy hours.

## Step 4. The five features

**"Usual"** is used by three features, each named after the value it compares with its usual (`<value>_vs_usual`): `consumption_vs_usual`, `voltage_gap_with_neighbours_vs_usual` and `transformer_voltage_vs_usual`. It always means the same thing:

- **Normal case:** the median of the same hour, on the same type of day, over the last 28 calendar days, **using trusted hours only**. It needs at least 3 trusted values. Using calendar days keeps the comparison in the current season.
- **Two types of day:** weekday, and weekend-or-holiday. Holidays join weekends because 4 weeks rarely contain more than one or two holidays, too few values for a median of their own.
- **Fewer than 3 trusted values:**
  - a home that has never had a valid usual (start of the data, a new connection) isn't scored yet; the definition rules still run on it. Its very first usual also counts its unchecked hours (no rule caught them, but nobody could check them): otherwise it could never get one, since an hour without a usual can't be scored and stays unchecked. From its first usual on, only trusted hours count. The same holds for each phase's first transformer voltage usual;
  - a home that already had one keeps its last valid usual for that hour and type of day, until it has 3 trusted values again. Otherwise a theft lasting more than about 3 weeks would leave too few trusted values, and the home would stop being scored while it's being robbed. The same applies to the transformer's voltage usual after a long grid event.

| Feature | How it's computed | Question it answers |
|---|---|---|
| `consumption_vs_usual` | this hour's kWh ÷ this home's usual kWh for this hour; a usual below a minimum (set in `settings.py`, checked on 2018) counts as that minimum, so a tiny night usual can't make the ratio jump | Is this home using more or less than usual? |
| `voltage_gap_with_neighbours_vs_usual` | the home's voltage gap with its neighbours (home voltage − median voltage of the other homes on its cable and phase) − this home's usual gap | Is this meter's voltage out of line with its neighbours? |
| `phase_unexplained_energy` | (transformer phase energy − sum of its meters − expected losses) ÷ transformer phase energy, in % | Is energy missing or appearing on this phase? |
| `transformer_voltage_vs_usual` | transformer phase voltage − its usual for this hour | Is the voltage arriving at the transformer off (a sag or surge from the grid)? |
| `hour` | 0–23 | What time is it? |

**These are general senses, not problem detectors.** Each one measures a basic relationship of a healthy street: a home vs its habits, a meter vs its neighbours, a phase's unexplained energy, the grid. Many problems disturb each one, and the forest decides what's abnormal.

**Why the comparisons work:**
- **Comparing with neighbours** cancels what the whole cable experiences: the voltage arriving at the transformer, and the load everyone shares.
- **Subtracting the home's usual gap** cancels its position on the cable. A home far from the transformer is always a bit lower.
- **Dividing the unexplained energy by the phase's energy** keeps the normal range the same at every load, because the transformer meter's error is a percentage of what it measures.

**The trust rule, so anomalies don't distort "usual":**
- **Two levels.** Each home row has a trust state (it feeds the home's consumption usual and its voltage gap's usual), and so does each phase row (it feeds the phase's transformer voltage usual).
- **Three states.** Only **trusted** rows feed "usual":
  - **trusted:** checked by the forest and found normal, or diagnosed as a real change in use. Estimated readings also count as trusted (the utility's own values, kept for history). During learning there's no forest yet, so a row is trusted if no rule caught it and it can be scored (it was measured, and all five features exist);
  - **untrusted:** a rule caught it, or the system diagnosed it as a fault or "unknown". It stays untrusted for as long as that incident lasts;
  - **unchecked:** the forest couldn't score it (see below). Nobody knows whether it's normal, so it stays out of "usual" (except toward a very first usual, see above); the fallback above covers long gaps. A phase row is unchecked while its transformer voltage has no usual to compare with.
- **A grid event untrusts everything it touches; a home problem untrusts only that home.** An outage, a sag or a surge on a phase untrusts the phase row and every home row on that phase: during a sag, the homes' consumption is shaped by the grid, not by the families. A theft or a broken meter untrusts only that home's row.
- **Hours diagnosed as a real change in use** (holiday, new appliance) **stay trusted**, so "usual" follows real life.
- **Neighbours** are the other homes on the same cable and phase (7 or 8 on the real street). Those whose reading is untrusted are left out of the neighbours' median; if none is left, `voltage_gap_with_neighbours_vs_usual` can't be computed. An estimated reading still counts (the rules don't untrust it, and one guessed voltage barely moves the median). Each hour's gap is computed once, after the rules, and kept in the memory with the home's row: the usual gap is the median of past gaps.
- **When one of a home's features can't be computed, its hour is unchecked:**
  - the home has no usual yet;
  - none of its neighbours is usable;
  - its phase's unexplained energy can't be computed: a meter on that phase was caught by a rule (missing, impossible or stuck), so the sum of the phase's meters is wrong; a meter on that phase sent an estimated reading (the utility's guess, not a measurement); or the transformer measured no energy on that phase (the unexplained energy is a percentage of it);
  - the transformer's reading on its own phase is broken.

  The other phases are still scored.
- **Trust comes only from the system's own decisions** (and later, operators'). Never from the answer key.

*Effect:* once a theft is flagged, its hours stop feeding the home's "usual", so the drop stays visible for as long as the theft lasts. A family that really changed habits gets a new "usual" within 4 weeks.

**Left out of the features:**
- **temperature:** weather changes what everyone uses, which is real life, not a fault. The energy still adds up, so the diagnosis calls it `normal_real_change` (no alert), and "usual" follows the season within 4 weeks;
- **the street's total compared with its usual** (`street_vs_usual`, dropped): it detects no problem (a cold snap is a real change the diagnosis already names); a broken transformer reading on one phase would have left all 75 homes unchecked; and a feature that carries no problem information wastes some of the forest's random splits;
- **day type:** already built into "usual";
- **current:** in this data it's computed from kWh;
- **raw kWh and raw voltage:** their normal range differs by home, and the forest can't tell homes apart.

## Step 5. Train the Isolation Forest

- **Rows:** every trusted, measured 2018 home-hour from 29 January onward (rows caught by a rule, unchecked rows and estimated readings are left out). Starting on 29 January means every usual in training comes from a full 28-day window, as it does for most hours afterwards (a median of 3 values is noisier than a median of 20, and the forest shouldn't learn that noise as normal). That's about 600,000 rows of 5 features.
- **Settings:** 256 rows per tree (the default), fixed seed.
- **Number of trees:** start at 100 and increase (200, 300, 500) until two forests trained with different seeds flag almost the same 2018 rows (for example, 95% overlap); expect 200–500. More trees only make scores stable near T. Keep 256 rows per tree: bigger samples make unusual points harder to isolate.
- **Threshold T:** start at the lowest 0.1% of 2018 scores, then tune in practice.

*Why:* the forest needs no labels, and STEG has none. It learns which **combinations** of the five senses are normal, so it can flag problems nobody wrote a rule for.

## Step 6. Explain and diagnose each flag

**Explain:** for each flagged row, SHAP gives how much each feature pushed it toward "unusual". The top two pushes are the reasons.

**Diagnose:** match the reasons to a known pattern.

| Main reason | Extra check | Verdict |
|---|---|---|
| `transformer_voltage_vs_usual` | most homes on the phase flagged in the same hour | `voltage_sag_upstream` / `voltage_surge_upstream` |
| `voltage_gap_with_neighbours_vs_usual` | only this home | `meter_voltage_drift` |
| `phase_unexplained_energy`, energy **missing** | compare the energy missing over the last 24 h with each home's drop below its usual over the same hours | `under_reporting` (home named, or candidates listed), or `energy_missing_unattributed` if no home's drop matches it |
| `consumption_vs_usual` | `phase_unexplained_energy` barely contributed | `normal_real_change`: logged, **no alert** |
| anything else | | `unknown`: a person decides |

**Naming the home:** a home is named only if the match is decent: its 24-hour drop is about the same as the energy missing over the same hours (how close is a setting, tuned in practice). If several homes match, they're all listed, closest match first. If none matches, nobody is named: `energy_missing_unattributed`, and a person decides. Why: a thief can hide a brand-new appliance behind a bypass, so their recorded use doesn't drop and the missing energy matches nobody; naming the closest home anyway would accuse an innocent one. Even a decent match makes a home a suspect, never proof.

There's **no pattern for energy appearing** (meters reporting more than the transformer delivered), so the held-out over-reporting can only show up as `unknown`. That's what keeps the test fair.

*Why:* the forest decides **whether** something is abnormal. The diagnosis only decides **what kind**, using the forest's own reasons plus one physical check.

## Step 7. Incidents, scores, explanation card

**Incidents:**
- Flagged hours with the same verdict on the same home or phase become one incident.
- An incident closes after 6 normal hours in a row (tuned in practice).
- A phase incident absorbs its homes' rows, so one problem gives one alert.

**Two separate scores:**
- **Unusualness:** "more unusual than 99.97% of normal hours" (the row's score compared with all 2018 scores).
- **Confidence:** how often this verdict type was right in practice, computed as (right + 1) ÷ (total + 2) and shown with the count. "Unknown" has no confidence.

**The card** (illustrative numbers):

> **Under-reporting suspected at H073** · confidence 83% (9 practice cases) · more unusual than 99.97% of normal hours
>
> **Why flagged:** phase B is missing 1.9% of its energy; H073 is using 75% less than usual.
>
> **Diagnosis:** phase B lost 2.1 kWh more than normal over the last 24 h; H073 used 2.0 kWh less than usual; no other home on phase B dropped that much.
>
> **Ruled out:** a holiday (energy is missing), a grid event (the transformer's voltage is normal).
>
> **Decision:** operator.

## Step 8. Operators and feedback

- **In the MVP, every incident goes to an operator.** Automation comes later, once the confidence numbers justify it.
- **The operator chooses:** confirm, reject, or "different cause" (with a reason). Every decision is logged in the incident table (step 11, item 6).

**How each decision improves the system:**

| Operator input | Effect |
|---|---|
| confirmed fault | those hours stay untrusted |
| rejected as normal | those hours become trusted, and their rows are added to the forest's training data at the next retraining, so it stops flagging that pattern |
| all decisions | update the confidence of each verdict type |
| the same "unknown" confirmed several times | becomes a new pattern in the diagnosis |

## Step 9. The hourly loop

Every hour, the operator app sends the new hour's readings plus a memory: the past hours' home and phase rows (about 28 days, what the usual needs), each with its trust state (and, for a home row, its voltage gap with its neighbours), plus the last valid usual of each home and phase (the fallback when fewer than 3 trusted values remain). The detector stores nothing itself; it judges the new hour once and returns the new rows with their trust state (and gap) and the updated last usuals, which the app keeps.

1. Run the definition rules.
2. Compute each home's voltage gap with its neighbours, then the five features, using trusted past hours only.
3. Score the 75 home rows with the forest.
4. Explain each flagged row (SHAP) and diagnose it.
5. Update incidents and update trust flags.
6. Send cards to the operator, and log everything.

## Step 10. Learning, practice, final test

**Held-out types:**
- `over_reporting` stays held out (no diagnosis pattern for it).
- Once the features and patterns are frozen, a teammate secretly adds **one more** held-out type to the final-test months: a real fault with no rule or pattern written for it.

| Phase | Data | What you do |
|---|---|---|
| **Learning** (2018) | readings, transformer, wiring, context | run the rules over the year, then fit the losses regression on the clean hours; replay 2018 hour by hour through the same functions as live (rules, features, trust) to build the forest's rows; train the forest, set the starting T, freeze |
| **Practice** (odd months of 2019) | the same, plus the practice answers | run the hourly loop as if live; compare incidents with the answers; tune T, the candidate rule and the incident closing, **one change at a time**; fill the confidence table; freeze |
| **Final test** (even months of 2019) | the same, plus the test answers | run **once**, then report |

**Report:**
- recall and precision, per type and size;
- false alarms per 100 meter-days;
- false fraud suspicions;
- grid events confused with house problems;
- hours until the first alert;
- held-out problems caught as "unknown" on the right meter;
- whether each verdict type's confidence (from practice) holds up: compared with how often that verdict was right in the final test. The final test never changes the confidence.

**To show the forest earns its place:** compare it with plain thresholds on the same five features (one limit per feature, tuned on practice).

## Step 11. Later improvements (after the MVP works)

1. **24-hour averages** of `phase_unexplained_energy` and `voltage_gap_with_neighbours_vs_usual`, so small, slow problems show up sooner.
2. **Better home naming:** match the shape of the missing energy hour by hour against each home's readings.
3. **Transformer and network-map checks** when onboarding a new site.
4. **Safe automation** for verdict types with proven confidence.
5. **A confidence model learned from operator decisions** (logistic regression).
6. **A supervised model trained on operator decisions.** Once enough incidents are labelled, train a classifier that predicts the anomaly category from the evidence. It doesn't replace the forest, which is still needed for unknown problems; it sharpens the diagnosis and the confidence. For this to be possible later, every decision must be stored **from the first day**, in one structured table in the system's database:

| Field | Why |
|---|---|
| incident ID, home(s), phase, start, end | what happened, and where |
| the five feature values for every hour of the incident, **as computed at the time** | the model's inputs. Recomputing them later would give different values, because the usual and the trust flags change |
| forest score and SHAP reasons | the evidence the system saw |
| system verdict and confidence | to compare the system with the outcome |
| operator label, **chosen from a fixed list** (the verdict names, plus `normal` and `other`) | a clean target. Free text can't be trained on |
| operator reason (free text, stored separately) | context for people |
| field inspection result, when one happens | stronger evidence than an operator's judgement |
| operator ID, decision time, model and settings versions | traceability |

Two rules for this table:
- **Labels only exist for what the system flagged,** so a model trained on them could never learn what the system misses. Also store a small random sample of unflagged home-days that operators reviewed.
- **The table holds personal data** (Law 2004-63), so access to it is restricted and logged.

**Known limits of the MVP:**
- Small, slow problems will be caught late or missed until improvement 1.
- Subtle thefts are often reported as a list of candidates, not one named home.
- A theft hidden behind a brand-new appliance can't be tied to a home: it's reported as energy missing on the phase.
- A stuck meter is recognised at its 4th identical reading, so the alarm comes 2 hours after it froze (honest meters repeat a reading up to 3 times in a row). Those 2 earlier hours stay as they were judged, and during them the frozen reading still goes into its phase's unexplained energy, so the phase can look like it's missing or gaining energy.
- While a meter's reading is caught by a rule (missing, impossible or stuck) or is estimated, its phase's unexplained energy can't be computed, so that phase's homes aren't scored by the forest; the rules still judge them.
- One home's supply cut, recorded as 0 V while its phase has power, isn't named by a rule: the forest flags it (its voltage is far below its neighbours') and an operator decides (generally, a supply cut wouldn't send anything anyway so it would count as missing)
- A problem the system misses feeds the usual. The median barely moves until the problem fills half of its values (about 2 weeks); after that, it looks normal to the features that use a usual. Its hours leave the usual within 4 weeks after it ends. The unexplained energy uses no usual, so a missed theft still shows as energy missing on the phase.
- Learning has no forest yet, so it treats the history as mostly normal: a long problem hidden in the history is learned as normal. Likewise, a home's very first usual comes from hours nobody checked: a problem present from its first days becomes its usual.
- A home alone on its cable and phase has no neighbours, so its voltage gap can't be computed: its gap feature is always empty, and the forest never scores it. The rules still judge it, and energy going missing at its home still shows in its phase's unexplained energy.
- The physics in the dataset is simulated, so the results are an upper bound.
