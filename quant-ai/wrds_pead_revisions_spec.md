# WRDS Earnings Revisions + PEAD Blueprint

## Objective

Build the first truly informational alpha block for the project:
- analyst estimate revisions
- post-earnings-announcement drift (PEAD)

The design goal is strict point-in-time integrity:
- no use of report period end as signal availability
- no use of fundamentals before filing availability
- no same-day trading unless exact announcement timestamps are available

## Primary WRDS datasets

### Required
- `IBES Detail History - Estimates`
  - WRDS SAS name: `ibes.detu` for unadjusted detail estimates
  - Official source: WRDS "A Note on IBES Unadjusted Data"
- `IBES Detail History - Actuals`
  - WRDS SAS name: `ibes.actu`
- `IBES Adjustment Factors`
  - WRDS SAS name: `ibes.adj`
- `IBES Detail ID`
  - WRDS SAS name: `ibes.id`
- `IBES-CRSP Link`
  - Use the WRDS linking suite / query form if available, or the historical `ICLINK` macro output
- `CRSP US Stock Daily`
  - Prefer current WRDS CIZ daily stock file
  - If your institution still exposes legacy style tables, legacy `dsf` is also acceptable

### Optional next layer
- `S&P Compustat Point in Time (Charter Oaks)` schema `comp_pit`
- `S&P Compustat Unrestated Quarterly (Charter Oaks)` schema `comp_urq`

## Why this stack

### Revisions
Revisions are a pure information alpha:
- estimate changes
- estimate breadth
- changes in analyst disagreement

### PEAD
PEAD is the cleanest market underreaction anomaly:
- compare actual EPS to the most recent pre-announcement consensus
- trade only after the market could have learned the information

## Official WRDS references

- IBES unadjusted note:
  - https://wrds-www.wharton.upenn.edu/documents/5/A_Note_on_IBES_Unadjusted_Data_pdf.pdf
- WRDS linking matrix:
  - https://wrds-www.wharton.upenn.edu/pages/wrds-research/database-linking-matrix/
- IBES to CRSP:
  - https://wrds-www.wharton.upenn.edu/pages/wrds-research/database-linking-matrix/linking-ibes-with-crsp/
- Compustat product overview showing current Charter Oaks schemas:
  - https://wrds-www.wharton.upenn.edu/pages/about/data-vendors/sp-global-market-intelligence/
- CRSP CIZ transition note:
  - https://wrds-www.wharton.upenn.edu/pages/data-announcements/changes-to-crsp-data/

## Exact timing rules

### Revisions signal
- Build consensus from IBES estimates snapshots.
- Signal availability date is the estimate snapshot date, not the fiscal period end date.
- If using summary history, use `statpers`.
- If using detail history, use the latest estimate record available by `estdats`.
- Trade no earlier than the next trading day after the snapshot date.

### PEAD signal
- Use the earnings announcement date from IBES actuals, not the accounting period end date.
- In WRDS examples this is the report date field `repdats`.
- Consensus must be the latest estimate snapshot strictly before `repdats`.
- If you do not have intraday announcement timestamps, trade from `t+1`, not `t`.
- If you later obtain exact announcement times, you can split into:
  - before-open announcements -> trade same-day open
  - after-close announcements -> trade next-day open

### Split-adjustment rule
- Follow the WRDS IBES unadjusted note.
- Use unadjusted estimates and unadjusted actuals, then align them with split adjustment factors.
- Relevant WRDS SAS files from the note:
  - `ibes.detu`
  - `ibes.actu`
  - `ibes.adj`
  - `ibes.id`
- The note explicitly warns that mismatched split bases can produce wrong surprises.

## Minimum fields to export

### IBES detail estimates
- `ticker`
- `fpedats`
- `estdats`
- `value`
- `measure`
- `fpi`
- `usfirm`
- `analys`
- `broker`
- `revdats`

### IBES actuals
- `ticker`
- `pends`
- `repdats`
- `pdicity`
- `value`
- `usfirm`

### IBES adjustments
- `ticker`
- `spdates`
- `adj`
- `usfirm`

### IBES identifiers
- `ticker`
- `oftic`
- `cusip`
- `cname`
- `sdates`
- `usfirm`

### IBES-CRSP link
- `ticker`
- `permno`
- `sdate`
- `edate`
- `score`

### CRSP daily stock data
- `permno`
- daily date
- daily return
- daily close price
- delisting return if available

## Preferred filters

### Universe
- U.S. common stocks only
- keep active and inactive securities
- exclude ETFs, ADRs, preferreds, units, warrants

### IBES
- `usfirm = 1`
- `measure = 'EPS'`
- use quarterly forecast horizon first
- start with `fpi = '1'` if that corresponds to next quarterly earnings in your subscription

### Link quality
- Keep only high-quality IBES-CRSP links
- If using `ICLINK`, keep scores `0, 1, 2` first

## Signal definitions

### Revisions
- `rev_21d = medest_t - medest_{t-21d}`
- `rev_63d = medest_t - medest_{t-63d}`
- `rev_pct_21d = medest_t / |medest_{t-21d}| - 1`
- `dispersion_change = stdev_t - stdev_{t-21d}`
- `breadth` if available from detail-level analyst estimate changes

### PEAD
- `surprise_raw = actual_aligned - medest_pre_announcement`
- `surprise_to_price = surprise_raw / price_{t-1}`
- `surprise_to_dispersion = surprise_raw / stdev_pre_announcement`

### Cross-sectional composite
- positive:
  - `rev_21d`
  - `rev_63d`
  - `surprise_to_price`
  - `surprise_to_dispersion`
- negative:
  - analyst disagreement spike
  - stale estimates

## Trading rule

### Conservative first version
- formation date:
  - revisions -> next trading day after estimate snapshot
  - PEAD -> next trading day after `repdats`
- hold:
  - 20 trading days for PEAD leg
  - rolling 20 or 40 trading days for revisions leg
- rank cross-sectionally within sector ETF universes

## Anti-bias checklist

- never use `fpedats` as the signal date
- never use `repdats` to trade on the same day without intraday timestamps
- never mix adjusted estimates with unadjusted actuals
- never use current identifiers to backfill old mappings without historical validity windows
- always enforce link validity windows from the IBES-CRSP link
- if using Compustat later, lag by filing availability, not fiscal quarter end

## Recommended implementation order

### Phase 1
- revisions only
- PEAD event table only
- validate dates and split-adjustment logic

### Phase 2
- combine revisions + PEAD
- add sector-neutral ranking

### Phase 3
- add `comp_urq` or `comp_pit` quality overlay

## What I would ask you to export first from WRDS

1. `IBES-CRSP Link` for all U.S. names
2. `ibes.detu`
3. `ibes.actu`
4. `ibes.adj`
5. `ibes.id`
6. CRSP daily stock file for linked `permno`s

Once we have those, we can build the first clean alpha without needing more tuning.
