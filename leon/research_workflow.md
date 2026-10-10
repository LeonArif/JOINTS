# Research workflow: what makes a film keep selling after its first 3 days

## Goal

The model predicts D4-D10 tickets as a multiple of the D1-D3 average (the "hold-up").
So a factor only matters if it changes how well a film keeps selling after its opening, not how big the opening is (the opening is already known from D1-D3).
Example: a big star raises every day's sales equally, so the hold-up stays the same; a kids film on a school break keeps selling on weekdays, so the hold-up rises.

## Workflow (applied to every factor)

1. Hypothesis: how the factor should change the hold-up, and for which rows (films, cinemas, dates).
2. Legit data: a source dated on or before 2025-09-30 (or fixed in advance, like a calendar); never test-film sales after the cutoff.
3. Evidence from the past: check the effect on data we already have (train rows, or the national Cinepoint charts 2023-09 .. 2025-09).
4. Test:
   - present in train (varies between train films / dates): add as a feature, LightGBM CV on 3 seeds (noise about 0.003 for one seed, about 0.001 for a 3-seed mean);
   - only in test (a season train never saw): a postprocessing factor measured on past data, scored by distance to main's file (scoreboard only, never fitted);
   - in both: both checks.
5. Decision: adopt only if the evidence and the test agree; log the result in `logs.md` either way.

## Factor list

Status: done = tested and decided, todo = still to research.

| # | Factor | Why it could change the hold-up | Legit data | Test | Status |
|---|---|---|---|---|---|
| 1 | Holidays / days off (weekends, national holidays, cuti bersama) | Days off sell more; a holiday inside D4-D10 lifts the ratio | holidays.csv, cuti_bersama.csv | CV | done: detailed days-off features adopted |
| 2 | Ramadan / Lebaran | Ramadan lowers evening attendance; Lebaran week is the year's peak | past national charts, ramadan.csv | scoreboard | done: season formula |
| 3 | Christmas / New Year break | School-break weekdays sell like weekends | past national charts | scoreboard | done: x1.83 level |
| 4 | Kids / family films x school breaks | Children are free on weekdays, families go together | school calendars, genres | scoreboard + train OOF | done: x1.3 |
| 5 | Genre x season (horror, comedy, action ...) | Some genres may suit holidays better | past national charts, hand labels | analysis | done: same effect in all seasons, model already has genre |
| 6 | Film profile (local, sequel, kind, source, predecessor) | Sequels and fan films front-load (fans come on day 1), originals grow by word of mouth | film_profile.csv | CV | done: adopted |
| 7 | Cities / mudik | People leave big cities for Lebaran | none city-level | sources | done: no evidence, not used |
| 8 | Nyepi (Bali day of silence) | Everything in Bali, cinemas included, closes for 24 hours | official calendar, test_history | data check | done: the 4 Denpasar rows are already 0 |
| 9 | Payday (gajian, around the 25th to the 1st) | People spend more right after payday, so a target day after payday sells more | calendar only | evidence on charts + CV | done: no consistent effect, CV +0.0011 (worse) |
| 10 | Production house / director track record | Some studios make front-loaded films (e.g. fast horror), others long runners | movies.csv producer / director, train film results only (inside each fold) | CV | done: producer +0.0002, director +0.0008, not used |
| 11 | City demographics (population, income, cinemas per city) | Small towns with one cinema keep a film longer; rich cities have more choice | BPS 2024 population / GRDP (published before the cutoff), train cinema counts | CV | done: city_name already covers 67 of 69 test cities, not built |
| 12 | School exam weeks x teen films | Students stop going to the cinema in exam weeks | school_calendar_regional.csv (exam rows) | scoreboard | done: 1-4 films per case, direction flips, not used |
| 13 | Chinese New Year (Imlek) and cities with many Chinese Indonesians | Family holiday; Chinese-language films | calendar | scoreboard | done: legit already matches main around it |
| 14 | Audience age (age rating D17+ vs all ages) | Adult films lose the school-break boost | movies.csv age_rating (already a feature) | scoreboard by season | done: remaining gap is per film, nothing to add |
| 15 | Showtime cuts when new films open | Cinemas give screens to new releases, older films lose shows | competition features (in), Cinepoint showtimes column (pre-cutoff) | CV | done: covered by show1-3 and competition features; test-film showtimes are after the cutoff |
| 16 | Runtime | Long films get fewer shows per day | needs a runtime source dated before release | CV | done: test runtimes mostly published after the cutoff (censorship), not usable |
| 17 | Pre-release hype (trailers, Wikipedia, search) | Hyped films front-load | Wikipedia pageviews (tested: no gain); trailer views are counted after the cutoff | CV | done: not usable |
| 18 | Quality / word of mouth (ratings) | Good films hold; bad films drop | pre-release ratings rarely exist; D1-D3 trend already captures early word of mouth | CV | done: ratings not usable |
| 19 | Weather / rainy season | Rain sends people to malls or keeps them home | test-day weather is after the cutoff; only climate averages are legit | analysis | done: not usable (climate averages add nothing beyond the season handling) |
| 20 | Big competing events (football matches, concerts) | Evening events pull people away | match calendars published before the cutoff | analysis | done: Oct 2025 matches started 22:00-00:00 WIB, after most shows; negligible |
| 21 | Format versions (IMAX / 3D) | Same film as the normal version, fewer screens | base_title join (in) | train OOF | done: model's own prediction beats tying to the base film |
| 22 | Small films pushed out in Lebaran week | Screens go to the big Lebaran releases | none needed | scoreboard | done: too few rows to matter |

## Results so far

- Adopted in main_legit: detailed days off, film profile, Christmas level (x1.83 of the normal level), kids films on school breaks (x1.3), season formula for Ramadan / Lebaran.
- Biggest remaining gap to main: which single film holds up in Lebaran week (per-film), which only post-cutoff sales can tell.
- Candidates to try on main.ipynb: the Christmas level and the kids factor (check for double counting, since main sees each film's real national sales), greedy removal of the 17 new columns.
