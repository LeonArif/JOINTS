# Research notes: papers, audience statistics, test-film information

Written 2026-10-07 while the user was away.
Rule applied to every source: public on or before 2025-09-30, and nothing about a test film's own run.
Everything that was only seen in a search snippet, or whose date could not be read, is marked as a lead and is not in `external/`.

## 1. Papers on the same kind of task

None of these papers evaluates a season that is missing from the training period, and none uses Indonesian data.
Their accuracy numbers are not comparable to MASE, so they are used for dataset structure and ideas only.

### Groen (2023), "The Movie Scheduling Problem: Forecasting and Optimization", Erasmus University Rotterdam
Link: https://thesis.eur.nl/pub/70278/MasterThesis_MerelGroen_492835.pdf
Closest to our table: demand per movie session in a multiplex, forecast one week ahead.
Gradient tree boosting beat linear regression (OLS on log demand).
Accuracy on two held-out weeks: R2 0.725 and 0.708, RMSE 26.8 and 22.1 tickets per session, MAE 18.3 and 16.8.
Validation is rolling origin: fit on all earlier weeks, predict the next week, then add that week and predict the following one.
Tuned values: learning rate 0.1, 300 trees, min samples to split 30, max depth 8, 12 features per split.
Feature structure (their Table 5):
- General: movie indicator, start hour, weekday, public holiday flag, favourable weather flag.
- Movie: IMDb MOVIEmeter popularity rank, IMDb audience rating, language, genre, sequel flag, actors in IMDb top 100 / 1000, directors in top 1000 / 5000, distributor rank, budget above 80 million USD, weeks since release, tickets sold last week.
- Schedule (competition), the group the author found highly important: number of films of the same genre, of the same release week and of top-5 popularity that start within one hour of the session.
Mapping to our data:
- "Tickets last week" is our D1-D3.
- "Weeks since release" is our horizon.
- The competition group is the only one we do not have in main.ipynb. The log of the other session built a version from other films' D1-D3 rows (`c_new_films_T`, `own_show_share`), which fits this finding (LightGBM CV 0.3763 with the competition group alone against 0.3819 without it).
- Popularity and ratings are the analogue of film_profile and the Wikipedia pageviews; neither helped our CV.

### Baranowski, Korczak, Zajac (2020), "Forecasting Cinema Attendance at the Movie Show Level: Evidence from Poland"
Link: https://reference-global.com/article/10.2478/bsrj-2020-0006 (only the abstract was readable).
179,103 shows in 25 cinemas over 19 months, linear regressions re-estimated recursively for one-week-ahead forecasts.
Cinema characteristics cut RMSE by about 1%, region variables by about 0.3%, movie parameters and title popularity helped too.
We already carry cinema size (`cl_scale`, `cl_movies`) and city / province.

### Sawhney and Eliashberg (1996) and Neelamegham and Chintagunta (1999)
Already tested in baseline.ipynb on 2026-10-03 (exponential decay curve with shrinkage).
Better CV, worse LB: the decay curve does not transfer to the test period.

### Mestyan, Yasseri, Kertesz (2013), "Early Prediction of Movie Box Office Success based on Wikipedia Activity Big Data"
Link: https://arxiv.org/abs/1211.0970
Predicts a film's total box office from Wikipedia page views and editor activity before release, a film-level popularity signal.
Our version (pageviews in the 60 days before D1) had no CV gain, and only 12% of test rows have pageviews against 38% of train rows.

### Marshall, Dockendorff, Ibanez (2013), "A forecasting system for movie attendance", Journal of Business Research
Link: https://ideas.repec.org/a/eee/jbrese/v66y2013i10p1800-1806.html
117 Chilean films, Bass model against the Sawhney-Eliashberg model, in-sample prediction errors between 2.7% and 17.1% depending on horizon and history length.

### Seen and rejected
- Entropy 2022, "Predicting Box-Office Markets with Machine Learning Methods": annual national totals, 18 data points.
- Scientific Reports / PMC 2024 optimized XGBoost: classifies total revenue on the Kaggle Movies Dataset, a different task.
- "Artificial intelligence-based predictions of movie audiences on opening Saturday" (International Journal of Forecasting, 2020) and the daily-box-office DNN paper (Cognitive Systems Research, 2018): relevant, but the full text was blocked (HTTP 403).

### What to copy
1. Rolling-origin validation (train on earlier films, validate on later ones) as a second check next to the film-grouped CV.
   Both closest papers validate this way, and our CV has pointed the wrong way three times.
2. A competition block of features, built only from information that exists for test rows.
3. Keep GBDTs; the thesis found boosting clearly better than linear models, which matches our results.

## 2. Audience statistics found (all pages opened, dated on or before 2025-09-30)

File: `external/audience_statistics.csv` (8 rows, built in `external_add.ipynb`).

| Statement | Period | Source and date |
|---|---|---|
| Attendance drops 30 to 40% in Ramadan, also on weekends | Ramadan 2013 | Cineplex 21 via KapanLagi, 2013-07-12 |
| School breaks do not affect occupancy much; the industry relies on Eid and year-end | 2023 | GPBSI chair via Kontan, 2023-07-04 |
| Cinema XXI had more than 14 million admissions in April 2025, a monthly record (old record 11.7 million, June 2019) | April 2025 | Bareksa, 2025-05-02 |
| About 8 million in May 2025, 25 million in Q2 2024, 84 million in 2024 | 2024 to 2025 | Kontan, 2025-06-19 |
| Q1 2025 ticket revenue down 30% year on year, March (Ramadan) weaker, April admissions already close to the whole Q1 total | Q1 2025 | Indo Premier, 2025-04-25 |
| Five films above 1 million admissions in Q1 2025 against seven in Q1 2024 | Q1 2025 | Bareksa, 2025-05-02 |

Seasonal profile derived from these numbers (Cinema XXI only, rough):
- An average month of 2024 is 84 / 12 = 7.0 million admissions.
- April 2025, the Lebaran month with Eid on 31 March, is about 2.0 times an average month.
- May 2025 is about 1.1 times, Q2 2024 about 1.2 times.
- Q1 2025 is roughly 14 to 15 million for three months, so about 0.7 times an average month, and March (Ramadan) lower still.
- The school-break evidence agrees with our own train estimate: national tickets on school-holiday days were only 1.08 times other days.

Not found, despite searching:
- How many students from a school watch films on holidays, or the percentage of people who go to the cinema on holidays. Only general surveys exist (SMRC 2019: 67% of 15 to 38 year olds saw an Indonesian film in a cinema during 2019; Populix and IDN 2022: 95% of Gen Z and millennials like films, 21% go to a cinema). Their publication dates were not read, so they are leads, not data.
- The share of annual admissions that falls within 14 days of Eid: a figure of 17.25% (62.9 million over 2007 to 2024) is quoted in search summaries, but the only pages found were dated March 2026, so it is not used.
- Any monthly or daily series for Ramadan 2025 or Christmas 2024 beyond the film trajectories already in `season_trajectories.csv`.

Post-cutoff pages that appeared in search results and were discarded without storing their numbers: ANTARA articles dated 2026-03-31 (they contain admissions of test films), CNBC Indonesia 2026-03-04, Alonesia and Narasitoday March 2026 on Ramadan 2026 occupancy, Cinema XXI full-year 2025 results, and the 2026 Wikipedia pages of test films.

## 3. Test-film information before the cutoff

Almost nothing quantified exists before 2025-09-30 for the Oct 2025 to Mar 2026 films, and what exists cannot become a model feature without the same figure for the 237 train films.
- Official trailer of AGAK LAEN: MENYALA PANTIKU! was released on 2025-10-08, after the cutoff.
- ALAS ROBAN was first announced in November 2025, DANUR: THE LAST CHAPTER and most of the Q1 2026 films have their material from after the cutoff.
- Leads seen only in a search snippet (not opened, so not stored): Zootopia 2 trailer of 2025-05-20 with 130 million views in 24 hours and the final trailer of 2025-09-29 with 74 million.
- Franchise predecessors with admissions dated before the cutoff are already in `film_profile.csv` (Agak Laen 9.13 million, Sewu Dino 4.89 million, Danur 3 2.42 million and others), but only about 10% of test films have one.
- YouTube and video transcripts could not be read in this environment.

## 4. Ideas for a test-film-only focus

1. The test films' own D1-D3 rows are the only test-period information with a real signal, and they are already the model input (national and per-cinema D1-D3, trend, cinema count).
2. Seasonal priors (this file, `season_trajectories.csv`, `lebaran_admissions.csv`) are the only way to teach the model about Ramadan, Christmas and Lebaran, which train never saw; they act through postprocessing factors.
   The Lebaran x3 probe is still untested.
3. Weighting train films to look like test films was tried (adversarial weights, test-scale weights) and did not improve CV; CV cannot judge it, so it needs an LB test.
4. A rolling-origin check (train D1 before a cut date, validate after) would show how stable each feature is across the train season.
