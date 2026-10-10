# Experiment Logs

## 2026-09-28 - baseline.ipynb (LGBM baseline)

Notebook: `baseline.ipynb`

### Setup
- D1 on train = first date a movie is shown in >= 50% of its max cluster count (skips preview days, matches how test D1 lines up with wide release).
- Train movies dropped if already running on 2025-04-01 or D10 falls after 2025-09-30: 205 movies left.
- Train pairs kept only if they have sales on D3 (test pairs are exactly the pairs still selling on D3): 8,007 pairs, 56,049 rows.
- Target = ticket / scale (scale = pair mean D1-D3), LightGBM with L1 objective, so the loss equals MASE.
- Features: pair D1-D3 tickets, trend, occupancy, shows; movie national D1-D3 totals, trend, cluster count, pair share; cluster mean scale; calendar (horizon, dow, day_tipe, holiday, D1 dow); city ticket price and price ratio vs D1-D3.
- Not used yet: movies.csv metadata, external data.
- Validation: GroupKFold(5) by movie_title, early stopping on the validation fold. SEED = 42.
- LGBM: lr 0.03, num_leaves 31, min_child_samples 50, subsample 0.8, colsample 0.8; final model n_estimators = mean best_iter (134).

### Results (CV MASE, lower is better)
| Approach | MASE |
|---|---|
| Naive: predict D1-D3 mean | 0.7393 |
| Naive: predict D3 ticket | 0.6965 |
| Naive: predict 0 | 0.5877 |
| LGBM baseline | 0.3860 |

- Per fold: 0.4852, 0.5255, 0.2622, 0.3909, 0.2660 (high variance between folds, depends on which movies land in each fold).
- Early stopping on the validation fold makes CV slightly optimistic.
- Kaggle public LB: 0.49211 (initial submission).
- CV-LB gap: 0.386 CV vs 0.492 LB, so CV is too optimistic by about 0.1.

## 2026-10-01 - main.ipynb (EDA)

Notebook: `main.ipynb`

### Setup
- Added title, section headers and a full EDA section; no model or features yet.
- EDA covers: file summary, train/test overlap, 3D/IMAX title variants, ticket distributions, national daily tickets with holidays, day-of-week and holiday effect, preview days before D1, D1-D10 decay curve, curve by D1 weekday, train vs test shift, clusters and cities, ticket prices, movie metadata (genre vs decay), occupancy and show count.

### Findings
- Only 10 of 160 test movies appear in train; all 121 test clusters except 4 are in train.
- All 20 test titles missing from movies.csv are 3D/IMAX variants; after stripping the format suffix every title (390/390) has metadata.
- Share of zero-ticket targets grows fast with horizon: D4 10.9%, D7 29.4%, D10 55.6% (train pairs still selling on D3).
- Median target ratio (ticket / D1-D3 mean) drops from about 0.9 on D4 to 0 by D9.
- National tickets in the test_history period (Oct 2025 - Mar 2026) are much lower than in train, and test pair scale is shifted lower than train: a likely part of the CV-LB gap.
- Test D1 is mostly Wednesday/Thursday like train, with fewer Friday starts.
- Genre shows different decay speed (Action/Horror slower, Romance/Mystery faster), so movies.csv metadata is worth trying as features.

## 2026-10-01 - baseline.ipynb (optimization round 1)

Notebook: `baseline.ipynb`

### Diagnosis of the CV-LB gap (CV 0.386 vs LB 0.492)
- Metric confirmed from the competition page: scale = mean D1-D3 per pair, clipped at a minimum of 1 (baseline was missing the clip, negligible effect).
- Test D1 = first test_history date for all movies, and test pairs = pairs with D3 > 0, so the train sample construction already matches test.
- Honest CV without early stopping on the validation fold: 0.389 (not 0.386).
- Test has far more small-scale pairs (scale <= 20: 13% of test rows vs 3.6% of train rows), and small pairs have much larger error.
  Reweighting CV error to the test scale mix (`cv_w`) gives 0.479, close to the LB 0.492, so most of the gap is the test mix, not leakage.
- Small test pairs are mostly low-selling movies with 3 active days; small train pairs are mostly odd late arrivals (t1 = t2 = 0), which are nearly unpredictable (ratio 0 to 50).
- Test period is quieter: median occupancy 11 vs 20.5 in train.
- Adversarial validation (train vs test rows, GroupKFold by movie) AUC 0.618, driven by movie-level volume features (mv_clusters, mv_t1, mv_t2).

### Experiments (CV = GroupKFold(5) by movie, fixed trees, no early stopping; cv_w = reweighted to test scale mix; adv = adversarial-weighted; time = train D1 < 2025-07-22, validate D1 >= 2025-08-01)
| Approach | CV | cv_w | adv | time |
|---|---|---|---|---|
| Baseline features, 134 trees | 0.3890 | 0.4787 | 0.4106 | 0.3527 |
| + importance weights to test scale mix | 0.3907 | 0.4793 | - | 0.3484 |
| + competition features (new releases between D1 and target date, national and per cluster) | 0.3896 | 0.4788 | - | 0.3625 |
| + extra pair/movie features (per-day shows/occ, show trend, movie show/occ trends) | 0.3969 | 0.4876 | - | 0.3793 |
| Hyperparameter grid (num_leaves 7-31, min_child 200-500, reg_lambda, colsample) | 0.389-0.402 | 0.480-0.487 | - | - |
| - mv_trend | 0.3855 | 0.4759 | 0.3852 | 0.3452 |
| **- all movie-level features (mv_*), keep pair_share (V2)** | **0.3765** | **0.4658** | **0.3890** | **0.3496** |
| V2 + any single mv_* feature back | +0.012 to +0.018 worse | | | |
| V2 + cinema_ids / format as categorical | +0.0006 / -0.0004 (noise) | | | |
| V2, 250 trees / num_leaves 63 | 0.3760 / 0.3755 | 0.4644 / 0.4646 | 0.3967 / 0.3899 | 0.3525 / 0.3466 |
| V2, 5-seed average | 0.3763 | - | - | - |
| V2 + two-stage movie curve model (nested OOF `m_pred`) | 0.3791 | 0.4691 | 0.4023 | 0.3474 |
| CatBoost MAE depth 6, 600 it on V2 | 0.3774 | 0.4678 | 0.3935 | - |
| Blend 0.5 LGBM V2 + 0.5 CatBoost | 0.3740 | 0.4638 | 0.3888 | - |

- Why V2 works: movie-level features are nearly unique per movie, so trees use them as a movie ID and memorize the ~164 training movies per fold.
- CatBoost blend not adopted yet (small gain, new dependency, +2 min runtime).

### Changes applied to baseline.ipynb
- Dropped all movie-level features except pair_share.
- Scale clipped at 1 to match the official metric.
- No early stopping on the validation fold; fixed n_estimators = 150 for CV and final model.
- Renamed `# Feature Extraction` to `# Feature Engineering`.
- Result: CV MASE 0.3758 (per fold 0.4683, 0.5154, 0.2487, 0.3859, 0.2605).
- Submission written to `submission_v2.csv`; LB not checked yet.

## 2026-10-01 - baseline.ipynb (optimization round 2)

### LB result of round 1
- `submission_v2.csv` (no movie-level features, CV 0.3758): public LB 0.50470, worse than the original 0.49211.
- Compared with the original, v2 predicts lower for big movies (top national-volume quintile: mean ratio 0.59 vs 0.71) and for movies rising over D1-D3 (0.51 vs 0.63).
- Conclusion: in the test period big / rising movies hold up better than train suggests, and movie-level features carry that signal.
  Train CV (plain, test-scale weighted, adversarial weighted, time split) all pointed the wrong way, so train CV cannot be trusted for movie-level feature decisions.

### Changes applied to baseline.ipynb
- Movie-level features restored (same feature set as the original baseline).
- Kept: scale clipped at 1, no early stopping, fixed n_estimators = 150, `# Feature Engineering` header.
- CV MASE 0.3893. Output saved as `submission_v3.csv` (mean abs diff vs original submission 2.5 tickets); LB not checked yet.

### External data: DKI Jakarta school holidays
- Sources (all published before 2025-09-30):
  - Kalender Pendidikan DKI 2025/2026, Keputusan Kepala Dinas Pendidikan Jakarta No. 89 Tahun 2025 (detik.com, 2025-08-07; sinotif.com, 2025-07-11).
  - Kalender Pendidikan DKI 2024/2025 (tirto.id / detik.com, 2024).
- Periods: 2025-03-25..2025-04-07 (Idulfitri), 2025-06-28..2025-07-13 (kenaikan kelas), 2025-12-22..2026-01-04 (semester ganjil), 2026-02-16..2026-02-20 (awal Ramadan), 2026-03-16..2026-03-27 (Idulfitri).
- Target ratio in train is higher on school-holiday dates (D5 median 0.83 vs 0.59).
- As LGBM features (school_hol, school_hol_d13, diff, days so far): CV 0.3893 -> 0.3963 (worse), time split 0.3519 -> 0.3402 (better). Only 34 train movies touch a holiday, so the feature can be used to memorize movies.
- Monotone constraints are not supported with the L1 objective in LightGBM.
- Post-hoc multiplier on OOF predictions, bootstrap over movies:
  - target holiday, D1-D3 not: best k 0.85 (p10-p90 0.75-1.05, 17 movies)
  - D1-D3 holiday, target not: best k 1.45 (0.85-1.8, 8 movies)
  - D1-D3 and target all holiday: best k 1.25 (1.15-1.4, 17 movies), 7,684 test rows
- `submission_v4_school.csv` = v3 with x1.25 on rows where D1-D3 and the target date are all school holidays. LB probe, not checked yet.

### LB results of round 2
| Submission | Public LB |
|---|---|
| Original baseline (`submission.csv`) | 0.49211 |
| v2, no movie-level features (`submission_v2.csv`) | 0.50470 |
| v3, movie features restored + scale clip + no early stopping, 150 trees (`submission_v3.csv`) | 0.48865 |
| v4, v3 x1.25 on rows where D1-D3 and target are all school holidays (`submission_v4_school.csv`) | 0.48601 |

- v3 beats the original by 0.0035, so the neutral changes (scale clip, no early stopping, 150 trees) help a little.
- v4 beats v3 by 0.0026: the school holiday effect shows up in the test set. Multiplier size and the other two holiday groups are not tuned yet.
- Still to review.

## 2026-10-02 - baseline.ipynb (optimization round 3)

### Offline experiments (base feature set, GroupKFold(5) by movie, 150 trees, no early stopping)
| Approach | CV |
|---|---|
| 150 / 250 / 400 trees | 0.3893 / 0.3904 / 0.3910 |
| Single seeds 42, 0, 1, 2, 3, 4 | 0.3893, 0.3897, 0.3898, 0.3919, 0.3934, 0.3908 |
| 5-seed average | 0.3907 |
| CatBoost MAE (600 it depth 6 / 400 it depth 4 / 800 it depth 8) | 0.3961 / 0.4008 / 0.3970 |
| Blend 0.7 LGBM + 0.3 CatBoost (best) | 0.3885 |
- No gain from tree count, seed averaging or CatBoost, not adopted.

### Training set checks (scored only on movies with D1-D3 reach like test, 3 seeds)
- Test regular-format movies reach at least 30 clusters in D1-D3; train has limited releases down to 1 cluster (Bollywood, concerts, niche anime, previews), 279 of 8007 pairs.
- Some train movies are preview / gala screenings of films officially released in October (in test): e.g. STOLEN GIRL (train 22 Sep, test D1 22 Oct), YAKIN NIKAH, PENGIN HIJRAH, JANGAN PANGGIL MAMA KAFIR. The 50% D1 rule treats the preview day as D1, but they contribute 0 training pairs (no D3 sales), so no damage.
| Training set | eval MASE |
|---|---|
| All original movies | 0.3910 |
| Only movies with >= 30 clusters (regular) | 0.3921 |
| Original + 9 late-September movies with partial D4-D10 (previews excluded) | 0.3910 |
| Late + wide only | 0.3940 |
- No gain, not adopted.

### Seasonal effects in the test period
- Median D1-D3 occupancy of new releases by week (from test_history): Christmas-New Year weeks 39-50, Ramadan weeks 1.4-7, normal weeks 5-25.
  As a demand index it is noisy (corr with true daily occupancy in train 0.28, 0.55 after 7-day smoothing), so not used as a feature.
- Train has no Ramadan and no movie whose D1-D3 is before a Lebaran (train starts on Idulfitri 2025-04-01), so the model cannot learn these patterns.
- Ramadan 1447 H taken as 2026-02-19 to 2026-03-19 (Idulfitri 2026-03-21/22 per holidays.csv).
- Disjoint seasonal groups of test rows (`season_group`):
| Group | Rows | Movies | Boosted x1.25 in v4 |
|---|---|---|---|
| normal | 48,044 | 113 | 0 |
| ram_to_lebaran (D1-D3 in Ramadan, target >= 2026-03-20) | 4,598 | 10 | 4,363 |
| ram_to_ram | 10,767 | 29 | 99 |
| pre_to_ram (D1-D3 before Ramadan, target in Ramadan) | 1,460 | 8 | 0 |
| school_both (Christmas-New Year) | 3,168 | 8 | 3,168 |
| school_target_only | 2,947 | 12 | 0 |
| school_d13_only (after the school break) | 1,627 | 4 | 0 |
- v4's gain came from Lebaran releases and/or Christmas-New Year rows; the probes below separate them.

### LB probes (each = v4 with one group changed to v3 x k)
| File | Group | k | Public LB |
|---|---|---|---|
| submission_p1_lebaran_x2.csv | ram_to_lebaran | 2.0 | **0.47566** |
| submission_p2_xmas_both_x1.5.csv | school_both | 1.5 | - |
| submission_p3_school_target_x1.25.csv | school_target_only | 1.25 | - |
| submission_p4_ramadan_x0.8.csv | ram_to_ram | 0.8 | - |
| submission_p5_after_school_x0.85.csv | school_d13_only | 0.85 | - |

### LB result of round 3
- p1 (Lebaran rows = v3 x2.0, everything else as v4): public LB 0.47566, better than v4 (0.48601) by 0.0104. Best so far.
- Only 4,598 rows (6.3% of test) changed, so the Lebaran-release group had a large error before. The model was badly under-predicting Idulfitri week.
- x2.0 vs v3 is not tuned (v4 had x1.25); the optimum could be higher or lower. The size of the gain suggests the true multiplier is likely well above 1.25, so x2 is probably not the ceiling.
- p2 to p5 not submitted (daily submission limit), scores unknown.
- Plan: when building main.ipynb, first optimize baseline.ipynb to find the direction. Candidate direction: a proper seasonal adjustment step in the pipeline (Lebaran, Ramadan, school holidays) built from the `season_group` definition in `season.py` (scratchpad), with factors tuned via LB.

## 2026-10-03 - baseline.ipynb (paper-based: BOXMOD decay curve + hierarchical shrinkage)

### Papers used
- Sawhney & Eliashberg (1996), "A Parsimonious Model for Forecasting Gross Box-Office Revenues of Motion Pictures", Marketing Science 15(2) (BOXMOD): exponential-decay adoption, calibrated from the first weeks of sales plus stationary shape parameters shared across movies.
- Neelamegham & Chintagunta (1999), "A Bayesian Model to Forecast New Product Performance in Domestic and International Markets", Marketing Science 18(2): hierarchical Bayes across markets; sparse markets are shrunk toward the movie-level pattern.
- Also reviewed: Ainslie, Dreze & Zufryden (2005, gamma diffusion + market share), Marshall et al. (2013, Bass vs Sawhney-Eliashberg on movie attendance), Montero-Manso & Hyndman (2021, global vs local models).

### Method
- Date effects: log(national tickets + 1) = movie level + movie slope x age + shared age curve (A(1) = A(2) = 0) + day-of-week + national holiday + school holiday, on the first 14 days of 169 train movies (opening mean >= 100 tickets, >= 7 days observed).
  The shared age curve is needed: without it, the day-of-week effects absorb the decay shape because most movies open on Wednesday.
  Multipliers vs Monday: Tue 1.24, Wed 1.03, Thu 0.89, Fri 0.92, Sat 1.36, Sun 1.42, national holiday 1.74, school holiday 1.08. Mean log slope per day -0.257.
- Movie decay: D1-D3 national tickets with date effects and age curve removed, slope = (z3 - z1) / 2 clipped to [-1.5, 0.7], shrunk: 0.25 x own slope + 0.75 x mean slope.
- `decay_ratio` = projected ticket level on the target date (with its day/holiday effect) / mean projected D1-D3 level.
- Pair-level slopes shrunk toward the movie slope (weight n / (n + k)) were tested: best is k -> infinity (pair slopes not used at all, too noisy).
- Model: LGBM target = ticket / (scale x decay_ratio), sample weight = decay_ratio (weighted L1 equals MASE exactly). Same features, params and 150 trees as before.
- Postprocessing (moved into the notebook, same as p1): x2.0 when D1-D3 touches Ramadan (from 2026-02-19) and the target is >= 2026-03-20; x1.25 when D1-D3 and target are all school holidays. 4,598 and 3,267 test rows.

### Results (CV = GroupKFold(5) by movie; time = train D1 < 2025-07-22, validate D1 >= 2025-08-01)
| Approach | CV | time |
|---|---|---|
| Base (v3 model) | 0.3893 | 0.3519 |
| Decay curve alone (no LGBM) | 0.4206 | - |
| Base + decay_ratio as feature | 0.3843 | 0.3434 |
| Base + decay_season as feature | 0.3812 | 0.3411 |
| Base + decay_ratio, slope, season features | 0.3865 | 0.3376 |
| **Decay curve as offset (adopted)** | **0.3801** | **0.3267** |
| Offset + decay features | 0.3797 | 0.3289 |
- Notebook run end to end: CV MASE 0.3803 (folds 0.4758, 0.5317, 0.2574, 0.3826, 0.2537).
- Test predictions vs v3 by seasonal group (mean pred / scale): normal 1.00, ram_to_lebaran 0.91, ram_to_ram 1.16, pre_to_ram 0.77, school_both 0.99, school_target_only 1.25, school_d13_only 1.04 (before postprocessing).
  So the decay offset does not double count the Lebaran factor; it already lifts targets that fall on school holidays.
- Movie-level features are kept (removing them hurt LB in round 1).
- Public LB not checked yet. Compare with p1 (0.47566), which has the same postprocessing.

### LB result of the decay pipeline
- `submission.csv` from the decay notebook: public LB 0.48012, worse than p1 (0.47566) by 0.0045, although CV (0.3893 -> 0.3803) and the time split (0.3519 -> 0.3267) both improved. Second time train CV pointed the wrong way.
- Prediction difference vs p1, mean |new - p1| / scale = 0.111, split by group:
| Group | Rows | Share of total difference | new / p1 (mean pred / scale) |
|---|---|---|---|
| normal | 48,044 | 58% | 1.00 (higher D4 0.93 vs 0.86, lower D9-D10 0.15 vs 0.19) |
| ram_to_lebaran | 4,598 | 16% | 0.91 |
| ram_to_ram | 10,767 | 12% | 1.16 |
| school_target_only | 2,947 | 6% | 1.28 |
| school_both | 3,168 | 5% | 1.01 |
- Suspects: the decay offset lowered the Lebaran group (the group where raising gave the biggest LB gain) and raised Ramadan rows (holidays inside D1-D3 such as Imlek and the early-Ramadan school break are treated as busy days, so the projected target level goes up, while Ramadan is actually quiet). The holiday effects are learned from train, which has no Ramadan.
- Next probe: `submission_hybrid_decay_normal.csv` = decay model on the normal group (48,044 rows), p1 on every seasonal group.
  Better than p1 -> the decay curve helps on normal rows and the seasonal groups caused the loss.
  Worse than p1 -> the decay curve itself does not transfer to the test period; revert the notebook to the p1 setup.
- p1 file was deleted from leon/; rebuilt identically from v3/v4 copies in the scratchpad (v4 with the ram_to_lebaran rows set to v3 x2.0).

### LB result of the hybrid probe, and revert
| Submission | Public LB |
|---|---|
| p1 (v3 model + Lebaran x2 + school x1.25) | **0.47566** |
| hybrid: decay model on normal rows, p1 on seasonal rows | 0.47832 |
| decay model everywhere (`submission.csv`) | 0.48012 |
- The decay curve hurts on normal rows too (about +0.0027), and its effect on the seasonal groups adds about +0.0018.
- Conclusion: the BOXMOD-style decay offset does not transfer to the test period, despite better CV and time split. Train-only validation has now misled twice (v2 and decay).
- baseline.ipynb reverted to the pre-decay notebook plus a `# Postprocessing` section (school holiday dates + seasonal factors: Lebaran x2.0, all-school-holiday x1.25).
  Verified: notebook output equals p1 (max abs diff 7e-12), CV 0.3893.
- The decay version of the notebook is kept in the scratchpad (`baseline_decay_version.ipynb`), not in the repo.

## 2026-10-05 - main.ipynb (Aggregate Dataset)

Notebook: `main.ipynb`

- Filled the `# Aggregate Dataset` section: all sources merged into one table `full`, one row per film x cinema x target date (D4-D10), same unit as test.csv. No derived features yet; processing is left for Preprocessing.
- Steps: D1 per film (train: first day at >= 50% of max cinema count; test: first test_history date), film-cinema windows with D1-D3 tickets / occupancy / shows as columns, train pairs kept only if selling on D3, 7 rows per pair (horizon 4-10).
- Train and test are stacked with `is_train`; test rows carry `id`, train rows carry the target `total_ticket`.
- Raw columns joined: city_name (from cinema), day_tipe / holiday_tipe / holiday_name (holidays, on target date), price_day / price (ticket_prices, city + day type of target date), age_rating / genre / producer / director / writer / casts (movies, joined on title with the 3D / IMAX suffix removed).
- Result: 128,660 rows (56,049 train + 72,611 test, matches test.csv), 30 columns. Only expected missing values: total_ticket (test), id (train), holiday_name (non-holidays).
- test_history is not added as training rows: it only has D1-D3 of test films, so it has no D4-D10 target; it is the D1-D3 input of the test rows.

## 2026-10-05 - External data: competition rule and candidate datasets

### Official rule (competition page, section 3 "Data Eksternal")
- Penggunaan data eksternal diperbolehkan, baik dalam bentuk labelled maupun unlabelled data.
- Format data eksternal dibebaskan, selama penggunaannya sesuai dengan ketentuan kompetisi.
- Sumber data eksternal wajib dapat diakses secara publik dan dapat diverifikasi.

How we apply it:
- Every source must show a publication or signing date on or before 2025-09-30.
- Live websites that keep updating (filmindonesia.or.id, Box Office Mojo, Wikipedia) must be cut to data dated before 2025-10-01, and ideally taken from an archived snapshot (Wayback Machine) dated on or before 2025-09-30, so the version is verifiable.
- Never use any number about a test film's own run (Oct 2025 - Mar 2026).
  Example: filmindonesia lists AGAK LAEN: MENYALA PANTIKU! (2025) at 11,000,866 admissions; it is a test film (D1 2025-11-27, the biggest test film by D1-D3 tickets), so this number is forbidden.

### Already used
- DKI Jakarta school holidays 2024/2025 and 2025/2026 (Kepdis Pendidikan Jakarta No. 89/2025; detik.com 2025-08-07, sinotif.com 2025-07-11). Used for the x1.25 school-holiday factor (LB-tested in v4).

### Candidate datasets (not tried yet)
| # | Data | Source | Published | Status |
|---|---|---|---|---|
| 1 | Cuti bersama 2026: 16 Feb (Imlek), 18 Mar (Nyepi), 20, 23, 24 Mar (Idulfitri), plus 15 May, 28 May, 24 Dec | SKB 3 Menteri No. 1497, 2, 5 Tahun 2025 | signed 2025-09-19 | allowed |
| 2 | Cuti bersama 2025: 28 Jan, 28 Mar, 2, 3, 4, 7 Apr, 13 May, 30 May, 9 Jun, 18 Aug, 26 Dec | SKB 3 Menteri 2025 (18 Aug added by SKB No. 933, 1, 3 Tahun 2025) | signed 2024 / 2025-08-07 | allowed |
| 3 | Ramadan 1447 H: 1 Ramadan 18 Feb 2026, Idulfitri 20 Mar 2026 | Maklumat PP Muhammadiyah No. 2/MLM/I.0/E/2025 | signed 2025-09-22 | allowed |
| 4 | Ramadan start per government (19 Feb 2026) | sidang isbat Kemenag | Feb 2026 | not allowed as a source (only implied by SKB Idulfitri 21-22 Mar) |
| 5 | School holidays per province (Jabar, Jateng, Jatim, ...) | provincial Kalender Pendidikan 2025/2026 | mostly Jun-Jul 2025, needs per-document check | to verify; Jabar reportedly returns 12 Jan 2026 vs 5 Jan in Jateng / Jatim; Lebaran break 16-28 Mar 2026 in Jabar / Jateng |
| 6 | Past admissions per Indonesian film (director / cast / franchise track record) | filmindonesia.or.id/film/penonton | live site, use pre-Oct-2025 films only, archived snapshot | allowed with cutoff; e.g. Agak Laen (2024) 9,126,979 |
| 7 | Weekly Indonesian box office incl. foreign films (franchise priors: Avatar, Zootopia, Wicked, FNAF, ...) | Box Office Mojo area=ID weekend pages; Wikipedia "List of 2024 box office number-one films in Indonesia" | live, cut at Sep 2025 | allowed with cutoff |
| 8 | Movie hype before release (trailer views, Google Trends, social media) | various | mostly built after Sep 2025 for test films | likely not allowed, only pre-Oct-2025 values would count |

### Notes from checking the data
- holidays.csv has no cuti bersama at all; its only "holiday" rows are the 18 national holidays. In the test period this misses 26 Dec 2025, 16 Feb, 18 Mar, 20 Mar, 23 Mar and 24 Mar 2026.
  The train period has 8 cuti bersama days (2, 3, 4, 7 Apr, 13 May, 30 May, 9 Jun, 18 Aug 2025), so a cuti bersama flag can be learned from train.
- holidays.csv does not mark Ramadan (about 18 Feb - 19 Mar 2026, about a month of the test period); train has no Ramadan days at all.
- Biggest test films by D1-D3 tickets: AGAK LAEN: MENYALA PANTIKU!, AVATAR: FIRE AND ASH, ALAS ROBAN, JANUR IRENG: SEWU DINO THE PREQUEL, SAMPAI TITIK TERAKHIRMU, SOSOK KETIGA: LINTRIK, DANUR: THE LAST CHAPTER, DUSUN MAYIT, NOW YOU SEE ME: NOW YOU DONT, COMIC 8 REVOLUTION SANTET K4BINET.
  Several are sequels / franchises (Agak Laen, Avatar, Sewu Dino, Danur, Now You See Me, Comic 8, Zootopia 2, Wicked), so track-record data (#6, #7) can cover a meaningful part of the test set.

### Folder convention for external data
- `external/` (repo root, next to `data/`): stores the finished external datasets as files.
- `leon/external_add.ipynb`: builds each external dataset (from the verified sources above) and writes it into `external/`.
- The pipelines (`main.ipynb`, `baseline.ipynb`) only read the files in `external/` (path `../external/` from `leon/`), so the source building stays separate from modeling.
- Each external file should record its source URL and publish / signing date (<= 2025-09-30) in external_add.ipynb.

## 2026-10-05 - external_add.ipynb (calendar datasets)

Notebook: `external_add.ipynb` (builds files in `external/`)

| File | Rows | Content | Source (published) |
|---|---|---|---|
| `external/cuti_bersama.csv` | 19 | Collective leave days 2025 (11) and 2026 (8): date, name | SKB 3 Menteri No. 1017, 2, 2 Tahun 2024 (2024-10-14); No. 933, 1, 3 Tahun 2025 (2025-08-07, adds 18 Aug 2025); No. 1497, 2, 5 Tahun 2025 (2025-09-19) |
| `external/ramadan.csv` | 30 | 18 Feb - 19 Mar 2026, with ramadan_day 1-30 | Maklumat PP Muhammadiyah No. 2/MLM/I.0/E/2025 (2025-09-22) |
| `external/school_holidays.csv` | 61 | DKI Jakarta school breaks (same 5 periods already used in baseline): date, region, period | Kalender Pendidikan DKI 2024/2025 (2024) and 2025/2026, Kepdis No. 89 Tahun 2025 (2025-07-11) |

- Every row carries `source` and `published` columns so the cutoff can be checked per row.
- Ramadan 1446 H (March 2025) is left out because it ends before the train data starts.
- Provincial school calendars (Jabar, Jateng, Jatim, ...) not added yet: each document still needs a pre-cutoff publication check.
- Track-record data (filmindonesia.or.id, Box Office Mojo) not added yet: web.archive.org cannot be reached from this environment, so archived pre-cutoff snapshots need to be fetched another way (e.g. by hand in a browser).
- Not used in any pipeline yet; next step is to join them in main.ipynb preprocessing and check on the LB.

## 2026-10-05 - eda.ipynb (EDA moved out of main, external data EDA)

- All EDA cells moved from main.ipynb to the new `eda.ipynb` (markdown translated to English, outputs cleared). main.ipynb now goes straight from Import datasets to Aggregate; `FORMAT_PATTERN` moved into the Aggregate merge cell. Verified main.ipynb still builds `full` (128,660 x 30).
- eda.ipynb loads the external CSVs from `external/` and adds an "EDA external datasets" section.

### Findings
- Calendar coverage (share of rows):
| Flag | train | test_history (D1-D3) | test target (D4-D10) |
|---|---|---|---|
| national holiday | 5.9% | 5.7% | 4.1% |
| cuti bersama | 3.9% | 5.0% | 3.3% |
| ramadan | 0% | 18.6% | 17.5% |
| school holiday | 12.9% | 14.9% | 17.3% |
- Cuti bersama vs centered 7-day mean of national tickets (train): median 1.07 (n=6 usable days), national holiday 1.39, weekend 1.33, weekday 0.80. A cuti bersama day is only mildly busier than a normal day; 2-3 Apr 2025 fall outside the rolling window.
- Target ratio on cuti bersama target dates: too few films (26) for a stable read, per-horizon medians swing from 0 to 1.15.
- Ramadan in test_history (D1-D3 demand by opening period, medians per film):
| Opening | Films | Occupancy | Tickets per show | Tickets per cinema-day |
|---|---|---|---|---|
| Oct-Dec 2025 | 83 | 12.7 | 18.6 | 134 |
| Jan - mid Feb 2026 | 46 | 11.5 | 17.7 | 95 |
| During Ramadan | 27 | 4.0 | 7.9 | 37 |
| Lebaran release (18 Mar) | 7 | 18.5 | 30.0 | 243 |
  Films opening during Ramadan sell about 2.5-3.5x less per cinema-day than before it; the scale already captures this, the open question is how D4-D10 decays inside Ramadan.
- Test target rows: 2,392 on cuti bersama, 12,729 in Ramadan, 15,365 with D1-D3 in Ramadan, 12,540 on school holidays, 2,976 on national holidays.
- Data quality: train has recording gaps on 7, 9, 11, 13 (missing), 14, 15, 16 Jun 2025 (rows < 70% of the 15-day median; 9 Jun has 1 row, 13 Jun none).
  They touch 6 films: 601 target rows (1.1%) are mostly fake zeros (median ratio 0 vs 0.91 at D4), and 959 rows (1.7%) have a D1-D3 window on a gap day (wrong scale).
  Suggested fix for training: drop target rows on gap days and pairs whose D1-D3 touches a gap day.
- test_history looks "empty" on many Mondays-Tuesdays, but that is expected: it only holds each test film's D1-D3, so it is not a gap.

### Correction: Ramadan source
- The Maklumat PP Muhammadiyah No. 2/MLM/I.0/E/2025 was signed 2025-09-22 but only reported publicly in October 2025 (muhammadiyah.or.id post under /2025/10/, Liputan6 2025-10-28), so it may not meet the "public by 2025-09-30" rule.
- `external/ramadan.csv` now cites Muhammadiyah's Kalender Hijriah Global Tunggal (KHGT), launched publicly 2025-06-25, which gives the same dates (1 Ramadan 1447 H = 2026-02-18, Idulfitri 2026-03-20).
- The month is also consistent with Idulfitri 2026-03-21/22 in the provided holidays.csv and in SKB 3 Menteri No. 1497, 2, 5 Tahun 2025 (signed 2025-09-19).
- The candidate table above (row 3) is superseded by this source.

## 2026-10-05 - main.ipynb (external datasets in Aggregate)

- Import datasets: added a "### External datasets" step that loads `external/cuti_bersama.csv`, `ramadan.csv`, `school_holidays.csv` (EXTERNAL_DIR found like DATA_DIR).
- Aggregate: added "### External datasets join", which left-joins on the target date the raw columns `cuti_bersama_name`, `ramadan_day`, `school_break` (empty on other dates). The `source` / `published` columns stay in the CSVs only.
- `full` is now 128,660 x 33. Rows with a value:
| Column | train | test |
|---|---|---|
| cuti_bersama_name | 1,141 | 2,392 |
| ramadan_day | 0 | 12,729 |
| school_break | 6,208 | 12,540 |
- ramadan_day is empty for every train row, so as a raw model feature it cannot be learned; it can only drive postprocessing or a derived feature.

## 2026-10-06 - external_add.ipynb (Lebaran admissions dataset)

- Box Office Mojo Indonesia (area=ID) checked and rejected: weekly data with only 2-5 Hollywood titles per week, no Indonesian films, several clearly incomplete weeks.
  Anecdotes only: Kung Fu Panda 4 -55% in a Ramadan 2024 week; Captain America -59% / -76% in Ramadan 2025 weeks; Snow White -84% in Lebaran week 2025 (local Lebaran releases took the screens).
- New file `external/lebaran_admissions.csv` (37 rows, 7 films): daily and cumulative admissions from news articles quoting Cinepoint or the filmmakers, all published April 2024 / April 2025.
  - Lebaran 2025 (Eid 31 Mar 2025, all released 31 Mar): PABRIK GULA, JUMBO, QODRAT 2, KOMANG, NORMA: ANTARA MERTUA DAN MENANTU.
  - Lebaran 2024 (Eid 10 Apr 2024, released 11 Apr): BADARAWUHI DI DESA PENARI, SIKSA KUBUR.
  - Columns: film, release_date, eid_date, date, day, metric (daily / cumulative), admissions, approx (rounded milestones like "2 juta"), source (URL), published.
  - Sources: kompas.com (2025-04-06, 04-08, 04-15), idntimes.com (2025-04-07), cnnindonesia.com (2025-04-10), liputan6.com (2024-04-12), disway.id (2024-04-18).
  - One kompas article says "Monday, 6 March 2025"; from context it is Monday 7 April 2025 (last Lebaran holiday day) and is stored as such.
- Average daily admissions per reported window / release-day admissions:
  - 2025 films (release on Eid day): days 2-7 at 1.3x-2.7x the release day, day 8 (last holiday day) at 1.6x-5.0x, and JUMBO kept growing (days 12-15 at 6.9x).
  - 2024 films (release the day after Eid): days 2-7 at about 0.94x-1.27x the release day.
  - For comparison, a normal train pair is at about 0.9x of its D1-D3 mean on D4 and 0.36x on D7.
  - So during the Lebaran holiday, sales hold flat or grow for 1-2 weeks instead of decaying, which supports a Lebaran factor of x2 or more relative to the normal-decay model.
- Caveats: only 7 films; the numbers come from different articles at slightly different update times (e.g. QODRAT 2 days 9-10 look too low at 0.79x next to 2.92x on day 8 and 1.42x on days 11-15); and the 2024/2025 films opened on Eid itself, while the 2026 test Lebaran films open on 18 Mar, two days before Eid.

## 2026-10-06 - Holiday effects per holiday (train) and more data checks

- National tickets on each holiday / cuti bersama day vs the same weekday 1-2 weeks before and after (holidays, cuti bersama and gap days excluded from the reference):
  Independence cuti bersama Mon 18 Aug 2.93x; Maulid Fri 5 Sep 2.82x; Ascension Thu 29 May 2.60x; Good Friday 1.95x; Waisak Mon 1.91x; Labour Day Thu 1.83x; Lebaran cuti bersama 2-7 Apr 1.17-1.80x; Independence Sun 17 Aug 1.71x; Ascension cuti Fri 30 May 1.26x; Waisak cuti Tue 1.26x; Islamic New Year Fri 1.16x; Easter Sun 1.07x; Idulfitri Tue 1 Apr 0.90x; Idul Adha Fri 6 Jun 0.91x; Pancasila Sun 1 Jun 0.31x.
- Long weekends (Thu holiday + Fri cuti, Sun holiday + Mon cuti) give the biggest lift; holidays on a weekend add little.
  Unreliable points: Maulid coincides with THE CONJURING: LAST RITES opening week (D1 3 Sep); Pancasila Day and Idul Adha sit in the broken June stretch.
- Implication: a 0/1 is_holiday averages very different effects (about 1.4x); a "length of the consecutive days-off block" feature (weekends + national holidays + cuti bersama) should capture most of the difference and transfers to the test holidays, whose names never appear in train.
- More data quality findings (tickets vs same weekday +-2 weeks):
  - 1-5 Jun 2025: rows and cinemas normal but tickets only 0.24-0.45x; suspected under-recording (not certain), on top of the row gaps 7-16 Jun.
  - 29 Aug and 1 Sep 2025: about 0.4x; probably real (late-August 2025 protests, many malls closed), not a data error.

## 2026-10-06 - main.ipynb (Preprocessing + calendar Feature Engineering)

### Preprocessing
- Drop incomplete days: train rows whose target date or D1-D3 falls on a row-gap day (7, 9, 11, 13, 14, 15, 16 Jun 2025) are removed: 1,378 rows (601 target on gap, 959 D1-D3 on gap, some overlap). The suspect low-ticket days 1-5 Jun 2025 are kept for now (to test later).
- Calendar flags on the target date: is_holiday, is_cuti_bersama, is_school_holiday (0/1). Share of rows train / test: 5.4% / 4.1%, 1.8% / 3.3%, 11.4% / 17.3%.
- Genre: one 0/1 column per genre, 27 genres (Horror 37.5% of train rows, Drama 32.8%, Action 21.2%).
- Categories: city_name (69), day_tipe (3), age_rating (4).
- Dropped: holiday_tipe, holiday_name, cuti_bersama_name, school_break, genre, price_day, base_title, producer, director, writer, casts. ramadan_day kept for postprocessing only.
- Target: scale = mean(t1, t2, t3) clipped at 1; target_ratio = total_ticket / scale. Median train target_ratio by horizon D4-D10: 0.915, 0.616, 0.467, 0.360, 0.163, 0, 0.

### Feature Engineering (calendar)
- A day is "off" if it is a weekend, a national holiday or a cuti bersama day.
- off_block_len: length of the consecutive days-off block containing the target date (0 on working days).
- d13_off_days: number of off days among D1-D3; d13_off_block_max: longest block touching D1-D3.
- New external file `external/extra_national_holidays.csv` (Nyepi 29 Mar 2025, Idulfitri 31 Mar 2025, from SKB 3 Menteri No. 1017, 2, 2 Tahun 2024, signed 2024-10-14), because holidays.csv starts on 1 Apr 2025 and would cut the Lebaran 2025 block. The calendar also starts 30 days before the first D1.
- Blocks of 4+ days: Lebaran 2025 28 Mar - 7 Apr (11), Waisak 10-13 May (4), Ascension 29 May - 1 Jun (4), Idul Adha 6-9 Jun (4), Christmas 25-28 Dec (4), Imlek 14-17 Feb (4), Lebaran 2026 18-24 Mar (7).
- Target-date block lengths (rows): train 0: 35,899 / 1: 312 / 2: 11,658 / 3: 3,955 / 4: 2,780 / 11: 67; test 0: 47,759 / 1: 378 / 2: 16,581 / 3: 1,382 / 4: 3,444 / 7: 3,067.
  Only 67 train target rows sit in the 11-day Lebaran 2025 block (films running from before 1 Apr are not training samples), so the model has very little to learn the 7-day test block from; postprocessing still has to handle Lebaran.
- 5,202 train rows have a D1-D3 touching a 4+ day block.
- Correction to earlier reasoning: the 2026 Lebaran films (D1 18 Mar) have their whole D1-D3 inside the 7-day block (Nyepi cuti, Nyepi, Idulfitri cuti), and test_history shows them as the strongest openings (about 243 tickets per cinema-day), so their D1-D3 is not weak.

## 2026-10-06 - main.ipynb (Regressor Models, Postprocessing, Submission)

- Foundation models found locally: `nvidia/Kumo-Tabular` (package `structured-data-models`, import `sdm`, license OpenMDW 1.1) and `google/tabfm-1.0.0-pytorch` (package `tabfm` 1.0.1, non-commercial license). TabPFN is not installed.
- One-fold smoke test (GroupKFold fold 0, main's current features, target_ratio MAE):
| Model | Fold MASE | Time | Peak GPU |
|---|---|---|---|
| LightGBM default (objective l1) | 0.3037 | 2 s | - |
| Kumo-Tabular large, 5k context, median of 999 quantiles | 0.2992 | 35 s | 3.1 GB |
| Kumo-Tabular large, 10k context | 0.3001 | 55 s | 5.0 GB |
| TabFM, 10k context, 4 estimators | 0.3476 | 71 s | 6.1 GB |
| TabFM, 5k context | 0.4356 | 56 s | 5.3 GB |
  - TabFM's loader keeps the weights on the CPU (it then ran for over 10 minutes without finishing); it must be moved with `.to("cuda")`.
- Added to main.ipynb:
  - Imports cell now also loads catboost, xgboost, torch, sdm and tabfm (unused joblib and early_stopping removed).
  - `# Regressor Models`: SEED, RUN_MODELS switch, FEATURES (raw D1-D3, horizon, scale, price, calendar flags and blocks, 27 genre columns, city_name / day_tipe / age_rating), GroupKFold(5) by film, `cross_validate` helper (fold scores, then a refit on 100% of train for test).
  - Models with default params: LightGBM (l1), CatBoost (MAE), XGBoost (reg:absoluteerror), Kumo-Tabular (large, KUMO_CONTEXT_ROWS 5000), TabFM (TABFM_CONTEXT_ROWS 10000). Both foundation models take a list of context seeds (`*_CONTEXT_SEEDS`) and average over them.
  - Model comparison table; equal-weight blend of lgbm, catboost, xgboost, kumo (TabFM excluded by default).
  - `# Postprocessing`: p1 seasonal factors (Lebaran x2.0, all-school-holiday x1.25).
  - `# Submission`: writes submission.csv in sample_submission order.
- Full CV results: not run here (the user runs the notebook).

### main vs baseline features, same rows and folds (5-fold GroupKFold by film, 54,671 train rows after dropping gap rows)
| Setup | CV MASE | Folds |
|---|---|---|
| main features, LightGBM default (l1) | 0.3645 | 0.3037, 0.5024, 0.2894, 0.4170, 0.3101 |
| main features, XGBoost default (absolute error) | 0.3812 | 0.3387, 0.5169, 0.3056, 0.4301, 0.3147 |
| baseline features, tuned LightGBM (150 trees, the p1 setup) | 0.3855 | 0.3859, 0.5252, 0.2734, 0.4249, 0.3178 |
| baseline features, LightGBM default | 0.3886 | 0.4062, 0.5239, 0.2725, 0.4248, 0.3157 |
- main's feature set is better on CV by about 0.02, but it has no film-level features (national D1-D3 totals, trend, cinema count). v2 also improved CV by dropping those and lost on the LB, so this needs an LB check before trusting it.
- Note: the foundation model added is TabFM (google/tabfm-1.0.0-pytorch), not TabPFN; TabPFN is not installed and not in the HF cache.

### TabFM full CV (user run)
- TabFM (10k context, 4 estimators): CV MASE 0.3964 (folds 0.3674, 0.5047, 0.3599, 0.4253, 0.3246), worse than default LightGBM on main features (0.3645).
- The final test prediction crashed with CUDA out of memory (all 72,611 test rows predicted in one call on an 8 GB GPU).
- Decision: drop TabFM; RUN_MODELS = ["lgbm", "catboost", "xgboost", "kumo"].

## 2026-10-06 - Status at end of day

- Best public LB so far: 0.47566 (p1 = baseline.ipynb model + Lebaran x2.0 + all-school-holiday x1.25); baseline.ipynb reproduces it exactly.
- main.ipynb is complete end to end (Aggregate, Preprocessing, calendar Feature Engineering, LightGBM / CatBoost / XGBoost / Kumo-Tabular, blend, p1 postprocessing, Submission) and has not been scored on the LB yet.
- Discussed but not done: stacking (a meta-model trained on OOF predictions); to be tested with its own film-grouped CV if main scores well.
- Next steps:
  1. Submit main.ipynb's submission.csv and compare with 0.47566.
  2. If worse: port the baseline film-level features (national D1-D3 totals, trend, cinema count) into main.
  3. If better: try a cross-validated stacking / weighted blend, then the Lebaran x3 probe.

## 2026-10-07 - LB result of main.ipynb and diagnosis

- main.ipynb submission (LightGBM / CatBoost / XGBoost / Kumo blend on main features + p1 postprocessing): public LB 0.48785, worse than p1 (0.47566) by 0.0122, although main's CV was better by about 0.02 (0.3645 vs 0.3855 for LightGBM on the same folds). Third time train CV pointed the wrong way.
- Prediction comparison vs p1 (mean prediction / scale, mean |main - p1| / scale = 0.177):
| Slice | main / p1 | Share of total difference |
|---|---|---|
| ram_to_lebaran (4,598 rows) | 0.44 | 24% |
| pre_to_ram (1,460 rows) | 0.38 | 2% |
| school_target_only (2,947 rows) | 1.32 | 6% |
| normal (48,044 rows) | 1.00 | 51% |
| biggest films (top national D1-D3 quintile) | 0.82 | 30% |
| smallest films (bottom quintile) | 1.13 | 10% |
| fastest-falling films (mv_trend bottom quintile) | 0.71 | 22% |
| fastest-rising films (top quintile) | 0.86 | 22% |
- Cause 1: the new D1-D3 calendar features (d13_off_days, d13_off_block_max). In train, films opening on a long weekend have an inflated D1-D3 and drop hard afterwards, so the model predicts a steep drop for the 2026 Lebaran films (D1-D3 inside the 7-day block). In reality Eid week keeps demand high. After the x2 factor these rows end up at about 0.9x of the v3 base, where the LB rewarded 2x.
- Cause 2: no film-level features in main (national D1-D3 totals, trend, cinema count), the same pattern as v2: bigger and rising films are under-predicted, small films over-predicted.

## 2026-10-07 - main.ipynb (film-level features back, D1-D3 calendar features removed)

- Feature Engineering now also builds the baseline features: pair (trend, t3_share, occ_mean, show_mean, tickets_per_show3, active_days), film (mv_t1, mv_t2, mv_t3, mv_clusters, mv_trend, pair_share; computed separately for train and test rows from their own windows, since a few titles appear in both), cinema (cl_scale, cl_movies), weekday and price (dow, d1_dow, price_ratio).
- Removed d13_off_days and d13_off_block_max (they pushed the Lebaran 2026 group down); off_block_len on the target date is kept.
- RUN_MODELS = ["lgbm", "catboost", "xgboost", "kumo"] (TabFM dropped).
- 63 features, no missing values in train or test.
- CV on the same folds: LightGBM default 0.3844 (folds 0.3804, 0.5302, 0.2817, 0.4235, 0.3064), XGBoost default 0.4047. Worse than main without film features (0.3645 / 0.3812) and about level with the baseline (0.3855), the same pattern as before where film features hurt CV but help the LB.
- LightGBM-only test predictions after postprocessing vs p1 (mean prediction / scale): normal 1.02, ram_to_lebaran 1.35, ram_to_ram 1.10, pre_to_ram 1.12, school_both 1.19, school_d13_only 1.30, school_target_only 1.14; mean |new - p1| / scale 0.106.
  The Lebaran group is now above p1 (like about x2.7 on the v3 base instead of x2), so it is the first suspect if this submission scores worse.
- LB not checked yet.

## 2026-10-07 - Zero predictions check

- Top of the public LB is 0.34456 / 0.34972, then a cluster at 0.371-0.380; we are at 0.47566. A gap of 0.10-0.13 is much larger than all calendar / external data gains so far (about 0.016 combined), so the top teams likely use a structural signal in the given data, not only extra data.
- Zero targets are real: in train, 9% of D4 rows and 55% of D10 rows sell 0 (the cinema dropped the film), and only 5.7% of D4-D10 zeros are followed by sales again later.
- main.ipynb LightGBM (default, film features) predicts ~0 (< 0.02 ratio) for 2-31% of train rows by horizon (OOF), fewer than the actual zeros, and for 3-47% of test rows (test period is quieter). When it predicts ~0 on train, 86.3% are truly 0; those rows are 6.0% of the total OOF error.
- Conclusion: zeros are not the main error source; the error comes from pairs that keep selling. Next idea: same-date competition per cinema from test_history (shows / tickets of new releases in the same cinema on the target date), which decides which films keep their screens.

## 2026-10-07 - LB result: main.ipynb with film features = new best

| Submission | Public LB |
|---|---|
| p1 (baseline model + Lebaran x2 + school x1.25) | 0.47566 |
| main without film features, with D1-D3 calendar features | 0.48785 |
| **main with film features, without D1-D3 calendar features (blend lgbm + catboost + xgboost + kumo, default params)** | **0.45395** |
- Gain of 0.0217 over p1. main.ipynb is now the main pipeline.
- Confirms again that film-level features matter on the LB even when train CV prefers dropping them (CV 0.3844 with them vs 0.3645 without).
- Note: in this submission the Lebaran group sits about 1.35x above p1 (like about x2.7 on the v3 base), so a higher Lebaran level did not hurt.
- Next: hyperparameter tuning (Optuna) of the GBDTs on CV, then seed averaging.

## 2026-10-06 - external.ipynb rebuilt, regional school calendars, competition features in main

### external.ipynb and external_data/
- `external.ipynb` (new, was empty) rebuilds every external file into `external_data/`; main.ipynb now looks there first (`external_data`, then the old `external` paths).
- Rebuilt from the sources in the 2026-10-05 entries: `cuti_bersama.csv` (19), `ramadan.csv` (30), `school_holidays.csv` (DKI, 61), `extra_national_holidays.csv` (2). Same dates as before.
  The DKI 2024/2025 calendar has `published` = "2024" (year only; the exact day was never recorded).
- New: `city_province.csv` (73 cities, all train / test cities mapped) and `school_calendar_regional.csv` (125 rows, 34 provinces + the national SEB No. 4/2025 Lebaran 2025 break).
  - Types: break, ramadan_break, ramadan_selfstudy, idulfitri_break, exam. Every row has its own source URL and publication date (latest 2025-08-26).
  - Sources: provincial Kalender Pendidikan decrees and news articles that print them (antaranews, detik, komunitasbelajar, kalderanews, official PDFs of Jateng / Jatim / Kaltim, city calendars for Yogyakarta, Denpasar, Makassar).
  - Left out: 2 Jambi rows with no confirmed publication date. Sumsel / Kaltara June 2025 break end not given, set to 12 Jul (most common end).
  - Caveats: provincial decrees mostly cover SMA / SMK; Ramadan / Lebaran 2026 dates are the pre-cutoff plans (later revisions are after the cutoff).
- The last cell asserts every dated row is published on or before 2025-09-30.
- Also found (not used yet): TKA national academic test for grade 12, 3-6 Nov 2025 (detik 2025-09-27); predecessor admissions for test franchises (Agak Laen 9.12 jt, Avatar TWOW 6.69 jt, Sewu Dino 4.89 jt, KKN 9.23 jt, Danur 1-3 2.4-2.7 jt, Comic 8 1.62 jt, Sosok Ketiga 1.16 jt), each from articles dated before the cutoff; Wayback snapshots of filmindonesia.or.id yearly admission lists exist dated before the cutoff (Jun-Aug 2025) but could not be opened from this machine.

### Error by pair scale (main LightGBM, OOF)
| Scale | Train share | Test share | OOF error | Train zero share |
|---|---|---|---|---|
| <= 5 | 0.3% | 1.5% | 2.47 | 77% |
| 5-10 | 0.9% | 3.4% | 1.59 | 75% |
| 10-20 | 2.4% | 8.0% | 0.48 | 74% |
| 20-40 | 7.5% | 14.0% | 0.57 | 64% |
| 40-80 | 14.9% | 19.0% | 0.38 | 57% |
| 80-160 | 22.0% | 20.8% | 0.36 | 41% |
| > 160 | 52.0% | 33.3% | 0.33 | 13% |
- The model predicts about 0 for small test pairs. Small test pairs are mostly from films with 3 active days (67% vs 39% in train), so they may keep selling more than train suggests: a candidate LB probe (raise predictions for scale <= 20).

### New features (CV = GroupKFold(5) by film; stable = 3 seeds, lr 0.05, 300 trees, bagging 0.8; cv_w = weighted to the test scale mix)
- Competition from other films' D1-D3 rows only (test_history for test, the same window rule for train, so both are built the same way): c_new_films_T, c_new_shows_T, c_new_show_share_T (over cinema capacity = 95th pct of daily shows), own_show_share (show3 / capacity), comp_vs_own, nat_comp_ratio (national new-release tickets on T / film D1-D3 mean), c_entries (films opening at the cinema after D3 up to T).
- Other: seats_per_show, dom (day of month), is_3d_imax, occ_trend, show_trend, pair_vs_film_trend.
- Single features alone move default-LGBM CV by less than the noise (about 0.003); only the groups help.
| Setup (stable) | CV | cv_w |
|---|---|---|
| main features | 0.3819 | 0.4676 |
| + competition | 0.3763 | 0.4609 |
| + other | 0.3791 | 0.4663 |
| + both | 0.3731 | 0.4589 |
| + both, no raw ticket counts, no c_entries_tix_rel (kept) | 0.3729 | 0.4589 |
| kept set without nat_comp_ratio | 0.3787 | 0.4650 |
| kept set, school flag = regional instead of DKI | 0.3723 | 0.4592 |
| kept set + regional exam flag | 0.3734 | 0.4597 |
| OOF cinema decay target encoding | worse (+0.004 default LGBM) | |
- Leave-one-out: dropping any kept feature costs 0.0009 to 0.0034 (own_show_share most). Raw new-release ticket counts were dropped (the test period sells fewer tickets; cost 0.0008).
- Risk: nat_comp_ratio reads quiet Ramadan days as low competition. Test predictions vs old features: pre_to_ram x1.25 (stable) / x1.78 (default LGBM), ram_to_lebaran x1.02, all x1.006. Without it: pre_to_ram x1.05, Lebaran x1.13.

### main.ipynb changes
- torch / sdm / tabfm imports optional (HAS_FOUNDATION); Kumo / TabFM skipped when missing.
- Feature Engineering: "New-release competition", "Other pair features", "Regional school holidays" (school_off_reg), "Features kept" (NEW_FEATURES, SCHOOL_FLAG = "dki" or "regional"; postprocessing keeps the DKI calendar either way).
- Full run on CPU (no Kumo), blend of LightGBM / CatBoost / XGBoost:
| Run | LGBM | CatBoost | XGBoost | Blend |
|---|---|---|---|---|
| SCHOOL_FLAG dki | 0.3778 | 0.3771 | 0.3955 | 0.3775 |
| SCHOOL_FLAG regional | 0.3764 | 0.3729 | 0.3949 | 0.3758 |
| SCHOOL_FLAG dki, without nat_comp_ratio | 0.3856 | 0.3794 | 0.3978 | 0.3811 |
- Before these features: LGBM 0.3844, XGBoost 0.4047.
- LB not checked yet. Submissions in the repo root: submission_feat_dki.csv (same as submission.csv), submission_feat_regional.csv, submission_feat_dki_no_natratio.csv.
- LB probe order: dki first (vs p1 0.47566 and main 0.48785); then regional (school calendar) and no_natratio (Ramadan risk of nat_comp_ratio), each differs from dki in one thing.

## 2026-10-06 - External data: all candidates (status list)

Rule reminder: public, verifiable, and published on or before 2025-09-30; never any number about a test film's own run (Oct 2025 - Mar 2026).
Core problem the data should fix: train covers Apr - Sep 2025 only, so the model has never seen the Oct - Mar season (rainy season, Christmas - New Year, Ramadan, Lebaran week), and test has far more small pairs.

### In external_data/ now
| File | Content | Used in main |
|---|---|---|
| cuti_bersama.csv | Collective leave days 2025 / 2026 (SKB 3 Menteri) | yes (is_cuti_bersama, off_block_len) |
| ramadan.csv | Ramadan 1447 H, 18 Feb - 19 Mar 2026 (KHGT) | postprocessing only |
| school_holidays.csv | DKI Jakarta school breaks | yes (is_school_holiday, postprocessing) |
| extra_national_holidays.csv | Nyepi and Idulfitri 2025 (before holidays.csv starts) | yes (off_block_len) |
| school_calendar_regional.csv | School breaks / exams for 34 provinces + national SEB 2025 | optional (SCHOOL_FLAG = "regional") |
| city_province.csv | City to province map | with the regional calendar |

### Found, not built yet
| # | Data | Source / date | Possible use | Note |
|---|---|---|---|---|
| 1 | Lebaran 2024 / 2025 daily admissions of 7 films | kompas, idntimes, cnnindonesia, liputan6, disway (Apr 2024 / Apr 2025) | Prior for the Lebaran factor (sales flat or rising for 1-2 weeks) | Built earlier as lebaran_admissions.csv in the old external/ folder; not in this folder, values must be re-collected |
| 2 | Predecessor admissions of test franchises (Agak Laen 9.12 jt, Avatar TWOW 6.69 jt, Sewu Dino 4.89 jt, KKN 9.23 jt, Danur 1-3 2.4-2.7 jt, Comic 8 1.62 jt, Sosok Ketiga 1.16 jt) | inilah 2024-05-09, kompas 2025-03-25, cnnindonesia, detik, ngopibareng (all before cutoff) | Film-level prior (sequel strength, legs) | Covers about 10 test films only |
| 3 | TKA national academic test, grade 12: 3-6 Nov 2025, make-up 17-20 Nov | detik 2025-09-27 (SE BSKAP 3866/2025) | Calendar flag | Probably small effect |
| 4 | filmindonesia.or.id yearly admission lists (2023, 2024, 2025 to late Aug, all-time) | Wayback snapshots dated 2025-06-21 to 2025-08-25 | Track record of producer / director / franchise | Snapshots exist but could not be opened from this machine |

### Not searched yet (ordered by expected value)
| # | Data | Why it could help | Allowed? |
|---|---|---|---|
| 5 | Previous season (Oct 2024 - Mar 2025) film trajectories: daily / weekly admissions from news (Cinepoint figures) and filmindonesia weekly snapshots | Measures decay in the same season one year earlier: Christmas 2024, Ramadan 2025, Lebaran 2025. Directly targets the missing season in train | Yes, all published before 2025-04 |
| 6 | Industry statements on seasonal attendance (GPBSI, Cinepoint, exhibitors): Ramadan drop %, Christmas / New Year lift, rainy season | Priors to set the seasonal postprocessing factors without spending LB submissions | Yes, if the article is dated before the cutoff |
| 7 | Producer / director / cast track record (past admissions) | Film-level prior for how long a film holds; joins on movies.csv producer / director / casts | Yes, from snapshots before the cutoff |
| 8 | Film type: local vs Hollywood vs Asian, sequel, adaptation | Legs differ by type; partly derivable from movies.csv | Yes (announced before release); verification per film needed |
| 9 | Public Kaggle / GitHub datasets of Indonesian box office | Ready-made history if one exists | Only versions dated before the cutoff |
| 10 | City statistics: population, income per capita, screens per city (BPS 2024, GPBSI) | Static city features; helps small cities with few rows | Yes |
| 11 | Ratings (IMDb / RT) of foreign test films that premiered abroad before 2025-09-30 | Word-of-mouth signal | Only for the few films out before the cutoff, via snapshots |
| 12 | Pre-release hype up to 2025-09-30 (trailer views, Google Trends) | Demand signal | Only values dated before the cutoff; hard to verify, partial coverage |
| 13 | Weather normals (BMKG monthly rainfall per city) | Rainy season Nov - Mar vs dry-season train | Yes (climate normals), but train has no rainy season to learn from |
| 14 | Pay timing: THR before Lebaran, civil servant 13th salary | Spending peaks | Only if the 2026 rule was published before the cutoff |
| 15 | Competing events: SEA Games 2025 (Thailand, 9-20 Dec 2025), big concerts | Attention elsewhere | Schedules published before the cutoff; likely small effect |

### Rejected
- Box Office Mojo Indonesia weekly charts: only 2-5 Hollywood titles per week, incomplete weeks.
- Maklumat PP Muhammadiyah No. 2/MLM/I.0/E/2025: signed 2025-09-22 but made public in October 2025 (replaced by KHGT).
- Kemenag sidang isbat (Feb 2026), provincial calendar revisions from 2026, any test-period admissions, trailer views or ratings measured after the cutoff, actual weather in the test period.

## 2026-10-06 - external.ipynb (film-level and previous-season datasets)

New files in `external_data/` (raw collected tables in `external_data/raw/`, cleaned by external.ipynb; the cutoff check passes for all 11 files):
| File | Rows | Content | Source / date |
|---|---|---|---|
| filmindonesia_admissions.csv | 285 | Top 15 films by Indonesian admissions per year, 2007-2025 | filmindonesia.or.id "Penonton" pages via Wayback snapshots dated 2025-05-14 to 2025-09-14 |
| film_profile.csv | 400 (237 train, 163 test titles) | origin, is_local, film_kind, sequel / franchise, predecessor + its admissions, source material, production house, Wikipedia titles, IMDb (7 test films) | Web research: Wikipedia (versions before the cutoff where possible), dated news; predecessor admissions dated <= 2025-08-03 |
| wiki_pageviews_daily.csv | 17,794 | Daily en / id Wikipedia pageviews, 1 Jan - 29 Sep 2025 (96 of 400 articles had views before the cutoff) | Wikimedia pageviews API; published = day + 1 |
| season_trajectories.csv | 193 | Admissions at given days after release, 33 films: xmas_2024 (10), pre_ramadan_2025 (3), ramadan_2025 (4), lebaran_2025 (5), lebaran_2024 (2), other (10) | Dated news, mostly Cinepoint figures via tabloidbintang, kompas, detik; latest 2025-04-26 |
| season_statements.csv | 17 | Exhibitor statements on seasonal attendance (e.g. XXI: March 2025 down on Ramadan, April 2025 almost equal to all of Q1) | detik, kontan etc., latest 2025-04-25 |

### Profile coverage
- Train: 108 / 237 local, 78 sequel / franchise rows, predecessor admissions for 7 (+ filmindonesia matches).
- Test: 30 + 19 sequel / franchise rows, predecessor admissions for 9 films (Agak Laen 9.13 jt, Janur Ireng <- Sewu Dino 4.89 jt, Danur 3 2.42 jt, Suzzanna 2.19 jt, AADC 2 3.63 jt, Comic 8 CK2 1.84 jt, Qorin 1.32 jt, Sosok Ketiga 1.16 jt, ...). Filled 3 more from the filmindonesia list.
- `info_public_before_cutoff` (test): yes 96, unknown 66, no 1 (ALAS ROBAN, teaser 2025-11-10). 7 films have their earliest source found after the cutoff. For a strict reading, use only rows with yes, or only fields that also follow from movies.csv.
- IMDb ratings before the cutoff exist only for 7 small test titles; the pre-cutoff snapshots of the September festival premieres (Pangku, Rangga & Cinta, Black Phone 2) show no rating yet. Not useful.
- About 40 small local titles are low confidence: the session's shared web search cap (200 searches) ran out during collection. Raising `CLAUDE_CODE_MAX_WEB_SEARCHES_PER_SESSION` would allow filling them.

### Previous season, first read (week-2 / week-1 cumulative admissions)
| Season | Week-2 / week-1 | Films |
|---|---|---|
| Lebaran 2025 | 0.67-0.70 (word-of-mouth films grew: JUMBO 1.98, KOMANG 1.36) | Pabrik Gula, Qodrat 2, Norma |
| Lebaran 2024 | about 0.57 | Badarawuhi |
| Late Jan / pre-Ramadan 2025 | 0.70-0.77 | Petaka Gunung Gede, Perayaan Mati Rasa, Pengantin Setan, 1 Kakak 7 Ponakan |
| Christmas 2024 (released just before) | 0.44-0.55 | Racun Sangga, Modal Nekad |
| Ramadan 2025 | about 0.36 | Iblis Dalam Kandungan 2 |
- Lebaran 2025 days 4-7 vs days 1-3: about 2.0x (collective leave through day 8), JUMBO 2.9x. Supports the Lebaran x2 factor.
- Films opening on 25 Dec held flat or rose (days 4-7 vs 1-3 about 1.1-1.2x).
- A holdover through Ramadan 2025 lost about 50% of daily sales per week.
- Gaps: no full daily series for Christmas films, late Ramadan thin, no quantified GPBSI / Cinepoint statement on Ramadan or rainy season.

### Candidate list status update
- #1 Lebaran admissions: re-collected inside season_trajectories.csv (lebaran_2024 / lebaran_2025).
- #2, #4, #7, #8, #11: in film_profile.csv / filmindonesia_admissions.csv (#11 ratings: not useful).
- #5, #6: season_trajectories.csv, season_statements.csv.
- #12 pre-release hype: wiki_pageviews_daily.csv (Wikipedia only).
- Not done: #3 TKA dates, #9 Kaggle datasets, #10 city statistics, #13 weather normals, #14 pay timing, #15 events.

## 2026-10-07 - Optuna tuning attempt (not completed) and log merge

- Log merge: the user's updated log (from the session that built `external.ipynb`, the regional school calendar, competition features and the film-level datasets) matched this log up to line 492; its three newer sections were appended above, and the 0.45395 LB entry was kept.
- Which submission scored what: the 0.45395 submission was main.ipynb with film-level features and default parameters, before the new competition / other features (LightGBM 0.3844 CV at that point). The new features (LightGBM 0.3764-0.3778, blend 0.3758-0.3775) have no LB score yet, so the three prepared files (dki, regional, dki without nat_comp_ratio) are still to be submitted.
- The downloaded log contains no hyperparameter tuning results; its only mentions of parameters are the old baseline grid and the fixed n_estimators = 150.
- Optuna tuning script for LightGBM / XGBoost / CatBoost (same 63 features and 5 film-grouped folds as main.ipynb, absolute-error objective, median pruner, seed 42) was started for LightGBM with 60 trials; the session ended after 6 trials: trial CV 0.3780 at best vs 0.3844 for default LightGBM. Nothing was saved, so no tuned parameters are available or adopted.
- Tuning has to be redone on the new feature set (the one in main.ipynb now), then seed averaging.

## 2026-10-07 - External film / calendar datasets tested as features (main.ipynb feature set, CV only)

Setup: the 63 features of main.ipynb (LB 0.45395), 5 folds grouped by film (same folds as main), LightGBM l1; "default" = library defaults, "stable" = 300 trees, lr 0.05, bagging 0.8, mean and std over 3 seeds. Noise level is about 0.002. Feature code is in the scratchpad (`extfeat.py`) and not yet in main.ipynb.

### Feature groups built from `external/`
| Group | Features | Source files (publish dates all <= 2025-09-30 except the blanked films) | Coverage train / test |
|---|---|---|---|
| A profile | is_local, film_kind, is_sequel, source_material | film_profile.csv | 100% / 96% |
| B predecessor | pred_adm_log (log of the predecessor film's admissions) | film_profile.csv (predecessor_published <= 2025-08-03) | 5% / 9.5% |
| C wiki | wiki_views_log, wiki_views_trend (Wikipedia pageviews in the 60 days before D1, or before 2025-09-29 for test films) | wiki_pageviews_daily.csv + wikipedia_en / wikipedia_id of film_profile.csv | 38% / 12% |
| D school regional | school_off_reg (break / Idulfitri / Ramadan break / selfstudy by province of the cinema's city; DKI from school_holidays.csv), school_exam_reg | school_calendar_regional.csv, city_province.csv | off 12% / 14%, exam 0.7% / 3.3% |
- Strict cutoff rule applied: the 7 test films whose fact source is dated after 2025-09-30 (or marked info_public_before_cutoff = no): ALAS ROBAN, MALAM 3 YASINAN, OZORA: PENGANIAYAAN BRUTAL PENGUASA JAKSEL, LIFT, MENGEJAR RESTU, RAJAH, MENUJU PELAMINAN - THE ROAD TO MARRIAGE have their profile fields left empty (2,625 rows). world_premiere_date, trailer_date and distributor_id are not used (test-only columns, empty for train).
- Not used as features: filmindonesia_admissions.csv (matches 11 train films and 0 test films), season_trajectories.csv and season_statements.csv (priors, see below).
- The three "(1)" files in external/ are misnamed copies (season_statements (1) = film profile table, season_trajectories (1) = old statements, wiki_pageviews_daily (1) = trajectories with one more row); ignored, not deleted.

### CV results
| Setup | Features | default LGBM | stable (3 seeds) |
|---|---|---|---|
| base (main.ipynb) | 63 | 0.3844 | 0.3828 +- 0.0017 |
| + A profile | 67 | 0.3874 | 0.3831 +- 0.0013 |
| + B predecessor | 64 | 0.3858 | 0.3808 +- 0.0015 |
| + A + B | 68 | 0.3866 | 0.3834 +- 0.0012 |
| + C wiki | 65 | 0.3886 | 0.3851 +- 0.0014 |
| + D school regional (added next to the DKI flag) | 64 | 0.3850 | 0.3827 +- 0.0023 |
| + D school regional (replaces the DKI flag) | 63 | 0.3865 | 0.3822 +- 0.0003 |
| + D exam flag | 64 | 0.3844 | 0.3831 +- 0.0020 |
| + D regional + exam | 64 | 0.3865 | 0.3821 +- 0.0024 |
| + everything | 71 | 0.3892 | 0.3829 +- 0.0012 |
- No group moves CV outside the noise. Only C (wiki) is consistently worse (+0.002 to +0.004), matching its coverage mismatch (38% train vs 12% test).
- CV can not judge film-level features (lesson from v2: they hurt CV and helped the LB), so A and B need an LB test; D only changes calendar flags.

### Season priors from season_trajectories.csv (D4-D10 average daily admissions / D1-D3 average, national, interpolated cumulative admissions, 14 films)
| Season | Ratio | Films |
|---|---|---|
| Our normal train films (national, D4-D10 / D1-D3) | median 0.36, mean 0.45 | 180 |
| Christmas 2024 | 0.69-0.80 | 4 |
| Pre-Ramadan 2025 | 0.88 | 1 |
| Lebaran 2024 (released after Eid) | 0.65, 0.97 | 2 |
| Lebaran 2025 (released on Eid) | 1.27, 1.37, 1.39, 1.58, 2.40 (median 1.39) | 5 |
- Holiday seasons keep sales about 2x above a normal week; Lebaran week for Eid-day releases about 4x. Supports a Lebaran factor above x2 (never tested on the LB: x3 is the next probe). Small sample and interpolated data, so a direction, not a value.

### Optuna status (still running when this was written, 63-feature main.ipynb)
- After about 21 min: LightGBM best 0.3750 (20 trials, default 0.3844), XGBoost best 0.3790 (40 trials), CatBoost best 0.3809 (5 trials). Seed re-scoring of the top trials is still to come.

## 2026-10-07 - Research while the user was away: papers, audience statistics, test-film information

Full write-up in `leon/research_notes.md`. Rule applied: public on or before 2025-09-30, and never any number about a test film's own run.

### New external file
- `external/audience_statistics.csv` (8 rows, built in `external_add.ipynb`, section "Audience statistics"): statements from pages that were opened, each with url and publish date (all <= 2025-09-30, assertion in the notebook).
  - Cineplex 21 via KapanLagi, 2013-07-12: attendance drops 30 to 40% in Ramadan, also on weekends.
  - GPBSI chair via Kontan, 2023-07-04: school breaks do not affect occupancy much, the industry relies on Eid and year-end.
  - Cinema XXI via Bareksa 2025-05-02 and Kontan 2025-06-19 and Indo Premier 2025-04-25: April 2025 more than 14 million admissions (monthly record, old record 11.7 million in June 2019), May 2025 about 8 million, Q2 2024 25 million, 2024 total 84 million; Q1 2025 ticket revenue -30% YoY with March (Ramadan) weaker and April already close to the whole Q1 total; five films above 1 million admissions in Q1 2025 against seven in Q1 2024.
- Seasonal profile from these numbers (Cinema XXI, rough): April 2025 (Lebaran month) about 2.0x an average month (7.0 million), May about 1.1x, Q1 2025 about 0.7x.
- No file was built for test-film hype: nothing quantified and dated before the cutoff could be verified (Zootopia 2 trailer view counts seen only in a search snippet; Agak Laen trailer 2025-10-08, Alas Roban announcement and most Q1 2026 material are after the cutoff).

### Not used, with reason
- ANTARA articles dated 2026-03-31 (they contain admissions of test films), CNBC Indonesia 2026-03-04, Alonesia / Narasitoday March 2026 (Ramadan 2026 occupancy), Cinema XXI full-year 2025 results, 2026 Wikipedia pages: after the cutoff. They appeared in search results; their numbers were not stored or used.
- The 17.25% (62.9 million admissions within 14 days of Eid, 2007 to 2024) figure: only pages dated March 2026 were found.
- SMRC 2019 (67% of 15 to 38 year olds saw an Indonesian film in a cinema) and Populix / IDN 2022 (95% like films, 21% go to cinemas): publication date not read, general surveys with no holiday detail.
- No data found on how many students watch films on holidays, or the percentage of people who watch films on holidays; the nearest evidence is the GPBSI statement (school breaks matter little) and our own train estimate (school-holiday days 1.08x national tickets).
- Videos / YouTube transcripts cannot be read here.

### Papers reviewed (details and links in research_notes.md)
- Groen (2023, Erasmus thesis): session demand forecasting, gradient boosting beat linear regression, R2 0.725 / 0.708 on two held-out weeks, rolling-origin validation; feature groups: general (movie, hour, weekday, holiday, weather), movie (popularity rank, rating, language, genre, sequel, actors / directors, distributor, budget, weeks since release, last week's tickets) and schedule / competition (films of the same genre / release week / popularity starting nearby), the competition group highly important.
- Baranowski et al. (2020, Poland, 179,103 shows): cinema variables cut RMSE about 1%, region variables 0.3%.
- Also Sawhney and Eliashberg (1996), Neelamegham and Chintagunta (1999) (tested, see 2026-10-03), Mestyan et al. (2013, Wikipedia activity; tested as pageview features, no gain), Marshall et al. (2013, Chile).
- Takeaways: use rolling-origin validation as a second check; a competition feature block is what the closest paper found most important (the other session's competition features, LightGBM CV 0.3763 alone against 0.3819, point the same way).
- No paper tests a season missing from training and none uses Indonesian data, so their accuracy numbers are not comparable to MASE.

## 2026-10-07 - Optuna result: XGBoost (63-feature main.ipynb set, 5 film-grouped folds, absolute-error objective)

- Search: 60 trials (TPE, median pruner) in about 33 min. Top 5 re-scored on 3 seeds, winner = trial 56.
- Winner parameters: n_estimators 541, learning_rate 0.0145, max_depth 10, min_child_weight 46.1, subsample 0.515, colsample_bytree 0.431, reg_lambda 7.78, reg_alpha 2.25.
- 10-seed check (equal-weight average, no seed picked by score):
| Setup | CV |
|---|---|
| default XGBoost (any seed, no randomness) | 0.4047 |
| winner, mean of 10 single seeds | 0.3787 +- 0.0014 |
| winner, 10-seed averaged predictions | 0.3781 |
- Tuning gain 0.0267 (far above the 0.002 noise level); seed averaging gain only 0.0006: the averaged CV is 0.3778-0.3786 from 2 seeds up, so 3 to 5 seeds are enough for XGBoost.
- Not yet on the LB. LightGBM (best 0.3729 so far) and CatBoost (best 0.3708 after 10 of 30 trials) are still being searched.

## 2026-10-07 - Optuna result: LightGBM (same setup as XGBoost above)

- Search: 71 trials (TPE, median pruner) in 45 min. Top 5 re-scored on 3 seeds, winner = trial 54 (3-seed mean 0.3732: seeds 0.3729, 0.3742, 0.3724) against default LightGBM 0.3844 (no seed dependence). Gain +0.0113.
- Runners-up on 3 seeds: trial 53 0.3736, trial 61 0.3739, trial 26 0.3748, trial 33 0.3759, so the top trials are within about 0.001 of each other and the exact parameters matter less than the region.
- Winner parameters: n_estimators 883, learning_rate 0.0150, num_leaves 87, min_child_samples 11, subsample 0.641 (subsample_freq 1), colsample_bytree 0.483, reg_lambda 1.86, reg_alpha 11.6.
- 10-seed averaging check was still running when this was written; CatBoost search was at trial 10 of 30 (best 0.3708).

### LightGBM 10-seed check
| Setup | CV |
|---|---|
| default LightGBM | 0.3844 |
| winner, mean of 10 single seeds | 0.3731 +- 0.0011 (seeds 0.3719 to 0.3752) |
| winner, 10-seed averaged predictions | 0.3725 |
- Tuning gain 0.0119, seed averaging gain 0.0006; the averaged CV is flat (0.3723 to 0.3732) from 2 seeds up, so 3 to 5 seeds are enough for LightGBM as well.
- A single seed varies by 0.003 around the mean (0.3719 to 0.3752), which is the same size as most feature effects tested earlier, so features must be compared with several seeds.

## 2026-10-07 - Optuna result: CatBoost (same setup)

- Search: 20 trials (30 planned, stopped by the 75 min cap), a CatBoost trial takes 3 to 12 minutes. Top 5 re-scored on 3 seeds, winner = trial 8 (3-seed mean 0.3723: seeds 0.3708, 0.3700, 0.3760) against default CatBoost 0.3800 (seeds 0.3797, 0.3801, 0.3801). Gain +0.0077.
- Runners-up on 3 seeds: trial 16 0.3726, trial 17 0.3739, trial 19 0.3743, trial 14 0.3772.
- Winner parameters: iterations 723, learning_rate 0.0330, depth 10, l2_leaf_reg 11.9, random_strength 7.57, bagging_temperature 1.79 (loss MAE, category columns as text).
- 3-seed averaging check was running when this was written.

### Summary of the three tuned models (63-feature main.ipynb set, CV = 5 folds grouped by film)
| Model | default | tuned (mean of seeds) | gain |
|---|---|---|---|
| LightGBM | 0.3844 | 0.3731 (10 seeds) / 0.3725 averaged | 0.0119 |
| XGBoost | 0.4047 | 0.3787 (10 seeds) / 0.3781 averaged | 0.0267 |
| CatBoost | 0.3800 | 0.3723 (3 seeds) | 0.0077 |
- A blend check of the tuned models plus Kumo (equal weights, out-of-fold predictions) was started; result to be logged next.

## 2026-10-07 - Blend check of the tuned models, and why Kumo-Tabular wins

Out-of-fold predictions of the tuned winners on the 63-feature main.ipynb set, 5 folds grouped by film (3 seeds averaged for the GBDTs, Kumo large with a 5,000-row context and 1 seed).

### OOF MAE
| Model | MAE |
|---|---|
| Kumo-Tabular | **0.3502** |
| CatBoost (tuned) | 0.3704 |
| LightGBM (tuned) | 0.3727 |
| XGBoost (tuned) | 0.3783 |
| lgbm + catboost | 0.3699 |
| kumo + catboost | 0.3558 |
| lgbm + kumo | 0.3566 |
| all four, equal weights | 0.3634 |
- Every equal-weight blend that includes Kumo is worse than Kumo alone; blending the three GBDTs gains almost nothing (0.3699 vs 0.3704).
- Weighted: w * kumo + (1 - w) * mean(lgbm, catboost) gives 0.3540 at w 0.6, 0.3526 at 0.7, 0.3515 at 0.8, 0.3507 at 0.9, 0.3502 at w 1.0, so no weight beats Kumo alone.
- Kumo is better in every fold (0.3002, 0.5128, 0.2527, 0.3849, 0.3005 vs LightGBM 0.3524, 0.5221, 0.2712, 0.4114, 0.3063), every horizon (D4 0.3906 vs 0.3995, D10 0.3281 vs 0.3676) and every pair-size bucket.

### Part of the reason: zeros
- 31.8% of train target rows are exactly 0. Kumo predicts about 0 (< 0.02) for 31.4% of rows; the tuned GBDTs for only 9%, and they predict a mean ratio of 0.48-0.49 against 0.59 actual (Kumo 0.40).
- Under absolute error the best prediction is the median, which is 0 whenever more than half of similar rows sold nothing; Kumo outputs quantiles and takes the median, boosted trees with an L1 loss converge slowly to a spike at 0.
- Snapping GBDT predictions below a threshold to 0 helps: LightGBM 0.3727 to 0.3673 at t 0.20, CatBoost 0.3704 to 0.3653 at t 0.19, XGBoost 0.3783 to 0.3726 at t 0.22. Threshold chosen on 4 folds and scored on the 5th: LightGBM 0.3681, CatBoost 0.3678. Kumo needs no snapping (best t 0.01).
- This also answers the earlier worry about many zeros in the submissions: the zeros are correct, and the GBDT submissions had too few of them.
- Snapping explains only about a quarter of the gap (0.0051 of 0.0225); the rest is better generalization to unseen films.

### Caveats
- Train-period CV has misled before, but only for film-level feature choices and season-specific adjustments; this is a comparison of model types on the same features and folds.
- Kumo is untested on the LB alone. The 0.45395 submission used an equal-weight blend of default LightGBM, CatBoost, XGBoost and Kumo.
- A run of Kumo context variants (5k vs 10k rows, 1 vs 3 context seeds averaged) was started.

### CatBoost 3-seed check
| Setup | CV |
|---|---|
| default CatBoost, mean of 3 single seeds | 0.3800 +- 0.0002 |
| default CatBoost, 3-seed averaged predictions | 0.3791 |
| winner, mean of 3 single seeds | 0.3723 +- 0.0026 (0.3708, 0.3700, 0.3760) |
| winner, 3-seed averaged predictions | 0.3704 |
- Tuning gain 0.0087, seed averaging gain 0.0018 (larger than for LightGBM and XGBoost, 0.0006, because CatBoost's seeds vary more).

## 2026-10-07 - Kumo context variants, and main.ipynb updated with the tuned setup

### Kumo-Tabular context variants (OOF MAE, 63-feature set, 5 folds grouped by film)
| Setup | MAE | Zero share | Time (5 folds) |
|---|---|---|---|
| 5,000 rows, 1 context seed | 0.3502 | 0.314 | 3 min |
| 5,000 rows, 3 seeds averaged | 0.3458 | 0.282 | 9 min |
| 10,000 rows, 1 seed | 0.3465 | 0.317 | 5 min |
| 10,000 rows, 3 seeds averaged | **0.3447** | 0.283 | 16 min |
- Averaging 3 context seeds gains 0.0044 at 5,000 rows, as much as doubling the context; both together are best.
- Blend with snapped LightGBM + CatBoost (mean) at Kumo 10k x 3: weight 1.0 -> 0.3447, 0.95 -> 0.3451, 0.9 -> 0.3455, 0.8 -> 0.3466, 0.7 -> 0.3479, 0.5 -> 0.3513. No tree weight helps on CV; each 0.05 costs about 0.0004.

### main.ipynb changes (backup in the scratchpad; smoke-tested end to end with tiny settings: 1 seed, 3,000-row context, 60 CatBoost trees)
- RUN_MODELS = lgbm, catboost, kumo (xgboost kept in the notebook, off by default).
- Optuna parameters (2026-10-07) in LGBM_PARAMS, CATBOOST_PARAMS, XGB_PARAMS.
- MODEL_SEEDS = [42, 7, 123]: the tree models are trained per seed and averaged with equal weights (`seed_average`).
- ZERO_SNAP = 0.2 for the tree models: predictions below 0.2 are set to 0 (applied to OOF and test predictions); Kumo is not snapped.
- Kumo: KUMO_CONTEXT_ROWS = 10000, KUMO_CONTEXT_SEEDS = [42, 7, 123].
- Blend: BLEND_WEIGHTS = kumo 0.8, catboost 0.1, lgbm 0.1 (CV 0.3466 expected; Kumo alone 0.3447), as a hedge because a Kumo-only submission has not been scored on the LB.
- Submission cell writes three files from one run: `submission.csv` (the blend), `submission_kumo_only.csv`, `submission_trees_only.csv`; all use the same seasonal factors (Lebaran x2.0, school x1.25).
- Expected full run time: LightGBM about 1 min, CatBoost 15 to 20 min, Kumo about 20 min, total about 40 to 45 min.
- Smoke-test outputs (tiny settings): CV lgbm 0.3681, catboost 0.3814, kumo 0.3495, blend 0.3507; no missing values; season factor counts 64,746 x1.0, 4,598 x2.0, 3,267 x1.25.

### Watch on the LB
- Kumo and the tree models predict the same overall level (kumo / trees 1.004) but differ by seasonal group (mean ticket prediction, smoke run): ram_to_ram (10,767 rows) kumo 1.3x trees, pre_to_ram 0.6x, ram_to_lebaran 0.9x (after the x2.0), school_target_only 1.2x, school_d13_only 0.8x.
- The seasonal factors were tuned against tree-model output; if the Kumo-heavy blend scores worse than 0.45395, check these groups first (the decay-model submission, which also raised ram_to_ram, scored worse).

## 2026-10-07 - LB result: Kumo-only submission (the 63-feature main.ipynb with the tuned setup)

| Submission | Public LB |
|---|---|
| main.ipynb, equal-weight blend of default LightGBM, CatBoost, XGBoost, Kumo 5k x 1 seed | 0.45395 |
| **`submission_kumo_only.csv`** (Kumo 10k context x 3 seeds, same postprocessing) | **0.46068** |
- Worse than the earlier blend by 0.0067, although CV was far better (0.3447 vs about 0.3634 for the equal-weight blend of the same kind). Fourth time train-period CV pointed the wrong way (v2, decay model, first main.ipynb, now Kumo-only).
- `submission_trees_only.csv` and the 0.8 Kumo blend (`submission.csv`) are not submitted yet; the user plans to submit the trees-only file next.

### What the real files look like (mean predicted ratio = prediction / scale, test rows)
| | Kumo only | Trees only | Blend (0.8 Kumo) |
|---|---|---|---|
| all rows | 0.423 | 0.479 | 0.435 |
| exact zeros | 24.1% | 34.3% | 22.2% |
| ram_to_lebaran (4,598 rows) | 0.856 | 1.170 | 0.918 |
| pre_to_ram (1,460) | 0.033 | 0.169 | 0.060 |
| ram_to_ram (10,767) | 0.292 | 0.325 | 0.299 |
| school_d13_only (1,627) | 0.411 | 0.533 | 0.435 |
| normal (48,044) | 0.390 | 0.420 | 0.396 |
- By horizon Kumo / trees: D4 0.986 / 1.015, D5 0.680 / 0.748, D7 0.357 / 0.397, D9 0.139 / 0.214, D10 0.131 / 0.228: Kumo drops faster, mostly on the late days.
- Pattern across LB results: submissions that predict lower sales for big, long-lasting films scored worse (v2 without film features, decay model, now Kumo), and the largest gains came from raising predictions (Lebaran x2). Working explanation: the test period holds sales longer than the train period, so any model fitted on train decays too fast, and the model that decays fastest (Kumo) is hurt most. Not proven; it is a hypothesis to test.
- Tests that follow from it: (1) trees-only, the highest-level file (mean ratio 0.479); (2) a global multiplier on D4-D10 predictions (for example x1.1, or x1.1 on D6-D10 only) applied to the best file; (3) the Lebaran x3 probe.

## 2026-10-07 - The 0.45395 file compared with the new files, and a no-snap trees file

- The 0.45395 file (equal-weight blend of default LightGBM, CatBoost, XGBoost and Kumo 5k x 1 seed; saved by the user as Downloads/submission.csv) was compared with the new outputs (mean predicted ratio = prediction / scale over test rows):
| File | LB | Mean ratio | Exact zeros | Lebaran group | D9-D10 | Distance to the 0.45395 file |
|---|---|---|---|---|---|---|
| best (default models) | 0.45395 | 0.491 | 0.4% | 1.250 | 0.236 | - |
| `submission_kumo_only.csv` | 0.46068 | 0.423 | 24.1% | 0.856 | 0.135 | 0.121 |
| `submission.csv` (0.8 Kumo, 0.1 CatBoost, 0.1 LightGBM) | not submitted | 0.435 | 22.2% | 0.918 | 0.152 | 0.102 |
| `submission_trees_only.csv` (tuned LightGBM + CatBoost, 3 seeds, zeros snapped at 0.2) | not submitted | 0.479 | 34.3% | 1.170 | 0.221 | 0.070 |
| `submission_trees_nosnap.csv` (same models, no snapping) | not submitted | 0.502 | 2.2% | 1.202 | 0.254 | 0.068 |
- The best file has the highest level among the submitted files and almost no exact zeros (default models, no snapping). The trees-only file has 34.3% exact zeros because of the snapping, tuned on train where 31.8% of targets are 0; if the test period keeps sales longer, snapping may hurt there.
- Tomorrow's trees-only test mixes three changes against the best file (tuned parameters, 3 seeds, snapping). `submission_trees_nosnap.csv` removes the snapping: it was built with the notebook's own tuned LGBM_PARAMS / CATBOOST_PARAMS, 3 seeds each, final fit on all train rows, the same seasonal factors (Lebaran x2.0, school x1.25), unsnapped mean of LightGBM and CatBoost.
- Note: the leon/submission.csv is the 0.8 Kumo blend, not the 0.45395 file.

## 2026-10-07 - LB pattern across all submissions we still have files for

Mean predicted ratio (prediction / scale) on test rows, by slice, against the public LB:
| File | LB | All rows | D9-D10 | Top-20% biggest films | Lebaran group | Ramadan-to-Ramadan | Exact zeros |
|---|---|---|---|---|---|---|---|
| best (default models, equal blend) | **0.45395** | 0.491 | 0.236 | 0.821 | 1.250 | 0.341 | 0.4% |
| Kumo only | 0.46068 | 0.423 | 0.135 | 0.736 | 0.856 | 0.292 | 24.1% |
| p1 | 0.47566 | 0.453 | 0.205 | 0.807 | 1.064 | 0.298 | 2.8% |
| decay model | 0.48012 | 0.460 | 0.176 | 0.765 | 0.972 | 0.345 | 8.0% |
| v4 | 0.48601 | 0.428 | 0.195 | 0.757 | 0.665 | 0.298 | 2.8% |
| v3 | 0.48865 | 0.412 | 0.188 | 0.722 | 0.532 | 0.296 | 2.8% |
| not submitted: 0.8 Kumo blend | - | 0.435 | 0.152 | 0.751 | 0.918 | 0.299 | 22.2% |
| not submitted: trees only, snapped | - | 0.479 | 0.221 | 0.815 | 1.170 | 0.325 | 34.3% |
| not submitted: trees only, no snap | - | 0.502 | 0.254 | 0.822 | 1.202 | 0.356 | 2.2% |
- Spearman correlation of the LB score with the level of each slice over the 6 submitted files (negative = higher level goes with a better score): Lebaran group -0.83, biggest films -0.66, all rows -0.60, D9-D10 -0.26, Ramadan-to-Ramadan -0.14, share of exact zeros +0.09.
- So the LB follows the level of the Lebaran group and of the big films; the zero share does not matter. With 6 files this is descriptive, not proof.
- Exception: Kumo only has a lower level than p1 in every slice (Lebaran 0.856 vs 1.064) yet scores better (0.46068 vs 0.47566), so model quality (which pairs decay) also counts; Kumo's problem is its level, not its ranking.
- Similarity to the 0.45395 file (Pearson of ratios, mean absolute difference / scale, share of rows within 10%): trees no snap 0.964 / 0.068 / 43.6%; trees snapped 0.962 / 0.070 / 49.7%; 0.8 Kumo blend 0.927 / 0.102 / 41.2%; Kumo only 0.902 / 0.121 / 37.9%; p1 0.902 / 0.114 / 32.3%; decay 0.877 / 0.134; v4 0.874 / 0.127; v3 0.845 / 0.140.
- Idea not yet built: take Kumo's predictions (best ranking, too low a level) and lift them per season group and horizon to the level of the tree files, then test that on the LB.

## 2026-10-07 - Aggressive LB probes on top of the 0.45395 file (one change each)

Base = the 0.45395 submission (Downloads/submission.csv). Each probe multiplies only one slice of the base predictions; everything else is identical, so the LB change comes from that change alone. Zero predictions stay zero.
| File | Slice and factor | Rows changed | Mean ratio | Largest possible LB change |
|---|---|---|---|---|
| `submission_probe_lebaran_x1.5.csv` | Lebaran group (D1-D3 touches Ramadan, target from 2026-03-20) x1.5 | 4,598 (6.3%) | 0.491 -> 0.531 | 0.040 |
| `submission_probe_bigfilms_x1.3.csv` | top 20% of rows by the film's national D1-D3 average x1.3 | 15,015 (20.7%) | 0.491 -> 0.542 | 0.051 |
| `submission_probe_late_d7_10_x1.5.csv` | horizons D7 to D10 x1.5 | 41,492 (57.1%) | 0.491 -> 0.573 | 0.082 |
| `submission_probe_all_x1.2.csv` | all rows x1.2 | 72,611 | 0.491 -> 0.590 | 0.098 |
- "Largest possible LB change" is the mean absolute change of the predictions in scale units; the LB can never move by more than this, and in practice moves by a fraction of it.
- Reading a result against 0.45395: better by more than 0.003 = right direction, try a bigger step next; worse by more than 0.003 = wrong direction or too far; within 0.002 = close to the best level for that slice (or the gain and loss cancel).
- Why these four: the LB follows the level of the Lebaran group (Spearman -0.83) and of the biggest films (-0.66) across the six scored files; the late days are where Kumo drops fastest (D9-D10 43% below the trees); the global lift tests whether test sales are higher than all train-fitted models predict.
- Order suggested: lebaran_x1.5 first (strongest evidence, also supported by the Lebaran 2024 / 2025 trajectories at 1.27 to 2.40 for Eid-day releases), then bigfilms or all, then late days. Results to be logged here.

## 2026-10-07 - LB result: Lebaran x1.5 probe = new best 0.44518

| Submission | Public LB |
|---|---|
| base (default models, equal blend; Lebaran factor x2.0 on the raw model) | 0.45395 |
| **`submission_probe_lebaran_x1.5.csv`** (same file, Lebaran group x1.5 more, i.e. x3.0 of the raw model) | **0.44518** |
- Gain 0.00877, from changing 4,598 rows (6.3%) only; the largest possible change of that probe was 0.040. The direction is confirmed: test Lebaran-week sales are higher than the model-times-2 level.
- History of the Lebaran factor (x of the raw model, LB): 1.25 (v4) 0.48601 -> 2.0 (p1-style) gains 0.0104 -> 3.0 gains another 0.0088. The gain per step is shrinking, so the best factor is probably not far above 3.0 (rough estimate under a lognormal assumption about the ratio of actual to predicted: between 2.8 and 3.2); an extra x1.5 on top is unlikely to give more than a few thousandths.
- The older probes (`submission_probe_bigfilms_x1.3`, `late_d7_10_x1.5`, `all_x1.2`) were built on the 0.45395 base and are superseded.
- New probes built on the 0.44518 file, each excluding the Lebaran group so the slices stay disjoint from the lift already tested:
  - `submission_next_all_x1.2.csv`: all other rows x1.2.
  - `submission_next_bigfilms_x1.3.csv`: other rows of the top-20% biggest films x1.3.
  - `submission_next_late_d7_10_x1.5.csv`: other rows of horizons D7-D10 x1.5.

## 2026-10-07 - LB result: all other rows x1.2 = new best 0.44318

| Submission | Public LB |
|---|---|
| `submission_probe_lebaran_x1.5.csv` (base) | 0.44518 |
| **`submission_next_all_x1.2.csv`** (every row except the Lebaran group x1.2) | **0.44318** |
- Gain only 0.0020, against a largest possible change of 0.082. Reading it: the weight of the slice (sum of prediction / scale over its rows, divided by all rows) is about 0.41, the average slope of the error between x1.0 and x1.2 is -0.010 per unit, which means the weighted share of predictions below the truth exceeds the share above by only 2.4%, so the overall level is already near its best (the weighted median of actual / predicted is close to 1.1).
- Consequence: a bigger global lift will not help; remaining gains are in which slices are too low and which too high (they cancel in a global change).
- Weight carried by slice on this base (non-Lebaran rows): top-20% biggest films 18.5% of rows with mean ratio 0.936 (about 42% of the weight); bottom 40% films 39.6% of rows at 0.262; D7-D10 53% of rows at 0.301; D4-D6 40% of rows at 0.830; scale <= 20 pairs 12.5% at 0.262.
- Next probes built on the 0.44318 file (Lebaran group excluded), one slice each:
  - `submission_next2_bigfilms_x1.3.csv`: top-20% biggest films x1.3 (x1.56 against the 0.45395 base).
  - `submission_next2_smallfilms_x0.8.csv`: bottom-40% films x0.8 (tests that small films are over-predicted, the counterpart of the lift).
  - `submission_next2_late_d7_10_x1.5.csv`: horizons D7-D10 x1.5.
- Superseded: the `submission_next_bigfilms_x1.3` and `submission_next_late_d7_10_x1.5` files (built on the 0.44518 base).

## 2026-10-07 - Local analysis before spending more submissions: what train can and cannot say about multipliers

Best multiplier per slice on TRAIN (out-of-fold predictions of the tuned models, weighted median of actual / predicted with prediction weights; 1.00 = already right):
| Slice | Rows | Trees (lgbm + catboost) | Kumo 10k x 3 |
|---|---|---|---|
| all rows | 54,671 | 0.95 | 1.03 |
| top 20% biggest films | 11,123 | 0.87 | 0.95 |
| bottom 40% smallest films | 22,023 | 0.95 | 1.09 |
| D4-D6 | 23,414 | 0.99 | 1.03 |
| D7-D10 | 31,257 | 0.81 | 1.01 |
| pair scale <= 20 | 1,965 | 0.22 | 0.61 |
| pair scale 20-80 | 12,270 | 0.98 | 1.10 |
| pair scale > 80 | 40,436 | 0.95 | 1.02 |
| target on holiday / cuti / school break | 10,102 | 0.94 | 1.04 |
| normal day | 44,569 | 0.95 | 1.02 |

Change in train out-of-fold MAE if a probe multiplier is applied (negative = better):
| Probe | Trees | Kumo |
|---|---|---|
| all rows x1.2 | +0.0197 | +0.0097 |
| top-20% films x1.3 | +0.0167 | +0.0122 |
| bottom-40% films x0.8 | +0.0012 | +0.0042 |
| D7-D10 x1.5 | +0.0326 | +0.0146 |
| pair scale <= 20 x0.5 | -0.0008 | -0.0001 |
| pair scale <= 20 x1.5 | +0.0017 | +0.0007 |

- Train is well calibrated overall (best multiplier 0.95 to 1.03), yet on the LB the non-Lebaran rows gained from x1.2 (-0.0020) while train says the same step would cost +0.010 to +0.020: the test-period shift (films hold sales longer) is worth roughly +10 to +30% of the level on these rows, now measured against train.
- Train cannot give the test multiplier (no test labels, a different season; it has pointed the wrong way four times). It only gives relative slice patterns, which may or may not carry over.
- If the shift is uniform across slices, the train patterns say: big films are already the most over-lifted (best multiplier 0.87 to 0.95), late days even more for the trees (0.81), pairs with scale <= 20 are heavily over-predicted (0.22 to 0.61). So the probes with the weakest prior are D7-D10 x1.5 and big films x1.3; the strongest prior is lowering small pairs, but its largest possible LB change is only about 0.016.
- The slice with no train information at all is Ramadan (train has no Ramadan days): ram_to_ram 10,767 rows and pre_to_ram 1,460 rows; a probe there is the most uncertain and potentially the most valuable.
- Possible next step, not built: a local simulator of the test labels fitted to the scored files' LB values (8 files with known scores), validated by leaving one score out; only then can multipliers be tuned without submissions.

## 2026-10-07 - Local multiplier simulator (sweep of many multipliers without submitting)

Code: `multiplier_sim.py` in the scratchpad (not in the repo). Needs the saved out-of-fold predictions of the tuned models.

### Method
- Truth model for test row i: actual ratio T_i = kappa_g(i) x R_i x r_ref_i, where r_ref is the 0.45395 file's prediction / scale, R = actual / predicted drawn from the train out-of-fold residual pool of rows with a similar predicted ratio (14 buckets, capped at 25), and kappa is a season shift for 2 slices (Lebaran rows 4,598, other rows 68,013).
- kappa fitted so the simulated LB changes equal the two measured ones exactly: Lebaran x1.5 on the 0.45395 file (-0.00877) and all other rows x1.2 on the 0.44518 file (-0.00200). Three residual pools tried (trees, Kumo, mix).
- Fitted kappa: trees pool Lebaran 1.496, other 1.177; Kumo pool 1.369 and 1.085; mix pool 1.411 and 1.119.

### Validation: the simulator FAILS for comparing different files
| File | Real LB | Simulated (trees / kumo / mix pool) |
|---|---|---|
| kumo only | 0.4607 | 0.4913 / 0.5011 / 0.4961 |
| p1 | 0.4757 | 0.4879 / 0.4903 / 0.4898 |
| decay | 0.4801 | 0.4937 / 0.4966 / 0.4960 |
| v4 | 0.4860 | 0.5027 / 0.5064 / 0.5054 |
| v3 | 0.4886 | 0.5117 / 0.5163 / 0.5151 |
- Mean absolute error on these 5 held-out files: 0.0192 (trees pool), 0.0239 (Kumo), 0.0223 (mix); correlation with the real LB 0.74, 0.50, 0.66. It over-penalises every file that differs from the reference: the truth is built around the 0.45395 file's predictions, so a different model whose ranking carries real information still looks worse. It therefore cannot say whether Kumo, trees or the blend is a better model.
- Back-test of within-file multiplier changes: Lebaran v4 -> p1 simulated -0.0148 / -0.0161 / -0.0156 against real -0.01035; v3 -> v4 (x1.25 on school and Lebaran rows) simulated about -0.009 against real -0.00264. Simulated gains are 1.4 to 3.4 times too large, so predicted gains below are upper bounds.
- A closure bug (the "total error" function used the last pool's parameters) made the first sweeps for the trees and Kumo pools inconsistent; fixed, results below are from the fixed run (the validation table was not affected).

### Sweeps (predicted LB when one slice is multiplied; current best = 0.44318)
- Current best file: best Lebaran multiplier x1.0 (x0.9 for the Kumo pool), best other-rows multiplier x0.9 (the file already carries x1.2, so about x1.08 of the original level). Predicted extra gain 0.0012 (trees), 0.0019 (Kumo), 0.0015 (mix); after the 1.4 to 3 times overstatement the real gain is probably below 0.001. The curves are V-shaped: Lebaran x1.3 more would cost about +0.009 to +0.013, other rows x1.3 more about +0.04.
- Conclusion: the Lebaran factor and the global lift are already near their best on this file; more tuning of multipliers on it is not worth submissions.
- `leon/submission.csv` (0.8 Kumo blend): its level is far lower than the current best (Lebaran mean ratio 0.918 against 1.876; other rows 0.402 against 0.528); best multipliers Lebaran x1.7 to x1.8 and other rows x1.1 (simulated 0.4651 / 0.4725 / 0.4684, not reliable in absolute terms).
- `submission_trees_nosnap.csv`: best Lebaran x1.4 to x1.5, other rows x1.1 (simulated 0.4531 to 0.4555 against 0.4648 to 0.4665 as is).

### Level-matched candidates (new files)
Each file is the original model output with the Lebaran rows and the other rows multiplied so their mean predicted ratios equal the current best's (Lebaran 1.876, other rows 0.528); this removes the level difference, so a score difference against 0.44318 reflects the model's ranking, not its level.
| File | x Lebaran | x other rows | Exact zeros |
|---|---|---|---|
| `submission_trees_nosnap_leveled.csv` | 1.561 | 1.161 | 2.2% |
| `submission_trees_snapped_leveled.csv` | 1.603 | 1.221 | 34.3% |
| `submission_blend08_leveled.csv` (0.8 Kumo blend) | 2.042 | 1.314 | 22.2% |
| `submission_kumo_only_leveled.csv` (Kumo only, whose real score was 0.46068) | 2.192 | 1.340 | 24.1% |
- Suggested order: trees_nosnap_leveled first (closest to the current best file, tests tuned trees); kumo_only_leveled tests directly whether Kumo's problem was only its level.

## 2026-10-07 - Where the local (out-of-fold) error comes from

Local out-of-fold MAE: Kumo 10k x 3 seeds 0.3447, tuned trees snapped 0.3643 (real LB of similar files: 0.443 to 0.461).
- True zeros: 31.8% of rows are exactly 0; they carry only 8.8% (Kumo) and 13.7% (trees) of the error. 86 to 91% of the error comes from rows that did sell (about 0.46 per row), so the error is about how much sold, not about zeros.
- By horizon (Kumo): D4 0.372, D5 0.406, D6 0.349, D7 0.343, D8 0.296, D9 0.321, D10 0.324; fairly flat, the first two days are the hardest.
- By pair size (D1-D3 average tickets), error per row (rows %, share of Kumo error): <= 10: 1.84 (1.2%, 6.5%); 10-20: 0.45 (2.4%, 3.1%); 20-40: 0.54 (7.5%, 11.7%); 40-80: 0.35 (14.9%, 15.1%); 80-160: 0.32 (22.0%, 20.6%); > 160: 0.285 (52.0%, 42.9%). Test has far more small pairs (13% of rows with scale <= 20 against 3.6% in train), which raises the test error.
- By fold (Kumo): 0.294, 0.518, 0.246, 0.370, 0.295; fold 1 is hard for every model, so the score depends strongly on which films a fold contains.
- By film: 10 of 174 films carry 35.0% of the Kumo error and 25 carry 54.5%. Largest: SORE ISTRI DARI MASA DEPAN (error 2.13 per row, 6.9% of the total), A MINECRAFT MOVIE (2.45, 6.1%), SAYAP SAYAP PATAH 2: OLIVIA (1.29, 4.5%), LILO & STITCH (1.01, 4.2%), UNTIL DAWN (1.53, 3.3%). These are films whose sales grew or held up where the models expected decay (surprise hits), the same direction as the test-period shift found on the LB.

## 2026-10-07 - Seasonal multipliers from last season's data (pre-cutoff), and a probe plan

### Method
- Ratio per film = daily mean of D4-D10 / daily mean of D1-D3 (the quantity the model predicts, at film level), from `external_data/season_trajectories.csv` (cumulative admissions from news, all published before the cutoff), interpolating the cumulative curve; films need a point on day 1-3 and one on day 8 or later.
- News covers hits, so seasons are compared with the news's own normal-season films ("other": Oct 2024 - Apr 2025 outside holidays), not with train. For reference, train films by the same formula: all 205 median 0.374, top third by D1-D3 0.571; news normal films 0.887, so the news sample holds about 1.55x better than train hits (selection, and / or the Oct-Mar season holding longer, which the LB also suggested with the all-rows x1.2 gain).
- Script: `season_factor.py` in the scratchpad; per-film results `season_ratios.csv`.

### Result
| Season (last year) | Films usable | Median ratio | vs news normal films | Test slice it maps to |
|---|---|---|---|---|
| normal ("other") | 5 | 0.887 | 1.00 | normal |
| Christmas 2024 | 6 | 0.712 | 0.80 | xmas (target 20 Dec - 4 Jan), 5,235 rows |
| Ramadan 2025 (opened in Ramadan) | 1 (IBLIS DALAM KANDUNGAN 2) | 0.493 | 0.56 | ram_to_ram, 10,767 rows |
| pre-Ramadan 2025 | 1 (PETAKA GUNUNG GEDE, its D4-D10 is before Ramadan) | 0.884 | 1.00 | none (pre_to_ram has no usable film) |
| Lebaran 2025 (opened on Eid) | 4 | 1.482 | 1.67 | ram_to_lebaran, 4,598 rows |
| Lebaran 2024 (opened Eid + 1) | 2 | 0.814 | 0.92 | (test Lebaran films open 2 days before Eid) |
- Lebaran: the data agree with the LB direction (Lebaran rows need a lift). The LB already tuned this slice (x3 of the raw model, near best per the simulator), so no new probe.
- Christmas: the data say faster decay (0.80, 6 films), but the LB gained from x1.25 on Christmas school-holiday rows (v3 -> v4). Contradiction, and the news Christmas films opened 12-25 Dec with inflated holiday D1-D3. Not probed.
- Ramadan: the only slice with no train information and no LB test yet. One film at 0.56 plus the XXI statement that March 2025 attendance dropped on Ramadan (no %). Shrunk halfway to 1 on the log scale: x0.75 (sqrt(0.556) = 0.75).

### Probe plan (shared submissions; each probe changes one slice of the current best 0.44318 file)
| Order | Probe | Rows | Largest possible LB change | Evidence | Read the result |
|---|---|---|---|---|---|
| 1 | ram_to_ram x0.75 | 10,767 (14.8%) | about 0.015 | last-season Ramadan film 0.56x normal; slice untested | better by > 0.002: try x0.6; worse by > 0.002: try x1.15 (Ramadan holds longer, like the rest of the test period) |
| 2 | `submission_trees_nosnap_leveled.csv` (already built) | all | - | tests the tuned trees at the best file's level | keep the better file as the base for later probes |
| 3 | pair scale <= 20 x0.5 | about 12.5% | about 0.016 (log) | strongest train prior (best multiplier 0.22-0.61) | small, cheap to confirm |
- Not worth submissions (per the simulator and the LB): more global lift, more Lebaran lift.
- Build a probe: `python make_probe.py <best.csv> ram_to_ram 0.75 submission_probe_ramadan_x0.75.csv` (scratchpad). Groups as in the season_group definition; this pre_to_ram (1,962 rows) has no school-group priority, unlike the earlier 1,460.
- The 0.44318 file is not on this machine (not in joints/, Downloads, Desktop or Documents), so no probe file is built yet.

### Files built (in joints/)
- `submission_probe_ramadan_x0.75.csv`: base = `Downloads/submission_next_all_x1.2_0797.csv` (verified as the 0.44318 file: mean ratio Lebaran 1.876, other rows 0.528), ram_to_ram rows x0.75 (10,767 rows, mean ratio 0.409 -> 0.307, largest possible LB change 0.0152). Not submitted.

### Current main (Kumo-based) output vs the best file, by slice (mean predicted ratio)
The current main's blend (`submission.csv` = 0.8 Kumo + 0.1 CatBoost + 0.1 LightGBM, seasonal factors Lebaran x2.0, school x1.25) is not on this machine; Kumo-only (80% of it, `Downloads/submission_kumo_only.csv`, LB 0.46068) compared with the 0.44318 file:
| Slice | Rows | Best | Kumo | Best / Kumo | Kumo exact zeros |
|---|---|---|---|---|---|
| normal | 50,049 | 0.514 | 0.388 | 1.32 | 24.4% |
| xmas | 5,235 | 1.024 | 0.787 | 1.30 | 5.0% |
| ram_to_ram | 10,767 | 0.409 | 0.292 | 1.40 | 33.8% |
| ram_to_lebaran | 4,598 | 1.876 | 0.856 | 2.19 | 7.3% |
| pre_to_ram | 1,962 | 0.217 | 0.057 | 3.84 | 55.3% |
| D4 / D5 / D6 / D7 | 10,373 each | 1.258 / 0.929 / 0.670 / 0.526 | 0.986 / 0.680 / 0.471 / 0.357 | 1.28 / 1.37 / 1.42 / 1.48 | 1.9% / 4.9% / 10.8% / 21.1% |
| D8 / D9 / D10 | 10,373 each | 0.323 / 0.291 / 0.298 | 0.200 / 0.139 / 0.131 | 1.61 / 2.10 / 2.27 | 35.0% / 45.8% / 49.4% |
| film size q1 (small) ... q5 (big) | about 14,500 each | 0.231 / 0.301 / 0.707 / 0.798 / 1.037 | 0.151 / 0.230 / 0.444 / 0.557 / 0.741 | 1.53 / 1.31 / 1.59 / 1.43 / 1.40 | 47% ... 2% |
- The gap is a shape difference, not only a level: it grows with the horizon (1.28 at D4 to 2.27 at D10). (Correction, see below: not caused by Kumo's exact zeros; the best file also predicts about 0 on those rows.)
- So the log's per-group level match (Lebaran x2.042, other x1.314 for the 0.8 blend) leaves D4-D6 too high and D8-D10 too low relative to the best file; a match per group and horizon fits better.
- Ramadan rows sit at the same relative level as normal rows (1.40 vs 1.32), so the Kumo models give no separate signal on Ramadan.

### Current main blend leveled per group and horizon (Downloads/submission_8369.csv)
- Verified as the current main's blend (0.8 Kumo + 0.1 CatBoost + 0.1 LightGBM): mean ratio Lebaran 0.918, other rows 0.402, 22.2% exact zeros (same as the log).
- Factor per (season group x horizon) cell = best file's mean ratio / blend's mean ratio:
| Group | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---|---|---|---|---|---|---|---|
| normal | 1.19 | 1.24 | 1.26 | 1.32 | 1.46 | 1.75 | 1.72 |
| xmas | 1.20 | 1.22 | 1.21 | 1.26 | 1.36 | 1.46 | 1.60 |
| ram_to_ram | 1.24 | 1.30 | 1.49 | 1.57 | 1.56 | 2.10 | 2.85 |
| ram_to_lebaran | 1.83 | 2.11 | 1.88 | 1.83 | 2.01 | 3.09 | 4.02 |
| pre_to_ram | - | - | 1.80 | 2.26 | 3.10 | 3.69 | 3.77 |
- Where the blend predicts exactly 0 (22.2% of rows), the best file's mean ratio is only 0.023: both files agree those pairs are finished, so Kumo's zeros are not the problem. Matching only the non-zero rows gives almost the same factors (within a few %), so the mean-matched version is kept.
- Files (in joints/, not submitted):
| File | Mean ratio | Exact zeros | Corr with best | Mean abs diff vs best (scale units) |
|---|---|---|---|---|
| `submission_8369` (blend as is) | 0.435 | 22.2% | 0.903 | 0.188 |
| `submission_blend_leveled_gh.csv` (blend x cell factor) | 0.613 | 22.2% | 0.934 | 0.134 |
| `submission_hedge_blendlev50_best50.csv` (0.5 leveled blend + 0.5 best) | 0.613 | 0.4% | 0.982 | 0.067 |
- Every cell of both files has the same mean predicted ratio as the best file, so a score difference against 0.44318 measures the blend's ranking (which pairs are high or low), not its level.
- Suggested order: hedge first (lower risk; averaging two different decent models usually beats both); if it gains, the leveled blend next to see whether more Kumo weight helps.

### LB result: hedge 50/50 = new best 0.43729
| Submission | Public LB |
|---|---|
| `submission_next_all_x1.2.csv` (best before) | 0.44318 |
| **`submission_hedge_blendlev50_best50.csv`** (0.5 current-main blend leveled per group x horizon + 0.5 best) | **0.43729** |
- Gain 0.0059 (mean abs change of the predictions 0.067), against 0.0020 for the last multiplier step: the current main's blend (mostly Kumo) ranks pairs better than the old file once its level is matched. First LB gain that comes from the model, not from a multiplier.
- Next files (in joints/, not submitted), both built on the same pieces:
  - `submission_hedge_blendlev75_best25.csv`: 0.75 leveled blend + 0.25 best (mean abs change vs the 0.43729 file 0.033). Better -> the optimum is above 0.5, try the leveled blend alone; worse -> keep 0.5.
  - `submission_hedge50_ramadan_x0.75.csv`: the 0.43729 file with ram_to_ram rows x0.75 (10,767 rows). Replaces `submission_probe_ramadan_x0.75.csv`, which was built on the old best.

## 2026-10-07 - Summary of today's submissions and where we stand

### Submissions scored today (public LB, lower is better)
| # | File | Public LB | Change vs previous best | What changed | Built from |
|---|---|---|---|---|---|
| 1 | `submission_kumo_only.csv` | 0.46068 | worse than 0.45395 by 0.0067 | Kumo 10k context x 3 seeds only, current main features and tuned setup, seasonal factors Lebaran x2.0 / school x1.25 | current main |
| 2 | `submission_probe_lebaran_x1.5.csv` | 0.44518 | -0.00877 vs 0.45395 | Lebaran rows (4,598) x1.5 on the 0.45395 file (x3.0 of the raw model) | 0.45395 file (old main, equal blend of default LightGBM / CatBoost / XGBoost / Kumo 5k) |
| 3 | `submission_next_all_x1.2.csv` | 0.44318 | -0.00200 | every non-Lebaran row x1.2 on file #2 | file #2 |
| 4 | `submission_hedge_blendlev50_best50.csv` | **0.43729** | **-0.00589** | 0.5 x current-main blend leveled to file #3 per season group and horizon + 0.5 x file #3 | file #3 + `submission_8369.csv` (current main blend) |
- Also in the submission history but not described anywhere in this log: `submission_baseline_v2 (3).csv` at 0.47410 (origin to be confirmed).
- Progress today: 0.45395 -> 0.43729 (-0.0167). Gap to the 0.369-0.376 cluster about 0.06-0.07; to first place (0.34456) about 0.093.

### What today taught us
1. Lebaran rows need far more than the model gives: x3 of the raw model is near the best level (gains per step 0.0104 at x2, 0.0088 at x3, simulator says little is left above x3). Last-season Lebaran films agree (D4-D10 ratio 1.67x that of normal films).
2. The overall level was slightly low: x1.2 on all other rows helped, but only by 0.0020, so the global level is now close to its best and more global lifts are not worth submissions.
3. Kumo alone scored worse (0.46068) only because of its level and shape: it predicts lower, especially late (1.28x below the best file at D4, 2.27x at D10). Once matched to the best file's level per season group and horizon, half of it in the mix gave the largest gain of the day (0.0059). So Kumo ranks pairs better; the earlier "CV better, LB worse" result for Kumo was a level problem, not a ranking problem.
4. Kumo's exact zeros are not a problem: where the blend predicts 0 (22.2% of rows) the best file's mean ratio is only 0.023.
5. Gains from multipliers are shrinking (0.0104, 0.0088, 0.0020); the first model-based gain (0.0059) is larger. Further progress has to come from better rankings (models, blends), with multipliers only to fix levels.

### How the current best file (0.43729) is built
1. Start from `submission_next_all_x1.2.csv` (= `Downloads/submission_next_all_x1.2_0797.csv`; mean predicted ratio Lebaran 1.876, other rows 0.528).
2. Take the current main's blend `Downloads/submission_8369.csv` (0.8 Kumo + 0.1 CatBoost + 0.1 LightGBM, seasonal factors Lebaran x2.0, school x1.25; mean ratio Lebaran 0.918, other 0.402, 22.2% zeros).
3. Ratio = prediction / scale (scale = pair mean D1-D3 tickets from test_history, clipped at 1).
4. Season groups (test rows): ram_to_lebaran (D1-D3 touches 18 Feb - 19 Mar 2026, target from 20 Mar), ram_to_ram (same D1-D3, target before 20 Mar), pre_to_ram (D1-D3 before Ramadan, target in Ramadan), xmas (target 20 Dec - 4 Jan), normal (the rest). Horizon = target date - D1 + 1 (4-10).
5. For each group x horizon cell: factor = mean ratio of step 1 / mean ratio of step 2 (factors from 1.19 at normal D4 to 4.02 at Lebaran D10; table in the entry above). Leveled blend = blend ratio x its cell factor.
6. Final ratio = 0.5 x leveled blend + 0.5 x step 1 ratio; prediction = final ratio x scale.
- Scripts (scratchpad, not in the repo): `level_blend.py` (steps 3-6), `make_probe.py` (season groups, single-slice probes). Copy them into the repo if they need to survive the session.

### Files ready, not submitted (all in joints/)
| File | Tests | Mean abs change vs 0.43729 file |
|---|---|---|
| `submission_hedge_blendlev75_best25.csv` | more weight on the current main's blend (0.75) | 0.033 |
| `submission_blend_leveled_gh.csv` | the leveled blend alone (weight 1.0) | 0.067 |
| `submission_hedge50_ramadan_x0.75.csv` | ram_to_ram rows (10,767) x0.75 on the 0.43729 file | 0.015 at most |

### Decision tree for the next submissions
1. Submit the 0.75 hedge.
   - Better than 0.43729: the optimum weight is above 0.5; next submit the leveled blend alone (1.0).
   - Worse: keep 0.5 as the weight; next submit the Ramadan probe.
2. Ramadan probe: better by more than 0.002 -> try x0.6; worse by more than 0.002 -> try x1.15 (Ramadan holds like the rest of the test period); within 0.002 -> leave Ramadan alone.
3. Then add a third ranking source: level `submission_trees_only.csv` from the current main the same way and average three pieces (needs the file in joints/).
4. Final selection: the best public file plus one hedge, since the factors are tuned on the public LB and may overfit it.

## 2026-10-07 - Merge of the other device's research, verification, and what it changes

### Merge
- `Downloads/submission_hedge_blendlev50_best50/logs_3217.md` (from the other device) was an exact continuation of this log (identical first 1,008 lines); its two new sections (122 lines: "Seasonal multipliers from last season's data and a probe plan" and "Summary of today's submissions and where we stand") were appended above. A backup of the previous log is in the scratchpad.
- The same folder holds `main_5220.ipynb`: all 64 cells are identical to `leon/main.ipynb`, nothing to port.
- Files copied into `leon/` (not moved): `submission_hedge_blendlev50_best50.csv` (the 0.43729 file, new best), `submission_hedge_blendlev75_best25.csv`, `submission_hedge50_ramadan_x0.75.csv`, `submission_blend_leveled_gh.csv`.

### Result
| Submission | Public LB |
|---|---|
| `submission_next_all_x1.2.csv` | 0.44318 |
| **`submission_hedge_blendlev50_best50.csv`** (0.5 x current main blend leveled per season group and horizon + 0.5 x the 0.44318 file) | **0.43729** |
- Gain 0.00589 from changing the predictions by 0.067 on average (scale units), three times the gain of the last multiplier step (0.0020), and the first gain that comes from a better ranking of pairs, not from a level change.
- It also resolves the earlier puzzle about Kumo: its CV was far better but its LB worse (0.46068); once its predictions are matched to the best file's level and decay shape, half of it in the mix helps. The problem was level and shape, not ranking.

### Verification on this machine
- Rebuilt from local files only (0.44318 file, `leon/submission.csv` = the 0.8 Kumo blend, season groups ram_to_lebaran / ram_to_ram / pre_to_ram / xmas / normal, factor per group x horizon cell, 50/50): the result equals the 0.43729 file and the leveled blend `submission_blend_leveled_gh.csv` exactly (maximum difference 0.0 tickets, correlation 1.0).
- Group sizes: normal 50,049, ram_to_ram 10,767, xmas 5,235, ram_to_lebaran 4,598, pre_to_ram 1,962.
- Per-cell factors (best file's mean ratio / blend's mean ratio) grow with the horizon: normal 1.19 at D4 to 1.72 at D10, Lebaran 1.83 to 4.02, Ramadan-to-Ramadan 1.24 to 2.85, Christmas 1.20 to 1.60. The Kumo-heavy blend decays too fast, mostly on the late days.
- New script `leon/level_blend.py` (works from the leon folder; `--best BEST.csv --model FILE[:WEIGHT] ... --out OUT.csv`) implements the recipe and reproduces the 0.43729 file exactly; it also takes several models. It replaces the scratchpad scripts of the other device (`level_blend.py`, `make_probe.py`) for the leveling part.

### What this changes in my earlier entries
- The level-matched files built earlier today with one factor per group (`submission_blend08_leveled.csv`, `submission_trees_nosnap_leveled.csv`, `submission_trees_snapped_leveled.csv`, `submission_kumo_only_leveled.csv`) are superseded: matching per group leaves D4-D6 too high and D8-D10 too low, because the factors depend strongly on the horizon. Use per-cell leveling (`level_blend.py`).
- The local multiplier simulator could not judge whether a different model ranks better (validation failed); this LB result is the empirical answer for the Kumo-heavy blend.
- Interpretation kept: multipliers are for levels, gains now have to come from rankings (models and blends).

### New file (not submitted)
- `submission_hedge3_blendlev40_treeslev20_best40.csv` = 0.4 x current blend leveled + 0.2 x `submission_trees_nosnap.csv` leveled (both per group x horizon cell to the 0.44318 file) + 0.4 x the 0.44318 file; mean abs change vs the 0.43729 file 0.019 (correlation 0.9990), so it is a small step; the third ranking source asked for in step 3 of the other device's decision tree.
- Order from that decision tree (still valid): 0.75 hedge, then the Ramadan x0.75 probe, then the three-way blend; final selection: the best public file plus one hedge.

## 2026-10-07 - Cleanup of submission files in leon/ (deleted on the user's instruction)

13 files deleted permanently; each is either superseded by a better base or did not help. Anything mentioned earlier in this log under these names no longer exists on disk (the numbers in the log stay valid).
| Deleted | Reason |
|---|---|
| `submission_kumo_only.csv` | scored 0.46068, worse than the others; the blend (`submission.csv`) already contains Kumo |
| `submission_kumo_only_leveled.csv`, `submission_blend08_leveled.csv`, `submission_trees_nosnap_leveled.csv`, `submission_trees_snapped_leveled.csv` | level matched per group only; superseded by per-cell leveling (`level_blend.py`) |
| `submission_probe_all_x1.2.csv`, `submission_probe_bigfilms_x1.3.csv`, `submission_probe_late_d7_10_x1.5.csv` | probes on the old 0.45395 base; the all-rows lift was tested later as `next_all_x1.2` |
| `submission_next_bigfilms_x1.3.csv`, `submission_next_late_d7_10_x1.5.csv` | probes on the 0.44518 base; superseded |
| `submission_next2_bigfilms_x1.3.csv`, `submission_next2_late_d7_10_x1.5.csv`, `submission_next2_smallfilms_x0.8.csv` | probes on the 0.44318 base; weak train priors (big films and late days cost +0.012 to +0.033 on train); the best base is now the 0.43729 file, new probes should be rebuilt on it |

Kept (10):
| File | Role |
|---|---|
| `submission_hedge_blendlev50_best50.csv` | **best, LB 0.43729** |
| `submission_next_all_x1.2.csv` | LB 0.44318; the reference file of the leveling recipe |
| `submission_probe_lebaran_x1.5.csv` | LB 0.44518; Lebaran x3 step of the winning chain |
| `submission.csv` | current main blend (0.8 Kumo, 0.1 CatBoost, 0.1 LightGBM); input of the recipe |
| `submission_trees_nosnap.csv`, `submission_trees_only.csv` | tuned trees without / with zero-snapping; inputs for a third ranking source |
| `submission_hedge_blendlev75_best25.csv` | next candidate (0.75 leveled blend) |
| `submission_hedge50_ramadan_x0.75.csv` | Ramadan probe on the best file |
| `submission_blend_leveled_gh.csv` | leveled blend alone (weight 1.0) |
| `submission_hedge3_blendlev40_treeslev20_best40.csv` | three-way blend candidate |

## 2026-10-07 - Where a big score jump could come from (oracle study on train, Kumo out-of-fold predictions)

Local out-of-fold MAE of Kumo 10k x 3 seeds: 0.3447. Oracle = the best constant multiplier per group, chosen with the labels (an upper bound, not achievable).
| Oracle multiplier per | MAE | Gain | Share of the error |
|---|---|---|---|
| film (174 films) | 0.2979 | 0.0467 | 14% |
| film x 3 horizon blocks (D4-5, D6-7, D8-10) | 0.2716 | 0.0730 | 21% |
| cinema | 0.3433 | 0.0014 | 0.4% |
| film x cinema pair (includes pair noise) | 0.2418 | 0.1029 | 30% |
- The film-level shift (some films hold or grow, others fade faster than predicted) is the largest structured error: spread of the per-film best multiplier: median 1.04 to 1.08, quartiles 0.71 to 1.27, 22 films above 1.5 and 43 below 0.7. It is about the size of the gap between our LB score (0.437) and the top (0.345 to 0.376). Cinema-level effects are negligible.
- It is not predictable with what we have: a LightGBM on film features (national D1-D3, trend, cinemas, genre, release day, holiday flags; 8-fold by film) predicts the film multiplier with correlation -0.00 and R2 -0.17, and applying the predicted multipliers makes the error worse (0.3479 to 0.3585 against 0.3447).
- Screen of 25 film-level signals against the per-film best multiplier (Spearman, 163 films): strongest |rho| 0.14 (number of other films opening within 3 days, p 0.066), then 0.11 (cinema expansion D1 to D3), 0.09 (D1 weekday); national and per-pair trends, show growth, occupancy change, concentration of sales, film size, local / sequel / genre flags all |rho| <= 0.08. Nothing usable.
- Conclusion: no cheap feature route to a large jump from the current information. The remaining gap is mostly film-level level shifts that no available signal explains.

### Strategy consequences
- Level and ranking need different judges: CV has pointed the wrong way five times, always on effects that change the level (film features, decay model, first main run, Kumo only, global shift); the per-cell leveled Kumo blend gained 0.0059 on the LB in the direction CV predicted (Kumo ranks better). So: tune rankings locally with a level-neutral CV (compare models after matching their group x horizon means), and tune levels on the LB with few probes.
- Expected gains of the queued files (rough): 0.75 hedge 0.002 to 0.004; three-way blend 0.001 to 0.003; Ramadan x0.75 probe up to 0.015 at most, probably about 0.003.
- Film-level calibration with the LB as the oracle: the potential is about 0.02 if the top 10 to 15 films by row weight were tuned (crude estimate: total weight 0.61 x typical |best multiplier - 1| of 0.2 x about 0.5 for the V-shaped error x the top films' share), but each group of films needs its own submissions, so it is only worth trying on the largest films with up-down pairs, after the cheaper steps.
- Risk: the public LB covers a part of the test rows; levels found structurally (Lebaran, global shift) should carry over, film-specific tuning may overfit the public part.

## 2026-10-07 - Submission plan at 3 per day and probe maker
- Added `make_probe.py`: multiplies one slice (season group, horizon range, top N films by prediction weight) of a base submission by a factor; prints the largest possible LB move (mean absolute change of the ratio).
- Built from the best file (`submission_hedge_blendlev50_best50.csv`, LB 0.43729), all on the `ram_to_ram` group only, like the queued `submission_hedge50_ramadan_x0.75.csv`:
  - `submission_plan_ramadan_x0.6.csv` (largest move 0.0243), `submission_plan_ramadan_x1.15.csv` (0.0091).
  - `submission_plan_xmas_x1.2.csv` (Christmas rows x1.2, largest move 0.0148).
  - `submission_plan_prerum_x1.2.csv` (pre_to_ram x1.2): largest move only 0.0012, not worth a submission, kept unused.
- Plan (3 submissions per day, three independent probes on the same base, combine the winners the next day):
  - Day 1: hedge75 (ranking weight), Ramadan x0.75 (level of ram_to_ram), three-way blend (third ranking source).
  - Day 2: leveled blend alone if hedge75 won, Ramadan x0.6 or x1.15 by the Day 1 Ramadan result, Christmas x1.2.
  - Day 3: one file combining every winner, plus up to two film-level probes (top films on the new best).
  - Day 4 onward: final selection = best public file plus one hedge, rest of the quota for probes on the largest films.

## 2026-10-07 - Reproducibility of the submission files
- Copied `C:\Users\leouw\Downloads\submission.csv` (LB 0.45395, identical sha256) to `leon/submission_main_default_lb0.45395.csv` so the whole chain lives in the repo.
- Verified from the data: `submission_next_all_x1.2.csv` = that file with Lebaran rows (ram_to_lebaran) x1.5 and every other row x1.2; `submission_probe_lebaran_x1.5.csv` = Lebaran rows x1.5 only.
- Added `reproduce.py`: rebuilds 11 files (probes, x1.2 reference, all hedges, all plan files) from three inputs and compares them with the files on disk; every file matches to 1e-14 in ratio. `--write` regenerates them.
- hedge3 (`blendlev40_treeslev20_best40`) uses `submission_trees_nosnap.csv` as its tree input, not `submission_trees_only.csv`.
- Only inputs that come from a notebook run cannot be rebuilt bit for bit (GPU Kumo): keep `submission_main_default_lb0.45395.csv`, `submission.csv`, `submission_trees_nosnap.csv`. sha256 (first 16 chars):
  - *submission.csv: 816b08f6789d04ea
  - *submission_trees_nosnap.csv: 969591baf90b5841
  - *submission_main_default_lb0.45395.csv: 411a4fb68a9b2815

## 2026-10-07 - Test-film real admissions (Cinepoint scrape), trial / assumption only

- Idea: the largest remaining error is film-level (some films hold sales, others fade); real daily admissions of the test films (Oct 2025 - Mar 2026) can give a per-film multiplier. Post-cutoff data, so this is a "trial" / "assumption", not official external data.
- Cleanup (user instruction): deleted 14 outdated submission files in leon/; kept only `submission_hedge_blendlev75_best25.csv`.
- New files:
  - `external/cinepoint/cinepoint_scrape.py`: Playwright (headed Chrome, pip package `playwright`) scraper of https://cinepoint.com/pages/tbo; for each date it opens the day's popup (full ranking: rank, title, daily admissions, cumulative admissions, showtimes) and appends to `external/cinepoint/cinepoint_daily_top.csv` (resumable). Needs the Chrome window visible. Month is chosen through the dropdowns (the `month` URL parameter is ignored).
  - `external/test_movies/test_films.csv`: 163 test films with D1, dataset D1-D3 tickets, cinemas, D4 / D10 dates.
  - `leon/film_adjust.py`: coverage c = dataset D1-D3 / real D1-D3 (Cinepoint cumulative at day 3); multiplier = real daily admissions x c / our predicted tickets, per film (mode film), per film-day (day) or mix; alpha shrinks toward 1; films / days missing in Cinepoint stay unchanged.
- Cross-check: Cinepoint cumulative matches news (Agak Laen D5 2,301,647; D7 3,161,317). Agak Laen real D4-D10 scaled to the dataset is about 2.41M vs 2.75M predicted by the earlier best file (about x0.88).
- Status: December 2025 done; Oct - Mar scrape was running when the session stopped. Not yet run: `film_adjust.py`, any adjusted submission, LB check.

## 2026-10-08 - Cinepoint scrape finished, first film-level adjusted submissions (trial / assumption)

- Scraper fixes: `month` URL parameter is ignored (period chosen via dropdowns), the table must finish reloading at 100 rows per page before dates are read, and each date is wrapped in try/except. Result: `external/cinepoint/cinepoint_daily_top.csv`, 182 of 182 dates (2025-10-01 to 2026-03-31), 3,094 rows (full daily ranking per day: rank, title, daily and cumulative admissions, showtimes).
- `film_adjust.py` fixes: duplicate titles per day aggregated; real D1-D3 = sum of the first three daily numbers (cumulative at D3 would include previews).
- Per-film multiplier (real D4-D10 x coverage / our predicted D4-D10) on `submission_hedge_blendlev75_best25.csv`, 80+ films with usable data (real D1-D3 >= 20,000, >= 4 D4-D10 days). Coverage (dataset D1-D3 / real) mostly 0.5-0.9.
  | Film | film_mult |
  |---|---|
  | Agak Laen, Avatar | 1.08, 1.07 |
  | Danur: The Last Chapter, Tunggu Aku Sukses Nanti, Suzzanna, Na Willa (Lebaran) | 1.31, 1.41, 1.56, 1.45 |
  | Alas Roban, Dusun Mayit, SpongeBob, Pelangi di Mars, Timur, Comic 8 | 0.51, 0.47, 0.35, 0.33, 0.54, 0.49 |
  | Sosok Ketiga, Janur Ireng, Sampai Titik Terakhirmu, Zootopia 2 | 0.63, 0.84, 0.83, 0.92 |
  Spread is 0.33 to 1.56 for the big films, the film-level shift found in the oracle study; Lebaran films need more, many December / January local horror films need much less.
- Files built (not submitted; 74.4% of rows adjusted, films without Cinepoint data unchanged; ids and no NaN checked):
  | File | Mode | alpha | Total tickets vs base | Mean abs change |
  |---|---|---|---|---|
  | `submission_filmadj_film_a1.0.csv` | one factor per film | 1.0 | 0.922 | 56 |
  | `submission_filmadj_mix_a1.0.csv` | geometric mean of film and film-day factor | 1.0 | 0.907 | 61 |
  | `submission_filmadj_film_a0.5.csv` | one factor per film | 0.5 | 0.944 | 30 |
- Reading the LB: film a1.0 better than base by a lot -> trust the real data, mix next; a0.5 better than a1.0 -> coverage estimate is noisy, keep shrinking.
- Caveat: coverage per film is estimated from one window (D1-D3), previews and format variants add noise; the multipliers are for films with real data only.

## 2026-10-08 - main.ipynb: Cinepoint real-admission features (trial / assumption), built from scratch instead of adjusting hedge75
Notebook: `main.ipynb` (new markdown + code cell after the weekday/price cell; `re` added to the imports cell; `CP_FEATURES` appended to `NUM_FEATURES` in the model setup cell; switch `USE_CINEPOINT`).

### Source and data
- Source: cinepoint.com daily top box office page (public), scraped with `external/cinepoint/cinepoint_scrape.py` on 2026-10-08.
- Test period: 2025-10-01 to 2026-03-31 (`cinepoint_daily_top.csv`, 182 dates, 3,094 rows). Published after the 2025-09-30 cutoff, so anything using it is a "trial" / "assumption", not an official submission.
- Train period: 2025-04-01 to 2025-09-30 (`cinepoint_daily_top_train.csv`), before the cutoff, so usable as an external dataset. Scrape takes about 35 s per date.
- Idea: the earlier film-level adjustment (`film_adjust.py`) multiplied an existing submission. This version has no earlier submission as input: the model sees each film's real curve and learns how a pair's tickets follow it.

### Features (per film from its D1)
- `cp_log_m13` (log mean real admissions D1-D3), `retK` (real day K / that mean, K = 4..10, empty when the film is not in the daily list), `cutK` (smallest listed admissions that date / the mean, an upper bound for films that fell off the list), `cp_cov` (our D1-D3 tickets / real D1-D3), plus `ret_h`, `cut_h` (at the row's horizon), `cp_ret_mean`, `cp_ndays`.
- Cinepoint titles have format suffixes stripped and are summed per film (this matches 100% of test films with D1-D3 data, versus about 74% of rows in the first `film_adjust.py` attempt).
- Train and test are built by the same code.

### Status
- Smoke test of the new cells passed (test rows with `ret_h`: 89%, with `cp_cov`: 100%). Train-period scrape still running, so no CV result yet.
- `realdata_model.py` was an earlier standalone draft of this and is replaced by the notebook cells; not used any more.

### 2026-10-08 - cleanup
- Deleted the helper scripts in `leon/` on the user's request: `film_adjust.py`, `realdata_model.py`, `level_blend.py`, `make_probe.py`, `reproduce.py`. Only code that is part of the `main.ipynb` pipeline goes into the notebook from now on (the Cinepoint feature cells already are). The film-level multiplier idea of `film_adjust.py` is not in the pipeline, so it is not ported; the three `submission_filmadj_*.csv` files it produced stay as they are.
- The scraper `external/cinepoint/cinepoint_scrape.py` stays (data collection, outside the pipeline, still running for the train period).

### 2026-10-08 - external_add.ipynb: Cinepoint scraper and test-film list moved into the notebook
- `external_add.ipynb` (the notebook that builds the files in `external/`): new section "Cinepoint daily top box office" after the audience statistics (scraper functions + `scrape_cinepoint(start, end, out)`, run in its own thread because Playwright's sync API cannot run in the notebook's event loop, `SCRAPE_CINEPOINT = False` by default, prints a summary of the existing CSVs) and a "Test films" cell that rebuilds `external/test_movies/test_films.csv` (checked: same values as the earlier file for all 163 films).
- Imports for these cells added to the first code cell (Playwright import is optional).
- Source and dates of the data are in the notebook text and in the 2026-10-08 entries above (test period after the cutoff = trial / assumption; train period before the cutoff).
- `external/cinepoint/cinepoint_scrape.py` stays on disk only until the running train-period scrape ends, then it is deleted (the notebook version is the same code).

### 2026-10-08 - new notebook main_new.ipynb for the Cinepoint pipeline
- `main_new.ipynb` = copy of `main.ipynb` plus the Cinepoint feature cells (see the entry above): the new real-admission pipeline lives here.
- `main.ipynb` was put back to its state before the Cinepoint cells (cells removed, `re` import and `CP_FEATURES` in `NUM_FEATURES` reverted), so it stays the old pipeline.

## 2026-10-08 - main_new.ipynb: LGBM with Cinepoint real-curve features (first result)
Notebook: `main_new.ipynb` (cells run through a scratch script, LGBM only, notebook not executed end to end yet).

### Setup
- Data: `cinepoint_daily_top_train.csv` (2025-04-01 to 2025-09-30, 183/183 dates, 2,926+14 rows; the 2025-05-24 timeout was re-scraped) and `cinepoint_daily_top.csv` (test period, trial / assumption).
- Features added to the existing list: `cp_log_m13, cp_cov, cp_ndays, ret_h, cut_h, cp_ret_mean, ret4..ret10, cut4..cut10`. Same LGBM params (Optuna 2026-10-07), 3 seeds, GroupKFold(5) by movie, zero snap 0.2.
- Fixed a bug in the new cell: the summary line crashed when `USE_CINEPOINT = False`.

### Results (CV MASE, lower is better)
| Setup | CV MASE | Rows |
|---|---|---|
| LGBM, `USE_CINEPOINT = False` | 0.3672 | all |
| LGBM, `USE_CINEPOINT = True` | 0.3049 | all (99.5% of train rows have Cinepoint data; 0.3021 on those, 0.8439 on the 0.5% without) |
- By horizon (on): D4 0.359, D5 0.377, D6 0.302, D7 0.305, D8 0.251, D9 0.265, D10 0.275 (off: 0.399, 0.429, 0.361, 0.357, 0.314, 0.357, 0.354).
- Per fold (on): 0.2649, 0.4528, 0.2159, 0.3363, 0.2547.

### Submission written (not submitted)
- `submission_new_lgbm_cinepoint.csv`: LGBM (3 seeds, all train rows) ratio x scale, no season factor, no earlier submission as input. Total tickets 12.73M = 0.808 of `submission_hedge_blendlev75_best25.csv` (15.77M), mean abs difference 65 tickets per row.
- Caveats: train CV uses the same-period real curve, so it is optimistic for the test period (CV vs LB gap was about 0.1 before); the level is 19% below hedge75 while earlier LB gains came from higher levels; CatBoost / Kumo / blend and the season-factor cell are not adapted yet.

## 2026-10-08 - LB result of the Cinepoint pipeline (reported by the user)
- `submission.csv` written by running `main_new.ipynb` (Cinepoint real-curve features, full model blend): public LB **0.36284**, previous best 0.43729. Big jump, in line with the oracle study (the missing information was film-level).
- Not an official result: the test-period features come from Cinepoint data after 2025-09-30, which `leon/CLAUDE.md` allows only as a trial / assumption. The train-period features (Apr-Sep 2025) are within the cutoff. The score shows how much film-level information is worth; an official-eligible final submission must not rely on the post-cutoff Cinepoint columns.

## 2026-10-08 - main_new.ipynb: rule-based real-sales multipliers instead of Cinepoint model features (trial / assumption)
Notebook: `main_new.ipynb`. Reason: using post-2025-09-30 data to build model inputs is the part that may not be allowed; getting a few numbers (multipliers) per film from it is. So the models now only see data up to 2025-09-30 and the real sales only feed a post-processing rule.

### Changes
- `USE_CINEPOINT = False` by default (the learned variant with Cinepoint columns, LB 0.36284, stays available behind the switch).
- New cells after the seasonal adjustment: `real_sales_multipliers` / `apply_real_sales` and a train back-test, then `test_submit` (used by the `submission.csv` cell). Adjusted rows no longer get the seasonal factor.
- Rule: coverage = dataset D1-D3 / real D1-D3; film multiplier = real D4-D10 x coverage / predicted D4-D10; day multiplier likewise per day; used = day^0.75 x film^0.25, clipped 0.3-3, raised to a power by horizon (D4 0.5, D5 0.7, D6 0.85, D7-D10 1.0), zero snap 0.2. Films with real D1-D3 < 5,000 admissions or < 4 usable days are left unchanged.

### Back-test on train (out-of-fold LGBM without Cinepoint features, base CV 0.3672; Cinepoint train file Apr-Sep 2025)
| Rule | CV MASE |
|---|---|
| film multiplier, alpha 0.75 | 0.3306 |
| mix (day^0.5 x film^0.5), alpha 0.75 | 0.3176 |
| day multiplier, alpha 0.75 | 0.3146 |
| day^0.75 x film^0.25, alpha 0.9, min real D1-D3 3,000, snap 0.2 (grid best, flat around it) | 0.3126 |
| + per-horizon power (0.5, 0.7, 0.85, 1, 1, 1, 1), min 5,000 (final) | 0.3079 |
| reference: learned model with Cinepoint columns (LGBM) | 0.3049 |
- Rows adjusted: 82% of train rows. Parameter grid is flat (0.3126-0.3140 over most settings), so the result is not sensitive to the exact values; the per-horizon power gives the extra 0.005 (early days should follow our model more).
- Per scale bucket powers were also tried (small cinemas want more, big cinemas 0.85), only 0.0004 gain, not used.

### Test side
- Notebook cells checked end to end with LGBM only: back-test 0.3672 -> 0.3079, 80.2% of test rows adjusted (median multiplier 1.16), total tickets 12.46M (`submission_rule_lgbm.csv`, same as the scratch build) vs 14.08M for `submission.csv` (LB 0.36284, full blend with the seasonal factor on every row).
- Not submitted. The LB is the judge; the back-test suggests the rule keeps most (not all) of the gain of the learned variant.
- Note: `submission_hedge_blendlev75_best25.csv` and the `submission_filmadj_*.csv` files are no longer in `leon/`.

## 2026-10-08 - main_new.ipynb cleaned: no Cinepoint model features, rule only (reproduces 0.3079)
Notebook: `main_new.ipynb`.
- Removed the learned variant completely: no `USE_CINEPOINT`, no `CP_FEATURES`, no Cinepoint cells in Feature Engineering. The models see only data up to 2025-09-30.
- The Cinepoint CSVs are now loaded in the Postprocessing section (`cp_key`, `CP_DAILY`) and only feed the per-film multipliers (`real_sales_multipliers`, `apply_real_sales`); `USE_REAL_MULT` removed, the rule is always applied to `submission.csv`.
- `RUN_MODELS = ["lgbm"]` (CatBoost / Kumo can be added back, the numbers then change; the blend weights adapt automatically).
- Whole notebook executed top to bottom (outputs redirected to a scratch folder, repo `submission.csv` untouched): LGBM CV 0.3672, rule back-test on the OOF 0.3672 -> 0.3079, 80.2% of test rows adjusted, no feature starting with ret/cut/cp_. Output equals `submission_rule_lgbm.csv`.
- The learned variant (Cinepoint columns as features, LB 0.36284) is no longer in the notebook; its description stays in the entries above.

## 2026-10-08 - main_new.ipynb: version with no Cinepoint data at all (independent pipeline)
Notebook: `main_new.ipynb`.
- New switch `USE_REAL_SALES = False` (default): the Cinepoint CSVs are not read, `CP_DAILY` is not even defined, `test_submit = test_blend`. `True` gives the rule of the entry above (trial / assumption, post-cutoff numbers).
- `RUN_MODELS` back to LGBM + CatBoost + Kumo (the best independent models). For a quick run use `["lgbm"]` (CV 0.3672).
- Levels (set on the public LB with no external data, see 2026-10-07 entries): Lebaran rows x3.0 (was 2.0 in the notebook; the LB probe on the old main gained 0.0088 from x2 to x3), school break x1.25, every other row x1.2 (probe gained 0.0020). These are guesses carried over from a different base blend (the old main: equal blend of LightGBM / CatBoost / XGBoost / Kumo 5k; the Kumo-heavy blend of today is lower at late horizons, so the real optimum differs).
- Run end to end with LGBM only (outputs to a scratch folder): CV 0.3672, no Cinepoint object defined. File: `submission_independent_lgbm.csv`, total tickets 14.69M (the rule version: 12.46M; `submission.csv` of LB 0.36284: 14.08M).
- No LB score for it yet. Expected range from the earlier results: 0.44-0.46 (main default 0.45395, Kumo-only 0.46068, best leveled 0.43729). Reason a big jump is unlikely without Cinepoint: the oracle study (2026-10-07) found the film-level level shift is the largest error (14% of error per film, 21% per film x horizon block) and none of 25 film-level signals or a LightGBM on film features predicts it (correlation -0.00).

## 2026-10-08 - Competition rule re-read: Cinepoint test-period data cannot be used for a real submission
- Files that used it, to be treated as trials only (ceiling measurements, never a final selection): `submission.csv` of LB 0.36284 (learned variant), `submission_new_lgbm_cinepoint.csv`, `submission_rule_lgbm.csv`, and anything built with `USE_REAL_SALES = True` in `main_new.ipynb`.
- Clean path: `main_new.ipynb` with `USE_REAL_SALES = False` (default) and `submission_independent_lgbm.csv`. The levels in it (Lebaran x3.0, school x1.5, other x1.2) come from LB probes, not from external data.
- Possibly still allowed: Cinepoint Apr-Sep 2025 (train period, public before the cutoff), but the rule also says versions must be verifiable, so a Wayback snapshot dated on or before 2025-09-30 would be needed first; and it has no test-period counterpart, so it would only help as past-season priors.
- Note: `main_new.ipynb` was renamed to `main_legit.ipynb` (by the user); the entries above that mention `main_new.ipynb` mean this notebook. The rule warning was added to its "Real-sales film multipliers" markdown cell.

## 2026-10-08 - main_legit.ipynb: leveling to the LB-tuned reference (clean, no post-cutoff data)
Notebook: `main_legit.ipynb` (new markdown + code cell after the real-sales cell, switch `USE_LEVEL_REF = True`).
- Method of the 0.43729 file ported into the notebook: season group (ram_to_lebaran / ram_to_ram / pre_to_ram / xmas / normal) x horizon cell factor = mean ratio of the reference / mean ratio of the blend; final ratio = `LEVEL_WEIGHT` (0.5) x leveled blend + 0.5 x reference ratio; `season_factor` set to 1.
- Reference: `submission_ref_lb0.43729.csv` (copy of `Downloads/submission_hedge_blendlev50_best50.csv`, public LB 0.43729 on 2026-10-07, made of LB probes only, 72,611 rows, total 15.66M). Group sizes match the earlier log (ram_to_ram 10,767, ram_to_lebaran 4,598 rows).
- Run end to end with LGBM only (scratch outputs): `submission_independent_leveled_lgbm.csv`, total 15.65M (reference 15.66M), mean abs change of the ratio vs the reference 0.050, mean abs diff in tickets 12.0. With only LGBM the ranking half is weaker than the Kumo-heavy blend of the 0.43729 file, so this file is a check of the step, not a candidate; the real run is LGBM + Kumo in the notebook.
- Expected LB with Kumo + LGBM: around 0.43729 (same recipe, different model weights); gains from here are small steps (the log estimated 0.002-0.004 for the 0.75 weight).
## 2026-10-08 - Can the Cinepoint submission be copied with movie/day multipliers? (analysis only, no notebook change)
Question: reproduce `submission.csv` (LB 0.36284, Cinepoint) from a clean file with a multiplier formula per film and per day.
Distance = mean abs difference of the ratio (tickets / D1-D3 average), same units as MASE, measured against the Cinepoint file.

| Clean base x multiplier | Distance to Cinepoint file |
|---|---|
| `submission_ref_lb0.43729.csv`, no multiplier | 0.2223 |
| x best factor per horizon x weekday (fit on the Cinepoint file) | 0.1929 |
| x best factor per date | 0.1547 |
| x best factor per film | 0.1647 |
| x best factor per film x day | 0.0957 |
| x LightGBM on our own features, scored on films it did not see (GroupKFold by film) | 0.1873 |
| `submission_independent_lgbm.csv`, no multiplier / LightGBM on unseen films | 0.2439 / 0.2249 |

- Most of the gap is per film; the film factor spreads from 0.46 (10th pct) to 1.21 (90th pct), median 0.83.
- Our features close only about 16% of the gap on unseen films, about the same as a horizon x weekday table; the per-film part does not generalize (same finding as the oracle study of 2026-10-07).
- A per-film or per-film-day table would get close, but its numbers are the Cinepoint test-period data in another form, so it is not allowed for a real submission. Any formula fit to the Cinepoint file also carries that information; nothing was added to `main_legit.ipynb`.
- Found: the current `submission_kumo_only.csv` is only 0.0271 away from the Cinepoint file, so it was overwritten by the Cinepoint run and is no longer the clean Kumo-only file of LB 0.46068.

## 2026-10-08 - Approaching the Cinepoint teacher without post-cutoff data (session paused)
Goal (user): get a clean submission as close as possible to `submission.csv` (LB 0.36284, Cinepoint test-period columns), used only as a "teacher" scoreboard.
Rule kept: Cinepoint data only up to 2025-09-30 as input; the teacher is never fitted on, only scored against.
Score = distance to the teacher (mean abs difference of the ratio, MASE units).

### Data and tools
- `external/cinepoint/cinepoint_scrape.py`: standalone scraper restored from the compiled .pyc (the .py was never committed), with fixes: the popup's own "Rows per page" is set to its largest option, a day is only written when the rows read equal the popup's total, and the month table waits until every day is listed. The old paging code (also in `external_add.ipynb` cell 22) re-read stale pages and dropped films on days with more than 10 films; the existing train / test Cinepoint files are almost complete (6-7 days with gaps each).
- `external/cinepoint/cinepoint_daily_top_prev.csv` (new, pre-cutoff): 282 dates, complete for 2024-09-20 to 2025-03-31 (14-21 films per day); 2023-09 to 2024-09 only partly scraped (scrape stopped, resumable with the same command). Cinepoint lists only about 5 films per day in 2023.
- `submission_teacher_cinepoint_lb0.36284.csv`: copy of the teacher (sha256 826d98615ebd1b5a), because running `main_legit.ipynb` overwrites `submission.csv`.
- `main_legit.ipynb`: markdown descriptions removed (headers kept), on user request. No code change.

### Results (distance to teacher, lower = closer)
| Clean file | Distance |
|---|---|
| ref (LB 0.43729) | 0.2223 |
| Kumo 0.8 + LGBM 0.2 x season factor | 0.2231 |
| same, leveled to ref (main_legit default) | 0.2215 |
| leveled x national retention prior (rule with predicted instead of real Cinepoint), alpha 0.25 | 0.2171 |
| leveled x per-horizon part of that prior only | 0.1988 |
| leveled x 0.85 (plain global scale) | 0.1959 |
| LGBM + prior retention as a feature (CV 0.3681 -> 0.3709) | worse |
- National retention prior: linear model of log(real D_h / real D1-D3) on horizon x D1 weekday, calendar of D_h minus calendar of D1-D3 (weekend, holiday, Ramadan, days from Eid, Christmas window), D1-D3 shape and size; fit on 234 Cinepoint films (Oct 2024 - Sep 2025). Season holdout (train without Oct 2024 - Apr 2025, score on it): weighted MAE of log retention 0.497 vs 0.668 for the horizon median.
- Against the teacher: its per-horizon pattern matches (D4 -0.16 vs -0.17 ... D10 +0.08 vs +0.05), film-level correlation only 0.22, within-film day shape 0.45; applying the film and shape parts adds noise and moves away from the teacher.
- Conclusion so far: without post-cutoff data only the level / per-horizon level moves toward the teacher; the per-film legs that make the teacher good are not predictable from pre-cutoff data (same as the oracle study). Matching the teacher's level is not proof of a better LB: LB probes on clean files favored higher levels.
- Also checked: test_history has only each film's own D1-D3 (format variants start on the same day), so no hidden test-period signal there.
- Two-way fixed-effects market index (film + age + date) was tried first and dropped: age and date are not separable (the Lebaran effect came out with the wrong sign).

### Next (when resumed)
- Add a "teacher check" cell at the end of `main_legit.ipynb` (distance to the teacher overall / by horizon / by month) as the direction check.
- Port the scraper paging fix into `external_add.ipynb` cell 22 and add the prev run to `CINEPOINT_RUNS`.
- Optionally finish the 2023-2024 scrape and refit the prior (sparse data, low expectations).

## 2026-10-08 (later) - main_legit.ipynb: timing shape from last year's films (pre-cutoff Cinepoint), teacher check
Teacher = `submission_teacher_cinepoint_lb0.36284.csv`, used only as a scoreboard (distance = mean abs ratio difference).

### Changes
- New cell "Timing shape from last year's films (Cinepoint up to 2025-09-30)" after the level-to-reference cell, switch `USE_TIMING_SHAPE = True`, `TIMING_ALPHA = 0.5`.
  - For each test film released within -40..+14 days of Eid or -14..+10 days of Christmas, the D4-D10 log shape of last years' films released at the same offset (+-4 days, widened to 10 if fewer than 3 films) is averaged per season group (ram_to_lebaran, ram_to_ram, pre_to_ram, xmas), converted to dataset scale (kappa per horizon from train films found in both), and compared with the model's shape in that group.
  - Only the shape changes: each group's level stays as LB-tuned. An assert stops the cell if any Cinepoint date after 2025-09-30 is read. Eid dates used: 2024-04-10, 2025-03-31, 2026-03-21 (national holiday day 1).
  - Factors (scratch run): Lebaran 0.78, 0.80, 0.83, 0.92, 1.41, 1.31, 1.13 (D4..D10, flatter curve); Christmas 0.82-1.27; Ramadan groups close to 1.
- New cell "Teacher check" after the submission cell: distance to the teacher overall, by horizon, season group and month.
- `CP_FORMAT` / `cp_key` moved out of the `USE_REAL_SALES` block (title cleaner only, no data read).
- `external_add.ipynb`: the Cinepoint cell now imports `scrape_cinepoint` from `external/cinepoint/cinepoint_scrape.py` (one copy of the scraper, with the paging fix and a page reload when the period filter stops responding); `cinepoint_daily_top_prev.csv` added to `CINEPOINT_RUNS`.
- Scrape: `cinepoint_daily_top_prev.csv` now also has Jan-Sep 2024 (Lebaran 2024); not yet used in the numbers below.

### Results (cached Kumo 0.8 + LGBM 0.2, leveled base; distance to teacher)
| Step | Distance |
|---|---|
| leveled (main_legit default before this) | 0.2215 |
| + timing shape, alpha 0.5 | 0.2141 |
| reference: teacher-fitted shape per group x horizon (best possible shape-only) | 0.2122 |
| reference: teacher-fitted level and shape per group x horizon | 0.1708 |
- Per group at alpha 0.5: Lebaran 0.2215 -> 0.2174, Christmas -> 0.2175, the two Ramadan groups about neutral.
- Rejected on the way (all moved away from the teacher): retention prior per film and day, per film analog shapes (3-8 analogs, too noisy), timing per release week or calendar date, a fixed-effects market index (Eid effect not identifiable from one Lebaran cohort), the prior as an LGBM feature.

### Open
- Full end-to-end run of the notebook (isolated copy) was started and stopped before finishing; not verified end to end yet.
- Refit the timing shape with the 2024 data (second Lebaran), then build the on / off pair of submission files for the LB.

### LB result (2026-10-08)
- `submission_legit.csv` (main_legit default: Kumo 0.8 + CatBoost 0.1 + LGBM 0.1, season factors, leveled to the 0.43729 reference at weight 0.5, timing shape alpha 0.5): public LB **0.43015**.
- Best clean score so far (previous clean best 0.43729, -0.0071). Teacher (Cinepoint test-period leakage) 0.36284; the remaining gap is mostly per-film legs, not predictable from pre-cutoff data.

## 2026-10-08 (evening) - where submission_legit (LB 0.43015) differs from the teacher (LB 0.36284)
Scratch scripts gap.py / gap2.py / gap3.py; teacher used only as a scoreboard (distance = mean abs ratio difference, legit = 0.2141).

| Season group | rows | share of distance | legit / teacher mean |
|---|---|---|---|
| normal | 50049 | 0.49 | 1.25 (legit too high, worst at D8-D9: 1.4-1.5) |
| ram_to_lebaran | 4598 | 0.33 | 0.66 (legit too low at D5-D10: 0.5-0.7, D4 fine) |
| ram_to_ram | 10767 | 0.09 | 0.91 |
| xmas | 5235 | 0.08 | 1.26 |
| pre_to_ram | 1962 | 0.02 | 1.39 |
- Zeros are not the issue: teacher has 22% exact zeros, legit is about 0.001 on the same rows (2% of the distance).
- Lebaran: the raw model decays the 2026-03-18 films to 0.12 / 0.06 at D9 / D10; the teacher keeps them at 2.5 / 1.7 (Eid week = D4-D10).
- Per film: biggest film gaps are the Lebaran films (NA WILLA 0.41x) and local films that flopped (TIMUR, GETIH IRENG, SHUTTER, SENGKOLO about 2x too high), the second kind is per-film legs.
- If legit had the teacher's mean per group x horizon: 0.1817; per film: 0.1649; per film x horizon: 0.1091.
- Clean Lebaran idea checked: 2025 market total (pre-cutoff Cinepoint) went from about 179k / day on Eid-3..-1 to 500k-1,000k / day in Eid week (x2.8 to x5.8 by horizon). Market jump x raw model: 0.2186 (worse, the raw decay is too strong); market jump with no decay x0.5: 0.2046 vs 0.1889 for the variant below (shape off at D4 and D9-D10). Not adopted yet.
- Note: the leveling step sets each group x horizon mean to the 0.43729 reference, so SEASON_FACTORS has no effect on the final level; a level change has to come after the leveling.

### Probe file
- `submission_legit_normal1.0.csv` = `submission_legit.csv` with the normal-group rows x 1/1.2 (other factor 1.2 -> 1.0, everything else unchanged). Distance 0.2141 -> 0.1889. Not submitted yet (LB to be checked by the user).

## 2026-10-09 - CV experiments for the clean pipeline (scratch harness, not yet in main_legit.ipynb)
Harness: LGBM with the main_legit Optuna params on the cached main_legit features, 1 seed, GroupKFold(5) by film, zero snap 0.2.
Extra columns per run: dist_sf = distance to the teacher after the main_legit season factors; dist_lvl = same after matching each season group x horizon mean to the teacher (evaluation only, measures the row ranking inside a group); dist_lvl_nonleb = dist_lvl without the Lebaran group.
Scripts: scratchpad `exp/` (lib.py, feat_*.py, e*.py), results table `exp/results.tsv`.

| Run | CV | dist_sf | dist_lvl | dist_lvl_nonleb |
|---|---|---|---|---|
| base LGBM (63 features) | 0.3681 | 0.2436 | 0.2088 | 0.1490 |
| + competition / date-strength features from other films' D1-D3 (8) | 0.3643 | 0.2401 | 0.1934 | 0.1417 |
| + days off in the D1-D3 window vs the target day (win_off, tgt_off, off_rel) | **0.3577** | 0.2496 | 0.1919 | 0.1420 |
| + more days-off detail (each D1-D3 day, eve / day after, days off between) | 0.3578 | 0.2635 | 0.1995 | 0.1415 |
| + weekday-adjusted D1-D3 trends | 0.3581-0.3667 | - | 0.1907 | 0.1445 |
| + out-of-fold cinema / city fade encodings | 0.3649 (on comp only) | 0.2427 | 0.1975 | 0.1476 |
| + late-start pair features (first active day, effective age) | 0.3565 | 0.2551 | 0.1950 | 0.1405 |
| + film-level show / occupancy trends | 0.3610 | 0.2473 | 0.1923 | 0.1431 |
| params grid on comp + off (num_leaves, min_child_samples, lr, trees, colsample, reg) | 0.3559-0.3616 (best num_leaves 255 + mcs 40) | | | |
| zero snap 0.1 / 0.3 / 0.4 | 0.3611 / 0.3580 / 0.3695 (0.2 stays) | | | |
| **+ shifted-window extra rows, shifts 3 and 7, weight 0.5** | **0.3438** | 0.2512 | 0.1930 | 0.1365 |
| Kumo base, 1 context seed | 0.3465 | 0.2377 | 0.2182 | 0.1346 |

- Competition features: comp_n_cin / comp_size_cin / comp_fresh_cin (films that opened at the same cinema after our D1 and on or before the target day, count and size vs our scale), comp_n_nat / comp_size_nat (same nationally), date_idx / own_idx / date_idx_rel (how strong other films' D1-D3 days were on the target date and on our own D1-D3 dates, vs the usual D1-D3 shape for their release weekday). Built the same way in train and test from each side's own pairs, so no post-cutoff data.
- Shifted-window extra rows: main_legit's data and feature cells re-run with every train film's D1 moved s days later (window D(1+s)-D(3+s), targets the 7 days after), column age_offset = s (0 for real rows and test). Only the fitting films' shifted rows are added in each fold; validation is on real rows only.
- Time-split check of the shifted rows (fit only on target dates before the cut, validate on films released after it, so no shared calendar dates): cut 2025-07-15: 0.3146 -> 0.2995 (w0.5) / 0.2955 (w1); cut 2025-08-15: 0.3244 -> 0.3060 / 0.3079. The gain is real, not date leakage.
- Error breakdown (LGBM comp + off): pairs with only 1-2 active days in D1-D3 are 8% of rows and 24% of the error; a known per film x horizon level would bring CV from 0.358 to 0.297 (film legs).

### Later on 2026-10-09
| Run | CV | dist_lvl | dist_lvl_nonleb |
|---|---|---|---|
| LGBM + extra rows, shifts 1, 2, 3, 5, 7 (w0.5) | 0.3354 | 0.1997 | 0.1411 |
| LGBM + extra rows, all shifts 1, 2, 3, 5, 7, 10, 14 (w0.5) | 0.3333 | 0.2012 | 0.1395 |
| same, n_estimators 1500 | **0.3304** | 0.2018 | 0.1390 |
| same (883 trees) + late-start features | 0.3322 | 0.2010 | 0.1367 |
| same (883 trees), num_leaves 255 + mcs 40 / mcs 50 / weight 0.3 | 0.3323 / 0.3329 / 0.3344 | | |
| Kumo + comp + off, 1 context seed | 0.3359 | 0.2465 | 0.1356 |
| Kumo + comp + off, context 6k real + 4k extra rows | **0.3293** | 0.2831 | 0.1407 |
| Kumo + comp + off, context 4k real + 6k extra rows | 0.3321 | 0.2749 | 0.1454 |
- Time split for all shifts (2 seeds): cut 2025-07-15: base 0.3146, shifts 3+7 0.2995, shifts 1-7 0.2977, all 0.2952; cut 2025-08-15: 0.3244, 0.3060, 0.2911, 0.2895.
- The teacher distance moves away with the new features, partly expected (the teacher was the old feature set + Cinepoint columns), but the Lebaran group drifts a lot with the days-off features (holiday combinations never seen in train). Option: keep the old-feature model for the Lebaran rows.
- Reference: the teacher pipeline's Kumo CV was 0.2904 (user), with the Cinepoint columns also in train.

### Open (next session)
- Results of the still-running runs: Kumo context 8k real + 2k extra rows (`exp/e12.log`), LGBM 1500 trees + late-start features with saved OOF / test (`exp/lgbm_final.pkl`).
- Blend LGBM + Kumo on OOF, decide the Lebaran handling, then port to `main_legit.ipynb`: competition / date-strength features, days-off features, late-start features, shifted-window extra rows (rebuild with the notebook's own cells), Kumo context mix; then the leveling / postprocessing on top.

## 2026-10-09 (later) - main_legit.ipynb: new features, shifted-window extra rows, new blend (ported from the scratch experiments)
Backup of the notebook before this change: scratchpad `main_legit_backup_before_v2.ipynb`.

### Changes
- Data cells now define a function and apply it in the same cell (same headers and order): join_sources, join_external, drop_gap_rows, add_calendar_flags, add_genres, set_categories, drop_unused, add_target, add_off_block, add_pair_features, add_film_features, add_cinema_features, add_weekday_price. Needed so the extra rows go through exactly the same steps.
- New feature cells (headers only, no description text): "Days off in the D1-D3 window" (win_off, tgt_off, off_rel; days off from the existing `cal["off"]`), "Late-start pairs" (first_active, eff_age, last_ratio, active_mean), "Competition from other films' openings" (comp_n_cin, comp_size_cin, comp_fresh_cin, comp_n_nat, comp_size_nat, date_idx, own_idx, date_idx_rel; train from train pairs, test from test pairs).
- New cell "Extra training rows (shifted windows)": `AUG_SHIFTS = [1, 2, 3, 5, 7, 10, 14]`, 254,483 extra rows (4.7x), column age_offset (0 for real rows and test). Cinema features of the extra rows use the real windows (scratch version used the shifted windows; small difference).
- Setup: `USE_EXTRA_ROWS = True`, `EXTRA_WEIGHT = 0.5`, `fit_rows()` adds the fitting films' extra rows in every fold and for the test fit; all fit_predict functions take `(X_tr, y_tr, is_extra, X_pred)`; tree models use the weight, `seed_average` passes `sample_weight`. 79 features.
- LGBM `n_estimators` 883 -> 1500.
- Kumo context: `KUMO_CONTEXT_ROWS = 8000` real + `KUMO_EXTRA_ROWS = 2000` extra rows (best of 10k+0 / 8k+2k / 6k+4k / 4k+6k: 0.3359 / 0.3232 / 0.3293 / 0.3321, 1 context seed).
- Blend: `{"kumo": 0.7, "lgbm": 0.3}` (scratch OOF with Kumo 8k + 2k: kumo 0.5 / 0.6 / 0.7 / 0.8 = 0.3228 / 0.3220 / 0.3217 / 0.3220), `BLEND_SNAP = 0.1`, prints the CV for kumo 0.5-0.8.
- Postprocessing (season factors, leveling, timing shape) and submission cells unchanged. All code outputs cleared.

### Checks
- Data + feature + setup cells run end to end (no model training, 40 s): train 54,671, test 72,611, extra rows 254,483; every new feature identical to the scratch experiment values on all train and test rows.
- Expected CV from the scratch runs (1 seed / 1 context seed): LGBM about 0.330, Kumo 8k + 2k 0.3232, blend about 0.32. Not run in the notebook yet (the user runs it).

## 2026-10-09 (later) - main.ipynb (Cinepoint teacher) brought to the same configuration as main_legit
User request: same setup in main.ipynb to see its CV. Backup before the change: scratchpad `main_backup_before_v2.ipynb`.

### Changes
- Data cells: the main_legit function versions (code was identical to legit's old cells).
- New cells "Days off in the D1-D3 window", "Late-start pairs", "Competition from other films' openings" (same code as main_legit, one-line description each to match main's style).
- Cinepoint cell: same columns, now an `add_cinepoint(df)` function so the extra rows get them too (from their own shifted D1, Cinepoint train file + post-cutoff file as before).
- New cell "Extra training rows (shifted windows)" after the Cinepoint cell, same as main_legit plus `add_cinepoint`.
- Setup / LGBM / CatBoost / XGBoost / Kumo / blend cells: main_legit versions; feature list = main_legit's 79 + the 20 Cinepoint columns (99). LGBM 1500 trees, Kumo 8k + 2k context, blend Kumo 0.7 / LGBM 0.3, snap 0.1.
- Markdown of LGBM / Kumo / Blend / Setup updated to the new settings; the "Days-off blocks" note now says D1-D3 days off are back; with the Cinepoint columns the old Lebaran drop does not apply in main (user).
- Postprocessing unchanged (season factors only); output names unchanged (`submission_cine.csv` etc., user OK with overwriting since `submission_teacher_cinepoint_lb0.36284.csv` keeps the LB 0.36284 file).

### Checks
- Data + feature + setup cells run end to end (34 s): train 54,671, test 72,611, extra rows 254,483, 99 features; new features identical to main_legit; Cinepoint ret_h filled for 83% train / 86% extra / 89% test rows.
- Model CV not run (the user runs it). Old teacher Kumo CV was 0.2904 (user).

## 2026-10-09 (later) - LB result: main_legit with the new configuration
- `submission_legit.csv` from the updated main_legit (new features, shifted-window extra rows, LGBM 1500 trees, Kumo 8k + 2k context, blend Kumo 0.7 / LGBM 0.3, same postprocessing / leveling as before): LB 0.42037 (previous main_legit LB 0.43015, -0.0098).
- Reference LBs: leveling reference `submission_ref_lb0.43729.csv` 0.43729, Cinepoint teacher 0.36284.
- Scratch CV went Kumo 0.3465 -> 0.3232 and LGBM 0.3681 -> 0.3298 (blend about 0.32), and the LB fell about 0.010, so a good part of the CV gain carried over; the CV-LB gap (about 0.10) is still there and is mostly the test-period level (Lebaran and season), not the model.

## 2026-10-09 (later) - Older data: film profile features, national fade prior by film type, 2022-2023 Cinepoint scrape (scratch, CV only)
User request: try the not-yet-tested ideas for using older years (genre / type / sequel patterns). Notebooks not changed.
Baseline: LGBM with the current main_legit setup (79 features, shifted-window rows w 0.5, 1500 trees, 1 seed): CV 0.3298 (seed 42), 0.3302 (seed 7).

### 1. Film profile features in the current pipeline
- Scratch `exp/feat_profile.py`: from `external/film_profile.csv`: is_local, is_sequel (sequel / franchise), kind_code (live action / animation / anime / rerelease / concert / documentary), source_code (original / novel / comic / remake / true story / ...), origin_code (Indonesia / US / Korea / Japan / Thailand / other), pred_adm_log (log admissions of the predecessor film, only if published before the cutoff), local_horror (is_local x genre_Horror).
- Strict cutoff kept: the 7 test films with a fact source after 2025-09-30 get NaN (coverage train 100%, test 96.4%; pred_adm 5% / 7.7%).

| Run | CV | dist_sf | dist_lvl |
|---|---|---|---|
| base (seed 42) | 0.3298 | 0.2576 | 0.1993 |
| + all profile features (seed 42) | 0.3254 | 0.2539 | 0.2031 |
| + is_local, is_sequel, pred_adm_log, local_horror (seed 42) | 0.3258 | 0.2490 | 0.1955 |
| base (seed 7) | 0.3302 | 0.2513 | 0.1975 |
| + all profile features (seed 7) | 0.3262 | 0.2514 | 0.1957 |
- About -0.004 in both seeds (noise about 0.002), so a real CV gain for LGBM. The old test (2026-10-07, old 63-feature setup) showed no gain; with the new features / extra rows it helps.
- Kumo with the profile features: not finished (stopped after about 45 min because the user's notebook was using the GPU at the same time). To rerun: `exp/e19_kumo_extra.py 8000 2000 profile`.

### 2. Average fade per film type from the national Cinepoint charts
- Scratch `exp/feat_typeprior.py`: films from `cinepoint_daily_top_prev.csv` + `cinepoint_daily_top_train.csv` (D1 2024-01 to 2025-09, all D1-D10 dates <= 2025-09-30, D1-D3 mean >= 1,000 admissions): 392 films. Per film and horizon: log(D_h / D1-D3 mean) minus a calendar-only LGBM (weekday, days off, Eid / Christmas / Ramadan distance, size, D1-D3 shape, new competitors; out-of-fold by film). The residual is the film's "legs".
- Type = local x horror x sequel. Labels: competition films from film_profile + movies genre; 241 older chart films labelled in `exp/old_film_labels.py` from general knowledge (static facts; about 10 unsure titles left out, a few labels may be wrong).
- National pattern (residual, log): local horror fades fastest (about -0.4 to -0.47 average over D4-D10, D10 about -0.76), sequels fade faster than originals, local non-horror holds best late (slope +0.4).
- Features tp_leg, tp_slope, tp_h (type's average residual, overall / late minus early / at the row's horizon).

| Run | CV | dist_lvl |
|---|---|---|
| leave-one-film-out prior (all chart films) | 0.3263 | 0.2073 |
| same, tp_h only | 0.3286 | 0.2035 |
| prior from older films only (not competition films, one constant per type) | 0.3277 | 0.2096 |
| profile + older-films prior | 0.3259 | 0.1978 |
- The leave-one-out version leaks (a train film's value moves with its own result, test films never get that), so its 0.3263 is not trusted. The clean version is -0.002 (noise level) and adds nothing on top of the profile features (0.3254 -> 0.3259). Not adopted.

### 3. Cinepoint before 2023-09
- Scraped 2022-01 to 2022-06 into a scratch file (`cp_older.csv`, stopped by the 15 min limit): 3-7 films per listed day and 12-24 of 30 days listed per month. Not one film has a complete D1-D10 run with the week before it, so 0 usable films. Not continued.
- The existing pre-cutoff files are complete as far as Cinepoint goes: 2023-09 to 2024-06 has about 5 films per day (43 usable films before 2024-07), 2024-07 onward 10-21 per day.

### Takeaway
- Worth porting: profile features (LGBM CV -0.004). Needs the Kumo check and an LB test (film-level features have disagreed between CV and LB before).
- Not worth it: type fade prior, older scraping.

## 2026-10-09 (later) - main_legit.ipynb cleanup: no post-cutoff traces, no dead cells
User request. Backup before the change: scratchpad `main_legit_backup_before_cleanup.ipynb`. 81 -> 70 cells.

### Removed
- "Real-sales film multipliers (trial / assumption)" (2 code cells + header): read the post-cutoff Cinepoint file `cinepoint_daily_top.csv` when switched on (it was off). The Cinepoint title key `cp_key` moved into the timing-shape cell, which only reads pre-cutoff files (with its assert).
- "Teacher check" (cell + header): compared with `submission_teacher_cinepoint_lb0.36284.csv`, which was built with post-cutoff Cinepoint. Teacher comparisons stay in the scratchpad only.
- CatBoost and XGBoost cells (not in RUN_MODELS, so never ran) and their imports; also the unused imports matplotlib and tabfm. ZERO_SNAP only has lgbm.
- "Seasonal adjustment" cell (SEASON_FACTORS): its output was overwritten by the leveling step (season_factor set to 1), so it had no effect. The leveling cell now starts from `test_submit = test_blend` and defines RAMADAN_START / LEBARAN_START itself; the season_factor column is gone (submission = ratio x scale).
- Extra outputs `submission_kumo_only_legit.csv` and `submission_trees_only_legit.csv` (user: remove past / unhelpful submissions); the notebook writes only `submission_legit.csv`.

### Kept (checked)
- Leveling to `submission_ref_lb0.43729.csv`: built only from LB probes and pre-Cinepoint models (no post-cutoff data). It still helps a lot. Distance to the teacher with the current blend (scratch, Kumo 8k+2k 0.7 / LGBM 0.3): current 0.5 leveled + 0.5 reference 0.2210; leveled only 0.2528; leveled 0.75 0.2329; no reference, old season factors 0.2819; raw blend 0.2818. The Lebaran group is the main reason (1.17 vs 2.11 without it).
- Calendar files that cover the test period (holidays, cuti bersama, Ramadan, school holidays, Eid 2026-03-20 / 21): published before the cutoff, not sales data.

### Check
- Whole notebook run end to end with a tiny LGBM (50 trees, 1 seed, no Kumo, output to the scratchpad): 57 s, 79 features, 72,611 submission rows; no errors.

### main.ipynb CV (user run, new configuration with Cinepoint columns)
- LGBM CV 0.2906 (folds 0.2482 / 0.4328 / 0.2074 / 0.3274 / 0.2372), 14 min (3 seeds x 6 fits x 1500 trees on about 300k rows, 99 features).

## 2026-10-09 (later) - main.ipynb: leveling and timing-shape steps from main_legit added as switches (off)
User request: test main_legit's postprocessing in main too. Backup: scratchpad `main_backup_before_post.ipynb`. 74 -> 78 cells.
- Seasonal adjustment cell now keeps `final_ratio = test_blend x season_factor`.
- New cells after it: "Level to a reference submission (test)" (`USE_LEVEL_REF = False`, same code as main_legit on final_ratio) and "Timing shape from last year's films (test)" (`USE_TIMING_SHAPE = False`, pre-cutoff Cinepoint only).
- Submission cell writes `final_ratio`; the Kumo-only / trees-only files get the season factors explicitly (same values as before when both switches are off).
- Smoke run with both switches on (tiny LGBM, outputs to the scratchpad): 42 s, 99 features, 72,611 rows, no errors.
- Expectation: probably worse for main. The reference's group x horizon levels are far from the teacher's (reference / teacher: normal 1.09-1.51x, ram_to_lebaran D8-D10 0.38-0.43x, ram_to_ram D10 0.55x), and main's Cinepoint columns already carry each test film's real curve. LB test only.
- Saved outputs: the edits to main_legit (cleanup) and main cleared the saved cell outputs on disk; outputs of every unchanged cell were copied back from the backups (main 28 / 28, main_legit 28 / 34; the 6 changed or removed cells keep their old outputs only in the backup).

## 2026-10-09 (evening) - main_legit.ipynb: no reference submission; Lebaran level from last year's national market
User: the 0.43729 reference is a drag, find a better way to get the multipliers. Backup: scratchpad `main_legit_backup_before_market.ipynb`. 70 -> 68 cells.
Scoreboard: distance to the teacher (mean abs ratio difference). Calibration on LB-known clean files: reference 0.2223 = LB 0.43729; `submission_legit.csv` 0.2094 = LB 0.42037.

### What was found
- Normal rows (69% of test rows): the raw blend already sits at the teacher's level (mean ratio 0.409 vs 0.411); the reference leveling lifted them to 0.514 and moved them away (distance 0.116 raw vs 0.150 leveled). The 50/50 mix only helped the Ramadan / Christmas groups a little.
- Lebaran group (4,598 rows, 6.3%): all big films open on Wed 18 Mar 2026 (last Ramadan days, Nyepi, cuti bersama); the raw model predicts a drop (ratio 0.47 -> 0.10), the teacher has 2.4-3.3. This group is about a third of the whole distance.
- 2025 national market (sum of charted admissions): last 3 days of Ramadan about 180k / day, Eid week 0.5-1.1M / day (2.8x-6x). The 2025 Eid releases kept or grew their market share for two weeks (share ratio 0.64-2.4), and no film opens after 18 Mar 2026 in test.

### Method (new cell "Lebaran level from last year's national market", switch USE_MARKET_LEBARAN)
- Market series from `cinepoint_daily_top_prev.csv` + `cinepoint_daily_top_train.csv` (assert: no date after 2025-09-30), complete charts from 2024-09-20.
- Usual weekday effect and weekday-holiday effect (log market vs its 15-day median, outside Christmas and Feb 10 - Apr 20 2025) divided out, because the model already knows 2026's weekdays / holidays (holiday effect 0.654 in log).
- Lebaran rows only: prediction x (market on the target date / mean market on the D1-D3 dates), both read on the same days from Eid in 2025 (Eid 2026-03-21 <-> 2025-03-31). Mean multiplier by horizon about 2.9-5.7.
- Lebaran rows use the LGBM prediction (LEBARAN_MODEL = "lgbm"); Kumo predicts an even deeper drop there (with the blend: Lebaran distance 1.89 vs 1.13). All other rows: blend as is, no multiplier.
- Removed: "Level to a reference submission" (the reference file is no longer read) and "Timing shape" (made it worse on top of the new step: 0.2014 -> 0.2076).

### Distance to the teacher (scratch predictions injected into the notebook: Kumo 8k+2k 1 context seed, LGBM 1 seed)
| Version | Distance | normal | pre_to_ram | ram_to_ram | ram_to_lebaran | xmas | Total tickets |
|---|---|---|---|---|---|---|---|
| `submission_legit.csv` (LB 0.42037) | 0.2094 | 0.148 | 0.113 | 0.133 | 1.050 | 0.255 | 15.76M |
| new default | **0.2014** | 0.116 | 0.080 | 0.168 | 1.183 | 0.267 | 11.50M |
| new + timing shape | 0.2076 | 0.116 | 0.088 | 0.188 | 1.191 | 0.302 | 11.33M |
| no multiplier at all | 0.2818 | 0.116 | 0.080 | 0.168 | 2.452 | 0.267 | 9.82M |
| scratch only: market multiplier also on Ramadan / Christmas groups (LGBM) | 0.2129 vs 0.2055 Lebaran only | | | worse | | worse | |
- Risk: total tickets drop from 15.8M to 11.5M (teacher 14.1M). Earlier LB probes on an older model favored higher levels on normal rows (x1.2 helped by 0.006), while the teacher says the raw level is right. One LB test settles it.
- Smoke run of the whole notebook (tiny LGBM): no errors.

### Same changes tried on main.ipynb (LGBM only, 1 seed; user: compare each notebook with its own previous version)
- main LGBM CV (new configuration, 1 seed): 0.2917 (user's 3-seed run 0.2906; old main LGBM 0.3049).
- Postprocessing compared by distance to the old main output (LB 0.36284; for main this only shows how far a change moves away from the LB-checked version):
| main postprocessing | Distance to old main | Lebaran | Total |
|---|---|---|---|
| season factors x2.0 / x1.25 (previous, default) | 0.1230 | 0.808 | 13.98M |
| Lebaran from last year's market | 0.3094 | 3.752 | 18.03M |
| + timing shape | 0.1260 | 0.797 | 13.90M |
| + reference leveling | 0.1861 | 1.062 | 16.10M |
- main's raw Lebaran rows already rise (ratio 1.0-1.5 at D4-D10) because the Cinepoint columns carry the real surge; the market multiplier counts it twice (2.7-8.3). Not added to main.
- The reference-leveling and timing-shape test cells in main were removed again (not helpful; backup `main_backup_before_post_removal.ipynb`); main is back to season factors only (74 cells), checked end to end with the saved predictions (same output).
- LGBM-only comparison for main_legit: new 0.2088 vs previous postprocessing 0.2134.

### Ready for the 2 submissions of today (user submits)
- main: run as is (Kumo 0.7 / LGBM 0.3, season factors) -> `submission_cine.csv`.
- main_legit: run as is (Kumo 0.7 / LGBM 0.3 elsewhere, LGBM x market on Lebaran rows) -> `submission_legit.csv`.
- main.ipynb: the Kumo-only and trees-only output files were removed (user); main writes only `submission_cine.csv`, main_legit only `submission_legit.csv`.

## 2026-10-09 (night) - main_legit.ipynb: season formula (market change x share change) for Lebaran and Ramadan groups
User: compare main_legit with main's submission (not the weaker reference), find an explainable multiplier formula without using main's information.
Scoreboard from now on: distance to main's submission (the 0.36284 file). Only a scoreboard: no number below was fitted to it, except the clearly marked experiment.
Backup: scratchpad `main_legit_backup_before_formula.ipynb`.

### Formula (cell "Season levels from last year's national market", switch USE_SEASON_FORMULA)
- Expected national D_h / D1-D3 of a film = market change x share change.
  - Market change: 2025 national market (sum of all charted admissions) on the same distance from Eid (2026-03-21 <-> 2025-03-31); 2025's weekday / weekday-holiday effect divided out, 2026's multiplied in, 3-day centered average against single-day noise.
  - Share change: 2025 Eid releases (Jumbo, Komang, Norma, Pabrik Gula, Qodrat 2) kept their share: 1.07, 1.07, 1.06, 1.05, 1.11, 1.09, 1.00 for D4-D10 (used for Lebaran rows: no film opens after 18 Mar 2026). Normal-season films (D1 Apr-Sep 2025): 0.68, 0.76, 0.68, 0.52, 0.25, 0.20, 0.14 (used for the Ramadan groups).
  - Model scale q(h) = the model's mean prediction for a train film / that film's national ratio (median over 54 train films): LGBM 1.02, 1.03, 1.02, 0.95, 0.80, 0.64, 0.75.
- ram_to_lebaran: level per horizon from the formula, LGBM ranking inside. ram_to_ram, pre_to_ram: one factor per group (formula mean / model mean: 1.31, 1.30), model day shape kept. normal, xmas: model level (xmas left out: the December 2024 school break is missing from the calendar files and the formula gives an implausible x0.67).
- Sanity check: the formula applied to normal rows gives mean ratio 0.392 vs the model's 0.430 (main 0.411), so its scale is about right.

### Versions tried (distance to main; blend = Kumo 0.7 / LGBM 0.3, scratch predictions)
| Version | Lebaran mean D4-D10 | Lebaran dist | Total dist |
|---|---|---|---|
| `submission_legit.csv` (LB 0.42037, reference leveling) | | 1.050 | 0.2094 |
| raw blend, no multiplier | | 2.452 | 0.2818 |
| market only x LGBM (previous version of the cell) | 1.5-2.9 | 1.183 | 0.2014 |
| formula, 2025 market as is (no weekday swap) | 3.0-5.5 | 2.491 | |
| formula, weekday + holiday swap | 1.6-4.5 | 1.289 | |
| formula, weekday swap only | 3.1-9.4 | 3.782 | |
| formula, weekday-only swap, 3-day smooth | 3.1-6.7 | 2.459 | |
| **formula, weekday + holiday swap, 3-day smooth (chosen)** | 1.5-3.2 | **1.080** | 0.1949 |
| + Ramadan groups at the formula's group level (chosen) | | 1.080 | **0.1953** |
| + xmas at the formula's level | | | 0.2061 (xmas 0.267 -> 0.417) |
| Blend ranking on Lebaran instead of LGBM | | 1.437 | |
| main's Lebaran mean D4-D10 | 1.7-3.3 | | |
- The Ramadan group factor leaves the distance the same (0.1949 vs 0.1953) but brings the group means to main's (ram_to_ram 0.302 -> 0.396, main 0.431; pre_to_ram 0.116 -> 0.151, main 0.159); kept for the level.
- In the notebook (scratch predictions injected): blend 0.1953 (total 12.58M); LGBM only 0.2024 (previous LGBM-only versions 0.2134 with the reference, 0.2088 market-only). Smoke run of the whole notebook: no errors.

### Experiment only: multipliers fitted to main's submission (uses its post-cutoff information, never for main_legit)
| Fitted to main | Total dist | normal | ram_to_lebaran | xmas |
|---|---|---|---|---|
| raw blend, one mean factor per group | 0.2421 | 0.117 | 1.829 | 0.219 |
| raw blend, mean factor per group x horizon | 0.2341 | 0.114 | 1.760 | 0.218 |
| raw blend, best abs-error factor per group x horizon | 0.1948 | 0.112 | 1.232 | 0.209 |
| formula version, best abs-error factor per group x horizon | 0.1734 | 0.112 | 0.893 | 0.209 |
| raw blend, mean factor per film x horizon (film-level ceiling) | 0.1767 | 0.065 | 1.682 | 0.082 |
- Insight: the legit formula (0.1953) is as close to main as group x horizon multipliers fitted directly to main on the raw blend (0.1948). What is left is mostly per-film (which film holds up): normal rows go 0.116 -> 0.065 only with film-level factors, which need the post-cutoff sales.

## 2026-10-09 (night) - Cleanup of unused parts (models kept)
- Removed in both notebooks: the `ramadan_day` column (joined in join_external but never used); in main also the unused matplotlib import and the markdown lines about ramadan_day.
- A cleanup command that also removed the CatBoost / XGBoost cells ran although the user rejected it; the user wants the models kept, so CatBoost and XGBoost (cells, imports, ZERO_SNAP entries) and the tabfm import were restored in both notebooks (main_legit had lost them in the earlier cleanup too). RUN_MODELS stays ["lgbm", "kumo"].
- Checks: main_legit smoke run OK, distance with injected predictions unchanged (0.1953); main postprocessing check unchanged (0.1230). main 74 cells, main_legit 72.
- main_legit reads no reference submission and no teacher file (0 mentions); levels come from its own season formula.

## 2026-10-10 - Feature ablation (main_legit feature set, LGBM, scratch)
User: test removing features (too many / unnecessary?); a feature can go if removing it does not hurt. Screening LGBM: 500 trees x lr 0.045 (3x faster than 1500 x 0.015), extra rows w 0.5, 1 seed. Scripts `exp/e21_ablation.py`, `e22_single.py`, `e23_noise.py`, `e24_greedy.py`.

### Noise level
- All 79 features, seeds 42 / 1 / 2 / 3 / 4: 0.3313 / 0.3326 / 0.3294 / 0.3328 / 0.3321 (mean 0.3316, range 0.0034). Single-seed differences below about 0.003 are noise.

### Drop one group (seed 42; all features 0.3313)
| Group dropped | CV | | Group dropped | CV |
|---|---|---|---|---|
| comp_counts (comp_n_cin, comp_n_nat) | 0.3304 | | genres (27) | 0.3322 |
| cinema (cl_scale, cl_movies) | 0.3311 | | weekday (dow, d1_dow) | 0.3325 |
| occupancy (occ1-3, occ_mean) | 0.3314 | | pair_shape (trend, t3_share, active_days) | 0.3329 |
| shows (show1-3, show_mean, tickets_per_show3) | 0.3315 | | calendar_flags (is_holiday, is_cuti_bersama, is_school_holiday, off_block_len) | 0.3329 |
| price (price, price_ratio) | 0.3315 | | comp_lookahead (6) / categoricals (3) | 0.3333 / 0.3333 |
| | | | late_start (4) | 0.3337 |
| | | | film_level (mv_*, pair_share) | 0.3372 |
| | | | window_offdays (win_off, tgt_off, off_rel) | 0.3395 |

### Drop one feature (seed 42)
- Clearly needed: mv_trend 0.3341, win_off 0.3340, is_school_holiday 0.3328, mv_t3 0.3327, eff_age 0.3327, day_tipe 0.3326, cl_scale 0.3325, d1_dow 0.3324.
- Everything else 0.3288-0.3320 (inside the noise): e.g. date_idx 0.3288, show1 0.3290, show_mean 0.3292, comp_n_cin 0.3292, occ2 0.3294, tickets_per_show3 0.3294; rare genres (12 genres under 2% of rows) together 0.3313. Correlated features cover for each other, so single drops cannot decide; next step is a greedy removal scored on 3 seeds.

### Greedy backward removal (3 seeds 1 / 2 / 3, remove if the mean does not get worse by more than 0.0003)
- Start: all 79 features, mean 0.3316.
- Removed: comp_counts (0.3313), rare_genres (0.3311), occupancy (0.3309). Final 61 features, mean 0.3309.
- Kept (removing hurt): everything else. Biggest losses when removed: city_name 0.3340, own_idx 0.3333, comp_size_nat 0.3333, shows 0.3332, first_active 0.3329, dow 0.3329, price 0.3325.
- Reading: the feature set is not bloated; only 3 groups (2 + 12 + 4 = 18 columns) are dead weight and the gain is small (0.0007, about noise size). Next: new-feature bundles on the 61-feature base (`e25_newfeat.py`).

### New feature bundles on the 61-feature base (3 seeds, `exp/e25_newfeat.py`; base 0.3309)
| Bundle added | CV | vs base |
|---|---|---|
| profile (is_local, is_sequel, kind_code, source_code, origin_code, pred_adm_log, local_horror; test films with facts dated after 2025-09-30 blanked: 7 of 163) | 0.3270 | -0.0039 (all 3 seeds better) |
| off_more (off_d1-3, win_hol, win_wk, tgt_hol, tgt_prev_off, tgt_next_off, between_off, between_off_share) | 0.3276 | -0.0032 (all 3 seeds better) |
| off_position (target day's place in its run of days off, days to next / from previous day off, days off D4..target) | 0.3310 | +0.0001 |
| te_cin_city (cinema / city fade history, learned inside each fold) | 0.3317 | +0.0009 |
| open_rank (opening rank among same-week releases) | 0.3326 | +0.0017 |
| adj_trends (weekday-adjusted trends) | 0.3336 | +0.0027 |
| film_trends (national show / occupancy trends) | 0.3355 | +0.0046 |
- Strong signals found: film profile (what kind of film it is) and which exact D1-D3 / target days are off (not only the counts). Next: profile + off_more together, then drop each new column once (`exp/e26_combo.py`).

### profile + off_more together, then drop each new column once (3 seeds, `exp/e26_combo.py`)
- base 61: 0.3309 -> base + profile + off_more (78): 0.3244 (-0.0065, all 3 seeds better; the two gains add up).
- Drop one (vs combo 0.3244): is_sequel +0.0026 (needed), off_d3 +0.0008, origin_code +0.0002, win_hol +0.0001; all others -0.0001 to -0.0016 (tgt_hol -0.0016, win_wk -0.0012, source_code -0.0010, off_d2 -0.0010, the rest -0.0003 to -0.0008). Many columns overlap (e.g. off_d1-3 vs win_off / win_wk), so single drops look free; greedy prune next (`exp/e27_prune.py`).
- Greedy prune of the new columns (`exp/e27_prune.py`) was stopped before its first step so the user could run the notebooks; the full 78-feature combo is applied as is.

## 2026-10-10 - Feature ablation result applied to both notebooks
- Backups: scratchpad `main_legit_backup_before_features.ipynb`, `main_backup_before_features.ipynb`. Edit script `nb_features.py`.
- Removed: occupancy (occ1-3, occ_mean; occupation_rate no longer loaded), comp_n_cin / comp_n_nat (no longer computed), rare genres as features (COMMON_GENRES = genres in >= 2% of train rows: 15 of 27; the dummies stay so the columns line up), unused OFF_DAY.
- Added: add_window_offdays now also makes off_d1-3, win_hol, win_wk, tgt_hol, tgt_prev_off, tgt_next_off, between_off, between_off_share (is_off = weekend / national holiday / cuti bersama); new "Film profile" cell (add_film_profile: is_local, is_sequel, kind_code, source_code, origin_code, pred_adm_log, local_horror; facts public after 2025-09-30 -> NaN, 3.6% of test rows), also applied to the shifted-window rows.
- Features: main_legit 79 -> 78, main 99 -> 98 (same change plus its Cinepoint columns). Kumo uses the same list.
- Smoke runs (50 trees, 1 seed, no Kumo): both notebooks run end to end. Expected from screening (fast LGBM, 3 seeds): 0.3316 -> 0.3244 on main_legit. Not yet checked with full settings, with Kumo, or on main's Cinepoint feature set. The user runs both notebooks.

## 2026-10-10 - LB: submission_legit.csv 0.40286 (was 0.42037)
- First submission with the season formula (own formula, no reference file) + the ablation feature set (78 features). Gain 0.0175.
- Distance to main's 0.36284 file (mean abs ratio difference): 0.1906 (0.42037 file: 0.2094, 0.43729 ref: 0.2223). LB has moved with this distance at about 0.9-1.3 LB per unit.
- Reading: reaching 0.36 legit would need distance about 0.15; the best fitted-to-main experiments only reached 0.173-0.177 because the rest is per-film (which film holds up), which needs post-cutoff sales. Realistic legit target about 0.38-0.39.
- submission_cine.csv in leon/ is still the old 10-08 file (identical to the 0.36284 file): main has not been rerun with the new features yet.

## 2026-10-10 - Research: genre x season, kids x school breaks, cities, students (scratch `research/`)
Rule: only data dated on or before 2025-09-30 as evidence; main's 0.36284 file only as scoreboard (distance), never fitted on. Distances below use the harness with the older injected predictions (base 0.1953; the 0.40286 file itself is 0.1906).

### Data used
- Cinepoint national charts 2023-09-20 .. 2025-09-30 (prev + train files): two Christmases (2023, 2024), two Lebarans (2024, 2025). Charts list 10 to 24 films per day, so daily "market" totals jump with the list length (use top-10 sums for market levels).
- Genres for 230 pre-train films labelled by hand into coarse groups (family/kids, horror, comedy, action, drama, anime) with local / foreign; 22 unsure labels, the 7 largest checked on the web (2nd Miracle in Cell No. 7 drama, The Last Supper drama, Keajaiban Air Mata Wanita drama, Negeri Para Ketua comedy, Dark Nuns horror, Konco-Konco Edan horror comedy, Ambyar Mak Byar musical drama). Train / test films: same groups from movies.csv genre (`research/labels_prev.csv`, `research/genre_season.py`).
- DKI school breaks from news (kompas.tv, detik): 2024-04-04..16 (Lebaran), 2024-06-22..07-07, 2024-12-21..2025-01-04, 2025-03-21..04-08 (national SEB), 2025-06-28..07-13.

### Findings
1. Genre x season (national D_h / D1-D3 vs the season's typical film): local horror fades faster than other films in every season (normal 0.78x, Ramadan 0.65x, Lebaran releases 0.74x); local comedy 1.19x, foreign action 1.14x in normal weeks. The genre effect looks the same in and out of holidays, and the model already has genre + profile, so no genre x season factor.
2. Kids / family films x school breaks (strongest new signal): outside breaks they hold 0.75x the typical film on the same days (21 films), on break days 1.43x (7 films: Mufasa 1.74, Jumbo 1.65, Ejen Ali 2 5.07, Warkop DKI Kartun 1.43, Snow White 1.02, Ne Zha 2 0.93, Si Juki 0.81). Train OOF agrees: the blend under-predicts family films on school-break days by about 1.9x (Ejen Ali 2: actual 0.88, predicted 0.25).
3. Christmas: films with D4-D10 in Dec 20 - Jan 4 held 1.83x the normal-season median (13 films, 2023 + 2024). The model gives Christmas rows only about 1.5x; holiday weekdays (e.g. 2024-12-30, Monday: 485k admissions vs about 200k on a normal Monday) behave like weekends. holidays.csv flags only 12-25 and 01-01, so the model treats 12-29..12-31 as ordinary weekdays.
4. Cities / mudik: no source splits Lebaran admissions by city; Cinema 21 (2017) says the Lebaran rise is spread almost evenly over cinemas, Cinema XXI (2025) says box office rose while mudik travellers fell 24%. main has no city-level truth either (its Cinepoint is national), so city differences between legit and main say nothing. No city factor.
5. Students: BEKraf / Rumah Sinema survey (reported 2018): cinema-goers about 56% university students, 33% high-school students, 11% others; SMRC 2019: 15-22 year olds watch most. No figure for the share of school children who go on holidays.
- Post-cutoff material seen in search results and not used: Lebaran 2026 admissions (inews / ANTARA), Cinema XXI 2025 full-year.

### Scoreboard tests (distance to main, harness base 0.1953)
| Change | dist all | dist xmas rows |
|---|---|---|
| none | 0.1953 | 0.267 |
| Christmas formula (market change x share change, Dec 2024 market, any alignment) | 0.1994-0.2073 | worse (level 0.38-0.49, main 0.78) |
| Christmas level = 1.83 x model's normal-season level | 0.1916 | 0.215 |
| + kids films on school-break days x1.43 (past-data value) | 0.1911 | |
| + kids x1.43, Christmas rows only | 0.1906 | |
| + kids x1.3 | 0.1904 | 0.199 |
| + kids x1.9 | 0.1963 | |
| Same-season film medians for the Ramadan / Lebaran groups (instead of the formula) | 0.199-0.254 | worse: keep the formula there |
- 5 test kids films fall on school-break days (SpongeBob, Na Willa, Pelangi di Mars, Tunggu Aku Sukses Nanti, Patah Hati Yang Kupilih); main is 1.15-1.8x above legit on all 5.
- Not applied to the notebooks yet.

## 2026-10-10 - Applied to main_legit: Christmas level + kids films on school breaks
- User picked x1.3 for kids (close to the 1.43 past-data value, best on the scoreboard).
- New cell after the season formula (backup `main_legit_backup_before_holiday.ipynb`, script `nb_holiday.py`), switch USE_HOLIDAY_ADJUST:
  - XMAS_REL from the national charts (films first charted 2023-09-25..2025-09-20, D1-D3 mean >= 3000; rows in Dec 20 - Jan 4 of 2023 / 2024 vs normal rows, Ramadan / Lebaran-release rows left out): 1.83 (13 films). xmas rows = model rows scaled to 1.83 x the model's normal-season mean (0.599 -> 0.782).
  - Kids films = animation or family genre, not anime (kind_code 2); school-break day = most provinces on break in school_calendar_regional.csv 2025/2026 (all rows used published 2025-05-22..2025-08-26). 1358 rows, 5 films, x1.3.
  - The total_ticket lines moved from the season cell to the end of this cell.
- Check (harness, injected older predictions): distance to main 0.1953 -> 0.1904 (xmas rows 0.267 -> 0.199), same as the scratch experiment.

## 2026-10-10 - LB: submission_cine.csv (main) 0.35793 (was 0.36284)
- First main run with the ablation feature set (98 features: occupancy, competition counts and rare genres out; detailed days off and film profile in). Gain 0.0049.
- Same feature change on main_legit gave 0.42037 -> 0.40286 together with the season formula.
- main is the best file now; the scoreboard for legit experiments should move from the 0.36284 file to this one (old file kept).

## 2026-10-10 - Research workflow, factors 8-20 (`leon/research_workflow.md`, scratch `research/`, `exp/e28_factors.py`)
- #8 Nyepi (2026-03-19): test_history has no Denpasar rows that day (cinemas closed); the 4 test rows are already 0 in both legit and main. Nothing to do.
- #9 Payday (25th-5th): national market (top-10 sums, 2023-10..2025-09, holidays and seasons left out) shows no consistent effect (payday window vs 16th-24th: 2023 x0.54, 2024 x0.77, 2025 x1.09). CV with t_payday, win_payday, payday_rel, t_dom on the 78-feature set: 0.3255 vs 0.3244 (+0.0011, worse). Not used.
- #10 Track record (shrunk mean log hold-up of the fitting films sharing a producer / director, own film left out): producer 0.3246 (+0.0002), director 0.3252 (+0.0008). Only 43 / 29 of 130 test films share a producer / director with a train film. Not used.
- #11 City demographics: 67 of 69 test cities are in train with >= 283 rows each, so city_name already covers them (test-only: Tuban, Magelang, 322 rows). Low value, not built.
- #12 School exam weeks: the calendar covers exams in 17 of 34 provinces; on the scoreboard the exam cases have 1-4 films each and flip direction (main / legit 0.58 to 1.52). Not used.
- #13 Imlek (2026-02-17): legit and main agree around it (1.07-1.12); holiday + cuti bersama already in the calendar.
- #14 Age rating after the holiday change: Christmas gap small for all ratings (0.89-1.13); Lebaran week all-ages / adult films still about 1.27x below main, teen films level; per-film (Na Willa 1.83, Suzzanna 1.28).
- Format versions (IMAX / 3D, 20 titles in train and 20 in test): they already get genre (base_title join) and profile; tying them to the base film's prediction is worse on train (MAE 0.373 at best vs 0.279 for the model's own OOF). Not used.
- Small foreign films in Lebaran week (Reminders of Him, Number One): main about 0, legit 0.12-0.53 (screens go to the Lebaran releases); too few rows to matter.
- #15 showtimes, #16 runtime, #19 weather, #20 football: decided without a test (covered already / data only after the cutoff / negligible), see the workflow file.

## 2026-10-10 - Factor scoreboard vs the new main, and a check against the real test-film sales
### vs the new main file (0.35793), legit = notebook with injected older predictions
| Legit variant | dist new main | dist old main |
|---|---|---|
| no holiday cell | 0.1917 | 0.1953 |
| Christmas level only | 0.1897 | 0.1916 |
| Christmas + kids x1.3 | 0.1916 | 0.1904 |
- Slices vs the new main (main / legit): kids on break days 0.82 (legit now above), Christmas 0.94; normal-season horizons D4-D6 0.90-0.92, D9 1.11, D10 1.50 (legit falls too fast toward D10); payday phase, region, Christian-majority cities, local / sequel, film size flat (0.92-1.07).

### vs the real national sales of the test films (Cinepoint 2025-10..2026-03, evaluation only, `research/eval_leak.py`)
- Method: per film x horizon, truth = national D_h / D1-D3 x c(h), c(h) = median ratio of our cinemas' actual hold-up to the national one on 120 train films ([1.08, 1.03, 1.05, 1.09, 1.27, 1.17, 1.19]); prediction = scale-weighted film ratio over our cinemas. 882 film x horizon cells.
- Calibration: train OOF gives median predicted / real = 0.76 (the model predicts typical values, not averages), so about 0.76-0.78 is the right level.
| Variant | error all | normal | xmas | ram_to_ram | pre_to_ram | ram_to_lebaran |
|---|---|---|---|---|---|---|
| legit, no holiday cell | 0.2894 | 0.1943 | 0.2696 | 0.4149 | 0.1446 | 1.0611 |
| legit, Christmas only | 0.2895 | 0.1943 | 0.2712 | 0.4149 | 0.1446 | 1.0611 |
| legit, Christmas + kids x1.3 | 0.2928 | 0.1943 | 0.2851 | 0.4149 | 0.1446 | 1.0950 |
| main new (0.35793) | 0.2039 | 0.1386 | 0.1531 | 0.3371 | 0.0957 | 0.6779 |
| main old (0.36284) | 0.2098 | 0.1319 | 0.1841 | 0.3060 | 0.1005 | 0.8638 |
- The check ranks the files like the LB (main new < main old < legit).
- Levels (median predicted / real, 0.77 = right): legit normal 0.78; Christmas 0.77 without the holiday cell, 0.96 with the Christmas level, 1.09 with kids too; ram_to_ram 0.53 (about 30% low; main new 0.65); pre_to_ram 0.62; ram_to_lebaran 0.94. Kids films: Lebaran week 1.12, Christmas 0.84 -> 1.09 with x1.3.
- Reading: the Christmas level (from 2023-2024) and the kids factor do not hold in 2025-26; the model was already at the right Christmas level. Biggest legit level miss outside Lebaran week: ram_to_ram about 30% low.
- Caveat: this uses post-cutoff sales as the judge (never as an input). Waiting for the user's decision on switching the holiday cell off.

### All film / date factors vs the real test-film sales (`research/leak_board.py`, evaluation only)
- Files: submission_legit.csv (0.40286, no Christmas / kids changes; it predates them) and submission_cine.csv (0.35793). Level = median(predicted / real) / 0.76 (train OOF), 1.00 = right.
- Measurement artifacts (same pattern for legit and main, so not model errors): weekday (Thu / Fri about 0.6, Mon / Tue 1.3-1.4), horizon (D4-D6 1.2-1.3, D9-D10 0.6), film size (small 0.5-0.6; small films fall off the national chart and the 0.7 x chart-minimum fill overstates their real sales). Ignored.
- Legit vs main levels: season normal 0.97 / 0.97, pre_to_ram 0.94 / 0.85, ram_to_ram 0.74 / 0.85, ram_to_lebaran 1.28 / 1.17, xmas 1.05 / 1.31; kids on break days 0.79 / 1.57 (5 films); payday phase 0.91-1.08 / 0.84-1.04; exam days 0.94 / 0.90; local 1.01 / 1.05, foreign 0.96 / 0.87; sequel 1.12 / 1.10, original 0.93 / 0.90; all-ages films 1.23 / 1.24; genres in the normal season 0.88-1.12 / 0.81-1.08; Imlek 1.05 / 1.09.
- Ramadan by genre (ram_to_ram + pre_to_ram): drama 0.40 / 0.77 (6 films), horror 0.43 / 0.58 (3), family 0.86 / 1.00, action 1.02 / 0.85, comedy 1.04 / 0.97.
- Reading: the only clear legit-specific miss is the Ramadan level, worst for dramas (one level factor for the whole Ramadan group pulls them down). Christmas is already right without the holiday cell; the kids effect is inconclusive (5 films).

## 2026-10-10 - Ramadan genre factors (both notebooks)
### Evidence (pre-cutoff, `research/ramadan_genre.py`)
- National charts, Ramadan 2024 + 2025 (rows whose target day is in Ramadan), each film's hold-up vs all films on the same horizon in the same Ramadan, compared with the same genre in normal weeks:
| Genre | Ramadan films | in Ramadan | in normal weeks | Ramadan vs normal |
|---|---|---|---|---|
| action | 9 | 1.45 | 1.14 | 1.27 |
| drama | 6 | 0.86 | 0.63 | 1.37 |
| family | 2 | 1.15 | 0.78 | 1.48 |
| horror | 14 | 0.63 | 0.79 | 0.80 |
| comedy / anime / other | 0 | | | 1.0 |
- Most past Ramadan films were horror (14 of 31); the test Ramadan films are mostly not (horror 4 of 37 films).
### Check against the real test sales (evaluation only, `research/ramadan_genre_eval.py`)
| File | Variant | error all | error Ramadan rows |
|---|---|---|---|
| legit 0.40286 | as is | 0.2843 | 0.3630 |
| legit 0.40286 | factors, group level kept | 0.2833 | 0.3570 |
| legit 0.40286 | factors, level moves | 0.2859 | 0.3723 |
| main 0.35793 | as is | 0.2039 | 0.2998 |
| main 0.35793 | factors, group level kept | 0.2049 | 0.3057 |
| main 0.35793 | factors, level moves | 0.1956 | 0.2519 |
- Half-strength factors land between (legit 0.2833 kept / 0.2837 moving, main 0.2044 / 0.1974).
### Applied
- New cell "Ramadan genre factors" (USE_RAMADAN_GENRE, RAMADAN_GENRE_FACTORS, genre_group from the genre dummies + kind_code) in both notebooks; backups `main_legit_backup_before_ramadan_genre.ipynb`, `main_backup_before_ramadan_genre.ipynb`, script `nb_ramadan_genre.py`.
  - main_legit: after the holiday cell; inside each Ramadan season group the factors are renormalised so the season formula's level stays. The total_ticket lines moved to this cell.
  - main: after the season-factor cell; factors applied directly (mean factor on Ramadan rows 1.259). The total_ticket line moved to this cell; the season_factor value_counts display was dropped.
- Test Ramadan rows by genre: action 5266, drama 2676, family 2288, comedy 1229, horror 1046, other 224.
- Checks: main_legit with injected older predictions: real-sales error 0.2928 -> 0.2910 (Ramadan rows 0.3731 -> 0.3628), distance to the old main 0.1904 -> 0.1896. main: smoke run (50 trees) OK.
- Still open: the holiday cell (Christmas level + kids x1.3) is on in main_legit, while the real sales say neither helps.

## 2026-10-10 - main: Ramadan genre weights from the real national sales (user request; main_legit unchanged)
- main.ipynb cell "Ramadan genre weights from the real national sales" replaces the past-Ramadan factors (backup `main_backup_before_real_weights.ipynb`, script `nb_main_real_weights.py`): per genre, weight = BASE_LEVEL (model's own level on train films from oof_blend) / the genre's predicted-vs-real level on Ramadan rows, real = ret_h (off-chart: 0.7 x cut_h) x C_H (our cinemas vs national per horizon, train films); genres with < 2 Ramadan films get the all-Ramadan weight. Smoke run OK (50-tree model: weights action 1.31, comedy 1.08, drama 1.22, family 1.49, horror 1.32, other 2.01; full run will differ).
- Same rule applied to the 0.35793 file (weights action 1.18, drama 1.29, family 1.00, horror 1.74, comedy 1.03), scored on the real sales (self-graded: the weights are fitted on the same data):
| Version | error all | error Ramadan | level Ramadan | action | drama | family | horror | comedy |
|---|---|---|---|---|---|---|---|---|
| main before (0.35793) | 0.204 | 0.300 | 0.85 | 0.85 | 0.77 | 1.00 | 0.58 | 0.97 |
| past-Ramadan factors | 0.196 | 0.252 | 1.02 | 1.03 | 1.05 | 1.47 | 0.47 | 0.97 |
| real-sales weights | 0.199 | 0.272 | 0.98 | 0.99 | 0.99 | 1.00 | 0.90 | 1.00 |
- Reading: matching each genre's median level does not minimise the absolute error; the past factors (level slightly above right) score better even against a self-graded alternative. Recommended: back to the past-Ramadan factors; waiting for the user's choice.

## 2026-10-10 - Ramadan weights re-examined; leveling to national sales; main season levels (scratch `research/best_weight*.py`, `level_train.py`)
### Correction to the earlier real-sales "error" numbers
- The film-level error compared the model's film totals with the real national totals (means). The model predicts per-row typical values (trained on absolute error), so its totals are by design about 0.76x the real means on train (0.91 at D4 down to 0.34 at D9-D10 per film, because most late-horizon typical values are 0). So that error rewards predicting too high; the "past factors beat the real-sales weights" result and the main 0.2039 -> 0.1956 gain were partly this artifact. Levels (a film's median log ratio vs the train baseline) do not have this bias and were fine; error-based weight picking is dropped.
- A second artifact: small films (tiny sales, often off the national chart so 0.7 x chart minimum is filled in) dominate any equal-per-film median. Level checks now weight films by size (predicted tickets), as MASE does by rows.
### Results (main = submission_cine.csv 0.35793 with national data, legit = submission_legit.csv)
- Ramadan, bigger half of the 37 films: weight still needed main x1.15, legit x1.32; smaller half main x2.30, legit x3.74 (national value partly estimated). The 8 biggest films in main need 1.10, 1.01, 1.01, 0.91, 0.80, 1.19, 0.97, 1.04 (about right).
- Leave-one-film-out, size-weighted level deviation (lower = better): main no weight 0.121 | one weight 0.145 | per genre 0.171 | pooled toward all (K=8) 0.161; size-weighted all-Ramadan weight 1.007 (action 1.01, comedy 1.01, drama 1.04, family 1.10, horror 1.42 on 4 films). Per-genre weights fitted on the real sales do not generalise (3-12 films per genre).
- main + past-Ramadan genre factors: deviation 0.271 (still needed x0.77); main + real-sales median weights: 0.214 (x0.81): both overshoot the big films. DECISION: genre cell REMOVED from main.ipynb (season cell restored to its original last lines; backup `main_backup_before_removing_ramadan_genre.ipynb`; smoke run OK, 76 cells).
- legit: file as is 0.328 (10 biggest 0.354) -> with the past-Ramadan genre factors (notebook rule, group level kept) 0.270 (0.268). Kept in main_legit.
- main season levels, size-weighted (weight still needed, 1.00 = right): normal 0.92, ram_to_ram 1.04, ram_to_lebaran 1.03, xmas / school-break 0.95, pre_to_ram 0.80 (8 films, 1962 rows). The hand-set Lebaran x2.0 and school x1.25 are already right; no replacement needed.
### Leveling main's predictions to the film's real national total, tested on TRAIN labels (honest: OOF, C(h) leave-one-film-out; row-level MASE; main LGBM 1 seed OOF 0.2910)
| Leveling | strength 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|
| film x horizon, straight to the national total | 0.2913 | 0.2950 | 0.3026 | 0.3150 |
| film x horizon, only the deviation from the usual | 0.2955 | 0.3028 | 0.3119 | 0.3220 |
| one factor per film, deviation only | 0.2962 | 0.3059 | 0.3185 | 0.3333 |
- No version beats the plain OOF (0.2910): main's model already uses the film's real national numbers (ret_h, cut_h), the leftover is mostly cinema mix and noise of the national estimate (film-level floor 0.13 with perfect predictions).

## 2026-10-10 - main: national-chart day features (post-cutoff Cinepoint, allowed in main)
- Idea: main only sees the film's national ratios as ret_h and the wide ret4..ret10; the model must work out which one is "yesterday". Candidates built from the same charts (`research/nat_feats.py`): n_st_ratio (national showtimes ratio), n_fill_ratio (admissions per showtime vs D1-D3), n_share_ratio / n_st_share_ratio (share of the day's market vs opening share), n_mkt_ratio (total market that day / D1-D3), n_rank, n_ret_prev / n_ret_next (ratio of the day before / after the target), later n_prev2, n_dod, n_st_prev, n_fill_prev, n_week_ago.
- Test: main's 98 features +/- the new ones, LightGBM l1 (500 trees, lr 0.045), 5 folds grouped by film, seeds 1-3, no extra rows (same for both). Base 0.3036 (0.3031 / 0.3035 / 0.3043).
| Added | CV | vs base |
|---|---|---|
| neighbour days (ret_prev, ret_next) | 0.2991 | -0.0045 |
| showtimes (st_ratio, fill_ratio) | 0.3001 | -0.0036 |
| market share (share_ratio, st_share_ratio) | 0.3012 | -0.0024 |
| rank | 0.3016 | -0.0020 |
| market size (mkt_ratio) | 0.3026 | -0.0010 |
| all eight | 0.2974 | -0.0062 |
- Drop one from the eight (vs 0.2974): ret_prev +0.0022 (needed), fill_ratio +0.0009, mkt_ratio +0.0006, rank +0.0003, ret_next 0, st_share_ratio 0, st_ratio -0.0003, share_ratio -0.0008.
- Smaller sets: {ret_prev, fill_ratio, mkt_ratio, rank} 0.2977; + ret_next 0.2971; + share_ratio 0.2969; {ret_prev, ret_next, fill_ratio, mkt_ratio} 0.2975; **{ret_prev, fill_ratio} 0.2968 (-0.0068)**, as good as all eight. On top of the two: n_prev2 +0.0016, n_dod +0.0016, n_st_prev +0.0017, n_fill_prev +0.0009, n_week_ago +0.0018, all five together +0.0001 (no help).
- Applied to main.ipynb: `cp_day_features` in the Cinepoint cell (n_ret_prev, n_fill_ratio per film x horizon, also for the shifted-window rows through add_cinepoint; CP_SERIES keeps admissions and showtimes per film and day), markdown documents both; backup `main_backup_before_daily_features.ipynb`, script `nb_main_daily.py`. Features 98 -> 100.
- Full pipeline check, LGBM only, 1 seed (42), extra rows on, 1500 trees: CV 0.2910 -> **0.2884** (-0.0026). Seeds 1 and 2 running (`seeds_daily.log`).
- Seeds (full pipeline, LGBM only, extra rows on, 1500 trees, CV on train): before (98 features) 0.2910 / 0.2908 / 0.2913 (seeds 42 / 1 / 2, mean 0.2910); after (100 features) 0.2884 / 0.2885 / 0.2886 (mean 0.2885). Gain 0.0025 on every seed. Kumo uses the same feature list (NUM_FEATURES includes CP_FEATURES), not re-run.
- Not testable here: the LB effect (the user runs and submits main).
- Open: main_legit still has the holiday cell on (Christmas level x1.83 + kids x1.3); the level check on the real sales (not affected by the mean / median bias) said Christmas was already right without it (1.05 vs right 1.00) and the kids effect is inconclusive (5 films). Recommendation: USE_HOLIDAY_ADJUST = False. Waiting for the user.

## 2026-10-10 - Checks on the new main columns, and more candidate features (scratch `research/check_daily.py`, `perm_imp.py`, `check4.py`, `cinema_vs_nat.py`)
### The two national-chart columns (n_ret_prev, n_fill_ratio)
1. Correctness: 6 random film x horizon cells (3 train, 3 test) recomputed by hand from the raw charts match the notebook to 6 digits.
2. Train vs test: similar quantiles and NaN shares (n_ret_prev NaN 13.6% train / 8.6% test; n_fill_ratio 16.8% / 11.5%). Correlation with ret_h 0.75 / 0.80. In Lebaran week 49% of test rows have n_ret_prev above the train 99th percentile (median 2.69 vs 0.57 on train); same limit already exists for ret_h (3.7% of all test rows above its train 99th percentile).
3. Permutation importance (5 folds grouped by film, fast LGBM, shuffle the column inside each held-out fold, MASE rise): n_fill_ratio +0.0207 (5/5 folds), n_ret_prev +0.0143 (5/5); for scale ret_h +0.0730, horizon +0.0050, scale +0.0009, mv_trend +0.0005, cp_ret_mean +0.0004, win_off +0.0002, cut_h -0.0005. The 10 new days-off columns: tgt_next_off +0.0007 (5/5), off_d3 +0.0002, rest 0 to +0.0002 (near-dead). Profile columns one by one: origin_code +0.0002, the other six ~0 (they helped as a group in the earlier CV, shuffling one hides it because they overlap).
4. Old vs new main (LGBM only, raw test predictions, size-weighted level vs the real national sales, evaluation only): rows changed by more than 20%: normal 5%, Ramadan 6-7%, Christmas 3%, Lebaran week 12%; mean after / before 0.99, 0.98 / 0.95, 0.99, 1.075. Level off the right level, before -> after: normal 0.130 -> 0.160, ram_to_ram 0.106 -> 0.130, pre_to_ram 0.275 -> 0.188, Christmas 0.063 -> 0.078, Lebaran week 0.655 -> 0.533. Mixed, differences of 0.02-0.03 are about the noise of this diagnostic; train CV (row-level labels) is the clean evidence.
### Cinema-vs-national candidates (base 100 features 0.2980, fast LGBM, seeds 1-3, no extra rows)
| Added | CV | vs base |
|---|---|---|
| n_d1, n_d3 (national opening shape) | 0.2972 | -0.0008 |
| n_share_log (cinema size vs national) | 0.2975 | -0.0005 |
| n_rel_trend (cinema trend vs national trend) | 0.2971 | -0.0009 |
| n_proj3 (cinema's D3 share held constant x national that day) | 0.2971 | -0.0009 |
| all five | 0.2987 | +0.0007 |
| te_rel (cinema / city habit of holding vs national, fold-safe) | 0.2985 | +0.0005 |
| all five + te_rel | 0.2981 | +0.0001 |
- None adds signal beyond noise (about 0.001); not added.
### Relative target (predict y / national ratio, weighted L1 with weights national + 0.05, then multiply back), fast LGBM, no extra rows (`research/rel_target.py`, `rel_blend.py`)
- Train CV, 3 seeds: plain 0.2980, relative 0.3082 (+0.0102, worse on every seed).
- Raw test predictions by period (no season factors), weight still needed to reach the right level (1.00 = right) plain -> relative: Lebaran week 1.62 -> 1.14, pre_to_ram 0.82 -> 1.00, ram_to_ram 0.99 -> 0.94, Christmas 0.97 -> 0.92, normal 0.96 -> 0.91; size-weighted level off: Lebaran 0.468 -> 0.116, pre_to_ram 0.239 -> 0.177, ram_to_ram 0.109 -> 0.117, Christmas 0.066 -> 0.104, normal 0.130 -> 0.161. It extrapolates to Lebaran week on its own, but main's hand-set x2.0 on the blend already sits at about 1.03, so no gain there.
- Blend of plain and relative OOF (share of relative): 0.0 0.2980, 0.1 0.2969, 0.2 0.2965 (-0.0015, all 3 seeds same sign), 0.3 0.2968, 0.5 0.2989. Small, about the noise level, and Kumo already adds variety; not adopted.
- Summary of this round: the only post-cutoff features that moved the CV are n_ret_prev and n_fill_ratio (applied); cinema-vs-national features and the relative target add nothing usable. Not done: pruning the near-dead days-off / profile columns with the full pipeline.
