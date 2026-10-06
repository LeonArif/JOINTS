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
- Data eksternal hanya boleh digunakan apabila informasi atau versi data tersebut telah tersedia untuk publik paling lambat 30 September 2025.
- Batas waktu tersebut berlaku untuk seluruh film uji.

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
