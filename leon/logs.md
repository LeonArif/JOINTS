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
