# Experiment Logs (fayadh)

## 2026-10-08 - main.ipynb (market index features + LB-calibrated Lebaran factor)

Notebook: `main.ipynb`. Outputs: `submission.csv` (main), `fayadh_model.pkl` (5 LGBM models, 2.3 MB), `submission_p1_recalibrated.csv` (safe fallback, see below).

### Starting point
- Leon's `baseline.ipynb` (p1: v3 model + Lebaran x2.0 + all-school-holiday x1.25) = best public LB 0.47566. Reproduced exactly (CV 0.3893; predictions identical to p1 on all seasonal rows of `leon/submission_hybrid_decay_normal.csv`).
- Note for the team: `leon/submission.csv` is the decay-curve version (LB 0.48012), not p1.

### Diagnosis
- CV reweighted to the test composition (D1-D3 active-day pattern x pair scale) is about 0.398, but LB is 0.476. Test composition explains only part of the gap. The rest comes from the test period itself.
- The test period market is much quieter: median D1-D3 occupancy of new releases is 10.0 (test) vs 20.1 (train), and tickets per show 18.6 vs 32.7. Test movies are also wider (median 70 clusters vs 33).
- Market by month (median occupancy of pairs in D1-D3 of new releases, +-10 days): Apr 25.5, Sep 13.4, Oct 8.1, Dec 25.8 (Christmas), Feb 5.8, Mar 6.3. The last Ramadan week is about 1.4.
- Hypothesis: absolute features (occ_mean, tickets_per_show3, mv_t1..3) read a normal movie in a quiet market as a weak movie that will drop fast, and a normal movie in a busy market (Christmas) as a strong one. This fits the LB history: Lebaran rows needed a large boost, the Christmas x1.25 did not help (see below), and Leon found that big movies hold up better in test than train suggests.

### LB calibration of seasonal factors (no new submissions used)
- MASE as a function of a multiplier k on a group of rows is convex. The OOF residual distribution (Y / pred per horizon x pred bin) is used as a pseudo ground truth with an unknown group level c (actual = c x train-like).
- Lebaran group (D1-D3 touches Ramadan, target >= 2026-03-20; 4,598 rows): the observed v4 -> p1 delta (-0.01035) gives c of about 2.4, so the best k for the p1 model is about 2.4 (expected gain about 0.0015 vs x2.0).
- With that c, the Lebaran part of v3 -> v4 is about -0.0051, so the Christmas school-holiday x1.25 (3,267 rows) cost about +0.0025. Best k for that group is about 0.9 (c about 0.93).

### Experiments (LGBM L1 on ticket/scale, 150 trees, same params as Leon)
cv = GroupKFold(5) by movie; cv_w = cv reweighted to the test pattern x scale mix; time = train D1 < 2025-07-22, validate D1 >= 2025-08-01; time_w = time reweighted. Single seed 42 unless noted.

| Approach | cv | cv_w | time | time_w |
|---|---|---|---|---|
| Base (p1 features) | 0.3893 | 0.3985 | 0.3519 | 0.3301 |
| + market-relative feats at release (occ_rel, tps_rel, mvt_rel, mv_pc_rel) | 0.3840 | 0.3964 | 0.3473 | 0.3285 |
| Relative feats replacing absolute ones | 0.3942 | 0.4107 | 0.3734 | 0.3749 |
| Fully relative (pair volumes too) | 0.3952 | 0.4148 | 0.3717 | 0.3781 |
| + M_occ level | 0.3864 | 0.4010 | 0.3466 | 0.3349 |
| + target-date market ratio M_occ, window +-3 | 0.3806 | 0.3920 | 0.3447 | 0.3270 |
| same, window +-7 | 0.3844 | 0.3961 | 0.3554 | 0.3337 |
| target market ratio as multiplicative offset (beta 0.1-0.5) | 0.3844-0.3946 | worse | - | - |
| ratio variants M_tps / M_tk, windows 2-5 | 0.3791-0.3862 | 0.3909-0.3978 | 0.3446-0.3580 | - |
| **+ both ratios mr_M_occ + mr_M_tps (+-3) = final features** | **0.3795** | **0.3908** | **0.3412** | **0.3235** |
| final + movies.csv metadata (genre flags, age rating, format) | 0.3815 | 0.3926 | 0.3462 | 0.3291 |
| final + 5 main genres | 0.3799 | 0.3920 | 0.3435 | 0.3254 |
| CatBoost MAE 600 it depth 6 / blends 0.3-0.7 with LGBM | 0.3887 / 0.3801-0.3839 | worse | worse | worse |
| **Final, 5-seed average (seeds 42-46, in notebook)** | **0.3789** | **0.3907** | **0.3439** | **0.3250** |

- Market index: for each date, the median occupancy / tickets per show / tickets of all pairs that are in their D1-D3 (new releases) within +-w days. It is built the same way from train.csv and test_history.csv and never uses D4-D10 targets. Dates with no new releases are linearly interpolated.
- Not adopted: metadata (small or negative gain, and a risk of acting as a movie ID like the mv_* features), CatBoost, replacing absolute features.

### Effect on test predictions (final model vs p1 model, mean predicted ratio, before postprocessing)
| Group | Rows | final / p1 |
|---|---|---|
| normal | 48,044 | 1.11 |
| ram_to_lebaran | 4,598 | 1.07-1.09 |
| ram_to_ram | 10,767 | 1.26 |
| pre_to_ram | 1,460 | 1.40 |
| school_both (Christmas) | 3,168 | 0.88 |
| school_target_only | 2,947 | 0.91 |
| school_d13_only | 1,627 | 1.18 |
- The final model lowers Christmas by itself to about 0.88-0.90, close to the k of about 0.9 implied by the LB calibration, and raises the Lebaran rows. The direction matches the LB history without using it.

### Round 1 output (superseded by round 2 below)
- Lebaran group x2.2, no school-holiday factor. CV 0.3789 (5 seeds). Not submitted.
- A fallback `submission_p1_recalibrated.csv` (p1 model, Lebaran x2.4, no Christmas factor, expected about 0.471) was made and later removed, because the team has only one submission left.

## 2026-10-08 - main.ipynb round 2 (augmentation + external calendar + capacity)

Context: the team has a single submission left, so every decision below is backed by train validation and, for the out-of-distribution seasonal groups, by the existing LB probes.

### Validation used for decisions
cv / cv_w / time / time_w as in round 1, plus a **cross-market split**. Train movies are ranked by the market index at release (median occupancy of new releases, +-10 days). hi>lo = train on the top 40% (busy market), validate on the bottom 40% (quiet market, like test). lo>hi = the reverse. avg = mean of cv, cv_w, time, hi>lo, lo>hi.
- Bug found and fixed in my own harness: in the time and cross-market splits, augmented rows of movies outside the training side (gap or middle-market movies) were added to training. Every number below uses the fixed harness. GroupKFold numbers were not affected.

### Experiments (single seed 42)
| Approach | cv | cv_w | time | hi>lo | lo>hi | avg |
|---|---|---|---|---|---|---|
| Round 1 final features (market index), 150 trees, 31 leaves | 0.3795 | 0.3908 | 0.3412 | 0.3687 | 0.4425 | 0.3845 |
| cluster retention target encoding (OOF) | 0.3810 | 0.3921 | 0.3420 | 0.3656 | 0.4525 | 0.3866 |
| recency weights / hyperparameter grid (no augmentation) | 0.377-0.385 | | | | | 0.3837-0.3918 |
| **+ augmentation, shifted windows D1' = D1 + s, s = 1..8, 300 trees** | 0.3606 | 0.3778 | 0.3403 | 0.3317 | 0.4396 | 0.3700 |
| aug 1..8, 500 trees | 0.3566 | 0.3747 | 0.3414 | 0.3304 | 0.4398 | 0.3686 |
| aug 1..8,10,12,14, 300 / 500 trees | 0.3602 / 0.3566 | | | | | 0.3702 / 0.3685 |
| **aug 1..8 300 + cuti bersama / offday features** (is_cuti, is_offday, offday_d13) | 0.3524 | 0.3705 | 0.3229 | 0.3271 | 0.4340 | 0.3614 |
| + school holiday features too (+ offday_win3) | 0.3521 | 0.3711 | 0.3242 | 0.3261 | 0.4299 | 0.3607 |
| + offday_win3 / offday counts to target / prev-next offday | | | | | | 0.3639 / 0.3635 / 0.3620 |
| colsample 0.6 | | | | | | 0.3608 |
| 63 leaves (300 trees) | 0.3489 | 0.3671 | 0.3199 | 0.3231 | 0.4320 | 0.3582 |
| 600 trees (31 leaves) | 0.3495 | 0.3680 | 0.3185 | 0.3232 | 0.4333 | 0.3585 |
| 63 leaves, 600 trees | 0.3465 | 0.3651 | 0.3154 | 0.3210 | 0.4319 | 0.3560 |
| 127 leaves, min_child 100, 600 trees | 0.3418 | 0.3604 | 0.3100 | 0.3212 | 0.4364 | 0.3539 |
| 63 leaves, 600 trees, aug 1..8,10,12,14 | 0.3440 | 0.3625 | 0.3123 | 0.3170 | 0.4267 | 0.3525 |
| **63 leaves, 600 trees, aug 1..14 (chosen)** | 0.3427 | 0.3617 | 0.3109 | 0.3175 | 0.4255 | 0.3517 |
| aug 1..14,16,18,21 | 0.3433 | 0.3619 | 0.3081 | 0.3162 | 0.4256 | 0.3510 |
| 127 leaves, aug 1..14 | 0.3402 | 0.3587 | 0.3069 | 0.3188 | 0.4311 | 0.3511 |
- 127 leaves has slightly better cv/time but worse cross-market lo>hi, so 63 leaves was kept (the test period is a market shift).
- More shifts beyond 14 days: saturated.

### External data (all published before 2025-09-30)
- Cuti bersama 2025: SKB 3 Menteri (Oct 2024) + SKB for 18 Aug 2025 (Aug 2025). Dates: 28 Jan, 28 Mar, 2-4 & 7 Apr, 13 May, 30 May, 9 Jun, 18 Aug, 26 Dec 2025.
- Cuti bersama 2026: SKB No. 1497/2/5 Tahun 2025, signed 19 Sep 2025 (setneg.go.id; cnbcindonesia.com 2025-09-19). Dates in range: 16 Feb, 18 Mar, 20 Mar, 23-24 Mar 2026.
- School holidays: DKI Jakarta education calendars (same sources as Leon's round 2 entry).
- `holidays.csv` has no cuti bersama, which is why these were added.

### Seasonal-group check of the final model (mean predicted ratio vs p1 base model, before postprocessing)
| Group | v2: + offday (no school) | v3: + offday + school (final) | LB-implied level |
|---|---|---|---|
| normal | 1.09 | 1.08 | - |
| school_both (Christmas) | 0.73 | 0.89 | about 0.93 |
| ram_to_lebaran (target >= 2026-03-20) | 0.48 | 0.60 | about 2.43 |
| ram_to_ram | 1.24 | 1.27 | - |
| pre_to_ram | 2.0 | 2.0 | - (tiny absolute values: base predicted about 0.01-0.03) |
- Without school holidays, the offday features make the model expect a collapse after a holiday-heavy D1-D3, including 29-31 Dec, which are formally weekdays but still the peak of the year-end school holiday. Adding the school-holiday features brings Christmas back to the LB-implied level, so they were kept although their CV effect is neutral.
- The Lebaran rows (96% are the 18 Mar releases, whose D1-D3 are cuti bersama, Nyepi, cuti bersama) stay far from the LB-implied level for any model trained on train.

### Postprocessing
- The Lebaran group is exactly the rows with target date >= 2026-03-20 (identical to the old Ramadan-based rule, so no Ramadan start date is needed).
- These 4,598 rows use the **baseline (p1) model x 2.4**: the configuration the LB probes were run on, with the calibrated multiplier. All other rows use the final model with no multiplier. The Christmas x1.25 factor is dropped.

### Notebook (main.ipynb) ablation, as executed
| Step | cv | cv_w | time | time_w |
|---|---|---|---|---|
| 1. team baseline | 0.3893 | 0.3985 | 0.3519 | 0.3301 |
| 2. + market index | 0.3795 | 0.3908 | 0.3412 | 0.3235 |
| 3. + augmentation shift 1-14 | 0.3670 | 0.3834 | 0.3415 | 0.3282 |
| 4. + offday, cuti bersama, school holiday | 0.3597 | 0.3774 | 0.3319 | 0.3210 |
| **5. + 63 leaves, 600 trees = FINAL** | **0.3419** | **0.3614** | **0.3146** | **0.3068** |
- Final training data: 56,049 original + 464,639 augmented rows; 5 seeds (42-46); model file 19 MB. Runtime about 6 minutes.
- `submission.csv` matches the independent scratch pipeline (max diff 5e-12). Mean |diff| vs p1 = 0.147 x scale; overall level 1.078 x p1.

### Expectation
- Train-based validation improved by 0.035-0.047 on every scheme (cv 0.3893 -> 0.3419).
- The LB/private effect cannot be measured before submitting. If the train gain transferred fully, the public LB would land around 0.43-0.44. Train CV misled the team twice before, which is why the seasonal groups are anchored to LB evidence.

## 2026-10-08 - main.ipynb round 3 (level calibration per season x horizon) = FINAL SUBMISSION

### Trigger: Leon's newer log (leon/logs.md, entries up to 2026-10-08)
- The team's public LB is now 0.43729 (Leon), not 0.47566. Facts from that log that change my postprocessing:
  - Lebaran rows x3.0 of the raw model beat x2.0 (0.45395 -> 0.44518); all other rows x1.2 helped (-> 0.44318).
  - Leveling another model to the same mean prediction per (season group x horizon) cell, then averaging, gave 0.43729: rankings can be swapped while the LB-tested levels are kept.
- My round-2 Lebaran choice (p1 base x 2.4, mean ratio 1.277) is far below the LB-tested Lebaran level (1.876), so the pseudo-truth calibration underestimated it. Round-2 postprocessing is dropped.

### Method
- Season groups (same definitions as Leon): ram_to_lebaran (D1-D3 touches Ramadan 18 Feb - 19 Mar 2026, target >= 20 Mar), ram_to_ram, pre_to_ram, xmas (target 20 Dec - 4 Jan), normal. Sizes 4,598 / 10,767 / 1,962 / 5,235 / 50,049, identical to Leon's.
- Target level per cell = mean(prediction / scale) of `leon/submission_hedge_blendlev75_best25.csv`. That file is built so that every cell has the same mean as the 0.44318 file (and as the 0.43729 file): Lebaran 1.876, other rows 0.528, checked.
- The 34 cell targets are hard-coded in main.ipynb (`LEVEL_TARGETS`), so the notebook reproduces the submission from the raw data only. No CSV from another notebook is read.
- Final ratio = model ratio x (cell target / model's cell mean). Within each cell, the ranking of pairs is the final model's.
- Factors for my model: normal 1.25 / 1.20 / 1.21 / 1.20 at D4-D7, about 1.0 at D8-D10; xmas about 1.6; ram_to_ram 1.5 at D4 down to 0.51 at D10; pre_to_ram 0.7-1.2; Lebaran 4.3-7.7.
- Correlation of the leveled ratios with the hedge75 file 0.888, mean |diff| 0.19 x scale, so it is a different ranking at the same levels.
- Blending the final model with the p1 base model hurts OOF (0.3419 -> 0.3437 at weight 0.1, 0.3583 at 0.5), so the final model is used alone.

### Rule audit (my notebook)
| Rule | Status |
|---|---|
| External data public and published <= 2025-09-30 | cuti bersama SKB 2025 (Oct 2024, + 18 Aug SKB Aug 2025), SKB 2026 (signed and reported 19 Sep 2025), DKI school calendars (2024, Jul 2025), Ramadan dates from KHGT Muhammadiyah (public 25 Jun 2025). Dates only, sources in the notebook |
| No information about test films' own run | none used (no Cinepoint test-period data, no box office of test films) |
| No AutoML, no LLM inference | LightGBM only, fixed hyperparameters |
| Seeds | SEED = 42, models 42-46, deterministic=True; two full runs gave an identical submission.csv |
| First cell pip install with versions, -q | yes |
| Model file <= 200 MB | 19 MB |
| LB-derived constants | level targets come from the team's own public-LB submissions (normal leaderboard probing, not test labels) and are documented in the notebook |

### Rule warnings for the team (from leon/)
- `external/cinepoint/cinepoint_daily_top.csv` (Oct 2025 - Mar 2026) holds real admissions of the test films, published after the cutoff. Any file built from it is a rule violation (sections 5.2 and 7.4): `submission_filmadj_*.csv`, `submission_new_lgbm_cinepoint.csv` and `main_new.ipynb`. Leon labeled them "trial / assumption"; they must never be submitted or selected as final.
- The train-period Cinepoint scrape (Apr - Sep 2025) was taken from a live site in Oct 2026; its pre-cutoff version cannot be proven, so it is risky too.
- The 0.43729 chain (Kumo on GPU + CSV leveling scripts that were deleted) has to be rebuilt by one notebook if it is selected as the final submission, otherwise reproduction by the committee can fail.

## 2026-10-08 - LB result of round 3, and the search for a structural gain

### LB
| File | Public LB |
|---|---|
| team best (Leon, 0.5 leveled Kumo blend + 0.5 file 0.44318) | 0.43729 |
| file 0.44318 (Leon; same per-cell levels as below) | 0.44318 |
| **fayadh/submission.csv** (my final model, leveled to the same per-cell levels) | **0.45688** |
- The levels are identical to the 0.44318 file, so the +0.0137 is pure ranking: my model ranks pairs worse on the test period than Leon's default blend, although its train CV is far better (0.342 vs about 0.37-0.38). This is the sixth time train CV pointed the wrong way.

### Checks done afterwards (no new LB)
- D1 alignment train vs test: fine (cinemas D1/D3 median 1.01 test vs 1.05 train; national D1/D2 1.25 vs 1.22).
- The 0.44318 / 0.43729 files cannot be rebuilt: `leon/submission.csv` was overwritten (sha 826d... instead of 816b...), and git history does not contain them.
- Cinema programming signal ("squeeze": shows that new releases get at cinema c on date X relative to expectation, capacity share, film-level weighted contrast), built the same way from train and test_history: no gain (avg of 5 validations 0.3845 -> 0.3851-0.3858).
- Late-start pairs: pattern 001 (sales only on D3) keeps selling in 82% of D4 rows at a median ratio of about 3 (the scale is t3 / 3), the model predicts 0.71 on average vs 2.92 actual (error 2.64 per row, 8x a normal pair). Reparametrised target y / b with b = mean tickets on active D1-D3 days and weight b / scale (same L1 objective): small gain only (001 error 2.64 -> 2.49; avg 0.3845 -> 0.3833 with market features). The pattern is bimodal (18% stop, 82% continue), so most of that error is irreducible.

### Why scores below 0.40 are probably not reachable with allowed data
- Leon's Cinepoint run (real daily admissions of the test films, published after the cutoff, forbidden): CV 0.367 -> 0.305 (-0.062). Applied to our LB of about 0.437 that lands around 0.37-0.38, exactly the cluster of scores at 0.371-0.380 on the public LB. Leon's oracle study also showed the remaining error is mostly film-level (which films hold up), and nothing in the allowed data predicts it (best |rho| 0.14).
- So the teams below 0.40 very likely use post-cutoff box office data of the test films. That breaks rule 5.2 and they risk disqualification at verification; we should not follow.

### Probes prepared (leveled to the same per-cell levels as the 0.45688 file, so only the ranking differs)
| File | Ranking from | Question |
|---|---|---|
| `probe_A_p1_ranking_leveled.csv` | p1 base model (Leon's baseline features, 150 trees) | is a simple model's ranking better than mine on the test period? |
| `probe_B_no_calendar_leveled.csv` | my final model without offday / cuti / school features | do the D1-D3 holiday features hurt the test ranking (Leon saw this on 0.48785)? |
| `probe_C_no_aug_leveled.csv` | my final model without augmentation, 150 trees | does the shifted-window augmentation hurt the test ranking? |
