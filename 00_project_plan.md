# 00 — Project Plan: Fraud Detection in Electricity and Gas Consumption

Written by Sean, with research assistance from Claude, for the joint course project with Roy.
Originally drafted against a separate `sean-eda`-branch version of `01_eda.ipynb`; that
notebook's EDA findings have since been folded into this branch's `01_eda.ipynb` (Roy's
structure — numbered "Step 1–7" sections rather than this doc's "§" section numbers). Where a
citation below says "`01_eda.ipynb` §2.5" or similar, the equivalent content in the current
notebook is Step 6 (meter-reading reconciliation); "§2.6" is Step 7 (leakage audit); the
`client_catg`/`region`/`disrict` segment breakdown is in Step 3. The data dictionary, Kaggle/
Zindi research, and open questions below are otherwise unaffected by that reorganization.

This document answers the two open questions from our kickoff discussion, summarizes what we
learned from public notebooks on this dataset, and lays out an end-to-end plan. The EDA section
is deliberately the most detailed part — it is the highest-leverage phase for this dataset. The
later phases are sketched at a level that should be enough to start, and are expected to be
refined once EDA findings are in.

The dataset is a Kaggle mirror of a Zindi Africa competition run by **STEG** (Société Tunisienne
de l'Électricité et du Gaz), the Tunisian national electricity and gas utility. STEG reported
losses on the order of 200 million Tunisian Dinars from meter tampering and other consumption
fraud, and released ~15 years (2005–2019) of client and billing records to build a model that
ranks clients for investigation.

Sources consulted: the [Kaggle dataset page](https://www.kaggle.com/datasets/mrmorj/fraud-detection-in-electricity-and-gas-consumption),
its [notebooks tab](https://www.kaggle.com/datasets/mrmorj/fraud-detection-in-electricity-and-gas-consumption/code),
[Kevanoo's "Predicting Utility Fraud Cases"](https://www.kaggle.com/code/kevanoo/predicting-utility-fraud-cases-machine-learning) notebook
(read cell-by-cell via the Kaggle API — see §4.1),
the original [Zindi competition data page](https://zindi.africa/competitions/fraud-detection-in-electricity-and-gas-consumption-challenge/data)
(the authoritative data dictionary — Kaggle only mirrors this), and several other public
notebooks/repos for the same dataset (linked inline below).

**Note on how this research was done:** Kaggle's notebook pages are JavaScript-rendered — a plain
fetch returns an empty page shell with no cells, code, or output, confirmed with a raw `curl`.
Once a Kaggle API token was set up, four notebooks were pulled and read directly, cell by cell,
via `kaggle kernels pull`: Kevanoo's (§4.1), the two most-upvoted kernels on the dataset —
[imgremlin's 4th-place Zindi solution](https://www.kaggle.com/code/imgremlin/4th-place-in-fraud-detection-from-zindi)
and [fouedayedi's EDA and modeling notebook](https://www.kaggle.com/code/fouedayedi/eda-and-modeling-fraud-detection-in-elec)
(§4.4, §4.5) — and the CallmeMehdi starter notebook (§4.2, pulled as raw JSON from GitHub instead).
The one exception is **advalhakim's EDA kernel**, referenced only in passing below — it wasn't
pulled, so treat that one mention as lower-confidence than everything else in this document.

---

## 1. Answers to the two open questions

### 1.1 What do the columns mean (especially in the invoice data)?

Verified directly against `client_train.parquet` / `client_test.parquet` / `invoice_train.parquet`
in `data/parquet/`, and cross-checked against the [Zindi data dictionary](https://zindi.africa/competitions/fraud-detection-in-electricity-and-gas-consumption-challenge/data).

**Client table** (`client_train` / `client_test`, one row per client):

| Column | Meaning | Notes from the actual data |
|---|---|---|
| `client_id` | Unique client identifier | Key only — never a model feature |
| `disrict` *(sic)* | District code | 4 values in-sample (60, 62, 63, 69); misspelling preserved from source |
| `client_catg` | Consumer category code | Dominated by `11` (~97%); `12` and `51` are small minority categories |
| `region` | Region code (coarser geography than district) | ~15+ values, uneven sizes (101 alone ≈ 25% of clients) |
| `creation_date` | Date the client account was created | Ranges 1977–2019 in `client_train` |
| `target` | **Fraud label: 1 = fraud, 0 = not fraud** | Present only in `client_train`; fraud rate ≈ **5.58%** |

**Invoice table** (`invoice_train` / `invoice_test`, many rows per client — one per billing event):

| Column | Meaning | Notes from the actual data |
|---|---|---|
| `client_id` | Foreign key to the client table | Median ≈ 30 invoices/client, mean ≈ 33, max 439 — history length varies a lot |
| `invoice_date` | Date of the billing/reading event | Ranges 1978–2019 |
| `tarif_type` | Tariff/rate-plan code applied to this invoice | `11` and `40` dominate; several rare codes (8, 9, 12–15, 29, 30, 45) |
| `counter_number` | Physical meter identifier | Not a feature — occasionally useful to detect meter swaps |
| `counter_statue` *(sic)* | Meter status code recorded at the visit | Almost all rows are `0`; codes 1–5 look like anomaly/condition flags. The 4th-place Zindi solution (§4.4) had to clean values like `769`, `618`, `269375`, `46`, `420`, and `'A'` — confirmed present in our own full file, but **only 47 of 4.47M rows (0.001%)**, per the executed EDA in `01_eda.ipynb` §4. A tiny, easy cleaning step, not a major blocker — **investigate in EDA**, see §2.6 |
| `counter_code` | Meter type/model code | Wide spread of codes (203, 5, 207, 413, 202, …) |
| `reading_remarque` | Numeric code for the note a **STEG agent writes during their site visit** (e.g. "meter looks tampered") | Only 4 values seen (6, 7, 8, 9); recorded by a human at inspection time — **possible leakage source**, see §2.6 |
| `counter_coefficient` | Multiplier applied once consumption exceeds a standard threshold | Overwhelmingly `1`; rare non-1 values |
| `consommation_level_1..4` | Consumption split across 4 progressive tariff tiers (like a tiered pricing ladder) | `level_1` carries most volume; `level_2–4` are near-zero most of the time but very heavy-tailed when non-zero (max seen: 93k / 210k / 45k / 344k in a 500k-row sample) |
| `old_index` / `new_index` | Meter reading at the start / end of the billing period | `new_index − old_index` should roughly reconcile with total consumption — mismatches are a strong data-quality/fraud signal, see §2.6 |
| `months_number` | Number of months the invoice covers | Usually 1, 4, 6, or 12 — but has extreme outliers (one sampled batch had a max of ~231,602 months), clearly bad data that needs capping/investigation |
| `counter_type` | Meter/utility type | Only `ELEC` or `GAZ` — some clients have both |

Two spellings are preserved on purpose because they match the original source schema:
`disrict` and `counter_statue`. Keep them as-is rather than "fixing" the name, or joins/merges
against fresh Kaggle downloads will silently break.

### 1.2 The test set has no `target` — what's going on, and does the fifth file solve it?

This is a **standard supervised-learning competition split**, not a data bug:

- `client_train.csv` / `invoice_train.csv` (**labeled**) — the only place `target` exists. This is
  the only data we can use to both train and honestly evaluate a model.
- `client_test.csv` / `invoice_test.csv` (**unlabeled**) — same schema minus `target`. This is the
  Kaggle/Zindi competition's hold-out set; the label exists only on Zindi's servers, so **we
  cannot score against it locally.**
- **The fifth file, `SampleSubmission.csv`, does *not* contain the missing labels.** It's a
  *submission-format template*: one row per test `client_id` with a placeholder `target` column
  showing the two columns and their names/order a submission must have. Every public kernel that
  references it uses it purely to shape its final output file, never as a source of ground truth.
  Confirmed against the [Zindi data page](https://zindi.africa/competitions/fraud-detection-in-electricity-and-gas-consumption-challenge/data)
  and multiple kernels (e.g. the [CallmeMehdi starter notebook](https://github.com/CallmeMehdi/Fraud-Detection-in-Electricity-and-Gas-Consumption-Challenge/blob/master/Starter_notebook.ipynb),
  which merges predictions into it as the very last step).

**Implication for us — since this is a course project, not an active leaderboard we can submit
to:** the Kaggle/Zindi "test" set is not usable for evaluating logistic regression / random
forest / anything else we build, because we would have no way to check whether our predictions
were right. **All model development and evaluation has to happen on `client_train`**, split
ourselves into a training/validation portion and a held-out portion — which is exactly what
`01_eda.ipynb` already does (`development` vs. `final_holdout`, stratified by `target`, split at
the **client** level so no client's invoices leak across the split). That existing split is the
correct answer to "where does our target come from" — keep it.

The Kaggle `client_test`/`invoice_test`/`SampleSubmission.csv` files are still useful, just for a
narrower purpose: (a) sanity-checking that our feature pipeline runs on genuinely unseen clients
without errors, and (b) optionally producing a Kaggle-format submission file as a nice demo
artifact for the presentation. They are **not** part of our validation strategy and shouldn't be
used to claim any accuracy/precision number — we have no labels to check them against.

---

## 2. Detailed EDA plan

This is the most important phase, and the one we should spend the most joint time on before
touching a model. Each subsection below is a concrete set of things to compute/plot, not just a
topic. `01_eda.ipynb` already implements a first pass of several of these (schema checks, target
balance, client-level aggregation, some segment fraud rates) — treat this list as the fuller
checklist to work through and extend that notebook against.

### 2.1 Schema and inventory sanity
- Row/column counts, dtypes, and memory footprint for all 4 tables (done in `01_eda.ipynb`).
- Uniqueness of `client_id` in the client tables; cardinality of every categorical column.
- Confirm `client_test` truly has no `target` column (it doesn't — verified above) and that
  `client_train`/`client_test` share identical dtypes for every other column.

### 2.2 Target and class balance
- Fraud rate on `client_train` (**5.58%** confirmed) — establish this number once and reuse it
  everywhere else (e.g. as the naive baseline for a dummy classifier).
- Fraud rate broken out by `client_catg`, `region`, `disrict`, with a minimum-sample-size guard
  (already implemented as `fraud_rate_by_group` in `01_eda.ipynb`) — small categories will show
  noisy rates and shouldn't be over-interpreted on sample size alone, but check before discounting
  one: **`client_catg == 51` is a verified, high-priority lead, not noise** — 1,678 clients
  (large enough to trust) at a **16.9% fraud rate**, roughly 3× the overall 5.58% base rate. This
  matches (and confirms) what fouedayedi's public notebook reports (§4.5). `disrict` fraud rates
  also verified directly: 60→3.6%, 62→5.2%, 63→6.5%, 69→7.1% (only 4 district codes exist in the
  data — fouedayedi's notebook mislabels the highest one "district 29"; it's actually `69`, a good
  reminder to verify a public notebook's prose against our own data rather than quoting it as-is).

### 2.3 Observation-window / history-length bias (do this early, it can invalidate everything else)
- Distribution of invoices-per-client and active-history span (`last_invoice − first_invoice`),
  split by target. Median is 30 invoices/client but the range is wide (1–439).
- Explicitly test whether `target` correlates with *how long a client has been observed* rather
  than with genuine behavior — e.g. compare fraud rate across quintiles of invoice count/tenure
  (`01_eda.ipynb` already bins this). If longer-observed clients show much higher fraud rates, a
  naive model will partly learn "has been a customer a long time" instead of "is committing
  fraud," which won't generalize to newer clients.
- Check whether `creation_date` ever falls *after* `first_invoice` (data-entry inconsistency) and
  whether `target` correlates with customer tenure at time of first/last invoice.

### 2.4 Consumption behavior
- `consommation_level_1..4`: distribution per tier (heavy right skew — use `log1p` for plots, not
  for the raw stats), share of invoices with zero total consumption, and whether tier-2/3/4 usage
  (rare but very large when present) differs by target — tier overflows might indicate unusually
  high, undeclared, or irregular usage.
- Total/mean/max consumption per client by target, on log scale (`01_eda.ipynb` already does a
  first pass of this).
- `counter_type` mix (`ELEC` vs `GAZ`, and clients with both) vs. fraud rate — fraud mechanics
  likely differ between electricity meters and gas meters and may deserve separate models later.
- Zero-consumption runs: clients with long streaks of zero consumption followed by a resumption
  are a classic tampering pattern worth a dedicated feature, not just an overall rate.

### 2.5 Meter-reading integrity checks (this dataset's real fraud signal likely lives here)
- **Reconcile `new_index − old_index` against `consommation_level_1..4` summed.** These should
  roughly agree; systematic mismatches (index barely moves while consumption is reported as high,
  or vice versa) are a strong, physically-motivated fraud signal — arguably more meaningful than
  any category code.
- Frequency and magnitude of negative `index_delta` (meter went backwards) — could be genuine
  meter replacement/reset, or tampering. `01_eda.ipynb` already counts these; extend to check
  whether they correlate with target and whether they cluster right before/after an
  investigation.
- `months_number` outliers (values in the tens of thousands seen in-sample) — these are clearly
  corrupted or mis-recorded and need capping/flagging before they're allowed anywhere near a
  mean/sum aggregation, since a single such row would dominate a client-level sum.
- Rare `counter_statue` codes (1–5, vs. the dominant `0`) and rare `counter_coefficient` values
  (anything ≠ 1) — check fraud rate conditional on ever having a non-default code, not just on the
  most recent one.

### 2.6 Leakage audit — specific to this dataset, and not something the public kernels we found discuss
- `reading_remarque` is explicitly described (Zindi data dictionary) as **a note the STEG agent
  writes during a physical site visit**. If agents visit *because* fraud is suspected, or write a
  distinctive remark *as a result of* what they find, this column (and possibly `counter_statue`)
  risks encoding the outcome of the investigation itself rather than an independent predictor
  available *before* a decision to investigate.
- Concretely check: does the distribution of `reading_remarque`/`counter_statue` values differ
  between a client's *last* invoice vs. earlier invoices, and does that difference line up with
  `target`? If suspicious codes cluster right at the end of fraudulent clients' histories, treating
  the full-history aggregate as a predictor is likely leaking the label backward in time.
- Decide, and write down the decision: keep these fields but only from *early* invoices, drop them
  entirely, or keep them as-is with an explicit caveat that the model is partly learning from
  investigation outcomes. Whatever we choose, say so explicitly in the slides — this is exactly
  the kind of thing an instructor will ask about.
- More generally: for every candidate feature, ask "would this value have been known before a
  human already looked into this specific client?" `client_id`, `counter_number`, and any
  post-hoc administrative code deserve this question.

### 2.7 Data quality audit
- Duplicate rows (`01_eda.ipynb` already hashes rows per chunk to catch exact duplicates without
  holding the full 4.5M-row table in memory) and duplicate `client_id`s in the client tables.
- Missing values per column, per table.
- `invoice_train` clients missing from `client_train` and vice versa (should be zero/near-zero;
  confirm and explain any exceptions).
- Internal consistency: `creation_date` after first invoice, negative tenure, and the index/
  consumption/months_number anomalies from §2.5.

### 2.8 Segment robustness (first pass now, full pass during error analysis)
- Fraud rate and, once a model exists, model performance broken out by region, district,
  category, counter type, and history-length bucket. A single global PR-AUC can hide a model that
  is useless (or unfair) for a specific segment — flag this early so it isn't a surprise on Day 3.

### 2.9 Suggested plot/table set to actually produce
- Class balance bar chart (have it).
- Fraud rate by category/region/district bar charts with sample-size annotations.
- Log-scale boxplots/violins of consumption and index-delta by target (have a first version).
- Invoice-count / active-days histograms by target, and the quintile-binned fraud-rate bar chart
  (have it).
- Scatter of `index_delta` vs. total consumption, colored by `counter_statue`/target, to visually
  spot the reconciliation mismatches from §2.5.
- A simple time trend of aggregate consumption/invoice volume over calendar years, to catch
  cohort effects (e.g. did billing practices change over the 1978–2019 span in a way that would
  confound a naive "recent vs. historical" feature).

### 2.10 EDA deliverable
A short written findings memo (the "Record findings before modeling" section already scaffolded
in `01_eda.ipynb`) answering: which anomalies are real fraud signal vs. data artifacts, which
features look predictive vs. leaky, whether history-length bias is a real risk, and what the
segment picture looks like — before any model is trained on more than a dummy baseline.

### 2.11 Status: this EDA has now been executed — headline real results

Everything in §2.1–2.9 above has been implemented and run end-to-end in `01_eda.ipynb` on the real
full 4.47M-row training file (not a sample). Full write-up is in that notebook's §7 findings
section; the results that most change the picture from this plan's original assumptions:

- **Observation-window bias is real and severe**: fraud rate is **9.6×** higher in the
  longest-history quintile (9.0%) than the shortest (0.9%) — the single biggest risk to address
  before modeling, exactly as §2.3 anticipated, but bigger than expected.
- **A previously-unknown 2005 cohort effect**: invoice record-keeping only ramps up to full scale
  in 2005 (2,159 invoices in 2004 → 64,663 in 2005 → 200k–410k/year after); pre-2005 history is a
  thin trickle relative to `creation_date`'s 1977 start. Not anticipated anywhere above — factor
  this into any "recent vs. historical" feature.
- **The index-reconciliation check (§2.5) found a real signal**: mean absolute
  index/consumption mismatch is 76.5 for non-fraud clients vs. 126.2 for fraud clients — worth
  carrying into the baseline model, though it's driven by a subset of large mismatches, not a
  uniform shift (median is 0 for both groups).
- **`counter_statue` cleaning turned out to be a non-issue in volume** (47 of 4.47M rows — see the
  correction in §1.1) — worth doing, but not worth over-budgeting time for.
- **`months_number` needed a threshold fix mid-analysis**: a first attempt flagged 7% of rows as
  outliers using too narrow a "valid" set; the real distribution shows several more legitimate
  multi-period billing values. A physically-implausible threshold (>36 months) is the defensible
  version and only flags ~0.04% of rows — a good example of checking a first heuristic against the
  full data rather than trusting it.
- **The leakage audit (§2.6) came back genuinely two-directional, not a single smoking gun**:
  `reading_remarque` shifts toward non-modal codes at a client's last invoice for *everyone*
  (49%→71% non-fraud, 51%→79% fraud), while `counter_statue` shifts the *opposite* direction for
  everyone too. Something changes for all clients near their last invoice, somewhat more so for
  fraud clients — real caution for feature design, not proof of a single mechanism.

---

## 3. The rest of the plan (sketched)

Sketched at the level of "what to do and why," informed by the pattern across public kernels for
this dataset (mean/count aggregation → label encoding → LightGBM/XGBoost, generally *without*
addressing class imbalance, leakage, or a defensible validation split — see §4). Refine each of
these once §2's findings are in.

### 3.1 Feature engineering
- One row per `client_id`, built only from `client_train`/`client_test` + aggregated invoices —
  never invoice rows directly.
- Per-client aggregates beyond mean/count (which is as far as most public kernels go): sum, mean,
  std, min/max, zero-rate, and quantiles of consumption and index deltas; counts of distinct
  `tarif_type`/`counter_type`/`counter_code`/`counter_statue` values seen; a flag for any
  non-default `counter_statue`/`counter_coefficient` ever observed; invoice count and active-days
  span; customer tenure at last invoice.
- Recency features: split each client's history (e.g. last 6–12 months vs. earlier) and compute
  ratios/deltas — sudden drops or spikes are more informative than lifetime averages.
- Treat `region`/`district`/`client_catg` as categorical (one-hot or target-safe encoding fit on
  training folds only); never use `client_id` or `counter_number` as a predictor.
- Apply the leakage decision from §2.6 consistently to train and Kaggle-test feature building.
- Borrowed concretely from imgremlin's 4th-place solution (§4.4), which is worth reproducing:
  `coop_time` (months of tenure as of a fixed reference date — a cleaner version of our own
  tenure feature), `delta_time` (days between consecutive invoices per client — directly captures
  billing-cycle irregularity), and `region_group` (bucketing `region` into low/mid/high numeric
  ranges on the hypothesis that the codes aren't randomly assigned). Also their systematic
  `_range` (`max − min`) and `_max_mean` (`max / mean`) pass over every aggregated column — cheap
  to generate broadly and let feature selection/importance decide what survives.
- Feature pruning: either a Pearson-correlation threshold (Kevanoo's approach, §4.1 — drop one of
  any pair with |r| ≥ 0.90) or backward selection informed by feature importance (imgremlin's
  approach, §4.4). Worth trying the cheap correlation-threshold pass first, since it needs no
  model refit loop.

### 3.2 Validation strategy
- Reuse the existing client-level `development` / `final_holdout` split of `client_train` (already
  in `01_eda.ipynb`) — this *is* our answer to "where's the target," per §1.2.
- Inside `development`, use stratified k-fold (grouped at client level) for model comparison and
  any hyperparameter tuning; touch `final_holdout` exactly once, at the end.

### 3.3 Modeling
- Dummy baseline (predict the training prior, ≈5.58%) to anchor every later metric.
- Logistic regression next (scaled features, `class_weight="balanced"` or manual threshold
  tuning) — simple, fast, interpretable coefficients that are useful for the slides.
- Random forest / gradient boosting (scikit-learn's `HistGradientBoostingClassifier`, or add
  LightGBM/XGBoost as an optional dependency if we want what most public kernels use) with
  imbalance handling via `class_weight`/`scale_pos_weight`; `imbalanced-learn` is already a
  project dependency if we want to compare resampling (SMOTE/undersampling) against class
  weighting rather than defaulting to resampling as most kernels implicitly assume.
- Worth knowing before over-investing in resampling: the actual **4th-place Zindi solution used
  no SMOTE/undersampling at all** (§4.4) — its lift came from rich per-client feature engineering
  plus a properly cross-validated, early-stopped LightGBM tuned directly against its evaluation
  metric. Class-weighting/resampling is one lever, not necessarily the highest-leverage one;
  feature richness and correct CV are worth trying first, and comparing against a resampled
  version rather than assuming resampling is required.
- Compare models on the metrics in §3.4, not accuracy.

### 3.4 Evaluation
- Primary: PR-AUC (average precision), because of the ~5.58% imbalance.
- Operational: precision@k / recall@k at a plausible inspection capacity `k` (ask the instructor
  or pick a few illustrative `k`s — e.g. top 1%/5%/10% of clients by score — since no real
  operational capacity number exists for a course project).
- Secondary: ROC-AUC, confusion matrix at the chosen threshold.
- Report all of the above on `development` folds; run once on `final_holdout` right before the
  deadline.

### 3.5 Error analysis
- False positives/negatives by region, category, counter type, tenure/history-length bucket
  (extends §2.8 once a model exists).
- Feature-importance / coefficient review, plus an explicit ablation with vs. without the §2.6
  leakage-suspect features, so we can honestly state how much of the model's lift depends on them.
- Inspect a handful of specific misclassified clients' histories qualitatively — good material for
  the slides.

### 3.6 Optional: Kaggle-format submission (demo only, not a graded metric)
- Run the frozen pipeline on `client_test`/`invoice_test`, format the output to match
  `SampleSubmission.csv` (`client_id`, `target` probability), purely as a demonstration of what a
  "deployed" scoring pipeline would produce. We cannot self-evaluate this file's accuracy (§1.2).

### 3.7 Deliverables and timeline
Follows the milestones already agreed in `README.md` (Day 2 baseline, Day 3 slides + error
analysis, Day 4 final model + presentation) — no change proposed here, just confirming this plan
feeds directly into that schedule.

---

## 4. What we're taking from public kernels, and what we're doing differently

### 4.1 Kevanoo's "Predicting Utility Fraud Cases" — read directly, cell by cell

This is the notebook Sean flagged as a good reference, and it's a genuine step up from the other
kernels below — it's the only one of the bunch that engages with the imbalance problem at all.
What it actually does:

- **EDA:** computes the fraud/non-fraud ratio (0.0591 — consistent with our verified 5.58% fraud
  rate: `0.0591 ≈ 0.0558 / (1 − 0.0558)`), confirms no missing values, aggregates
  `consommation_level_1..4` **and `reading_remarque`** per client (mean/std/min/max) and compares
  the group means between fraud and non-fraud clients with bar charts, and plots fraud percentage
  by `region` and by `disrict`.
- **Feature engineering:** drops several raw columns up front (`counter_code`, `old_index`,
  `new_index`, `months_number`, `counter_statue`, `tarif_type`, `invoice_date`, `counter_type`),
  aggregates the rest per client, then **prunes features with a Pearson correlation matrix**
  (drops one of any pair with |r| ≥ 0.90) — a concrete technique worth borrowing for our own
  feature set in §3.1.
- **Class imbalance:** explicitly names the skew as a problem and applies **SMOTE to the training
  split only** (test set left in its natural distribution) — correct practice, and further than
  either of the two GitHub-hosted kernels below get.
- **Modeling:** three models — Logistic Regression, Random Forest, kNN — each wrapped in a
  `Pipeline` with `StandardScaler` + `PCA(n_components=0.95)`, tuned via `GridSearchCV`/
  `RandomizedSearchCV` optimizing ROC-AUC. Random Forest also gets a feature-importance-based
  pruning pass before its own hyperparameter search.
- **Reported results (test split):** Logistic Regression — accuracy 0.610, recall 0.699→0.684,
  ROC-AUC 0.652→0.656 after tuning. Random Forest — accuracy 0.915→0.862, recall
  **0.112→0.237**, ROC-AUC 0.536→0.698 after tuning. kNN — accuracy 0.755→0.740, recall
  0.379→0.465, ROC-AUC 0.577→0.610 after tuning.
- **Conclusion:** frames the LR-vs-RF choice as a business tradeoff (LR catches more fraud but
  flags more clients; RF is more "accurate" but misses most fraud), and ultimately calls Random
  Forest "superior" because of its higher ROC-AUC — while its own numbers show RF's recall
  (0.237) is the worst of the three models.

**What we're borrowing:** the correlation-based feature pruning (§3.1), and the general
three-model comparison structure (§3.3) — logistic regression against a tree ensemble is exactly
what we already planned.

**What we're doing differently, and why:**
- **PR-AUC over ROC-AUC as the headline metric.** The notebook itself argues that both false
  positives and false negatives matter, which is precisely the argument *for* average precision
  under ~5.6% prevalence — ROC-AUC can look reasonable (0.698) while the same model's recall is
  only 0.237, because ROC-AUC is computed against the *rate* of negatives and is far less sensitive
  to this imbalance than precision-based metrics are. This is a real, general pitfall in these
  results, not a nitpick, and it's exactly why `README.md` already fixes on PR-AUC + precision@k/
  recall@k instead. Reading this notebook is a good, concrete justification to put in the slides
  for *why* we chose PR-AUC over accuracy/ROC-AUC.
- **Threshold as a business decision, not `predict()`'s default 0.5.** Random Forest's recall
  collapsing to 0.112–0.237 despite SMOTE-balanced training is a classic symptom of scoring with
  the default 0.5 cutoff on the original-distribution test set. Tuning the decision threshold (or
  reporting precision@k/recall@k across a few `k`) would very likely change which model "wins" —
  worth demonstrating explicitly if we reproduce a similar comparison.
- **Stratified split.** `train_test_split` here has no `stratify=y` — at 135k rows this is
  probably fine in practice, but our own split already stratifies (and does so at the client
  level for the reasons in §1.2/§3.2), so let's keep that.
- **The `reading_remarque` leakage question is raised but not resolved.** The notebook checks
  whether `reading_remarque_mean` differs by target (it says "minimal differences" and states it
  will drop the variable) — but then keeps `reading_remarque_std/min/max` in the actual feature
  set used for modeling. It never asks *why* an agent's site-visit note might correlate with fraud
  in the first place, i.e. whether it's a symptom of the same investigation that produced the
  label. That's exactly the audit in §2.6, and it's still an open, unanswered question after
  reading this notebook — not something we can consider solved by it.

### 4.2 The rest: CallmeMehdi's starter notebook and BHafsa/STEG-Fraud-Detection

The [CallmeMehdi starter notebook](https://github.com/CallmeMehdi/Fraud-Detection-in-Electricity-and-Gas-Consumption-Challenge/blob/master/Starter_notebook.ipynb)
was also read directly (pulled as raw JSON from GitHub, not summarized). It aggregates only the
*mean* of `consommation_level_1..4` plus a transaction count, label-encodes `disrict`/`client_catg`,
and fits a single default-hyperparameter LightGBM **on 100% of `client_train` with no train/
validation split at all** — it even imports `StratifiedKFold` and never uses it — then writes a
Kaggle-format submission with no metric evaluated anywhere in the notebook. `BHafsa/STEG-Fraud-Detection`
(read as a README summary only, not the notebook itself) uses XGBoost with "hyper-parameters ...
picked randomly" by the author's own admission, and its README explicitly flags the absence of
precision/recall-style metrics as a gap.

Neither of these two engages with class imbalance, a real validation split, or the leakage
question at all — that part of the earlier draft of this section still holds. Kevanoo's notebook
is the outlier in a good way, which is presumably why Sean picked it out.

### 4.4 imgremlin's 4th-place Zindi solution — the most upvoted kernel, and a genuinely different tier

This one is not a course-project-style walkthrough — it's team GORNYAKI's (Ukraine, KPI/IASA)
actual 4th-place entry on the original Zindi leaderboard, read directly cell by cell. It's the
strongest engineering in anything we looked at, and the one most worth learning from:

- **Feature engineering, not resampling, did the work.** `coop_time` (tenure in months as of
  2019), `delta_time` (days between a client's consecutive invoices), `region_group` (bucketing
  `region` into <100 / 100–300 / >300 — an explicit bet that the region codes carry ordinal
  structure), a cleaned `counter_statue` (see the corrected data-dictionary note in §1.1), and
  extracted `invoice_month`/`invoice_year`/`is_weekday`. Every one of ~16 columns then gets
  mean/std/min/max aggregated per client, plus a `_range` (max−min) and `_max_mean` (max/mean)
  pass over all of them — a large, systematic feature explosion followed by pruning.
- **No SMOTE, no class-weighting, no resampling anywhere in the notebook.** The imbalance is
  handled implicitly by using a rank-based metric (AUC) and letting LightGBM + rich features do
  the separating — a real data point against reflexively reaching for resampling as step one.
- **Rigorous tuning:** Optuna-driven Bayesian hyperparameter search over LightGBM, evaluated with
  5-fold `StratifiedKFold` and early stopping, explicitly checked that the local CV score "almost
  matched" the leaderboard score — i.e. they verified their validation setup wasn't lying to them
  before trusting it. That's a good habit to copy regardless of model choice.
- **Important framing difference for us:** every metric in this notebook is **ROC-AUC** — that
  was very likely the actual Zindi competition's scoring metric. We are deliberately choosing a
  different metric (PR-AUC + precision@k/recall@k, per `README.md`) because our framing is an
  *operational inspection tool with limited capacity*, not a leaderboard rank. Worth saying this
  explicitly in the slides: we're not trying to beat this notebook's AUC number, we're solving a
  differently-framed problem on the same data, and that's a deliberate choice, not an oversight.

### 4.5 fouedayedi's EDA and modeling notebook — mostly original EDA, borrowed modeling

Also read directly. Its modeling section turns out to be a credited, near-verbatim reuse of
imgremlin's feature engineering and LightGBM setup from §4.4 (the notebook says so itself in a
code comment) — so this isn't really a fifth independent approach, it's "one strong original
solution plus one EDA-heavy writeup built on top of it." The EDA half adds real value though:

- Confirms our own numbers independently: 68.8% ELEC / 31.2% GAZ by invoice count (our own sample
  came out 68.6%/31.4% — good cross-check), and a target-imbalance framing consistent with ours.
- **Flags `client_catg == 51` as the highest-fraud-rate category (reports "over 17%")** — verified
  directly against `client_train.parquet`: **16.9% on 1,678 clients**, a real, large-enough-to-trust
  finding, not noise (now folded into §2.2 above).
- Plots fraud rate directly against `counter_statue` and against `tarif_type` — the same kind of
  check §2.5/§2.6 proposes, though this notebook doesn't ask *why* those codes might correlate
  with the label (the temporal/leakage question), just *whether* they do.
- **Two things to actively avoid copying:** (1) it draws a causal claim ("the fraudulent customer
  is likely tampering with their meter") from a single fraud client vs. a single honest client's
  2005 consumption plot — an n=1-vs-n=1 anecdote, not evidence, and a good example of the kind of
  overclaim the "associations, not causes" caveat already in `01_eda.ipynb` is guarding against.
  (2) its district fraud-rate write-up labels the highest-fraud district "district 29" — there is
  no district 29; the real codes are only `{60, 62, 63, 69}` (verified directly), and the rate it
  quotes (7.1%) actually belongs to district `69`. A small slip, but exactly why every number that
  goes in our slides should be re-run against our own data rather than quoted from a public
  notebook, however well-regarded.

### 4.6 Net takeaway

This plan (and the existing `01_eda.ipynb`/`README.md`) already does better than the majority of
these on the validation-split and imbalance-metric fronts — imgremlin's solution (§4.4) is the one
kernel that's genuinely ahead of us on engineering rigor and is worth deliberately learning from,
not just critiquing. The concrete, still-unaddressed-by-any-kernel addition from this research
pass remains §2.5/§2.6 — the index-reconciliation check and the `reading_remarque`/`counter_statue`
leakage audit — plus a ready-made cautionary example from Kevanoo's notebook (RF's ROC-AUC vs.
recall split) and imgremlin's notebook (ROC-AUC was the actual leaderboard metric) for why we're
deliberately leading with PR-AUC and an operating-point-aware metric instead of matching what the
competition itself was scored on.

---

## 5. Open questions for the team / instructor

- Is there a realistic inspection capacity (`k`) to anchor precision@k/recall@k, or should we
  present a curve over several plausible `k` values?
- Do we want to spend time on the optional Kaggle-format submission (§3.6) for the presentation, or
  skip it since it can't be scored?
- Once §2.6's leakage audit is done: do we ship a model *with* `reading_remarque`/`counter_statue`
  (higher score, weaker story) or *without* them (lower score, cleaner story)? Worth deciding
  together rather than defaulting silently either way.
