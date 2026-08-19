# Slide blueprint

Keep one message per slide. Prefer one chart or table over many small outputs.

## Slide 1 — Title

- Predict which electricity and gas clients are most likely linked to fraud.
- Turn millions of invoices into a short, useful inspection list.
- Team: Roy and Sean.
- Link: Github Repo

**Visual:** title plus one meter/invoice image.

## Slide 2 — Business Case
Imagine that a utility company has a small team that can inspect only some clients.
We want to help them decide which clients to inspect first.

## Slide 3 — Value means finding fraud within a limited inspection budget

- Only **5.58%** of clients are labelled as fraud.
- Accuracy is misleading: predicting “not fraud” for everyone is about 94% accurate but finds no fraud.
- Main metric: **average precision**, which summarises the precision-recall curve. Operational metrics: **precision@k** and **recall@k**.

**Say:** “If the team can inspect `k` clients, how many real fraud cases are in that list?”

## Slide 4 — From invoices to one prediction per client

- 135,493 labelled clients and 4,476,749 invoice rows.
- One-to-Many: One client has many invoices; the target exists once per client.
- Streamed the 4.48 million invoice rows in batches to reduce peak memory use.
- Aggregated dates, consumption, meter readings, and billing behaviour into one feature row per client.

**Visual:** `client → many invoices → one client feature row → fraud score`.

**Course anchor:** Question → data → inspect/clean → explore → split → train → evaluate → explain.

## Slide 5 — What we actually split, and when

- The EDA uses **all 135,493 labelled clients** from `client_train`.
- Invoice rows are first aggregated to one row per client.
- WIP: `02_baseline_model.ipynb` then makes one **stratified 80/20 train/validation split of client rows**.
- WIP: Imputation, scaling, and one-hot encoding are fitted only on the 80% model-training rows through the model pipeline.
- Kaggle's separate, unlabeled `client_test`, `invoice_test`, `SampleSubmission.csv` data are not used in the current EDA or baseline evaluation.

**Say:** “Explore all labelled training data, create one row per client, then split clients for the baseline.”

**Limitation:** the validation rows do not fit the scaler, encoder, or model, but their labels were visible during EDA and influenced feature selection. The score is suitable for a first iteration, not a fully untouched estimate of future performance.

## Slide 6 — We turned raw invoices into behaviour features

The target is in `client_train`; the repeated behaviour is in `invoice_train`. Every created feature below is aggregated to **one row per client** before modelling.

| Feature(s), source, and status | **What** it measures | **Why** we created or prepared it | EDA evidence and effect on the data | How it helps / final decision |
|---|---|---|---|---|
| **Created from `invoice_train`:** `invoice_count`, `active_days` | Number of invoices and days from the first to last recorded invoice | Clients have very different amounts of recorded history. We need to represent exposure explicitly instead of letting every total silently stand for “more history.” | Fraud medians: **41 vs 29 invoices** and **4,727 vs 3,283 active days**. Fraud rate ranges from **0.94% to 9.15%** across invoice-count groups. | Both enter the baseline. They are useful candidate signals, but error analysis must test whether the model is mainly learning observation length. |
| **Created from `invoice_train`:** `mean_consumption`, `zero_consumption_rate`, `elec_share` | Average consumption per invoice, share of zero-consumption invoices, and share of electricity invoices | Raw sums and counts grow automatically when a client has more invoices. Rates and means make clients with different history lengths more comparable. | Median mean consumption is **466 for fraud vs 393 for non-fraud**. Converting counts to rates reduces, but does not completely remove, history-length effects. | All three enter the baseline to test typical usage, repeated zero readings, and electricity/gas mix independently of raw volume. |
| **Created from `invoice_train.invoice_date`:** `mean_invoice_gap_days`, `max_invoice_gap_days` | Average and longest spacing between a client's recorded invoice dates | `invoice_count` and `active_days` do not show whether invoices arrive regularly or contain long pauses. We created gaps to represent regularity. | No strong standalone fraud-rate result was established. Clients with fewer than two invoices receive 0, meaning “no measurable gap,” not proven zero-day spacing. | Use `mean_invoice_gap_days` in the baseline; later compare missing plus an availability flag. Hold back the outlier-sensitive maximum. |
| **Created from each invoice's indices:** `backwards_index_rate` | Proportion of invoices where `new_index − old_index < 0` | A raw backward-reading count would be larger for clients with more invoices. The rate makes clients more comparable and keeps possible resets, corrections, or unusual readings as information. | Clients with at least one backward index have **11.34% fraud vs 5.51%** without one. We keep the rows instead of treating the reading as an automatic error. | Enters the baseline as a candidate meter-behaviour signal; interpretation remains uncertain and must not be presented as proof of fraud. |
| **Created from `counter_number`, `new_index`, and `invoice_date`:** `meter_count`, `mean_monthly_submission_index_delta`, `backward_submission_rate` | Number of unique meter identifiers and changes between consecutive readings for the same identifier, ordered by invoice date and adjusted for elapsed days | The within-invoice index difference does not describe how the same identifier changes across records. Unequal time gaps also make raw changes difficult to compare. | Median backward-sequence rate: **11.94% fraud vs 8.82% non-fraud**. Highest monthly-change group: **7.39% fraud vs 3.75%** in the lowest. **5,333 clients** lack a valid transition. | All three enter the baseline. Later analysis must confirm that the signal is not only longer history and that identifiers represent the expected meter units. |
| **Created from indices and consumption columns:** `large_mismatch_count`, `reconciliation_gap_abs_mean` | Size and frequency of the absolute gap between `new_index − old_index` and summed `consommation_level_1..4` | Consumption and meter movement describe related quantities. Comparing them creates a candidate consistency check that neither value provides alone. | “Large” currently means an unvalidated gap **>10**. At least one: **8.82% of fraud vs 5.52% of non-fraud clients**. The mean is heavy-tail sensitive, and the count increases with invoice exposure. | Both enter the baseline as candidates. Domain validation should set the threshold; later compare a mismatch rate that adjusts for invoice count. |
| **Existing in `client_train`, prepared as categories:** `client_catg`, `disrict`, `region` | Customer-category and geographic group membership | Although stored as numbers, these are labels. Passing them as continuous numbers would invent false distances and ordering between codes. | Category 51: **16.87% fraud**; region 103: **10.30%**; overall fraud rate: **5.58%**. One-hot encoding expands each field into yes/no category columns. | All three enter the baseline as one-hot features. Error analysis must check group performance because these are associations, not causes. |

**Say:** “The raw data told us what was billed. These features describe each client's history, regularity, meter behaviour, and inconsistencies.”

**WIP: Final selection:** the baseline keeps **12 created numeric features plus 3 prepared categorical fields**. Other aggregates remain available for later experiments; they are not permanently deleted merely because they were left out of the first model.

## Slide 7 — EDA found strong patterns and one major bias risk

- Fraud rises from **0.94%** in the shortest-history group to **9.15%** in the longest-history group.
- `client_catg = 51`: **16.87% fraud**; region 103: **10.30% fraud**.
- A backward meter index: **11.34% fraud** versus **5.51%** without one.
- The highest monthly meter-change group has **7.39% fraud** versus **3.75%** in the lowest, while typical timing is almost identical.
- At least one large index-versus-consumption mismatch appears for **8.82% of fraud clients versus 5.52% of non-fraud clients**.

**Visual:** one compact findings table with the base fraud rate as a reference.

**Named bias — observation-window (exposure) bias:** clients observed for longer have more invoices, more recorded events, and more opportunities to have fraud detected. Because fraud rate rises sharply with invoice history, the model may learn **“long-known client”** instead of fraud behaviour that would generalise to a newer client.

**Possible label/inspection bias:** category and regional fraud rates may partly reflect where past inspections were concentrated, not only the underlying fraud risk. We must compare errors and recall across these groups.

**Model evidence:** `active_days` and `meter_count` have the two largest positive standardized numeric coefficients. Coefficients are not definitive feature importance, but they show that exposure/history reliance is already visible in the fitted baseline.

**Temporal/cohort risk:** invoice coverage jumps sharply around 2005. A random client split does not test a later time period; deployment requires an as-of prediction cutoff and temporal backtesting.

**Data note:** the sequence comparison excludes 5,333 clients without a valid transition between invoice-dated meter readings.

## Slide 8 — The EDA changed what entered the model

| Keep as baseline predictors | Exclude or postpone | What the baseline code does |
|---|---|---|
| 12 created numeric features | `client_id`, raw dates, invalid tenure, and possible last-invoice leakage fields | Numeric pipeline: median `SimpleImputer` → `StandardScaler` |
| `client_catg`, `disrict`, `region` | All `months_number` features and redundant/outlier-sensitive aggregates | Categorical pipeline: most-frequent `SimpleImputer` → `OneHotEncoder(handle_unknown="ignore")` |
| All client and invoice rows remain in the source data | 11 duplicate-looking rows are investigated, not automatically deleted | `ColumnTransformer` combines both pipelines; `.fit()` is called on `model_train` only |

**Say:** “An outlier is not automatically an error; investigate before changing it.”

**Where this appears:** these operations are implemented in `02_baseline_model.ipynb`, not in the EDA export. The EDA saves raw selected values; the model pipeline learns imputation, scaling, and one-hot categories after the split.

**Scaling limitation:** monthly index changes and reconciliation gaps are extremely heavy-tailed. `StandardScaler` is implemented, but it is not robust to extremes; compare a log transformation or `RobustScaler` later.

**Effect on the project:** the baseline uses 12 understandable numeric features and 3 categorical fields rather than every available column. Error analysis must specifically check history length and geography, while a real deployment would need a prediction-date cutoff to prevent future information from leaking in.

## Slide 9 — Baseline: one simple linear classifier

- Compare a `DummyClassifier` with **logistic regression**.
- Logistic regression is the linear model for a binary target; linear regression is for numeric targets.
- Use an explicit EDA-informed feature list, not every available column.
- Pipeline: median imputation → standard scaling / one-hot encoding → logistic regression.

**Visual:** small pipeline diagram.

## Slide 10 — Baseline results

Verified on the 20% validation split created from all labelled client rows:

| Model | Average precision | Precision@200 | Recall@200 | Lift@200 |
|---|---:|---:|---:|---:|
| Dummy | 0.056 | — | — | — |
| Logistic regression | 0.188 | 27.5% | 3.6% | 4.93x |

**Say:** 200 of 27,099 validation clients is the same population share as 1,000 of all 135,493 labelled clients. The model finds **55 fraud clients** in those 200, giving **27.5% precision** and about **4.9 times** the validation fraud concentration.

**Limitations:** the dummy gives every client the same score, so it cannot create a meaningful shortlist. A full-population budget of 1,000 is illustrative, not confirmed by a stakeholder. This baseline uses one random validation split after label-informed EDA and does not test a future time period; use repeated and temporal validation before making a firm performance claim.

## Slide 11 — What we will inspect in the errors

- False positives: clients inspected unnecessarily.
- False negatives: fraud clients the shortlist missed.
- Compare errors by category, region, electricity share, and amount of invoice history.
- Check whether performance comes mainly from history length or possible leakage.

**Visual:** confusion matrix plus two example client profiles.

## Slide 12 — Recommendation and limitations

- Use the model as a ranking tool for investigations, not proof of fraud.
- Choose `k` from the real inspection capacity and report precision and recall at that `k`.
- For real deployment, define a prediction date and use only information available before inspection.
- Next iteration should follow the error analysis, not add complexity without evidence.

**Closing line:** First understand the data. Then teach the model. Then check whether it learned fairly.
