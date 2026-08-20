# Fraud detection in electricity and gas consumption

## Presentation blueprint

This is the detailed speaking plan for the PowerPoint deck. Keep one main message per slide. The text here is more detailed than the final slides so Roy and Sean can choose what to say. Use the notebook tables and charts as the source of truth when a notebook is rerun.

## Slide 1 — Title

**Predicting which electricity and gas clients should be inspected first**

- Team: Roy and Sean
- Dataset: labelled electricity and gas consumption clients
- Goal: rank clients by fraud risk so a limited inspection team can start with the most promising cases
- Repository: link to the GitHub project

**Visual:** simple client → meter → invoice illustration.

**Say:** “We are not trying to prove fraud automatically. We are building a ranking tool to help decide who should be inspected first.”

## Slide 2 — Business case

A utility company cannot inspect every client. The practical question is:

> If the team can inspect only a limited number of clients, which clients should it inspect first?

The business value is finding more real fraud cases within the available inspection capacity while avoiding unnecessary investigations of non-fraud clients.

**Visual:** a funnel from all clients to a small inspection queue.

## Slide 3 — What does value mean?

The target is highly imbalanced:

- 135,493 labelled clients
- 7,566 fraud clients (**5.58%**)
- 127,927 non-fraud clients

Accuracy is not the main measure. Predicting “not fraud” for everybody would be about 94% accurate but would find no fraud.

**Primary metric:** average precision / PR-AUC, because it evaluates ranking quality when fraud is rare.

**Operational metrics:** precision and recall at a chosen inspection capacity, such as the top 10% of clients. The final threshold should be chosen with the inspection team, using the cost of missed fraud and unnecessary inspections.

**Say:** “Precision tells us how many flagged clients are actually fraud. Recall tells us how much of all fraud we found.”

## Slide 4 — Our data-science question

We treated this as a ranking problem, not an automatic fraud verdict. The investigation team has limited capacity, so the question is:

> Can the information available at client level produce a useful, explainable ranking — and where does that ranking fail?

Our lifecycle was:

1. Understand the client and invoice tables.
2. Check data quality and unusual readings.
3. Aggregate invoices into one row per client.
4. Select a small, explainable feature contract.
5. Compare a simple baseline with stronger model families.
6. Analyse where the models fail.
7. Tune one candidate and evaluate it once on a final in-memory holdout.

**Project decision:** build the simplest defensible baseline first, then use model comparison and error analysis to decide whether extra complexity is justified.


## Slide 4.5 The conclusion first

**Use the tuned LightGBM as a ranking aid for inspection prioritisation — not as an automatic fraud verdict.**

Why this is our recommendation:

- The client-level features contain real signal: final one-time holdout **PR-AUC 0.2647** versus the dummy ranking baseline of about **0.056**.
- The model is useful for ordering a limited inspection queue, but it does not separate fraud perfectly.
- Error analysis shows a structural blind spot: short-history clients are often missed, and performance varies by region.
- Human review, a capacity-based threshold, and segment monitoring are required before operational use.

The next slides show how we reached this conclusion:

`raw tables → EDA and data-quality checks → client features → baseline → model comparison → error analysis → final recommendation`

**Say:** “We will give the recommendation first, then walk through the evidence and decisions that support it.”

## Slide 5 — EDA: from invoices to one prediction row per client

- **135,493** labelled client rows in `client_train`.
- **4,476,749** invoice rows in `invoice_train` — a one-to-many relationship, with one target per client.
- We read invoices in chunks; the full aggregation completed in **16.4 seconds** in the executed notebook.
- We turned repeated dates, consumption, meter readings, gaps, and reconciliation checks into **one client-level feature row**.
- The saved EDA artifact is `data/processed/df_training.csv` (**135,493 rows, 17 columns: key, target, and 15 candidate inputs**).
- Raw source tables have no missing cells. Two transition features are unavailable for **5,333 clients** because there is no valid positive-day transition; imputation happens only inside modelling.

**Visual:** `client → many invoices → client-level feature row → model score`.

![EDA target balance](presentation/assets/eda_target_balance.png)

**Evaluation fact:** 01 is EDA and feature construction. 02–04 use five-fold stratified CV on the complete saved client table. 05 creates a one-time in-memory holdout after model choices are frozen.

## Slide 6 — EDA: the feature contract we built

The target is in `client_train`; repeated behaviour is in `invoice_train`. Every created feature is aggregated to one row per client before modelling.

| Feature family | Examples | Why this family matters | What we learned / decision |
|---|---|---|---|
| **History** — `invoice_count`, `active_days`, `mean_invoice_gap_days` | Created from invoice dates | Make exposure and regularity explicit; totals alone mostly measure how long we observed a client | Fraud medians are **41 vs 29 invoices** and **4,727 vs 3,283 active days**. Keep, but treat history as a bias risk. |
| **Consumption** — `mean_consumption`, `zero_consumption_rate`, `elec_share` | Created from consumption and meter type | Compare typical use and mix across clients with different invoice counts | Median mean consumption is **466 vs 393**. Keep means/rates; postpone raw totals and maxima initially. |
| **Meter behaviour** — `backwards_index_rate`, `meter_count`, `mean_monthly_submission_index_delta`, `backward_submission_rate` | Created from meter IDs, indices, invoice dates, and elapsed days | Capture corrections/resets and changes between consecutive readings for the same meter | Backward-index clients: **11.34% fraud vs 5.51%**. **5,333** clients lack a valid transition, so the model imputes those derived values. Keep as candidate signals, not proof. |
| **Reconciliation** — `large_mismatch_count`, `reconciliation_gap_abs_mean` | Created by comparing index change with summed consumption | Test whether billed consumption is consistent with meter movement | A large mismatch occurs for **8.82% of fraud vs 5.52% of non-fraud**. Keep; validate the `>10` threshold with domain experts. |
| **Category/geography** — `client_catg`, `disrict`, `region` | Original `client_train` fields, prepared as categorical inputs | Numeric codes are labels, not measurements; categorical handling avoids fake distances | Category 51 has **16.87% fraud** versus the **5.58%** base rate. Keep all three; check group performance for inspection bias. |

**Final contract:** **12 created numeric features + 3 original categorical fields = 15 client inputs.** Other aggregates remain available for later experiments; they were postponed, not silently discarded.

**Say:** “The raw data tells us what was billed. The feature contract turns that history into comparable signals about exposure, behaviour, regularity, and consistency.”

## Slide 7 — EDA: strong patterns and bias risks

### Strong patterns

- Fraud rate rises from **0.94%** in the 1–7 invoice group to **9.15%** in the longest-history group.
- Category 51 has **16.87% fraud**, roughly three times the overall base rate.
- A backward meter index is associated with **11.34% fraud**, compared with **5.51%** without one.
- The highest monthly meter-change group has **7.39% fraud**, compared with **3.75%** in the lowest group.
- A large index-versus-consumption mismatch appears for **8.82% of fraud clients** and **5.52% of non-fraud clients**.

### Named risks

- **Observation-window bias:** clients observed for longer have more invoices and more opportunities for fraud to be detected. The model may learn “long-known client” rather than behaviour that generalises to a newly observed client.
- **Inspection/label bias:** category and regional differences may partly reflect where inspections were historically concentrated.
- **Temporal/cohort risk:** invoice coverage changes strongly over calendar time. A random client split does not test future-time performance.
- **Leakage risk:** last-invoice `reading_remarque` and `counter_statue` may reflect later investigation or account events, so they are excluded.

**Visual:** history bias plus the reconciliation scatterplot from the notebook.

![Fraud rates by client category, district, and region](presentation/assets/eda_categorical_patterns.png)

![Index change versus reported consumption](presentation/assets/eda_reconciliation.png)

## Slide 8 — EDA conclusion: the 15 client inputs

The baseline uses **12 created numeric features plus 3 original categorical fields**.

| Group | Final inputs | Origin |
|---|---|---|
| Client category and geography | `client_catg`, `disrict`, `region` | Original `client_train` fields, prepared as categories |
| History and timing | `invoice_count`, `active_days`, `mean_invoice_gap_days` | Created from invoice dates |
| Consumption | `mean_consumption`, `zero_consumption_rate`, `elec_share` | Created from invoice consumption and meter type |
| Meter behaviour | `backwards_index_rate`, `meter_count`, `mean_monthly_submission_index_delta`, `backward_submission_rate` | Created from meter identifiers, indices, and invoice dates |
| Reconciliation | `large_mismatch_count`, `reconciliation_gap_abs_mean` | Created from meter-index changes and summed consumption |

Before modelling:

- We keep all raw invoice rows; we do not automatically delete the 11 duplicate-looking rows or backward readings.
- We do not fill missing values in the EDA. The raw tables have no missing cells, but the two derived meter-transition features are missing for **5,333 clients**; the modelling pipeline median-imputes those values using each training fold only.
- We exclude `client_id`, `target`, raw dates, unreliable tenure, `months_number` features, split labels, and possible last-invoice leakage fields.
- Features left out initially are postponed, not permanently deleted.

**Handoff:** `client_train + invoice_train → one row per client → 15 inputs → df_training.csv → modelling`.

The EDA did not delete invoice rows. It made an explicit, reproducible decision about which client-level signals enter the first model and which questionable or redundant columns wait for error analysis.

![Client-level feature correlations](presentation/assets/eda_feature_correlations.png)

## Slide 9 — Reproducible modelling pipeline

- **01 EDA:** deterministic feature builder → `df_training.csv` + reusable `feature_engineering_pipeline.joblib`.
- **02 baseline:** dummy classifier versus regularised logistic regression; same 15-input contract.
- **03 extension:** same cross-validation and inputs across six model families.
- **04 error analysis:** out-of-fold predictions expose agreement, blind spots, and segment weaknesses.
- **05 final model:** tune the leading family, reject an unhelpful ensemble, then score one in-memory holdout once.
- Preprocessing is fitted within each training fold: numeric median imputation/scaling and categorical handling. The EDA itself does not fill source-table missing cells.

**Visual:** `raw files → feature builder → 15 inputs → fold-safe preprocessing → CV/model → error analysis → one-time holdout`.

## Slide 10 — Baseline model and results

We started with a deliberately simple reference: a stratified dummy classifier and regularised logistic regression. This answers the first ML question: do the selected features contain signal beyond the class prior?

| Model | Five-fold PR-AUC | ROC-AUC |
|---|---:|---:|
| Dummy classifier | **0.056** | **0.502** |
| Logistic regression | **0.170** | **0.795** |

Interpretation:

- Logistic regression improves PR-AUC from **0.056 to 0.170** — about **3× the dummy reference**.
- ROC-AUC of **0.795** confirms useful separation, but PR-AUC remains far from 1.0: fraud and non-fraud clients still overlap heavily.
- The simple model is explainable and establishes a fair floor for every later model family.

**Visual:** keep this slide visual-light: the result table is the evidence. The more informative precision-recall comparison appears after all candidate families are evaluated.

## Slide 11 — Model extensions and results

We compared linear, distance-based, tree, and anomaly-detection approaches using the same 15-input contract and five-fold CV.

| Model | PR-AUC | ROC-AUC | Decision |
|---|---:|---:|---|
| LightGBM | **0.247** | **0.831** | Leading untuned candidate |
| XGBoost | 0.232 | 0.812 | Useful reference, behind LightGBM |
| Random Forest | 0.227 | 0.816 | Useful reference, behind LightGBM |
| Logistic regression | 0.170 | 0.795 | Simple baseline |
| KNN | 0.139 | 0.687 | Weaker ranking |
| Isolation Forest | 0.072 | 0.560 | Close to the dummy reference |

**Decision:** continue with LightGBM for tuning. It improves on logistic regression by **0.077 PR-AUC**, while the other tree models remain close behind. The gain is evidence for non-linear combinations, not evidence that one feature proves fraud.

![Precision-recall curves for the model extensions](presentation/assets/model_precision_recall_comparison.png)

## Slide 12 — Error analysis: where the models fail

The three tree models agree strongly:

- The agreement plot counts fraud clients by how many of the three models **missed** them: **2,250 missed by 0**, **1,066 by 1**, **1,002 by 2**, and **3,248 by all 3**.
- **3,248 fraud clients (42.9% of all fraud)** were missed by all three models.
- **2,250 fraud clients** were caught by all three.
- The remaining fraud clients were split between partial agreement groups.
- The shortest-history group (1–7 invoices) has only **2.3% LightGBM recall**, **5.8% Random Forest recall**, and **6.2% XGBoost recall** in the top-10% inspection analysis.
- Among long-history fraud clients, **3,017** were still missed by every model; this is not explained by short history alone.
- The current feature comparison among that long-history missed group shows the largest median difference for `mean_invoice_gap_days` (effect size **−0.20**), but this is exploratory, not causal evidence.

**Message:** changing model family alone will not solve the main blind spot. We need better signal for short-history and consistently missed clients, and we need to verify that the ranking is useful in the intended inspection segments.

![Fraud clients caught by zero, one, two, or all three models](presentation/assets/error_models_caught_0_to_3.png)

**How to read it:** `0` means all three models caught the client; `3` means all three models missed the client. This is the clearest evidence that the main limitation is shared missing signal, not only the choice of model family.

![Missed-by-all fraud split by history](presentation/assets/error_missed_history_split.png)

## Slide 13 — Final tuned model

We tuned LightGBM and compared one-hot preprocessing with native categorical handling.

| Candidate | Development CV PR-AUC |
|---|---:|
| Untuned LightGBM | **0.2379** |
| Tuned LightGBM, one-hot | **0.2470** |
| Tuned LightGBM, native categorical | **0.2514** during tuning; **0.2506** OOF comparison |
| Tuned LightGBM + Random Forest ensemble | **0.2499** |

**Decision:** keep tuned LightGBM with native categorical handling. It is the best development result, and the ensemble was rejected because it was slightly worse (**0.2499 vs 0.2506**) and did not justify extra complexity.

**One-time in-memory holdout:** PR-AUC **0.2647**, ROC-AUC **0.8354**.

**Caveat:** this final holdout is a useful final check created from `df_training.csv`, but it is not the separate unlabelled Kaggle test set and should not be reused for repeated tuning.

![Final holdout recall by invoice-history group](presentation/assets/final_holdout_recall_by_invoice_history.png)

## Slide 14 — Kaggle submission: a credible first result

After submitting the final model, our public leaderboard result was:

- **Rank 275 of 710 actually rated participants** — approximately the **top 39%** of the rated leaderboard.
- **Public score: 0.826022007**.
- **One submission** recorded at the time of the screenshot.

This is a positive external check: the model was not only strong in our local validation, it produced a competitive first submission. The public leaderboard is provisional because Kaggle notes that it reflects only part of the test set until the competition closes.

![Kaggle public leaderboard result](assets/kaggle_leaderboard_rank_275.png)

**Interpretation:** rank 275 is not a “winning” claim, but it is a credible result for a first submitted model and supports the decision to focus next on feature quality, temporal validation, and error analysis rather than blindly tuning more.

## Slide 15 — Recommendation

Use the tuned LightGBM as a ranking aid for inspection prioritisation, not as an automatic fraud verdict.

The model has useful lift over the baseline, but the error analysis identifies important limits:

- It performs poorly for clients with very short invoice histories.
- Regional performance is uneven: holdout recall was **19.9% in region 104**, **29.1% in region 107**, and **68.2% in region 311** at the same inspection rule.
- A strong overall metric can hide weak performance in specific client groups.
- Human review and a capacity-based threshold remain necessary.

**What we can claim:** the final model ranks better than the baseline and passes a one-time holdout check.

**What we cannot claim:** that it proves fraud, that it is equally reliable in every region, or that the random split represents future deployment performance.

## Slide 16 — Next steps

1. **Define the operating point:** inspection capacity, cost of false positives, and cost of missed fraud.
2. **Add short-history signal:** relative or sequence features that work when only a few invoices exist.
3. **Validate semantics:** confirm meter identifiers and the reconciliation threshold with utility experts.
4. **Test time:** use an as-of date and temporal backtesting instead of relying only on a random split.
5. **Check segments:** repeat precision/recall by region, category, history length, and meter availability.
6. **Protect the holdout:** freeze the feature contract and threshold before any further final evaluation.

## Slide 17 — Extra / bonus: reproducibility and CI/CD

We added engineering practices beyond the model itself:

- **GitHub Actions workflow:** on pull requests and pushes to `main`, execute the EDA and baseline notebooks headlessly with `nbconvert`; a notebook exception fails CI.
- **Reproducible data path:** committed Parquet inputs avoid repeatedly loading the large CSVs in the browser; `.env` controls whether generated artifacts are saved.
- **Reusable feature pipeline:** `feature_engineering_pipeline.joblib` stores the same client-level feature logic for future data, rather than rebuilding features by hand.
- **Explicit feature contract:** the 15 inputs and excluded/leakage-prone fields are documented in the EDA and carried into 02–05.
- **Presentation evidence:** charts are extracted from executed notebook outputs and stored alongside the markdown blueprint.

**Bonus lesson:** reproducibility is part of the ML deliverable. A good score is not enough if the next run cannot recreate the table, preprocessing, and evaluation.

**Closing sentence:** “The project gives us a reproducible ranking baseline, identifies where it fails, and makes the next data-collection and validation steps explicit.”
