# Fraud Detection in Electricity and Gas Consumption

Collaborative data-science challenge project by Roy and Sean. The goal is to identify clients who may be committing electricity or gas consumption fraud and to turn model scores into a useful, capacity-aware inspection shortlist.

Dataset: [Fraud Detection in Electricity and Gas Consumption on Kaggle](https://www.kaggle.com/datasets/mrmorj/fraud-detection-in-electricity-and-gas-consumption)

## Stakeholder framing

The operational stakeholder is assumed to be a utility fraud-investigation team with limited inspection capacity. A useful model should find as many fraudulent clients as possible while keeping unproductive inspections manageable.

Primary model-selection metric: **average precision (PR-AUC)**. It measures ranking quality under the strong class imbalance (about 5.6% fraud in the training clients) and is more informative here than accuracy.

**ROC-AUC** is reported alongside it as a secondary ranking metric, but is not sufficient on its own: with only ~5.6% fraud, it can look strong even for a model that isn't very useful for a capacity-limited inspection team, since it's dominated by how well the model ranks the (very common) non-fraud majority.

## Project structure

| Path | Purpose |
|---|---|
| `01_eda.ipynb` | Runnable, memory-aware EDA and client-level invoice aggregation |
| `02_baseline_model.ipynb` | Day 2 baseline-model scaffold |
| `03_error_analysis.ipynb` | Day 3 threshold and error-analysis scaffold |
| `04_final_model.ipynb` | Final comparison and holdout-evaluation scaffold |
| `scripts/csv_to_parquet.py` | Headless, streaming CSV-to-Parquet conversion |
| `data/` | Source CSVs, committed Parquet inputs, and local generated features |

The numbered notebooks are intentionally separate so the analysis, baseline, error analysis, and final model can evolve independently.

## Data setup

Download and unzip the Kaggle dataset into `data/`. `01_eda.ipynb` reads the committed Parquet inputs; the CSVs remain useful as the original source and for regenerating those files.

```text
data/
├── client_train.csv
├── invoice_train.csv
├── client_test.csv
├── invoice_test.csv
├── SampleSubmission.csv
└── parquet/
    ├── client_train.parquet
    ├── client_test.parquet
    ├── invoice_train.parquet
    └── invoice_test.parquet
```

The raw files are deliberately excluded from Git. Each collaborator should download the same dataset locally.

If you want Parquet copies for faster repeated reads, run the conversion outside Jupyter:

```bash
docker run --rm -it \\
  -v "$PWD":/home/jovyan/work \\
  -w /home/jovyan/work \\
  quay.io/jupyter/scipy-notebook:2025-12-31 \\
  python scripts/csv_to_parquet.py
```

The output is written to `data/parquet/` for all four client/invoice train and test tables. The converter streams row groups and explicitly handles the mixed `counter_statue` values, so it avoids the browser-kernel crash/schema error from converting the full invoice CSV with pandas inference. `SampleSubmission.csv` stays CSV because it is already a small Kaggle submission template.

## Run with local Jupyter Docker

From the repository root:

```bash
docker run --rm -it \
  -p 8888:8888 \
  -v "$PWD":/home/jovyan/work \
  quay.io/jupyter/scipy-notebook:2025-12-31
```

Open the URL printed by Jupyter, enter the `work` folder, and start with `01_eda.ipynb`. The SciPy notebook image includes the pandas, NumPy, Matplotlib, Seaborn, and scikit-learn packages used here.

The EDA reads invoices in chunks and aggregates them to one row per client. A chunk size of `250_000` is the default; allow roughly 1 GB of container memory (the full verified run peaked near 600 MB). Lowering `CHUNK_SIZE` reduces transient chunk memory, although client aggregates and duplicate-row hashes still accumulate across the file. The dated Jupyter image tag is pinned so Roy and Sean receive the same package environment.

### Local notebook settings

`01_eda.ipynb` loads local settings from `.env`. Copy `.env.example` if needed; `.env` is ignored by Git, so each collaborator can use their own values.

The default is `SAVE_TRAINING_ARTIFACTS=false`, which means the EDA explores the data without writing generated files. Set it to `true` when you are ready to save the complete `df_training.csv` and the fitted feature-engineering pipeline in `data/processed/`:

```env
SAVE_TRAINING_ARTIFACTS=true
```

GitHub Actions sets this value automatically when it runs the notebooks.

## Recommended workflow and milestones

- **Milestone 1 — baseline, Day 2 at 17:00:** complete EDA, freeze a client-level validation split, and compare a dummy model with one simple classifier.
- **Milestone 2 — slides draft, Day 3 at 12:00:** draft the stakeholder problem, data, validation design, baseline, and next-step slides.
- **Milestone 3 — model plus error analysis, Day 3 at 16:00:** analyze false positives/negatives and performance by client segment, then revise features or threshold.
- **Milestone 4 — final deliverables, Day 4 at 13:00:** select the model using validation data, evaluate once on the untouched holdout, finish slides, and rehearse.

Suggested GitHub issues/checklists:

- Problem framing and definition of inspection value
- Data audit and EDA
- Validation strategy and leakage checks
- Baseline model
- Feature engineering and model iteration
- Error and segment analysis
- Slides and presentation
- Reproducibility and final QA

## What the EDA should focus on

1. **Unit of prediction and leakage.** The target lives at client level while invoices form a one-to-many history. Aggregate first and split clients—not invoice rows. Fit every learned transformation using training clients only.
2. **Class imbalance.** Quantify the 5.6% fraud rate and use stratified client-level splits. Avoid accuracy as the headline metric.
3. **Observation-window bias.** Invoice histories range widely in length. Check whether fraud labels correlate with number of invoices, customer age, or first/last invoice date; otherwise the model may learn who was observed longer.
4. **Consumption behavior.** Compare total/average/max consumption, zero-consumption frequency, index changes, tariff levels, and electricity versus gas activity by target. Heavy-tailed variables should be viewed on log scales.
5. **Temporal behavior.** Create recency, active span, changes over time, sudden drops/spikes, and variability features. For later iterations, compute recent-versus-historical ratios.
6. **Data quality.** Audit duplicate invoices, missing values, impossible date order, index inconsistencies, unusual status/category codes, and spelling-preserved source columns such as `disrict` and `counter_statue`.
7. **Segment robustness.** Compare performance by region, client category, meter type, history length, and customer tenure. A strong global score can hide weak or unfair operational segments.
8. **Capacity-aware evaluation.** Translate model scores into precision and recall at the actual number of investigations available.

## Modeling guardrails

- Create and retain the deterministic `split` column in the EDA feature table. All target-led EDA and model iteration use development clients; final-holdout clients are scored only once after the approach is frozen.
- Use cross-validation only inside the development partition.
- For a real operational backtest, set an invoice `AS_OF_DATE` and confirm the fraud label occurred after that cutoff. The Kaggle snapshot alone does not provide a deployment-safe label timestamp.
- Keep feature aggregation identical for train and Kaggle test clients.
- Do not use `client_id` as a numeric/categorical predictor; retain it only as a key.
- Build the first model before elaborate feature engineering: dummy baseline, logistic regression or tree ensemble, PR-AUC, then error analysis.
- Record assumptions and decisions in Markdown cells so the notebooks can feed directly into the slides.

## Collaboration

Work in small branches, avoid committing raw data or notebook checkpoints, and agree who owns each numbered notebook before editing. Before merging, restart the kernel and run all cells from top to bottom.
