# SMS Spam Detection

A reproducible English SMS classification baseline with TF-IDF, Naive Bayes, logistic regression, and error analysis.

## Project status

New portfolio implementation prepared in October 2026 with coding-assistant support.
This is not a historical project submission, employer deliverable, or deployed system.
The bundled data is synthetic. See [DATA_CARD.md](DATA_CARD.md) before interpreting outputs.

## Run locally

Python 3.11 is recommended. From this repository folder:

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python model.py --data data/demo.csv
python -m pytest -q
python -m streamlit run app.py
```

Commands use paths relative to the repository root. Outputs are written to `results/`.
The Streamlit app reads local sample data until a CSV is uploaded (where supported).
CSV exports can contain source text; review sensitive content before sharing them.

## Design

The core module is usable without Streamlit. The CLI exports machine-readable results;
the dashboard provides an interactive view. Tests exercise failure cases and invariants,
including duplicate handling, invalid inputs, and distribution checks where applicable.
The original inputs are preserved. There is no automatic overwrite of source data.

## Results and limitations

Root outputs in `results/` use synthetic fixtures. `results/uci/metrics.json` records
a separate run on the public UCI dataset downloaded on 1 October 2026. Dependency lower/upper bounds
support installation; `results/environment.txt` records the actual validation environment.
Run the commands yourself before presenting or extending this project.

## Next steps

Read the code, explain each decision, and replace fixtures with appropriately licensed
data. Keep a record of your experiments, review failures, and add results from a real
held-out dataset where relevant. Document changes and remaining limitations.

## License

Code: MIT. External datasets retain their own licenses and attribution requirements.

## Public-data evaluation

```bash
python download_data.py
python model.py --data data/SMSSpamCollection
```

The logistic regression model choice is fixed before reporting held-out metrics.
TF-IDF is fitted only on the training partition. Compare spam precision, recall, F1,
and the confusion matrix against the majority baseline; accuracy alone is misleading.
No hyperparameter search on the test partition is performed.

## Recorded UCI baseline

Downloaded file: 5,572 rows; after normalized exact deduplication: 5,157.
Seed 42 stratified split: 4,125 train / 1,032 test (128 spam).

| Model | Accuracy | Spam precision | Spam recall | Spam F1 |
|---|---:|---:|---:|---:|
| majority | 0.8760 | 0.0000 | 0.0000 | 0.0000 |
| naive_bayes | 0.9409 | 1.0000 | 0.5234 | 0.6872 |
| logistic_regression | 0.9593 | 1.0000 | 0.6719 | 0.8037 |

The logistic regression baseline misses 42 of 128 spam messages despite 95.93%
accuracy. Improving recall is an important next experiment. Perfect observed
precision on this split is not a guarantee. Future tuning must use a validation
partition or cross-validation within training data, leaving a final test untouched.
Near-duplicate overlap and historical-data bias remain limitations.
