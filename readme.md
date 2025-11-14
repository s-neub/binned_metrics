# Binned Performance Metrics Monitor

## Overview

This monitor calculates model performance metrics (e.g., Sensitivity, Specificity) and/or numeric aggregations (e.g., mean of `age`) over specified time windows.

The primary output is a set of JSON objects formatted for ModelOp Center's timeline graphs, allowing users to track model performance week-over-week, month-over-month, and year-over-year.

## Key Features

* **Binned Metrics:** Calculates 20+ performance metrics (like `SEN`, `SP`, `F1`, `ACC`, `MCC`) for binary classifiers.
* **Binned Aggregations:** Calculates simple aggregations (like `mean`, `median`, `sum`) on any numeric column.
* **Flexible Time Windows:** Supports any standard pandas time frequency string (e.g., `W` for Weekly, `MS` for Month-Start, `Q` for Quarterly).
* **Configurable:** All columns, values, and metrics are configurable via job parameters, making the monitor highly reusable.

## 📦 ModelOp Center Configuration

When creating a new monitor in ModelOp Center, you will use these files.

    binned_metrics_monitor/
    ├── .gitignore
    ├── binned_metrics.py     # Your existing script
    ├── binned_metrics_development.ipynb    # development sandbox before configuring monitor in MOC
    ├── job_parameters.json   # For local testing (replaces hardcoded params)
    ├── readme.md             # This file: monitor documentation
    ├── required_assets.json  # Tells MOC what data to provide
    └── synthetic_2021_2024_prediction_data.csv # Small sample data to make main() run

### Entry Point

* **Init Function:** `init`
* **Metrics Function:** `metrics`
* **Python File:** `binned_metrics.py` (Note: MOC may default to `metrics.py`. If so, either rename your file or specify `binned_metrics.py` as the source file).

### Required Assets (`required_assets.json`)

This monitor requires one asset: the "baseline" DataFrame. The `required_assets.json` file is already configured for this.

### Job Parameters

These parameters are defined in the `init()` function and can be set in the ModelOp Center UI to configure the monitor's behavior.

| Parameter | Type | Default Value | Description |
| :--- | :--- | :--- | :--- |
| `TIMESTAMP_COLUMN` | String | `'Date'` | The name of the DataFrame column that contains the datetime for each record. |
| `LABEL_COLUMN_NAME` | String | `'label'` | The name of the column containing the **actual** (ground truth) values. |
| `LABEL_FALSE_VALUE` | Any | `False` | The specific value in the `LABEL_COLUMN_NAME` column that represents "False" (0). |
| `LABEL_TRUE_VALUE` | Any | `True` | The specific value in the `LABEL_COLUMN_NAME` column that represents "True" (1). |
| `SCORE_COLUMN_NAME` | String | `'score'` | The name of the column containing the **predicted** values from the model. |
| `SCORE_FALSE_VALUE` | Any | `'NO'` | The specific value in the `SCORE_COLUMN_NAME` column that represents "False" (0). |
| `SCORE_TRUE_VALUE` | Any | `'YES'` | The specific value in the `SCORE_COLUMN_NAME` column that represents "True" (1). |
| `METRICS_TO_CALC` | List[Str] | `['SEN', 'SP']` | A list of performance metric abbreviations to calculate. |
| `BINS_TO_CALC` | List[Str] | `['W', 'MS', 'YS']` | A list of pandas time-bin strings (e.g., `W`eekly, `M`onth-`S`tart, `Y`ear-`S`tart). |
| `NUMERIC_AGGS_TO_CALC` | Dict | `{'patient_age': 'mean'}` | A dictionary of numeric aggregations. Format: `{"col_name": "agg_func"}`. |

## 📈 Output

The `metrics()` function yields a JSON object containing data for three timeline graphs:

* `baseline_time_line_graph_weekly`
* `baseline_time_line_graph_monthly`
* `baseline_time_line_graph_yearly`

Each graph object contains a `data` key, which holds the time-series data for each calculated metric (e.g., `SEN`, `SP`, `patient_age_mean`).

## 💻 Local Development Workflow

We recommend a two-stage development process: first, use the **Jupyter Notebook** for interactive development and testing, then use the **Python script** for a final validation.

### 1. Interactive Development (Jupyter Notebook)

The file `binned_metrics_development.ipynb` is a "development sandbox" for interactively building and testing your monitor.

The notebook allows you to:
1.  **Test Components:** Interactively test the core logic (like `_calculate_metrics`) and the main data-processing function (`calculate_binned_metrics`) with small, custom-built DataFrames.
2.  **Iterate Quickly:** Create test data that targets edge cases (e.g., missing data, unmapped values, empty time bins) and see the results immediately.
3.  **Simulate ModelOp Center:** The notebook has a final section that perfectly simulates the MOC environment by calling `init()` with test parameters and then passing a DataFrame to `metrics()`.
4.  **Validate JSON:** You can print and visually inspect the *exact* final JSON output that MOC will use for its graphs.

**Recommended Flow:**
1.  Open and run the `binned_metrics_development.ipynb` notebook.
2.  Follow the steps inside to test the core logic.
3.  If you find a bug or need to make a change, modify the main code block (Cell 3) in the notebook.
4.  Re-run the code block and your test cells until the logic is correct and the final JSON output is perfect.
5.  Once you are satisfied, **copy the entire Python script from Cell 3 of the notebook and paste it into your `binned_metrics.py` file**.

### 2. Final Validation (Python Script)

After developing your logic in the notebook, you can run a final test using the `main()` function in the `binned_metrics.py` script. This validates that the script runs as a standalone file.

1.  **Create Environment:** Create and activate your `venv38` environment (see `requirements.txt` and previous instructions).
2.  **Install Dependencies:** `pip install -r requirements.txt`
3.  **Run Test:**
    ```bash
    python binned_metrics.py
    ```
4.  **Check Output:** This will run the `init()` and `metrics()` functions using the sample data in `synthetic_2021_2024_prediction_data.csv` and the *hardcoded* parameters in `main()`. It will produce the file `universal_performance_2021_2024_metrics_example_output.json`.

#### Using `job_parameters.json` for Better Testing

The default `main()` function in `binned_metrics.py` hardcodes test parameters. For better testing, you can modify `main()` to read parameters from `job_parameters.json` instead.

**To modify `main()` to use this file:**

Replace the *entire* `test_params` block in `main()` with this:

```python
    # ... inside main() ...
    
    # 1. initialize global variables
    print("--- Calling init() ---")
    
    # Load test parameters from the JSON file
    try:
        with open('job_parameters.json', 'r') as f:
            job_params_content = json.load(f)
            
        test_params = {
            "rawJson": json.dumps({
                "jobParameters": job_params_content
            })
        }
        print("Successfully loaded 'job_parameters.json'")
    except Exception as e:
        print(f"Error loading 'job_parameters.json', falling back to defaults. Error: {e}")
        test_params = {"rawJson": "{}"} # Fallback to defaults in init()

    init(test_params)
    print("--- init() complete ---")
    
    # 2. Read local data
    # ... rest of main() ...
```
## 💡Appendix

### Confusion-matrix terms
| &darr; _Model_ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; _Actual_&rarr;  | ACTUALLY TRUE: T_ *(label=1)* | ACTUALLY FALSE: F_ *(label=0)* |
| :--- | :--- | :--- |
| **PREDICTED TRUE : _P** *(score=1)* &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | **True Positive: TP** *(number of cases correctly identified as patient)* | **False Positive: FP** *(number of cases incorrectly identified as patient)* |  
| **PREDICTED FLASE: _N** *(score=0)* &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | **True Negative: TN** (*number of cases correctly identified as healthy)*  | **False Negative: FN** *(number of cases incorrectly identified as healthy)* |

### Performance Metrics
| Statistic | Abbreviation | Formula | Explanation of Additional Data/Statistical Elements Required | Description (Plain Language) | Common Use Cases |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Sensitivity (Recall) | SEN | $\frac{TP}{TP+FN}$ | None. (Eq 5) | "Of all the people who *actually* had the condition, what percentage did our model correctly identify?" | **Medical Diagnostics:** Finding all sick patients (missing one is very bad). **Fraud Detection:** Catching as many fraudulent transactions as possible. |
| Specificity | SP | $\frac{TN}{TN+FP}$ | None. (Eq 6) | "Of all the people who were *actually* healthy, what percentage did our model correctly identify?" | **Spam Filtering:** Ensuring non-spam emails (healthy) are *not* put in the spam folder (a false positive is very annoying). **Medical Screening:** Avoiding telling a healthy person they are sick. |
| Precision | PPV | $\frac{TP}{TP+FP}$ | None. (Eq 7) | "Of all the times our model *predicted* someone had the condition, what percentage was it correct?" | **Search Engines/E-commerce:** The top results *must* be relevant (a bad prediction ruins user trust). **Recommendation Systems:** Suggestions must be high-quality. |
| Accuracy | ACC | $\frac{TP+TN}{TP+TN+FP+FN}$ | None. (Eq 13) | "Overall, what percentage of *all* predictions (both positive and negative) did the model get right?" | **General Classification:** Good for balanced datasets where error costs are equal. **Stakeholder Reporting:** Easiest metric to explain as a starting point. |
| F1 Score | F1 | $2\cdot \frac{\text{Precision}\cdot \text{Recall}}{\text{Precision}+\text{Recall}} \text{ or } \frac{2\cdot TP}{2\cdot TP+FP+FN}$ | Requires Precision and Recall. (Eq 14) | "A single score that balances the trade-off between Sensitivity (finding all positives) and Precision (not making false positive predictions). Best for 'imbalanced' data." | **Imbalanced Data:** The go-to metric when one class is rare, such as **Fraud Detection**, **Rare Disease Diagnosis**, or **System Anomaly Detection**. |
| Negative Predictive Value | NPV | $\frac{TN}{TN+FN}$ | None. (Eq 8) | "Of all the times our model *predicted* someone was healthy, what percentage was it correct?" | **Medical Testing:** If a test comes back negative, this tells you how much to trust that result. **Quality Control:** If a part passes inspection, what's the confidence it's truly non-defective? |
| False Positive Rate | FPR | $\frac{FP}{FP+TN}$ | None. (Eq 10) | "What percentage of *all healthy people* did we incorrectly flag as sick? This is the 'false alarm' rate." | **Spam Filtering:** Measures how often real email is sent to spam. **Security Alerts:** Measures how many alerts are "crying wolf," which can lead to "alert fatigue" in IT operations. |
| False Negative Rate | FNR | $\frac{FN}{FN+TP}$ | None. (Eq 9) | "What percentage of *all sick people* did we incorrectly label as healthy? This is the 'miss' rate." | **Critical Disease Screening:** The single most important metric for a cancer test. **Airport Security:** Measures how many threats are missed. **Cybersecurity:** How many intrusions were not detected. |
| False Discovery Rate | FDR | $\frac{FP}{TP+FP}$ | None. (Note: $1-\text{Precision}$) (Eq 11) | "Of all the 'positive' predictions we made, what percentage were actually wrong?" | **Scientific Research (Genomics, Physics):** When running thousands of tests, this controls the rate of "discoveries" that are actually just random chance. |
| False Omission Rate | FOR | $\frac{FN}{TN+FN}$ | None. (Note: $1-\text{NPV}$) (Eq 12) | "Of all the 'negative' predictions we made, what percentage were actually wrong?" | **Loan Applications:** Of all the people the model rejected, what percentage would have actually been good customers? **AI Fairness Audits:** Used to check for bias in "cleared" or "rejected" groups. |
| Error Rate | ERR | $\frac{FP+FN}{TP+TN+FP+FN}$ | None. (Note: $1-\text{Accuracy}$) | "Overall, what percentage of *all* predictions did the model get wrong?" | **General Classification:** A simple way to express the total number of mistakes. The inverse of Accuracy. |
| Prevalence | PR | $\frac{TP+FN}{TP+TN+FP+FN}$ | None. (Eq 18) | "What percentage of the *entire population* actually has the condition? This isn't a model metric, but it's the most important context." | **Epidemiology:** Understanding the base rate of a disease. **Risk Modeling:** What is the baseline rate of fraud or customer churn before any model is applied? |
| Matthews Correlation Coefficient | MCC | $\frac{TP\cdot TN-FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$ | None. (Eq 15) | "A single, robust score from -1 to +1 that measures prediction quality. +1 is perfect, 0 is random, -1 is perfectly wrong. It's the most reliable single metric for imbalanced data." | **Bioinformatics:** A standard metric for predicting protein structures. **Software Defect Prediction:** Highly preferred over F1/Accuracy for finding rare bugs. |
| Informedness | INF / BM | $\frac{TP}{TP+FN}+\frac{TN}{TN+FP}-1$ | Requires Sensitivity and Specificity. (Eq 16) | "How well does the model *inform* us about the true state, separating the sick from the healthy? It's (How well we find positives) + (How well we find negatives) - 1." | **Medical Diagnostics:** Used to summarize the total discriminative power of a diagnostic test. |
| Markedness | MK / ΔP | $PPV + NPV - 1$ | Requires Precision and NPV. (Eq 17) | "How trustworthy are the model's predictions? It's (How trustworthy our 'positive' predictions are) + (How trustworthy our 'negative' predictions are) - 1." | **Advanced Model Diagnostics:** Used to analyze the *predictive* quality (PPV/NPV) of a model, as a counterpart to Informedness (which measures *discriminative* quality - SEN/SP). |
| Prevalence Threshold | PT | $\frac{\sqrt{SEN \cdot (1-SP)} + SP - 1}{SEN + SP - 1}$ | Requires Sensitivity and Specificity. (Eq 19) | "This isn't a performance metric, but a calculated value. It tells you the prevalence rate at which the model's F1 score or Accuracy becomes no better than random guessing." | **Advanced Model Auditing:** Used to understand a model's 'break-even' point and its theoretical limits, especially in scenarios with very low prevalence. |
| Threat Score | TS / CSI | $\frac{TP}{TP+FN+FP}$ | None. (Eq 20) | "Of all the times the event *either* happened *or* was predicted to happen, what percentage of the time was it correctly predicted? (It ignores correct 'no' predictions)." | **Weather Forecasting:** The standard metric for rare, severe events (hurricanes, tornadoes). A correct "no tornado" prediction is useless; this focuses only on the "threat." |
| Youden's Index | J | $\frac{TP}{TP+FN}+\frac{TN}{TN+FP}-1$ | Requires Sensitivity and Specificity. (Same as Informedness) (Eq 16) | "Same as Informedness. It's (True Positive Rate) - (False Positive Rate). It measures the model's ability to avoid both kinds of errors." | **Medical Epidemiology:** Used to find the 'optimal' threshold on an ROC curve that best separates the sick and healthy populations. |
| Diagnostic Odds Ratio | DOR | $\frac{TP/FP}{FN/TN} \text{ or } \frac{TP\cdot TN}{FP\cdot FN}$ | None. (Eq 21) | "A single number that summarizes the 'odds' of a positive test in a sick person compared to a healthy person. A high number means the test is very good at discriminating." | **Medical Meta-Analysis:** Used by researchers to compare the performance of different diagnostic tests (e.g., Test A vs. Test B) across different studies, as it's independent of prevalence. |
| AUC-ROC / AUC-PR | AUC | N/A (Cannot be calculated from a single matrix) | These require the model's predicted probabilities for all instances, not just the final binary class predictions. | "A single score (0 to 1) that measures the model's performance across *all possible thresholds*. A score of 1.0 is a perfect model, 0.5 is a random guess." | **Model Comparison:** The primary way data scientists compare different models (e.g., "Model A's AUC is 0.92, Model B's is 0.88"). **AUC-ROC** is standard; **AUC-PR** is for highly imbalanced data. |


