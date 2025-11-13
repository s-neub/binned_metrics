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