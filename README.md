# Statistics Module Assignment

A reproducible Python statistics assignment exploring a small mobile-phone specifications dataset.

## Project scope

The notebook completes nine learning tasks:

1. Load and inspect the dataset.
2. Convert Bluetooth, Wi-Fi, touchscreen and external-memory flags into binary values, then build a combined logical condition.
3. Calculate total pixel count (pixel height × pixel width) and compare phones against the dataset median.
4. Calculate screen diagonal from screen height and width, then apply the upper-quartile threshold.
5. Explore primary and front camera resolutions using histograms, means and medians.
6. Compare internal memory, battery capacity and RAM against their means.
7. Explore phone depth and weight, using medians to identify thinner and lighter records.
8. Combine all six conditions and sort matching phones by listed price.
9. Compare numerical-feature variability with standard-deviation-to-mean ratios.

## Dataset

- File: `Dataset.csv`
- Shape: 50 rows × 17 columns
- Identifier: `PID`
- Binary features: `Blue`, `Wi_Fi`, `Tch_Scr`, `Ext_Mem`
- Measurements and listed price: pixel dimensions, screen dimensions, camera resolutions, internal memory, battery power, RAM, depth, weight and `Price`.

The notebook loads the repository CSV when it is present and checks it against an embedded fallback copy. This keeps the notebook usable in both a repository checkout and standalone Google Colab. The analysis stops with a clear error if the schema, identifiers, binary values or numeric measurements fail validation.

## Run locally

Python 3.11 is recommended.

```bash
python -m pip install pandas numpy matplotlib seaborn jupyter nbconvert nbformat pytest
pytest -q
jupyter nbconvert --to notebook --execute "Statistics_Module_Assignment_SINGLE_FILE (1).ipynb" --output executed_statistics.ipynb --ExecutePreprocessor.timeout=600
```

Open the notebook in Jupyter or upload it to Google Colab to explore the analysis interactively.

## Repository layout

- `Dataset.csv`: version-controlled dataset.
- `Statistics_Module_Assignment_SINGLE_FILE (1).ipynb`: full analysis and visualisations.
- `tests/test_dataset.py`: dataset schema, identifier, binary-value and numeric-integrity tests.
- `.github/workflows/project-checks.yml`: automated notebook execution and validation.

## Interpretation and limitations

This is a descriptive educational exercise on a small 50-row dataset. The repository does not document an independently verifiable data source, collection method, sampling frame or date. Do not treat the records as representative of the phone market, infer causal relationships, or use the listed prices as current market prices. The standard-deviation-to-mean ratio is a descriptive comparison and is not meaningful for every possible distribution or zero-mean feature.

## Tools

Python, pandas, NumPy, Matplotlib, Seaborn, Jupyter Notebook and Google Colab.
