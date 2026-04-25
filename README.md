# Signature Detection and Verification

Final project for the Introduction to Computer Vision course.

## Features

- Signature Detection: detect, draw a bounding box, and crop the signature region from a document image.
- Signature Verification: compare two signature images and classify them as `genuine` or `forged`.

## Environment Requirements

- Python 3.10 or newer.
- Anaconda or Miniconda is recommended.
- Required packages are listed in `requirements.txt`.

## Run With Anaconda

Open Anaconda Prompt or a terminal in the project folder, then run:

```bash
conda create -n signature-pipeline python=3.11 -y
conda activate signature-pipeline
pip install -r requirements.txt
python run.py
```
## Run With Standard Python

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

## How To Use The Application

1. Open the web application in the browser.
2. Select the `Detection` tab to upload a document image and crop the detected signature region.
3. Select the `Verification` tab to upload two signature images for comparison.
4. Click the run button.
5. Review the prediction, score, threshold, baseline score, and evaluation metrics.

## Dataset Evaluation

Create a CSV file with three columns:

```csv
signature_a,signature_b,label
genuine/001_01.png,genuine/001_02.png,1
genuine/001_01.png,forged/001_f01.png,0
```

Label format:

- `label = 1`: both signatures belong to the same writer.
- `label = 0`: the pair is forged or belongs to different writers.

Run evaluation:

```bash
python tools/evaluate_pairs.py --pairs pairs.csv --image-root path/to/dataset --output docs/requirement4_results.json
```

The output JSON contains accuracy, precision, recall, and F1-score for both the baseline model and the improved model.


The internal benchmark output used in the report is available at:

```text
docs/sanity_benchmark_results.json
```
