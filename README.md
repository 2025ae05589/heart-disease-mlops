# Heart Disease Prediction - MLOps Assignment 1

Course: Machine Learning Operations (AIMLCZG523), BITS Pilani WILP

This project builds a classifier to predict the risk of heart disease from
patient health data (UCI Heart Disease dataset). The trained model will be
served as an API with CI/CD, Docker, Kubernetes and monitoring.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data

Source: [UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)

### What the UCI page says about this data

The dataset has patient records from four hospitals: Cleveland, Hungary,
Switzerland and VA Long Beach. The dataset page gives this information:

> This database contains 76 attributes, but all published experiments refer to
> using a subset of 14 of them. In particular, the Cleveland database is the
> only one that has been used by ML researchers to date.

> The "goal" field refers to the presence of heart disease in the patient. It
> is integer valued from 0 (no presence) to 4. Experiments with the Cleveland
> database have concentrated on simply attempting to distinguish presence
> (values 1,2,3,4) from absence (value 0).

> The names and social security numbers of the patients were recently removed
> from the database, replaced with dummy values.

### How it is used in this project

Based on the above, and on our own check of the files:

- Only the "processed" files with 14 columns are used. These files have no
  patient names or ID numbers.
- The Cleveland file (303 patients) is used for modelling.
- The target is made binary: 0 = no disease, 1 = disease (values 1 to 4).
- The files of the other three hospitals are kept only for a comparison in
  the EDA. They are not used for modelling because most of their values for
  `ca`, `thal` and `slope` are missing.

| Hospital | Patients | Rows with no missing value |
|---|---|---|
| Cleveland | 303 | 297 |
| Hungarian | 294 | 1 |
| Switzerland | 123 | 0 |
| VA Long Beach | 200 | 1 |

### Download

The raw files are already kept in `data/raw/`. To download them again:

```bash
python scripts/download_data.py
```

More steps will be added here as the project grows.
