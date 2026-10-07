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

### Information given on the UCI page

The dataset contains patient records from four hospitals: Cleveland, Hungary,
Switzerland and VA Long Beach. The following is stated on the dataset page:

> This database contains 76 attributes, but all published experiments refer to
> using a subset of 14 of them. In particular, the Cleveland database is the
> only one that has been used by ML researchers to date.

> The "goal" field refers to the presence of heart disease in the patient. It
> is integer valued from 0 (no presence) to 4. Experiments with the Cleveland
> database have concentrated on simply attempting to distinguish presence
> (values 1,2,3,4) from absence (value 0).

> The names and social security numbers of the patients were recently removed
> from the database, replaced with dummy values.

### Use of the data in this project

The statements above were verified against the files (see
`notebooks/01_data_acquisition_and_eda.ipynb`). The following decisions were
taken:

- Only the 14-column files provided by UCI (named `processed.*.data`) are
  used. These files contain no patient names or ID numbers. The word
  "processed" in the file names refers to the reduction from 76 to 14 columns,
  done before the files were published. In this project these files are the
  raw data.
- The Cleveland file (303 patients) is used for modelling.
- The target is made binary: 0 = no disease, 1 = disease (values 1 to 4).
- The files of the other three hospitals are retained only for a comparison
  in the EDA. They are not used for modelling because most of their values
  for `ca`, `thal` and `slope` are missing.

| Hospital | Patients | Rows with no missing value |
|---|---|---|
| Cleveland | 303 | 297 |
| Hungarian | 294 | 1 |
| Switzerland | 123 | 0 |
| VA Long Beach | 200 | 1 |

### Acknowledgement

The UCI documentation requests that any work using this data names the
principal investigators responsible for the data collection:

- Hungarian Institute of Cardiology, Budapest: Andras Janosi, M.D.
- University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
- University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
- V.A. Medical Center, Long Beach and Cleveland Clinic Foundation:
  Robert Detrano, M.D., Ph.D.

### Download

The raw files are kept in `data/raw/`. They can be downloaded again with:

```bash
python scripts/download_data.py
```

### Cleaning

The cleaned file `data/processed/heart_clean.csv` is produced from the raw
Cleveland file with:

```bash
python src/data_prep.py
```

The six missing values (4 in `ca`, 2 in `thal`) are intentionally left in the
cleaned file. They are filled inside the model pipeline, so that the fill
value is learnt from the training data only.

### Preprocessing

The steps that learn from the data (filling of missing values, one-hot
encoding and scaling) are defined in `src/preprocessing.py` and are used as
the first step of the model pipeline. Their effect on the data is shown in
section 14 of the notebook.

## Data acquisition and exploratory data analysis

The analysis is in `notebooks/01_data_acquisition_and_eda.ipynb`. The download
and cleaning scripts are called from within the notebook, so it can be run on
its own.

Further sections will be added as the project progresses.
