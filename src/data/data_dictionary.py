"""The data dictionary of the patient features: names, meanings and valid
values.

It is used by the training code, the data checks and the API, so that the
same rules apply to the training data and to a new patient. Only facts about
the features are kept here. The training settings are in src/config.py.
"""

# "ca" is a count, but it is treated as categorical (EDA notebook, section 11)
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

# the order in which the model was trained
FEATURES = NUMERIC + CATEGORICAL

# meaning of each feature, from the UCI documentation (heart-disease.names)
DESCRIPTIONS = {
    "age": "age in years",
    "trestbps": "resting blood pressure in mm Hg",
    "chol": "serum cholesterol in mg/dl",
    "thalach": "maximum heart rate achieved",
    "oldpeak": "ST depression induced by exercise relative to rest",
    "sex": "1 = male, 0 = female",
    "cp": "chest pain type (1 to 4)",
    "fbs": "fasting blood sugar > 120 mg/dl (1 = true, 0 = false)",
    "restecg": "resting electrocardiographic results (0 to 2)",
    "exang": "exercise induced angina (1 = yes, 0 = no)",
    "slope": "slope of the peak exercise ST segment (1 to 3)",
    "ca": "number of major vessels (0 to 3) colored by fluoroscopy",
    "thal": "3 = normal, 6 = fixed defect, 7 = reversible defect",
}

# values allowed for the coded features, from the UCI documentation
VALID_CODES = {
    "sex": [0, 1],
    "cp": [1, 2, 3, 4],
    "fbs": [0, 1],
    "restecg": [0, 1, 2],
    "exang": [0, 1],
    "slope": [1, 2, 3],
    "ca": [0, 1, 2, 3],
    "thal": [3, 6, 7],
}

# numerical features that have to be above 0
ABOVE_ZERO = ["age", "trestbps", "chol", "thalach"]
# numerical features that can be 0, but not below
ZERO_OR_ABOVE = ["oldpeak"]

# features that may be empty in a request to the API; these are the ones
# with missing values in the data
MAY_BE_MISSING = ["ca", "thal"]
