"""The shape of the requests and the replies of the API.

The request for one patient is built from the data dictionary
(src/data/data_dictionary.py), so the API applies the same rules as the data
checks: valid codes for the coded features, and no value of 0 or below for
the numerical ones. FastAPI refuses a request that does not follow them.

Used by api/main.py.
"""
from typing import Annotated, Any, Literal, Optional

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, create_model

from src.data.data_dictionary import (
    ABOVE_ZERO,
    DESCRIPTIONS,
    FEATURES,
    MAY_BE_MISSING,
    VALID_CODES,
    ZERO_OR_ABOVE,
)

# one patient of the training data, shown as the example on the /docs page
EXAMPLE_PATIENT = {
    "age": 48,
    "trestbps": 124,
    "chol": 274,
    "thalach": 166,
    "oldpeak": 0.5,
    "sex": 1,
    "cp": 4,
    "fbs": 0,
    "restecg": 2,
    "exang": 0,
    "slope": 2,
    "ca": 0,
    "thal": 7,
}


def code_check(name):
    """Return a function that refuses a value which is not a valid code of
    this feature."""
    allowed = VALID_CODES[name]

    def check(value):
        if value not in allowed:
            listed = ", ".join(str(code) for code in allowed)
            raise ValueError(f"should be one of {listed}")
        return value

    return check


def feature_type(name):
    """Return the type of one feature for the request: a number with its
    check. A feature in MAY_BE_MISSING may also be null.

    strict=True accepts only a number (1 or 1.0). Without it, text such as
    "1" and the value true would also be taken as the number 1.
    """
    if name in VALID_CODES:
        # "enum" shows the allowed codes on the /docs page
        codes = Field(strict=True, json_schema_extra={"enum": VALID_CODES[name]})
        kind = Annotated[float, codes, AfterValidator(code_check(name))]
    elif name in ABOVE_ZERO:
        kind = Annotated[float, Field(strict=True, gt=0)]
    elif name in ZERO_OR_ABOVE:
        kind = Annotated[float, Field(strict=True, ge=0)]
    else:
        raise ValueError(f"no rule in the data dictionary for {name}")

    if name in MAY_BE_MISSING:
        return Optional[kind]
    return kind


def build_patient():
    """Return the request class for one patient, with one field for each
    feature of the data dictionary. Every field has to be present, and a
    field that is not known is refused."""
    fields = {}
    for name in FEATURES:
        # "..." means the field has no default value, so it has to be given
        fields[name] = (feature_type(name), Field(..., description=DESCRIPTIONS[name]))

    return create_model(
        "Patient",
        __config__=ConfigDict(extra="forbid", json_schema_extra={"example": EXAMPLE_PATIENT}),
        **fields,
    )


Patient = build_patient()


# largest number of patients in one batch request; a larger request is
# refused, so that one request cannot keep the service busy for long
MAX_BATCH_SIZE = 100


class BatchRequest(BaseModel):
    """A list of 1 to MAX_BATCH_SIZE patients. Each row is checked on its own,
    so one wrong row does not stop the others."""
    patients: list[Any] = Field(
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description=f"one entry for each patient (at most {MAX_BATCH_SIZE}), "
                    "with the same fields as in /predict",
        json_schema_extra={"example": [EXAMPLE_PATIENT]},
    )


class Prediction(BaseModel):
    prediction: int = Field(description="1 = disease, 0 = no disease")
    label: str = Field(description="the prediction in words")
    confidence: float = Field(description="probability of the predicted class, 0.5 to 1")
    filled_fields: list[str] = Field(
        description="features that came as null and were filled by the model"
    )


class PredictResponse(Prediction):
    model_version: str


class RowResult(BaseModel):
    """The result for one row of a batch: a prediction, or the errors."""
    row: int = Field(description="position of the patient in the request, from 1")
    status: Literal["ok", "error"]
    prediction: Optional[int] = None
    label: Optional[str] = None
    confidence: Optional[float] = None
    filled_fields: Optional[list[str]] = None
    errors: Optional[list[str]] = None


class BatchResponse(BaseModel):
    results: list[RowResult]
    count: int = Field(description="number of rows in the request")
    predicted: int = Field(description="rows that got a prediction")
    failed: int = Field(description="rows that were refused")
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    run_id: str = Field(description="MLflow run that trained the model")
    trained_at: str = Field(description="time the model was trained (UTC)")
