# Image of the prediction API with the champion model inside.
#
# The model folder models/champion/ is created by the training pipeline (step 6),
# so the pipeline has to be run before the image is built.
#
#   docker build -t heart-disease-api .
#   docker run --rm -p 8000:8000 heart-disease-api

FROM python:3.12-slim
WORKDIR /app

# packages used in training, as recorded by MLflow with the model
COPY models/champion/requirements.txt model-requirements.txt
RUN pip install --no-cache-dir -r model-requirements.txt

# packages of the API
COPY api/requirements.txt api-requirements.txt
RUN pip install --no-cache-dir -r api-requirements.txt

# only the data dictionary is copied from src
COPY src/__init__.py src/
COPY src/data/__init__.py src/data/data_dictionary.py src/data/
COPY api/ api/
COPY models/champion/ models/champion/

# the service runs as a non-root user
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
# uvicorn's access log is disabled, as the API writes its own log line for
# each request
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
