
FROM python:3.12-slim

WORKDIR /app

COPY requirements-docker.txt .

RUN pip install --no-cache-dir -r requirements-docker.txt

COPY api/ ./api/
COPY models/ ./models/

ENV MODEL_PATH=models/vehicle_damage_augmented.keras

EXPOSE 8501

CMD ["streamlit", "run", "api/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
