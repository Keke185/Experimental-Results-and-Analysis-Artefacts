FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir scikit-learn
COPY fit_and_save_pipeline.py .
RUN python fit_and_save_pipeline.py
COPY main.py .
ENV ARTIFACT_PATH=/data/distribution.pkl
ENV MODEL_PATH=sentence-transformers/all-MiniLM-L6-v2
ENV NODE_ROLE=edge
ENV NEW_LOGIC_PATH=/app/new_logic_pipeline.pkl
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
