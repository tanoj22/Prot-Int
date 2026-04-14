FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt
COPY app/ ./app/
COPY src/ ./src/
COPY models/best_residue_model.pt ./models/best_residue_model.pt
ENV TRANSFORMERS_CACHE=/app/cache
ENV HF_HOME=/app/cache
RUN mkdir -p /app/cache
RUN python -c "from transformers import AutoModel, AutoTokenizer; \
    AutoModel.from_pretrained('facebook/esm2_t33_650M_UR50D'); \
    AutoTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D')"
EXPOSE 7860
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "7860"]
