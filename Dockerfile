FROM python:3.10-slim

WORKDIR /app

COPY supply_chain_forecast/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY supply_chain_forecast/ .

EXPOSE 8000 8501

CMD ["bash", "-c", "uvicorn api.main:app --host 0.0.0.0 --port 8000 & streamlit run dashboard/app.py --server.port 8501"]