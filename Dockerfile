FROM python:3.11-slim

WORKDIR /app

COPY requirements/runtime.txt ./requirements/runtime.txt

RUN pip install --no-cache-dir -r requirements/runtime.txt

COPY src ./src
COPY models ./models

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]