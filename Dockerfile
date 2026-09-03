FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt pytest

COPY reproduce_article_figures.py extended_validation.py pytest.ini ./
COPY tests ./tests

RUN mkdir -p /app/logs /app/results

CMD ["python", "-m", "pytest", "-q"]
