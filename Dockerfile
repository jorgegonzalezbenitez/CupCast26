# ── Imagen base ───────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Metadatos
LABEL maintainer="CupCast26"
LABEL description="Predictor del Mundial 2026 — Elo + FIFA + Montecarlo"

# ── Variables de entorno ──────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

# ── Directorio de trabajo ─────────────────────────────────────────────────────
WORKDIR /app

# ── Dependencias del sistema ──────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Dependencias Python ───────────────────────────────────────────────────────
# Copiar solo requirements primero para aprovechar la caché de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Código fuente ─────────────────────────────────────────────────────────────
COPY src/       ./src/
COPY api/       ./api/
COPY web/       ./web/
COPY data/      ./data/

# ── Carpeta de outputs (gráficos) ─────────────────────────────────────────────
RUN mkdir -p /app/outputs /app/data/raw /app/data/processed

# ── Puerto expuesto ───────────────────────────────────────────────────────────
EXPOSE 8000

# ── Healthcheck ───────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/status || exit 1

# ── Comando de arranque ───────────────────────────────────────────────────────
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]