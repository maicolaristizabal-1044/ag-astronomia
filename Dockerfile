FROM python:3.11-slim

LABEL description="Pipeline de optimización con Algoritmos Genéticos sobre datos SDSS"

# Sin archivos .pyc y con salida sin buffer, para ver los logs en tiempo real
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

# Las dependencias se copian primero para aprovechar la caché de capas:
# si solo cambia el código, no se reinstalan los paquetes.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY tests/ ./tests/
COPY data/ ./data/
COPY main.py pytest.ini ./

RUN mkdir -p outputs

# Ejecuta el script principal automáticamente al levantar el contenedor.
# Para correr solo las pruebas:
#   docker run --rm ag-astronomia pytest -v
CMD ["python", "main.py", "--data", "data/sdss_sample.csv", "--outputs", "outputs"]
