# Pipeline reproducible de optimización con Algoritmos Genéticos

**Inteligencia Computacional — Universidad de San Buenaventura**

Pipeline que aplica Algoritmos Genéticos (AG) a tres subproblemas de optimización sobre el dataset astronómico `sdss_sample.csv` (1000 objetos del Sloan Digital Sky Survey, clasificados como `Galaxy`, `QSO` o `Star`).

---

## 1. Estructura del proyecto

```
ag-astronomia/
├── data/
│   └── sdss_sample.csv          # Dataset (1000 filas, 9 columnas)
├── src/
│   ├── encoding.py              # Codificación cromosómica
│   ├── operators.py             # Operadores evolutivos
│   ├── fitness.py               # Funciones de fitness
│   ├── evaluation.py            # Métricas, historial y gráficas
│   ├── feature_selection.py     # Subproblema 1
│   ├── hyperparameter_tuning.py # Subproblema 2
│   └── clustering.py            # Subproblema 3
├── tests/
│   ├── test_dataset.py          # Integridad del dataset
│   ├── test_encoding.py         # Codificación cromosómica
│   └── test_operators.py        # Operadores y fitness
├── outputs/                     # Métricas, gráficas y logs (generado)
├── main.py                      # Punto de entrada
├── requirements.txt
├── pytest.ini
├── Dockerfile
├── Jenkinsfile
└── README.md
```

La separación en `encoding` / `operators` / `fitness` / `evaluation` es deliberada: el núcleo evolutivo es el mismo para los tres subproblemas y solo cambian la representación y la función objetivo. Esto evita duplicar el bucle genético tres veces.

---

## 2. Ejecución

### Local

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

pytest tests/ -v                  # 42 pruebas
python main.py
```

Opciones: `python main.py --data data/sdss_sample.csv --outputs outputs --seed 42`

### Docker

```bash
docker build -t ag-astronomia .
docker run --rm -v "$(pwd)/outputs:/app/outputs" ag-astronomia
```

El volumen es necesario para que las gráficas y métricas queden en tu máquina; sin él se pierden al terminar el contenedor.

Solo las pruebas: `docker run --rm ag-astronomia pytest -v`

### Jenkins

Crear un *Pipeline* apuntando al repositorio con *Script Path* = `Jenkinsfile`. Etapas: Checkout → Instalación de dependencias → Pruebas básicas → Ejecución del pipeline → Verificación de resultados → Build de la imagen Docker. Los artefactos de `outputs/` se archivan en cada build.

> La etapa de Docker requiere que el agente de Jenkins tenga el demonio de Docker disponible. Si no lo tiene, comentar esa etapa.

---

## 3. Diseño de los tres Algoritmos Genéticos

| | **1. Selección de características** | **2. Hiperparámetros** | **3. Clustering** |
|---|---|---|---|
| **Variables** | u, g, r, i, z, redshift → `class` | u, g, r, i, z → `redshift` | u, g, r, i, z |
| **Codificación** | Binaria, 6 genes | Real, 1 gen (k ∈ [1,50]) | Real, 15 genes (3 centroides × 5 dim.) |
| **Fitness** | Accuracy KNN k=5 (70/30) | 1 / MSE | 1 / SSE |
| **Selección** | Torneo (t=3) | Ruleta | Torneo (t=3) |
| **Cruce** | Un punto (0.9) | Aritmético (0.9) | Aritmético (0.9) |
| **Mutación** | Bit-flip (0.1) | Gaussiana (0.2, σ=2.0) | Gaussiana (0.25, σ: 1.0 → 0.02) |
| **Elitismo** | 2 individuos | 1 individuo | 2 individuos |
| **Población × generaciones** | 20 × 15 | 20 × 20 | 60 × 150 |

### Justificación de los parámetros evolutivos

**Por qué torneo en 1 y 3, ruleta en 2.** El torneo es indiferente a la escala del fitness. En el subproblema 1 el fitness es una accuracy entre 0.95 y 0.99: con ruleta las probabilidades de selección serían casi idénticas y la presión selectiva se anularía. En el subproblema 2 el fitness (1/MSE) sí tiene rango amplio, así que la ruleta funciona y cumple el requisito de la actividad de usar ruleta o torneo.

**Por qué σ adaptativo en el clustering.** Con 15 genes, una σ fija obliga a elegir entre explorar (σ grande, nunca refina) o refinar (σ pequeña, nunca escapa del óptimo local). El decaimiento lineal de 1.0 a 0.02 explora al principio y afina al final. Se ve en la curva: la diversidad baja de 1.5 a casi 0 mientras el fitness promedio se acerca al del mejor individuo.

**Por qué inicialización híbrida en el clustering.** El 50% de la población inicial se genera muestreando 3 objetos reales del dataset como centroides. Con inicialización uniforme en las 15 dimensiones, casi todos los centroides iniciales caen en zonas vacías del espacio y el AG converge a un SSE ~47% peor que KMeans. Con la siembra, el AG llega a 0.22% de KMeans. El otro 50% sigue siendo aleatorio para no perder exploración.

**Por qué el elitismo.** Sin él, el cruce y la mutación pueden destruir el mejor individuo de una generación. Con elitismo la curva del mejor individuo es monótona no decreciente, que es exactamente lo que se observa en las tres curvas de convergencia.

---

## 4. Resultados

Ejecución con `seed=42` (reproducible). Tiempo total: ~4 s.

### 4.1 Selección de características

| | |
|---|---|
| Mejor cromosoma | `[0, 1, 1, 1, 1, 1]` |
| Variables seleccionadas | g, r, i, z, redshift (5 de 6) |
| Accuracy | **0.9933** |
| Accuracy con las 6 variables | 0.9933 |

Matriz de confusión (300 muestras de test):

| Real \ Predicho | Galaxy | QSO | Star |
|---|---|---|---|
| **Galaxy** | 105 | 0 | 0 |
| **QSO** | 2 | 88 | 0 |
| **Star** | 0 | 0 | 105 |

**Interpretación.** El AG descarta la magnitud `u` sin perder ni un punto de accuracy: el mismo rendimiento con una variable menos, que es justamente el objetivo de la selección de características. El AG converge en la generación 1 porque el problema es fácil (solo 2⁶ = 64 combinaciones posibles y varias alcanzan el óptimo); la diversidad se mantiene alta porque muchos subconjuntos distintos dan el mismo fitness, así que no hay presión para converger a uno solo. Los 2 únicos errores son QSO clasificados como Galaxy, lo cual tiene sentido físico: los cuásares son núcleos galácticos activos y sus magnitudes se solapan con las de las galaxias. Las estrellas se separan perfectamente.

### 4.2 Hiperparámetros

| | AG | Línea base (k=5) |
|---|---|---|
| k | **37** | 5 |
| MSE | **0.1468** | 0.1698 |
| R² | **0.7616** | 0.7242 |

**Interpretación.** El AG mejora el MSE un 13.5% frente al k por defecto. Un k alto (37 vecinos sobre 700 muestras de entrenamiento) indica que el redshift es una función suave de las magnitudes con ruido considerable: promediar muchos vecinos reduce la varianza. El R² de 0.76 dice que las cinco magnitudes explican el 76% de la varianza del redshift, resultado razonable para *photometric redshift* con solo cinco bandas. La curva de convergencia es casi plana desde la generación 0 (el fitness inicial ya es 6.77 de 6.81 final) porque el espacio de búsqueda es unidimensional y la población inicial de 20 individuos ya lo cubre bien; el AG aquí aporta poco frente a una búsqueda exhaustiva, y eso es un hallazgo legítimo que vale la pena decir en la sustentación.

### 4.3 Clustering

| | AG | KMeans |
|---|---|---|
| SSE | **545.92** | 544.72 |
| Silhouette | 0.5338 | 0.5337 |
| ARI vs. clases reales | 0.9433 | 0.9488 |
| Tamaños de cluster | 298 / 352 / 350 | 352 / 350 / 298 |

**Interpretación.** El AG llega a un SSE 0.22% mayor que KMeans, prácticamente la misma solución: los tamaños de cluster coinciden (298/350/352, solo permutados) y el silhouette es idéntico hasta el cuarto decimal. Esto valida la implementación: un AG bien configurado alcanza el óptimo de un algoritmo especializado, aunque necesita 150 generaciones × 60 individuos = 9000 evaluaciones frente a las ~30 iteraciones de KMeans. La ventaja del AG no es la velocidad sino la flexibilidad: permitiría optimizar funciones objetivo no diferenciables donde KMeans no aplica.

El ARI de 0.94 frente a las clases reales indica que los clusters recuperan casi exactamente las tres clases astronómicas **sin haberlas visto**. En la proyección PCA (PC1 explica el 96.2% de la varianza) se ve por qué: las estrellas forman un grupo completamente separado a la izquierda, mientras que galaxias y cuásares se solapan en la frontera, que es donde se concentran las discrepancias — el mismo solapamiento Galaxy/QSO que apareció en la matriz de confusión del subproblema 1.

---

## 5. Archivos generados en `outputs/`

| Archivo | Contenido |
|---|---|
| `metrics.json` | Todas las métricas, parámetros y mejores cromosomas |
| `fs_fitness_curve.png` | Convergencia — selección de características |
| `fs_confusion_matrix.png` | Matriz de confusión del mejor subconjunto |
| `ht_fitness_curve.png` | Convergencia — hiperparámetros |
| `cl_fitness_curve.png` | Convergencia — clustering |
| `clustering_comparison.png` | AG vs. KMeans vs. clases reales (PCA 2D) |
| `fs/ht/cl_fitness_history.csv` | Fitness mejor, promedio y diversidad por generación |

Las curvas incluyen el **fitness promedio** y la **diversidad poblacional** además del mejor individuo, porque la distancia entre la línea del mejor y la del promedio es el indicador directo de si la población está convergiendo o estancada.

---

## 6. Reproducibilidad

Todo el azar pasa por un único `numpy.random.default_rng(seed)` por módulo, propagado explícitamente a cada operador. Las dependencias están fijadas por versión en `requirements.txt`. Dos ejecuciones con la misma semilla dan resultados idénticos.
