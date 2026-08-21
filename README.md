# Financial Time Series Forecasting with Deep Learning

> **AVISO LEGAL**: Este repositorio tiene fines exclusivamente académicos y de investigación. Ninguna sección del código o documentación constituye una recomendación de inversión o asesoramiento financiero profesional.


---

## Descripción del Proyecto

Repositorio oficial del Trabajo de Fin de Máster:  
**"Comparación de modelos para la predicción de series temporales financieras: ARIMA, LSTM/GRU, TCN y Transformers"**.

El proyecto implementa un pipeline de aprendizaje profundo y finanzas cuantitativas diseñado para evaluar la capacidad predictiva y la rentabilidad económica ajustada al riesgo sobre el índice **S&P 500 (^GSPC)**.

---

## Metodología Cuantitativa

### 1. Formulación del Target Relativo
Para evitar predecir el drift alcista natural del índice, el modelo clasifica el exceso de retorno frente a la mediana histórica móvil:
$$Target_t = \mathbb{I}\left(\ln\left(\frac{P_{t+K}}{P_t}\right) - \text{Median}_{60}(\text{Ret}_5) > 0\right)$$
donde $K = 5$ días de negociación.

### 2. Purged Walk-Forward Cross-Validation
* **Esquema:** *Expanding or Rolling Window* (ventana de entrenamiento acumulativa o deslizante).
* **Purging Gap ($K=5$):** Eliminación de solapamiento temporal entre bloques de entrenamiento, validación y test para evitar *Data Leakage*.
* **Evaluación Fuera de Muestra:** Pliegues de test ciego cubriendo periodos de estrés de mercado (COVID-19 en 2020, ciclo de tipos/inflación en 2022).

### 3. Modelo de Decisión Financiera
* **Filtro de Régimen Macro:** Detección de tendencia secular mediante la media móvil de 200 sesiones ($\text{SMA}_{200}$).
* **Asignación Asimétrica de Capital ($w_t$):** Modulación de la exposición entre 50% y 100% en regímenes alcistas (*Bull*) y repliegue a liquidez o cobertura defensiva (0% a 50%) en regímenes bajistas (*Bear*).
* **Fricción Operativa:** Descuento de costes de transacción continuos (5 bps por rotación).

---


## Estructura del Repositorio

```
financial-deep-learning-tfm/
│
├── config.py             # Hiperparámetros globales (semillas, K, ventanas, costes)
├── data_loader.py            # Descarga, alineación macro (VIX, TNX, IRX) e ingeniería de features
├── models.py                 # Definición de PyTorch: LSTM, GRU, TCN, Transformer
├── validation.py             # Motor Purged Walk-Forward CV y evaluación del rendimiento del modelo
├── backtesting.py            # Asignación de capital híbrida, métricas y multi-benchmark
├── main.py                   # Script de orquestación y ejecución comparativa
│
├── data/                     # Caché local de series históricas en CSV
└── results/                  # Guardado automático de resultados
    ├── figures/              # Curvas ROC, Precision-Recall y Curvas de Riqueza (PNG/PDF)
    ├── metrics/              # Resúmenes tabulares (CSV) y snapshots completos (JSON)
    └── models/               # Checkpoints de pesos por pliegue (.pt)
