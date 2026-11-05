# Evaluación 4 — Equipo 9: fallas APS en camiones Scania

**Docente:** Daniel Alexis Nieto Mora · **Entrega:** 13 de noviembre de 2026.
**Dataset:** [APS Failure at Scania Trucks (UCI)](https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks).

## Alcance académico
- **Supervisado:** clasificar falla relacionada con APS (`pos`) frente a falla no relacionada con APS (`neg`) utilizando regresión logística y Random Forest.
- **No supervisado:** identificar patrones con K-Means y anomalías con Isolation Forest, sin utilizar `class` para entrenar ni seleccionar características.
- **Evaluación:** Recall, Precision, F1, PR-AUC, matriz de confusión; Silhouette (para clustering), perfiles y correspondencia posterior con las etiquetas.
- Interpretar el desbalance, faltantes, costo de falsos positivos/falsos negativos, limitaciones y recomendaciones preventivas.

**Importante:** `neg` NO garantiza un camión sano; representa fallas ajenas al componente APS estudiado. Las 170 características están anonimizadas. No inventar sus significados físicos.

## Primeros pasos
Desde esta carpeta, con Python 3.11+:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python src/download_data.py
python src/run_analysis.py
```

El script descarga los CSV oficiales en `data/raw/`, separados en entrenamiento y prueba. Los datos y artefactos generados no se versionan. Se crea `results/` con tablas y gráficas. No se usa GitHub Actions.

## Estructura
```text
evaluacion 4/
  src/download_data.py       # Descarga y lectura del formato UCI
  src/run_analysis.py        # EDA + 2 supervisados + 2 no supervisados
  requirements.txt
  .gitignore
  README.md
  data/raw/                  # Local; no subir CSV grandes
  results/                   # Gráficas y métricas generadas
```

## Precauciones metodológicas
1. Separar un conjunto de validación a partir del entrenamiento, preservando clases.
2. Aprender imputación, selección de características y escalado **solo** de entrenamiento; usar pipelines.
3. Mantener el test oficial independiente para la evaluación final.
4. No usar `class` como entrada para los métodos no supervisados; usarla después para interpretar.
5. Las decisiones sobre umbral o hiperparámetros se toman en validación, nunca en test.
6. Los modelos incluidos son una base reproducible, no resultados medidos hasta ejecutarlos con el dataset.

## Pendiente de completar en equipo
- Integrantes y reparto de aportes, commits individuales.
- Interpretar resultados reales y elaborar conclusiones.
- Discusión comparativa (8 preguntas de la guía), recomendaciones de mantenimiento y video de máximo 12 minutos.
- Verificar procedencia y términos de redistribución antes de publicar datos.
