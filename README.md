# Predicción de fallas en camiones industriales

Mantenimiento predictivo para flotas de camiones industriales: detectar fallas antes de
que ocurran usando telemetría y datos de sensores, en lugar de mantenimiento por horas de
servicio o correctivo.

## Problema

En una flota industrial una falla no programada cuesta tres veces: la reparación, el
tiempo muerto del camión y el incumplimiento con el cliente. El mantenimiento por
calendario gasta repuestos que todavía servían; el correctivo siempre llega tarde.
En medio queda una ventana: los datos del camión avisan antes de que falle.

## Enfoque

1. **Ingesta** — telemetría del camión (horas de motor, temperatura, presión de aceite,
   RPM, consumo, códigos de diagnóstico) y registro de órdenes de trabajo.
2. **Etiquetado** — unir telemetría con las fallas reales reportadas en mantenimiento.
   Sin esto no hay aprendizaje supervisado honesto.
3. **Features** — ventanas móviles, tendencias y desviaciones contra la línea base de
   cada unidad; el mismo valor significa cosas distintas para dos camiones.
4. **Modelo** — clasificación de riesgo por componente (motor, frenos, transmisión,
   hidráulico) con horizonte definido, no una probabilidad suelta.
5. **Salida** — alerta priorizada para el jefe de taller, con la señal que la disparó.
   Una alerta que no se puede explicar no se atiende.

El punto difícil no es el modelo: es el etiquetado y la calidad de los datos. Un proyecto
de este tipo se cae por eso, no por el algoritmo.

## Estado

Repositorio recién creado. Todavía no hay datos, pipeline ni modelo. Este README define
el alcance; la estructura se irá armando sobre trabajo real, no como esqueleto vacío.

## Estructura prevista

```
data/          # crudo y procesado (no versionado)
notebooks/     # exploración
src/           # ingesta, features, entrenamiento, evaluación
models/        # artefactos entrenados
reports/       # métricas y resultados
```

## Requisitos

Python 3.11+, pandas, scikit-learn. Se agregarán dependencias reales cuando exista código
que las use.

## Licencia

Sin definir.
