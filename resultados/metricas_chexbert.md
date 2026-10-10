# Métricas CheXbert: MAIRA-2 con y sin CCD

IU X-Ray (sección *findings*), 30 estudios con ≥ 1 de los 5 hallazgos principales. Micro-promedio. IC 95 % por bootstrap pareado (2000 remuestreos). Salida de la celda C del notebook.

```
5 hallazgos principales  (n=30 estudios con hallazgos)
  Precisión original 1.000 → CCD 0.625   Δ IC95 [-0.667, +0.875]   P(mejora) = 37%
  Recall    original 0.030 → CCD 0.152   Δ IC95 [+0.000, +0.267]   P(mejora) = 95%
  F1        original 0.059 → CCD 0.244   Δ IC95 [-0.011, +0.391]   P(mejora) = 95%
  AUC       original 0.515 → CCD 0.563   Δ IC95 [-0.015, +0.122]   P(mejora) = 93%

14 hallazgos  (n=30 estudios con hallazgos)
  Precisión original 0.031 → CCD 0.161   Δ IC95 [+0.002, +0.269]   P(mejora) = 98%
  Recall    original 0.019 → CCD 0.094   Δ IC95 [+0.000, +0.167]   P(mejora) = 95%
  F1        original 0.024 → CCD 0.119   Δ IC95 [+0.001, +0.205]   P(mejora) = 98%
  AUC       original 0.467 → CCD 0.512   Δ IC95 [+0.003, +0.092]   P(mejora) = 98%
```

Macro-promedio (celda 5):

```
5 hallazgos  macro | Precisión 0.200→0.143 | Recall 0.020→0.050 | F1 0.036→0.074  (ΔF1 IC95 [-0.041, +0.110])
14 hallazgos macro | Precisión 0.071→0.051 | Recall 0.007→0.018 | F1 0.013→0.026  (ΔF1 IC95 [-0.015, +0.039])
```

Matriz de confusión (5 hallazgos × 30 estudios = 150 pares):

| | Original | CCD |
|---|---|---|
| Verdaderos positivos | 1 | 5 |
| Falsos negativos (omitidos) | 32 | 28 |
| Falsos positivos (inventados) | 0 | 3 |
| Verdaderos negativos | 117 | 114 |

Por paciente: mejoró 6 · igual 21 · empeoró 3.

Tiempo (A100): ~1,4 s por informe sin restricción · ~11,5 s con CCD.
