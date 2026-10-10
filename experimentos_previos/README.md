# Experimentos previos

Restricciones post-entrenamiento que probamos antes del caso principal (CCD). Se incluyen porque muestran **cuándo una restricción no ayuda**, que es la otra mitad de la lección.

## `biomedclip_pcam/`: BiomedCLIP + PatchCamelyon

Modelo: `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` (congelado). Datos: PCam (split *test*, parquet de `zacharielegault/PatchCamelyon`), 1000 parches balanceados.

| Archivo | Qué hace |
|---|---|
| `embed.py` | Embeddings de cada parche en sus 8 orientaciones (grupo D4) |
| `analyze.py` | Restricción D4 (promedio sobre el grupo): AUC, recall, precisión, tasa de diagnósticos que cambian al girar |
| `demo.py` / `demo_biomedclip_pcam.ipynb` | Demo completa con figuras (ROC, PR, IC 95 %, restricción de sensibilidad ≥ 95 % con umbral conformal, oclusión) |
| `stain_test.py` | Restricción de invariancia a la tinción (normalización H&E) |
| `crop_test.py` | Restricción espacial: mirar solo el centro del parche |
| `transclip.py` | Port a numpy de TransCLIP (Zanella et al., NeurIPS 2024) desde el código oficial |
| `c17_embed.py` | Embeddings de Camelyon17-WILDS (hospitales de origen vs. hospital nuevo) para probar CORAL |

Resultados en la sección 5 del [README principal](../README.md#5-experimentos-previos-cuándo-una-restricción-no-ayuda).

## `llm_permutaciones/`: invariancia al orden de las opciones en un LLM

`llm_test.py` le pasa a Qwen2.5-0.5B-Instruct 60 preguntas de MMLU *clinical knowledge* en las 4 rotaciones cíclicas de las opciones y guarda las log-probabilidades de A/B/C/D. `llm_eval.py` compara la exactitud del orden original con la de la restricción (promedio sobre órdenes). Corre en CPU en ~3,5 min.

```bash
python llm_test.py 60
python llm_eval.py cache/llm_Qwen2.5-0.5B-Instruct_60.npz
```
