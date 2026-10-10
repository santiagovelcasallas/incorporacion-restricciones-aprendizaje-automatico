# Experimentos previos

Restricciones post-entrenamiento que probamos antes del caso principal (CCD). Se incluyen porque muestran **cuándo una restricción no ayuda**, que es la otra mitad de la lección. Todo corre en **CPU**.

## Cómo reproducir

```bash
pip install -r requirements.txt
python descargar_datos.py --c17        # PCam (136 MB) + Camelyon17 (390 MB); sin --c17 solo PCam
cd biomedclip_pcam
python embed.py 1000                   # embeddings en 8 orientaciones (~16 min en CPU de 16 hilos; segundos en GPU)
python evaluar.py d4                   # restricción D4
python stain_test.py                   # normalización de tinción (~2 min)
python crop_test.py && python evaluar.py recortes
python evaluar.py transclip
python c17_embed.py && python evaluar.py coral
cd ../llm_permutaciones
python llm_test.py 60                  # ~3,5 min en CPU
python llm_eval.py cache/llm_Qwen2.5-0.5B-Instruct_60.npz
```

Salida esperada de `evaluar.py` (la misma de la sección 5 del [README principal](../README.md#5-experimentos-previos-cuándo-una-restricción-no-ayuda)):

```
d4:        AUC mitad de prueba 0.825 → D4 0.832  IC95 Δ [-0.007 0.021] · mejora en 10/10 particiones
           diagnóstico distinto según el giro: 30.6 % · κ de Fleiss entre orientaciones: 0.742
recortes:  centro 64 px 0.798 → 0.780 · centro 48 px 0.798 → 0.760
transclip: zero-shot 0.810 · TransCLIP 0.765 · Histo-TransCLIP (sin KL) 0.644
coral:     origen 0.951 · hospital nuevo 0.962 · hospital nuevo + CORAL 0.936
```

Las semillas son fijas, así que los números deberían coincidir; pueden variar en el tercer decimal según la versión de PyTorch y el hardware.

## `biomedclip_pcam/`: BiomedCLIP + PatchCamelyon

Modelo: `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` (congelado). Datos: PCam (split *test*, `zacharielegault/PatchCamelyon`), 1000 parches balanceados con semilla fija.

| Archivo | Qué hace |
|---|---|
| `embed.py` | Embeddings de cada parche en sus 8 orientaciones (grupo D4) y de los textos de las clases |
| `evaluar.py` | Evalúa las restricciones D4, recortes, TransCLIP y CORAL (ver arriba) |
| `analyze.py` | Primera evaluación de D4 (zero-shot y sonda lineal) |
| `demo.py` / `demo_biomedclip_pcam.ipynb` | Demo completa con figuras: ROC, PR, IC 95 %, restricción de sensibilidad ≥ 95 % con umbral conformal y mapas de oclusión. `run_local.py` la corre guardando las figuras en `figs/` |
| `stain_test.py` | Restricción de invariancia a la tinción (normalización H&E), con y sin un cambio de tinción simulado |
| `crop_test.py` | Restricción espacial: mirar solo el centro del parche |
| `transclip.py` | Port a numpy de TransCLIP (Zanella et al., NeurIPS 2024) desde el código oficial |
| `c17_embed.py` | Embeddings de Camelyon17-WILDS (hospitales de origen vs. hospital nuevo) |
| `figuras/` | Figuras de la demo (`fig1`–`fig5`) y ejemplos de normalización de tinción |

## `llm_permutaciones/`: invariancia al orden de las opciones en un LLM

`llm_test.py` le pasa a Qwen2.5-0.5B-Instruct 60 preguntas de MMLU *clinical knowledge* en las 4 rotaciones cíclicas de las opciones y guarda las log-probabilidades de A/B/C/D. `llm_eval.py` compara la exactitud en cada orden con la de la restricción (promedio sobre órdenes) y muestra el sesgo por letra.
