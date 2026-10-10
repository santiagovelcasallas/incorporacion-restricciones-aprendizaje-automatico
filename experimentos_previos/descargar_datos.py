"""Descarga los datos públicos de los experimentos previos en biomedclip_pcam/data/.

- PatchCamelyon, split test (136 MB)                   → data/test.parquet      (embed.py, demo.py)
- Camelyon17-WILDS, id_val (107 MB) y ood_test (283 MB) → data/c17_*.parquet     (c17_embed.py; opcional)

Uso:  python descargar_datos.py            # solo PCam
      python descargar_datos.py --c17      # PCam + Camelyon17
"""
import os, shutil, sys
from huggingface_hub import hf_hub_download

DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "biomedclip_pcam", "data")
os.makedirs(DST, exist_ok=True)
ARCHIVOS = [("zacharielegault/PatchCamelyon", "data/test-00000-of-00001.parquet", "test.parquet")]
if "--c17" in sys.argv:
    ARCHIVOS += [("jxie/camelyon17", "data/id_val-00000-of-00001-2016d95c986aa53e.parquet", "c17_idval.parquet"),
                 ("jxie/camelyon17", "data/ood_test-00000-of-00001-4e2cffbdce8ec122.parquet", "c17_ood.parquet")]

for repo, ruta, nombre in ARCHIVOS:
    destino = os.path.join(DST, nombre)
    if os.path.exists(destino):
        print("ya existe:", destino); continue
    shutil.copy(hf_hub_download(repo, ruta, repo_type="dataset"), destino)
    print("descargado:", destino)
