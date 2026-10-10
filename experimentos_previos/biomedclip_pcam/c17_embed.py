"""Camelyon17-WILDS: ¿empeora BiomedCLIP al cambiar de hospital? (para probar la restricción CORAL).

Se calculan embeddings de 400 parches (balanceados) de los hospitales de origen (split id_val) y de
400 del hospital nuevo (split ood_test). Luego se entrena una sonda lineal en el origen y se evalúa en el
hospital nuevo, con y sin alinear la media/covarianza (CORAL). Resultado (README, sección 5): el modelo
NO empeora en el hospital nuevo (AUC 0,951 → 0,962), así que la restricción no tiene nada que corregir.

Requiere: data/c17_idval.parquet y data/c17_ood.parquet (python ../descargar_datos.py --c17). Uso: python c17_embed.py
"""
import io, os, time, numpy as np, torch, pyarrow.parquet as pq, open_clip
from PIL import Image

os.makedirs("cache", exist_ok=True)
torch.set_num_threads(16); M = 400                            # parches por hospital (mitad con tumor)
model, pre = open_clip.create_model_from_pretrained("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
model.eval()
out = {}; t0 = time.time()
for name, f in [("src", "data/c17_idval.parquet"), ("tgt", "data/c17_ood.parquet")]:
    t = pq.read_table(f); lab = np.array(t.column("label").to_pylist()); r = np.random.default_rng(0)
    # muestra balanceada con semilla fija
    idx = np.concatenate([r.choice(np.where(lab == c)[0], M // 2, replace=False) for c in (0, 1)]); r.shuffle(idx)
    col = t.column("image")
    imgs = np.stack([np.array(Image.open(io.BytesIO(col[int(i)].as_py()["bytes"])).convert("RGB")) for i in idx])
    with torch.no_grad():
        E = np.concatenate([torch.nn.functional.normalize(model.encode_image(torch.stack(
            [pre(Image.fromarray(a)) for a in imgs[s:s + 64]])), dim=-1).numpy() for s in range(0, M, 64)])
    out[name] = E; out[name + "_y"] = lab[idx]; out[name + "_img"] = imgs
    print(name, f"{time.time()-t0:.0f}s", flush=True)
np.savez_compressed("cache/c17.npz", **out)
