"""Restricción espacial post-entrenamiento: obligar a BiomedCLIP a mirar solo el centro del parche.

En PCam la etiqueta depende únicamente del cuadro central de 32×32 px (Veeling et al., 2018), pero el
modelo ve los 96×96 px completos. Aquí se recorta el centro (64 y 48 px) y se calculan sus embeddings.
La evaluación (AUC zero-shot) está en la sección 5 del README: la restricción EMPEORA (0,798 → 0,780 / 0,760),
porque al ampliar un recorte pequeño a 224 px el modelo pierde detalle y contexto.

Requiere: cache/emb.npz (generado por embed.py). Uso: python crop_test.py
"""
import time, numpy as np, torch, open_clip
from PIL import Image

torch.set_num_threads(16)
d = np.load("cache/emb.npz"); imgs = d["imgs"][:300]          # 300 parches: prueba rápida en CPU (~1 min)
model, pre = open_clip.create_model_from_pretrained("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
model.eval()


@torch.no_grad()
def enc(arrs):
    """Embeddings normalizados de una lista de imágenes (en lotes de 64)."""
    out = []
    for s in range(0, len(arrs), 64):
        x = torch.stack([pre(Image.fromarray(np.ascontiguousarray(a))) for a in arrs[s:s + 64]])
        out.append(torch.nn.functional.normalize(model.encode_image(x), dim=-1).numpy())
    return np.concatenate(out)


t0 = time.time(); res = {}
for c in (64, 48):                                            # tamaño del recorte central, en píxeles
    o = (96 - c) // 2
    res[f"c{c}"] = enc([im[o:o + c, o:o + c] for im in imgs])
    print(c, f"{time.time()-t0:.0f}s", flush=True)
np.savez("cache/crops.npz", **res)
