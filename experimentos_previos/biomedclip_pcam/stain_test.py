"""Prueba rápida: restricción de invariancia a la tinción (post-entrenamiento).
Restricción = normalizar la tinción de cada parche a una referencia común en espacio H&E
(proyección: se elimina la dirección "cuánto hematoxilina/eosina puso ESTE laboratorio").
Se evalúa en datos originales y con un corrimiento de tinción simulado (otro hospital)."""
import time, numpy as np, torch, open_clip
from PIL import Image
from sklearn.metrics import roc_auc_score
torch.set_num_threads(16)
M = 400
d = np.load("cache/emb.npz"); y = d["labels"][:M]; imgs = d["imgs"][:M]; emb0 = d["emb"][:M, 0]
norm = lambda a: a / np.linalg.norm(a, axis=-1, keepdims=True)
w = norm(d["txt1"].mean(0)) - norm(d["txt0"].mean(0))

# Ruifrok & Johnston: matriz de tinciones H, E, DAB
RGB_FROM_HED = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]])
HED_FROM_RGB = np.linalg.inv(RGB_FROM_HED)
to_hed = lambda im: (-np.log(np.maximum(im.astype(np.float64), 1) / 255.0)) @ HED_FROM_RGB
to_rgb = lambda h: np.clip(255 * np.exp(-(h @ RGB_FROM_HED)), 0, 255).astype(np.uint8)

ref_hed = np.stack([to_hed(im).reshape(-1, 3) for im in imgs])
REF_MU, REF_SD = ref_hed.mean((0, 1)), ref_hed.std(1).mean(0)


def normalize_stain(im):  # la restricción
    h = to_hed(im).reshape(-1, 3); mu, sd = h.mean(0), h.std(0) + 1e-6
    return to_rgb(((h - mu) / sd * REF_SD + REF_MU).reshape(im.shape))


rng = np.random.default_rng(1)
def shift(im):  # "otro hospital": más/menos hematoxilina y eosina
    h = to_hed(im); a = rng.uniform(0.6, 1.4, 3); b = rng.uniform(-0.05, 0.05, 3); a[2], b[2] = 1, 0
    return to_rgb(h * a + b)


model, pre = open_clip.create_model_from_pretrained("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
model.eval()
@torch.no_grad()
def enc(arrs):
    out = []
    for s in range(0, len(arrs), 64):
        x = torch.stack([pre(Image.fromarray(a)) for a in arrs[s:s + 64]])
        out.append(torch.nn.functional.normalize(model.encode_image(x), dim=-1).numpy())
    return np.concatenate(out)


t0 = time.time()
shifted = [shift(im) for im in imgs]
S = {"original": emb0 @ w,
     "original + restricción tinción": enc([normalize_stain(im) for im in imgs]) @ w}
print(f"{time.time()-t0:.0f}s", flush=True)
S["otro hospital"] = enc(shifted) @ w
S["otro hospital + restricción tinción"] = enc([normalize_stain(im) for im in shifted]) @ w
print(f"{time.time()-t0:.0f}s")


def ci(a, b, n=2000):
    r = np.random.default_rng(0); out = []
    for _ in range(n):
        i = r.integers(0, M, M)
        if y[i].min() != y[i].max(): out.append(roc_auc_score(y[i], b[i]) - roc_auc_score(y[i], a[i]))
    return np.percentile(out, [2.5, 50, 97.5])


for k, s in S.items(): print(f"{k:38s} AUC={roc_auc_score(y, s):.3f}")
print("Δ restricción (original):     ", np.round(ci(S["original"], S["original + restricción tinción"]), 3))
print("Δ restricción (otro hospital):", np.round(ci(S["otro hospital"], S["otro hospital + restricción tinción"]), 3))
Image.fromarray(np.concatenate([np.concatenate([imgs[i], shifted[i], normalize_stain(shifted[i])], 1) for i in range(4)], 0)).save("figs/stain_examples.png")
