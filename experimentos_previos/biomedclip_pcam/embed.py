"""Paso 1: embeddings de BiomedCLIP (requiere data/test.parquet: python ../descargar_datos.py).
Se corre UNA vez: embeddings BiomedCLIP de cada parche
en sus 8 orientaciones D4. Guarda todo en cache/emb.npz.
Uso: python embed.py N   (N imágenes balanceadas del split test de PCam)"""
import os, sys, io, time
import numpy as np, torch, pyarrow.parquet as pq
from PIL import Image
import open_clip

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
SEED = 0
torch.set_num_threads(16)
os.makedirs("cache", exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"

t0 = time.time()
model, preprocess = open_clip.create_model_from_pretrained(
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
tokenizer = open_clip.get_tokenizer(
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
model = model.to(dev).eval()
print(f"modelo cargado {time.time()-t0:.1f}s en {dev}")

t = pq.read_table("data/test.parquet")
labels_all = np.array(t.column("label").to_pylist())
rng = np.random.default_rng(SEED)
idx = np.concatenate([rng.choice(np.where(labels_all == c)[0], N // 2, replace=False) for c in (0, 1)])
rng.shuffle(idx)
imgs_col = t.column("image")
imgs = np.stack([np.array(Image.open(io.BytesIO(imgs_col[int(i)].as_py()["bytes"])).convert("RGB")) for i in idx])
labels = labels_all[idx]


def d4(x):
    """8 elementos del grupo diédrico D4 sobre un array HxWxC."""
    out = []
    for k in range(4):
        r = np.rot90(x, k)
        out += [r, r[:, ::-1]]
    return out  # orden: (rot0, rot0+flip, rot90, rot90+flip, ...)


@torch.no_grad()
def encode(batch_np):
    x = torch.stack([preprocess(Image.fromarray(np.ascontiguousarray(a))) for a in batch_np]).to(dev)
    f = model.encode_image(x)
    return torch.nn.functional.normalize(f, dim=-1).float().cpu().numpy()


t0 = time.time()
emb = np.zeros((N, 8, 512), np.float32)
B = 32
for s in range(0, N, B):
    chunk = imgs[s:s + B]
    views = [v for a in chunk for v in d4(a)]
    emb[s:s + len(chunk)] = encode(views).reshape(len(chunk), 8, -1)
    if s % (B * 4) == 0:
        print(f"  {s+len(chunk)}/{N}  {time.time()-t0:.0f}s", flush=True)
        np.save("cache/emb_partial.npy", emb[:s + len(chunk)])
print(f"embeddings {time.time()-t0:.1f}s")

# textos: plantillas x descripciones por clase
TEMPL = ["this is a photo of {}", "an H&E stained histopathology image of {}",
         "a lymph node histology patch showing {}", "a microscopy image of {}"]
CLASSES = {
    0: ["normal lymph node tissue", "benign lymphoid tissue", "normal lymphocytes", "healthy tissue"],
    1: ["metastatic breast cancer in a lymph node", "tumor tissue", "metastatic tumor cells", "cancer cells"],
}
with torch.no_grad():
    txt = {}
    for c, descs in CLASSES.items():
        prompts = [tp.format(d) for d in descs for tp in TEMPL]
        f = model.encode_text(tokenizer(prompts, context_length=256).to(dev))
        txt[c] = torch.nn.functional.normalize(f, dim=-1).float().cpu().numpy()
scale = float(model.logit_scale.exp())

np.savez_compressed("cache/emb.npz", emb=emb, labels=labels, imgs=imgs, idx=idx,
                    txt0=txt[0], txt1=txt[1], logit_scale=scale)
print("guardado cache/emb.npz")
