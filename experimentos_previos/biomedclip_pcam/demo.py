# %% [markdown]
# # Restricciones post-entrenamiento en un Transformer médico
# **¿Puede una restricción —sin reentrenar nada— hacer que un modelo detecte mejor el cáncer?**
#
# - **Modelo:** BiomedCLIP (Vision Transformer de Microsoft, entrenado con 15 M de pares imagen–texto biomédicos). Lo usamos **congelado**.
# - **Datos:** PatchCamelyon (PCam): parches de 96×96 px de ganglios linfáticos teñidos con H&E. Etiqueta = hay metástasis de cáncer de mama en el **cuadro central de 32×32 px**.
# - **Restricción 1 (invariancia, hard):** un tumor en una laminilla no tiene "arriba" ni "abajo". Obligamos al modelo a dar **el mismo diagnóstico en las 8 rotaciones/reflexiones** (grupo D4) promediando su representación sobre el grupo.
# - **Restricción 2 (decisión):** *detectar al menos el 95 % de los tumores* y, sujeto a eso, minimizar las falsas alarmas.
#
# > **La idea:** restringir lo irrelevante libera capacidad para lo relevante.
#
# *Demostración académica, no es una herramienta clínica.*

# %%
#! pip -q install open_clip_torch
import os, io, time
import numpy as np, torch, pyarrow.parquet as pq, open_clip
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from huggingface_hub import hf_hub_download

T0 = time.time()
tic = lambda msg: print(f"⏱ {time.time()-T0:5.1f}s  {msg}")
dev = "cuda" if torch.cuda.is_available() else "cpu"
N = 1000          # parches (mitad con tumor, mitad sin tumor)
SEED = 0

# Paleta (fija para todo el notebook)
C_BASE, C_REST, C_SENS = "#9a9890", "#2a78d6", "#4a3aa7"   # modelo original / con restricción / + sensibilidad
C_TUM, C_NOR = "#eb6834", "#1baf7a"                         # clase tumor / normal
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.titleweight": "bold",
                     "axes.titlesize": 11, "legend.frameon": False})

# %% [markdown]
# ## 1. Cargar el modelo y los datos

# %%
HUB = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
model, preprocess = open_clip.create_model_from_pretrained(HUB)
tokenizer = open_clip.get_tokenizer(HUB)
model = model.to(dev).eval()
if dev == "cuda": model = model.half()
tic(f"BiomedCLIP cargado en {dev}")

path = "data/test.parquet" if os.path.exists("data/test.parquet") else hf_hub_download(
    "zacharielegault/PatchCamelyon", "data/test-00000-of-00001.parquet", repo_type="dataset")
table = pq.read_table(path)
labels_all = np.array(table.column("label").to_pylist())
rng = np.random.default_rng(SEED)
idx = np.concatenate([rng.choice(np.where(labels_all == c)[0], N // 2, replace=False) for c in (0, 1)])
rng.shuffle(idx)
col = table.column("image")
imgs = np.stack([np.array(Image.open(io.BytesIO(col[int(i)].as_py()["bytes"])).convert("RGB")) for i in idx])
y = labels_all[idx]
tic(f"{N} parches de PCam cargados ({y.sum()} con tumor)")

# %% [markdown]
# ## 2. Representaciones del Transformer en las 8 orientaciones (grupo D4)
# El modelo **no se toca**. Sólo le mostramos cada parche girado/reflejado.

# %%
def d4(x):
    out = []
    for k in range(4):
        r = np.rot90(x, k); out += [r, r[:, ::-1]]
    return out


@torch.no_grad()
def encode(arrs):
    x = torch.stack([preprocess(Image.fromarray(np.ascontiguousarray(a))) for a in arrs]).to(dev)
    if dev == "cuda": x = x.half()
    return torch.nn.functional.normalize(model.encode_image(x).float(), dim=-1).cpu().numpy()


CACHE = "cache/emb.npz"
if os.path.exists(CACHE) and np.array_equal(np.load(CACHE)["idx"], idx):
    emb = np.load(CACHE)["emb"]; tic("embeddings leídos de caché")
else:
    emb = np.zeros((N, 8, 512), np.float32)
    for s in range(0, N, 64):
        ch = imgs[s:s + 64]
        emb[s:s + len(ch)] = encode([v for a in ch for v in d4(a)]).reshape(len(ch), 8, -1)
    os.makedirs("cache", exist_ok=True); np.savez_compressed(CACHE, emb=emb, idx=idx)
    tic(f"{N*8} imágenes codificadas")

# Clasificador "zero-shot": el modelo compara la imagen contra descripciones en texto
TEMPL = ["this is a photo of {}", "an H&E stained histopathology image of {}",
         "a lymph node histology patch showing {}", "a microscopy image of {}"]
DESC = {0: ["normal lymph node tissue", "benign lymphoid tissue", "normal lymphocytes", "healthy tissue"],
        1: ["metastatic breast cancer in a lymph node", "tumor tissue", "metastatic tumor cells", "cancer cells"]}
with torch.no_grad():
    T = {c: torch.nn.functional.normalize(model.encode_text(tokenizer(
        [t.format(d) for d in ds for t in TEMPL], context_length=256).to(dev)).float(), dim=-1).mean(0).cpu().numpy()
         for c, ds in DESC.items()}
w_text = T[1] / np.linalg.norm(T[1]) - T[0] / np.linalg.norm(T[0])
tic("textos codificados")

# %% [markdown]
# ## 3. Aplicar la restricción (segundos)
# - **Modelo original:** mira el parche tal como llegó (una orientación).
# - **Con restricción D4:** promedia la representación sobre las 8 orientaciones → la función queda *forzada* a ser invariante. Es una **proyección**: elimina de la representación todas las direcciones que sólo codifican "cómo estaba girada la laminilla".
#
# Separamos los parches en **calibración** (para fijar umbrales) y **prueba** (donde se mide todo).

# %%
MODO = "zero-shot"   # "zero-shot" (sin ninguna etiqueta) o "sonda" (cabeza lineal ajustada en calibración)
norm = lambda a: a / np.linalg.norm(a, axis=-1, keepdims=True)
ical, ite = train_test_split(np.arange(N), test_size=0.5, stratify=y, random_state=SEED)

if MODO == "sonda":
    head = LogisticRegression(C=1.0, max_iter=3000).fit(emb[ical, 0], y[ical])
    raw = lambda X: head.decision_function(X.reshape(-1, 512)).reshape(X.shape[:-1])
else:
    raw = lambda X: X @ w_text

s_orbit = raw(emb)                 # puntaje del modelo original en cada una de las 8 orientaciones
s_base = s_orbit[:, 0]             # modelo original: la orientación en que llegó el parche
s_rest = raw(norm(emb.mean(1)))    # con restricción D4

# Probabilidades calibradas (Platt) en el conjunto de calibración: P>0.5 => "tumor"
platt_b = LogisticRegression().fit(s_base[ical, None], y[ical])
platt_r = LogisticRegression().fit(s_rest[ical, None], y[ical])
p_orbit = platt_b.predict_proba(s_orbit.reshape(-1, 1))[:, 1].reshape(N, 8)
p_base, p_rest = p_orbit[:, 0], platt_r.predict_proba(s_rest[:, None])[:, 1]
tic("restricción aplicada")

# %% [markdown]
# ## 4. El problema: el modelo original cambia de opinión si giras la laminilla

# %%
dec_orbit = p_orbit >= 0.5
flips = dec_orbit.min(1) != dec_orbit.max(1)
rango = p_orbit.max(1) - p_orbit.min(1)
# parche con tumor donde la opinión del modelo original más depende del giro y la restricción acierta
cand = [i for i in ite if y[i] == 1 and flips[i] and p_rest[i] >= 0.5]
k = max(cand, key=lambda i: rango[i]) if cand else int(np.argmax(rango * y))

fig = plt.figure(figsize=(12, 3.6))
names = ["original", "reflejo", "90°", "90°+reflejo", "180°", "180°+reflejo", "270°", "270°+reflejo"]
for j, v in enumerate(d4(imgs[k])):
    ax = fig.add_axes([0.01 + j * 0.105, 0.22, 0.095, 0.6]); ax.imshow(v); ax.set_xticks([]); ax.set_yticks([])
    ax.add_patch(Rectangle((31.5, 31.5), 32, 32, fill=False, ec="white", lw=1.2, ls="--"))
    t = p_orbit[k, j] >= 0.5
    ax.set_title(f"{names[j]}\nP(tumor)={p_orbit[k, j]:.2f}", fontsize=9, color=INK)
    ax.text(48, 106, "TUMOR ✓" if t else "NORMAL ✗", ha="center", va="top", fontsize=9, weight="bold",
            color=C_TUM if t else INK2)
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_color(C_TUM if t else C_BASE); sp.set_linewidth(2.5)
ax = fig.add_axes([0.86, 0.22, 0.13, 0.6]); ax.axis("off")
ax.text(0, 0.75, "Con restricción D4", fontsize=11, weight="bold", color=C_REST)
ax.text(0, 0.5, f"P(tumor) = {p_rest[k]:.2f}\nen las 8 orientaciones", fontsize=10, color=INK)
ax.text(0, 0.18, "TUMOR ✓", fontsize=13, weight="bold", color=C_TUM)
fig.suptitle("El mismo parche con tumor (verdad: TUMOR). Al modelo original lo cambia el giro de la laminilla",
             x=0.01, ha="left", fontsize=12, weight="bold", color=INK)
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw={"width_ratios": [2.2, 1]})
axs[0].hist(rango[ite], bins=30, color=C_BASE, ec="white", lw=1)
axs[0].axvline(0, color=C_REST, lw=3)
axs[0].annotate("con restricción: variación = 0\n(por construcción)", xy=(0, axs[0].get_ylim()[1] * 0.85),
                xytext=(0.08, axs[0].get_ylim()[1] * 0.85), color=C_REST, fontsize=9, va="center",
                arrowprops=dict(arrowstyle="->", color=C_REST))
axs[0].set(title="¿Cuánto cambia P(tumor) al girar el MISMO parche?", xlabel="máx − mín de P(tumor) entre las 8 orientaciones",
           ylabel="n.º de parches")
vals = [flips[ite].mean() * 100, 0.0]
b = axs[1].bar(["Modelo original", "Con restricción"], vals, color=[C_BASE, C_REST], width=0.55)
for r, v in zip(b, vals): axs[1].text(r.get_x() + r.get_width() / 2, v + 0.5, f"{v:.1f}%", ha="center", color=INK, weight="bold")
axs[1].set(title="Parches cuyo diagnóstico\ncambia según el giro", ylabel="% de parches de prueba", ylim=(0, max(vals) * 1.3 + 1))
axs[1].grid(axis="x", visible=False)
plt.tight_layout(); plt.show()
print(f"{flips[ite].sum()} de {len(ite)} parches de prueba reciben diagnósticos distintos según cómo se gire la laminilla.")

# %% [markdown]
# ## 5. ¿Qué hace la restricción por dentro? Elimina dimensiones que no tienen nada que ver con el cáncer
# Cada parche, girado de 8 formas, es una **nube** de 8 puntos en el espacio del Transformer. Esa dispersión es ruido puro: el diagnóstico no depende del giro.
# La restricción colapsa cada nube a su centro → las clases se separan mejor sobre el eje que importa.

# %%
from sklearn.decomposition import PCA
fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw={"width_ratios": [1.15, 1, 1]})
pca = PCA(2, random_state=0).fit(emb[ite].reshape(-1, 512))
sub = np.random.default_rng(1).choice(ite, 40, replace=False)
for i in sub:
    P = pca.transform(emb[i]); c = pca.transform(norm(emb[i].mean(0))[None])[0]
    col_ = C_TUM if y[i] else C_NOR
    for q in P: axs[0].plot([q[0], c[0]], [q[1], c[1]], color=col_, lw=0.6, alpha=0.45)
    axs[0].scatter(P[:, 0], P[:, 1], s=9, color=col_, alpha=0.5, lw=0)
    axs[0].scatter(*c, s=46, color=col_, ec="white", lw=1.2, zorder=3)
axs[0].scatter([], [], s=46, color=C_TUM, label="tumor"); axs[0].scatter([], [], s=46, color=C_NOR, label="normal")
axs[0].legend(loc="best"); axs[0].set(title="40 parches × 8 giros (PCA)\npuntos chicos = giros · grande = con restricción",
                                     xlabel="componente 1", ylabel="componente 2")


def dprime(s, yy): return (s[yy == 1].mean() - s[yy == 0].mean()) / np.sqrt(0.5 * (s[yy == 1].var() + s[yy == 0].var()))


lo, hi = np.percentile(np.r_[s_base[ite], s_rest[ite]], [0.5, 99.5]); bins = np.linspace(lo, hi, 40)
for ax, s, ttl in [(axs[1], s_base, "Modelo original"), (axs[2], s_rest, "Con restricción D4")]:
    ax.hist(s[ite][y[ite] == 0], bins=bins, color=C_NOR, alpha=0.75, label="normal", ec="white", lw=0.5)
    ax.hist(s[ite][y[ite] == 1], bins=bins, color=C_TUM, alpha=0.75, label="tumor", ec="white", lw=0.5)
    ax.set(title=f"{ttl}\nseparación entre clases d′ = {dprime(s[ite], y[ite]):.2f}", xlabel="puntaje del modelo (→ más tumor)",
           ylabel="n.º de parches"); ax.legend()
plt.tight_layout(); plt.show()
w_var = emb[ite].var(1).sum(-1).mean(); t_var = emb[ite].reshape(-1, 512).var(0).sum()
print(f"El {100*w_var/t_var:.1f}% de toda la variación de las representaciones se debe SÓLO al giro de la laminilla.")

# %% [markdown]
# ## 6. Resultado: AUC, recall y precisión (con intervalos de confianza)

# %%
yt = y[ite]


def metrics(p, thr=0.5, yy=None, ii=None):
    yy = yt if yy is None else yy; p = p[ite] if ii is None else p
    pr = p >= thr; tp = (pr & (yy == 1)).sum(); fp = (pr & (yy == 0)).sum(); fn = (~pr & (yy == 1)).sum()
    rec = tp / max(tp + fn, 1); prec = tp / max(tp + fp, 1)
    return dict(AUC=roc_auc_score(yy, p), Recall=rec, Precisión=prec, F1=2 * rec * prec / max(rec + prec, 1e-9))


def boot(pa, pb, n=2000):
    r = np.random.default_rng(0); A, B = [], []
    for _ in range(n):
        j = r.integers(0, len(ite), len(ite)); yy = yt[j]
        if yy.min() == yy.max(): continue
        A.append(metrics(pa[ite][j], yy=yy, ii=1)); B.append(metrics(pb[ite][j], yy=yy, ii=1))
    keys = A[0].keys()
    ci = lambda L, k: np.percentile([d[k] for d in L], [2.5, 97.5])
    diff = {k: np.percentile([b[k] - a[k] for a, b in zip(A, B)], [2.5, 50, 97.5]) for k in keys}
    return {k: ci(A, k) for k in keys}, {k: ci(B, k) for k in keys}, diff


mb, mr = metrics(p_base), metrics(p_rest)
cib, cir, diff = boot(p_base, p_rest)

fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw={"width_ratios": [1, 1, 1.4]})
for p, c, lab in [(p_base, C_BASE, "Modelo original"), (p_rest, C_REST, "Con restricción D4")]:
    fpr, tpr, _ = roc_curve(yt, p[ite]); axs[0].plot(fpr, tpr, color=c, lw=2, label=f"{lab}  AUC={roc_auc_score(yt, p[ite]):.3f}")
    pr_, rc_, _ = precision_recall_curve(yt, p[ite]); axs[1].plot(rc_, pr_, color=c, lw=2, label=f"{lab}  AP={average_precision_score(yt, p[ite]):.3f}")
axs[0].plot([0, 1], [0, 1], color=GRID, lw=1, ls="--")
axs[0].set(title="Curva ROC", xlabel="tasa de falsas alarmas", ylabel="tumores detectados (recall)"); axs[0].legend(loc="lower right", fontsize=8.5)
axs[1].set(title="Curva precisión–recall", xlabel="recall", ylabel="precisión", ylim=(0.4, 1.02)); axs[1].legend(loc="lower left", fontsize=8.5)

keys = list(mb.keys()); x = np.arange(len(keys)); wd = 0.36
for off, m, ci, c, lab in [(-wd / 2 - 0.01, mb, cib, C_BASE, "Modelo original"), (wd / 2 + 0.01, mr, cir, C_REST, "Con restricción D4")]:
    v = np.array([m[k] for k in keys]); e = np.array([[v[i] - ci[k][0], ci[k][1] - v[i]] for i, k in enumerate(keys)]).T
    axs[2].bar(x + off, v, wd, color=c, label=lab, yerr=e, ecolor=INK2, capsize=3, error_kw=dict(lw=1))
    for xi, vi in zip(x + off, v): axs[2].text(xi, 0.03, f"{vi:.2f}", ha="center", color="white", weight="bold", fontsize=9)
axs[2].set_xticks(x, [f"{k}\nΔ {diff[k][1]:+.3f}\nIC95 [{diff[k][0]:+.3f}, {diff[k][2]:+.3f}]" for k in keys], fontsize=8.5)
axs[2].set(title="Métricas en prueba (umbral P>0.5) · barras = IC 95 % bootstrap", ylim=(0, 1.05)); axs[2].legend(loc="upper right", fontsize=8.5)
axs[2].grid(axis="x", visible=False)
plt.tight_layout(); plt.show()
for k in keys: print(f"{k:10s} original {mb[k]:.3f}  →  restricción {mr[k]:.3f}   Δ={diff[k][1]:+.3f}  IC95 [{diff[k][0]:+.3f}, {diff[k][2]:+.3f}]")

# %% [markdown]
# ## 7. Restricción 2: "no se me puede escapar más del 5 % de los tumores"
# Problema de optimización: **minimizar falsas alarmas sujeto a recall ≥ 95 %**. El umbral se fija en calibración con un cuantil conformal (garantía estadística, no un ajuste a ojo) y se evalúa en prueba.
# Se aplica a los dos modelos: así vemos cuántas falsas alarmas cuesta cumplir la meta **con y sin** la restricción D4.

# %%
ALPHA = 0.05
def conformal_thr(p):
    sc = np.sort(p[ical][y[ical] == 1]); k = int(np.floor(ALPHA * (len(sc) + 1)))
    return sc[k - 1] if k >= 1 else -np.inf


thr_b, thr_r = conformal_thr(p_base), conformal_thr(p_rest)


def conf(p, thr):
    pr = p[ite] >= thr
    return np.array([[(~pr & (yt == 0)).sum(), (pr & (yt == 0)).sum()], [(~pr & (yt == 1)).sum(), (pr & (yt == 1)).sum()]])


cases = [("Modelo original\n(umbral 0.5)", conf(p_base, 0.5), C_BASE),
         ("Con restricción D4\n(umbral 0.5)", conf(p_rest, 0.5), C_REST),
         ("Original + recall≥95 %", conf(p_base, thr_b), C_BASE),
         ("D4 + recall≥95 %", conf(p_rest, thr_r), C_SENS)]
fig, axs = plt.subplots(1, 5, figsize=(16, 3.7), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.7]})
for ax, (ttl, M, c) in zip(axs[:4], cases):
    ax.imshow(np.zeros((2, 2)), cmap="Greys", vmin=0, vmax=1); ax.grid(False)
    for (r, cc), v in np.ndenumerate(M):
        bad = (r != cc)
        ax.add_patch(Rectangle((cc - 0.48, r - 0.48), 0.96, 0.96, color=(C_TUM if (r == 1 and cc == 0) else (C_BASE if bad else c)),
                               alpha=0.9 if (r == 1 and cc == 0) else (0.25 if bad else 0.75)))
        ax.text(cc, r, f"{v}", ha="center", va="center", fontsize=15, weight="bold", color=INK if bad else "white")
    ax.set_xticks([0, 1], ["dice\nnormal", "dice\ntumor"]); ax.set_yticks([0, 1], ["es normal", "es tumor"])
    ax.set_title(f"{ttl}\n{M[1,0]} tumores perdidos · {M[0,1]} falsas alarmas", fontsize=9.5)
    for sp in ax.spines.values(): sp.set_visible(False)
ts = np.linspace(0, 1, 201)
for p, c, lab in [(p_base, C_BASE, "original"), (p_rest, C_REST, "con restricción D4")]:
    fn = [((p[ite] < t) & (yt == 1)).sum() for t in ts]; fp = [((p[ite] >= t) & (yt == 0)).sum() for t in ts]
    axs[4].plot(fp, fn, color=c, lw=2, label=lab)
axs[4].axhline(int(np.ceil(ALPHA * (yt == 1).sum())), color=C_TUM, lw=1, ls="--")
axs[4].text(axs[4].get_xlim()[1] if False else 5, int(np.ceil(ALPHA * (yt == 1).sum())) + 3, "meta: ≤5 % de tumores perdidos", color=C_TUM, fontsize=8.5)
for M, c in [(cases[2][1], C_BASE), (cases[3][1], C_SENS)]:
    axs[4].scatter(M[0, 1], M[1, 0], s=70, color=c, ec="white", lw=1.5, zorder=3)
axs[4].set(title="El costo de no perder tumores", xlabel="falsas alarmas", ylabel="tumores perdidos"); axs[4].legend(fontsize=8.5)
plt.tight_layout(); plt.show()
print(f"Para cumplir recall≥95 %: original necesita {cases[2][1][0,1]} falsas alarmas; con restricción D4, {cases[3][1][0,1]}.")

# %% [markdown]
# ## 8. ¿Dónde mira el modelo? Tumores que el original no vio y la restricción sí
# Mapa de oclusión: tapamos cada región del parche y medimos cuánto baja P(tumor). Rojo = la región que convenció al modelo.
# Recuadro punteado = zona central de 32×32 px donde el patólogo marcó el tumor.

# %%
@torch.no_grad()
def occlusion(img, constrained=True, win=24, stride=12):
    ys = list(range(0, 96 - win + 1, stride)); views, H = [], np.zeros((len(ys), len(ys)))
    fill = img.reshape(-1, 3).mean(0).astype(np.uint8)
    for a in ys:
        for b in ys:
            m = img.copy(); m[a:a + win, b:b + win] = fill; views.append(m)
    f = lambda arrs: raw(norm(encode([v for a in arrs for v in d4(a)]).reshape(len(arrs), 8, -1).mean(1))) if constrained \
        else raw(encode(arrs))
    s0 = f([img])[0]; s = np.concatenate([f(views[i:i + 8]) for i in range(0, len(views), 8)])
    H = np.maximum(s0 - s, 0).reshape(len(ys), len(ys))
    big = np.zeros((96, 96)); cnt = np.zeros((96, 96))
    for (ia, a) in enumerate(ys):
        for (ib, b) in enumerate(ys): big[a:a + win, b:b + win] += H[ia, ib]; cnt[a:a + win, b:b + win] += 1
    return big / np.maximum(cnt, 1)


te = np.array(ite)
rescued = te[(y[te] == 1) & (p_base[te] < 0.5) & (p_rest[te] >= 0.5)]
lost = te[(y[te] == 1) & (p_base[te] >= 0.5) & (p_rest[te] < 0.5)]
fixed_fa = te[(y[te] == 0) & (p_base[te] >= 0.5) & (p_rest[te] < 0.5)]
new_fa = te[(y[te] == 0) & (p_base[te] < 0.5) & (p_rest[te] >= 0.5)]
show = rescued[np.argsort(-(p_rest[rescued] - p_base[rescued]))][:5]
from matplotlib.colors import LinearSegmentedColormap
heat = LinearSegmentedColormap.from_list("heat", [(1, 1, 1, 0), (0.89, 0.29, 0.28, 0.85)])
fig, axs = plt.subplots(2, len(show), figsize=(2.6 * len(show), 5.6), squeeze=False)
for j, i in enumerate(show):
    for r, overlay in enumerate([False, True]):
        ax = axs[r, j]; ax.imshow(imgs[i]); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        if overlay:
            Hm = occlusion(imgs[i]); ax.imshow(Hm / (Hm.max() + 1e-9), cmap=heat, vmin=0, vmax=1)
        ax.add_patch(Rectangle((31.5, 31.5), 32, 32, fill=False, ec="white" if not overlay else INK, lw=1.3, ls="--"))
    axs[0, j].set_title(f"original: P={p_base[i]:.2f} ✗\nrestricción: P={p_rest[i]:.2f} ✓", fontsize=9)
axs[0, 0].set_ylabel("parche (verdad: TUMOR)"); axs[1, 0].set_ylabel("dónde mira el modelo\ncon restricción")
fig.suptitle(f"Tumores rescatados por la restricción: {len(rescued)}  ·  tumores que la restricción perdió: {len(lost)}   |   "
             f"falsas alarmas corregidas: {len(fixed_fa)} · nuevas: {len(new_fa)}", fontsize=11, weight="bold", color=INK)
plt.tight_layout(); plt.show()
tic("fin")

# %% [markdown]
# ## Conclusión
# - El Transformer **no se reentrenó**: el mismo modelo, los mismos pesos.
# - La restricción D4 le quitó al modelo la libertad de "opinar" sobre algo irrelevante (el giro de la laminilla). Esa parte de su representación era ruido; al eliminarla, la señal del tumor queda más limpia.
# - La restricción de sensibilidad convierte una exigencia clínica ("no perder más del 5 % de tumores") en una regla con garantía estadística, y la restricción D4 abarata su costo en falsas alarmas.
# - **Restringir lo irrelevante libera capacidad para lo relevante.** Es la misma idea detrás de las CNN (invariancia a traslaciones), de la decodificación restringida en LLMs (PICARD) y de los modelos equivariantes en ciencia.
