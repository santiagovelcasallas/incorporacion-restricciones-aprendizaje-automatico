"""Evaluación de las restricciones de los experimentos previos (números de la sección 5 del README).

Uso (desde esta carpeta, después de generar los embeddings):
    python evaluar.py d4          # invariancia D4: AUC por partición, % de diagnósticos que cambian al girar, κ de Fleiss
    python evaluar.py recortes    # mirar solo el centro (requiere crop_test.py)
    python evaluar.py transclip   # transducción TransCLIP / Histo-TransCLIP (requiere transclip.py)
    python evaluar.py coral       # cambio de hospital Camelyon17 + alineación CORAL (requiere c17_embed.py)
La normalización de tinción se evalúa dentro de stain_test.py.
"""
import sys, numpy as np
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_predict

norm = lambda a: a / np.linalg.norm(a, axis=-1, keepdims=True)


def zero_shot():
    """Embeddings PCam (8 orientaciones) y dirección zero-shot tumor − normal a partir de los textos."""
    d = np.load("cache/emb.npz")
    w = norm(d["txt1"].mean(0)) - norm(d["txt0"].mean(0))
    return d, d["emb"], d["labels"], w


def ic_auc(y, a, b, n=2000):
    """IC 95 % bootstrap de AUC(b) − AUC(a), remuestreando los mismos parches."""
    r = np.random.default_rng(0); D = []
    for _ in range(n):
        i = r.integers(0, len(y), len(y))
        if y[i].min() != y[i].max(): D.append(roc_auc_score(y[i], b[i]) - roc_auc_score(y[i], a[i]))
    return np.percentile(D, [2.5, 97.5])


def d4():
    """Restricción D4: promediar el embedding sobre las 8 rotaciones/reflexiones del parche."""
    d, emb, y, w = zero_shot()
    s0, sd = emb[:, 0] @ w, norm(emb.mean(1)) @ w                 # original (1 orientación) vs restricción
    _, it0 = train_test_split(np.arange(len(y)), test_size=0.5, stratify=y, random_state=0)
    print(f"AUC mitad de prueba (500, la del README) original {roc_auc_score(y[it0], s0[it0]):.3f} → D4 {roc_auc_score(y[it0], sd[it0]):.3f}"
          f"  IC95 Δ {np.round(ic_auc(y[it0], s0[it0], sd[it0]), 3)}")
    print(f"AUC los 1000 parches                     original {roc_auc_score(y, s0):.3f} → D4 {roc_auc_score(y, sd):.3f}")
    gana = 0
    for seed in range(10):                                          # 10 particiones calibración / prueba
        _, it = train_test_split(np.arange(len(y)), test_size=0.5, stratify=y, random_state=seed)
        gana += roc_auc_score(y[it], sd[it]) > roc_auc_score(y[it], s0[it])
    print(f"D4 mejora el AUC en {gana}/10 particiones")
    # % de parches de prueba cuyo diagnóstico cambia según la orientación (umbral de Youden fijado en calibración)
    so = emb @ w
    ic, it = train_test_split(np.arange(len(y)), test_size=0.5, stratify=y, random_state=0)
    youden = lambda t: recall_score(y[ic], s0[ic] >= t) + 1 - (s0[ic][y[ic] == 0] >= t).mean()
    thr = max(np.unique(s0[ic]), key=youden); dec = so[it] >= thr
    print(f"Diagnóstico distinto según el giro: {(dec.min(1) != dec.max(1)).mean():.1%} (con D4: 0 % por construcción)")
    # κ de Fleiss entre las 8 orientaciones (umbral P>0.5 con calibración de Platt), media de 10 particiones
    K = []
    for seed in range(10):
        ic, it = train_test_split(np.arange(len(y)), test_size=0.5, stratify=y, random_state=seed)
        pl = LogisticRegression().fit(s0[ic, None], y[ic])
        D = (pl.predict_proba(so[it].reshape(-1, 1))[:, 1] >= 0.5).reshape(-1, 8).astype(float)
        p1 = D.mean(1); Pi = ((p1 ** 2 + (1 - p1) ** 2) * 8 - 1) / 7
        pj = np.array([1 - D.mean(), D.mean()]); Pe = (pj ** 2).sum(); K.append((Pi.mean() - Pe) / (1 - Pe))
    print(f"κ de Fleiss entre orientaciones: {np.mean(K):.3f} (con D4: 1,000)")


def recortes():
    """Restricción espacial: embeddings del centro (64 / 48 px) vs el parche completo; AUC zero-shot."""
    d, emb, y, w = zero_shot(); c = np.load("cache/crops.npz"); M = len(c["c64"]); y = y[:M]
    base = emb[:M, 0] @ w
    for k in ["c64", "c48"]:
        s = c[k] @ w
        print(f"centro {k[1:]} px: AUC {roc_auc_score(y, base):.3f} → {roc_auc_score(y, s):.3f}  IC95 Δ {np.round(ic_auc(y, base, s), 3)}")


def transclip_eval():
    """Transducción (Zanella et al., 2024): agrupar los parches de prueba guiado por el texto."""
    from transclip import transclip
    d, emb, y, w = zero_shot()
    T = np.stack([norm(d["txt0"].mean(0)), norm(d["txt1"].mean(0))])
    for nombre, F in [("1 orientación", emb[:, 0]), ("D4", norm(emb.mean(1)))]:
        z, yh = transclip(F, T); z2, _ = transclip(F, T, use_text_kl=False)
        print(f"{nombre:14s} zero-shot AUC {roc_auc_score(y, yh[:, 1]):.3f} · TransCLIP {roc_auc_score(y, z[:, 1]):.3f}"
              f" · Histo-TransCLIP (sin KL) {roc_auc_score(y, z2[:, 1]):.3f}")


def coral():
    """Sonda lineal entrenada en los hospitales de origen, evaluada en el hospital nuevo, con y sin CORAL."""
    d = np.load("cache/c17.npz"); Xs, ys, Xt, yt = d["src"], d["src_y"], d["tgt"], d["tgt_y"]

    def alinear(Xt, Xs, eps=1.0):  # media y covarianza del hospital nuevo := las del origen (sin etiquetas)
        def msqrt(C, inv=False):
            w_, V = np.linalg.eigh(C); w_ = np.maximum(w_, 1e-9); return (V * (w_ ** (-0.5 if inv else 0.5))) @ V.T
        reg = lambda X: np.cov(X.T) + eps * np.eye(X.shape[1]) * np.trace(np.cov(X.T)) / X.shape[1]
        return (Xt - Xt.mean(0)) @ msqrt(reg(Xt), True) @ msqrt(reg(Xs)) + Xs.mean(0)

    h = LogisticRegression(C=1.0, max_iter=5000).fit(Xs, ys)
    cv = cross_val_predict(LogisticRegression(C=1.0, max_iter=5000), Xs, ys, cv=5, method="predict_proba")[:, 1]
    print(f"origen (val. cruzada) AUC {roc_auc_score(ys, cv):.3f}")
    print(f"hospital nuevo        AUC {roc_auc_score(yt, h.predict_proba(Xt)[:, 1]):.3f}")
    print(f"hospital nuevo + CORAL AUC {roc_auc_score(yt, h.predict_proba(alinear(Xt, Xs))[:, 1]):.3f}")


if __name__ == "__main__":
    {"d4": d4, "recortes": recortes, "transclip": transclip_eval, "coral": coral}[sys.argv[1]]()
