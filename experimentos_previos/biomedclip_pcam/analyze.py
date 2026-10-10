"""Paso 2 (en vivo, segundos): aplicar las restricciones sobre los embeddings
guardados y medir. Sin figuras: sólo números, para validar la hipótesis."""
import numpy as np
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

d = np.load("cache/emb.npz")
emb, y = d["emb"], d["labels"]
m0, m1 = d["txt0"].mean(0), d["txt1"].mean(0)
w_zs = m1 / np.linalg.norm(m1) - m0 / np.linalg.norm(m0)

norm = lambda a: a / np.linalg.norm(a, axis=-1, keepdims=True)
x_base = emb[:, 0]                 # el modelo tal cual: una sola orientación
x_d4 = norm(emb.mean(1))           # restricción: promedio sobre el grupo D4

ical, ite = train_test_split(np.arange(len(y)), test_size=0.5, stratify=y, random_state=0)


def boot_auc_diff(y, a, b, n=2000, seed=0):
    r = np.random.default_rng(seed); diffs = []
    for _ in range(n):
        i = r.integers(0, len(y), len(y))
        if y[i].min() == y[i].max(): continue
        diffs.append(roc_auc_score(y[i], b[i]) - roc_auc_score(y[i], a[i]))
    return np.percentile(diffs, [2.5, 50, 97.5])


def report(name, s_base, s_d4, s_orbit):
    yc, yt = y[ical], y[ite]
    print(f"\n=== {name} ===")
    for lab, s in [("base", s_base), ("D4", s_d4)]:
        thr = np.median(s[ical])  # umbral simple fijado en calibración
        # Youden en calibración
        cand = np.unique(s[ical]); best = max(cand, key=lambda t: recall_score(yc, s[ical] >= t) + 1 - (s[ical][yc == 0] >= t).mean())
        p = s[ite] >= best
        print(f"{lab:5s} AUC={roc_auc_score(yt, s[ite]):.3f}  recall={recall_score(yt, p):.3f}  prec={precision_score(yt, p):.3f}")
    print("dAUC (D4-base) IC95:", np.round(boot_auc_diff(yt, s_base[ite], s_d4[ite]), 4))
    # inestabilidad: ¿cuántas imágenes cambian de diagnóstico según cómo se gire?
    cand = np.unique(s_base[ical]); thr = max(cand, key=lambda t: recall_score(yc, s_base[ical] >= t) + 1 - (s_base[ical][yc == 0] >= t).mean())
    dec = s_orbit[ite] >= thr
    flip = (dec.min(1) != dec.max(1)).mean()
    print(f"imágenes cuyo diagnóstico cambia al rotarlas (modelo base): {flip:.1%}")


# A) zero-shot puro: el modelo no ve ninguna etiqueta
report("zero-shot BiomedCLIP", x_base @ w_zs, x_d4 @ w_zs, emb @ w_zs)

# B) sonda lineal entrenada en calibración con UNA orientación; la restricción se aplica sólo al evaluar
clf = LogisticRegression(C=1.0, max_iter=2000).fit(x_base[ical], y[ical])
f = lambda X: clf.decision_function(X.reshape(-1, X.shape[-1])).reshape(X.shape[:-1])
report("sonda lineal", f(x_base), f(x_d4), f(emb))
