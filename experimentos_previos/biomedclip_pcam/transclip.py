"""Restricción transductiva TransCLIP (Zanella, Gérin, Ben Ayed — NeurIPS 2024), versión zero-shot,
portada a numpy desde el código oficial (github.com/MaxZanella/transduction-for-vlms, TransCLIP_utils.py).
Sin etiquetas y sin tocar los pesos del modelo. Para todos los parches a la vez, busca asignaciones z que:
  1) formen grupos compactos por clase (verosimilitud de una mezcla gaussiana),
  2) sean iguales entre parches vecinos (término Laplaciano, k=3 vecinos),
  3) no se alejen de la predicción del texto (término KL: yhat^lambda).
Hiperparámetros = los del código oficial (k=3, lambda=1, 10 iteraciones, temperatura 1/50)."""
import numpy as np


def _softmax(a):
    a = a - a.max(1, keepdims=True); e = np.exp(a); return e / e.sum(1, keepdims=True)


def transclip(F, T, k=3, lam=1.0, max_iter=10, use_text_kl=True):
    """F: (n,d) embeddings de imagen normalizados. T: (K,d) prototipos de texto normalizados.
    Devuelve (z, yhat): probabilidades transductivas y zero-shot, ambas (n,K)."""
    n, d = F.shape; K = T.shape[0]
    yhat = _softmax(100 * F @ T.T); z = yhat.copy()
    # mu: media de los 8 parches más confiables de cada clase
    mu = np.zeros((K, d))
    for c in range(K):
        top = np.argsort(-z[:, c])[:8]; mu[c] = (F[top] * z[top, c:c + 1]).mean(0)
    mu /= np.linalg.norm(mu, axis=1, keepdims=True)
    std = np.full(d, 1.0 / d)
    # afinidad: k vecinos por producto interno
    S = F @ F.T; nn = np.argsort(-S, axis=1)[:, 1:k + 1]
    W = np.zeros((n, n)); W[np.arange(n)[:, None], nn] = S[np.arange(n)[:, None], nn]
    for it in range(max_iter + 1):
        ll = np.stack([-0.5 * (((F - mu[c]) ** 2) / std).sum(1) for c in range(K)], 1)
        for _ in range(5):
            a = ll + (50 / (k * 2)) * (W.T @ z + W @ z)
            a = a - a.max(1, keepdims=True)
            e = (yhat ** lam if use_text_kl else 1.0) * np.exp(a / 50)
            z = e / e.sum(1, keepdims=True)
        if it == max_iter: break
        mu = (z.T @ F) / z.sum(0)[:, None]; mu /= np.linalg.norm(mu, axis=1, keepdims=True)
        std = sum((z[:, c:c + 1] * (F - mu[c]) ** 2).sum(0) for c in range(K)) / z.sum()
    return z, yhat
