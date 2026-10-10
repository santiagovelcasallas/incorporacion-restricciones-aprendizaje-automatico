"""Evalúa la restricción de invariancia al orden de las opciones (salida de llm_test.py).

Compara la exactitud del LLM en cada uno de los 4 órdenes de opciones con la de la restricción
(promediar las log-probabilidades sobre los 4 órdenes), y muestra el sesgo por letra.
Uso: python llm_eval.py cache/llm_Qwen2.5-0.5B-Instruct_60.npz
"""
import sys, numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
d=np.load(sys.argv[1]); LP,y=d["LP"],d["y"]; n=len(y)
P_orig=LP.argmax(2)                 # (n,4): respuesta (opción original) en cada uno de los 4 órdenes
P_rest=LP.mean(1).argmax(1)         # restricción: promedio sobre órdenes
acc_each=[(P_orig[:,k]==y).mean() for k in range(4)]
print(f"n={n} preguntas")
print("Exactitud original en cada orden:", np.round(acc_each,3), "→ media", round(np.mean(acc_each),3))
print("Exactitud con restricción:", round((P_rest==y).mean(),3))
# métricas por clase (clase = opción correcta), promediadas sobre los 4 órdenes para el original
def m(pred): p,r,f,_=precision_recall_fscore_support(y,pred,average="macro",zero_division=0); return p,r,f
mo=np.mean([m(P_orig[:,k]) for k in range(4)],0); mr=m(P_rest)
print(f"Macro precisión {mo[0]:.3f}→{mr[0]:.3f} | macro recall {mo[1]:.3f}→{mr[1]:.3f} | macro F1 {mo[2]:.3f}→{mr[2]:.3f}")
# sesgo de posición: qué LETRA elige (posición mostrada) en todos los órdenes
pos=np.array([[(P_orig[i,k]-k)%4 for k in range(4)] for i in range(n)]).ravel()
print("Letra elegida (original, todas las presentaciones):", {c:f"{(pos==j).mean():.0%}" for j,c in enumerate('ABCD')}, " (lo correcto sería ~25% c/u)")
inc=(P_orig.min(1)!=P_orig.max(1)).mean(); print(f"Preguntas cuya respuesta cambia al reordenar opciones: {inc:.0%}")
k0=P_orig[:,0]; print(f"vs orden original: arregladas {((k0!=y)&(P_rest==y)).sum()} · dañadas {((k0==y)&(P_rest!=y)).sum()}")
