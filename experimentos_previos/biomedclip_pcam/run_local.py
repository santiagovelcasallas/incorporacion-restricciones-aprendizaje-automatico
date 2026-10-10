"""Corre demo.py localmente guardando cada figura en figs/ (en Colab se ven inline)."""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.makedirs("figs", exist_ok=True)
_n = [0]
def _show(*a, **k):
    _n[0] += 1; plt.savefig(f"figs/fig{_n[0]}.png", dpi=110, bbox_inches="tight"); plt.close("all")
plt.show = _show
exec(compile(open("demo.py", encoding="utf-8").read(), "demo.py", "exec"))
