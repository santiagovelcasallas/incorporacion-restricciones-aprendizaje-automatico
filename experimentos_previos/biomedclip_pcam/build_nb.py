"""Convierte demo.py (celdas '# %%') en el notebook de Colab demo_restricciones.ipynb."""
import re, nbformat as nbf

import sys
SRC = sys.argv[1] if len(sys.argv) > 1 else "demo.py"
src = open(SRC, encoding="utf-8").read()
cells = []
for block in re.split(r"^# %%", src, flags=re.M)[1:]:
    if block.startswith(" [markdown]"):
        body = block.split("\n", 1)[1]
        md = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.strip().splitlines())
        cells.append(nbf.v4.new_markdown_cell(md))
    else:
        code = block.split("\n", 1)[1].strip().replace("#! ", "!")
        cells.append(nbf.v4.new_code_cell(code))
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {"accelerator": "GPU", "colab": {"gpuType": "A100", "provenance": []},
               "kernelspec": {"name": "python3", "display_name": "Python 3"}}
nbf.write(nb, SRC.replace(".py", ".ipynb"))
print(len(cells), "celdas")
