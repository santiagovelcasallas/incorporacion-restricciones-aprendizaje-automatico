"""Restricción de invariancia a permutaciones en un LLM (análogo de D4 para texto).
Preguntas médicas MMLU (clinical_knowledge). El modelo lee la pregunta y le leemos la probabilidad de A/B/C/D.
Original: un solo orden de opciones. Restricción: la respuesta no puede depender del orden ->
promediamos log-probabilidades sobre las 4 rotaciones cíclicas de las opciones."""
import sys, time, numpy as np, torch, pandas as pd
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
MODEL = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen2.5-0.5B-Instruct"
torch.set_num_threads(16)
df = pd.read_parquet(hf_hub_download("cais/mmlu", "clinical_knowledge/test-00000-of-00001.parquet", repo_type="dataset"))
df = df.sample(n=min(N, len(df)), random_state=0).reset_index(drop=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
L = "ABCD"; ids = [tok.encode(c, add_special_tokens=False)[0] for c in L]


@torch.no_grad()
def letter_logprobs(q, opts):
    body = q + "\n" + "\n".join(f"{L[j]}. {o}" for j, o in enumerate(opts))
    msgs = [{"role": "user", "content": "Answer the following medical multiple-choice question with only the letter of the correct option.\n\n" + body}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + "The answer is "
    x = tok(text, return_tensors="pt")
    logits = model(**x).logits[0, -1, ids]
    return torch.log_softmax(logits, -1).numpy()


t0 = time.time(); LP = np.zeros((len(df), 4, 4))  # [pregunta, permutación k, opción ORIGINAL]
for i, r in df.iterrows():
    opts = list(r["choices"])
    for k in range(4):
        perm = [(j + k) % 4 for j in range(4)]           # posición j muestra la opción original perm[j]
        lp = letter_logprobs(r["question"], [opts[p] for p in perm])
        for j, p in enumerate(perm): LP[i, k, p] = lp[j]
    if i % 20 == 0: print(f"{i}/{len(df)} {time.time()-t0:.0f}s", flush=True)
y = df["answer"].values
np.savez(f"cache/llm_{MODEL.split('/')[-1]}_{N}.npz", LP=LP, y=y)
print(f"listo {time.time()-t0:.0f}s")
