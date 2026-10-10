import time, numpy as np, torch, open_clip
from PIL import Image
torch.set_num_threads(16)
d=np.load("cache/emb.npz"); imgs=d["imgs"][:300]
model,pre=open_clip.create_model_from_pretrained("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"); model.eval()
@torch.no_grad()
def enc(arrs):
    out=[]
    for s in range(0,len(arrs),64):
        x=torch.stack([pre(Image.fromarray(np.ascontiguousarray(a))) for a in arrs[s:s+64]])
        out.append(torch.nn.functional.normalize(model.encode_image(x),dim=-1).numpy())
    return np.concatenate(out)
t0=time.time(); res={}
for c in (64,48):
    o=(96-c)//2; res[f"c{c}"]=enc([im[o:o+c,o:o+c] for im in imgs]); print(c,f"{time.time()-t0:.0f}s",flush=True)
np.savez("cache/crops.npz",**res)
