import torch, numpy as np, pprint
p = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/measurement_flow_g0_ngmix_ablate_s2c_lt500_dom6x6_s501_swaavg.pt"
ck = torch.load(p, map_location="cpu", weights_only=False)
def show(d, name):
    print("="*70); print(name); print("="*70)
    for k,v in d.items():
        if isinstance(v,(np.ndarray,)) or hasattr(v,"shape"):
            arr = np.asarray(v)
            print(f"  {k}: array{arr.shape} = {np.round(arr,5).tolist() if arr.size<40 else '...'}")
        else:
            print(f"  {k}: {type(v).__name__} = {repr(v)[:600]}")
show(ck["condition_preprocessor"], "CONDITION_PREPROCESSOR")
show(ck["target_transform"], "TARGET_TRANSFORM")
print("="*70); print("STATE_DICT (95 keys)"); print("="*70)
for k,v in ck["state_dict"].items():
    print(f"  {k}: {tuple(v.shape)}")
