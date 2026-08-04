import torch, pprint
p = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/measurement_flow_g0_ngmix_ablate_s2c_lt500_dom6x6_s501_swaavg.pt"
ck = torch.load(p, map_location="cpu", weights_only=False)
print("="*70); print("MODEL_CONFIG"); print("="*70)
pprint.pprint(ck["model_config"], width=160)
print("="*70); print("METADATA"); print("="*70)
pprint.pprint(ck["metadata"], width=160)
print("="*70); print("CONDITION_PREPROCESSOR keys"); print("="*70)
for k,v in ck["condition_preprocessor"].items():
    if hasattr(v,"shape"): print(f"  {k}: tensor{tuple(v.shape)} -> {v.tolist() if v.numel()<40 else '...'}")
    else: print(f"  {k}: {type(v).__name__} = {repr(v)[:800]}")
print("="*70); print("TARGET_TRANSFORM"); print("="*70)
for k,v in ck["target_transform"].items():
    if hasattr(v,"shape"): print(f"  {k}: tensor{tuple(v.shape)} -> {v.tolist() if v.numel()<40 else '...'}")
    else: print(f"  {k}: {type(v).__name__} = {repr(v)[:800]}")
