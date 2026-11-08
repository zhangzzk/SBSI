"""Device-independent latent streams for matched constant-shear comparisons.

NumPy PCG64 produces the standard normals on CPU before transfer to the GPU.
The same bank is reused across shear legs and models. This intentionally differs
from legacy CUDA RNG streams, whose reproducibility across devices failed replay.
"""
import hashlib
import numpy as np
import torch

def latent_bank(batch,draws,dimension,seed):
    if draws<2 or draws%2:raise ValueError('Even antithetic draw count required')
    half=np.random.Generator(np.random.PCG64(int(seed))).standard_normal((batch,draws//2,dimension)).astype(np.float32)
    return np.concatenate((half,-half),axis=1)

def physical_context_draws(model,contexts,draws,seed):
    batch=len(contexts[0]);bank=latent_bank(batch,draws,model.target_dim,seed)
    base=torch.from_numpy(bank).to(device=contexts[0].device,dtype=contexts[0].dtype).reshape(batch*draws,model.target_dim)
    result=[]
    for context in contexts:
        expanded=context[:,None,:].expand(batch,draws,model.context_dim).reshape(batch*draws,model.context_dim)
        sample,_=model.forward_physical(base,expanded)
        result.append(sample.reshape(batch,draws,model.target_dim))
    return result
