"""Function-preserving extra conditioning in every photometry/shape coupling."""
import importlib.util
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sbsi.selection_model import TabularPreprocessor
from sbsi.measurement_model import TargetStandardizer
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('neighbour_common_model',HERE/'common.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
FORMAT='sbsi_experimental_neighbour_factorized_v1'
class Condition:
    def __init__(self,base,mean,scale,use_score):
        self.base=base;self.mean=float(mean);self.scale=float(scale);self.use_score=bool(use_score)
        assert np.isfinite([self.mean,self.scale]).all() and self.scale>0
        self.means=base.means;self.scales=base.scales
    def transform_frame(self,frame):
        old=self.base.transform_frame(frame)
        score=frame[c.FEATURE].to_numpy(float)
        if not np.isfinite(score).all():raise ValueError('nonfinite neighbour input')
        extra=(score-self.mean)/self.scale if self.use_score else np.zeros(len(score))
        return np.column_stack([old,extra]).astype(np.float32)
    def to_state(self):return {'base':self.base.to_state(),'mean':self.mean,'scale':self.scale,'use_score':self.use_score}
    @classmethod
    def from_state(cls,d):return cls(TabularPreprocessor.from_state(d['base']),d['mean'],d['scale'],d['use_score'])
class ExtraInputLinear(nn.Module):
    """Equivalent to inserting a zero column, preserving the old GEMM at init."""
    def __init__(self,original,index):
        super().__init__();self.original=original;self.index=int(index)
        self.extra_weight=nn.Parameter(torch.zeros(original.out_features,device=original.weight.device,dtype=original.weight.dtype))
    def forward(self,x):
        old=torch.cat([x[:,:self.index],x[:,self.index+1:]],-1)
        return self.original(old)+x[:,self.index:self.index+1]*self.extra_weight
def augment(model):
    old_width=model.context_dim
    for flow in [model.base.base.photometry,model.base.base.shape]:
        for layer in flow.layers:
            assert isinstance(layer.net[0],nn.Linear)
            layer.net[0]=ExtraInputLinear(layer.net[0],2+old_width)
        flow.context_dim+=1
    model.context_dim=old_width+1
    return model
def initialize(mean,scale,use_score,device='cuda'):
    assert c.sha(c.BASE)==c.BASE_SHA
    model,base,target,artifact=c.fm.load(c.BASE,device=device)
    condition=Condition(base,mean,scale,use_score)
    return augment(model),condition,target,artifact['config']
def save(path,model,condition,target,config,metadata):
    torch.save({'format':FORMAT,'state_dict':model.state_dict(),'condition':condition.to_state(),
        'target':target.to_state(),'config':config,'metadata':metadata},path)
def load(path,device='cuda'):
    d=torch.load(path,map_location=device,weights_only=False);assert d['format']==FORMAT
    condition=Condition.from_state(d['condition']);target=TargetStandardizer.from_state(d['target'])
    model=augment(c.fm.FrameFlow(d['config'],condition.base,'cartesian')).to(device)
    model.load_state_dict(d['state_dict']);model.eval()
    return model,condition,target,d
