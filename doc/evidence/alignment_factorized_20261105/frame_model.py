"""Experimental density wrapper with an optional intrinsic-shape output frame."""
import torch
from torch import nn
from sbsi.measurement_model import build_flow, ConditionalAffineFlow

FORMAT='sbsi_experimental_factorized_frame_v1'

class PhotometryShapeFlow(nn.Module):
    """Triangular transport: photometry first, then conditional shape.

    The four coordinates keep their original ordering [shape2, photometry2].
    Both flows are on the original standardized unbounded coordinates.
    """
    def __init__(self,config):
        super().__init__()
        assert config['target_dim']==4 and config['n_flows']==10
        kw={k:config[k] for k in ['hidden_dim','n_layers','scale_limit','activation']}
        self.photometry=ConditionalAffineFlow(2,config['context_dim'],n_flows=5,**kw)
        self.shape=ConditionalAffineFlow(2,config['context_dim']+2,n_flows=5,**kw)

    def forward(self,z,context):
        phot,lp=self.photometry.forward(z[:,2:],context)
        shape,ls=self.shape.forward(z[:,:2],torch.cat([context,phot],-1))
        return torch.cat([shape,phot],-1),lp+ls

    def inverse(self,y,context):
        zp,lp=self.photometry.inverse(y[:,2:],context)
        zs,ls=self.shape.inverse(y[:,:2],torch.cat([context,y[:,2:]],-1))
        return torch.cat([zs,zp],-1),lp+ls

class FrameFlow(nn.Module):
    def __init__(self,config,condition,kind):
        super().__init__()
        if kind not in ['cartesian','frame']:raise ValueError(kind)
        self.kind=kind;self.base=build_flow(config)
        self.base.base=PhotometryShapeFlow(config)
        self.target_dim=self.base.target_dim;self.context_dim=self.base.context_dim
        self.register_buffer('input_means',torch.as_tensor(condition.means[:2],dtype=torch.float64))
        self.register_buffer('input_scales',torch.as_tensor(condition.scales[:2],dtype=torch.float64))

    def rotation(self,context):
        e=context[:,:2].double()*self.input_scales+self.input_means
        norm=torch.linalg.vector_norm(e,dim=-1)
        safe=norm.clamp_min(1e-8)
        c=torch.where(norm>=1e-8,e[:,0]/safe,torch.ones_like(norm))
        s=torch.where(norm>=1e-8,e[:,1]/safe,torch.zeros_like(norm))
        return c,s

    def rotate(self,e,context,inverse=False):
        c,s=self.rotation(context)
        if inverse:s=-s
        return torch.stack([c*e[:,0]+s*e[:,1],-s*e[:,0]+c*e[:,1]],-1)

    def log_prob(self,y,context):
        if self.kind=='cartesian':return self.base.log_prob(y,context)
        mean,scale=self.base.disk_shape_means,self.base.disk_shape_scales
        shape=self.rotate(y[:,:2]*scale+mean,context)
        local=torch.cat([(shape-mean)/scale,y[:,2:]],-1)
        return self.base.log_prob(local,context)

    def forward_physical(self,z,context):
        physical,ld=self.base.forward_physical(z,context)
        if self.kind=='frame':physical=torch.cat([self.rotate(physical[:,:2],context,inverse=True),physical[:,2:]],-1)
        return physical,ld

def save(path,model,condition,target,config,metadata):
    torch.save({'format':FORMAT,'kind':model.kind,'state_dict':model.state_dict(),
                'condition':condition.to_state(),'target':target.to_state(),'config':config,'metadata':metadata},path)

def load(path,device='cuda'):
    from sbsi.selection_model import TabularPreprocessor
    from sbsi.measurement_model import TargetStandardizer
    d=torch.load(path,map_location=device,weights_only=False)
    assert d['format']==FORMAT
    condition=TabularPreprocessor.from_state(d['condition'])
    target=TargetStandardizer.from_state(d['target'])
    model=FrameFlow(d['config'],condition,d['kind']).to(device)
    model.load_state_dict(d['state_dict']);model.eval()
    return model,condition,target,d
