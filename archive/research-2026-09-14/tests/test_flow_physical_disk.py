import numpy as np
import pytest
import torch
from sbsi.measurement_model import build_flow,TargetStandardizer,save_measurement_model,load_measurement_model
from sbsi.selection_model import TabularPreprocessor
from sbsi.flow_physical_disk import positive_inverse

def config():
 return dict(flow_type='physical_disk_affine',target_dim=4,context_dim=2,hidden_dim=8,n_layers=1,n_flows=2,scale_limit=1.,disk_map='radial_tanh',disk_shape_means=[0.,0.],disk_shape_scales=[.3,.3],disk_coordinate_means=[0.,0.],disk_coordinate_scales=[.5,.5],physical_map='softplus',physical_means=[3.,100.],physical_scales=[2.,80.],physical_units=[2.,80.],physical_coordinate_means=[1.,0.],physical_coordinate_scales=[1.,2.])

def test_positive_map_extremes():
 u=torch.tensor([-100.,-10.,0.,10.,100.,1e5],dtype=torch.float64,requires_grad=True)
 p=torch.logaddexp(u,torch.zeros_like(u));torch.testing.assert_close(positive_inverse(p),u)
 assert p[-1]==1e5
 torch.testing.assert_close(torch.autograd.grad(p.sum(),u)[0],torch.sigmoid(u))

def test_density_roundtrip_jacobian_and_support():
 torch.manual_seed(3);m=build_flow(config());c=torch.zeros((1,2));z=torch.tensor([[.2,-.1,.4,.6]])
 y,ld=m(z,c);back,ild=m.inverse(y,c)
 torch.testing.assert_close(back,z,rtol=1e-5,atol=1e-6);torch.testing.assert_close(ld+ild,torch.zeros_like(ld),rtol=0,atol=1e-6)
 jac=torch.autograd.functional.jacobian(lambda x:m(x[None],c)[0][0],z[0]);torch.testing.assert_close(torch.linalg.slogdet(jac).logabsdet.double(),ld[0],rtol=1e-5,atol=1e-5)
 physical=y[:,2:]*m.physical_scales+m.physical_means;assert bool((physical>0).all())
 bad=y.detach().clone();bad[:,2]=-2;assert torch.isneginf(m.log_prob(bad,c)).all()
 loss=-m.log_prob(y.detach(),c).mean();loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())

def test_checkpoint_metadata(tmp_path):
 import pandas as pd
 cfg=config();m=build_flow(cfg);pre=TabularPreprocessor.fit(pd.DataFrame({'a':[0.,1.],'b':[1.,2.]}),['a','b'],log_features=[])
 target=TargetStandardizer(['measured_ngmix_g1','measured_ngmix_g2','measured_flux_radius','measured_flux_from_mag_auto'],np.array([0.,0.,3.,100.]),np.array([.3,.3,2.,80.]),dtype='float64')
 p=tmp_path/'m.pt';save_measurement_model(p,m,pre,target,cfg);load_measurement_model(p)
 cfg['physical_means']=[4.,100.]
 with pytest.raises(ValueError):save_measurement_model(p,m,pre,target,cfg)

def test_reject_swallowed_physical_options():
 cfg=config();cfg['flow_type']='affine'
 for k in list(cfg):
  if k.startswith('disk_'):del cfg[k]
 with pytest.raises(ValueError,match='physical'):build_flow(cfg)

def test_common_nll_reference_jacobian():
 from sbsi.flow_physical_response import physical_outputs
 oldmu=torch.tensor([0.,0.,24.,.5],dtype=torch.float64);oldsd=torch.tensor([.3,.4,2.,.7],dtype=torch.float64)
 newmu=torch.tensor([0.,0.,3.,100.],dtype=torch.float64);newsd=torch.tensor([.3,.4,2.,80.],dtype=torch.float64)
 z=torch.tensor([.1,-.2,.3,.4],dtype=torch.float64);raw=z*oldsd+oldmu
 jac=torch.autograd.functional.jacobian(lambda x:(physical_outputs(x*oldsd+oldmu)-newmu)/newsd,z)
 k=.4*np.log(10);formula=oldsd.log().sum()-newsd.log().sum()+raw[3]+k*(30-raw[2])+np.log(k)
 torch.testing.assert_close(torch.linalg.slogdet(jac).logabsdet,formula)

def test_direct_physical_map_preserves_tiny_positive_flux_and_density_jacobian():
 cfg=config();cfg['physical_coordinate_means']=[-60.,-60.]
 m=build_flow(cfg);z=torch.zeros((1,4));c=torch.zeros((1,2))
 p,ld=m.forward_physical(z,c);y,yld=m(z,c)
 assert bool((p[:,2:]>0).all())
 reconstructed=y[:,2:]*m.physical_scales+m.physical_means
 assert bool((reconstructed==0).all())
 scales=torch.cat((m.disk_shape_scales,m.physical_scales))
 torch.testing.assert_close(ld-yld,scales.log().sum().expand_as(ld))
 p[:,2:].sum().backward()
 assert all(x.grad is None or torch.isfinite(x.grad).all() for x in m.parameters())

def test_bundle_sampling_preserves_physical_tiny_positive_values():
 import pandas as pd
 from sbsi.measurement_model import MeasurementModelBundle
 cfg=config();cfg['context_dim']=4;cfg['physical_coordinate_means']=[-60.,-60.]
 model=build_flow(cfg);frame=pd.DataFrame({'a':[0.,1.],'b':[1.,2.]})
 pre=TabularPreprocessor.fit(frame,['a','b'],log_features=[])
 target=TargetStandardizer(['measured_ngmix_g1','measured_ngmix_g2','measured_flux_radius','measured_flux_from_mag_auto'],np.array([0.,0.,3.,100.]),np.array([.3,.3,2.,80.]),dtype='float64')
 bundle=MeasurementModelBundle(model,pre,target)
 sample=bundle.sample(frame,n_samples=4,batch_size=1,qmc=True)
 assert np.all(sample[...,2:]>0)
