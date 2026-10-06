import copy

import numpy as np
import pandas as pd
import torch

from scripts.continue_circularized_flow import continue_fit, shape_epoch, nll_pass
from scripts.trial_circularized_flow import fit_nll_arm
from sbsi.measurement_model import TargetStandardizer, build_flow
from sbsi.selection_model import TabularPreprocessor
from sbsi.flow_physical_population import PhysicalResponsePopulation
from test_flow_physical_disk import config


def training_fixture():
    torch.manual_seed(87)
    context = torch.randn(24, 2)
    model = build_flow(config())
    with torch.no_grad():
        values = model(torch.randn(24, 4), context)[0]
    frame = pd.DataFrame(context.numpy(), columns=['e1_input_p', 'Re_input_p'])
    pre = TabularPreprocessor.fit(frame, list(frame), add_missing_indicators=False)
    target = TargetStandardizer(
        ['measured_ngmix_g1','measured_ngmix_g2','measured_flux_radius','measured_flux_from_mag_auto'],
        np.array([0.,0.,3.,100.]), np.array([.3,.3,2.,80.]), dtype='float64')
    return model, values, context, pre, target


def test_continuation_matches_uninterrupted_nll_weights_and_adam_state(tmp_path):
    _, values, context, pre, target = training_fixture()
    recipe = dict(seed=501, learning_rate=1e-4, epochs=4, rows_per_epoch=13,
                  batch_size=6, early_stopping_patience=20)
    train, val = (values[:18],context[:18]), (values[18:],context[18:])
    fit_nll_arm(tmp_path/'whole','whole',config(),pre,target,train,val,recipe,{})
    _, selected = fit_nll_arm(tmp_path/'first','first',config(),pre,target,train,val,
                              recipe | dict(epochs=2),{})
    initial = torch.load(tmp_path/'first/optimizer.pt',weights_only=False)
    parent = torch.load(tmp_path/'first/epoch002.pt',weights_only=False)
    continued_recipe = recipe | dict(max_epoch=3, min_lr=1e-6, diagnostic_interval=2)
    diagnostic = lambda model,epoch:dict(shape_loss=0.)
    continue_fit(tmp_path/'continued','nll',parent,initial,selected['best_checkpoint'],
                 train,val,continued_recipe,{},diagnostic)
    # Simulate wall-time interruption after epoch 3, then resume the committed state.
    continue_fit(tmp_path/'continued','nll',parent,initial,selected['best_checkpoint'],
                 train,val,continued_recipe | dict(max_epoch=4),{},diagnostic,resume=True)
    a = torch.load(tmp_path/'whole/optimizer.pt',weights_only=False)
    b = torch.load(tmp_path/'continued/optimizer.pt',weights_only=False)
    for key in a['state_dict']:
        torch.testing.assert_close(a['state_dict'][key],b['state_dict'][key],rtol=0,atol=0)
    for parameter in a['optimizer']['state']:
        for key in a['optimizer']['state'][parameter]:
            torch.testing.assert_close(a['optimizer']['state'][parameter][key],
                                       b['optimizer']['state'][parameter][key],rtol=0,atol=0)
    assert a['scheduler']==b['scheduler']


def test_zero_shape_weight_preserves_nll_updates_exactly():
    model, values, context, _, _ = training_fixture()
    clone = copy.deepcopy(model)
    rng = np.random.default_rng(9)
    statistics = dict(target=rng.normal(size=(2,4)), precision=np.tile(np.eye(4),(2,1,1)),
                      counts=np.array([6,6]), shape_design=rng.normal(size=(12,2)),
                      scalar_design=rng.normal(size=(12,2)))
    population = PhysicalResponsePopulation(context,np.arange(24).reshape(12,2),np.repeat([0,1],6),statistics)
    optimizers = [torch.optim.AdamW(m.parameters(),lr=1e-5,weight_decay=0.) for m in (model,clone)]
    recipe = dict(batch_size=7,cells_per_step=2,pairs_per_cell=3,training_draws=4,
                  response_batch_size=3,training_response_seed=123000,shape_weight=0.)
    expected = nll_pass(model,values,context,7,optimizer=optimizers[0],seed=51,clip=5.)
    actual = shape_epoch(clone,values,context,population,optimizers[1],recipe,51)
    assert actual['nll']==expected['nll']
    for a,b in zip(model.parameters(),clone.parameters()):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
