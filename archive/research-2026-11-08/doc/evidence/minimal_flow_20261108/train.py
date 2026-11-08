"""Fixed-endpoint transport and two-axis response distillation on training contexts."""
import json
import os
import time
import types
from pathlib import Path
import numpy as np
import torch
from sbsi.shear_map import apply_shear_to_ellipticity

H = Path(__file__).resolve().parent
c = types.ModuleType('minimal_train_common'); c.__file__ = str(H / 'common.py')
exec(compile((H / 'common.py').read_bytes(), c.__file__, 'exec'), c.__dict__)
m = c.load('minimal_student', H / 'model.py')

def sheared_contexts(x, condition):
    # Preprocessor first two columns are linearly standardized true e1/e2.
    assert condition.base.feature_names[:2] == ['e1', 'e2'] or all(
        ('e1' if i == 0 else 'e2') in name for i, name in enumerate(condition.base.feature_names[:2]))
    assert not set(condition.base.feature_names[:2]) & set(condition.base.log_features)
    mean = torch.as_tensor(condition.means[:2], device=x.device, dtype=torch.float64)
    scale = torch.as_tensor(condition.scales[:2], device=x.device, dtype=torch.float64)
    e = x[:, :2].double() * scale + mean
    eps = torch.complex(e[:, 0], e[:, 1])
    out = [x]
    for axis in (1, 2):
        for sign in (1, -1):
            g = complex(sign * .02, 0.) if axis == 1 else complex(0., sign * .02)
            v = (eps + g) / (1 + np.conj(g) * eps)
            q = x.clone()
            q[:, :2] = ((torch.stack((v.real, v.imag), -1) - mean) / scale).to(x.dtype)
            out.append(q)
    return out

def transport(model, z, x):
    return model.base.base.forward(z, x)[0]

def loss_parts(student_values, teacher_values):
    base = (student_values[0] - teacher_values[0]).double().square().mean()
    response = torch.stack([
        ((student_values[p] - student_values[p+1]).double() -
         (teacher_values[p] - teacher_values[p+1]).double()).square().mean() / .04**2
        for p in (1, 3)]).mean()
    return base, response

def main():
    assert os.environ.get('SLURM_JOB_ID')
    c.verify(); cfg = c.config(); arm = cfg['arms'][int(os.environ['SLURM_ARRAY_TASK_ID'])]
    gate = json.loads((c.ROOT/'numerical_passed.json').read_text())
    assert gate['manifest_sha256'] == c.sha(H/'manifest.json')
    tc = cfg['training']; out = c.ROOT / arm['name']; out.mkdir(exist_ok=True)
    provenance = dict(manifest=c.sha(H/'manifest.json'), sources=c.sha(H/'source_pins.json'), arm=arm)
    if (out/'completed.json').exists():
        receipt = json.loads((out/'completed.json').read_text())
        assert receipt['provenance'] == provenance and c.sha(out/'final.pt') == receipt['sha256']
        print('TRAIN_ALREADY_COMPLETE', arm['name'], flush=True); return
    torch.set_num_threads(4); torch.manual_seed(tc['seed']); torch.cuda.manual_seed_all(tc['seed'])
    assert c.sha(cfg['teacher']['path']) == cfg['teacher']['sha256']
    teacher, condition, target, artifact = c.old.load(cfg['teacher']['path'])
    teacher.requires_grad_(False)
    student = m.initialize(teacher, arm); student.requires_grad_(True)
    # Physical transforms/buffers are inherited unchanged; only coupling networks train.
    parameters = list(student.parameters())
    assert all(name.startswith('base.base.') for name, _ in student.named_parameters())
    prep = Path(cfg['prepared_root'])
    assert c.sha(prep/'completed.json') == cfg['prepared_receipt_sha256']
    # 2.7GB original training contexts; no held-out rows are used for gradients.
    x = torch.from_numpy(np.load(prep/'train/context.npy')).to('cuda')
    vc = torch.from_numpy(np.load(prep/'validation/nll_context.npy')).to('cuda')
    vt = torch.from_numpy(np.load(Path(cfg['original_prepared_root'])/'validation/nll_target.npy')).to('cuda')
    assert x.dtype == torch.float32 and x.shape[1] == teacher.context_dim
    print('DATA_READY', arm, 'rows', len(x), 'gpu', torch.cuda.get_device_name(), flush=True)
    checks = verify(student, teacher, condition, target, arm, x[:32])
    counts = {'teacher': sum(p.numel() for p in teacher.parameters()), 'student': sum(p.numel() for p in parameters)}
    c.write(out/'startup.json', dict(arm=arm, config=artifact['config'], counts=counts, checks=checks,
                                  gpu=torch.cuda.get_device_name(), provenance=provenance))
    optimizer = torch.optim.AdamW(parameters, lr=tc['learning_rate'], weight_decay=0.)
    history = []; start_step = 0
    if (out/'resume.pt').exists():
        resume = torch.load(out/'resume.pt', map_location='cuda', weights_only=False)
        assert resume['provenance'] == provenance
        student.load_state_dict(resume['model']); optimizer.load_state_dict(resume['optimizer'])
        history = resume['history']; start_step = resume['step']
    started = time.monotonic()
    for step in range(start_step, tc['steps']):
        seed = tc['seed'] + 100003 * step
        rng = np.random.default_rng(seed)
        rows = torch.as_tensor(rng.integers(len(x), size=tc['rows']), device='cuda')
        z = torch.from_numpy(c.portable.latent_bank(tc['rows'], tc['draws'], 4, seed+10000000)).to('cuda')
        optimizer.zero_grad(set_to_none=True); values = np.zeros(2)
        for lo in range(0, tc['rows'], tc['chunk']):
            hi = min(lo+tc['chunk'], tc['rows']); n = hi-lo
            ctx = x[rows[lo:hi]][:, None].expand(n, tc['draws'], x.shape[1]).reshape(-1, x.shape[1])
            latent = z[lo:hi].reshape(-1, 4)
            frames = sheared_contexts(ctx, condition)
            with torch.no_grad(): tv = [transport(teacher, latent, f) for f in frames]
            sv = [transport(student, latent, m.contexts(f, arm)) for f in frames]
            base, response = loss_parts(sv, tv)
            loss = (base + tc['response_weight'] * response) * n / tc['rows']
            assert torch.isfinite(loss)
            loss.backward(); values += [float(base.detach())*n/tc['rows'], float(response.detach())*n/tc['rows']]
        norm = torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
        optimizer.step()
        history.append(dict(step=step+1, transport=float(values[0]), response=float(values[1]), gradient_norm=float(norm)))
        if step == start_step or (step+1) % 32 == 0:
            print('PROGRESS', arm['name'], history[-1], 'seconds', round(time.monotonic()-started, 1), flush=True)
        if (step+1) % 128 == 0 or step+1 == tc['steps']:
            temp = out/'resume.tmp'
            torch.save(dict(provenance=provenance, step=step+1, model=student.state_dict(), optimizer=optimizer.state_dict(), history=history), temp)
            temp.replace(out/'resume.pt')
    # Fixed endpoint, never selected by held-out likelihood or shear outcomes.
    m.save(out/'final.pt', student, condition, target, artifact['config'], arm, provenance)
    loaded, _, _, _ = m.load(out/'final.pt')
    z = torch.from_numpy(c.portable.latent_bank(128, 4, 4, 202611081901)).to('cuda').reshape(-1, 4)
    ctx = vc[:len(z)]
    with torch.no_grad():
        expected = student.forward_physical(z, m.contexts(ctx, arm))[0]
        actual = loaded.forward_physical(z, m.contexts(ctx, arm))[0]
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        probe = diagnostic(student, teacher, condition, target, arm, vc, vt, tc['validation_rows'])
    c.write(out/'history.json', history)
    c.write(out/'completed.json', dict(provenance=provenance, sha256=c.sha(out/'final.pt'), reload_exact=True,
                                     counts=counts, diagnostic=probe, seconds=time.monotonic()-started))
    print('TRAIN_COMPLETE', arm['name'], probe, flush=True)

def verify(student, teacher, condition, target, arm, x):
    z = torch.from_numpy(c.portable.latent_bank(len(x), 2, 4, 202611081701)).to(x.device)[:, 0]
    frames = sheared_contexts(x, condition)
    e = x[:, :2].double().cpu().numpy()*condition.scales[:2]+condition.means[:2]
    for j, (axis, sign) in enumerate([(1,1),(1,-1),(2,1),(2,-1)], 1):
        expected = np.column_stack(apply_shear_to_ellipticity(e[:,0], e[:,1], .02*sign if axis==1 else 0., .02*sign if axis==2 else 0.))
        actual = frames[j][:,:2].double().cpu().numpy()*condition.scales[:2]+condition.means[:2]
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    with torch.no_grad():
        xx = m.contexts(x, arm)
        y, ld = student.base.base.forward(z, xx)
        zz, ild = student.base.base.inverse(y, xx)
        torch.testing.assert_close(z, zz, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(ld+ild, torch.zeros_like(ld), rtol=0, atol=1e-4)
        physical, _ = student.forward_physical(z, xx)
        assert torch.isfinite(physical).all() and (physical[:,:2].square().sum(-1)<1+1e-12).all() and (physical[:,2:]>0).all()
        # Removing the score must equal evaluating the teacher at standardized score zero
        # before truncation (for depth 5), including finite-density consistency.
        if not arm['score'] and arm['depth']==5:
            zero = x.clone(); zero[:,-1]=0
            torch.testing.assert_close(physical, teacher.forward_physical(z, zero)[0], rtol=0, atol=0)
    # Chunked and direct gradients on an uneven final chunk.
    student.zero_grad(set_to_none=True)
    with torch.no_grad(): tv = [transport(teacher,z,f) for f in frames]
    sv = [transport(student,z,m.contexts(f,arm)) for f in frames]
    a,b=loss_parts(sv,tv); (a+.1*b).backward()
    direct=[p.grad.detach().clone() for p in student.parameters()]
    student.zero_grad(set_to_none=True)
    for lo in range(0,len(x),13):
        hi=min(lo+13,len(x)); sv=[transport(student,z[lo:hi],m.contexts(f[lo:hi],arm)) for f in frames]
        a,b=loss_parts(sv,[v[lo:hi] for v in tv]); ((a+.1*b)*(hi-lo)/len(x)).backward()
    error=sum(float((p.grad-g).double().square().sum()) for p,g in zip(student.parameters(),direct))
    norm=sum(float(g.double().square().sum()) for g in direct)
    relative=(error/max(norm,1e-30))**.5
    assert relative<3e-4,relative
    student.zero_grad(set_to_none=True)
    return dict(shear_numpy_replay=True, inverse_max_abs=float((zz-z).abs().max()), physical_support=True,
                uneven_chunk_gradient_relative_error=relative)

def diagnostic(student, teacher, condition, target, arm, vc, vt, count):
    # Fixed validation rows; no gradients or endpoint selection.
    ids=np.random.default_rng(202611081801).choice(len(vc),count,replace=False)
    result=[]
    for lo in range(0,count,256):
        ix=torch.as_tensor(ids[lo:lo+256],device='cuda'); x=vc[ix]; y=vt[ix]
        z=torch.from_numpy(c.portable.latent_bank(len(x),4,4,202611081802+lo)).to('cuda')
        xx=x[:,None].expand(len(x),4,x.shape[1]).reshape(-1,x.shape[1]);zz=z.reshape(-1,4)
        frames=sheared_contexts(xx,condition)
        tv=[transport(teacher,zz,f) for f in frames];sv=[transport(student,zz,m.contexts(f,arm)) for f in frames]
        a,b=loss_parts(sv,tv)
        lp=teacher.log_prob(y,x);ls=student.log_prob(y,m.contexts(x,arm))
        assert torch.isfinite(ls).all()
        result.append([len(x),float(a),float(b),float((-lp).mean()),float((-ls).mean())])
    r=np.array(result);v=(r[:,0,None]*r[:,1:]).sum(0)/r[:,0].sum()
    return dict(rows=count,transport_mse=v[0],response_mse=v[1],teacher_nll=v[2],student_nll=v[3])

if __name__=='__main__': main()
