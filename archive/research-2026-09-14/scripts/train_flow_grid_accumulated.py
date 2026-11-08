"""NLL-parent grid-response trials with larger effective batches and replay.

Use the scheduler for real data. All epochs, including the unchanged parent,
are retained for independent validation; the best-file selector is a noisy
screening score, not an accuracy certificate.
"""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from sbsi.flow_grid_accumulation import accumulated_epoch, evaluate_validation_groups
from sbsi.flow_grid_response import GridResponsePopulation
from sbsi.measurement_model import (
    build_flow, save_measurement_model, TargetStandardizer, _validate_disk_contract,
)
from sbsi.selection_model import TabularPreprocessor
from sbsi.training import GPUBatches, epoch_nll
from scripts.train_flow_grid_response import DEFINITION, digest, load_torch


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--initial-model',help='completed NLL-only disk refit; default is the original prepared affine parent')
    parser.add_argument('--raw-root',help='verified original float64 recovery output, required for a disk initial model')
    parser.add_argument('--weight',type=float,required=True)
    parser.add_argument('--mode',choices=['sampled','full'],required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--epochs',type=int,default=3)
    parser.add_argument('--patience',type=int,default=15)
    parser.add_argument('--nll-batch-size',type=int,default=131072)
    parser.add_argument('--nll-chunk-size',type=int,default=4096)
    parser.add_argument('--cells-per-step',type=int,default=16)
    parser.add_argument('--pairs-per-cell',type=int,default=1024)
    parser.add_argument('--response-chunk-size',type=int,default=256)
    parser.add_argument('--draws',type=int,default=16)
    parser.add_argument('--validation-draws',type=int,default=64)
    parser.add_argument('--validation-groups',type=int,default=4)
    parser.add_argument('--validation-batch-size',type=int,default=1024)
    parser.add_argument('--validation-seed',type=int,default=7301)
    parser.add_argument('--seed',type=int,default=501)
    parser.add_argument('--lr',type=float,default=1e-5)
    parser.add_argument('--max-grad-norm',type=float,default=5.)
    parser.add_argument('--max-val-nll-increase',type=float,default=.05)
    parser.add_argument('--disable-nll-guard',action='store_true',
                        help='select and stop on the combined objective without an NLL ceiling')
    parser.add_argument('--max-tail-rate-ratio',type=float,default=4.)
    parser.add_argument('--max-train-nll-rows',type=int)
    parser.add_argument('--response-pairs-per-epoch',type=int,help='with-replacement occurrences per independent group')
    parser.add_argument('--validation-nll-rows',type=int)
    parser.add_argument('--validation-pairs',type=int)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args(argv)
    positive = ('epochs','patience','nll_batch_size','nll_chunk_size','cells_per_step',
                'pairs_per_cell','response_chunk_size','validation_batch_size')
    if (any(getattr(args,k) < 1 for k in positive)
            or any(not math.isfinite(v) or v <= 0 for v in (args.weight,args.lr,args.max_grad_norm))
            or any(v < 2 or v%2 for v in (args.draws,args.validation_draws))
            or args.validation_groups < 4 or args.validation_groups%2
            or not math.isfinite(args.max_val_nll_increase) or args.max_val_nll_increase < 0
            or not math.isfinite(args.max_tail_rate_ratio) or args.max_tail_rate_ratio < 1
            or min(args.seed,args.validation_seed) < 0):
        parser.error('positive training options, even draws and an even validation group count >=4 required')
    caps = (args.max_train_nll_rows,args.response_pairs_per_epoch,args.validation_nll_rows,args.validation_pairs)
    if any(v is not None and v < 1 for v in caps):
        parser.error('data caps must be positive')
    if any(v is not None for v in caps) and args.mode != 'sampled':
        parser.error('data caps require sampled mode')
    return args


def nll_initialization(args, manifest):
    """Select an NLL-only checkpoint while preserving the original data contract.

    The prepared grid/covariance remains immutable. A disk refit changes only
    the starting density and the precision of the original NLL target values;
    its physical target/context coordinate system must be the same.
    """
    original = load_torch(manifest['parent'])
    if original['model_config']['flow_type'] != 'affine':
        raise ValueError('prepared data must retain the original plain affine NLL parent')
    original_path = Path(manifest['parent'])
    initial = getattr(args, 'initial_model', None)
    raw_root = getattr(args, 'raw_root', None)
    path = Path(initial) if initial else original_path
    reference = load_torch(path) if initial else original
    config, metadata = reference['model_config'], reference.get('metadata', {})
    kind = config['flow_type']
    if kind not in ('affine', 'disk_affine') or config.get('mean_hidden', 0):
        raise ValueError('plain affine or disk-affine NLL parent required')
    prior_weights = ('response_weight', 'coupling_weight', 'grid_response_weight', 'ratio_response_weight',
                     'paired_transport_weight', 'distribution_response_weight', 'response_dispersion_weight',
                     'ra_weight', 'preserve_weight')
    if any(float(candidate.get('metadata', {}).get(k, 0) or 0) != 0
           for candidate in (original, reference) for k in prior_weights):
        raise ValueError('parent metadata contains a nonzero response regularizer; NLL-only parent required')
    hashes = {str(original_path): manifest['parent_sha256']}
    if initial:
        hashes[str(path)] = digest(path)
    recovery = None
    if kind == 'affine':
        if initial or raw_root:
            raise ValueError('an explicit initial model/raw recovery is supported only for a disk NLL refit')
        return reference, path, hashes, recovery, prior_weights
    if not initial or not raw_root:
        raise ValueError('disk NLL parent requires explicit --initial-model and --raw-root')
    if metadata.get('training_phase') != 'NLL_only_disk_refit' or metadata.get('epoch', 0) < 1:
        raise ValueError('completed NLL-only disk refit required; coordinate initialization/probes are not a baseline')
    if metadata.get('nll_weight') != 1 or metadata.get('parent_sha256') != manifest['parent_sha256']:
        raise ValueError('disk baseline must descend from the original pure-NLL parent')
    # The only allowed target metadata extension is its explicit dtype. These
    # checks also keep all original photometry standardizations and input names.
    for name in ('condition_preprocessor', 'target_transform'):
        actual, expected = reference[name], original[name]
        fields = set(actual)
        if name == 'target_transform':
            fields.discard('dtype')
            if actual.get('dtype') != 'float64':
                raise ValueError('disk baseline requires float64 physical targets')
        if fields != set(expected):
            raise ValueError('disk baseline changes serialized physical/context transform fields')
        if any(not np.array_equal(np.asarray(actual[k]), np.asarray(expected[k])) for k in expected):
            raise ValueError('disk baseline changes original physical/context transforms')
    raw_root = Path(raw_root)
    recovery_path = raw_root/'results.json'
    recovery = json.loads(recovery_path.read_text())
    hashes[str(recovery_path)] = digest(recovery_path)
    if recovery['original_parent_sha256'] != manifest['parent_sha256']:
        raise ValueError('raw recovery belongs to a different original NLL parent')
    nll_sources = metadata.get('source_hashes', {})
    def recorded_hash(path):
        # Prepared roots can be immutable symlink aliases in a new trial.
        matches = [value for name,value in nll_sources.items()
                   if Path(name).resolve() == Path(path).resolve()]
        return matches[0] if matches and len(set(matches)) == 1 else None
    prepared_path = Path(args.root)/'manifest.json'
    if (recorded_hash(prepared_path) != digest(prepared_path)
            or recorded_hash(recovery_path) != hashes[str(recovery_path)]):
        raise ValueError('disk baseline was not trained on this verified prepared/raw population')
    for split in ('train', 'validation'):
        info = recovery['splits'][split]
        if (not info['all_cached_targets_and_contexts_exactly_reproduced']
                or not info['original_pair_keys_and_shear_vectors_exactly_reproduced']
                or info['raw_shape_minimum_one_minus_norm_squared'] <= 0
                or info['nll_rows'] != manifest['splits'][split]['nll_rows']):
            raise ValueError('original raw support/population verification required')
        raw_path = raw_root/(split+'_raw.npy')
        expected = recovery['files'][raw_path.name]
        if digest(raw_path) != expected or recorded_hash(raw_path) != expected:
            raise ValueError('original raw targets differ from the NLL baseline population')
        hashes[str(raw_path)] = expected
    return reference, path, hashes, recovery, prior_weights


def nll_values(payload, split, target, raw_root=None, recovery=None):
    if raw_root is None:
        if target.dtype != 'float32':
            raise ValueError('float64 disk density cannot use cached float32 targets')
        return payload['values']
    if target.dtype != 'float64' or recovery is None:
        raise ValueError('original float64 recovery and disk target transform required')
    raw = np.load(Path(raw_root)/(split+'_raw.npy'))
    if (raw.dtype != np.float64 or raw.shape != (len(payload['context']), target.dim)
            or len(raw) != recovery['splits'][split]['nll_rows']):
        raise ValueError('raw NLL target precision or row alignment changed')
    return torch.from_numpy(target.transform_array(raw))


def train(args):
    root,output = Path(args.root),Path(args.output)
    resume_path = output.with_suffix('.resume.pt')
    if not args.resume and any(p.exists() for p in (output,resume_path,output.with_suffix('.history.json'))):
        raise FileExistsError('refusing to overwrite an existing trial without --resume')
    if args.resume and not resume_path.exists():
        raise FileNotFoundError('no committed epoch snapshot to resume')
    manifest_path = root/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    prepared_sha = digest(manifest_path)
    if manifest['version'] != 1 or manifest['definition'] != DEFINITION:
        raise ValueError('unsupported prepared data')
    if digest(manifest['parent']) != manifest['parent_sha256']:
        raise ValueError('parent checkpoint changed')
    for name,expected in manifest['files'].items():
        if digest(root/name) != expected:
            raise ValueError(f'prepared data changed: {name}')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('real trials require scheduled CUDA')
    device = torch.device(args.device)
    reference, initial_path, input_hashes, recovery, prior_weights = nll_initialization(args, manifest)
    input_hashes.update({str(manifest_path): prepared_sha,
                         **{str(root/name): value for name,value in manifest['files'].items()}})
    config = dict(reference['model_config'])
    parent_metadata = reference.get('metadata',{})
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = build_flow(config).to(device)
    model.load_state_dict(reference['state_dict'],strict=True)
    condition = TabularPreprocessor.from_state(reference['condition_preprocessor'])
    target = TargetStandardizer.from_state(reference['target_transform'])
    _validate_disk_contract(model, target, config)
    if target.target_names[:2] != ['measured_ngmix_g1','measured_ngmix_g2'] or target.dim != 4:
        raise ValueError('expected shape-first four-output parent')
    populations,data = {},{}
    subset_report = {}
    for split in ('train','validation'):
        payload = load_torch(root/f'{split}.pt')
        values = nll_values(payload, split, target, getattr(args, 'raw_root', None), recovery)
        data[split] = [values.to(device), payload['context'].to(device)]
        if split == 'validation' and args.validation_pairs is not None:
            from sbsi.flow_grid_caps import validation_pair_subset
            if args.raw_root is None:
                raise ValueError('capped response validation requires original physical float64 targets')
            raw_shapes = np.load(Path(args.raw_root)/'validation_raw.npy',mmap_mode='r')
            payload,subset_report = validation_pair_subset(payload,raw_shapes,args.validation_pairs,args.seed+900000019)
        populations[split] = GridResponsePopulation(data[split][1],payload,target.scales[:2])
        del payload
    validation_data = data['validation']
    if args.validation_nll_rows is not None:
        if args.validation_nll_rows > len(validation_data[0]):
            raise ValueError('validation NLL cap exceeds available rows')
        indices = np.sort(np.random.default_rng(args.seed+900000003).choice(
            len(validation_data[0]),args.validation_nll_rows,replace=False))
        subset_report['selected_nll_indices'] = indices
        validation_data = [a[torch.as_tensor(indices,device=device)] for a in validation_data]
    if args.max_train_nll_rows is not None and args.max_train_nll_rows > len(data['train'][0]):
        raise ValueError('training NLL cap exceeds available rows')
    if subset_report:
        output.parent.mkdir(parents=True,exist_ok=True)
        subset_path = output.with_suffix('.validation_subset.npz')
        if args.resume:
            with np.load(subset_path) as saved:
                if set(saved.files) != set(subset_report) or any(not np.array_equal(saved[k],v) for k,v in subset_report.items()):
                    raise ValueError('fixed validation subset changed on resume')
        else:
            if subset_path.exists(): raise FileExistsError('refusing to overwrite fixed validation subset')
            np.savez(subset_path,**subset_report)
        input_hashes[str(subset_path)] = digest(subset_path)
    validation_loader = GPUBatches(validation_data,args.nll_chunk_size,False,device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0)
    metadata = dict(parent_metadata)
    metadata.update(grid_response_weight=args.weight,grid_loss_reduction='mean',
                    grid_loss_divisor=4*manifest['n_cells'],parent=str(initial_path),
                    parent_sha256=input_hashes[str(initial_path)],prepared_manifest_sha256=prepared_sha,
                    original_nll_parent=manifest['parent'],original_nll_parent_sha256=manifest['parent_sha256'],
                    input_hashes=input_hashes,training_phase='grid_response_finetune',
                    nll_target_precision=target.dtype,
                    arguments=vars(args),response_estimator='complete_flow',response_components='matrix',
                    response_difference='forward',response_delta=.05,response_draws=args.draws,
                    grid_response_definition={**DEFINITION,
                        'loss':'NLL + lambda/(2*4*n_cells) * sum_B (mu_A-target)^T precision_B (mu_C-target)',
                        'cell_weighting':'mean over cells and four components; no occupancy weighting',
                        'stochastic_estimator':('all galaxies per selected cell; independent latent groups; exact replay'
                                                if args.mode == 'full' else
                                                'independent sampled galaxies and latent groups; exact replay'),
                        'validation':('fixed random matched-pair subset; paired subset WLS; original precision weights; independent latent groups'
                                      if args.validation_pairs is not None else
                                      'all matched held-out pairs; all distinct cross-products of independent integration groups')},
                    selection_interpretation='noisy validation screening; every epoch retained for independent audit',
                    population=manifest['splits'],
                    data_caps=dict(train_nll_rows=args.max_train_nll_rows,
                        response_pair_occurrences_per_group=args.response_pairs_per_epoch,
                        validation_nll_rows=args.validation_nll_rows,validation_pairs=args.validation_pairs),
                    validation_population_definition=('fixed random subset, paired measured/model WLS; original full-validation precision weights'
                        if args.validation_pairs is not None else 'original full validation population'))
    for key in prior_weights:
        if key != 'grid_response_weight':
            metadata[key] = 0.
    output.parent.mkdir(parents=True,exist_ok=True)
    def epoch_path(epoch):
        return output.with_name(f'{output.stem}_epoch{epoch:03d}.pt')
    def copy_best(epoch):
        temporary = output.with_name(output.name+'.tmp')
        shutil.copyfile(epoch_path(epoch),temporary)
        os.replace(temporary,output)
    history,best,best_epoch,wait,first_epoch = [],float('inf'),0,0,0
    elapsed_before = 0.
    if args.resume:
        state = load_torch(resume_path)
        if state['history'][-1].get('stopped_on_guard',False):
            raise ValueError('guard-stopped trial requires a reviewed new recipe, not an automatic resume')
        allowed = {'epochs','resume'}
        if any(v != vars(args).get(k) for k,v in state['arguments'].items() if k not in allowed):
            raise ValueError('resume options changed; only --epochs may increase')
        if state['prepared_manifest_sha256'] != prepared_sha:
            raise ValueError('prepared manifest changed since committed epoch')
        if (state.get('input_hashes', input_hashes) != input_hashes
                or (getattr(args, 'initial_model', None) and 'input_hashes' not in state)):
            raise ValueError('NLL initialization or raw inputs changed since committed epoch')
        if args.epochs <= state['epoch']:
            raise ValueError('--epochs must exceed the committed epoch')
        model.load_state_dict(state['state_dict'],strict=True)
        optimizer.load_state_dict(state['optimizer'])
        history,best,best_epoch,wait = (state[k] for k in ('history','best','best_epoch','wait'))
        first_epoch = state['epoch']+1
        elapsed_before = history[-1]['elapsed_seconds']
        torch.set_rng_state(state['rng'])
        if args.device == 'cuda':
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        copy_best(best_epoch)  # recover a best pointer written after a partial epoch
    started = time.monotonic()
    for epoch in range(first_epoch,args.epochs+1):
        training = None
        if epoch:
            training = accumulated_epoch(model,*data['train'],populations['train'],optimizer,
                weight=args.weight,mode=args.mode,nll_batch_size=args.nll_batch_size,
                nll_chunk_size=args.nll_chunk_size,cells_per_step=args.cells_per_step,
                pairs_per_cell=args.pairs_per_cell,draws=args.draws,response_chunk_size=args.response_chunk_size,
                seed=args.seed+epoch*1000000007,max_grad_norm=args.max_grad_norm,
                max_nll_rows=args.max_train_nll_rows,response_pairs_per_epoch=args.response_pairs_per_epoch)
        model.eval()
        val_nll = epoch_nll(model,validation_loader,device)
        validation = evaluate_validation_groups(model,populations['validation'],args.validation_draws,
                                               args.validation_groups,args.validation_batch_size,args.validation_seed,
                                               shape_means=target.means[:2])
        objective = val_nll+args.weight*validation['loss']
        if not np.isfinite(objective):
            raise ValueError('nonfinite validation objective')
        record = dict(epoch=epoch,val_nll=val_nll,val_grid=validation['loss'],val_total=objective,
                      validation_nll_rows=len(validation_data[0]),
                      validation_pairs=len(populations['validation'].pairs),
                      val_grid_mc_jackknife_se=validation['mc_jackknife_se'],
                      val_mean_quadratic=validation['mean_quadratic'],
                      val_integration_quadratic=validation['integration_quadratic'],
                      val_tail_thresholds=validation['tail_thresholds'].tolist(),
                      val_tail_counts=validation['tail_exceedance_counts'].tolist(),
                      val_tail_draws=validation['tail_draw_count'],
                      val_max_shape_amplitude=validation['max_shape_amplitude'],
                      val_shape_support_labels=validation['shape_support_labels'].tolist(),
                      val_shape_support_counts=validation['shape_support_counts'].tolist(),
                      val_shape_squared_radius_band=validation['shape_squared_radius_band'],
                      val_max_shape_squared_radius=validation['max_shape_squared_radius'],
                      elapsed_seconds=elapsed_before+time.monotonic()-started)
        parent_record = history[0] if history else record
        nll_ok = val_nll <= parent_record['val_nll']+args.max_val_nll_increase
        # Compare the rate beyond |e|=2 with the same parent's actual draws.
        # The five-event denominator floor avoids a zero-rate ratio when the
        # parent audit is too small to resolve a very rare tail. No samples or
        # model outputs are clipped, removed or reweighted.
        rate = record['val_tail_counts'][1]/record['val_tail_draws']
        parent_rate = max(parent_record['val_tail_counts'][1],5)/parent_record['val_tail_draws']
        tail_ok = rate <= args.max_tail_rate_ratio*parent_rate
        nll_guard_enforced = not args.disable_nll_guard
        eligible = (nll_ok or not nll_guard_enforced) and tail_ok
        stopped_on_guard = bool(epoch and not eligible)
        record.update(checkpoint_eligible=eligible,nll_guard_pass=nll_ok,
                      nll_guard_enforced=nll_guard_enforced,tail_guard_pass=tail_ok,
                      stopped_on_guard=stopped_on_guard)
        if training is not None:
            norms = np.asarray(training['gradient_norms'])
            record.update(train_nll=training['nll'],train_grid=training['grid'],
                          nll_rows=training['nll_rows'],response_pairs_per_group=training['response_pairs_per_group'],
                          response_pair_coverage=training['response_pair_coverage'],optimizer_steps=training['optimizer_steps'],
                          mean_nll_rows_per_step=training['nll_rows']/training['optimizer_steps'],
                          mean_response_pairs_per_group_per_step=training['response_pairs_per_group']/training['optimizer_steps'],
                          gradient_norm_median=float(np.median(norms)),gradient_norm_max=float(norms.max()),
                          clipped_step_fraction=float(np.mean(norms>args.max_grad_norm)))
        history.append(record)
        if eligible and objective < best:
            best,best_epoch,wait = objective,epoch,0
        else:
            wait += 1
        path = epoch_path(epoch)
        temporary = path.with_name(path.name+'.tmp')
        save_measurement_model(str(temporary),model,condition,target,config,
                               metadata={**metadata,'epoch':epoch,'best_epoch':best_epoch,
                                         'best_val_objective':best,'history':history})
        os.replace(temporary,path)
        validation_path = output.with_name(f'{output.stem}_epoch{epoch:03d}_validation.npz')
        with validation_path.with_suffix('.tmp').open('wb') as f:
            np.savez(f,**validation,gradient_norms=training['gradient_norms'] if training else [])
        os.replace(validation_path.with_suffix('.tmp'),validation_path)
        if best_epoch == epoch:
            copy_best(epoch)
        state = dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,
                     history=history,best=best,best_epoch=best_epoch,wait=wait,arguments=vars(args),
                     prepared_manifest_sha256=prepared_sha,input_hashes=input_hashes,rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if args.device == 'cuda' else [])
        torch.save(state,resume_path.with_suffix('.tmp'))
        os.replace(resume_path.with_suffix('.tmp'),resume_path)
        history_path = output.with_suffix('.history.json')
        history_path.with_suffix('.tmp').write_text(json.dumps(dict(history=history,best_epoch=best_epoch,
            best_val_objective=best,no_improvement_epochs=wait,stopped_on_patience=wait>=args.patience,
            stopped_on_guard=stopped_on_guard),indent=2)+'\n')
        os.replace(history_path.with_suffix('.tmp'),history_path)
        print(json.dumps(record),flush=True)
        if wait >= args.patience or stopped_on_guard:
            break
    for path, expected in input_hashes.items():
        if digest(path) != expected:
            raise ValueError('input changed during response training: '+path)
    print(f'ACCUMULATED_GRID_TRAINING_COMPLETE epoch={epoch} best_epoch={best_epoch}',flush=True)


def main(argv=None):
    train(arguments(argv))


if __name__ == '__main__':
    main()
