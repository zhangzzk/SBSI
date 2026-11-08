import os,json,time,hashlib
from pathlib import Path
import numpy as np
import xgboost as xgb
assert os.environ.get('SLURM_JOB_ID')
D=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches');R=D/'flow_photometry_information_20261107'
BASE=Path('/home/z/Zekang.Zhang/Documents/Codex/2026-11-07/size-and-flux-supplied-to-model')
DOMAIN=D/'galsbi_f8c8_20261102/harmonized/domain';F=D/'neighbour_conditioned_flow_20261106/features'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
dm=json.loads((DOMAIN/'manifest.json').read_text());rng=np.random.default_rng(202611071901)
traincases=rng.permutation(dm['split']['train_cases']).tolist();split={'train':traincases[32:],'validation':traincases[:32],'test':dm['split']['validation_cases']}
assert [len(split[k]) for k in split]==[128,32,40]
assert not any(set(split[a])&set(split[b]) for a,b in [('train','validation'),('train','test'),('validation','test')])
start=time.monotonic();data={};receipts=[]
for name,cases in split.items():
    xs=[];ys=[];ws=[];cs=[];ids=[];shuffles=[];flows=[]
    for k,case in enumerate(cases):
        src=DOMAIN/'flow/g0'/f'case{case:03d}.npz';assert sha(src)==dm['flow']['g0_cases'][str(case)]['sha256']
        with np.load(src) as z:raw=z['context'];target=z['target'];ii=z['input_index'];gamma=z['gamma']
        assert np.all(gamma==0) and len(np.unique(ii))==len(ii)
        if name=='test':
            with np.load(D/'flow_truth_diagnostics_20261107'/f'case{case}.npz') as z:
                selected=z['input_index'];order=np.argsort(ii);loc=order[np.searchsorted(ii[order],selected)]
                np.testing.assert_array_equal(ii[loc],selected);np.testing.assert_array_equal(raw[loc],z['context']);flow=z['predicted'][:,:,[0,1,4]].transpose(1,0,2);flows.append(flow)
                np.testing.assert_allclose(target[loc,2],z['measured'][:,0],rtol=0,atol=0)
        else:loc=np.sort(np.random.default_rng(202611071902+case).choice(len(ii),2048,replace=False))
        source=F/f'case{case}.npz';receipt=json.loads(source.with_suffix('.json').read_text());assert sha(source)==receipt['data_sha256'] and receipt['unmatched_parents']==0
        with np.load(source) as z:fid=z['input_index'];score=z['score']
        assert np.all(fid[:-1]<fid[1:]);j=np.searchsorted(fid,ii[loc]);np.testing.assert_array_equal(fid[j],ii[loc]);B=score[j]
        xx=np.column_stack([raw[loc],B]);t=target[loc];assert np.all(t[:,2:]>0)
        yy=np.column_stack([t[:,2],30-2.5*np.log10(t[:,3]),(t[:,2]>3)&(t[:,3]>100)])
        assert np.isfinite(xx).all() and np.isfinite(yy).all()
        xs.append(xx);ys.append(yy);ws.extend([len(ii)/len(loc)]*len(loc));cs.extend([case]*len(loc));ids.append(ii[loc]);shuffles.append(np.random.default_rng(202611071904+case).permutation(B))
        receipts.append(dict(split=name,case=case,sample=len(loc),population=len(ii),domain_sha256=dm['flow']['g0_cases'][str(case)]['sha256'],feature_sha256=receipt['data_sha256'],unmatched=0))
        if (k+1)%16==0 or k==len(cases)-1:print('PREP',name,k+1,'/',len(cases),round(time.monotonic()-start,1),flush=True)
    data[name]=dict(x=np.concatenate(xs),y=np.concatenate(ys),w=np.array(ws),case=np.array(cs),ids=np.concatenate(ids),shuffled=np.concatenate(shuffles))
    if flows:data[name]['flow']=np.concatenate(flows)
    np.savez_compressed(R/f'{name}.npz',**data[name])
write(R/'prepared.json',dict(split=split,receipts=receipts,source_sha256=sha(__file__),protocol_sha256=sha(BASE/'outputs/information_test/protocol.md')))
print('PREPARED', {k:len(v['x']) for k,v in data.items()},flush=True)
arms=[('existing_d4',8,4,False),('existing_d8',8,8,False),('score_d4',9,4,False),('score_d8',9,8,False),('shuffled_d8',9,8,True)]
Y=['radius','magnitude','joint_cut'];models=R/'models';models.mkdir(exist_ok=True);results=[]
for name,width,depth,shuffle in arms:
    xx={k:d['x'][:,:width].copy() for k,d in data.items()}
    if shuffle:
        for k,d in data.items():xx[k][:,-1]=d['shuffled']
    # Reuse feature binning across target fits.
    mats={k:xgb.DMatrix(v,weight=data[k]['w']/data[k]['w'].mean(),nthread=8) for k,v in xx.items()}
    for q,targetname in enumerate(Y):
        path=models/f'{name}_{targetname}.json';recpath=models/f'{name}_{targetname}_receipt.json';predpath=models/f'{name}_{targetname}_test.npy'
        for k,d in data.items():mats[k].set_label(d['y'][:,q])
        params=dict(objective='binary:logistic' if q==2 else 'reg:squarederror',eval_metric='logloss' if q==2 else 'rmse',tree_method='hist',max_bin=256,max_depth=depth,eta=.05,min_child_weight=50,reg_lambda=5,subsample=1.,colsample_bytree=1.,seed=202611071903,nthread=8,device='cpu')
        print('FIT',name,targetname,round(time.monotonic()-start,1),flush=True)
        history={};model=xgb.train(params,mats['train'],num_boost_round=500,evals=[(mats['train'],'train'),(mats['validation'],'validation')],early_stopping_rounds=40,evals_result=history,verbose_eval=100)
        best=int(model.best_iteration);model=model[:best+1]
        pred=model.predict(mats['test'],output_margin=q==2).astype(float)
        assert np.isfinite(pred).all();model.save_model(path)
        reload=xgb.Booster(model_file=path);reload.set_param({'nthread':8});np.testing.assert_array_equal(pred,reload.predict(mats['test'],output_margin=q==2).astype(float))
        rec=dict(arm=name,target=targetname,depth=depth,width=width,shuffled=shuffle,best_rounds=best+1,validation_loss=float(history['validation'][params['eval_metric']][best]),model_sha256=sha(path),reload_exact=True,params=params)
        np.save(predpath,pred);write(recpath,rec);write(models/f'{name}_{targetname}_history.json',history);results.append(rec)
        print('FIT_DONE',name,targetname,best+1,'seconds',round(time.monotonic()-start,1),flush=True)
write(R/'completed.json',dict(results=results,split=split,receipts=receipts,xgboost=xgb.__version__,seconds=time.monotonic()-start,job=os.environ['SLURM_JOB_ID'],source_sha256=sha(__file__),protocol_sha256=sha(BASE/'outputs/information_test/protocol.md')))
print('ALL FITS COMPLETE',flush=True)
