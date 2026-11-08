import hashlib,json,os,time
from pathlib import Path
import numpy as np
import xgboost as xgb
HERE=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/selection_information_20261106')
CASES=[1000,1040,1080,1120]
ARMS=[('existing8',8,6),('existing8_capacity',8,9),('plus_scene_score',9,6)]
def main():
    assert os.environ.get('SLURM_JOB_ID')
    fold=int(os.environ['SLURM_ARRAY_TASK_ID']);test=CASES[fold];val=CASES[(fold+1)%4]
    train=[c for c in CASES if c not in [test,val]];data={}
    for case in CASES:
        with np.load(ROOT/f'case{case}.npz') as z:data[case]={k:z[k] for k in ['x','target','flow','score','input_index']}
    dest=ROOT/f'fold{fold}';dest.mkdir(exist_ok=True);assert not (dest/'completed.json').exists()
    xt=np.concatenate([data[c]['x'] for c in train]);yt=np.concatenate([data[c]['target'] for c in train])
    predictions=[data[test]['flow']];margins=[np.full(len(data[test]['flow']),np.nan)];receipts=[];started=time.monotonic()
    print('READY',fold,'train',train,'validation',val,'test',test,'xgboost',xgb.__version__,flush=True)
    for name,width,depth in ARMS:
        dt=xgb.DMatrix(xt[:,:width],label=yt,nthread=8);dv=xgb.DMatrix(data[val]['x'][:,:width],label=data[val]['target'],nthread=8)
        de=xgb.DMatrix(data[test]['x'][:,:width],nthread=8)
        params={'objective':'reg:logistic','eval_metric':'logloss','tree_method':'hist','max_bin':256,
                'max_depth':depth,'eta':.05,'min_child_weight':50,'lambda':1,'subsample':1.,'colsample_bytree':1.,
                'seed':202611069701+fold,'nthread':8,'device':'cpu'}
        history={};print('TRAIN',fold,name,flush=True)
        model=xgb.train(params,dt,num_boost_round=600,evals=[(dt,'train'),(dv,'validation')],
                        early_stopping_rounds=50,evals_result=history,verbose_eval=50)
        best=int(model.best_iteration);model=model[:best+1]
        margin=model.predict(de,output_margin=True).astype(float);assert np.isfinite(margin).all()
        pred=1/(1+np.exp(-margin));assert np.isfinite(pred).all() and np.all((pred>=0)&(pred<=1))
        path=dest/(name+'.json');model.save_model(path);loaded=xgb.Booster(model_file=path);loaded.set_param({'nthread':8,'device':'cpu'})
        np.testing.assert_array_equal(margin,loaded.predict(de,output_margin=True).astype(float))
        validation=model.predict(dv,output_margin=True).astype(float);target=data[val]['target']
        loss=float(np.mean(np.logaddexp(0,validation)-target*validation))
        np.testing.assert_allclose(loss,float(history['validation']['logloss'][best]),rtol=0,atol=2e-6)
        predictions.append(pred);margins.append(margin);receipts.append({'arm':name,'width':width,'depth':depth,'best_iteration':best,
            'validation_logloss':loss,'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'parameters':params})
        (dest/(name+'_history.json')).write_text(json.dumps(history)+'\n')
        print('ARM_DONE',fold,name,'rounds',best+1,'seconds',round(time.monotonic()-started,1),flush=True)
    np.savez(dest/'predictions.npz',probabilities=np.array(predictions),margins=np.array(margins),target=data[test]['target'],score=data[test]['score'],
             input_index=data[test]['input_index'],case=test,arms=['flow',*[a[0] for a in ARMS]])
    out={'fold':fold,'training_cases':train,'validation_case':val,'test_case':test,'arms':receipts,
         'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'xgboost_version':xgb.__version__,
         'reload_exact':True,'seconds':time.monotonic()-started}
    (dest/'completed.json').write_text(json.dumps(out,indent=2)+'\n')
    print('DONE',fold,out['seconds'],flush=True)
if __name__=='__main__':main()
