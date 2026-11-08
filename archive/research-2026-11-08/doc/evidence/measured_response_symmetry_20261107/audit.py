"""Measured response symmetry and cross-axis configuration checks; no model fit."""
import configparser,csv,hashlib,json,os
from pathlib import Path
import numpy as np
from astropy.io import fits
H=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')
BASE=Path('/project/ls-gruen/users/zekang.zhang/sbsi/galsbi_resim')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def values(s,w):
    q=np.einsum('bc,caltd->baltd',w,s)
    means=q[...,1:]/q[...,:1]
    matrix=((means[:,:,0]-means[:,:,1])/.04).transpose(0,2,3,1)
    return {'matrix':matrix,'diagonal_difference':matrix[...,0,0]-matrix[...,1,1],
            'offdiagonal_difference':matrix[...,0,1]-matrix[...,1,0]}
def entry(v,n):
    jk=v[-n:]
    return dict(value=v[0].tolist(),jackknife_se=np.sqrt((n-1)/n*np.square(jk-jk.mean(0)).sum(0)).tolist(),bootstrap_se=v[1:10001].std(0,ddof=1).tolist(),bootstrap_ci95=np.quantile(v[1:10001],[.025,.975],axis=0).tolist())
def main():
    assert os.environ.get('SLURM_JOB_ID')
    cfg=read(H.parent/'current_measured_photometry_20261107/manifest.json');cases=cfg['cases'];cuts=[v[0] for v in cfg['cuts']]
    sim=[];files=[];settings=[];psf_reference=None;psf_max_abs=0.;config_changes=set()
    for case in cases:
        legs=[]
        for axis in [1,2]:
            p=ROOT/'current_measured_photometry_20261107'/f'axis{axis}/case{case}.npz';rec=read(p.with_suffix('.json'));assert sha(p)==rec['data_sha256']
            with np.load(p) as z:np.testing.assert_array_equal(z['cuts'],cuts);legs.append(z['sim'].sum(-2))
            files.append({'path':str(p),'sha256':rec['data_sha256']})
        sim.append(legs)
        for leg in ['0.02','-0.02']:
            dirs=[BASE/'constant_g002/sims',BASE/'constant_axis2_20261105/sims']
            configs=[]
            for d in dirs:
                p=d/f'sim_config_case{case}_{leg}.ini';c=configparser.ConfigParser();c.read(p)
                configs.append({sec:dict(c[sec]) for sec in c.sections()})
                f=d/f'case{case}_{leg}/real0/images/original/psf_tile180.0_-0.5_bandr/psf_ima.fits'
                psf=fits.getdata(f).astype(float)
                if psf_reference is None:psf_reference=psf
                assert psf.shape==psf_reference.shape
                psf_max_abs=max(psf_max_abs,float(np.max(np.abs(psf-psf_reference))))
                settings.append({'case':case,'leg':leg,'root':str(d),'config_sha256':sha(p),'psf_sha256':sha(f)})
            assert configs[0].keys()==configs[1].keys()
            changed=[]
            for section in configs[0]:
                assert configs[0][section].keys()==configs[1][section].keys()
                for key,v in configs[0][section].items():
                    other=configs[1][section][key]
                    if v!=other:
                        changed.append([section,key]);config_changes.add((section,key))
                        assert v.replace(str(dirs[0]),str(dirs[1]))==other,(case,leg,section,key)
            rec=read(dirs[1].parent/f'case{case}_{leg}_verified.json');assert rec['non_shear_truth_replay']=='passed' and rec['g1']==0 and rec['g2']==float(leg)
    assert config_changes<={('Paths','out_dir'),('Paths','tmp_dir'),('GalInfo','cata_file'),('NoiseInfo','cata_file')}
    assert sha(BASE/'constant_g002/sims/noise.csv')==sha(BASE/'constant_axis2_20261105/sims/noise.csv')
    noise=list(csv.DictReader((BASE/'constant_g002/sims/noise.csv').open()))
    assert all(float(x['seeing_e1_r'])==float(x['seeing_e2_r'])==0 for x in noise)
    sim=np.array(sim);out={'source_sha256':sha(__file__),'job':os.environ['SLURM_JOB_ID'],'files':files,'configuration_checks':{'records':settings,'allowed_path_changes':sorted(config_changes),'noise_sha256':sha(BASE/'constant_g002/sims/noise.csv'),'noise_parameters':noise[0],'psf_max_pixel_difference_all_cases_legs_axes':psf_max_abs,'truth_axis_verifications':80},'cohorts':{}}
    for cohort,ix in [('original_four',list(range(4))),('additional_thirty_six',list(range(4,40))),('all_forty',list(range(40)))]:
        n=len(ix);w=np.vstack([np.ones(n),np.random.default_rng(202611071940).multinomial(n,np.ones(n)/n,10000),1-np.eye(n)])
        v=values(sim[ix],w);rows={cut:{k:entry(q[:,ci],n) for k,q in v.items()} for ci,cut in enumerate(cuts)}
        central=cuts.index('mag25_radius_gt0.60');jk=v['diagonal_difference'][-n:,central]
        percase=values(sim[ix],np.eye(n));out['cohorts'][cohort]={'cases':[cases[i] for i in ix],'cuts':rows,'central_delete_one_diagonal_range':[float(jk.min()),float(jk.max())],'per_case_central_diagonal_difference':percase['diagonal_difference'][:,central].tolist(),'positive_diagonal_cases':int((percase['diagonal_difference'][:,central]>0).sum())}
    (H/'report.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    for co,d in out['cohorts'].items():print(co,d['cuts']['mag25_radius_gt0.60'],'leaveone',d['central_delete_one_diagonal_range'],'positive',d['positive_diagonal_cases'])
    print('PSF_MAX_ABS',psf_max_abs,'PATH_CHANGES',sorted(config_changes),flush=True)
    print('DONE_MEASURED_SYMMETRY',flush=True)
if __name__=='__main__':main()
