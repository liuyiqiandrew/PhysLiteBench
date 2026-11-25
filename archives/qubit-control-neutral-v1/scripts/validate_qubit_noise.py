"""Validate a shared-noise channel against a product-channel reduction."""
import copy
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'tasks/qubit-control'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    OUT=ROOT/'jobs/qubit-noise-diverse-validation'
    OUT.mkdir(parents=True,exist_ok=True)
    exact=load('noise_oracle',TASK/'solution/model.py')
    short=load('noise_baseline',ROOT/'scripts/qubit_noise_baseline.py')
    ref=load('noise_reference',TASK/'tests/reference.py')
    calibration=TASK/'environment/data/calibration.json'
    assert calibration.read_bytes()==(TASK/'tests/data/calibration.json').read_bytes()
    runs=json.loads(calibration.read_text())
    (OUT/'calibration.json').write_bytes(calibration.read_bytes())
    exps=[r['experiment'] for r in runs]
    truth=ref.reference(exps)
    groups=ref.hidden_groups();refs=[ref.reference(g) for g in groups]
    for es in [exps,*groups]:np.testing.assert_allclose(exact.probabilities(es,ref.TRUE_GAMMA),ref.reference(es),atol=2e-13)
    # A Schur channel is completely positive when its multiplier is PSD;
    # unit diagonal makes it trace preserving. Test both noise models.
    charges=np.array([1,0,0,-1]);bits=np.array([[0,0],[0,1],[1,0],[1,1]])
    distances=[(charges[:,None]-charges[None,:])**2,np.sum((bits[:,None,:]-bits[None,:,:])**2,axis=-1)]
    min_eig=1.
    for distance in distances:
        for gt in np.linspace(0,10,101):
            channel=np.exp(-gt*distance)
            np.testing.assert_array_equal(np.diag(channel),np.ones(4))
            minimum=np.linalg.eigvalsh(channel).min();min_eig=min(min_eig,float(minimum));assert minimum> -1e-12
    models={n:m.QubitModel().fit(copy.deepcopy(runs)) for n,m in [('oracle',exact),('shortcut',short)]}
    controls={n:ref.metrics(m,runs,refs) for n,m in models.items()}
    for name,m in controls.items():
        assert m['relative_parameter_error']<ref.PARAMETER_LIMIT
        assert m['calibration_reduced_chi2']<ref.CHI2_LIMIT
        hidden=np.array(m['hidden_probability_rmse'])
        assert (hidden<ref.PREDICTION_LIMIT).all() if name=='oracle' else (hidden>ref.PREDICTION_LIMIT).all()
        print(name,m,flush=True)
    rng=np.random.default_rng(1729);mc={'oracle':[],'shortcut':[]}
    for i in range(256):
        noisy=copy.deepcopy(runs)
        for r,t in zip(noisy,truth):r['probability']=float(t+rng.normal(0,r['sigma']))
        for name,module in [('oracle',exact),('shortcut',short)]:
            m=ref.metrics(module.QubitModel().fit(noisy),noisy,refs)
            mc[name].append([m['relative_parameter_error'],m['calibration_reduced_chi2'],*m['hidden_probability_rmse']])
        if (i+1)%64==0:print('validated',i+1,'/256',flush=True)
    stats={}
    for name,rows in mc.items():
        v=np.array(rows);visible=(v[:,0]<.05)&(v[:,1]<1.5);good=np.all(v[:,2:]<.03,axis=1)
        assert visible.all();assert good.all() if name=='oracle' else not good.any()
        assert np.max(v[:,2:])<.005 if name=='oracle' else np.min(v[:,2:])>.05
        stats[name]={'columns':['relative_parameter_error','calibration_reduced_chi2','hidden_rmse_parallel','hidden_rmse_quadrature'],
                     'min':v.min(axis=0).tolist(),'max':v.max(axis=0).tolist(),'median':np.median(v,axis=0).tolist(),
                     'visible_passes':int(visible.sum()),'all_passes':int((visible&good).sum())}
    summary={'true_gamma':ref.TRUE_GAMMA,'calibration_file':str(calibration.relative_to(ROOT)),'calibration_samples':len(runs),'monte_carlo_seed':1729,'samples':256,
             'thresholds':{'reduced_chi2':1.5,'relative_parameter_error':.05,'probability_rmse':.03},'minimum_channel_multiplier_eigenvalue':min_eig,'controls':controls,'monte_carlo':stats}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    data={'wait':[e['wait'] for e in exps],'observed':[r['probability'] for r in runs],'hidden_truth':refs}
    for name,m in models.items():
        data['cal_'+name]=m.predict(exps);data['hidden_'+name]=np.array([m.predict(g) for g in groups]);data['mc_'+name]=np.array(mc[name])
    np.savez_compressed(OUT/'plot-data.npz',**data)

if __name__=='__main__':main()
