"""Validate the current electroneutral shared-counterion task."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'reaction-diffusion'
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
ref=load('reference',TASK/'tests/reference.py')
Oracle=load('oracle',TASK/'solution/model.py').TransportModel
Shortcut=load('shortcut',ROOT/'scripts/reaction_baseline.py').TransportModel

def check(model,data,physical):
    result=ref.metrics(model,data)
    assert result['relative_diffusivity_error']<.05
    assert result['calibration_reduced_chi2']<1.5
    assert min(result['minimum_hidden_concentration'])>-1e-8
    assert max(result['maximum_mean_concentration_error'])<1e-8
    assert max(result['hidden_nrmse'])<.01 if physical else min(result['hidden_nrmse'])>.8
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--generate-data',action='store_true');args=parser.parse_args()
    out=ROOT/'jobs/reaction-validation-v6';out.mkdir(parents=True,exist_ok=True)
    x=np.linspace(0,ref.L,65);t=np.linspace(0,250,51);shape=np.cos(np.pi*x/ref.L)
    initial=[]
    for species,mean,amplitude in [(0,1.,.7),(0,.8,.5),(1,1.,.7),(1,1.4,.6)]:
        c=np.zeros((len(x),2));c[:,species]=mean+amplitude*shape;initial.append(c)
    initial=np.array(initial);physical=Oracle();physical.diffusivity=ref.D
    truth=np.array([physical.predict(t,x,c) for c in initial]);sigma=np.full_like(truth,.003)
    clean={'x':x,'t':t,'initial':initial,'sigma':sigma}
    if args.generate_data:
        np.savez(TASK/'environment/data/calibration.npz',**clean,concentration=truth+np.random.default_rng(20260921).normal(size=truth.shape)*sigma)
        shutil.copyfile(TASK/'environment/data/calibration.npz',TASK/'tests/data/calibration.npz')
    data=ref.load_data()
    controls={name:check(cls().fit(data),data,name=='oracle') for name,cls in [('oracle',Oracle),('shortcut',Shortcut)]}
    pure_error=float(np.max(abs(truth-np.array([Shortcut().fit(data).predict(t,x,c) for c in initial]))))
    physical.diffusivity=ref.D
    numerical=[]
    for ht,hx,c,expected in ref.hidden_cases():
        mean=np.trapezoid(c,hx,axis=0)/ref.L
        numerical.append((np.linalg.norm(physical.predict(ht,hx,c)-expected,axis=(0,1))/np.linalg.norm(expected-mean,axis=(0,1))).tolist())
    assert np.max(numerical)<.005
    convergence=[]
    for ht,hx,c,coarse in ref.hidden_cases():
        fine=ref.trajectory(ht,hx,c,refinement=8)
        mean=np.trapezoid(c,hx,axis=0)/ref.L
        convergence.append((np.linalg.norm(coarse-fine,axis=(0,1))/np.linalg.norm(fine-mean,axis=(0,1))).tolist())
    assert np.max(convergence)<.0001
    (out/'reference-convergence.json').write_text(json.dumps({'refinement4_vs8_species_nrmse':convergence},indent=2)+'\n')
    rng=np.random.default_rng(1729);rows={'oracle':[],'shortcut':[]}
    for index in range(256):
        trial={**clean,'concentration':truth+rng.normal(size=truth.shape)*sigma}
        # Both controls use the same exact binary calibration likelihood.
        shortcut=Shortcut().fit(trial);oracle=Oracle();oracle.diffusivity=shortcut.diffusivity
        for name,model in [('oracle',oracle),('shortcut',shortcut)]:
            result=check(model,trial,name=='oracle')
            rows[name].append([result['relative_diffusivity_error'],result['calibration_reduced_chi2'],*result['hidden_nrmse']])
        if (index+1)%64==0:print(f'Validated {index+1}/256 electrolyte controls',flush=True)
    summary={'controls':controls,'independent_reference_species_nrmse':numerical,'fixed_data_binary_fit_max_abs_error':pure_error,'monte_carlo':{}}
    for name,values in rows.items():
        a=np.array(values);summary['monte_carlo'][name]={'n':len(a),'min':a.min(axis=0).tolist(),'median':np.median(a,axis=0).tolist(),'max':a.max(axis=0).tolist(),'full_passes':int(np.sum(np.all(a[:,2:]<.05,axis=1)))}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    np.savez(out/'plot-data.npz',**data,oracle_mc=rows['oracle'],shortcut_mc=rows['shortcut'])
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
