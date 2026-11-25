"""Validate spatial Peltier heat against conservative reciprocal energy transport."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'tasks/thermoelectric-rod'


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result);return result


def settings(data,i):
    return (data['t'],data['x'],data['initial'][i],data['current'][i],data['boundary'][i],data['seebeck_span'][i])


def chi2(model,data):
    residuals=[]
    for i in range(len(data['initial'])):
        output=model.predict(*settings(data,i))
        for field in ['temperature','voltage']:
            residuals.extend(((output[field]-data[field][i])/data['sigma_'+field][i]).ravel())
    return float(np.sum(np.square(residuals))/(len(residuals)-1))


def noiseless(data,ref):
    runs=[ref.predict(*settings(data,i),ref.TRUE_CONDUCTIVITY,refinement=8) for i in range(len(data['initial']))]
    return {field:np.array([r[field] for r in runs]) for field in ['temperature','voltage']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate',action='store_true');parser.add_argument('--noise-trials',type=int,default=256)
    args=parser.parse_args();start=time.time()
    oracle=module('electric_oracle',TASK/'solution/model.py');shortcut=module('electric_shortcut',ROOT/'scripts/thermoelectric_rod_baseline.py')
    ref=module('electric_reference',TASK/'tests/reference.py')
    if args.generate:
        data=ref.calibration_settings();mean=noiseless(data,ref);rng=np.random.default_rng(31401)
        for field,sigma in [('temperature',.04),('voltage',2e-5)]:
            data['sigma_'+field]=np.full_like(mean[field],sigma)
            data[field]=mean[field]+rng.normal(size=mean[field].shape)*sigma
        path=TASK/'environment/data/calibration.npz';np.savez(path,**data);shutil.copyfile(path,TASK/'tests/data/calibration.npz')
    assert (TASK/'environment/data/calibration.npz').read_bytes()==(TASK/'tests/data/calibration.npz').read_bytes()
    data=dict(np.load(TASK/'environment/data/calibration.npz'))
    assert np.all(data['seebeck_span']==0) and np.all(data['current']!=0)
    result={'revision':2,'calibration_seed':31401,'noise_seed':41401,'noise_trials':args.noise_trials,'controls':{}}
    for name,source in [('oracle',oracle),('shortcut',shortcut)]:
        measured=ref.metrics(source.Model().fit(data),data);result['controls'][name]=measured
        assert measured['calibration_chi2']<1.5 and measured['parameter_relative_error']<.03
        assert all((v<.04)==(name=='oracle') for v in measured['hidden'].values())
        if name=='oracle':assert max(measured['voltage_absolute_error'])<1e-4
    a,b=oracle.Model(),shortcut.Model();a.conductivity=b.conductivity=ref.TRUE_CONDUCTIVITY
    difference=max(float(np.max(abs(a.predict(*settings(data,i))[f]-b.predict(*settings(data,i))[f])))
                   for i in range(len(data['initial'])) for f in ['temperature','voltage'])
    assert difference==0
    mean=noiseless(data,ref);rng=np.random.default_rng(41401);fitted=[];chi=[]
    for i in range(args.noise_trials):
        noisy=dict(data)
        for field in ['temperature','voltage']:
            noisy[field]=mean[field]+rng.normal(size=mean[field].shape)*data['sigma_'+field]
        model=oracle.Model().fit(noisy);control=shortcut.Model().fit(noisy)
        assert abs(model.conductivity-control.conductivity)<1e-12
        fitted.append(model.conductivity);chi.append(chi2(model,noisy))
        if (i+1)%32==0:print('noise fits',i+1,flush=True)
    assert max(chi)<1.5 and max(abs(np.array(fitted)/ref.TRUE_CONDUCTIVITY-1))<.03
    separation={'oracle':[],'shortcut':[]};voltages={'oracle':[],'shortcut':[]}
    convergence=[];true_error=[];oracle_refinement=[];temperature_range=[]
    for args_in in ref.hidden_inputs():
        truth=ref.predict(*args_in,ref.TRUE_CONDUCTIVITY);fine=ref.predict(*args_in,ref.TRUE_CONDUCTIVITY,refinement=8)
        convergence.append(float(np.max(abs(truth['temperature']-fine['temperature']))))
        true_error.append(float(np.max(abs(a.predict(*args_in)['temperature']-truth['temperature']))))
        coarse=oracle.thermal_solution(*args_in,ref.TRUE_CONDUCTIVITY,refinement=2)
        refined=oracle.thermal_solution(*args_in,ref.TRUE_CONDUCTIVITY,refinement=4)
        oracle_refinement.append(float(np.max(abs(coarse-refined))))
        temperature_range.extend([float(truth['temperature'].min()),float(truth['temperature'].max())])
        scale=np.sqrt(np.mean((truth['temperature']-np.linspace(*args_in[4],len(args_in[1])))**2))
        for name,source in [('oracle',oracle),('shortcut',shortcut)]:
            for k in [min(fitted),max(fitted)]:
                m=source.Model();m.conductivity=k;output=m.predict(*args_in)
                error=float(np.sqrt(np.mean((output['temperature']-truth['temperature'])**2))/scale)
                verror=float(np.max(abs(output['voltage']-truth['voltage'])))
                separation[name].append(error);voltages[name].append(verror)
                assert (error<.04)==(name=='oracle')
                if name=='oracle':assert verror<1e-4
    assert max(convergence)<.002 and max(true_error)<.02 and max(oracle_refinement)<.01
    args_in=ref.hidden_inputs()[0];t,x,initial,current,boundary,span=args_in
    direct=a.predict(*args_in)
    reverse=a.predict(t,x,initial[::-1],-current,boundary[::-1],-span)
    symmetry=float(np.max(abs(direct['temperature']-reverse['temperature'][:,::-1])))
    vsymmetry=float(np.max(abs(direct['voltage']+reverse['voltage'])))
    assert symmetry<1e-6 and vsymmetry<1e-8
    # Exact local heat balance on an analytic profile, independent of spatial discretization.
    z=np.linspace(0,.01,1001);temperature=300+20*z/.01+5*np.sin(np.pi*z/.01)
    grad=20/.01+5*np.pi/.01*np.cos(np.pi*z/.01);curv=-5*(np.pi/.01)**2*np.sin(np.pi*z/.01)
    s=2e-4+4e-6*(temperature-300)+span*(z/.01-.5);ds=4e-6*grad+span/.01
    minus_qx=-current*(ds*temperature+s*grad)+1.25*curv
    work=current*(2e-5*current+s*grad)
    expanded=1.25*curv+2e-5*current**2-current*temperature*(4e-6*grad+span/.01)
    balance=float(np.max(abs(minus_qx+work-expanded)))
    missing=-current*temperature*span/.01
    baseline=1.25*curv+2e-5*current**2-current*temperature*4e-6*grad
    omitted=float(np.max(abs(expanded-baseline-missing)))
    assert balance<1e-7 and omitted<1e-7
    zero_args=list(args_in);zero_args[3]=0.
    zero_current=max(float(np.max(abs(a.predict(*zero_args)[f]-b.predict(*zero_args)[f]))) for f in ['temperature','voltage'])
    assert zero_current==0
    result['physical_checks']={'homogeneous_nonzero_current_calibration_equivalence':difference,
        'graded_zero_current_equivalence':zero_current,'reference4_vs8_max_K':max(convergence),
        'oracle_vs_energy_flux_max_K':max(true_error),'oracle_refinement2_vs4_max_K':max(oracle_refinement),
        'current_grading_spatial_reversal_max_K':symmetry,'voltage_reversal_error_V':vsymmetry,
        'analytic_local_energy_balance_W_per_m3':balance,'analytic_omitted_spatial_peltier_identity_W_per_m3':omitted,
        'hidden_temperature_range_K':[min(temperature_range),max(temperature_range)]}
    result['noise']={'calibration_chi2_max':max(chi),'calibration_pass_fraction':1.,'parameter_min':min(fitted),
        'parameter_max':max(fitted),'parameter_relative_error_max':float(max(abs(np.array(fitted)/ref.TRUE_CONDUCTIVITY-1)))}
    result['hidden_parameter_extrema_check']={k:{'min':min(v),'max':max(v),'voltage_max_V':max(voltages[k])} for k,v in separation.items()}
    result['seconds']=time.time()-start
    result['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [TASK/'instruction.md',TASK/'environment/README.md',TASK/'environment/model.py',TASK/'solution/model.py',TASK/'tests/reference.py',TASK/'environment/data/calibration.npz',TASK/'tests/test_hidden.py',ROOT/'scripts/thermoelectric_rod_baseline.py',Path(__file__)]}
    output=ROOT/'jobs/thermoelectric-rod-r2-validation';output.mkdir(parents=True,exist_ok=True)
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS',output/'summary.json',result['seconds'])


if __name__=='__main__':main()
