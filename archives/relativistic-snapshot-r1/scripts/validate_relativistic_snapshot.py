"""Scientific controls only; no model-agent evaluation."""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import numpy as np
from scipy.special import kve

STAGE=Path(__file__).resolve().parents[1]
TASK=STAGE/'tasks/relativistic-snapshot'


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def tensor_energy(m,T,b,a):
    rest=m*kve(1,m/T)/kve(2,m/T)+3*T
    return ((1-a*b)*rest+b*(b-a)*T)/np.sqrt((1-a*a)*(1-b*b))


def local_test(model):
    with tempfile.TemporaryDirectory(prefix='relativistic-snapshot-control-') as d:
        app=Path(d);shutil.copytree(TASK/'environment',app,dirs_exist_ok=True)
        shutil.copy2(model,app/'model.py');start=time.monotonic()
        run=subprocess.run([sys.executable,'-m','pytest','-q',str(app/'test_public.py'),str(TASK/'tests/test_hidden.py')],cwd=app,env={**os.environ,'PYTHONPATH':str(app),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'},capture_output=True,text=True)
        return {'returncode':run.returncode,'elapsed_seconds':time.monotonic()-start,'stdout':run.stdout,'stderr':run.stderr}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--generate',action='store_true');args=parser.parse_args()
    start=time.monotonic()
    oracle=load(TASK/'solution/model.py','oracle');base=load(STAGE/'scripts/relativistic_snapshot_baseline.py','baseline');ref=load(TASK/'tests/reference.py','reference')
    inputs=ref.calibration_inputs();clean=ref.predict(inputs,ref.TRUE_PARAMETER)
    if args.generate:
        rng=np.random.default_rng(161051)
        rows=[{'input':e,'value':float(y),'sigma':ref.SIGMA} for e,y in zip(inputs,clean+rng.normal(0,ref.SIGMA,len(inputs)))]
        text=json.dumps(rows,indent=2)+'\n'
        for p in [TASK/'environment/data/calibration.json',TASK/'tests/data/calibration.json']:p.write_text(text)
    rows=json.loads((TASK/'environment/data/calibration.json').read_text())
    assert (TASK/'environment/data/calibration.json').read_bytes()==(TASK/'tests/data/calibration.json').read_bytes()
    a=oracle.Model().fit(rows);b=base.Model().fit(rows)
    ocal=oracle.predict_at(inputs,ref.TRUE_PARAMETER);bcal=base.predict_at(inputs,ref.TRUE_PARAMETER)
    assert np.max(abs(ocal-bcal))<1e-12
    groups=ref.hidden_inputs();hidden={};target={}
    for name,es in groups.items():
        y=ref.predict(es,ref.TRUE_PARAMETER);target[name]=y
        eo=np.linalg.norm(a.predict(es)-y)/np.linalg.norm(y);eb=np.linalg.norm(b.predict(es)-y)/np.linalg.norm(y)
        hidden[name]={'truth':y.tolist(),'oracle_nrms':float(eo),'shortcut_nrms':float(eb),'minimum_signal':float(y.min())}
        assert y.min()>.8 and eo<.04
        if name!='matched_anchors':assert eb>.04
    rng=np.random.default_rng(161053);fits=[];chis=[];maxfitdiff=0.;worst_o=0.;best_b=1e10
    for _ in range(256):
        values=clean+rng.normal(0,ref.SIGMA,len(clean))
        records=[{'input':e,'value':float(y),'sigma':ref.SIGMA} for e,y in zip(inputs,values)]
        om=oracle.Model().fit(records);bm=base.Model().fit(records)
        fits.append(om.mass);maxfitdiff=max(maxfitdiff,abs(om.mass-bm.mass))
        for model in [om,bm]:
            chi=float(np.sum(((model.predict(inputs)-values)/ref.SIGMA)**2)/(len(values)-1));chis.append(chi)
            assert abs(model.mass/ref.TRUE_PARAMETER-1)<.03 and chi<1.5
        for name,es in groups.items():
            y=target[name];eo=float(np.linalg.norm(om.predict(es)-y)/np.linalg.norm(y));eb=float(np.linalg.norm(bm.predict(es)-y)/np.linalg.norm(y))
            worst_o=max(worst_o,eo);assert eo<.04
            if name!='matched_anchors':best_b=min(best_b,eb);assert eb>.04
    # Independent quadrature and tensor/current checks throughout the allowed domain.
    corners=[]
    for m in [.8,1.07,1.2]:
        for T in [.2,1.]:
            for beta in [-.9,0.,.9]:
                for analysis in [-.9,0.,.9]:
                    o=oracle.mean_energy(m,T,beta,analysis);r=ref.energy(m,T,beta,analysis)
                    f=ref.energy(m,T,beta,analysis,288,192);t=tensor_energy(m,T,beta,analysis)
                    reflected=oracle.mean_energy(m,T,-beta,-analysis)
                    corners.append({'mass':m,'temperature':T,'gas_speed':beta,'analysis_speed':analysis,'value':o,'oracle_reference_relative_error':abs(o-r)/r,'reference_refinement_relative_change':abs(r-f)/f,'tensor_relative_error':abs(o-t)/t,'reflection_absolute_error':abs(o-reflected)})
                    assert max(abs(o-r)/r,abs(r-f)/f,abs(o-t)/t)<1e-9
                    assert abs(o-reflected)<1e-12 and o>=m
    # Check both exact calibration families and unique mass recovery, including endpoints.
    mass_range=[];derivatives=[]
    for m in np.linspace(.8,1.2,41):
        truth=ref.predict(inputs,float(m))
        noiseless=[{'input':e,'value':float(y),'sigma':ref.SIGMA} for e,y in zip(inputs,truth)]
        fitted=base.Model().fit(noiseless).mass
        equivalent=np.max(abs(oracle.predict_at(inputs,float(m))-base.predict_at(inputs,float(m))))
        derivative=(oracle.predict_at(inputs,float(m+1e-5))-oracle.predict_at(inputs,float(m-1e-5)))/(2e-5)
        derivatives.extend(derivative.tolist())
        gaps={}
        for name,es in groups.items():
            y=ref.predict(es,float(m));z=base.predict_at(es,float(m))
            gaps[name]=float(np.linalg.norm(z-y)/np.linalg.norm(y))
            if name!='matched_anchors':assert gaps[name]>.04
        mass_range.append({'mass':float(m),'recovered_mass':fitted,'absolute_error':abs(fitted-m),'calibration_equivalence':float(equivalent),'hidden_gaps':gaps})
        assert abs(fitted-m)<1e-6 and equivalent<1e-11 and derivative.min()>0
    # Correct rest gas equipartition and four-current normalization from a third radial integral.
    from scipy.integrate import quad
    thermodynamics=[]
    for m,T in [(.8,.2),(1.07,.55),(1.2,1.)]:
        density=lambda p:p*p*np.exp(-(np.sqrt(m*m+p*p)-m)/T)
        norm=quad(density,0,np.inf,epsabs=1e-12)[0]
        rest_energy=quad(lambda p:density(p)*np.sqrt(m*m+p*p),0,np.inf,epsabs=1e-12)[0]/norm
        pressure=quad(lambda p:density(p)*p*p/(3*np.sqrt(m*m+p*p)),0,np.inf,epsabs=1e-12)[0]/norm
        thermodynamics.append({'mass':m,'temperature':T,'pressure_per_particle':pressure,'equipartition_error':abs(pressure-T),'energy_error':abs(rest_energy-tensor_energy(m,T,0,0))})
        assert abs(pressure-T)<1e-10 and abs(rest_energy-tensor_energy(m,T,0,0))<1e-10
    # Nonrelativistic and ultrarelativistic asymptotic checks, outside scoring domain.
    limits=[]
    for T in [.002,.001,.0005]:
        e=tensor_energy(1.,T,0,0)
        limits.append({'limit':'nonrelativistic','temperature':T,'thermal_energy_ratio':(e-1)/(1.5*T)})
    for m in [.04,.02,.01]:
        e=tensor_energy(m,1.,0,0)
        limits.append({'limit':'ultrarelativistic','mass':m,'energy_ratio':e/3})
    assert abs(limits[2]['thermal_energy_ratio']-1)<.001 and abs(limits[-1]['energy_ratio']-1)<2e-5
    original=load(STAGE/'prototype/check.py','prototype')
    worldline=original.worldline_check(1.07,.55,.85,-.2)
    assert abs(worldline['sample_mean']-worldline['tensor_prediction'])<4*worldline['standard_error']
    residual=(a.predict(inputs)-np.array([r['value'] for r in rows]))/ref.SIGMA
    report={'status':'scientific_checks_passed','scope':'No model-agent evaluation.','calibration':{'records':len(rows),'unique_inputs':len({json.dumps(e,sort_keys=True) for e in inputs}),'oracle_mass':a.mass,'shortcut_mass':b.mass,'chi2':float(residual@residual)/(len(rows)-1),'maximum_equivalence_error':float(np.max(abs(ocal-bcal))),'maximum_numerical_bias_in_sigma':float(np.max(abs(ocal-clean))/ref.SIGMA),'minimum_sampled_derivative':min(derivatives),'fisher_information_at_truth':float(np.sum(((oracle.predict_at(inputs,ref.TRUE_PARAMETER+1e-5)-oracle.predict_at(inputs,ref.TRUE_PARAMETER-1e-5))/(2e-5*ref.SIGMA))**2)),'uncertainty':'Fixed sigma=.004, independent of mass and clean response.','identifiability':'Strictly positive de_rest/dm, proven in AUTHOR; positive known calibration boost factors.'},'hidden':hidden,'noise256':{'all_calibration_and_parameter_pass':True,'all_oracle_pass':True,'all_shortcut_diagnostic_fail':True,'maximum_chi2':max(chis),'mass_range':[min(fits),max(fits)],'maximum_control_fit_difference':maxfitdiff,'maximum_oracle_hidden_error':worst_o,'minimum_shortcut_diagnostic_error':best_b},'domain_corners':corners,'mass_range_checks':mass_range,'rest_equipartition':thermodynamics,'limits':limits,'worldline_reference':worldline,'elapsed_science_seconds':time.monotonic()-start}
    (STAGE/'results/relativistic-snapshot-r1-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    controls={'oracle':local_test(TASK/'solution/model.py'),'shortcut':local_test(STAGE/'scripts/relativistic_snapshot_baseline.py')}
    (STAGE/'results/relativistic-snapshot-r1-local-controls.json').write_text(json.dumps(controls,indent=2)+'\n')
    assert controls['oracle']['returncode']==0
    assert controls['shortcut']['returncode']==1 and '3 failed, 4 passed' in controls['shortcut']['stdout']
    print(json.dumps({'calibration':report['calibration'],'noise256':report['noise256'],'local_controls':{k:{s:v[s] for s in ['returncode','elapsed_seconds']} for k,v in controls.items()}},indent=2))

if __name__=='__main__':main()
