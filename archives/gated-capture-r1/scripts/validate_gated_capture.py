"""Author-only scientific validation and local controls; no agent model runs."""
from pathlib import Path
import argparse
import importlib.util
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import numpy as np

STAGE=Path(__file__).resolve().parents[1]
TASK=STAGE/'tasks/gated-capture'
KEYS=('radius','reactivity0','reactivity1','rate01','rate10')


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def local_test(model):
    with tempfile.TemporaryDirectory(prefix='gated-capture-control-') as d:
        app=Path(d);shutil.copytree(TASK/'environment',app,dirs_exist_ok=True)
        shutil.copy2(model,app/'model.py');start=time.monotonic()
        run=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',str(app/'test_public.py'),str(TASK/'tests/test_hidden.py')],cwd=app,env={**os.environ,'PYTHONPATH':str(app),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'},capture_output=True,text=True)
        return {'returncode':run.returncode,'elapsed_seconds':time.monotonic()-start,'stdout':run.stdout,'stderr':run.stderr}


def exact_profile(D,a,k0,k1,u,v):
    pi=np.array([v,u])/(u+v);mode=np.array([1.,-1.]);kap=np.array([k0,k1]);q=np.sqrt((u+v)/D)
    matrix=np.column_stack([pi*(D/a+kap),mode*(D*(1/a+q)+kap)])
    A,B=np.linalg.solve(matrix,-kap*pi)
    r=np.geomspace(a,a+40/q,320)
    c=pi[:,None]*(1+A*a/r)+mode[:,None]*B*a/r*np.exp(-q*(r-a))
    surface=pi*(1+A)+mode*B
    return {'minimum_density':float(c.min()),'maximum_excess_over_reservoir':float((c-pi[:,None]).max()),
            'direct_reaction':float(4*np.pi*a*a*np.dot(kap,surface)),
            'far_flux':float(-4*np.pi*a*D*A)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--generate',action='store_true');args=p.parse_args()
    start=time.monotonic()
    oracle=load(TASK/'solution/model.py','oracle');base=load(STAGE/'scripts/gated_capture_baseline.py','baseline');ref=load(TASK/'tests/reference.py','reference')
    inputs=ref.calibration_inputs();clean=ref.predict(inputs,ref.TRUE_PARAMETER)
    if args.generate:
        rng=np.random.default_rng(191069)
        rows=[{'input':e,'value':float(y),'sigma':ref.SIGMA} for e,y in zip(inputs,clean+rng.normal(0,ref.SIGMA,len(clean)))]
        text=json.dumps(rows,indent=2)+'\n'
        for f in [TASK/'environment/data/calibration.json',TASK/'tests/data/calibration.json']:f.write_text(text)
    rows=json.loads((TASK/'environment/data/calibration.json').read_text())
    assert (TASK/'environment/data/calibration.json').read_bytes()==(TASK/'tests/data/calibration.json').read_bytes()
    om=oracle.Model().fit(rows);bm=base.Model().fit(rows)
    oc=oracle.predict_at(inputs,ref.TRUE_PARAMETER);bc=base.predict_at(inputs,ref.TRUE_PARAMETER)
    assert np.max(abs(oc-bc))<1e-12
    groups=ref.hidden_inputs();hidden={};targets={}
    for name,es in groups.items():
        truth=ref.predict(es,ref.TRUE_PARAMETER);targets[name]=truth
        eo=float(np.linalg.norm(om.predict(es)-truth)/np.linalg.norm(truth));eb=float(np.linalg.norm(bm.predict(es)-truth)/np.linalg.norm(truth))
        hidden[name]={'truth':truth.tolist(),'minimum_signal':float(truth.min()),'oracle_nrms':eo,'shortcut_nrms':eb}
        assert truth.min()>.1 and eo<.04
        if name!='equal_reactivity_anchors':assert eb>.04
    rng=np.random.default_rng(191071);fits=[];chis=[];worst=0.;best=1e10;fitdifference=0.
    for _ in range(256):
        y=clean+rng.normal(0,ref.SIGMA,len(clean))
        records=[{'input':e,'value':float(value),'sigma':ref.SIGMA} for e,value in zip(inputs,y)]
        a=oracle.Model().fit(records);b=base.Model().fit(records)
        fits.append(a.diffusivity);fitdifference=max(fitdifference,abs(a.diffusivity-b.diffusivity))
        for model in [a,b]:
            chi=float(np.sum(((model.predict(inputs)-y)/ref.SIGMA)**2)/(len(y)-1));chis.append(chi)
            assert chi<1.5 and abs(model.diffusivity/ref.TRUE_PARAMETER-1)<.03
        for name,es in groups.items():
            truth=targets[name];eo=float(np.linalg.norm(a.predict(es)-truth)/np.linalg.norm(truth));eb=float(np.linalg.norm(b.predict(es)-truth)/np.linalg.norm(truth))
            worst=max(worst,eo);assert eo<.04
            if name!='equal_reactivity_anchors':best=min(best,eb);assert eb>.04
    domain=[]
    # Supported corners, including no reaction, unequal finite reaction and equal states.
    for D,a,k0,k1,u,v in itertools.product([.7,1.3],[.6,1.8],[0.,.5,15.],[0.,.5,15.],[.03,1.5],[.03,1.5]):
        controls=(D,a,k0,k1,u,v)
        exact=oracle.capture_rate(*controls);independent=ref.capture_rate(*controls);scale=max(abs(exact),1.)
        profile=exact_profile(*controls)
        swap=oracle.capture_rate(D,a,k1,k0,v,u)
        rate_scaled=oracle.capture_rate(2*D,a,2*k0,2*k1,2*u,2*v)
        detail={'controls':controls,'rate':exact,'reference_absolute_error':abs(exact-independent),'scaled_reference_error':abs(exact-independent)/scale,'profile':profile,'label_swap_error':abs(exact-swap),'time_scaling_error':abs(rate_scaled-2*exact)}
        domain.append(detail)
        assert abs(exact-independent)/scale<2e-6 and exact>=-1e-12
        assert profile['minimum_density']>-1e-12 and profile['maximum_excess_over_reservoir']<1e-12
        assert abs(profile['direct_reaction']-profile['far_flux'])<1e-11
        assert abs(exact-swap)<1e-11 and abs(rate_scaled-2*exact)<1e-11
        # The completed mean-Robin model has its own exact positive conservative solution.
        pi=base.gate_weights(u,v);kap=pi@np.array([k0,k1]);surface=D/(D+a*kap)
        source=base.capture_rate(*controls);source_far=4*np.pi*a*D*(1-surface)
        assert 0<=surface<=1 and abs(source-source_far)<1e-11
    refinements=[]
    for e in [groups['slow_switching'][0],groups['biased_gate'][0],groups['mixed_geometry'][-1]]:
        c=(ref.TRUE_PARAMETER,*[e[k] for k in KEYS]);r1=ref.capture_rate(*c);r2=ref.capture_rate(*c,refinement=2)
        refinements.append({'input':e,'relative_change':abs(r2-r1)/r2});assert abs(r2-r1)/r2<2e-6
    # Direct incoming and reaction flux, independently from the numerical radial BVP.
    balances=[]
    for es in groups.values():
        for e in es:
            c=(ref.TRUE_PARAMETER,*[e[k] for k in KEYS]);a=e['radius'];q=np.sqrt((e['rate01']+e['rate10'])/ref.TRUE_PARAMETER)
            flux,data=ref.finite_shell(*c,max(80*a,a+18/q))
            balances.append({'input':e,'rate':flux,**data});assert data['minimum_density']>=-1e-11 and data['balance_error']<1e-7
    sweep=[];derivative=[]
    for D in np.linspace(.7,1.3,31):
        truth=oracle.predict_at(inputs,float(D))
        equivalent=base.predict_at(inputs,float(D));assert np.max(abs(truth-equivalent))<1e-12
        records=[{'input':e,'value':float(y),'sigma':ref.SIGMA} for e,y in zip(inputs,truth)]
        recovered=base.Model().fit(records).diffusivity
        gaps={name:float(np.linalg.norm(base.predict_at(es,D)-oracle.predict_at(es,D))/np.linalg.norm(oracle.predict_at(es,D))) for name,es in groups.items()}
        sweep.append({'true_diffusivity':float(D),'fit':recovered,'absolute_error':abs(recovered-D),'hidden_gaps':gaps})
        assert abs(recovered-D)<1e-6
        assert all(gaps[n]>.04 for n in groups if n!='equal_reactivity_anchors')
        for e in inputs:
            a=e['radius'];k=e['reactivity0'];derivative.append(4*np.pi*a*(a*k)**2/(D+a*k)**2)
    assert min(derivative)>.2
    # Finite reactivities give regular fast- and slow-switching limits.
    D,a,k0,k1,u,v=1.,1.2,.1,7.,.1,.3;pi=base.gate_weights(u,v)
    slow=pi[0]*base.capture_rate(D,a,k0,k0,u,v)+pi[1]*base.capture_rate(D,a,k1,k1,u,v)
    fast=base.capture_rate(D,a,k0,k1,u,v)
    limits=[{'rate_multiplier':f,'oracle':oracle.capture_rate(D,a,k0,k1,f*u,f*v),'slow_limit':slow,'fast_limit':fast} for f in [1e-12,1e-8,1.,1e8,1e12]]
    assert abs(limits[0]['oracle']/slow-1)<2e-6 and abs(limits[-1]['oracle']/fast-1)<2e-5
    residual=(om.predict(inputs)-np.asarray([r['value'] for r in rows]))/ref.SIGMA
    sensitivity=(oracle.predict_at(inputs,ref.TRUE_PARAMETER+1e-5)-oracle.predict_at(inputs,ref.TRUE_PARAMETER-1e-5))/(2e-5)
    report={'status':'scientific_checks_passed','scope':'No model-agent evaluations.','calibration':{'records':len(rows),'unique_inputs':len({json.dumps(e,sort_keys=True) for e in inputs}),'oracle_diffusivity':om.diffusivity,'shortcut_diffusivity':bm.diffusivity,'chi2':float(residual@residual)/(len(rows)-1),'maximum_control_equivalence':float(np.max(abs(oc-bc))),'maximum_numerical_bias_in_sigma':float(np.max(abs(oc-clean))/ref.SIGMA),'minimum_sampled_derivative':min(derivative),'fisher_information_at_truth':float(np.sum((sensitivity/ref.SIGMA)**2)),'uncertainty':'Fixed instrument sigma=.01, independent of diffusivity and clean response.'},'hidden':hidden,'noise256':{'all_calibration_and_parameter_pass':True,'all_oracle_pass':True,'all_shortcut_diagnostic_fail':True,'maximum_chi2':max(chis),'diffusivity_range':[min(fits),max(fits)],'maximum_control_fit_difference':fitdifference,'maximum_oracle_hidden_error':worst,'minimum_shortcut_diagnostic_error':best},'domain_corners':domain,'reference_refinements':refinements,'independent_particle_balances':balances,'parameter_domain_sweep':sweep,'switching_limits':limits,'elapsed_science_seconds':time.monotonic()-start}
    (STAGE/'results/gated-capture-r1-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    controls={'oracle':local_test(TASK/'solution/model.py'),'shortcut':local_test(STAGE/'scripts/gated_capture_baseline.py')}
    (STAGE/'results/gated-capture-r1-local-controls.json').write_text(json.dumps(controls,indent=2)+'\n')
    assert controls['oracle']['returncode']==0
    assert controls['shortcut']['returncode']==1 and '3 failed, 4 passed' in controls['shortcut']['stdout']
    print(json.dumps({'calibration':report['calibration'],'noise256':report['noise256'],'local_controls':{k:{q:v[q] for q in ['returncode','elapsed_seconds']} for k,v in controls.items()}},indent=2))

if __name__=='__main__':main()
