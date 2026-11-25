"""Author-only science checks and local controls; no model-agent evaluations."""
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
from scipy.integrate import quad

STAGE=Path(__file__).resolve().parents[1]
TASK=STAGE/'tasks/collisionless-screening'


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def nrms(y,t):return float(np.linalg.norm(y-t)/np.linalg.norm(t))


def local_test(model):
    with tempfile.TemporaryDirectory(prefix='collisionless-control-') as d:
        app=Path(d);shutil.copytree(TASK/'environment',app,dirs_exist_ok=True)
        shutil.copy2(model,app/'model.py');start=time.monotonic()
        run=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',str(app/'test_public.py'),str(TASK/'tests/test_hidden.py')],cwd=app,env={**os.environ,'PYTHONPATH':str(app),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'},capture_output=True,text=True)
        return {'returncode':run.returncode,'elapsed_seconds':time.monotonic()-start,'stdout':run.stdout,'stderr':run.stderr}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--generate',action='store_true');args=parser.parse_args();start=time.monotonic()
    oracle=load(TASK/'solution/model.py','oracle');base=load(STAGE/'scripts/collisionless_screening_baseline.py','baseline');ref=load(TASK/'tests/reference.py','reference')
    inputs=ref.calibration_inputs();sigma=.0012;clean=ref.predict(inputs,ref.TRUE_PARAMETER)
    if args.generate:
        rng=np.random.default_rng(219031)
        rows=[{'input':e,'value':float(y),'sigma':sigma} for e,y in zip(inputs,clean+rng.normal(size=len(clean))*sigma)]
        text=json.dumps(rows,indent=2)+'\n'
        for f in [TASK/'environment/data/calibration.json',TASK/'tests/data/calibration.json']:f.write_text(text)
    rows=json.loads((TASK/'environment/data/calibration.json').read_text())
    assert (TASK/'environment/data/calibration.json').read_bytes()==(TASK/'tests/data/calibration.json').read_bytes()
    om=oracle.Model().fit(rows);bm=base.Model().fit(rows)
    exact=oracle.predict_at(inputs,ref.TRUE_PARAMETER);alternative=base.predict_at(inputs,ref.TRUE_PARAMETER)
    difference=float(np.max(abs(exact-alternative)));bias=float(np.max(abs(exact-clean)/sigma))
    assert difference<1e-13 and bias<.002
    groups=ref.hidden_inputs();targets={};hidden={};refinement=[]
    for name,es in groups.items():
        truth=ref.predict(es,ref.TRUE_PARAMETER);targets[name]=truth
        refined=ref.predict(es,ref.TRUE_PARAMETER,.5)
        change=float(np.max(abs(refined-truth)));refinement.append(change)
        eo=nrms(om.predict(es),truth);eb=nrms(bm.predict(es),truth)
        hidden[name]={'truth':truth.tolist(),'minimum_signal':float(truth.min()),'oracle_nrms':eo,'shortcut_nrms':eb,'reference_refinement':change}
        assert truth.min()>.05 and eo<.04 and change<2e-6
        assert eb<.04 if name=='nonresonant_anchors' else eb>.04
    rng=np.random.default_rng(219037);fits=[];chis=[];worst=0.;best=np.inf;fit_difference=0.
    for _ in range(256):
        y=clean+rng.normal(size=len(clean))*sigma
        records=[{'input':e,'value':float(v),'sigma':sigma} for e,v in zip(inputs,y)]
        a=oracle.Model().fit(records);b=base.Model().fit(records);fits.append(a.density)
        fit_difference=max(fit_difference,abs(a.density-b.density))
        for m in [a,b]:
            chi=float(np.sum(((m.predict(inputs)-y)/sigma)**2)/(len(y)-1));chis.append(chi)
            assert chi<1.5 and abs(m.density/ref.TRUE_PARAMETER-1)<.03
        for name,es in groups.items():
            eo=nrms(a.predict(es),targets[name]);eb=nrms(b.predict(es),targets[name]);worst=max(worst,eo)
            assert eo<.04
            if name=='nonresonant_anchors':assert eb<.04
            else:best=min(best,eb);assert eb>.04
    domain=[]
    controls=list(itertools.product([.6,1.4],[1.5,2.2],[.9,1.2],[0.,.2,.425,.65,1.5,2.5]))
    rng=np.random.default_rng(219041)
    for _ in range(24):
        density=float(rng.uniform(.6,1.4));k=float(rng.uniform(1.5,2.2));w=float(rng.uniform(.9,1.2))
        z=float(rng.uniform(.2,.65) if rng.random()<.6 else rng.uniform(1.5,2.5))
        controls.append((density,k,w,z))
    for density,k,w,z in controls:
        e=ref.experiment(k,w,z);expected=oracle.predict_at([e],density)[0]
        calculated=ref.response(e,density);fine=ref.response(e,density,.5)
        error=abs(calculated.real-expected);change=abs(calculated-fine)
        assert error<3e-6 and change<4e-6 and expected>0
        h=base.velocity_response(z);denom=1-density*h/(k*w)**2
        assert denom>.57
        reverse=ref.response(ref.experiment(k,w,-z),density) if z else calculated
        assert abs(reverse-calculated.conjugate())<1e-9
        if z and z<1:assert calculated.imag<0
        domain.append({'controls':[density,k,w,z],'oracle':expected,'reference_real_error':error,'reference_complex_refinement':change,'source_denominator':float(denom)})
    # Direct velocity quadrature independently checks the known characteristic and PV source.
    characteristic_error=0.;pv_error=0.
    for s in np.linspace(0,35,36):
        v=quad(lambda u:15/16*(1-u*u)**2*np.cos(s*u),-1,1,epsabs=1e-12)[0]
        characteristic_error=max(characteristic_error,abs(v-ref.characteristic(s)))
    for z in [-2.5,-1.5,-.65,-.2,0.,.2,.425,.65,1.5,2.5]:
        derivative=lambda u:15/4*(u**3-u)
        v=quad(derivative,-1,1,weight='cauchy',wvar=z,epsabs=1e-12)[0] if abs(z)<1 else quad(lambda u:derivative(u)/(u-z),-1,1,epsabs=1e-12)[0]
        pv_error=max(pv_error,abs(v-base.velocity_response(z)))
    assert characteristic_error<1e-12 and pv_error<1e-12
    memory_quadrature_refinement=max(abs(ref.memory(z,rate,16)-ref.memory(z,rate,24)) for z in [0.,.2,.65,1.5,2.5] for rate in [.012,.003,.0015])
    assert memory_quadrature_refinement<1e-10
    zero_density=max(abs(ref.response(ref.experiment(1.5,.9,z),0)-1) for z in [0,.3,.65,1.5,2.5])
    static_error=max(abs(oracle.predict_at([ref.experiment(1.7,1.1,0)],density)[0]-1/(1+5*density/(1.7*1.1)**2)) for density in [.6,1.4])
    assert zero_density<1e-12 and static_error<1e-14
    recovery=[]
    for density in np.linspace(.6,1.4,33):
        y=oracle.predict_at(inputs,float(density));records=[{'input':e,'value':float(v),'sigma':sigma} for e,v in zip(inputs,y)]
        fitted=base.Model().fit(records).density
        assert abs(fitted-density)<2e-7
        gap={name:nrms(base.predict_at(es,density),oracle.predict_at(es,density)) for name,es in groups.items()}
        assert all(gap[name]>.04 for name in groups if name!='nonresonant_anchors')
        recovery.append({'true_density':float(density),'fitted_density':float(fitted),'hidden_gaps':gap})
    # Exact derivative of every calibration response, including nonzero drives.
    kw=np.asarray([e['wavenumber']*e['velocity_width'] for e in inputs]);z=np.asarray([e['frequency'] for e in inputs])/kw
    h=base.velocity_response(z);derivative=(h/kw**2)/(1-ref.TRUE_PARAMETER*h/kw**2)**2
    residual=(om.predict(inputs)-np.asarray([r['value'] for r in rows]))/sigma
    report={'status':'scientific_checks_passed','model_evaluations':0,'calibration':{'records':len(rows),'unique_inputs':len({json.dumps(e,sort_keys=True) for e in inputs}),'oracle_density':om.density,'shortcut_density':bm.density,'chi2':float(residual@residual)/(len(rows)-1),'maximum_control_difference':difference,'maximum_reference_bias_in_sigma':bias,'fisher_information_at_truth':float(np.sum((derivative/sigma)**2)),'sigma':sigma,'identifiability':'Static transfer 1/(1+5*density/(k*width)^2) is strictly decreasing over the entire interval.'},'hidden':hidden,'noise256':{'all_calibration_and_parameter_pass':True,'all_oracle_pass':True,'all_shortcut_diagnostic_fail':True,'maximum_chi2':max(chis),'fitted_density_range':[min(fits),max(fits)],'maximum_control_fit_difference':fit_difference,'maximum_oracle_hidden_error':worst,'minimum_shortcut_diagnostic_error':best},'domain_checks':domain,'parameter_recovery':recovery,'limits':{'characteristic_quadrature_error':characteristic_error,'pv_quadrature_error':pv_error,'memory_quadrature_refinement':float(memory_quadrature_refinement),'zero_density_error':zero_density,'static_error':static_error},'domain_guards':{'minimum_source_denominator_bound':float(1-1.4*base.velocity_response(1.5)/(1.5*.9)**2),'minimum_resonant_source_denominator_bound':float(1-1.4*base.velocity_response(.65)/(1.5*.9)**2),'explanation':'H outside support is integral F/(u-z)^2, positive decreasing. H increases throughout [.2,.65]. In that interval the physical dielectric also has nonzero positive imaginary part.'},'elapsed_science_seconds':time.monotonic()-start}
    (STAGE/'results/collisionless-screening-r1-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    local={'oracle':local_test(TASK/'solution/model.py'),'shortcut':local_test(STAGE/'scripts/collisionless_screening_baseline.py')}
    (STAGE/'results/collisionless-screening-r1-local-controls.json').write_text(json.dumps(local,indent=2)+'\n')
    assert local['oracle']['returncode']==0
    assert local['shortcut']['returncode']==1 and '3 failed, 4 passed' in local['shortcut']['stdout']
    print(json.dumps({'calibration':report['calibration'],'noise256':report['noise256'],'domain_max_error':max(r['reference_real_error'] for r in domain),'local':{k:{q:v[q] for q in ['returncode','elapsed_seconds']} for k,v in local.items()}},indent=2))

if __name__=='__main__':main()
