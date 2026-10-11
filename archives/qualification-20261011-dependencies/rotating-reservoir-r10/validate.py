from pathlib import Path
import argparse,hashlib,importlib.util,json,time
import numpy as np
S=Path('/Users/yp9846/Desktop/PhysLiteBench/staging/overnight-20261011');T=S/'tasks/rotating-reservoir';A=S/'predecessors/rotating-reservoir-r9'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(n,p):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def err(a,b):return float(np.linalg.norm(a-b)/np.linalg.norm(b))
def main():
    pa=argparse.ArgumentParser();pa.add_argument('--freeze',action='store_true');args=pa.parse_args();start=time.monotonic()
    good=load('g',T/'solution/model.py');bad=load('b',S/'scripts/rotating_reservoir_baseline.py');ref=load('r',T/'tests/reference.py');prior=json.loads((A/'validation.json').read_text())
    assert sha(T/'solution/model.py')==sha(A/'task/solution/model.py')==prior['source_sha256']['solution/model.py']
    old_reference=(A/'task/tests/reference.py').read_text();assert (T/'tests/reference.py').read_text().startswith(old_reference)
    for name in ['environment/README.md','environment/data/calibration.json','tests/data/calibration.json','tests/test_hidden.py','task.toml','instruction.md']:
        assert sha(T/name)==sha(A/'task'/name)==prior['source_sha256'][name]
    records=json.loads((T/'environment/data/calibration.json').read_text());inputs=[r['input'] for r in records];y=np.array([r['value'] for r in records]);sigma=np.array([r['sigma'] for r in records]);hidden=ref.hidden_inputs();truth={}
    for name,es in hidden.items():truth[name]=ref.predict(es);print('Reference',name,round(time.monotonic()-start,1),flush=True)
    path=T/'tests/data/hidden_truth.json'
    if args.freeze:path.write_text(json.dumps(dict(reference_sha256=sha(T/'tests/reference.py'),description='Independent marked-edge finite-volume calorimeter, 81/162 velocity grids and Richardson extrapolation.',true_parameter=.67,groups={n:dict(inputs=hidden[n],truth=v.tolist()) for n,v in truth.items()}),indent=2)+'\n')
    frozen=json.loads(path.read_text());assert frozen['reference_sha256']==sha(T/'tests/reference.py') and all(err(truth[n],np.array(frozen['groups'][n]['truth']))<1e-10 for n in hidden)
    sensitive=['isothermal_driven_flux_third','cold_rotor_driven_flux_third']
    def scores(m,g):return {n:err(m.predict_at(es,g),truth[n]) for n,es in hidden.items()}
    controls={}
    for name,m in [('oracle',good),('shortcut',bad)]:
        fit=m.Model().fit(records);res=(fit.predict(inputs)-y)/sigma;row=dict(parameter=fit.drag,calibration_chi2=float(res@res)/(len(records)-1),hidden=scores(m,fit.drag))
        assert abs(fit.drag/.67-1)<.03 and row['calibration_chi2']<1.5
        if name=='oracle':assert max(row['hidden'].values())<.04
        else:assert all(row['hidden'][n]>.04 for n in sensitive) and row['hidden']['constant_contact_energy_third_cumulant']<.04 and row['hidden']['stationary_mean_and_uniform_noise']<.04
        controls[name]=row
    print('Controls',json.dumps(controls),flush=True)
    new_checks=[]
    for name in sensitive:
        es=hidden[name];higher=[];better=[]
        for e in es:
            a=good.kinetic_statistics(e,.67,degree=42);higher.append(a['third_cumulant']);better.append(ref.predict([e],.67,points=101)[0]);assert a['first_law_residual']<1e-9 and a['noise']>0
        row=dict(group=name,oracle_refinement_error=err(good.predict_at(es,.67),np.array(higher)),independent_error=err(np.array(higher),truth[name]),finite_volume_refinement_error=err(np.array(better),truth[name]))
        assert max(row[k] for k in row if k.endswith('error'))<.003,row;new_checks.append(row)
    checks=[]
    for i,item in enumerate(prior['physics']['settings']):
        e=item['input'];gamma=item['drag'];rate=bad.averaged_clock(e,gamma)
        a=bad.hermite(e,gamma,degree=6,variance=max(e['temperature_a'],e['temperature_b']),total_degree=True,noise=True,collision_rate=rate)
        b=bad.hermite(e,gamma,degree=8,variance=max(e['temperature_a'],e['temperature_b']),total_degree=True,noise=True,collision_rate=rate)
        row=dict(input=e,drag=gamma,clock=rate,noise=a['noise'],first_law_residual=a['first_law_residual'],degree_error=max(abs(a[k]-b[k]) for k in ['mean','noise','third_cumulant']))
        assert rate>0 and a['noise']>0 and row['first_law_residual']<1e-9 and row['degree_error']<1e-8
        if i%8==0:
            ref.RATE=rate;ee=dict(e,flux_fraction=0.)
            coarse=ref.finite_volume(ee,gamma,points=61,noise=True);fine=ref.finite_volume(ee,gamma,points=122,noise=True)
            independent=(4*fine['third_cumulant']-coarse['third_cumulant'])/3;row['shortcut_independent_scaled_error']=abs(a['third_cumulant']-independent)/(1+abs(a['third_cumulant']));assert row['shortcut_independent_scaled_error']<.005
            ref.RATE=3.6
        checks.append(row)
    print('Stronger shortcut physics PASS',flush=True)
    clean=np.array([good.calibration_mean(e,.67) for e in inputs]);fits=[]
    for draw in np.random.default_rng(146824).normal(size=(256,len(records))):
        noisy=clean+sigma*draw;sample=[dict(r,value=float(v)) for r,v in zip(records,noisy)];g=good.Model().fit(sample);b=bad.Model().fit(sample);res=(g.predict(inputs)-noisy)/sigma;chi=float(res@res)/(len(records)-1)
        assert abs(g.drag/.67-1)<.03 and abs(g.drag-b.drag)<1e-12 and chi<1.5;fits.append(dict(parameter=g.drag,chi2=chi))
    params=np.array([r['parameter'] for r in fits]);sensitivity=[]
    for gamma in np.quantile(params,np.linspace(0,1,9)):
        a=scores(good,float(gamma));b=scores(bad,float(gamma));assert max(a.values())<.04 and min(b[n] for n in sensitive)>.04
        sensitivity.append(dict(parameter=float(gamma),oracle=a,shortcut=b));print('Noise envelope',len(sensitivity),'of9',flush=True)
    report=dict(revision=10,calibration_regenerated=False,controls=controls,physics=dict(inherited_validation='predecessors/rotating-reservoir-r9/validation.json',inherited_sha256=sha(A/'validation.json'),unchanged_oracle_sha256=sha(T/'solution/model.py'),reference_change='Only new HIDDEN entries appended; all kinetic and calorimeter equations unchanged.',new_hidden_checks=new_checks,shortcut_checks=checks),noise=dict(seed=146824,trials=256,method='256 actual calibration fits; full hidden predictions at nine fitted-parameter quantiles including extrema, not 256 full hidden predictions.',fit_min=float(params.min()),fit_max=float(params.max()),max_chi2=max(r['chi2'] for r in fits),hidden_sensitivity=sensitivity),seconds=time.monotonic()-start,source_sha256={str(p.relative_to(T)):sha(p) for p in T.rglob('*') if p.is_file() and not any(x in p.parts for x in ['__pycache__','.pytest_cache'])})
    (S/'results/rotating-reservoir-r10-validation.json').write_text(json.dumps(report,indent=2)+'\n');print('PASS',round(report['seconds'],1),flush=True)
if __name__=='__main__':main()
