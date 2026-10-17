"""Validate third heat cumulants against independent marked-edge kinetics."""
from pathlib import Path
import argparse,hashlib,importlib.util,itertools,json,platform,time
import numpy as np
import scipy

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'tasks/rotating-reservoir'
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def error(a,b):return float(np.linalg.norm(a-b)/np.linalg.norm(b))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args()
    start=time.monotonic()
    good=load('oracle',TASK/'solution/model.py');bad=load('shortcut',ROOT/'scripts/rotating_reservoir_baseline.py');ref=load('ref',TASK/'tests/reference.py')
    records=json.loads((TASK/'environment/data/calibration.json').read_text())
    assert records==json.loads((TASK/'tests/data/calibration.json').read_text())
    public=(TASK/'environment/model.py').read_text()
    assert public.replace("        raise NotImplementedError('Infer the common drag from the calibration records.')\n",'        self.drag=fit_drag(records)\n        return self\n')==(ROOT/'scripts/rotating_reservoir_baseline.py').read_text()
    metadata=json.loads((TASK/'tests/metadata.json').read_text());assert metadata['revision']==9
    inputs=[r['input'] for r in records];observed=np.array([r['value'] for r in records]);sigma=np.array([r['sigma'] for r in records]);hidden=ref.hidden_inputs()
    truth={n:ref.predict(es) for n,es in hidden.items()}
    frozen_path=TASK/'tests/data/hidden_truth.json'
    if args.freeze:
        frozen=dict(description='Independent marked-edge finite-volume reference, two-grid Richardson extrapolation. Regenerate deliberately after reference changes.',reference_sha256=sha(TASK/'tests/reference.py'),true_parameter=.67,environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,machine=platform.machine()),groups={n:dict(inputs=hidden[n],truth=values.tolist()) for n,values in truth.items()})
        frozen_path.write_text(json.dumps(frozen,indent=2)+'\n')
    frozen=json.loads(frozen_path.read_text());assert frozen['reference_sha256']==sha(TASK/'tests/reference.py')
    assert all(error(np.array(frozen['groups'][n]['truth']),truth[n])<1e-10 for n in hidden)
    print('Frozen reference verified',flush=True)
    def scores(m,gamma):return {n:error(m.predict_at(es,gamma),truth[n]) for n,es in hidden.items()}
    sensitive=[n for n in hidden if 'third_cumulant' in n]
    controls={}
    for name,m in [('oracle',good),('shortcut',bad)]:
        fit=m.Model().fit(records);residual=(fit.predict(inputs)-observed)/sigma
        row=dict(parameter=fit.drag,parameter_relative_error=abs(fit.drag/.67-1),calibration_chi2=float(residual@residual)/(len(records)-1),hidden=scores(m,fit.drag))
        assert row['parameter_relative_error']<.03 and row['calibration_chi2']<1.5
        assert max(row['hidden'].values())<.04 if name=='oracle' else all(row['hidden'][n]>.04 for n in sensitive)
        assert row['hidden']['stationary_mean_and_uniform_noise']<.04
        controls[name]=row
    print('Controls',json.dumps(controls),flush=True)
    rng=np.random.default_rng(202610119);checks=[]
    settings=[]
    for i in range(32):
        settings.append((dict(temperature_a=float(rng.uniform(.4,2)),temperature_b=float(rng.uniform(.4,1.4)),relaxation_rate=float(rng.uniform(.2,3)),angular_speed=float(rng.uniform(-.65,.65)),flip_rate=float(rng.uniform(.8,2)),flux_fraction=float(rng.uniform(0,1))),float(rng.uniform(.4,1.1))))
    for ta,tb,lam,gamma in itertools.product([.4,2.],[.4,1.4],[.2,3.],[.4,1.1]):
        settings.append((dict(temperature_a=ta,temperature_b=tb,relaxation_rate=lam,angular_speed=.65,flip_rate=.8,flux_fraction=1.),gamma))
    for i,(e,gamma) in enumerate(settings):
        a=good.kinetic_statistics(e,gamma);b=good.kinetic_statistics(e,gamma,degree=42)
        c=ref.finite_volume(e,gamma,points=61,noise=True,details=True);d=ref.finite_volume(e,gamma,points=122,noise=True,details=True)
        extrap=(4*d['third_cumulant']-c['third_cumulant'])/3
        reverse=good.kinetic_statistics(dict(e,angular_speed=-e['angular_speed']),gamma)
        row=dict(input=e,drag=gamma,third=a['third_cumulant'],refined=b['third_cumulant'],reference=extrap,degree_scaled_error=abs(a['third_cumulant']-b['third_cumulant'])/(1+abs(b['third_cumulant'])),reference_scaled_error=abs(extrap-b['third_cumulant'])/(1+abs(b['third_cumulant'])),reflection_error=abs(a['third_cumulant']-reverse['third_cumulant']),minimum_probability=min(c['minimum_probability'],d['minimum_probability']),noise=a['noise'],first_law_residual=a['first_law_residual'],edge_probability=max(c['edge_probability'],d['edge_probability']))
        assert row['degree_scaled_error']<.003 and row['reference_scaled_error']<.005,row
        assert row['reflection_error']<1e-8 and row['minimum_probability']>-1e-12 and row['noise']>0 and row['first_law_residual']<1e-9 and row['edge_probability']<1e-7,row
        checks.append(row)
        if (i+1)%8==0:print('Independent physical settings',i+1,flush=True)
    limits=[]
    for gamma,t,lam,eta in itertools.product([.4,1.1],[.4,1.4],[.2,3.],[0.,1.]):
        e=dict(temperature_a=t,temperature_b=t,relaxation_rate=lam,angular_speed=0.,flip_rate=1.2,flux_fraction=eta)
        a=good.kinetic_statistics(e,gamma);isolated=good.kinetic_statistics(e,0.)
        h=1e-4;lo=good.kinetic_statistics(dict(e,temperature_a=t-h),gamma);hi=good.kinetic_statistics(dict(e,temperature_a=t+h),gamma)
        fdt=2*t*t*(hi['mean']-lo['mean'])/(2*h)
        assert abs(a['mean'])<1e-10 and abs(a['third_cumulant'])<1e-8 and abs(fdt-a['noise'])/a['noise']<2e-6
        assert max(abs(isolated[k]) for k in ['mean','noise','third_cumulant'])<1e-8
        limits.append(dict(input=e,drag=gamma,equilibrium_third=a['third_cumulant'],fdt_relative=abs(fdt-a['noise'])/a['noise'],isolated_third=isolated['third_cumulant']))
    for eta in [0.,.5,1.]:
        e=dict(temperature_a=1.6,temperature_b=.7,relaxation_rate=1.1,angular_speed=0.,flip_rate=1.2,flux_fraction=eta)
        a=good.kinetic_statistics(e,.6);reverse=good.kinetic_statistics(dict(e,temperature_a=.7,temperature_b=1.6,relaxation_rate=.6),1.1)
        assert abs(a['third_cumulant']+reverse['third_cumulant'])<1e-5
    print('Physical limits PASS',flush=True)
    clean=np.array([good.calibration_mean(e,.67) for e in inputs]);rows=[]
    for i,draw in enumerate(np.random.default_rng(146824).normal(size=(256,len(records)))):
        ys=clean+sigma*draw;data=[dict(r,value=float(y)) for r,y in zip(records,ys)]
        fit=good.Model().fit(data);res=(fit.predict(inputs)-ys)/sigma
        oe=scores(good,fit.drag);be=scores(bad,fit.drag)
        row=dict(parameter=fit.drag,chi2=float(res@res)/(len(records)-1),oracle_max=max(oe.values()),shortcut_min=min(be[n] for n in sensitive),anchor=be['stationary_mean_and_uniform_noise'])
        assert row['chi2']<1.5 and abs(fit.drag/.67-1)<.03 and row['oracle_max']<.04 and row['shortcut_min']>.04 and row['anchor']<.04
        rows.append(row)
        if (i+1)%32==0:print('Direct noise fits and all hidden predictions',i+1,round(time.monotonic()-start,1),flush=True)
    report=dict(revision=9,calibration_regenerated=False,controls=controls,physics=dict(settings=checks,limits=limits,max_reference_scaled_error=max(r['reference_scaled_error'] for r in checks),max_degree_scaled_error=max(r['degree_scaled_error'] for r in checks)),noise=dict(seed=146824,trials=256,method='Direct fit and prediction on every draw and every hidden input',oracle_worst=max(r['oracle_max'] for r in rows),shortcut_smallest_miss=min(r['shortcut_min'] for r in rows),max_calibration_chi2=max(r['chi2'] for r in rows)),frozen_truth_sha256=sha(frozen_path),seconds=time.monotonic()-start,source_sha256={str(p.relative_to(TASK)):sha(p) for p in TASK.rglob('*') if p.is_file() and not any(x in p.parts for x in ['__pycache__','.pytest_cache'])})
    (ROOT/'results/rotating-reservoir-r9-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',seconds=report['seconds'],noise=report['noise'])),flush=True)

if __name__=='__main__':main()
