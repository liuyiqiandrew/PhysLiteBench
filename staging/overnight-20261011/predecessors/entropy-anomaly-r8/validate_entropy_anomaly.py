"""Independent finite-mass validation of the cyclic-contact entropy task."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,platform,time
import numpy as np
import scipy

ROOT=Path(__file__).resolve().parents[1];TASK=ROOT/'tasks/entropy-anomaly'
def load(n,p):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def error(a,b):return float(np.linalg.norm(a-b)/np.linalg.norm(b))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args();start=time.monotonic()
    good=load('oracle',TASK/'solution/model.py');bad=load('shortcut',ROOT/'scripts/entropy_anomaly_baseline.py');ref=load('reference',TASK/'tests/reference.py')
    rs=json.loads((TASK/'environment/data/calibration.json').read_text());assert rs==json.loads((TASK/'tests/data/calibration.json').read_text())
    inputs=[r['input'] for r in rs];y=np.array([r['value'] for r in rs]);sigma=np.array([r['sigma'] for r in rs]);hidden=ref.hidden_inputs()
    truth={n:ref.predict(es) for n,es in hidden.items()};frozen_path=TASK/'tests/data/hidden_truth.json'
    if args.freeze:
        frozen_path.write_text(json.dumps(dict(description='Original finite-mass Hermite/Fourier equations, measured-residence subtraction, then three-mass extrapolation.',reference_sha256=sha(TASK/'tests/reference.py'),true_parameter=1.1,environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,machine=platform.machine()),groups={n:dict(inputs=hidden[n],truth=ys.tolist()) for n,ys in truth.items()}),indent=2)+'\n')
    frozen=json.loads(frozen_path.read_text());assert frozen['reference_sha256']==sha(TASK/'tests/reference.py')
    assert all(error(np.array(frozen['groups'][n]['truth']),truth[n])<1e-8 for n in hidden)
    def scores(module,friction):return {n:error(module.coefficients(es,friction),truth[n]) for n,es in hidden.items()}
    sensitive=['cyclic_anisotropic_contacts','cyclic_magnetic_drive'];anchors=[n for n in hidden if n not in sensitive]
    controls={}
    for name,m in [('oracle',good),('shortcut',bad)]:
        fit=m.Model().fit(rs);res=(fit.predict(inputs)-y)/sigma
        row=dict(parameter=fit.friction,calibration_chi2=float(res@res)/(len(rs)-1),hidden=scores(m,fit.friction))
        assert abs(fit.friction/1.1-1)<.03 and row['calibration_chi2']<1.5
        assert max(row['hidden'].values())<.04 if name=='oracle' else min(row['hidden'][n] for n in sensitive)>.04
        assert max(row['hidden'][n] for n in anchors)<.04
        controls[name]=row
    print('Controls',json.dumps(controls),flush=True)
    refinements=[]
    for n,es in hidden.items():
        fine=ref.predict(es,degree=14,spatial=51,epsilon=.003)
        normal=good.coefficients(es,1.1)
        dense=np.array([good.limiting_state(e,1.1,points=193)['dx']*np.sum(good.limiting_state(e,1.1,points=193)['local']*(1 if e.get('detector_phase') is None else (1+np.cos(good.limiting_state(e,1.1,points=193)['phase']-e['detector_phase']))/2)) for e in es])
        row=dict(group=n,independent_error=error(normal,truth[n]),reference_refinement=error(fine,truth[n]),spatial_refinement=error(dense,normal))
        assert row['independent_error']<.001 and row['reference_refinement']<.001 and row['spatial_refinement']<1e-8,row
        refinements.append(row);print('Hidden reference refinement',json.dumps(row),flush=True)
    rng=np.random.default_rng(202610118);checks=[]
    for i in range(32):
        gamma=float(rng.uniform(.7,1.6));e=ref.experiment(force_x=float(rng.uniform(-1.5,1.5)),force_y=float(rng.uniform(-1.5,1.5)),drag_ratio=float(np.exp(rng.uniform(np.log(.25),np.log(8.)))),magnetic=float(rng.uniform(-5,5)),temperature=float(rng.uniform(.8,1.4)),contrast=float(rng.uniform(0,.65)),wavenumber=int(rng.integers(1,4)),bath_contrast=float(rng.uniform(0,.65)),switch_rate=float(rng.uniform(.2,2.)))
        e.update(valve_cycle=float(rng.uniform(-.9,.9)),contact_drag_contrast=float(rng.uniform(0,.7)),detector_phase=float(rng.uniform(-np.pi,np.pi)))
        state=good.limiting_state(e,gamma);a=good.coefficients([e],gamma)[0];b=ref.predict([e],gamma,degree=12,spatial=41,epsilon=.002)[0]
        reverse=dict(e,force_x=-e['force_x'],force_y=-e['force_y'],detector_phase=-e['detector_phase'])
        parity=good.coefficients([reverse],gamma)[0]
        factors,gx,gy,q=good.contact_parameters(e,gamma)
        sum_correction=np.max(abs(np.sum(state['density_correction'][0,0],axis=0)))
        assert abs(a-b)/(1+abs(a))<.002 and abs(a-parity)<1e-8 and state['density'].min()>0 and state['diffusion'][0]>0 and sum_correction<1e-9
        assert np.max(abs(q.sum(axis=0)))<1e-12 and np.max(abs(q@np.ones(3)))<1e-12
        checks.append(dict(input=e,friction=gamma,oracle=float(a),reference=float(b),scaled_error=float(abs(a-b)/(1+abs(a))),parity_error=float(abs(a-parity)),minimum_density=float(state['density'].min()),diffusivity=float(state['diffusion'][0]),population_normalization_error=float(sum_correction)))
        if (i+1)%8==0:print('Independent random physics checks',i+1,flush=True)
    limits=[]
    for bias in [-.9,0.,.9]:
        for ratio in [.25,1.,8.]:
            e=ref.experiment(force_x=0.,force_y=0.,contrast=0.,bath_contrast=0.,drag_ratio=ratio)
            e.update(valve_cycle=bias,contact_drag_contrast=.7,detector_phase=.3)
            value=good.coefficients([e],1.1)[0];assert abs(value)<1e-8
            # For identical temperatures and drags, adding labels changes no motion.
            e.update(contrast=.5,force_x=.3,force_y=-.4,contact_drag_contrast=0.)
            old={k:v for k,v in e.items() if k not in ['valve_cycle','contact_drag_contrast']}
            difference=abs(good.coefficients([e],1.1)[0]-good.coefficients([old],1.1)[0]);assert difference<1e-8
            limits.append(dict(bias=bias,drag_ratio=ratio,equilibrium=float(value),identical_contact_error=float(difference)))
    clean=good.coefficients(inputs,1.1);rows=[]
    for i,draw in enumerate(np.random.default_rng(903118).normal(size=(256,len(rs)))):
        values=clean+sigma*draw;data=[dict(r,value=float(v)) for r,v in zip(rs,values)]
        fit=good.Model().fit(data);res=(fit.predict(inputs)-values)/sigma;oe=scores(good,fit.friction);be=scores(bad,fit.friction)
        row=dict(parameter=fit.friction,chi2=float(res@res)/(len(rs)-1),oracle_max=max(oe.values()),shortcut_min=min(be[n] for n in sensitive),anchor_max=max(be[n] for n in anchors))
        assert abs(fit.friction/1.1-1)<.03 and row['chi2']<1.5 and row['oracle_max']<.04 and row['shortcut_min']>.04 and row['anchor_max']<.04,row
        rows.append(row)
        if (i+1)%64==0:print('Direct noise fits and hidden predictions',i+1,flush=True)
    report=dict(revision=8,calibration_regenerated=False,controls=controls,physics=dict(hidden_refinements=refinements,random_cases=checks,limits=limits),noise=dict(seed=903118,trials=256,method='Direct fit and prediction on every hidden input for every draw',oracle_worst=max(r['oracle_max'] for r in rows),shortcut_smallest_miss=min(r['shortcut_min'] for r in rows),anchor_max=max(r['anchor_max'] for r in rows),max_calibration_chi2=max(r['chi2'] for r in rows)),seconds=time.monotonic()-start,source_sha256={str(p.relative_to(TASK)):sha(p) for p in TASK.rglob('*') if p.is_file() and not any(x in p.parts for x in ['__pycache__','.pytest_cache'])})
    (ROOT/'results/entropy-anomaly-r8-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',seconds=report['seconds'],noise=report['noise'])),flush=True)
if __name__=='__main__':main()
