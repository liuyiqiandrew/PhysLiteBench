"""Deterministic author checks; no model-agent evaluations."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import numpy as np

STAGE=Path(__file__).resolve().parents[1]
TASK=STAGE/'tasks/active-wall-pressure'

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def local_test(model):
    with tempfile.TemporaryDirectory(prefix='active-wall-control-') as d:
        app=Path(d);shutil.copytree(TASK/'environment',app,dirs_exist_ok=True)
        shutil.copy2(model,app/'model.py')
        start=time.monotonic()
        run=subprocess.run([sys.executable,'-m','pytest','-q',str(app/'test_public.py'),str(TASK/'tests/test_hidden.py')],cwd=app,env={**os.environ,'PYTHONPATH':str(app),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'},capture_output=True,text=True)
        return {'returncode':run.returncode,'elapsed_seconds':time.monotonic()-start,'stdout':run.stdout,'stderr':run.stderr}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--generate',action='store_true');args=parser.parse_args()
    start=time.monotonic()
    oracle=load(TASK/'solution/model.py','oracle');base=load(STAGE/'scripts/active_wall_pressure_baseline.py','baseline');ref=load(TASK/'tests/reference.py','reference')
    inputs=ref.calibration_inputs();truth=ref.predict(inputs,ref.TRUE_PARAMETER)
    if args.generate:
        rng=np.random.default_rng(141031)
        rows=[{'input':e,'value':float(y),'sigma':ref.SIGMA} for e,y in zip(inputs,truth+rng.normal(0,ref.SIGMA,len(inputs)))]
        text=json.dumps(rows,indent=2)+'\n'
        for p in [TASK/'environment/data/calibration.json',TASK/'tests/data/calibration.json']:p.write_text(text)
    rows=json.loads((TASK/'environment/data/calibration.json').read_text());assert (TASK/'environment/data/calibration.json').read_bytes()==(TASK/'tests/data/calibration.json').read_bytes()
    a=oracle.Model().fit(rows);b=base.Model().fit(rows)
    oa=oracle.predict_at(inputs,1);ba=base.predict_at(inputs,1);assert np.array_equal(oa,ba)
    assert np.min(oa)>0
    calres=(a.predict(inputs)-np.asarray([r['value'] for r in rows]))/ref.SIGMA
    groups=ref.hidden_inputs();hidden={}
    for name,es in groups.items():
        y=ref.predict(es,ref.TRUE_PARAMETER);o=oracle.predict_at(es,1);s=base.predict_at(es,1)
        hidden[name]={'truth':y.tolist(),'oracle_unit':o.tolist(),'shortcut_unit':s.tolist(),'oracle_nrms':float(np.linalg.norm(a.loading*o-y)/np.linalg.norm(y)),'shortcut_nrms':float(np.linalg.norm(b.loading*s-y)/np.linalg.norm(y)),'reference_signal_min':float(np.min(y))}
        assert hidden[name]['oracle_nrms']<.04
        if name!='torque_free_anchors':assert hidden[name]['shortcut_nrms']>.04
    rng=np.random.default_rng(141037);fits=[];chis=[];worst_o=0;best_b=1e10
    for _ in range(256):
        ys=truth+rng.normal(0,ref.SIGMA,len(truth))
        fit=float(np.clip(np.dot(oa,ys)/np.dot(oa,oa),.8,1.2));fits.append(fit)
        chi=float(np.sum(((fit*oa-ys)/ref.SIGMA)**2)/(len(ys)-1));chis.append(chi)
        assert abs(fit/ref.TRUE_PARAMETER-1)<.03 and chi<1.5
        for name,row in hidden.items():
            y=np.array(row['truth']);o=fit*np.array(row['oracle_unit']);s=fit*np.array(row['shortcut_unit'])
            eo=float(np.linalg.norm(o-y)/np.linalg.norm(y));eb=float(np.linalg.norm(s-y)/np.linalg.norm(y));worst_o=max(worst_o,eo)
            assert eo<.04
            if name!='torque_free_anchors':best_b=min(best_b,eb);assert eb>.04
    # Independent numerical checks at all scored controls and supported corners.
    corners=[dict(speed=v,temperature=T,half_width=L,alignment=h) for v in [.9,1.8] for T in [.45,.7] for L in [1.,2.] for h in [-1.,0.,2.]]
    comparisons=[]
    for e in corners:
        vals=[e[k] for k in ('speed','temperature','half_width','alignment')]
        o=oracle.unit_pressure(*vals);r=ref.unit_pressure(*vals)
        fine=(4*oracle._pressure_on_grid(*vals,513)-oracle._pressure_on_grid(*vals,257))/3
        x,c=oracle._moments(*vals,257)
        theta=np.arange(128)*2*np.pi/128
        p=(c[:,0,None]+2*(c[:,1:]@np.cos(np.arange(1,c.shape[1])[:,None]*theta)))/(2*np.pi)
        comparisons.append({'input':e,'oracle_reference_relative_error':abs(o-r)/abs(r),'oracle_refinement_relative_change':abs(o-fine)/abs(fine),'minimum_density':float(p.min()),'normalization_error':float(abs(np.sum(c[:,0])*(x[1]-x[0])-1)),'signal':r})
        assert abs(o-r)/r<.002 and abs(o-fine)/fine<.002 and p.min()>-1e-5 and r>0
    # Resolve the most strongly aligned and cold endpoint with a finer independent grid.
    refinements=[]
    for e in [dict(speed=1.8,temperature=.45,half_width=1.,alignment=2.),dict(speed=1.8,temperature=.45,half_width=1.,alignment=-1.)]:
        vals=[e[k] for k in ('speed','temperature','half_width','alignment')]
        old=ref.unit_pressure(*vals);new=ref.unit_pressure(*vals,refine=2)
        refinements.append({'input':e,'relative_change':abs(new-old)/abs(new)});assert abs(new-old)/abs(new)<.002
    # At zero propulsion the specified common-temperature dynamics is equilibrium.
    # Direct partition quadrature gives the normalized coating force independently.
    from scipy.integrate import simpson
    passive=[]
    for h in [-1.,0.,2.]:
        T=.55;L=1.4;x=np.linspace(-L-3,L+3,16001);theta=np.arange(512)*2*np.pi/512
        V,Vx,_=ref.potential(x[:,None],theta[None,:],L,4,h,.6)
        w=np.exp(-V/T);norm=simpson(w.mean(axis=1),x=x)
        force=simpson((w*Vx).mean(axis=1)[x>=0],x=x[x>=0])/norm
        val=oracle.unit_pressure(0.,T,L,h)
        passive.append({'alignment':h,'relative_error':abs(val-force)/force});assert abs(val-force)/force<.002
    report={'status':'scientific_checks_passed','scope':'No agent model evaluations.','calibration':{'records':len(rows),'unique_inputs':len({json.dumps(e,sort_keys=True) for e in inputs}),'loading':a.loading,'relative_error':abs(a.loading/ref.TRUE_PARAMETER-1),'chi2':float(calres@calres)/(len(rows)-1),'control_equivalence_max':float(np.max(abs(oa-ba))),'minimum_slope':float(np.min(oa)),'fisher_information':float(np.sum((oa/ref.SIGMA)**2)),'maximum_numerical_calibration_bias_sigma':float(np.max(abs(ref.TRUE_PARAMETER*oa-truth))/ref.SIGMA),'uncertainty':'constant sigma=.001, independent of loading and response'},'hidden':hidden,'noise256':{'all_parameter_and_calibration_pass':True,'all_oracle_pass':True,'all_shortcut_diagnostic_fail':True,'max_chi2':max(chis),'max_relative_parameter_error':max(abs(np.array(fits)/ref.TRUE_PARAMETER-1)),'max_oracle_hidden_error':worst_o,'minimum_shortcut_diagnostic_error':best_b,'loading_range':[min(fits),max(fits)]},'corners':comparisons,'reference_refinement':refinements,'passive_equilibrium_partition_check':passive,'elapsed_science_seconds':time.monotonic()-start}
    out=STAGE/'results/active-wall-pressure-r1-validation.json';out.write_text(json.dumps(report,indent=2)+'\n')
    controls={'oracle':local_test(TASK/'solution/model.py'),'shortcut':local_test(STAGE/'scripts/active_wall_pressure_baseline.py')}
    (STAGE/'results/active-wall-pressure-r1-local-controls.json').write_text(json.dumps(controls,indent=2)+'\n')
    assert controls['oracle']['returncode']==0
    assert controls['shortcut']['returncode']==1 and '3 failed, 4 passed' in controls['shortcut']['stdout']
    print(json.dumps({'calibration':report['calibration'],'noise256':report['noise256'],'local_controls':{k:{q:v[q] for q in ['returncode','elapsed_seconds']} for k,v in controls.items()}},indent=2))

if __name__=='__main__':main()
