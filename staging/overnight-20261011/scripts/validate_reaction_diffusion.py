from pathlib import Path
import hashlib,importlib.util,json,time
import numpy as np
from scipy.linalg import expm
S=Path('/Users/yp9846/Desktop/PhysLiteBench/staging/overnight-20261011');T=S/'tasks/reaction-diffusion';A=S/'predecessors/reaction-diffusion-r9'
def load(n,p):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def relative(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))
def main():
    tick=time.monotonic();good=load('good',T/'solution/model.py');bad=load('bad',S/'scripts/reaction_baseline.py');ref=load('ref',T/'tests/reference.py');prior=json.loads((A/'validation.json').read_text())
    assert all(sha(A/'task'/p)==h for p,h in prior['source_sha256'].items())
    for name in ['solution/model.py','tests/reference.py']:assert (T/name).read_text().startswith((A/'task'/name).read_text())
    assert sha(T/'environment/data/calibration.npz')==sha(A/'task/environment/data/calibration.npz')==sha(T/'tests/data/calibration.npz')
    data=ref.load_data();controls={}
    for name,m in [('oracle',good),('shortcut',bad)]:
        start=time.monotonic();fit=m.TransportModel().fit(data);row=ref.metrics(fit,data);row['seconds']=time.monotonic()-start
        assert row['calibration_reduced_chi2']<1.5 and row['relative_diffusivity_error']<.05 and max(row['hidden_nrmse'])<.05 and row['seconds']<45
        assert max(row['fluctuation_nrmse'].values())<.05 if name=='oracle' else min(row['fluctuation_nrmse'].values())>.05
        controls[name]=row
    print('CONTROLS',json.dumps(controls),flush=True)
    rng=np.random.default_rng(2026101110);domain=[];times=np.array([0.,.5,2.,10.,40.,120.,300.,600.])
    for i in range(64):
        totals=rng.uniform(.2,2.,2);D=float(rng.uniform(.5e-9,1.5e-9));h=int(rng.integers(1,9))
        if i<8:totals=np.array([(.2,2.)[i&1],(.2,2.)[(i>>1)&1]]);D=(.5e-9,1.5e-9)[(i>>2)&1];h=[1,8][i&1]
        C,M=good.fluctuation_matrices(totals,D);k2=(h*np.pi/.001)**2;rate=k2*M@np.linalg.inv(C)
        result=good.equilibrium_correlations(times,totals,h,D);independent=ref.correlations(times,totals,D,h)
        fine=ref.correlations(times,totals,D,h,separation=20000.)
        row=dict(totals=totals.tolist(),diffusivity=D,harmonic=h,independent_error=relative(result,independent),refinement_error=relative(result,fine),minimum_static_covariance_eigenvalue=float(np.linalg.eigvalsh(C).min()),minimum_mobility_eigenvalue=float(np.linalg.eigvalsh(M).min()),onsager_error=float(np.max(abs(M-M.T))),fdt_error=float(np.max(abs(rate@C+C@rate.T-2*k2*M))),time_reversal_error=float(np.max(abs(result-result.transpose(0,2,1)))))
        assert row['independent_error']<2e-6 and row['refinement_error']<1e-6,row
        assert row['minimum_static_covariance_eigenvalue']>0 and row['minimum_mobility_eigenvalue']>0 and row['time_reversal_error']<1e-10 and row['fdt_error']<1e-10
        assert min(np.linalg.eigvalsh(s).min() for s in result)>-1e-12
        # Independent ideal-mixture chemical potentials after minimizing association.
        derivative=[];epsilon=1e-5
        for axis in [0,1]:
            change=np.eye(2)[axis]*epsilon
            plus=ref.associated_initial((totals+change)[None,:])[0];minus=ref.associated_initial((totals-change)[None,:])[0]
            derivative.append((np.log(plus[:2]*plus[2])-np.log(minus[:2]*minus[2]))/(2*epsilon))
        hessian=np.column_stack(derivative);row['thermodynamic_hessian_error']=relative(hessian,np.linalg.inv(C));assert row['thermodynamic_hessian_error']<2e-8,row
        # Linearize the actual deterministic four-species flux, retaining binding equilibrium.
        equilibrium_jac=np.column_stack([(good.equilibrium((totals+np.eye(2)[j]*epsilon)[None,:])[0]-good.equilibrium((totals-np.eye(2)[j]*epsilon)[None,:])[0])/(2*epsilon) for j in [0,1]])
        c=good.equilibrium(totals[None,:])[0];z=np.array([1.,1.,-1.,0.]);transport=D*np.array([1.,4.,2.,.15]);q=transport*c*z
        flux_matrix=np.diag(transport)-np.outer(q,transport*z)/(z@q);readout=np.array([[1.,0.,0.,1.],[0.,1.,0.,1.]])
        diffusion=readout@flux_matrix@equilibrium_jac;row['regression_flux_error']=relative(diffusion,M@np.linalg.inv(C));assert row['regression_flux_error']<2e-8,row
        scale=good.equilibrium_correlations(2*times,totals,h,D);double=good.equilibrium_correlations(times,totals,h,2*D);row['diffusivity_time_scaling_error']=relative(scale,double);assert row['diffusivity_time_scaling_error']<1e-12
        if i<4:
            x=np.linspace(0,.001,129);mode=np.cos(h*np.pi*x/.001);dtimes=np.array([0.,2.,10.]);seed=np.array([.6,-.8]);initial=totals+1e-5*mode[:,None]*seed
            model=good.TransportModel();model.diffusivity=D;observed=model.predict(dtimes,x,initial)
            amplitude=2*np.trapezoid((observed-totals)*mode[None,:,None],x,axis=1)/.001/1e-5
            discrete_k2=4/(x[1]-x[0])**2*np.sin(h*np.pi/(2*(len(x)-1)))**2
            expected=np.array([expm(-t*discrete_k2*diffusion)@seed for t in dtimes]);row['deterministic_regression_error']=relative(amplitude,expected);assert row['deterministic_regression_error']<.003,row
        domain.append(row)
    print('64 independent thermodynamic, kinetic and regression checks passed',flush=True)
    model=good.TransportModel();model.diffusivity=ref.D;clean=np.array([model._binary(data['t'],data['x'],initial) for initial in data['initial']]);rng=np.random.default_rng(6112030);fits=[]
    for _ in range(256):
        sample=dict(data,concentration=clean+rng.normal(size=clean.shape)*data['sigma']);a=good.TransportModel().fit(sample);b=bad.TransportModel().fit(sample);pred=np.array([a._binary(data['t'],data['x'],initial) for initial in data['initial']]);res=(pred-sample['concentration'])/data['sigma'];chi=float(np.sum(res**2))/(res.size-1)
        assert chi<1.5 and abs(a.diffusivity/ref.D-1)<.05 and abs(a.diffusivity-b.diffusivity)<1e-20;fits.append(dict(parameter=a.diffusivity,chi2=chi))
    params=np.array([r['parameter'] for r in fits]);sensitivity=[]
    for d in np.quantile(params,np.linspace(0,1,9)):
        a=good.TransportModel();b=bad.TransportModel();a.diffusivity=b.diffusivity=float(d);ae=ref.metrics(a,data);be=ref.metrics(b,data)
        assert max(ae['hidden_nrmse'])<.05 and max(ae['fluctuation_nrmse'].values())<.05 and min(be['fluctuation_nrmse'].values())>.05
        sensitivity.append(dict(parameter=float(d),oracle=ae,shortcut=be))
    report=dict(revision=10,calibration_regenerated=False,inherited_r9_validation_sha256=sha(A/'validation.json'),controls=controls,physics=dict(domain=domain),noise=dict(seed=6112030,trials=256,method='256 actual calibration fits; all mean and covariance predictions at nine fitted-parameter quantiles including extrema.',fit_min=float(params.min()),fit_max=float(params.max()),max_chi2=max(r['chi2'] for r in fits),hidden_sensitivity=sensitivity),seconds=time.monotonic()-tick,source_sha256={str(p.relative_to(T)):sha(p) for p in T.rglob('*') if p.is_file() and not any(x in p.parts for x in ['__pycache__','.pytest_cache'])})
    (S/'results/reaction-diffusion-r10-validation.json').write_text(json.dumps(report,indent=2)+'\n');print('PASS',round(report['seconds'],1),flush=True)
if __name__=='__main__':main()
