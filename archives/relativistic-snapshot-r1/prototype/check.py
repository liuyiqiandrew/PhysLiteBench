"""Unevaluated equal-laboratory-time particle-selection prototype."""
from pathlib import Path
import json
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import kve
from scipy.optimize import minimize_scalar


def gamma(v):return 1/np.sqrt(1-v*v)


def rest_mean_energy(mass,temperature):
    z=mass/temperature
    return mass*kve(1,z)/kve(2,z)+3*temperature


def tensor_reference(mass,temperature,gas_speed,analysis_speed):
    b,a=gas_speed,analysis_speed
    e=rest_mean_energy(mass,temperature)
    return gamma(a)*gamma(b)*((1-a*b)*e+b*(b-a)*temperature)


def rest_pushforward(mass,temperature,gas_speed,analysis_speed,n=96):
    z,w=leggauss(n);mu,wm=leggauss(48)
    qmax=np.arccosh(1+60*temperature/mass)
    q=(z+1)*qmax/2;w=w*qmax/2
    E=mass*np.cosh(q);p=mass*np.sinh(q)
    weights=w*p*p*E*np.exp(-(E-mass)/temperature)
    Elab=gamma(gas_speed)*(E[:,None]+gas_speed*p[:,None]*mu)
    pxlab=gamma(gas_speed)*(p[:,None]*mu+gas_speed*E[:,None])
    observable=gamma(analysis_speed)*(Elab-analysis_speed*pxlab)
    return float(np.sum(weights[:,None]*wm[None,:]*observable)/(2*np.sum(weights)))


def laboratory_integral(mass,temperature,gas_speed,analysis_speed,n=192,angular=128):
    # Integrate in lab momentum coordinates, not a reweighted rest quadrature.
    z,w=leggauss(n);mu,wm=leggauss(angular)
    Emax=gamma(gas_speed)*(1+abs(gas_speed))*(mass+60*temperature)
    qmax=np.arccosh(Emax/mass)
    q=(z+1)*qmax/2;w=w*qmax/2
    E=mass*np.cosh(q);p=mass*np.sinh(q)
    scalar=np.exp(-(gamma(gas_speed)*(E[:,None]-gas_speed*p[:,None]*mu)-mass)/temperature)
    weights=w[:,None]*p[:,None]**2*E[:,None]*wm[None,:]*scalar
    observable=gamma(analysis_speed)*(E[:,None]-analysis_speed*p[:,None]*mu)
    return float(np.sum(weights*observable)/np.sum(weights))


def worldline_check(mass,temperature,beta,analysis_speed,count=600000):
    # Independently draw a uniform rest-time spatial slice, then intersect each
    # inertial worldline with t_lab=0 before selecting the laboratory volume.
    rng=np.random.default_rng(151043)
    pgrid=np.linspace(0,60*temperature+20*mass,30001)
    Egrid=np.sqrt(mass*mass+pgrid*pgrid)
    pdf=pgrid*pgrid*np.exp(-(Egrid-mass)/temperature)
    cdf=np.r_[0,np.cumsum((pdf[1:]+pdf[:-1])*np.diff(pgrid)/2)];cdf/=cdf[-1]
    p=np.interp(rng.random(count),cdf,pgrid);E=np.sqrt(mass*mass+p*p)
    mu=rng.uniform(-1,1,count);px=p*mu;vx=px/E
    radius=gamma(beta)*(1+abs(beta));x0=rng.uniform(-radius,radius,count)
    t_rest=-beta*x0/(1+beta*vx)
    x_rest=x0+vx*t_rest
    x_lab=gamma(beta)*(x_rest+beta*t_rest)
    chosen=abs(x_lab)<1
    Elab=gamma(beta)*(E+beta*px);pxlab=gamma(beta)*(px+beta*E)
    measured=gamma(analysis_speed)*(Elab-analysis_speed*pxlab)
    data=measured[chosen]
    return {'samples':count,'selected':int(chosen.sum()),'sample_mean':float(data.mean()),'standard_error':float(data.std(ddof=1)/np.sqrt(len(data))),'tensor_prediction':tensor_reference(mass,temperature,beta,analysis_speed),'all_rest_labels_mean':float(measured.mean()),'rest_pushforward':rest_pushforward(mass,temperature,beta,analysis_speed)}


def main():
    cases=[]
    for mass in [.8,1.07,1.2]:
        for T in [.2,.5,.9]:
            for b,a in [(0,0),(.65,.65),(-.8,-.8),(.8,0),(.9,-.3),(-.85,.25)]:
                ref=tensor_reference(mass,T,b,a)
                direct=laboratory_integral(mass,T,b,a)
                source=rest_pushforward(mass,T,b,a)
                cases.append({'mass':mass,'temperature':T,'gas_speed':b,'analysis_speed':a,'reference':ref,'lab_integral':direct,'source':source,'relative_gap':abs(source-ref)/ref,'reference_error':abs(direct-ref)/ref})
                assert abs(direct-ref)/ref<2e-10
                if a==b:assert abs(source-ref)/ref<2e-12
    calibrations=[(T,b,b) for T in [.2,.35,.55,.8] for b in [-.8,0,.8]]
    massfit=[]
    for truth in np.linspace(.8,1.2,9):
        values=np.array([tensor_reference(truth,*c) for c in calibrations])
        fit=minimize_scalar(lambda m:np.sum((np.array([rest_pushforward(m,*c) for c in calibrations])-values)**2),bounds=(.8,1.2),method='bounded',options={'xatol':1e-12})
        massfit.append({'truth':float(truth),'fit':float(fit.x),'error':float(abs(fit.x-truth))})
    derivative=[]
    for m in np.linspace(.8,1.2,41):
        for T in [.2,.35,.55,.8]:
            derivative.append((rest_mean_energy(m+1e-5,T)-rest_mean_energy(m-1e-5,T))/(2e-5))
    report={'status':'prototype_passed','scope':'No task package or agent evaluations.','cases':cases,'calibration_noiseless_fits':massfit,'minimum_sampled_mass_derivative':float(min(derivative)),'worldline_reference':worldline_check(1.07,.55,.85,-.2),'calibration':'Analysis frame moves with gas, including nonzero gas boosts; fixed lab-time selection remains unchanged.','physics':'Rest-particle push-forward and laboratory-simultaneous volume sampling are different ensembles. Exact boosts alone do not transform the selected probability measure.'}
    p=Path(__file__).with_name('report.json');p.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'max_reference_error':max(c['reference_error'] for c in cases),'hidden_gap_range':[min(c['relative_gap'] for c in cases if c['analysis_speed']!=c['gas_speed']),max(c['relative_gap'] for c in cases if c['analysis_speed']!=c['gas_speed'])],'mass_derivative_min':min(derivative),'worldline_reference':report['worldline_reference']},indent=2))
if __name__=='__main__':main()
