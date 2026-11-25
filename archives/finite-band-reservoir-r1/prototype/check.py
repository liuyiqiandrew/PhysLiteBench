"""Prototype only: preparation memory of localized reservoir-coupled modes."""
from pathlib import Path
import json,time
from functools import lru_cache
import numpy as np
from scipy.special import expit
from scipy.linalg import eigh_tridiagonal
from scipy.fft import dst
from scipy.optimize import brentq

@lru_cache(None)
def quadrature(n):
    x,w=np.polynomial.legendre.leggauss(n)
    theta=np.pi*(x+1)/2;weights=w*np.pi/2
    return theta,weights

def spectral(epsilon,temperature,coupling,mu=0.,n=384):
    theta,w=quadrature(n);E=-2*np.cos(theta);sin=np.sin(theta)
    f=expit(-(E-mu)/temperature)
    surface=(E-2j*sin)/2
    density=-np.imag(1/(E-epsilon-coupling**2*surface))/np.pi
    continuous=float(w@(density*f*2*sin))
    mass=float(w@(density*2*sin))
    def outside(x):return (x-np.sign(x)*np.sqrt(x*x-4))/2
    def denominator(x):return x-epsilon-coupling**2*outside(x)
    roots=[]
    if coupling**2>2-epsilon:roots.append(brentq(denominator,2+1e-12,2+abs(epsilon)+2*coupling))
    if coupling**2>2+epsilon:roots.append(brentq(denominator,-2-abs(epsilon)-2*coupling,-2-1e-12))
    bound=[];correct=continuous;equilibrium=continuous
    for pole in roots:
        derivative=(1-abs(pole)/np.sqrt(pole*pole-4))/2
        residue=1/(1-coupling**2*derivative)
        occupied= residue*(1+coupling**2*float(w@(2*sin**2/np.pi*f/(pole-E)**2)))
        correct+=residue*occupied
        equilibrium+=residue*expit(-(pole-mu)/temperature)
        mass+=residue
        bound.append(dict(energy=pole,residue=residue,prepared_occupation=occupied,thermal_occupation=float(expit(-(pole-mu)/temperature))))
    return dict(correct=correct,shortcut=equilibrium,continuum=continuous,spectral_weight=mass,bound=bound)

def finite_chain(epsilon,temperature,coupling,length,mu=0.):
    diagonal=np.zeros(length+1);diagonal[0]=epsilon
    hopping=-np.ones(length);hopping[0]=coupling
    E,U=eigh_tridiagonal(diagonal,hopping)
    bath_energies=-2*np.cos(np.pi*np.arange(1,length+1)/(length+1))
    f=expit(-(bath_energies-mu)/temperature)
    projection=dst(U[1:,:],type=1,axis=0,norm='ortho')
    population=U[0]**2+(f[:,None]*projection**2).sum(axis=0)
    value=float((U[0]**2)@population)
    equilibrium=float((U[0]**2)@expit(-(E-mu)/temperature))
    return dict(length=length,correct=value,shortcut=equilibrium,total_particles=float(population.sum()),initial_particles=float(1+f.sum()),energy=float(E@population),initial_energy=float(epsilon+bath_energies@f),eigen_population_min=float(population.min()),eigen_population_max=float(population.max()))

def main():
    start=time.monotonic();report={'status':'prototype_only_no_task_or_model_evaluations'}
    calibration=[];derivatives=[]
    for ed in [.3,.5,.8]:
        for T in [.2,.4,.7]:
            control=dict(epsilon=ed,temperature=T,coupling=.8)
            r=spectral(**control); assert not r['bound']
            slopes=[]
            for g in np.linspace(.65,.95,31):
                slopes.append((spectral(ed,T,g+1e-5)['shortcut']-spectral(ed,T,g-1e-5)['shortcut'])/2e-5)
            derivatives+=slopes;calibration.append(dict(controls=control,**r,minimum_coupling_derivative=min(slopes)))
    hidden=[]
    for ed,T,V in [(.4,.3,2.2),(.8,.7,2.6),(-.3,.4,2.),(0.,.25,2.4)]:
        exact=spectral(ed,T,V);rows=[finite_chain(ed,T,V,N) for N in [128,256,512,1024]]
        ext=[2*rows[i]['correct']-rows[i-1]['correct'] for i in range(1,len(rows))]
        hidden.append(dict(controls=dict(epsilon=ed,temperature=T,coupling=V),spectral=exact,finite_chain=rows,linear_extrapolants=ext,finest_extrapolation_error=abs(ext[-1]-exact['correct']),shortcut_relative_error=abs(exact['shortcut']/exact['correct']-1),quadrature_refinement=max(abs(spectral(ed,T,V,n=768)[key]-exact[key]) for key in ['correct','shortcut'])))
    report.update(calibration=calibration,minimum_calibration_derivative=min(derivatives),calibration_information_fixed_sigma_0005=sum(r['minimum_coupling_derivative']**2/.0005**2 for r in calibration),hidden=hidden,elapsed_seconds=time.monotonic()-start)
    assert min(derivatives)>0
    report['minimum_prototype_shortcut_gap']=min(x['shortcut_relative_error'] for x in hidden)
    report['initial_15_percent_margin_check']=bool(report['minimum_prototype_shortcut_gap']>.15)
    assert max(x['finest_extrapolation_error'] for x in hidden)<1e-5
    path=Path(__file__).with_name('assessment.json');path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
