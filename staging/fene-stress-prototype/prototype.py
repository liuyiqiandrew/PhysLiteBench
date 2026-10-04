"""Steady planar FENE stress versus Peterlin preaveraging; no model runs."""
from pathlib import Path
import itertools,json,hashlib,time
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import roots_jacobi,iv
from scipy.optimize import brentq
from scipy.linalg import solve_continuous_lyapunov
from scipy.integrate import quad


def analytic(rate,temperature,drag,max_length,points=80):
    wi=drag*rate/4
    if max_length is None:
        c=np.diag([temperature/(1-2*wi),temperature/(1+2*wi)])
        return float(c[0,0]-c[1,1]),c
    b=max_length**2/temperature
    x,w=roots_jacobi(points,b/2,0);u=(x+1)/2
    z=np.dot(w,iv(0,wi*b*u))
    trace=temperature*b*np.dot(w,u*iv(0,wi*b*u))/z
    difference=temperature*b*np.dot(w,u*iv(1,wi*b*u))/z
    # Steady second-moment balance avoids a force singularity in the oracle.
    stress=2*wi*trace
    c=np.diag([(trace+difference)/2,(trace-difference)/2])
    return float(stress),c


def source(rate,temperature,drag,max_length):
    wi=drag*rate/4
    if max_length is None:return analytic(rate,temperature,drag,None)
    b=max_length**2/temperature
    lower=max(1.,2*abs(wi))+1e-12
    f=brentq(lambda f:1-1/f-(1/(f-2*wi)+1/(f+2*wi))/b,lower,100+2*abs(wi),xtol=1e-13)
    c=temperature*np.diag([1/(f-2*wi),1/(f+2*wi)])
    return float(f*(c[0,0]-c[1,1])),c


def cartesian(rate,temperature,drag,max_length,points=80):
    if max_length is None:
        drift=np.diag([rate,-rate])-2/drag*np.eye(2)
        c=solve_continuous_lyapunov(drift,-4*temperature/drag*np.eye(2))
        return float(c[0,0]-c[1,1]),c,0.
    q,w=leggauss(points)
    x=max_length*q[:,None]
    half=np.sqrt(max_length**2-x*x)
    y=half*q[None,:]
    weight=max_length*w[:,None]*half*w[None,:]
    gap=1-(x*x+y*y)/max_length**2
    potential=-max_length**2/2*np.log(gap)
    logdensity=(-potential+drag*rate*(x*x-y*y)/4)/temperature
    density=np.exp(logdensity-logdensity.max())
    weights=weight*density
    z=weights.sum()
    xx=float((weights*x*x).sum()/z);yy=float((weights*y*y).sum()/z)
    stress=float((weights*(x*x-y*y)/gap).sum()/z)
    # Pointwise stationary probability current, evaluated from the full force.
    dlogx=(-x/gap+drag*rate*x/2)/temperature
    dlogy=(-y/gap-drag*rate*y/2)/temperature
    currentx=rate*x-2*x/(drag*gap)-2*temperature/drag*dlogx
    currenty=-rate*y-2*y/(drag*gap)-2*temperature/drag*dlogy
    current=float(max(np.max(abs(currentx)),np.max(abs(currenty))))
    return stress,np.diag([xx,yy]),current


def direct_radial_force(rate,temperature,drag,max_length,points=80):
    b=max_length**2/temperature;wi=drag*rate/4
    x,w=roots_jacobi(points,b/2,0);u=(x+1)/2
    den=np.dot(w,iv(0,wi*b*u))
    x,w=roots_jacobi(points,b/2-1,0);u=(x+1)/2
    return float(2*temperature*b*np.dot(w,u*iv(1,wi*b*u))/den)


def main():
    start=time.monotonic();rng=np.random.default_rng(241131)
    cases=list(itertools.product([.35,.6,.95],[.8,1.2],[3.2,4.8],[np.sqrt(6),np.sqrt(12)]))
    cases += [(rng.uniform(.35,.95),rng.uniform(.8,1.2),rng.uniform(3.2,4.8),rng.uniform(np.sqrt(6),np.sqrt(12))) for _ in range(24)]
    rows=[]
    for rate,temp,drag,length in cases:
        exact,c=analytic(rate,temp,drag,length)
        wrong,cw=source(rate,temp,drag,length)
        ref,cr,current=cartesian(rate,temp,drag,length)
        finer,_,_=cartesian(rate,temp,drag,length,128)
        force=direct_radial_force(rate,temp,drag,length)
        f=1/(1-np.trace(cw)/length**2)
        kappa=np.diag([rate,-rate])
        residual=kappa@cw+cw@kappa-4/drag*(f*cw-temp*np.eye(2))
        rows.append(dict(rate=rate,temperature=temp,drag=drag,max_length=length,oracle=exact,source=wrong,
                    relative_error=wrong/exact-1,reference_relative_error=abs(ref/exact-1),
                    reference_refinement_relative_error=abs(finer/ref-1),force_average_relative_error=abs(force/exact-1),
                    source_equation_residual=float(np.max(abs(residual))),source_extension_fraction=float(np.trace(cw)/length**2),
                    exact_extension_fraction=float(np.trace(c)/length**2),stationary_current_max=current))
    calibration=[]
    for rate,temp,drag in itertools.product([-.22,-.12,.06,.18,.22],[.8,1.,1.2],[3.2,4.,4.8]):
        a,c=analytic(rate,temp,drag,None);s,_=source(rate,temp,drag,None);r,_,_=cartesian(rate,temp,drag,None)
        calibration.append(dict(rate=rate,temperature=temp,drag=drag,equivalence=abs(a-s),reference_error=abs(a-r),
                         stability_margin=2/drag-abs(rate),drag_derivative=temp*rate*(1+(drag*rate/2)**2)/(1-(drag*rate/2)**2)**2))
    limits=[]
    for length in [4,10,30,100]:
        b=length**2
        # Direct unnormalized radial integration avoids Jacobi normalizations
        # overflowing in this far-outside-domain asymptotic check. At wi=.1,
        # the omitted r²>200 tail is bounded by exp(-.4*r²).
        def moment(order):
            return quad(lambda q: q**order*np.exp(.5*b*np.log1p(-q/b))*iv(0,.1*q),
                        0,min(b,200.),epsabs=1e-12,epsrel=1e-12)[0]
        a=.2*moment(1)/moment(0);s,_=source(.1,1.,4.,length);h,_=analytic(.1,1.,4.,None)
        limits.append(dict(max_length=length,oracle=a,source=s,hookean=h))
    equilibrium=[]
    for temp,drag,length in itertools.product([.8,1.2],[3.2,4.8],[np.sqrt(6),np.sqrt(12)]):
        a,c=analytic(0.,temp,drag,length);r,cr,_=cartesian(0.,temp,drag,length,128)
        b=length**2/temp;known=2*temp*b/(b+4)
        equilibrium.append(max(abs(a),abs(r),abs(np.trace(c)-known),np.max(abs(c-cr))))
    report=dict(status='prototype_only_not_evaluated',cases=rows,
        case_count=len(rows),source_relative_error_range=[min(r['relative_error'] for r in rows),max(r['relative_error'] for r in rows)],
        signal_range=[min(r['oracle'] for r in rows),max(r['oracle'] for r in rows)],
        reference_relative_error_max=max(r['reference_relative_error'] for r in rows),
        reference_refinement_relative_error_max=max(r['reference_refinement_relative_error'] for r in rows),
        direct_force_average_relative_error_max=max(r['force_average_relative_error'] for r in rows),
        source_equation_residual_max=max(r['source_equation_residual'] for r in rows),
        extension_fraction_max=max(max(r['source_extension_fraction'],r['exact_extension_fraction']) for r in rows),
        stationary_current_max=max(r['stationary_current_max'] for r in rows),
        calibration=dict(cases=len(calibration),oracle_shortcut_max=max(r['equivalence'] for r in calibration),
              independent_ou_max=max(r['reference_error'] for r in calibration),stability_margin_min=min(r['stability_margin'] for r in calibration),
              nonzero_drag_sensitivity_min=min(abs(r['drag_derivative']) for r in calibration)),
        equilibrium_error_max=max(equilibrium),hookean_limit=limits,
        integrability='b=Qmax²/T in[5,15]; probability near the boundary scales as distance^(b/2). Both spring-force first and second moments are integrable.',
        notes='The first supplementary Hookean-limit check at b10000 overflowed the unnormalized Jacobi weights; final outside-domain limit uses direct radial quadrature with a controlled tail. Initial screen used the probability quadrature weight directly for the endpoint-singular force. Final direct-force check uses the correct Jacobi exponent b/2-1; an independent Cartesian force quadrature also verifies it.',
        seconds=time.monotonic()-start,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    assert all(np.isfinite(r['oracle']) for r in limits)
    assert abs(limits[-1]['oracle']/limits[-1]['hookean']-1)<.001
    assert report['reference_relative_error_max']<2e-6
    assert report['reference_refinement_relative_error_max']<2e-6
    assert report['direct_force_average_relative_error_max']<1e-10
    assert report['source_equation_residual_max']<1e-10
    assert report['source_relative_error_range'][0]>.2
    Path(__file__).with_name('report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))

if __name__=='__main__':main()
