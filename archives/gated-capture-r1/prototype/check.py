"""Stationary capture by a telegraph-gated sphere: unevaluated prototype."""
from pathlib import Path
import json
import time
import numpy as np
from scipy.integrate import solve_bvp
from scipy.optimize import minimize_scalar


def stationary_gate(k01,k10):
    return np.array([k10,k01])/(k01+k10)


def homogenized(D,a,kappa0,kappa1,k01,k10,concentration=1.):
    pi=stationary_gate(k01,k10)
    kappa=pi@np.array([kappa0,kappa1])
    return 4*np.pi*a*D*concentration*a*kappa/(D+a*kappa)


def modes(D,a,kappa0,kappa1,k01,k10,concentration=1.):
    pi=stationary_gate(k01,k10);v=np.array([1.,-1.]);kap=np.array([kappa0,kappa1])
    q=np.sqrt((k01+k10)/D)
    matrix=np.column_stack([pi*(D/a+kap),v*(D*(1/a+q)+kap)])
    A,B=np.linalg.solve(matrix,-kap*pi*concentration)
    surface=pi*(concentration+A)+v*B
    uptake=4*np.pi*a*a*np.dot(kap,surface)
    farflux=-4*np.pi*a*D*A
    assert abs(uptake-farflux)<2e-12*max(1.,abs(uptake))
    r=np.geomspace(a,a+40/q,400)
    density=pi[:,None]*(concentration+A*a/r)+v[:,None]*B*a/r*np.exp(-q*(r-a))
    assert density.min()>-1e-12 and np.max(density-pi[:,None]*concentration)<1e-12
    return {'uptake':float(uptake),'farflux':float(farflux),'surface':surface.tolist(),'A':float(A),'B':float(B),'minimum_density':float(density.min()),'screening_rate':float(q)}


def finite_shell(D,a,kappa0,kappa1,k01,k10,R,concentration=1.,tol=2e-9):
    """Independent direct radial diffusion/gate BVP, no mode substitution."""
    pi=stationary_gate(k01,k10);kap=np.array([kappa0,kappa1]);Q=np.array([[-k01,k10],[k01,-k10]])
    r=np.geomspace(a,R,240)
    def rhs(r,y):
        return np.vstack([y[2:]/(D*r*r),-r*r*(Q@y[:2])])
    def bc(ya,yb):return np.r_[ya[2:]-a*a*kap*ya[:2],yb[:2]-pi*concentration]
    guess=np.vstack([pi[:,None]*concentration*(1-.4*a/r),np.repeat((pi*concentration*.4*a*D)[:,None],len(r),axis=1)])
    sol=solve_bvp(rhs,bc,r,guess,tol=tol,bc_tol=tol,max_nodes=15000)
    assert sol.success,sol.message
    inner=sol.sol(a);outer=sol.sol(R)
    uptake=4*np.pi*a*a*np.dot(kap,inner[:2]);far=4*np.pi*np.sum(outer[2:])
    return {'uptake':float(uptake),'farflux':float(far),'balance_error':float(abs(uptake-far)),'nodes':sol.x.size,'minimum_density':float(sol.y[:2].min())}


def reference(D,a,kappa0,kappa1,k01,k10,concentration=1.,refinement=1):
    q=np.sqrt((k01+k10)/D)
    R0=max(80*a,a+18/q)*refinement
    Rs=R0*np.array([1,2,4])
    shells=[finite_shell(D,a,kappa0,kappa1,k01,k10,R,concentration) for R in Rs]
    # Remove the finite outer-reservoir location independently by a polynomial
    # in 1/R; no infinite-domain Robin admittance or mode coefficient is used.
    rate=np.polynomial.polynomial.polyfit(1/Rs,[s['uptake'] for s in shells],2)[0]
    return float(rate),{'radii':Rs.tolist(),'shells':shells}


def main():
    start=time.monotonic();rows=[]
    controls=[(.7,.6,.05,8.,.04,.12),(1.,1.,0.,10.,.1,.1),(1.3,1.7,.2,5.,.06,.2),
              (1.1,1.4,6.,.1,.3,.07),(.9,.8,.1,15.,.05,.25),(.8,1.8,10.,.1,.15,.15)]
    for args in controls:
        oracle=modes(*args);base=homogenized(*args);ref,detail=reference(*args)
        error=abs(ref-oracle['uptake'])/oracle['uptake'];gap=abs(base-oracle['uptake'])/oracle['uptake']
        assert error<2e-6 and gap>.04
        assert max(s['balance_error'] for s in detail['shells'])<1e-7
        rows.append({'controls':args,'oracle':oracle,'shortcut':base,'reference':ref,'reference_relative_error':error,'shortcut_relative_error':gap,'reference_details':detail})
    refinements=[]
    for args in [controls[0],controls[-1]]:
        r1,_=reference(*args);r2,_=reference(*args,refinement=2)
        refinements.append({'controls':args,'relative_change':abs(r2-r1)/r2});assert abs(r2-r1)/r2<2e-6
    # Calibration at equal surface reactivities is exact for every switching rate.
    calibration=[(a,kap,kap,k01,k10) for a in [.6,1.,1.7] for kap in [.5,1.5,5.] for k01,k10 in [(.04,.2),(1.,.3)]]
    equivalence=[];fits=[];derivative=[]
    for D in np.linspace(.7,1.3,31):
        y=np.array([modes(D,*e)['uptake'] for e in calibration])
        src=np.array([homogenized(D,*e) for e in calibration]);equivalence.extend(abs(y-src))
        fit=minimize_scalar(lambda d:np.sum((np.array([homogenized(d,*e) for e in calibration])-y)**2),bounds=(.7,1.3),method='bounded',options={'xatol':1e-12})
        fits.append({'true':float(D),'fit':float(fit.x),'absolute_error':float(abs(fit.x-D))})
        for a,kap,_,_,_ in calibration:derivative.append(4*np.pi*a*(a*kap)**2/(D+a*kap)**2)
    assert max(equivalence)<1e-12 and max(x['absolute_error'] for x in fits)<1e-6 and min(derivative)>.2
    limits=[];D,a,k0,k1,u,v=1.,1.2,.1,7.,.1,.3;pi=stationary_gate(u,v)
    slow=pi[0]*homogenized(D,a,k0,k0,u,v)+pi[1]*homogenized(D,a,k1,k1,u,v)
    fast=homogenized(D,a,k0,k1,u,v)
    for factor in [1e-12,1e-8,1.,1e8,1e12]:
        physical=modes(D,a,k0,k1,u*factor,v*factor)['uptake']
        limits.append({'rate_multiplier':factor,'uptake':physical,'slow_limit':slow,'fast_limit':fast})
    assert abs(limits[0]['uptake']/slow-1)<2e-6 and abs(limits[-1]['uptake']/fast-1)<2e-5
    sym=[]
    for args in controls:
        D,a,k0,k1,u,v=args;x=modes(*args)['uptake'];other=modes(D,a,k1,k0,v,u)['uptake']
        scaled=modes(2*D,a,2*k0,2*k1,2*u,2*v)['uptake']
        sym.append({'swap_error':abs(x-other),'time_scaling_error':abs(scaled-2*x)})
        assert max(sym[-1].values())<1e-12
    report={'status':'prototype_passed','scope':'No full task package or model evaluations.','cases':rows,'reference_refinements':refinements,'calibration':{'unique_controls':len(calibration),'maximum_equivalence_error':float(max(equivalence)),'minimum_diffusivity_derivative':float(min(derivative)),'full_range_noiseless_fits':fits},'limits':limits,'symmetry_scaling':sym,'elapsed_seconds':time.monotonic()-start}
    p=Path(__file__).with_name('report.json');p.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'maximum_reference_error':max(r['reference_relative_error'] for r in rows),'shortcut_gap_range':[min(r['shortcut_relative_error'] for r in rows),max(r['shortcut_relative_error'] for r in rows)],'calibration_equivalence':max(equivalence),'minimum_parameter_derivative':min(derivative),'elapsed':report['elapsed_seconds']},indent=2))

if __name__=='__main__':main()
