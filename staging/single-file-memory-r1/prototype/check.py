"""Author-only single-file covariance prototype; no benchmark/model evaluations."""
import json
from pathlib import Path
from functools import lru_cache
import numpy as np
from scipy.special import roots_legendre


def oracle(t, s, diffusivity=1., density=1.):
    return np.sqrt(diffusivity / np.pi) / density * (np.sqrt(t+s)-np.sqrt(abs(t-s)))


def shortcut(t, s, diffusivity=1., density=1.):
    # Positive fractional-Brownian covariance with the exact one-time coefficient.
    return np.sqrt(diffusivity / (2*np.pi)) / density * (np.sqrt(t)+np.sqrt(s)-np.sqrt(abs(t-s)))


@lru_cache(None)
def nodes(n):
    return roots_legendre(n)


def crossing_covariance(x, t, s, diffusivity, angular_nodes=96):
    """Covariance of correlated Brownian threshold indicators at each fixed x.

    Integrate the bivariate normal CDF's correlation derivative; c=sin(theta)
    removes its endpoint square root. No tagged-particle covariance formula.
    """
    u,w=nodes(angular_nodes)
    angle=np.arcsin(np.sqrt(min(t,s)/max(t,s)))
    theta=(u+1)*angle/2
    a=np.asarray(x)[:,None]/np.sqrt(2*diffusivity*t)
    b=np.asarray(x)[:,None]/np.sqrt(2*diffusivity*s)
    exponent=-(a*a-2*np.sin(theta)*a*b+b*b)/(2*np.cos(theta)**2)
    return np.exp(exponent) @ (w*angle/(4*np.pi))


def reference(t,s,diffusivity=1.,density=1.,n=512):
    # rho * integral Cov(I_t,I_s) dx / rho^2 from the conserved particle rank.
    u,w=nodes(n)
    radius=12*np.sqrt(2*diffusivity*max(t,s))
    return float(radius*np.dot(w,crossing_covariance(radius*u,t,s,diffusivity))/density)


def lattice_current(t,s,scale,diffusivity=1.,density=1.):
    radius=12*np.sqrt(2*diffusivity*scale*max(t,s))
    number=int(np.ceil(radius*density))
    x=np.arange(-number,number+1)/density
    return float(np.sum(crossing_covariance(x,scale*t,scale*s,diffusivity))/(density*density*np.sqrt(scale)))


def rank_check(scale, samples=32768):
    # Independent direct Jepsen order statistics, same paths at both times.
    # Finite time/size, finite Monte Carlo uncertainty: supplementary only.
    rng=np.random.default_rng(49023+scale)
    t,s=4.,1.;n=int(np.ceil(9*np.sqrt(2*scale*t)))
    initial=np.arange(-n,n+1,dtype=float)
    products=[];early=[];late=[]
    for start in range(0,samples,512):
        count=min(512,samples-start)
        positions=initial+np.sqrt(2*scale*s)*rng.normal(size=(count,len(initial)))
        xs=np.partition(positions,n,axis=1)[:,n]
        positions+=np.sqrt(2*scale*(t-s))*rng.normal(size=positions.shape)
        xt=np.partition(positions,n,axis=1)[:,n]
        early.extend(xs);late.extend(xt);products.extend(xs*xt/np.sqrt(scale))
    early=np.asarray(early);late=np.asarray(late);products=np.asarray(products)
    mean=float(products.mean()-early.mean()*late.mean()/np.sqrt(scale))
    return {'scale':scale,'particles':2*n+1,'samples':samples,'estimate':mean,
            'standard_error':float(products.std(ddof=1)/np.sqrt(samples)),
            'limiting_covariance':oracle(t,s),'finite_time_bias_not_removed':True}


def main():
    rng=np.random.default_rng(49021)
    points=[(t,s,d,rho) for t,s in [(1,1),(.25,4),(1,4),(.7,.71),(2,2)] for d,rho in [(.8,.7),(1.2,1.5)]]
    points += [(*np.sort(rng.uniform(.25,4,2)),rng.uniform(.8,1.2),rng.uniform(.7,1.5)) for _ in range(16)]
    errors=[];refinement=[];calibration=[];scaling=[];psd=[]
    for t,s,d,rho in points:
        truth=oracle(t,s,d,rho);ref=reference(t,s,d,rho)
        errors.append(abs(truth-ref));refinement.append(abs(ref-reference(t,s,d,rho,768)))
        calibration.append(abs(oracle(t,t,d,rho)-shortcut(t,t,d,rho)))
        scaling.append(abs(oracle(7*t,7*s,d,rho)-np.sqrt(7)*truth))
    for _ in range(24):
        times=np.sort(rng.uniform(.1,5,20))
        psd.append([float(np.linalg.eigvalsh(np.array([[f(t,s) for t in times] for s in times])).min()) for f in [oracle,shortcut]])
    lattice=[{'scale':R,'covariance':lattice_current(4,1,R),'error':abs(lattice_current(4,1,R)-oracle(4,1))} for R in [1,4,16,64,256]]
    # Noiseless identification is a one-parameter linear fit in sqrt(D).
    cal=[(t,rho) for t in [.25,.5,1,2,4] for rho in [.7,1,1.5]]
    design=np.array([shortcut(t,t,1,rho) for t,rho in cal])
    recover=[]
    for D in np.linspace(.8,1.2,17):
        y=np.array([oracle(t,t,D,rho) for t,rho in cal]);fit=(design@y/(design@design))**2
        recover.append(abs(fit-D))
    gaps=[{'t':t,'s':s,'oracle':oracle(t,s),'shortcut':shortcut(t,s),'relative_gap':shortcut(t,s)/oracle(t,s)-1} for t,s in [(1,.25),(2,.5),(4,1),(4,.25),(2,1)]]
    report={'status':'prototype_only_no_model_evaluation','independent_reference':'Correlated Brownian crossing-indicator CDF quadrature integrated over initial positions; conserved-rank conversion. No closed covariance formula is used by reference.',
            'points':len(points),'max_reference_error':max(errors),'max_reference_refinement':max(refinement),'exact_diagonal_calibration_error':max(calibration),'time_scaling_error':max(scaling),'minimum_covariance_eigenvalues':np.min(psd,axis=0).tolist(),'noiseless_fit_max_absolute_error':max(recover),'calibration_design_norm_squared':float(design@design),'diffusivity_sensitivity_min':float(np.min(design)/(2*np.sqrt(1.2))),
            'lattice_current_covariance_convergence':lattice,'independent_finite_rank_checks':[rank_check(R) for R in [16,64,256]],'example_hidden_gaps':gaps,
            'limitations':['Finite-rank Monte Carlo is a consistency check, not the limiting oracle or a proposed verifier.','The finite-lattice crossing covariance converges to the integral, but its equality to tracer covariance holds only at the stated joint long-time scale.','No empirical agent difficulty is claimed.']}
    assert max(errors)<2e-8 and max(refinement)<2e-8
    assert max(calibration)<1e-14 and min(np.min(psd,axis=0))>0
    assert max(recover)<1e-14 and min(x['relative_gap'] for x in gaps)>.3
    out=Path(__file__).with_name('assessment.json');out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
