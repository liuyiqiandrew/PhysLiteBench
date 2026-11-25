"""Independent stationary FENE solvers for rotating planar extension."""
import json
import time
from pathlib import Path
import numpy as np
from scipy.special import roots_jacobi, eval_jacobi
from scipy.linalg import solve, solve_continuous_lyapunov
from scipy.optimize import brentq
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def flow(extension, rotation):
    return np.array([[extension, -rotation], [rotation, -extension]])


def peterlin(extension, rotation, temperature, drag, length):
    kappa = flow(extension, rotation)
    def covariance(factor):
        a = 2*factor/drag*np.eye(2)-kappa
        return solve_continuous_lyapunov(a, 4*temperature/drag*np.eye(2))
    if length is None:
        c = covariance(1.0)
        return c[0, 0]-c[1, 1], c, 1.0
    lower = max(1., drag/2*np.sqrt(max(extension**2-rotation**2, 0)))+1e-11
    f = brentq(lambda f: 1-1/f-np.trace(covariance(f))/length**2,
               lower, 100+drag*(abs(extension)+abs(rotation)), xtol=1e-13)
    c = covariance(f)
    return f*(c[0, 0]-c[1, 1]), c, f


def spectral(extension, rotation, temperature, drag, length, degree=20):
    """Weighted disk-polynomial weak FP, no spring-force singular quadrature."""
    exponent = length**2/(2*temperature)
    nodes, weights = roots_jacobi(degree+4, exponent, 0)
    x = (nodes+1)/2
    r = np.sqrt(x)
    theta = 2*np.pi*np.arange(4*degree+8)/(4*degree+8)
    rr, tt = np.meshgrid(r, theta, indexing='ij')
    w = np.broadcast_to((weights/weights.sum())[:, None]/len(theta), rr.shape).ravel()
    vals, radial, angular = [], [], []
    for total in range(0, degree+1, 2):
        for m in range(0, total+1, 2):
            n = (total-m)//2
            p = eval_jacobi(n, exponent, m, nodes)
            rad = r**m*p
            der = m*r**(m-1)*p if m else np.zeros_like(r)
            if n:
                der += 2*r**(m+1)*(n+exponent+m+1)*eval_jacobi(n-1, exponent+1, m+1, nodes)
            for sine in ([False] if m==0 else [False, True]):
                angle = np.sin(m*theta) if sine else np.cos(m*theta)
                angle_der = m*np.cos(m*theta) if sine else -m*np.sin(m*theta)
                z = (rad[:, None]*angle).ravel()
                norm = np.sqrt(np.dot(w, z*z))
                vals.append(z/norm)
                radial.append((der[:, None]*angle).ravel()/norm)
                angular.append((rad[:, None]*angle_der/rr).ravel()/norm)
    v, gr, gt = np.array(vals), np.array(radial), np.array(angular)
    radius, angle = rr.ravel(), tt.ravel()
    diffusion = 2*temperature/(drag*length**2)
    advection = (extension*radius*np.cos(2*angle))*gr + (rotation-extension*np.sin(2*angle))*radius*gt
    a = -diffusion*((gr*w)@gr.T+(gt*w)@gt.T)+(advection*w)@v.T
    coefficients = np.r_[1., solve(a[1:, 1:], -a[1:, 0])]
    f = coefficients@v
    prob = w*f
    cx = length**2*np.dot(prob, radius**2*np.cos(angle)**2)
    cy = length**2*np.dot(prob, radius**2*np.sin(angle)**2)
    cxy = length**2*np.dot(prob, radius**2*np.cos(angle)*np.sin(angle))
    c = np.array([[cx,cxy],[cxy,cy]])
    stress = drag/2*(extension*np.trace(c)-2*rotation*cxy)
    force = length**2*np.dot(prob, radius**2*np.cos(2*angle)/(1-radius**2))
    return float(stress), c, {'minimum_relative_density':float(f.min()),
        'negative_probability_mass':float(-np.minimum(prob,0).sum()),
        'normalization_error':float(abs(prob.sum()-1)),
        'weak_residual':float(np.max(abs(a@coefficients))),
        'force_moment_error':float(abs(force-stress))}


def bernoulli(x):
    x=np.asarray(x)
    return np.divide(x, np.expm1(x), out=np.ones_like(x), where=abs(x)>1e-10)


def finite_volume(extension, rotation, temperature, drag, length, nr=80, nt=128):
    """Positive conservative polar SG balance, using actual probability masses."""
    dr=length/nr; dt=2*np.pi/nt
    r=(np.arange(nr)+.5)*dr; theta=(np.arange(nt)+.5)*dt
    volume=np.repeat(r*dr*dt,nt)
    ids=np.arange(nr*nt).reshape(nr,nt)
    diffusion=2*temperature/drag
    left=ids[:-1].ravel(); right=ids[1:].ravel()
    edge=diffusion*(r[:-1]+dr/2)*dt/dr
    pe=extension*(r[1:,None]**2-r[:-1,None]**2)*np.cos(2*theta)/(2*diffusion)
    pe+=length**2/(2*temperature)*np.log((1-r[1:,None]**2/length**2)/(1-r[:-1,None]**2/length**2))
    ab=np.broadcast_to(edge[:,None],pe.shape).ravel()*bernoulli(-pe.ravel())/volume[left]
    ba=np.broadcast_to(edge[:,None],pe.shape).ravel()*bernoulli(pe.ravel())/volume[right]
    l2=ids.ravel();r2=np.roll(ids,-1,axis=1).ravel()
    edge2=diffusion*dr/(r[:,None]*dt)
    pe2=r[:,None]**2/diffusion*(rotation*dt+extension/2*(np.cos(2*(theta+dt))-np.cos(2*theta)))
    ab2=(edge2*bernoulli(-pe2)).ravel()/volume[l2]
    ba2=(edge2*bernoulli(pe2)).ravel()/volume[r2]
    left=np.r_[left,l2];right=np.r_[right,r2];ab=np.r_[ab,ab2];ba=np.r_[ba,ba2]
    a=coo_matrix((np.r_[-ab,ab,ba,-ba],(np.r_[left,right,left,right],np.r_[left,left,right,right])),shape=(nr*nt,nr*nt)).tocsc()
    fixed=a[1:,1:];rhs=-a[1:,0].toarray().ravel()
    p=np.r_[1.,spsolve(fixed,rhs)];p/=p.sum()
    rr,tt=np.meshgrid(r,theta,indexing='ij')
    force=np.dot(p,(rr**2*np.cos(2*tt)/(1-rr**2/length**2)).ravel())
    cx=np.dot(p,(rr**2*np.cos(tt)**2).ravel());cy=np.dot(p,(rr**2*np.sin(tt)**2).ravel())
    cxy=np.dot(p,(rr**2*np.cos(tt)*np.sin(tt)).ravel());c=np.array([[cx,cxy],[cxy,cy]])
    stress=drag/2*(extension*np.trace(c)-2*rotation*cxy)
    current=ab*p[left]-ba*p[right]
    return float(force), c, {'moment_stress':float(stress),'force_moment_error':float(abs(force-stress)),
        'minimum_probability':float(p.min()),'conservation':float(np.max(abs(a@p))),
        'current_l1':float(np.sum(abs(current)))}


if __name__=='__main__':
    cases=[(.5,.3,1.,4.1,3.),(.8,.8,1.,4.1,3.),(.8,1.1,.8,4.8,np.sqrt(6)),(.9,.5,1.2,3.2,np.sqrt(12)),(.5,-.8,1.,4.1,3.)]
    out=[];start=time.perf_counter()
    for case in cases:
        exact,c,checks=spectral(*case)
        ref,rc,refchecks=finite_volume(*case)
        source,sc,f=peterlin(*case)
        refined,_,_=spectral(*case,degree=24)
        record={'case':list(case),'oracle':exact,'reference':ref,'source':source,'source_relative_error':abs(source/exact-1),
            'reference_error':abs(ref-exact),'spectral_refinement':abs(refined-exact),'checks':checks,'reference_checks':refchecks}
        out.append(record);print(json.dumps(record),flush=True)
    report={'status':'prototype','cases':out,'elapsed_seconds':time.perf_counter()-start}
    Path(__file__).with_name('screen.json').write_text(json.dumps(report,indent=2)+'\n')
