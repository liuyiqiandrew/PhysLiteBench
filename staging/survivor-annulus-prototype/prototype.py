"""Prototype only: survivor transport in an annular Couette/Poiseuille flow."""
from pathlib import Path
import argparse
import datetime
import hashlib
import itertools
import json
import time

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigs


HERE = Path(__file__).resolve().parent


def bernoulli(x):
    x = np.asarray(x, dtype=float)
    return np.divide(x, np.expm1(x), out=np.ones_like(x), where=np.abs(x) > 1e-12)


def operator(diffusivity, inner, outer, rotation, capture_low, capture_high,
             nr=24, nt=48):
    """Conservative annular generator in volume-symmetric coordinates.

    Inner cylinder rotates, reflecting tracers. Outer cylinder is stationary
    with positive kappa(theta)=low+(high-low)*(1+cos(theta))/2.
    The returned operator acts on sqrt(volume) times forward concentration.
    """
    edges = np.linspace(inner, outer, nr+1)
    r = (edges[:-1]+edges[1:])/2
    dr = (outer-inner)/nr
    dt = 2*np.pi/nt
    theta = (np.arange(nt)+.5)*dt
    vr = (edges[1:]**2-edges[:-1]**2)/2
    volume = np.repeat(vr*dt, nt)
    rr, cc, vv = [], [], []
    diagonal = np.zeros(nr*nt)

    def jump(i, j, rate):
        rr.append(i); cc.append(j); vv.append(rate)
        diagonal[i] -= rate

    for i in range(nr-1):
        conductance = diffusivity*edges[i+1]/dr
        for j in range(nt):
            jump(i*nt+j, (i+1)*nt+j, conductance/vr[i])
            jump((i+1)*nt+j, i*nt+j, conductance/vr[i+1])
    # Integrate angular diffusion and Couette flux over each radial cell.
    angular_diffusion = diffusivity*np.log(edges[1:]/edges[:-1])/vr
    factor = rotation*inner**2/(outer**2-inner**2)
    angular_drift = factor*(outer**2*np.log(edges[1:]/edges[:-1])-vr)/vr
    pe = angular_drift*dt/angular_diffusion
    forward = angular_diffusion/dt**2*bernoulli(-pe)
    backward = angular_diffusion/dt**2*bernoulli(pe)
    for i in range(nr):
        for j in range(nt):
            jump(i*nt+j, i*nt+(j+1)%nt, forward[i])
            jump(i*nt+j, i*nt+(j-1)%nt, backward[i])
    capture = capture_low+(capture_high-capture_low)*(1+np.cos(theta))/2
    diagonal[(nr-1)*nt:] -= outer*capture/(1+capture*dr/(2*diffusivity))/vr[-1]
    rr.extend(range(nr*nt)); cc.extend(range(nr*nt)); vv.extend(diagonal)
    backward_generator = coo_matrix((vv,(rr,cc)), shape=(nr*nt,nr*nt)).tocsr()
    sq = np.sqrt(volume)
    forward_symmetric = diags(1/sq) @ backward_generator.T @ diags(sq)
    # Ordinary annular pressure-driven profile, normalized to unit peak.
    coefficient = (outer**2-inner**2)/np.log(outer/inner)
    maximum_radius = np.sqrt(coefficient/2)
    def shape(x):
        return outer**2-x*x-coefficient*np.log(outer/x)
    velocity = np.repeat(shape(r)/shape(maximum_radius), nt)
    return forward_symmetric.tocsc(), volume, velocity


def principal(matrix, vector=False):
    values, vectors = eigs(matrix, k=1, sigma=0, which='LM', tol=1e-11)
    value = values[0]
    assert abs(value.imag) < 1e-8
    if not vector:
        return float(value.real)
    v = vectors[:,0].real
    if v.sum() < 0:
        v = -v
    assert v.min() > -1e-10
    residual = np.linalg.norm(matrix@v-value.real*v)/np.linalg.norm(v)
    return float(value.real), v, float(residual)


def moments(diffusivity, inner, outer, rotation, capture_low, capture_high,
            nr=24, nt=48, reference=False):
    a, volume, u = operator(diffusivity,inner,outer,rotation,capture_low,capture_high,nr,nt)
    value, right, residual = principal(a, True)
    left_value, left, left_residual = principal(a.T.tocsc(), True)
    occupation = left*right
    physical = float(occupation@u/occupation.sum())
    squared_right = float((right*right)@u/(right@right))
    endpoint = float((np.sqrt(volume)*right)@u/(np.sqrt(volume)@right))
    result = {'loss_rate':-value, 'physical_drift':physical,
              'source_drift':squared_right, 'endpoint_drift':endpoint,
              'fractional_error':abs(squared_right/physical-1),
              'eigenvalue_adjoint_error':abs(value-left_value),
              'residual':max(residual,left_residual)}
    if reference:
        def derivative(s):
            p = principal(a+diags(s*u+diffusivity*s*s))
            m = principal(a+diags(-s*u+diffusivity*s*s))
            return (p-m)/(2*s)
        d1,d2 = derivative(.04),derivative(.02)
        result['tilted_drift'] = (4*d2-d1)/3
        result['tilted_agreement'] = abs(result['tilted_drift']/physical-1)
    return result


def screen():
    rows=[]; start=time.monotonic()
    for inner,rotation,low,high in itertools.product(
            [.15,.35,.6], [0.,2.,8.,32.,128.], [.02], [2.,20.,200.]):
        controls=dict(diffusivity=1.,inner=inner,outer=1.,rotation=rotation,
                      capture_low=low,capture_high=high)
        rows.append({'controls':controls, **moments(**controls)})
    report={'status':'prototype_screen_only','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'development_timing':'Authorized and developed before any survivor-transport-r1 model outcome; separate same-family possible r2.',
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'grid':[24,48],'rows':rows,'seconds':time.monotonic()-start}
    (HERE/'initial-screen.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(sorted(rows,key=lambda x:x['fractional_error'],reverse=True)[:8],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--screen',action='store_true');args=p.parse_args()
    if args.screen:screen()
