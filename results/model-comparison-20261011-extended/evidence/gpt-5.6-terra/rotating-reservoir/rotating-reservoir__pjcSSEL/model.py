"""Spectral stationary predictions for the finite-contact calorimeter."""
from functools import lru_cache
import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.laguerre import laggauss
from scipy.linalg import solve
from scipy.optimize import minimize_scalar

RATE = 3.6
FIELDS = ('angular_speed', 'flip_rate', 'temperature_a', 'temperature_b',
          'flux_fraction', 'relaxation_rate')


def _h(x, d):
    """Normalized probabilists' Hermites."""
    x = np.asarray(x); a = np.empty((x.size, d + 1)); a[:, 0] = 1.
    if d: a[:, 1] = x
    for n in range(1, d):
        a[:, n + 1] = (x*a[:, n] - np.sqrt(n)*a[:, n-1])/np.sqrt(n+1)
    return a


@lru_cache(maxsize=128)
def _flux_matrix(d, var):
    """Weak matrix for |v-g|(swap-I), using radial Laguerre quadrature."""
    pairs = tuple((n, m) for n in range(d+1) for m in range(d+1-n))
    q = d + 4; xx, xw = hermgauss(q); yy, yw = laggauss(q)
    t = np.sqrt(2*var)*xx[None, :]
    bs = []; js = []; ws = []
    for sign in (-1., 1.):
        z = sign*np.sqrt(2*var*yy[:, None]); v = (t+z)/np.sqrt(2); g = (t-z)/np.sqrt(2)
        hv = _h((v/np.sqrt(var)).ravel(), d); hg = _h((g/np.sqrt(var)).ravel(), d)
        bs.append(np.column_stack([hv[:, n]*hg[:, m] for n, m in pairs]))
        js.append(np.column_stack([hg[:, n]*hv[:, m] for n, m in pairs]))
        # Gaussian measure, including |v-g|, after z^2/(2 var)=y.
        ws.append((np.sqrt(var)/np.pi*yw[:, None]*xw[None, :]).ravel())
    b = np.vstack(bs); return b.T @ (np.concatenate(ws)[:, None]*(np.vstack(js)-b))


def _operators(e, drag, d):
    ta, tb, om, lam, eta, flip = (float(e[x]) for x in
        ('temperature_a','temperature_b','angular_speed','relaxation_rate','flux_fraction','flip_rate'))
    var = max(ta, tb, .5)
    pairs = tuple((n,m) for n in range(d+1) for m in range(d+1-n)); ix = {x:i for i,x in enumerate(pairs)}
    n = len(pairs); b = np.zeros((n,n)); g = np.zeros_like(b); g2 = np.zeros_like(b); dg = np.zeros_like(b)
    for j, (a, c) in enumerate(pairs):
        b[j,j] -= drag*a + lam*c
        if a >= 2: b[ix[a-2,c],j] += drag*(ta/var-1)*np.sqrt(a*(a-1))
        if c >= 2: b[ix[a,c-2],j] += lam*(tb/var-1)*np.sqrt(c*(c-1))
        if c:
            dg[ix[a,c-1],j] = np.sqrt(c/var); g[ix[a,c-1],j] = np.sqrt(var*c)
        if (a,c+1) in ix: g[ix[a,c+1],j] = np.sqrt(var*(c+1))
        g2[j,j] = var*(2*c+1)
        if c >= 2: g2[ix[a,c-2],j] = var*np.sqrt(c*(c-1))
        if (a,c+2) in ix: g2[ix[a,c+2],j] = var*np.sqrt((c+1)*(c+2))
    swap = np.zeros_like(b)
    for j, (a,c) in enumerate(pairs): swap[ix[c,a],j] = 1
    b += RATE*(1-eta)*(swap-np.eye(n))
    if eta: b += RATE*eta*_flux_matrix(d,var)
    L = np.zeros((2*n,2*n)); L[:n,:n] = b; L[n:,n:] = b
    for k, sign in enumerate((1.,-1.)):
        lo = k*n; other = (1-k)*n
        for j,(a,c) in enumerate(pairs):
            if c: L[lo+ix[a,c-1],lo+j] += lam*sign*om*np.sqrt(c/var)
            L[lo+j,lo+j] -= flip; L[other+j,lo+j] += flip
    A1 = np.zeros_like(L); A2 = np.zeros_like(L); eye = np.eye(n)
    for k, sign in enumerate((1.,-1.)):
        x = g-sign*om*eye
        # Derivatives of the E[exp(-k Q)] tilted generator.
        q1 = 2*lam*tb*(x@dg)-lam*(g2-2*sign*om*g+(om*om-tb)*eye)
        q2 = 2*lam*tb*(g2-2*sign*om*g+om*om*eye)
        A1[k*n:(k+1)*n,k*n:(k+1)*n] = q1; A2[k*n:(k+1)*n,k*n:(k+1)*n] = q2
    one = np.zeros(2*n); one[0] = one[n] = 1
    station = L.T.copy(); station[0] = one; rhs = np.zeros(2*n); rhs[0] = 1
    p = solve(station, rhs, check_finite=False)
    return L, A1, A2, p, one


def _stats(e, drag, d):
    L, A1, A2, p, one = _operators(e,drag,d)
    # The documented gated approximation mixes phase independently and fast:
    # E(w)=1, while the local quadratic variation has E(w^2).
    eps = float(e.get('detector_contrast',0.)); A2 *= 1+eps*eps/2
    l1 = float(p@(A1@one)); sys = L.copy(); sys[0] = p
    def inv(r):
        r = np.asarray(r,float).copy(); r[0] = 0
        return solve(sys,r,check_finite=False)
    r1 = inv(l1*one-A1@one)
    l2 = float(p@(A2@one+2*A1@r1))
    r2 = inv(l2*one+2*l1*r1-2*A1@r1-A2@one)
    l3 = float(p@(3*A1@r2+3*A2@r1))
    return {'mean':-l1, 'noise':l2, 'third_cumulant':-l3}


def calibration_mean(e, drag):
    lam = float(e['relaxation_rate'])
    conductance = drag*lam*RATE/(2*drag*lam+RATE*(drag+lam))
    return conductance*(float(e['temperature_a'])-float(e['temperature_b']))


def fit_drag(records):
    if not records: raise ValueError('at least one calibration record is required')
    vals = np.array([float(r['value']) for r in records]); sig = np.array([float(r['sigma']) for r in records])
    if np.any(sig <= 0): raise ValueError('calibration uncertainties must be positive')
    ins = [r['input'] for r in records]
    def loss(x): return np.sum(((np.array([calibration_mean(e,x) for e in ins])-vals)/sig)**2)
    return float(minimize_scalar(loss,bounds=(.4,1.1),method='bounded',options={'xatol':1e-12}).x)


def _degree(e): return 18 if float(e['flux_fraction']) == 0 else 30


@lru_cache(maxsize=1024)
def _cached(values, contrast, drag):
    e = dict(zip(FIELDS,values)); e['detector_contrast'] = contrast
    return _stats(e,drag,_degree(e))


def predict_at(experiments,drag):
    if drag is None: raise RuntimeError('call fit(records) before predict(experiments)')
    out=[]
    for e in experiments:
        stat=e.get('statistic','mean')
        if stat not in ('mean','noise','third_cumulant'): raise ValueError('unknown calorimeter statistic')
        contrast=float(e.get('detector_contrast',0.))
        if contrast and float(e['angular_speed']) != 0: raise ValueError('Angularly gated preparations require zero angular_speed')
        values=tuple(float(e[k]) for k in FIELDS)
        if stat=='mean' and contrast==0 and values[0]==0 and values[4]==0: out.append(calibration_mean(e,drag))
        else: out.append(_cached(values,contrast,float(drag))[stat])
    return np.asarray(out,dtype=float)


class Model:
    def __init__(self): self.drag=None
    def fit(self,records): self.drag=fit_drag(records); return self
    def predict(self,experiments): return predict_at(experiments,self.drag)
