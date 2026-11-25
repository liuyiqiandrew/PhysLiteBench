"""Independent Chebyshev collocation of the continuous rotor jump generator."""
import numpy as np
from scipy.linalg import solve,eigvals
from numpy.polynomial.legendre import leggauss
from functools import lru_cache

def interpolation(nodes,weights,targets):
 delta=targets[:,None]-nodes[None,:]
 near=abs(delta)<1e-13
 safe=np.where(near,1.,delta)
 value=weights[None,:]/safe
 value/=value.sum(axis=1)[:,None]
 for i in np.flatnonzero(near.any(axis=1)):
  value[i]=0;value[i,np.argmax(near[i])]=1
 return value

@lru_cache(2048)
def state(gamma=.67,ta=1.,tb=1.,omega=.8,flip=1.,eta=1.,points=121,extent=9.,quadrature=140):
 width=extent*np.sqrt(max(ta,tb))+abs(omega)
 x=width*np.cos(np.arange(points)*np.pi/(points-1))
 w=(-1.)**np.arange(points);w[[0,-1]]*=.5
 delta=x[:,None]-x[None,:]
 D=(w[None,:]/w[:,None])/np.where(np.eye(points,dtype=bool),1.,delta)
 np.fill_diagonal(D,0);np.fill_diagonal(D,-D.sum(axis=1))
 # Enforce reflecting backward-Neumann boundaries by eliminating endpoint
 # values. Interior test functions remain freely parameterized.
 E=np.zeros((points,points-2));E[1:-1]=np.eye(points-2)
 E[[0,-1]]=solve(D[np.ix_([0,points-1],[0,points-1])],-D[[0,-1],1:-1])
 L=(-gamma*x[:,None]*D+gamma*ta*(D@D))@E
 nodes,weights=leggauss(quadrature);blocks=[];marks=[];squared=[];motors=[];size=points-2
 for sign in [1.,-1.]:
  u=sign*omega;J=np.zeros((size,size));A=np.zeros_like(J);b=np.zeros(size);motor=np.zeros(size)
  for i,v in enumerate(x[1:-1]):
   for lo,hi in [(-width,v),(v,width)]:
    g=(lo+hi)/2+(hi-lo)*nodes/2
    phi=np.exp(-(g-u)**2/(2*tb))/np.sqrt(2*np.pi*tb)
    rates=3.6*((1-eta)+eta*abs(v-g))*phi*weights*(hi-lo)/2
    mat=interpolation(x,w,g)@E
    q=.5*((v-u)**2-(g-u)**2)
    J[i]+=rates@mat;J[i,i]-=rates.sum()
    A[i]+=(rates*q)@mat;b[i]+=rates@q**2;motor[i]+=rates@(u*(g-v))
  blocks.append(L[1:-1]+J);marks.append(A);squared.append(b);motors.append(motor)
 I=np.eye(size)
 B=np.block([[blocks[0]-flip*I,flip*I],[flip*I,blocks[1]-flip*I]])
 F=B.T.copy();F[0]=1;rhs=np.zeros(2*size);rhs[0]=1
 prob=solve(F,rhs)
 a=np.r_[marks[0].sum(axis=1),marks[1].sum(axis=1)];b=np.r_[squared[0],squared[1]]
 mean=prob@a
 P=B.copy();pivot=int(np.argmax(abs(prob)));P[pivot]=prob;rhs=mean-a;rhs[pivot]=0
 phi=solve(P,rhs)
 correction=np.r_[marks[0]@phi[:size],marks[1]@phi[size:]]
 noise=prob@(b+2*correction)
 return {'mean':float(mean),'noise':float(noise),'shot':float(prob@b),'generator_constant_error':float(max(abs(B@np.ones(2*size)))),
         'stationary_residual':float(max(abs(B.T@prob))),'poisson_residual':float(max(abs(B@phi-(mean-a)))),
         'motor':float(prob@np.r_[motors[0],motors[1]]),'velocity_square':float(prob@np.tile(x[1:-1]**2,2)),
         'conditional_mass_error':float(abs(prob[:size].sum()-.5))}


TRUE_PARAMETER=.67
INPUT_FIELDS=('angular_speed','flip_rate','temperature_a','temperature_b','flux_fraction')


def experiment(omega=0.,flip=1.,ta=1.,tb=1.,eta=1.,statistic='noise'):
    return dict(angular_speed=omega,flip_rate=flip,temperature_a=ta,temperature_b=tb,flux_fraction=eta,statistic=statistic)


def hidden_inputs():
    return {'rotating_equal_temperature_bias': [{'angular_speed': 0.6, 'flip_rate': 0.8, 'temperature_a': 1.0, 'temperature_b': 1.0, 'flux_fraction': 1.0, 'heat_bias': 0.35, 'statistic': 'scgf'}, {'angular_speed': -0.65, 'flip_rate': 1.0, 'temperature_a': 1.0, 'temperature_b': 1.0, 'flux_fraction': 1.0, 'heat_bias': 0.35, 'statistic': 'scgf'}, {'angular_speed': 0.6, 'flip_rate': 1.5, 'temperature_a': 1.0, 'temperature_b': 1.0, 'flux_fraction': 1.0, 'heat_bias': 0.35, 'statistic': 'scgf'}, {'angular_speed': -0.65, 'flip_rate': 2.0, 'temperature_a': 1.0, 'temperature_b': 1.0, 'flux_fraction': 1.0, 'heat_bias': 0.35, 'statistic': 'scgf'}], 'rotating_cold_a_positive_bias': [{'angular_speed': 0.65, 'flip_rate': 0.8, 'temperature_a': 0.6, 'temperature_b': 1.4, 'flux_fraction': 1.0, 'heat_bias': 0.35, 'statistic': 'scgf'}, {'angular_speed': -0.65, 'flip_rate': 1.0, 'temperature_a': 0.6, 'temperature_b': 1.4, 'flux_fraction': 1.0, 'heat_bias': 0.35, 'statistic': 'scgf'}, {'angular_speed': 0.6, 'flip_rate': 1.5, 'temperature_a': 0.6, 'temperature_b': 1.4, 'flux_fraction': 1.0, 'heat_bias': 0.35, 'statistic': 'scgf'}, {'angular_speed': -0.6, 'flip_rate': 2.0, 'temperature_a': 0.6, 'temperature_b': 1.4, 'flux_fraction': 1.0, 'heat_bias': 0.35, 'statistic': 'scgf'}], 'rotating_hot_a_negative_bias': [{'angular_speed': 0.5, 'flip_rate': 0.8, 'temperature_a': 1.6, 'temperature_b': 0.6, 'flux_fraction': 1.0, 'heat_bias': -0.35, 'statistic': 'scgf'}, {'angular_speed': -0.65, 'flip_rate': 1.0, 'temperature_a': 1.4, 'temperature_b': 0.7, 'flux_fraction': 1.0, 'heat_bias': -0.35, 'statistic': 'scgf'}, {'angular_speed': 0.65, 'flip_rate': 1.5, 'temperature_a': 1.6, 'temperature_b': 0.8, 'flux_fraction': 1.0, 'heat_bias': -0.35, 'statistic': 'scgf'}, {'angular_speed': -0.5, 'flip_rate': 2.0, 'temperature_a': 1.5, 'temperature_b': 0.6, 'flux_fraction': 1.0, 'heat_bias': -0.35, 'statistic': 'scgf'}], 'stationary_statistics_anchor': [{'angular_speed': 0.0, 'flip_rate': 0.8, 'temperature_a': 1.4, 'temperature_b': 0.8, 'flux_fraction': 0.0, 'statistic': 'mean'}, {'angular_speed': 0.5, 'flip_rate': 1.0, 'temperature_a': 1.0, 'temperature_b': 1.0, 'flux_fraction': 1.0, 'statistic': 'noise'}, {'angular_speed': -0.65, 'flip_rate': 2.0, 'temperature_a': 0.8, 'temperature_b': 1.2, 'flux_fraction': 1.0, 'statistic': 'noise'}, {'angular_speed': 0.4, 'flip_rate': 1.5, 'temperature_a': 1.3, 'temperature_b': 0.9, 'flux_fraction': 0.0, 'statistic': 'mean'}]}


def predict(experiments,drag=TRUE_PARAMETER,points=121,extent=9.,quadrature=140):
    values=[]
    for e in experiments:
        statistic=e.get("statistic","mean")
        if statistic=="scgf":
            value=0. if e.get("heat_bias",0.)==0. else scgf(float(e["heat_bias"]),e["temperature_a"],e["temperature_b"],e["angular_speed"],e["flux_fraction"],e["flip_rate"],float(drag),points,extent,quadrature)["value"]
        else:
            value=state(float(drag),e["temperature_a"],e["temperature_b"],e["angular_speed"],e["flip_rate"],e["flux_fraction"],points,extent,quadrature)[statistic]
        values.append(value)
    return np.array(values)


def calibration_inputs():
    settings=[(ta,tb,flip) for ta,tb in [(1.5,.7),(.7,1.3),(1.4,.8),(.8,1.2)] for flip in [.8,1.2,2.]]
    return [dict(angular_speed=0.,flip_rate=flip,temperature_a=ta,temperature_b=tb,flux_fraction=0.) for _ in range(24) for ta,tb,flip in settings]


def heat_interpolation(nodes, weights, targets):
    delta = targets[:, None] - nodes[None, :]
    near = abs(delta) < 1e-13
    value = weights[None, :] / np.where(near, 1., delta)
    value /= value.sum(axis=1)[:, None]
    for i in np.flatnonzero(near.any(axis=1)):
        value[i] = 0.
        value[i, np.argmax(near[i])] = 1.
    return value


@lru_cache(2048)
def scgf(chi, ta, tb, omega=0., eta=0., flip=1., gamma=.67,
         points=121, extent=9., quadrature=140, gauge='piecewise',
         boundary='reflecting'):
    if gauge == 'piecewise':
        q = min(chi, 0.)
    elif gauge == 'full':
        q = chi
    elif gauge == 'none':
        q = 0.
    else:
        q = float(gauge)
    width = extent * np.sqrt(max(ta, tb)) + abs(omega)
    x = width * np.cos(np.arange(points) * np.pi / (points - 1))
    barycentric = (-1.) ** np.arange(points)
    barycentric[[0, -1]] *= .5
    delta = x[:, None] - x[None, :]
    derivative = (barycentric[None, :] / barycentric[:, None]) / np.where(
        np.eye(points, dtype=bool), 1., delta)
    np.fill_diagonal(derivative, 0.)
    np.fill_diagonal(derivative, -derivative.sum(axis=1))
    second = derivative @ derivative
    nodes, weights = leggauss(quadrature)
    size = points - 2
    extensions = []
    for sign in (1., -1.):
        u = sign * omega
        extension = np.zeros((points, size))
        extension[1:-1] = np.eye(size)
        rows = derivative[[0, -1]].copy()
        if boundary == 'reflecting':
            # f_old = exp[-q(v-u)^2/2] f_new. Original f_old'=0
            # becomes f_new' - q(v-u) f_new=0 at either cutoff.
            rows[0, 0] -= q * (x[0] - u)
            rows[1, -1] -= q * (x[-1] - u)
        elif boundary != 'gauge_neumann':
            raise ValueError('unknown boundary')
        extension[[0, -1]] = solve(rows[:, [0, points-1]], -rows[:, 1:-1])
        extensions.append(extension)
    blocks = []
    off = []
    for index, sign in enumerate((1., -1.)):
        u = sign * omega
        potential = gamma*q*x*(x-u) + gamma*ta*(q*q*(x-u)**2-q)
        drift = -gamma*x - 2*gamma*ta*q*(x-u)
        continuous = gamma*ta*second + drift[:, None]*derivative + np.diag(potential)
        block = (continuous @ extensions[index])[1:-1]
        jump = np.zeros((size, size))
        for i, v in enumerate(x[1:-1]):
            for lo, hi in ((-width, v), (v, width)):
                gas = (lo+hi)/2 + (hi-lo)*nodes/2
                density = np.exp(-(gas-u)**2/(2*tb))/np.sqrt(2*np.pi*tb)
                rates = 3.6*((1-eta)+eta*abs(v-gas))*density*weights*(hi-lo)/2
                heat = .5*((v-u)**2-(gas-u)**2)
                evaluation = heat_interpolation(x, barycentric, gas) @ extensions[index]
                jump[i] += (rates*np.exp(-(chi-q)*heat)) @ evaluation
                jump[i, i] -= rates.sum()
        blocks.append(block+jump-flip*np.eye(size))
        off.append(flip*np.exp(-2*q*sign*omega*x[1:-1])[:, None]
                   * extensions[1-index][1:-1])
    generator = np.block([[blocks[0], off[0]], [off[1], blocks[1]]])
    values = eigvals(generator, check_finite=False)
    ordered = values[np.argsort(-values.real)]
    principal = ordered[0]
    return dict(value=float(principal.real), imaginary=float(abs(principal.imag)),
                spectral_gap=float(principal.real-ordered[1].real),
                gauge=float(q), points=points, extent=extent,
                quadrature=quadrature, boundary=boundary)
