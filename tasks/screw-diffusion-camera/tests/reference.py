"""Approximate the generator by advancing complete finite screws at the material origin."""
from functools import lru_cache

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import special
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply
from scipy.spatial.transform import Rotation


TRUE_PARAMETER = .9138
rates = [.55,.82,1.12]
screws = np.array([[.27,.46,-.18],[.04,-.23,.33],[-.38,.17,.29]])


def harmonics(points, order):
    longitude = np.arctan2(points[:,1],points[:,0])
    latitude = np.arccos(np.clip(points[:,2],-1.,1.))
    labels = [(l,m) for l in range(order+1) for m in range(-l,l+1)]
    if hasattr(special, "sph_harm_y"):
        return np.column_stack([special.sph_harm_y(l,m,latitude,longitude) for l,m in labels])
    return np.column_stack([special.sph_harm(m,l,longitude,latitude) for l,m in labels])


@lru_cache(maxsize=4)
def grid(order, step):
    nz = order+7
    nphi = 2*nz+1
    z,w = leggauss(nz)
    longitude = np.tile(2*np.pi*np.arange(nphi)/nphi, nz)
    z = np.repeat(z,nphi)
    xy = np.sqrt(1-z*z)
    u = np.column_stack([xy*np.cos(longitude),xy*np.sin(longitude),z])
    weight = np.repeat(w*2*np.pi/nphi,nphi)
    y = harmonics(u,order)
    basis = []
    for axis,screw in zip(np.eye(3),screws):
        parallel = (axis@screw)*axis
        pair = []
        for h in (step,-step):
            end = u @ Rotation.from_rotvec(h*axis).as_matrix()
            displacement = h*parallel+np.sin(h)*(screw-parallel)+(1-np.cos(h))*np.cross(axis,screw)
            pair.append((harmonics(end,order),u@displacement))
        basis.append(pair)
    return u,weight,y,y.conj().T*weight,basis


@lru_cache(maxsize=128)
def generator(order, step, k, hand):
    u,_,y,adj,neighbors = grid(order,step)
    diff = -(k*k)*(u**2 @ [.18,.27,.36])[:,None]*y
    for r, shifted in zip(rates,neighbors):
        forward, backward = [np.exp(1j*k*hand*dist)[:,None]*basis for basis,dist in shifted]
        diff += r*(forward+backward-2*y)/(step**2)
    matrix = adj @ diff
    matrix[abs(matrix)<3e-9] = 0.
    return csr_matrix(matrix)


@lru_cache(maxsize=128)
def motor_cells(order, step, k, hand, bias, flux):
    u,_,_,adj,_=grid(order,step)
    base_rot=np.array([[1.,0.,0.],[0.,1.,0.]])
    base_vel=hand*np.array([[.3,1.1,-.6],[-.8,.2,1.2]])
    # Exponential tilting of the signed finite-state current. The principal
    # right eigenvector is constant. Its eigenvalue derivative fixes flux.
    plus,minus=1+bias,1-bias
    tilt=np.arcsinh(flux/(2*np.sqrt(plus*minus)))+.5*np.log(minus/plus)
    forward,backward=plus*np.exp(tilt),minus*np.exp(-tilt)
    Q=np.zeros((3,3))
    operators=[]
    for j in range(3):
        Q[j,(j+1)%3]=forward
        Q[j,(j-1)%3]=backward
        Q[j,j]=-forward-backward
        angle=2*np.pi*j/3
        combo=1.5*np.sqrt(2)*np.array([np.cos(angle),np.sin(angle)])
        omega=combo@base_rot
        vel=combo@base_vel
        norm=np.linalg.norm(omega)
        axis=omega/norm
        axial=(vel@axis)*axis
        motions=[]
        for delta in (step,-step):
            final=u@Rotation.from_rotvec(delta*omega).as_matrix()
            travel=delta*axial+np.sin(norm*delta)/norm*(vel-axial)
            travel+=(1-np.cos(norm*delta))/norm*np.cross(axis,vel)
            motions.append(np.exp(1j*k*(u@travel))[:,None]*harmonics(final,order))
        op=adj@((motions[0]-motions[1])/(2*step))
        op[abs(op)<3e-10]=0.
        operators.append(csr_matrix(op))
    pi=np.ones((3,3))/3
    green=np.linalg.inv(pi-Q)-pi
    return Q,green,operators




def measurement(q, mobility=TRUE_PARAMETER, order=9, step=.003):
    u,w,y,adj,_=grid(order,step)
    tag = u@q["tag"]
    gates = [(1+u@g)**2/4 for g in q["gates"]]
    def advance(coeff,k,t,interval,initial=False):
        matrix = generator(order,step,k,q["hand"])
        if 'motor' in q:
            bias=q['motor']['bias']
            flux=q['motor'].get('flux',[2*bias]*2)[interval]
            _,green,ops=motor_cells(order,step,k,q['hand'],bias,flux)
            matrix=matrix+sum(ops[i]@ops[j]*(green[i,j]/3) for i in range(3) for j in range(3))
        evolved=expm_multiply((mobility*t)*matrix,coeff)
        if initial:
            diff=green[q['pair'][0]]-green[q['pair'][1]]
            return sum(weight*(op@evolved) for weight,op in zip(diff,ops))
        return evolved
    k1,k2=q["wave"]
    t1,t2=q["times"]
    # All propagation follows the material origin. Tag offsets enter at the three exposures.
    v = advance(adj@np.exp(1j*k2*tag),k2,t2,1)
    middle=adj@(gates[1]*np.exp(1j*(k1-k2)*tag)*(y@v))
    v = advance(middle,k1,t1,0)
    num = (w*gates[0]*np.exp(-1j*k1*tag))@(y@v)
    norm = (w*gates[0])@(y@advance(adj@gates[1],0,t1,0))
    if "pair" in q:
        dn = (w*gates[0]*np.exp(-1j*k1*tag)) @ (y@advance(middle,k1,t1,0,True))
        dp = (w*gates[0]) @ (y@advance(adj@gates[1],0,t1,0,True))
        num = dn-num*dp/norm
    return float((np.exp(-1j*q["phase"])*num/norm).real)


def finite_motor(q, epsilon, mobility=TRUE_PARAMETER, order=6, step=.002):
    """Advance the original three-state rotor before taking its fast limit."""
    from scipy.linalg import expm, block_diag

    u, w, y, adj, _ = grid(order,step)
    n = y.shape[1]
    bias = q['motor']['bias']
    gates = [(1+u@a)**2/4 for a in q['gates']]
    tag = u@q['tag']
    k1,k2 = q['wave']
    t1,t2 = q['times']

    def evolve(values,k,t):
        Q,_,velocities = motor_cells(order,step,k,q['hand'],bias,2*bias)
        base = generator(order,step,k,q['hand'])
        # Rows index the starting motor state; the right edge is the future exposure.
        drift = block_diag(*[(base+v/epsilon).toarray() for v in velocities])
        op = drift+np.kron(Q/epsilon**2,np.eye(n))
        return (expm(t*mobility*op) @ values.reshape(-1)).reshape(3,n)

    right = evolve(np.tile(adj@np.exp(1j*k2*tag),(3,1)),k2,t2)
    middle = np.array([adj@(gates[1]*np.exp(1j*(k1-k2)*tag)*(y@r)) for r in right])
    v = evolve(middle,k1,t1)
    numerator = v @ ((w*gates[0]*np.exp(-1j*k1*tag))@y)
    denominator = evolve(np.tile(adj@gates[1],(3,1)),0,t1) @ ((w*gates[0])@y)
    fringe = (np.exp(-1j*q['phase'])*numerator/denominator).real
    return float((fringe[q['pair'][0]]-fringe[q['pair'][1]])/epsilon)
