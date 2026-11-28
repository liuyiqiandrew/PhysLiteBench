"""Fast motor response in coordinates attached to the observed tag."""
import numpy as np
from scipy.sparse import eye
from scipy.sparse.linalg import expm_multiply
from apparatus import ROT, TRANS, SCREWS
from passive import operators, project


def fringe(q, mobility, order=7, diffusion_only=False):
    if diffusion_only and "pair" in q:
        return 0.
    pos, spin = operators(order)
    count = (order+1)**2
    one = np.zeros(count,complex)
    one[0] = 1
    gates = []
    for axis in q["gates"]:
        factor = eye(pos[0].shape[0],format="csr")+project(pos,axis)
        gates.append((factor @ factor)[:count,:count]/4)
    tag = np.asarray(q["tag"])
    screw = q["hand"]*SCREWS

    def evolve(v,k,t,interval,initial_diff=False):
        G = -k*k*sum(d*(p@p) for d,p in zip(TRANS,pos))
        for j,r in enumerate(ROT):
            lever = np.cross(np.eye(3)[j],tag)
            lever = lever+screw[j]
            twist = spin[j] + (1j*k)*project(pos,lever)
            G += r*(twist@twist)
        if 'motor' in q:
            bias = q['motor']['bias']
            flux = q['motor'].get('flux',[2*bias]*2)[interval]
            # The product of the optimal positive and negative jump rates is
            # invariant under exponential tilting of their signed count.
            total = np.sqrt(flux**2+4*(1-bias*bias))
            decay = 1.5*total
            circulation = np.sqrt(3)*flux/2
            scale = decay**2+circulation**2
            a = spin[0]+1j*k*project(pos, q['hand']*np.array([.3,1.1,-.6])+np.cross([1,0,0],tag))
            b = spin[1]+1j*k*project(pos, q['hand']*np.array([-.8,.2,1.2])+np.cross([0,1,0],tag))
            G += 2.25*decay/scale*(a@a+b@b)
            if not diffusion_only:
                G += 2.25*circulation/scale*(a@b-b@a)
        result = expm_multiply((mobility*t)*G[:count,:count],v)
        if initial_diff:
            p,r = q['pair']
            theta = 2*np.pi*np.array([p,r])/3
            # Solve -Q C = V on the zero-mean sine/cosine pair.
            cos_diff, sin_diff = np.cos(theta[0])-np.cos(theta[1]), np.sin(theta[0])-np.sin(theta[1])
            ax = (decay*cos_diff-circulation*sin_diff)/scale
            bx = (circulation*cos_diff+decay*sin_diff)/scale
            correction = (1.5*np.sqrt(2))*(ax*a+bx*b)
            result = correction[:count,:count] @ result
        return result

    times = q["times"]
    first, second = q["wave"]
    right = evolve(one,second,times[1],1)
    signal = (gates[0] @ evolve(gates[1]@right,first,times[0],0))[0]
    accepted = (gates[0] @ evolve(gates[1]@one,0,times[0],0))[0].real
    if q.get("pair") is not None:
        ds = (gates[0] @ evolve(gates[1]@right,first,times[0],0,True))[0]
        da = (gates[0] @ evolve(gates[1]@one,0,times[0],0,True))[0].real
        signal = ds-signal*da/accepted
    return float((np.exp(-1j*q["phase"])*signal).real/accepted)
