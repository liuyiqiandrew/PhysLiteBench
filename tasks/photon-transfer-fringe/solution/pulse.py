"""Finite two-trigger evolution of a Gaussian purification."""
import numpy as np
from scipy.linalg import expm

from gaussian import Survey, click, mesh, source_pairs


def strength_circuit(q, points=64, coherent=True):
    n = len(q['source'])
    core = 2*n
    size = core+2
    bank = np.zeros(n)
    bank[q['bank']] = 1
    mean, squeeze, _ = np.array(q['source']).T
    norm = np.prod(1/((1+mean)*np.cosh(squeeze)))
    gauge = np.exp(2j*np.pi*np.arange(points)[:,None]*np.r_[bank,np.zeros(n)]/points)
    initial = np.zeros((points,size,size),complex)
    initial[:,:core,:core] = gauge[:,:,None]*source_pairs(q)*gauge[:,None,:]
    spec = q['switch']
    middle, before, after = [mesh(n,rows) for rows in (spec['middle'],q['before'],q['after'])]
    eye = np.eye(size)
    pulses=[]
    for k,(row,duration) in enumerate(zip(spec['clicks'],q['pulse'])):
        u,v = click(row,np.eye(n))
        h = core+k
        H = np.zeros((size,size),complex)
        pair = H.copy()
        H[h,:n], H[:n,h] = u,u.conj()
        pair[h,:n] = pair[:n,h] = v
        transform = expm(duration*np.block([[-1j*H,-1j*pair],[1j*pair.conj(),1j*H.conj()]]))
        pulses.append((transform[:size,:size],transform[:size,size:]))

    def move(state, network):
        mat = eye.astype(complex)
        mat[:n,:n] = network
        return mat@state@mat.T

    def states(counting):
        out = []
        entrance = before*np.exp(-1j*counting*bank)
        exit_net = np.exp(1j*counting*bank)[:,None]*after
        for first,second in [(0,1),(1,0)]:
            state = move(initial,entrance)
            coeff = np.full(points,np.sqrt(norm),complex)
            for step,(x,y) in enumerate([pulses[first],pulses[second]]):
                divisor = y.conj()@state + x.conj()
                coeff *= np.exp(-.5*np.log(np.linalg.eigvals(divisor)).sum(axis=-1))
                state = (x@state+y)@np.linalg.inv(divisor)
                state = move(state,middle if step==0 else exit_net)
            out.append((state[:,:core,:core],state[:,core,core+1],state[:,core,:core],
                        state[:,core+1,:core],coeff))
        return out

    left = states(0.)
    amps = [1.,np.exp(1j*spec['angle'])]
    def characteristic(right):
        total = 0j
        for i,(bl,dl,la,lb,cl) in enumerate(left):
            adj = bl.conj().swapaxes(-1,-2)
            for j,(br,dr,ra,rb,cr) in enumerate(right):
                if i!=j and not coherent:
                    continue
                eig = np.linalg.eigvals(adj@br)
                ov = cl.conj()*cr*np.exp(-.5*np.log1p(-eig).sum(axis=-1))
                u = np.stack([la.conj(),lb.conj(),0*ra,0*rb],axis=1)
                v = np.stack([0*la,0*lb,ra,rb],axis=1)
                gram = np.linalg.inv(np.eye(core)-br@adj)
                pair = (u+v@adj)@gram@(br@u.swapaxes(-1,-2)+v.swapaxes(-1,-2))
                both = ((dl.conj()+pair[:,0,1])*(dr+pair[:,2,3])
                        +pair[:,0,2]*pair[:,1,3]+pair[:,0,3]*pair[:,1,2])
                total += np.conj(amps[i])*amps[j]*np.mean(both*ov)/4
        return total
    rate = characteristic(left).real
    fringe = (np.exp(-1j*q['phase'])*characteristic(states(q['lambda']))).real/rate
    return float(fringe),float(rate)


class PulseSurvey(Survey):
    pulse_coherence = True

    def reading(self,q):
        if 'pulse' in q:
            return strength_circuit(q,coherent=self.pulse_coherence)[0]
        return super().reading(q)
