"""Normally ordered Gaussian contractions in real quadratures."""
import numpy as np

TRUE_PARAMETER = .8436


def interferometer(n, rows):
    out = np.eye(n, dtype=complex)
    for i,j,t,p in rows:
        unit = np.eye(n, dtype=complex)
        unit[i,i] = unit[j,j] = np.cos(t)
        unit[j,i] = np.exp(1j*p)*np.sin(t)
        unit[i,j] = -unit[j,i].conjugate()
        out = unit @ out
    return out


def reference(q, points=112):
    n = len(q['source'])
    identity = np.eye(n)
    to_alpha = np.hstack((identity,1j*identity))/np.sqrt(2)
    v = np.zeros((2*n,2*n))
    for j,(mean,r,phase) in enumerate(q['source']):
        c,s = np.cos(phase/2),np.sin(phase/2)
        rotation = np.array([[c,-s],[s,c]])
        v[np.ix_([j,n+j],[j,n+j])] = (mean+.5) * (
            rotation @ np.diag(np.exp([-2*r,2*r])) @ rotation.T)
    prep,before,after = [interferometer(n,q[k]) for k in ('prep','before','after')]
    rotation = np.block([[prep.real,-prep.imag],[prep.imag,prep.real]])
    normal = rotation @ v @ rotation.T - .5*np.eye(2*n)
    bank = np.zeros(n)
    bank[q['bank']] = 1
    angles = (np.arange(points)+.5)*2*np.pi/points
    twirl = np.exp(1j*angles[:,None]*bank)
    if 'switch' in q:
        return switched(q, normal, before, after, bank, angles, to_alpha)

    def expect(matrix):
        rotated = twirl.conj()[:,:,None] * matrix * twirl[:,None,:]
        h = to_alpha.conj().T @ (rotated-identity) @ to_alpha
        eigen = np.linalg.eigvals(normal @ (h+h.transpose(0,2,1)))
        return np.mean(np.exp(-.5*np.log1p(-eigen).sum(axis=-1)))

    gate = q['gate']
    branches = [after @ before]
    weights = [1.]
    if gate:
        signs = np.ones(n)
        signs[gate['rails']] = -1
        branches.append(after @ np.diag(signs) @ before)
        weights = [.5, .5*gate['parity']]

    def characteristic(angle):
        d = np.diag(np.exp(1j*angle*bank))
        return sum(lw*rw*expect(l.conj().T @ d @ r @ d.conj())
                   for lw,l in zip(weights,branches) for rw,r in zip(weights,branches))

    probability = characteristic(0.).real if gate else 1.
    value = (np.exp(-1j*q['phase'])*characteristic(q['lambda'])).real / probability
    return float(value), float(probability)


def switched(q, covariance, before, after, bank, angles, to_alpha):
    n = len(bank)
    eye = np.eye(n)
    c = np.cos(angles[:,None] * bank)[:,None,:]*eye
    s = np.sin(angles[:,None] * bank)[:,None,:]*eye
    rot = np.concatenate((np.concatenate((c,-s),2),np.concatenate((s,c),2)),1)
    cov = rot @ covariance @ rot.transpose(0,2,1)
    data = q['switch']
    middle = interferometer(n,data['middle'])
    ops = []
    for i,j,t,p,b,f in data['clicks']:
        vector = np.cos(t)*eye[i] + np.exp(1j*p)*np.sin(t)*eye[j]
        ops.append((np.cos(b)*vector,np.sin(b)*np.exp(1j*f)*vector.conj()))
    incoming = [1.,np.exp(1j*data['angle'])]

    def trace(angle):
        d = np.diag(np.exp(1j*angle*bank))
        pulled = after.conj().T @ d @ after
        k = before.conj().T @ middle.conj().T @ pulled @ middle @ before @ d.conj()
        h = to_alpha.conj().T @ (k-eye) @ to_alpha
        product = cov @ (h+h.T)
        normalizer = np.exp(-.5*np.log1p(-np.linalg.eigvals(product)).sum(axis=1))
        tilted = np.linalg.solve(np.eye(2*n)-product,cov)
        answer = 0j
        for left in range(2):
            for right in range(2):
                def adj(op):
                    return op[1].conj(),op[0].conj()
                factors = [before.conj().T,adj(ops[left]),middle.conj().T,adj(ops[1-left]),
                           pulled,ops[1-right],middle,ops[right]]
                walk = eye
                creators,annihilators = [],[]
                for elem in factors:
                    if isinstance(elem,tuple):
                        annihilators.append(elem[0] @ walk.conj().T)
                        creators.append(elem[1] @ walk.T)
                    else:
                        walk = walk @ elem
                u,v = np.array(annihilators),np.array(creators)
                # Normal ordering on a coherent ket K|alpha> leaves vacuum contractions
                # u_i . v_j for i preceding j. The other terms come from the P average.
                source = u @ k @ to_alpha + v @ to_alpha.conj()
                pair = source @ tilted @ source.T + u @ v.T
                fourth = sum(pair[:,0,j]*pair[:,l,m] for j,l,m in ((1,2,3),(2,1,3),(3,1,2)))
                answer += incoming[left].conjugate()*incoming[right]*np.mean(normalizer*fourth)
        return answer/4
    retention = trace(0.).real
    return float((np.exp(-1j*q['phase'])*trace(q['lambda'])).real/retention),float(retention)
