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
