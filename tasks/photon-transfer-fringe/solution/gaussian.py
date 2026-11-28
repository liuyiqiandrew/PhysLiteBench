"""Passive-optical overlaps of a thermofield purification."""
import numpy as np

from optics import Calibrated, mesh


def source_pairs(q):
    n, r, phase = np.array(q['source']).T
    size = len(n)
    thermal = np.sqrt(n / (1 + n))
    squeezed = -np.exp(1j*phase)*np.tanh(r)
    cross = thermal / np.cosh(r)
    pairs = np.block([[np.diag(squeezed), np.diag(cross)],
                      [np.diag(cross), np.diag(-squeezed.conj()*thermal**2)]])
    transform = np.eye(2*size, dtype=complex)
    transform[:size, :size] = mesh(size, q['prep'])
    return transform @ pairs @ transform.T


class Survey(Calibrated):
    coherent = True

    def reading(self, q):
        pairs = source_pairs(q)
        size = len(q['source'])
        mean, squeeze, _ = np.array(q['source']).T
        norm = np.prod(1/((1+mean)*np.cosh(squeeze)))
        before = mesh(size, q['before'])
        after = mesh(size, q['after'])
        bank = np.zeros(size)
        bank[q['bank']] = 1
        phase = np.arange(64)*2*np.pi/64
        rotation = np.exp(1j*phase[:, None]*np.r_[bank, np.zeros(size)])
        states = rotation[:, :, None]*pairs*rotation[:, None, :]

        if 'switch' in q:
            return switch_fringe(q, states, norm, bank, before, after, self.coherent)

        def overlap(operator):
            extension = np.eye(2*size, dtype=complex)
            extension[:size, :size] = operator
            propagated = extension @ states @ extension.T
            eigenvalues = np.linalg.eigvals(states.conj().transpose(0, 2, 1) @ propagated)
            return norm * np.mean(np.exp(-.5*np.log1p(-eigenvalues).sum(axis=-1)))

        d = np.exp(1j*q['lambda']*bank)
        final = after.conj().T @ (d[:, None]*after)
        right = before*d.conj()
        gate = q['gate']
        if gate is None:
            value = overlap(before.conj().T @ final @ right)
        else:
            sign = gate['parity']
            parity = np.ones(size)
            parity[gate['rails']] = -1
            arms = [before, parity[:, None]*before]
            value = sum(sign**(i+j)*overlap(left.conj().T @ final @ (arm*d.conj()))
                        for i, left in enumerate(arms) for j, arm in enumerate(arms))
            retention = (1 + sign*overlap(before.conj().T @ arms[1]).real)/2
            value /= 4*retention
        return float((np.exp(-1j*q['phase'])*value).real)


def click(setting, basis):
    i, j, theta, phase, mix, pump = setting
    mode = np.cos(theta)*basis[i] + np.exp(1j*phase)*np.sin(theta)*basis[j]
    return np.cos(mix)*mode, np.exp(1j*pump)*np.sin(mix)*mode.conj()


def switch_fringe(q, states, norm, bank, before, after, coherent):
    setting = q['switch']
    size = len(bank)
    middle = mesh(size, setting['middle']) @ before
    full = after @ middle
    clicks = setting['clicks']
    orders = [(click(clicks[i], before), click(clicks[1-i], middle)) for i in (0,1)]
    amplitudes = [1, np.exp(1j*setting['angle'])]
    adjoints = states.conj().transpose(0,2,1)
    identity = np.eye(2*size)

    def characteristic(angle):
        d = np.exp(1j*angle*bank)
        final = full.conj().T @ (d[:,None]*full)
        extension = identity.astype(complex)
        extension[:size,:size] = final*d.conj()
        right = extension @ states @ extension.T
        eigen = np.linalg.eigvals(adjoints @ right)
        overlaps = norm * np.exp(-.5*np.log1p(-eigen).sum(axis=-1))
        gram = np.linalg.inv(identity - right @ adjoints)
        value = 0j
        for i,(first,second) in enumerate(orders):
            left_ops = [(v.conj(),u.conj()) for u,v in (first,second)]
            for j,(first,second) in enumerate(orders):
                if i != j and not coherent:
                    continue
                right_ops = [(u @ final.conj().T,v @ final.T) for u,v in (second,first)]
                u,v = [np.pad(np.asarray(rows),((0,0),(0,size))) for rows in zip(*(left_ops+right_ops))]
                pairs = (u + v @ adjoints) @ gram @ (right @ u.T + v.T)
                wick = (pairs[:,0,1]*pairs[:,2,3] + pairs[:,0,2]*pairs[:,1,3]
                        + pairs[:,0,3]*pairs[:,1,2])
                value += amplitudes[i].conjugate()*amplitudes[j]*np.mean(overlaps*wick)
        return value/4

    ratio = characteristic(q['lambda']) / characteristic(0.).real
    return float((np.exp(-1j*q['phase'])*ratio).real)
