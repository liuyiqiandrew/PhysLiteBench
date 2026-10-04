import numpy as np

PARAMETER = 'rotation_gain'
TRUE_PARAMETER = 1.


def calibration_inputs():
    return [dict(lengths=[0.,float(length)],counts=[1,1]) for length in np.linspace(.03,1.,100)]


def hidden_inputs():
    return {
        'one_each':[dict(lengths=[0.,float(length),float(2*length)],counts=[1,1,1]) for length in np.linspace(.95,1.15,9)],
        'two_one':[dict(lengths=[.2,float(.2+length),float(.2+2*length)],counts=[2,1,0]) for length in np.linspace(.95,1.15,9)],
        'all_together':[dict(lengths=[0.,float(length),float(2*length)],counts=[3,0,0]) for length in np.linspace(.95,1.15,9)],
    }


def fock_distribution(unitary, polarizations):
    n = len(polarizations)
    internal_modes = polarizations.shape[1]
    amplitudes = {(0,)*(internal_modes*n):1.+0j}
    for incoming in range(n):
        updated = {}
        for occupation,amplitude in amplitudes.items():
            for output in range(n):
                for polarization in range(internal_modes):
                    mode = internal_modes*output+polarization
                    value = unitary[output,incoming]*polarizations[incoming,polarization]
                    target = list(occupation)
                    target[mode] += 1
                    target = tuple(target)
                    updated[target] = updated.get(target,0j)+amplitude*value*np.sqrt(target[mode])
        amplitudes = updated
    probabilities = {}
    for occupation,amplitude in amplitudes.items():
        counts = tuple(sum(occupation[internal_modes*k:internal_modes*(k+1)]) for k in range(n))
        probabilities[counts] = probabilities.get(counts,0.)+abs(amplitude)**2
    return probabilities


def predict(experiments, rotation_gain):
    out = []
    for e in experiments:
        n = len(e['lengths'])
        angles = rotation_gain*np.array(e['lengths'])
        polarizations = np.array([[np.cos(a),np.sin(a)] for a in angles])
        unitary = np.array([[np.exp(2j*np.pi*k*j/n)/np.sqrt(n) for j in range(n)] for k in range(n)])
        distribution = fock_distribution(unitary,polarizations)
        out.append(distribution[tuple(e['counts'])])
    return np.array(out)
