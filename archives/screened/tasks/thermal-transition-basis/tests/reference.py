"""Independent energy-population master equation and equilibrium Gibbs weights."""
from functools import lru_cache
import numpy as np
from scipy.linalg import expm

TRUE_RATE=.009


@lru_cache(maxsize=64)
def population_model(coupling):
    energies=[];states=[]
    for indices,diagonal in [([0,3],1.35),([1,2],-.35)]:
        values,basis=np.linalg.eigh(np.array([[diagonal,coupling],[coupling,-diagonal]]))
        for value,column in zip(values,basis.T):
            state=np.zeros(4);state[indices]=column
            energies.append(value);states.append(state)
    order=np.argsort(energies);energies=np.array(energies)[order];states=np.array(states)[order].T
    generator=np.zeros((4,4))
    for source in range(4):
        for target in range(4):
            if source==target:continue
            lost=energies[source]-energies[target]
            factor=lost/(1-np.exp(-lost/.65))
            strength=sum(abs(states[:,target]@states[np.arange(4)^mask,source])**2 for mask in [1,2])
            generator[target,source]=factor*strength
        generator[source,source]=-generator[:,source].sum()
    observable=dict(excitation_1=np.sum(states[[0,1]]**2,axis=0),
                    excitation_2=np.sum(states[[0,2]]**2,axis=0),
                    xx_correlation=np.sum(states*states[np.arange(4)^3],axis=0))
    equilibrium=np.exp(-(energies-energies.min())/.65);equilibrium/=equilibrium.sum()
    return generator,observable,equilibrium,states,energies


def predict(experiments,rate=TRUE_RATE):
    out=[]
    for e in experiments:
        generator,observable,equilibrium,states,energies=population_model(e['coupling'])
        if e['protocol']=='stationary':probability=equilibrium
        else:probability=expm(rate*e['time']*generator)[:,e['initial_level']]
        out.append(float(observable[e['observable']]@probability))
    return np.array(out)


def experiment(coupling,observable,time=0.,level=3,protocol='transient'):
    return dict(coupling=float(coupling),initial_level=level,protocol=protocol,time=float(time),observable=observable)


def calibration_inputs():
    return [experiment(0.,observable,t) for observable in ['excitation_1','excitation_2'] for t in np.geomspace(2.,300.,50)]


def hidden_inputs():
    return dict(stationary_populations=[experiment(g,o,protocol='stationary') for g in [.25,.35,.45,.55,.65,.7] for o in ['excitation_1','excitation_2']],
                stationary_correlation=[experiment(g,'xx_correlation',protocol='stationary') for g in [.25,.35,.45,.55,.65,.7]],
                interacting_transients=[experiment(g,o,t,level) for g in [.35,.65] for level in [0,3] for t in [15.,50.,130.,300.] for o in ['excitation_1','excitation_2','xx_correlation']])
