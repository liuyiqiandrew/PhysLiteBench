"""Independent full-interval finite-difference Hamiltonian and boundary stress."""
from functools import lru_cache
import numpy as np
from scipy.linalg import eigh_tridiagonal

TRUE_PARAMETER=.63


def experiment(length, strength, temperature):
    return dict(length=length,strength=strength,temperature=temperature)


@lru_cache(maxsize=4096)
def modes(length, strength, mass, grid):
    spacing=length/(grid+1)
    kinetic=1/(2*mass*spacing**2)
    diagonal=np.full(grid,2*kinetic)
    diagonal[grid//2]+=strength/spacing
    offdiagonal=np.full(grid-1,-kinetic)
    energy,vectors=eigh_tridiagonal(diagonal,offdiagonal,
        select='i',select_range=(0,31))
    boundary=abs(vectors[-1])**2/(2*mass*spacing**3)
    return energy,boundary


def grid_force(e, mass, grid):
    energies,boundary=modes(e['length'],e['strength'],mass,grid)
    weights=np.exp(-(energies-energies[0])/e['temperature'])
    return float(weights@boundary/weights.sum())


def force(e, mass, grid=1023):
    coarse=grid_force(e,mass,(grid-1)//2)
    fine=grid_force(e,mass,grid)
    return (4*fine-coarse)/3


def predict(experiments, mass=TRUE_PARAMETER):
    return np.array([force(e,mass) for e in experiments])


def calibration_inputs():
    return [experiment(L,0.,T) for _ in range(12)
            for L in [.85,1.05,1.25,1.55] for T in [.2,.6,1.2]]


def hidden_inputs():
    return {
        'temperature_sweep':[experiment(1.7,3.4,T) for T in [.2,.6,1.,1.5]],
        'length_sweep':[experiment(L,5.72/L,.8) for L in [1.,1.2,1.4,1.7]],
        'strength_sweep':[experiment(1.4,g,.7) for g in [3.6,4.,4.4,4.8]],
        'free_anchors':[experiment(L,0.,T) for L,T in [(1.,.3),(1.4,1.4)]]
    }
