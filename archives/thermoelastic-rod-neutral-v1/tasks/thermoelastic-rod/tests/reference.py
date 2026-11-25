"""Independent cell entropy balances and quasistatic mechanical constraints."""
from functools import lru_cache
import numpy as np
from scipy.linalg import solve, eig

PARAMETER = 'conductivity'
TRUE_PARAMETER = 145.
BOUNDS = (80., 220.)


@lru_cache(32)
def entropy_operator(conductivity, contact, cells):
    length, area, temperature, young, expansion, heat, bath = .05, 1e-4, 300., 2e9, .002, 1e6, 6.
    dx = length/cells
    volume = area*dx
    expansion=.002*(1+.65*np.cos(2*np.pi*(np.arange(cells)+.5)/cells))
    # Unknowns are cell temperatures, strains, and uniform stress divided by E.
    material = np.zeros((2*cells+1, 2*cells+1))
    material[:cells, :cells] = np.eye(cells)*heat/temperature
    material[:cells, cells:2*cells] = np.eye(cells)*young*expansion
    material[cells:2*cells, :cells] = -np.eye(cells)*expansion
    material[cells:2*cells, cells:2*cells] = np.eye(cells)
    material[cells:2*cells, -1] = -1.
    material[-1, cells:2*cells] = 1./cells
    forcing = np.zeros((2*cells+1, cells))
    forcing[:cells] = np.eye(cells)/volume
    response = solve(material, forcing)
    # Conserved states are each cell's entropy increment and the bath's energy.
    temperatures = np.zeros((cells+1, cells+1))
    temperatures[:cells, :cells] = response[:cells]
    temperatures[-1, -1] = 1./bath
    stiffness = np.zeros((cells+1, cells+1))
    for i in range(cells-1):
        v = np.zeros(cells+1); v[i], v[i+1] = 1., -1.
        stiffness += conductivity*area/dx*np.outer(v, v)
    effective_contact = 0. if contact == 0 else 1./(1./contact+dx/(2*conductivity*area))
    v = np.zeros(cells+1); v[0], v[-1] = 1., -1.
    stiffness += effective_contact*np.outer(v, v)
    rates_matrix = -np.diag(np.r_[np.full(cells, 1./temperature), 1.])@stiffness@temperatures
    rates, vectors = eig(rates_matrix)
    assert np.max(abs(rates.imag)) < 1e-8
    return rates.real, temperatures@vectors.real, solve(vectors.real,np.eye(cells+1))


def predict(experiments, conductivity, cells=128):
    out = []
    x = (np.arange(cells)+.5)/cells
    for e in experiments:
        initial_t = e['mean']+e['first']*np.cos(np.pi*x)+e['second']*np.cos(2*np.pi*x)
        expansion=.002*(1+.65*np.cos(2*np.pi*x))
        strain=expansion*initial_t-np.mean(expansion*initial_t)
        entropy = (.05*1e-4/cells)*(1e6/300*initial_t+2e9*expansion*strain)
        initial = np.r_[entropy, 6.*e['bath_initial']]
        rates,temperature_modes,inverse_modes=entropy_operator(conductivity,e['contact'],cells)
        state=temperature_modes@(np.exp(rates*e['time'])*(inverse_modes@initial))
        values = {'mean': state[:-1].mean(), 'bath': state[-1],
                  'first': state[:-1]@np.cos(np.pi*x)/cells,
                  'second': state[:-1]@np.cos(2*np.pi*x)/cells}
        out.append(values[e['observable']])
    return np.array(out)


def calibration_inputs():
    return [dict(mean=0.,first=.5,second=0.,bath_initial=0.,contact=0.,
                 time=float(t),observable='first') for t in np.geomspace(.15,40.,80)]


def hidden_inputs():
    result={f'contact_{h}':[dict(mean=a,first=b,second=c,bath_initial=bath,
                                contact=h,time=float(t),observable=name)
            for name in ['mean','second','bath'] for t in np.geomspace(.4,65.,18)]
            for h,a,b,c,bath in [(.65,-.2,.3,.1,.6),(1.1,.3,-.3,.15,-.7)]}
    result['insulated_even']=[dict(mean=0.,first=0.,second=.75,bath_initial=0.,
                                   contact=0.,time=float(t),observable=name)
            for name in ['mean','second'] for t in np.geomspace(.5,65.,24)]
    return result
