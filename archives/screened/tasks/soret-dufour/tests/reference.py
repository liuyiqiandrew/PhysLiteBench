"""Conservative finite-volume heat and species currents from entropy forces."""
import numpy as np
from scipy.sparse import diags,bmat
from scipy.sparse.linalg import expm_multiply
from functools import lru_cache

PARAMETER = 'diffusivity'
TRUE_PARAMETER = .04


def experiment(time,mode=1,length=1.,composition_amplitude=.08,temperature_amplitude=0.,isothermal=False,observable='temperature_amplitude'):
    return dict(time=float(time),mode=int(mode),length=float(length),composition_amplitude=float(composition_amplitude),
                temperature_amplitude=float(temperature_amplitude),isothermal=bool(isothermal),observable=observable)


def calibration_inputs():
    return [experiment(t,mode=mode,composition_amplitude=amp,isothermal=True,observable='composition_amplitude')
            for mode,amp in [(1,.08),(2,.06),(3,-.07),(1,-.05)] for t in np.linspace(.1,5.,25)]


def hidden_inputs():
    return {'salt_driven':[experiment(t) for t in np.linspace(.2,8.,16)],
            'joint_preparation':[experiment(t,composition_amplitude=-.07,temperature_amplitude=.15) for t in np.linspace(.2,8.,16)],
            'shorter_wavelength':[experiment(t,mode=2,length=1.3,composition_amplitude=.06) for t in np.linspace(.2,5.,16)]}


@lru_cache(2000)
def profile(time,mode,length,ca,ta,isothermal,d,cells):
    dx = length*1e-3/cells
    x = (np.arange(cells)+.5)/cells
    initial_mode = np.cos(mode*np.pi*x)
    lap = diags([np.ones(cells-1),np.r_[-1.,np.full(cells-2,-2.),-1.],np.ones(cells-1)],[-1,0,1],format='csr')/dx**2
    d_si = d*1e-6
    if isothermal:
        # This cosine is an exact eigenvector of the cell-centered FV operator.
        rate = 4*np.sin(mode*np.pi/(2*cells))**2/dx**2
        return np.zeros(cells),ca*np.exp(-d_si*rate*time)*initial_mode,initial_mode
    n,T,C,kappa,R,s = 1000.,300.,2e6,.2,8.314462618,.05
    # Forces are grad(1/T)=-grad(deltaT)/T0^2 and
    # -grad[(muA-muB)/T]=-R/[c0(1-c0)]*grad(delta c).
    L = np.array([[kappa*T*T,n*d_si*.25*s*T*T],
                  [n*d_si*.25*s*T*T,n*d_si*.25/R]])
    force = np.diag([1/(T*T),R/.25])
    storage = np.diag([1/C,1/n])
    coefficients = storage@L@force
    # The two species/heat amplitudes form an invariant FV cosine subspace.
    # Diagonalizing the resulting small conservative flux operator avoids a
    # long explicit integration while retaining independent FV eigenvalues.
    rate = 4*np.sin(mode*np.pi/(2*cells))**2/dx**2
    from scipy.linalg import expm
    state = expm(-time*rate*coefficients)@np.array([ta,ca])
    return state[0]*initial_mode,state[1]*initial_mode,initial_mode


def predict(experiments,diffusivity,cells=384):
    out = []
    for e in experiments:
        temp,comp,mode = profile(e['time'],e['mode'],e['length'],e['composition_amplitude'],
            e['temperature_amplitude'],e['isothermal'],float(diffusivity),cells)
        field = temp if e['observable']=='temperature_amplitude' else comp
        out.append(float(field@mode/(mode@mode)))
    return np.array(out)
