"""Conservative staggered acoustic fields and constrained mean mass/momentum balance."""
from functools import lru_cache
import numpy as np
from scipy.linalg import solve_banded
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

RHO=1.
SOUND=1.
EXPONENT=5.
TRUE_PARAMETER=.12


@lru_cache(4096)
def finite_volume(length,mode,ratio,force,viscosity,cells):
    """Solve conservative first-harmonic fields, then mean continuity/momentum and mass."""
    k=mode*np.pi/length;omega=ratio*SOUND*k;dx=length/cells
    faces=np.linspace(0,length,cells+1)
    diffusion=viscosity+1j*RHO*SOUND**2/omega
    bands=np.zeros((3,cells-1),complex)
    bands[0,1:]=-diffusion/dx**2;bands[2,:-1]=-diffusion/dx**2
    bands[1]=-1j*omega*RHO+2*diffusion/dx**2
    velocity=np.zeros(cells+1,complex)
    velocity[1:-1]=solve_banded((1,1),bands,force*np.sin(k*faces[1:-1]))
    density=RHO*np.diff(velocity)/(1j*omega*dx)
    pressure=SOUND**2*density
    wall_linear=(9*pressure[0]-pressure[1])/8
    # The mean density is an unknown, not set to zero locally.
    density_face=(density[:-1]+density[1:])/2
    transported_mass=np.zeros(cells+1)
    transported_mass[1:-1]=.5*np.real(density_face*velocity[1:-1].conj())
    mean_velocity=-transported_mass/RHO
    mean_viscous=viscosity*np.diff(mean_velocity)/dx
    velocity_center=(velocity[:-1]+velocity[1:])/2
    momentum_flux=.5*RHO*abs(velocity_center)**2
    eos_curvature=.25*(EXPONENT-1)*SOUND**2/RHO*abs(density)**2
    nonlinear_flux=momentum_flux+eos_curvature-mean_viscous
    # Difference of total momentum flux is zero; final row fixes total mass.
    matrix=lil_matrix((cells,cells))
    for i in range(cells-1):
        matrix[i,i]=-SOUND**2/dx;matrix[i,i+1]=SOUND**2/dx
    matrix[-1,:]=1/cells
    rhs=np.r_[-np.diff(nonlinear_flux)/dx,0.]
    mean_density=spsolve(matrix.tocsr(),rhs)
    mean_pressure=SOUND**2*mean_density+eos_curvature
    wall=(9*mean_pressure[0]-mean_pressure[1])/8-viscosity*mean_velocity[1]/dx
    total=mean_pressure+momentum_flux-mean_viscous
    energy_in=.5*force*np.sum(np.sin(k*faces[1:-1])*velocity[1:-1].conj()).real*dx
    dissipation=.5*viscosity*np.sum(abs(np.diff(velocity)/dx)**2)*dx
    return {'density':float(abs(wall_linear)/SOUND**2),'wall_pressure':float(wall),
        'mean_density_integral':float(np.mean(mean_density)),
        'mean_mass_flux_max':float(np.max(abs(RHO*mean_velocity+transported_mass))),
        'mean_momentum_flux_variation':float(np.ptp(total)),
        'mean_velocity_max':float(np.max(abs(mean_velocity))),
        'linear_energy_balance':float(abs(energy_in-dissipation)/max(1.,abs(energy_in),abs(dissipation))),
        'source_local_momentum_residual':float(np.max(abs(np.diff(nonlinear_flux)/dx)))}


@lru_cache(4096)
def response(length,mode,ratio,drive,viscosity,cells=256):
    coarse=finite_volume(length,mode,ratio,drive,viscosity,cells)
    fine=finite_volume(length,mode,ratio,drive,viscosity,2*cells)
    return {'density':(4*fine['density']-coarse['density'])/3,
            'force':(4*fine['wall_pressure']-coarse['wall_pressure'])/3}


def predict(experiments,viscosity=TRUE_PARAMETER,cells=256):
    return np.array([response(e['length'],e['mode'],e['frequency_ratio'],e['drive'],viscosity,cells)[e['readout']] for e in experiments])


def experiment(length=3.,mode=1,frequency_ratio=1.,drive=.2,readout='force'):
    return dict(length=float(length),mode=int(mode),frequency_ratio=float(frequency_ratio),drive=float(drive),readout=readout)


def calibration_inputs():
    unique=[experiment(L,m,r,f,'density') for L in [2.5,4.] for m in [1,2] for r in [.9,1.,1.1] for f in [.1,.3]]
    return [dict(e) for _ in range(12) for e in unique]


def hidden_inputs():
    return {
        'resonance':[experiment(L,m,1.,f) for L,m,f in [(2.7,1,.12),(3.6,2,.25),(3.,3,.3),(3.9,1,.2)]],
        'low_frequency':[experiment(L,m,r,f) for L,m,r,f in [(2.5,1,.85,.15),(3.,2,.9,.3),(4.,3,.95,.2),(3.3,1,.8,.25)]],
        'high_frequency':[experiment(L,m,r,f) for L,m,r,f in [(2.8,1,1.05,.2),(3.1,2,1.1,.3),(3.8,3,1.15,.25),(4.,2,1.12,.15)]],
        'density_anchors':[experiment(L,m,r,f,'density') for L,m,r,f in [(2.8,1,.85,.13),(3.4,2,1.05,.27),(3.8,3,1.,.2),(3.,3,1.18,.3)]]
    }
