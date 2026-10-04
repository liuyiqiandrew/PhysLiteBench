"""Fixed-mass acoustic wall pressure; prototype only, no task or model trials."""
from pathlib import Path
import hashlib,itertools,json,time
import numpy as np
from scipy.linalg import solve_banded
from scipy.sparse import diags,lil_matrix
from scipy.sparse.linalg import spsolve

RHO=1.
SOUND=1.
EXPONENT=5.
BULK=RHO*SOUND**2


def amplitude(length,mode,ratio,force,viscosity):
    k=mode*np.pi/length;omega=ratio*SOUND*k
    v=force/(viscosity*k*k+1j*RHO*(SOUND*SOUND*k*k/omega-omega))
    return k,omega,v


def analytic(length,mode,ratio,force,viscosity):
    k,omega,v=amplitude(length,mode,ratio,force,viscosity)
    density=-1j*RHO*k*v/omega
    p_second=(EXPONENT-1)*SOUND**2/RHO
    physical=RHO*abs(v)**2/4+p_second*abs(density)**2/8
    source=p_second*abs(density)**2/4
    return {'linear_pressure':float(SOUND**2*abs(density)),'wall_pressure':float(physical),
            'source_wall_pressure':float(source)}


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
    return {'linear_pressure':float(abs(wall_linear)),'wall_pressure':float(wall),
        'mean_density_integral':float(np.mean(mean_density)),
        'mean_mass_flux_max':float(np.max(abs(RHO*mean_velocity+transported_mass))),
        'mean_momentum_flux_variation':float(np.ptp(total)),
        'mean_velocity_max':float(np.max(abs(mean_velocity))),
        'linear_energy_balance':float(abs(energy_in-dissipation)/max(1.,abs(energy_in),abs(dissipation))),
        'source_local_momentum_residual':float(np.max(abs(np.diff(nonlinear_flux)/dx)))}


def main():
    start=time.monotonic();rows=[]
    for L,m,r,F,eta in itertools.product([2.5,4.],[1,2,3],[.85,1.,1.15],[.1,.3],[.08,.18]):
        exact=analytic(L,m,r,F,eta)
        coarse=finite_volume(L,m,r,F,eta,128);fine=finite_volume(L,m,r,F,eta,256)
        rich={key:(4*fine[key]-coarse[key])/3 for key in ['linear_pressure','wall_pressure']}
        rows.append({'input':dict(length=L,mode=m,frequency_ratio=r,force=F,viscosity=eta),
            'analytic':exact,'independent':rich,
            'relative_source_error':abs(exact['source_wall_pressure']/exact['wall_pressure']-1),
            'reference_relative_error':max(abs(rich[k]/exact[k]-1) for k in rich),
            'checks':fine})
    eta_grid=np.linspace(.08,.18,201)
    calibration=[(L,m,r,F) for L in [2.5,4.] for m in [1,2] for r in [.9,1.,1.1] for F in [.1,.3]]
    curves=np.array([[analytic(*x,eta)['linear_pressure'] for eta in eta_grid] for x in calibration])
    minimum_negative_slope=float(np.min(-np.diff(curves,axis=1)/np.diff(eta_grid)))
    assert minimum_negative_slope>0
    maximum_error=max(x['reference_relative_error'] for x in rows)
    assert maximum_error<1e-5
    assert min(x['relative_source_error'] for x in rows)>.2
    assert max(abs(x['checks']['mean_density_integral']) for x in rows)<1e-12
    assert max(x['checks']['mean_mass_flux_max'] for x in rows)<1e-12
    assert max(x['checks']['mean_momentum_flux_variation'] for x in rows)<1e-11
    assert max(x['checks']['linear_energy_balance'] for x in rows)<1e-10
    report={'status':'prototype_only','task_proposal':'acoustic-cavity-pressure','model_evaluations':0,
       'apparatus':'Fixed-mass barotropic liquid between fixed rigid walls. External Eulerian volume-force density eps*f*sin(m*pi*x/L)*cos(omega*t), constant positive longitudinal viscosity, uniform thermostat removing heat without applied momentum. Periodic steady state is taken before eps tends to zero. EOS exponent5 is known.',
       'calibration':'First-harmonic wall-pressure amplitude divided by eps; strictly decreasing with unknown viscosity.',
       'readout':'Time-averaged normal endwall traction minus rest pressure, divided by eps².',
       'shortcut':'Exact linear viscous velocity/density, with local EOS rectification around unchanged Eulerian mean density.',
       'physical_distinction':'Mean density redistributes to enforce local momentum balance while total mass stays fixed; no net mean flow for the single mode.',
       'independent_reference':'Staggered finite-volume first-harmonic fields and separately constrained second-order mean-density/velocity solve.128/256 Richardson; no oracle wall-pressure formula.',
       'checks':{'cases':len(rows),'oracle_reference_relative_error_max':maximum_error,
           'minimum_source_relative_error':min(x['relative_source_error'] for x in rows),
           'maximum_source_relative_error':max(x['relative_source_error'] for x in rows),
           'minimum_wall_pressure_signal':min(x['analytic']['wall_pressure'] for x in rows),
           'maximum_wall_pressure_signal':max(x['analytic']['wall_pressure'] for x in rows),
           'minimum_calibration_negative_slope':minimum_negative_slope,
           'mass_error_max':max(abs(x['checks']['mean_density_integral']) for x in rows),
           'mean_momentum_flux_variation_max':max(x['checks']['mean_momentum_flux_variation'] for x in rows),
           'mean_velocity_max':max(x['checks']['mean_velocity_max'] for x in rows),
           'energy_balance_max':max(x['checks']['linear_energy_balance'] for x in rows),
           'resonance_source_to_correct_ratio':2*(EXPONENT-1)/(EXPONENT+1)},
       'seconds':time.monotonic()-start,'cases':rows,
       'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    Path(__file__).with_name('report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))

if __name__=='__main__':main()
