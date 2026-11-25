"""Unevaluated geometric-rotor concept: local scalar energy versus spinor holonomy."""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import minimize_scalar


def effective(inertia,temperature,winding,tilt,flux=True,cutoff=24):
    n=np.arange(-cutoff,cutoff+1,dtype=float)
    connection=.5*winding*(1-np.cos(tilt)) if flux else 0.
    scalar=winding**2*np.sin(tilt)**2/(8*inertia)
    energy=(n-connection)**2/(2*inertia)+scalar
    weight=np.exp(-(energy-energy.min())/temperature);weight/=weight.sum()
    return float(weight@energy)


def microscopic(inertia,temperature,winding,tilt,gap,cutoff=24):
    """Exact finite-gap two-component periodic Hamiltonian, full thermal trace."""
    n=np.arange(-cutoff,cutoff+1,dtype=float)
    kinetic=np.column_stack(((n-winding)**2/(2*inertia),n**2/(2*inertia)))
    h=np.zeros((len(n),2,2))
    h[:,0,0]=kinetic[:,0]+gap*np.cos(tilt)
    h[:,1,1]=kinetic[:,1]-gap*np.cos(tilt)
    h[:,0,1]=h[:,1,0]=gap*np.sin(tilt)
    values,vectors=np.linalg.eigh(h)
    measured=np.einsum('nai,na,nai->ni',vectors,kinetic,vectors)
    population=np.exp(-(values-values.min())/temperature);population/=population.sum()
    return float(np.sum(population*measured))


def extrapolated(inertia,temperature,winding,tilt,base=256.):
    # Inverse-gap Taylor extrapolation; neither connection nor scalar term used.
    y=np.array([microscopic(inertia,temperature,winding,tilt,base*f) for f in [1,2,4]])
    return float(y@[1/3,-2,8/3])


def main():
    true=1.07
    cal=[(t,q,np.pi/2) for t in [.06,.1,.18,.3] for q in [2,4]]
    hidden=[(t,q,theta) for t in [.05,.1,.2] for q,theta in [(1,np.pi/2),(2,np.pi/3),(1,.65*np.pi)]]
    groups=[]
    for t,q,theta in hidden:
        good=effective(true,t,q,theta);bad=effective(true,t,q,theta,False)
        ref=extrapolated(true,t,q,theta);fine=extrapolated(true,t,q,theta,512.)
        groups.append(dict(temperature=t,winding=q,tilt=theta,kinetic_energy=good,scalar_only_energy=bad,relative_gap=abs(bad/good-1),finite_gap_extrapolation=ref,reference_error=abs(ref-good),reference_refinement=abs(ref-fine)))
    equivalence=max(abs(effective(i,t,q,theta)-effective(i,t,q,theta,False)) for i in [.8,1.07,1.2] for t,q,theta in cal)
    design=np.array([effective(true,*x) for x in cal])
    recovery=[];profile_minima=[]
    for test_inertia in [.8,1.,1.2]:
        exact=np.array([effective(test_inertia,*x) for x in cal])
        def loss(i):return sum((effective(i,*x,False)-y)**2 for x,y in zip(cal,exact))
        grid=np.linspace(.8,1.2,801);losses=np.array([loss(i) for i in grid])
        interior=int(np.sum((losses[1:-1]<losses[:-2])&(losses[1:-1]<losses[2:])))
        endpoint=int(losses[0]<losses[1])+int(losses[-1]<losses[-2])
        profile_minima.append(interior+endpoint)
        optimum=minimize_scalar(loss,bounds=(.8,1.2),method='bounded',options={'xatol':1e-14})
        candidates=[.8,1.2,optimum.x];fit=min(candidates,key=loss)
        recovery.append(abs(fit-test_inertia))
    # A constant gauge rephasing with integer winding shifts the connection
    # by an integer but cannot change the partition function or kinetic energy.
    gauge_error=[]
    n=np.arange(-32,33,dtype=float)
    for t,q,theta in hidden:
        a=.5*q*(1-np.cos(theta));phi=q*q*np.sin(theta)**2/(8*true)
        energies=(n-a-3)**2/(2*true)+phi
        weight=np.exp(-(energies-energies.min())/t);weight/=weight.sum()
        gauge_error.append(abs(weight@energies-effective(true,t,q,theta)))
    boundary_error=max(abs(microscopic(true,t,q,a,128.,24)-microscopic(true,t,q,a,128.,36)) for t,q,a in hidden)
    assert equivalence<1e-13 and max(profile_minima)==1 and max(recovery)<1e-7
    assert max(g['reference_error'] for g in groups)<1e-7 and boundary_error<1e-13
    assert max(gauge_error)<1e-13 and min(g['relative_gap'] for g in groups)>.01
    report=dict(status='prototype_only_unevaluated',candidate='geometric-rotor',unknown='inertia in[.8,1.2]',true_inertia=true,
        apparatus='Single angular coordinate with a laboratory-basis periodic two-component wavefunction. H=L^2/(2I)+Delta[cos(theta)sigma_z+sin(theta)(cos(q phi)sigma_x+sin(q phi)sigma_y)], integer q. Unconstrained global Gibbs at known T, equilibrium before Delta->infinity; measure rotor kinetic energy L^2/(2I), in hbar=kB=1 units.',
        shortcut='Exact local lower energy and diagonal geometric scalar cost q^2 sin(theta)^2/(8I), then ordinary periodic scalar free-rotor Gibbs states. Missing global electronic holonomy; this is a valid positive scalar Hamiltonian approximation, not inaccurate diagonalization.',
        calibration='theta=pi/2 and nonzero even q=2,4. The omitted connection q/2 is integer, so the complete energy set and kinetic readout agree exactly, including nonzero scalar cost. Multiple temperatures identify I.',
        hidden_protocol='Odd winding or tilted textures make the induced connection noninteger; the physical kinetic energy includes shifted angular sectors.',
        correct_effective_energy='(n-q*(1-cos(theta))/2)^2/(2I)+q^2 sin(theta)^2/(8I), n integer.',
        independent_reference='Exact thermal trace of finite-Delta 2x2 Fourier blocks in the lab spin basis |up,n-q>,|down,n>; measure the original diagonal kinetic operator, then inverse-gap extrapolate. No Berry-connection formula appears in that calculation.',
        calibration_equivalence_max=equivalence,calibration_loss_minima=profile_minima,noiseless_fit_error_max=max(recovery),calibration_design_range=[float(design.min()),float(design.max())],
        hidden=groups,microscopic_angular_cutoff_error=boundary_error,integer_gauge_invariance_error=max(gauge_error),
        distinctness='No geometric-rotor/Born-Oppenheimer holonomy task found in archive AUTHOR scan. Boson-hall uses Berry curvature but misses magnetization transport, not global wavefunction boundary conditions. Rigid-linkage uses classical canonical measure. Superconducting-ring retains winding correctly and misses kinetic mechanical stress.',
        limitations='Exact stated Delta->infinity order is essential: finite-Delta corrections would break exact calibration equivalence. Generic low-temperature equilibration must exchange angular momentum; fixed angular-sector preparation would be a different task. Difficulty remains unknown and the Berry mechanism may be recognized readily.',
        primary_sources=[dict(title='On the determination of Born-Oppenheimer nuclear motion wave functions including complications due to conical intersections and identical nuclei',authors='Mead and Truhlar',year=1979,url='https://doi.org/10.1063/1.437734'),dict(title='The geometric phase in molecular systems',author='C. Alden Mead',year=1992,url='https://doi.org/10.1103/RevModPhys.64.51')])
    Path('results/geometric-rotor-prototype.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
