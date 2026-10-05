"""Bounded author-only pulse/slab feasibility; no task harness or model runs."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
from numpy.polynomial.legendre import leggauss

ROOT=Path(__file__).resolve().parent

def spectral_average(function,center,width,order=96):
    u,w=leggauss(order)
    weights=w*(1-u*u)**8
    return float(np.dot(weights,function(center+width*u))/weights.sum())

def indices(omega,strength):
    den=1-omega*omega
    phase=1+strength/den
    group=1+strength*(1+omega*omega)/(den*den)
    return phase,group

def predict(strength,center,width,length,kind):
    if kind=='transit':
        return length*spectral_average(lambda omega:indices(omega,strength)[1],center,width)
    if kind=='physical_displacement':
        return length*spectral_average(lambda omega:indices(omega,strength)[1]-1,center,width)
    if kind=='canonical_displacement':
        return length*spectral_average(lambda omega:(1-indices(omega,strength)[0])*indices(omega,strength)[1],center,width)
    raise ValueError(kind)

def maxwell_transmission(omega,strength,length):
    # Solve actual E/H interface equations rather than insert the matched phase.
    eps=1+strength/(1-omega*omega)
    mu=1+strength/(1-omega*omega)
    k=omega*np.sqrt(eps*mu)
    impedance=np.sqrt(mu/eps)
    ep,em=np.exp(1j*k*length),np.exp(-1j*k*length)
    mat=np.zeros((len(omega),4,4),dtype=complex)
    mat[:,0,:]=[-1,1,1,0]
    mat[:,1,0]=1
    mat[:,1,1]=1/impedance
    mat[:,1,2]=-1/impedance
    mat[:,2,1]=ep;mat[:,2,2]=em;mat[:,2,3]=-1
    mat[:,3,1]=ep/impedance;mat[:,3,2]=-em/impedance;mat[:,3,3]=-1
    rhs=np.zeros((len(omega),4,1),dtype=complex);rhs[:,0,0]=rhs[:,1,0]=1
    solution=np.linalg.solve(mat,rhs)[...,0]
    return solution[:,3],solution[:,0]

def temporal_reference(strength,center,width,length,n=16384,dt=.5):
    offsets=2*np.pi*np.fft.fftfreq(n,d=dt)
    active=np.abs(offsets)<width
    amplitude=np.zeros(n,dtype=complex)
    amplitude[active]=(1-(offsets[active]/width)**2)**4
    transmission,reflection=maxwell_transmission(center+offsets[active],strength,length)
    transmitted=amplitude.copy();transmitted[active]*=transmission
    vacuum=amplitude*np.exp(1j*offsets*length)
    times=(np.arange(n)-n//2)*dt
    def moment(spec):
        # Carrier cancels from cycle-averaged flux; keep its envelope.
        flux=np.abs(np.fft.fftshift(np.fft.fft(spec)))**2
        total=flux.sum()
        centroid=float(np.dot(times,flux)/total)
        edge=float((flux[:n//16].sum()+flux[-n//16:].sum())/total)
        return centroid,float(total),edge
    incoming=moment(amplitude);outgoing=moment(transmitted);free=moment(vacuum)
    transit=outgoing[0]-incoming[0]
    # Isolated final center-of-energy balance: Mc² ΔX/E = c Δt, c=1.
    displacement=outgoing[0]-free[0]
    return {'transit':transit,'displacement':displacement,
            'reflection_max':float(np.max(np.abs(reflection))),
            'energy_relative_error':abs(outgoing[1]/incoming[1]-1),
            'edge_fraction':max(incoming[2],outgoing[2],free[2])}

def main():
    started=time.monotonic()
    rows=[]
    for f in np.linspace(.2,.4,41):
        for center in (1.5,1.6,1.7):
            for width in (.04,.07,.1):
                for length in (.5,1,2):
                    args=(float(f),center,width,length)
                    source=predict(*args,'canonical_displacement')
                    physical=predict(*args,'physical_displacement')
                    rows.append({'strength':float(f),'center':center,'width':width,'length':length,
                                 'source':source,'physical':physical,'relative_gap':abs(source/physical-1)})
    references=[]
    rng=np.random.default_rng(101509)
    cases=[(.2,1.5,.1,.5),(.4,1.5,.1,2),(.2,1.7,.1,2),(.4,1.7,.1,.5),(.3,1.6,.04,1)]
    cases += [(float(rng.uniform(.2,.4)),float(rng.uniform(1.5,1.7)),float(rng.uniform(.04,.1)),float(rng.uniform(.5,2))) for _ in range(19)]
    for args in cases:
        coarse=temporal_reference(*args)
        fine=temporal_reference(*args,n=32768)
        physical=predict(*args,'physical_displacement')
        transit=predict(*args,'transit')
        references.append({'controls':list(args),'reference':coarse,'refined':fine,
                           'displacement_relative_error':abs(coarse['displacement']/physical-1),
                           'transit_relative_error':abs(coarse['transit']/transit-1),
                           'refinement_relative':abs(coarse['displacement']-fine['displacement'])/physical})
    calibration=[(c,w,L) for c in (1.5,1.6,1.7) for w in (.04,.1) for L in (.5,1,2)]
    coefficients=np.array([predict(1,c,w,L,'physical_displacement') for c,w,L in calibration])
    lengths=np.array([L for c,w,L in calibration])
    fits=[]
    for f in np.linspace(.2,.4,41):
        data=np.array([predict(f,c,w,L,'transit') for c,w,L in calibration])
        recovered=float(coefficients@(data-lengths)/(coefficients@coefficients))
        fits.append({'truth':float(f),'recovered':recovered,'error':abs(recovered-f)})
    omega=np.linspace(1.4,1.8,401)
    phase,group=indices(omega[:,None],np.array([.2,.4])[None,:])
    derivative=( (omega+1e-5)*indices(omega+1e-5,.3)[0]-(omega-1e-5)*indices(omega-1e-5,.3)[0])/(2e-5)
    vacuum=temporal_reference(0,1.6,.08,1)
    # Preserve rejected sign-only below-resonance exploration.
    low=(.3,.6,.05,1)
    variants={'rejected_below_resonance':{'controls':list(low),'source':predict(*low,'canonical_displacement'),
               'physical':predict(*low,'physical_displacement'),'reason':'Opposite signs would expose the error immediately; not proposed.'},
              'rejected_nondispersive_phase_delay_closure':{'reason':'Substituting phase index for group index in energy transport is too close to archived resonator-energy; source instead keeps group response exact.'}}
    summary={'pointwise_count':len(rows),'minimum_gap':min(x['relative_gap'] for x in rows),
             'maximum_gap':max(x['relative_gap'] for x in rows),
             'physical_range':[min(x['physical'] for x in rows),max(x['physical'] for x in rows)],
             'source_range':[min(x['source'] for x in rows),max(x['source'] for x in rows)],
             'reference_count':len(references),
             'maximum_reference_relative_error':max(x['displacement_relative_error'] for x in references),
             'maximum_refinement_relative':max(x['refinement_relative'] for x in references),
             'maximum_reference_energy_error':max(x['reference']['energy_relative_error'] for x in references),
             'maximum_reflection_amplitude':max(x['reference']['reflection_max'] for x in references),
             'maximum_time_window_edge_fraction':max(x['reference']['edge_fraction'] for x in references),
             'minimum_phase_index':float(phase.min()),'maximum_phase_index':float(phase.max()),
             'minimum_group_index':float(group.min()),'maximum_group_index':float(group.max()),
             'group_derivative_error':float(np.max(np.abs(derivative-indices(omega,.3)[1]))),
             'minimum_calibration_derivative':float(coefficients.min()),
             'calibration_information_unit_sigma':float(coefficients@coefficients),
             'maximum_noiseless_fit_error':max(x['error'] for x in fits),
             'vacuum_displacement':vacuum['displacement'],
             'runtime_seconds':time.monotonic()-started}
    assert summary['minimum_gap']>.2 and min(summary['physical_range'])>.1 and min(summary['source_range'])>.1
    assert summary['maximum_reference_relative_error']<1e-6
    assert summary['maximum_refinement_relative']<1e-6
    assert summary['maximum_noiseless_fit_error']<1e-12
    assert phase.min()>0 and phase.max()<1 and group.min()>1
    report={'status':'bounded_feasibility_passed','empirical_model_difficulty':'unevaluated',
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'summary':summary,'separation':rows,'independent_time_domain_checks':references,
            'noiseless_fits':fits,'preserved_variants':variants,
            'reference_limit':'Independent Maxwell interface solve and temporal energy centroids, combined with isolated final center-of-energy balance. Not a microscopic material-force simulation.'}
    (ROOT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
