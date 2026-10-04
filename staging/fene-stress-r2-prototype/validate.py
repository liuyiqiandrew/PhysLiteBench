"""Refinement and physical checks for the FENE rotating-flow prototype."""
import itertools
import json
from pathlib import Path
import time
import numpy as np
from scipy.integrate import quad
from scipy.special import i0e
from prototype import spectral, finite_volume, peterlin, flow

start=time.perf_counter()
cases=[(.5,.3,1.,4.1,3.),(.8,.8,1.,4.1,3.),(.8,1.1,.8,4.8,np.sqrt(6)),(.9,.5,1.2,3.2,np.sqrt(12))]
refinements=[]
for case in cases:
    truth,_,_=spectral(*case,degree=24)
    coarse,_,_=finite_volume(*case,nr=80,nt=128)
    fine,_,checks=finite_volume(*case,nr=160,nt=256)
    refinements.append({'case':list(case),'oracle':truth,'coarse_error':abs(coarse-truth),'fine_error':abs(fine-truth),
                        'richardson_error':abs((4*fine-coarse)/3-truth),'checks':checks})
    print('refinement',refinements[-1],flush=True)
rng=np.random.default_rng(241223)
domain=list(itertools.product([-.95,.95],[-1.2,1.2],[.8,1.2],[3.2,4.8],[np.sqrt(6),np.sqrt(12)]))
domain.extend((rng.uniform(-.95,.95),rng.uniform(-1.2,1.2),rng.uniform(.8,1.2),rng.uniform(3.2,4.8),rng.uniform(np.sqrt(6),np.sqrt(12))) for _ in range(16))
checks={'spectral_refinement_max':0.,'negative_probability_mass_max':0.,'source_moment_residual_max':0.,
        'source_covariance_eigenvalue_min':float('inf'),'oracle_covariance_eigenvalue_min':float('inf'),
        'source_extension_fraction_max':0.,'source_stress_power_min':float('inf'),'oracle_stress_power_min':float('inf')}
for i,case in enumerate(domain):
    exact,c,diag=spectral(*case,degree=24);refined,_,_=spectral(*case,degree=28)
    source,sc,f=peterlin(*case);e,o,t,z,l=case;k=flow(e,o)
    residual=k@sc+sc@k.T-4*f/z*sc+4*t/z*np.eye(2)
    checks['spectral_refinement_max']=max(checks['spectral_refinement_max'],abs(exact-refined))
    checks['negative_probability_mass_max']=max(checks['negative_probability_mass_max'],diag['negative_probability_mass'])
    checks['source_moment_residual_max']=max(checks['source_moment_residual_max'],np.max(abs(residual)))
    checks['source_covariance_eigenvalue_min']=min(checks['source_covariance_eigenvalue_min'],np.linalg.eigvalsh(sc).min())
    checks['oracle_covariance_eigenvalue_min']=min(checks['oracle_covariance_eigenvalue_min'],np.linalg.eigvalsh(c).min())
    checks['source_extension_fraction_max']=max(checks['source_extension_fraction_max'],np.trace(sc)/l**2)
    checks['source_stress_power_min']=min(checks['source_stress_power_min'],source*e)
    checks['oracle_stress_power_min']=min(checks['oracle_stress_power_min'],exact*e)
    if i%12==0:print('domain',i,flush=True)
limits=[]
for e,o in [(0.,0.),(0.,.8),(.5,0.),(-.5,0.)]:
    case=(e,o,1.,4.1,3.);exact,c,d=spectral(*case,degree=24)
    if e==0:
        expected=0.;expected_c=np.eye(2)/(1+4/9)
    else:
        k=4.1*e*9/4;a=9/2
        weight=lambda x:(1-x)**a*np.exp(abs(k)*x)*i0e(k*x)
        expected=4.1*e/2*9*quad(lambda x:x*weight(x),0,1,epsabs=1e-12)[0]/quad(weight,0,1,epsabs=1e-12)[0]
        expected_c=c
    limits.append({'flow':[e,o],'stress_error':abs(exact-expected),'equilibrium_covariance_error':float(np.max(abs(c-expected_c)))})
# Axis reversal and vorticity reversal, using the exact microscopic model.
base,_,_=spectral(.7,.5,1.,4.1,3.,degree=24)
rev_e,_,_=spectral(-.7,.5,1.,4.1,3.,degree=24)
rev_o,_,_=spectral(.7,-.5,1.,4.1,3.,degree=24)
selected=[]
for e,o,t,l in [(.65,.2,.8,np.sqrt(6)),(.8,.35,1.,3.),(.95,.5,1.2,np.sqrt(12)),(-.75,-.3,1.1,2.8),(.7,-.45,.9,3.2)]:
    for z in [3.2,4.1,4.8]:
        exact,_,_=spectral(e,o,t,z,l,degree=24);source,_,_=peterlin(e,o,t,z,l)
        selected.append({'case':[e,o,t,z,l],'oracle':exact,'source':source,'relative_error':abs(source/exact-1)})
report={'status':'prototype scientifically viable; no packaged task or model evaluation','domain_cases':len(domain),
        'finite_volume_refinements':refinements,'domain_checks':checks,'limits':limits,
        'sign_symmetries':{'extension_reversal_error':abs(base+rev_e),'rotation_reversal_error':abs(base-rev_o)},
        'candidate_hidden_cases':selected,'elapsed_seconds':time.perf_counter()-start}
Path(__file__).with_name('report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
