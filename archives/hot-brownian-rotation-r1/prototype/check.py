import json
import numpy as np
from scipy.integrate import quad

def impedance(omega,eta=1.,rho=1.,a=1.):
    z=a*np.sqrt(-1j*omega*rho/eta)
    return 8*np.pi*eta*a**3*(1+z*z/(3*(1+z)))

def temperature_weight(omega,eta=1.,rho=1.,a=1.):
    if omega==0:return .75,8*np.pi*eta*a**3
    k=np.sqrt(-1j*omega*rho/eta)
    def strain(r):
        A=a**3/r**2*np.exp(-k*(r-a))*(1+k*r)/(1+k*a)
        return A*(-3/r-k+k/(1+k*r))
    def density(r):return r*r*abs(strain(r))**2
    den=quad(density,a,np.inf,epsabs=1e-10,epsrel=1e-11)[0]
    num=quad(lambda r:density(r)*a/r,a,np.inf,epsabs=1e-10,epsrel=1e-11)[0]
    return num/den,8*np.pi*eta*den/3

rows=[]
for omega in [0,.1,.5,1,3,10,30,100,300]:
    weight,diss=temperature_weight(omega)
    exact=1+2*weight;shortcut=2.5
    rows.append(dict(omega=omega,thermal_weight=weight,relative_shortcut_error=abs(shortcut/exact-1),integrated_dissipation=diss,real_impedance=float(impedance(omega).real)))
result={'status':'numerical_proposal_only_no_task_or_model_evaluations','mechanism':'Exact full unsteady rotational impedance but a single static effective bath temperature versus frequency-dependent spatial dissipation weighting.','rows':rows,'max_dissipation_impedance_error':max(abs(r['integrated_dissipation']-r['real_impedance']) for r in rows),'primary_source':'https://arxiv.org/abs/1403.4040','overlap':'Related to deterministic unsteady-sphere and modal-noise heated-filament, but neither tested a nonisothermal continuum contracted to one frequency-independent noise temperature.','calibration':'Zero-frequency rotational velocity spectrum at nonuniform radial temperatures; S(0)=2(T0+3*deltaT/4)/(8*pi*eta*a^3), strictly identifies eta.','public_needed':'Fixed sphere center, freely fluctuating rotations, no-slip, explicit constant-property incompressible linear fluctuating hydrodynamics with local thermal stress, prescribed T=T0+deltaT*a/r, no other torque bath; stationary two-sided angular-velocity PSD.'}
print(json.dumps(result,indent=2))
