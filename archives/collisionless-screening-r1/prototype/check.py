"""Author-only causal-response prototype; no task or model-agent run."""
from pathlib import Path
import json
import numpy as np
from scipy.integrate import quad
from scipy.special import spherical_jn


def standing_h(zeta):
    z = float(zeta)
    return 15 / 4 * (2*z*z - 4/3 + (z**3-z)*np.log(abs((z-1)/(z+1))))


def transfer(density, k, width, omega, causal=True):
    z = omega/(k*width)
    h = complex(standing_h(z))
    if causal and abs(z) < 1:
        h += 1j*np.pi*(15/4)*(z**3-z)
    return 1/(1-density*h/(k*width)**2)


def characteristic(s):
    """Velocity characteristic derived by integrating the specified density."""
    if abs(s) < 1e-3:
        return 1-s*s/14+s**4/504-s**6/33264
    return 15*spherical_jn(2, s)/(s*s)


def causal_memory(density, k, width, omega, eta):
    """Duhamel memory integral for an exp((eta-i*omega)*t) drive."""
    # Oscillatory infinite-time quadrature is independent of the PV/log oracle.
    kw = k*width
    def kernel(t):
        return t*characteristic(kw*t)*np.exp(-eta*t)
    real = (quad(kernel, 0, np.inf, weight='cos', wvar=omega,
                 epsabs=2e-10, limlst=300, limit=300)[0] if omega else
            quad(kernel, 0, np.inf, epsabs=2e-10, limit=1500)[0])
    imag = quad(kernel, 0, np.inf, weight='sin', wvar=omega,
                epsabs=2e-10, limlst=300, limit=300)[0] if omega else 0.
    return 1/(1+density*(real+1j*imag))


def memory_limit(density, k, width, omega, scale=1.):
    eta = scale*np.array([.04,.02,.01])
    values = np.array([causal_memory(density,k,width,omega,x) for x in eta])
    return np.polynomial.polynomial.polyfit(eta,values,2)[0], values


def pv_quadrature(z):
    def derivative(u):
        return 15/4*(u**3-u)
    if abs(z) < 1:
        return quad(derivative,-1,1,weight='cauchy',wvar=z,epsabs=2e-12)[0]
    return quad(lambda u: derivative(u)/(u-z),-1,1,epsabs=2e-12)[0]


def main():
    records=[]
    for density in [.65,1.05,1.35]:
        for k,width,z in [(1.5,.9,.25),(1.7,1.1,.45),(2.,1.,.65),(1.8,.9,1.5),(2.1,1.1,2.),(1.7,1.,0.)]:
            omega=k*width*z
            correct=transfer(density,k,width,omega)
            shortcut=transfer(density,k,width,omega,False)
            ref,values=memory_limit(density,k,width,omega)
            refined,_=memory_limit(density,k,width,omega,.5)
            records.append(dict(density=density,k=k,width=width,zeta=z,
                oracle=[correct.real,correct.imag],shortcut=[shortcut.real,shortcut.imag],
                reference=[ref.real,ref.imag],refined=[refined.real,refined.imag],
                absolute_reference_error=abs(ref-correct),
                absolute_refined_error=abs(refined-correct),
                in_phase_relative_gap=abs(correct.real-shortcut.real)/abs(correct.real),
                full_complex_relative_gap=abs(correct-shortcut)/abs(correct)))
    calibration=[r for r in records if r['zeta']==0 or abs(r['zeta'])>1]
    hidden=[r for r in records if 0<r['zeta']<1]
    report=dict(status='prototype_only_no_agent_evaluation',cases=records,
      maximum_reference_error=max(r['absolute_reference_error'] for r in records),
      maximum_refined_error=max(r['absolute_refined_error'] for r in records),
      minimum_hidden_in_phase_relative_gap=min(r['in_phase_relative_gap'] for r in hidden),
      minimum_hidden_complex_relative_gap=min(r['full_complex_relative_gap'] for r in hidden),
      maximum_calibration_difference=max(abs(complex(*r['oracle'])-complex(*r['shortcut'])) for r in calibration),
      maximum_pv_formula_error=max(abs(standing_h(z)-pv_quadrature(z)) for z in [-2.,-.65,-.25,0.,.25,.65,1.5,2.]),
      derivative_static_formula='d response/d density = -5/(k*width)^2 / (1+5*density/(k*width)^2)^2 < 0',
      limitations=['Only 18 deterministic science cases; no 256-noise study or packaged task.',
       'Causal reference extrapolates finite adiabatic switch rates; full allowed-domain bounds remain future work.',
       'PV response is the nonresonant standing-response approximation extended into the resonant domain; trial classification must distinguish that premise from accidental complex arithmetic.'])
    (Path(__file__).parent/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))

if __name__=='__main__':
    main()
