"""Standalone weak-RF mechanical torque prototype; no task or evaluation writes."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import root

GAMMA = 1.
ALPHA = .18
MOMENT = 1.07


def coefficients(bias,omega,moment=MOMENT):
    response = GAMMA/(GAMMA*bias-omega+1j*ALPHA*omega)
    dc = ALPHA*moment*omega*abs(response)**2/GAMMA
    mechanical = -ALPHA*moment*omega*response/GAMMA
    magnetic = 1j*moment*(1-bias*response)
    return dc,mechanical,magnetic


def gilbert(m,field):
    cross = np.cross(m,field)
    return -GAMMA/(1+ALPHA**2)*(cross+ALPHA*np.cross(m,cross))


def finite_amplitude_reference(bias,omega,amplitude,moment=MOMENT):
    """Full nonlinear Gilbert steady motion in rotating coordinates.

    Compute torque on the holding clamp from total angular momentum balance,
    without inserting the damping-torque or weak-response formulas.
    """
    z = np.array([0.,0.,1.])
    field = np.array([amplitude,0.,bias])
    def state(xy):
        return np.array([*xy,np.sqrt(1-xy@xy)])
    def residual(xy):
        m = state(xy)
        return (gilbert(m,field)-omega*np.cross(z,m))[:2]
    h = GAMMA/(GAMMA*bias-omega+1j*ALPHA*omega)
    guess = amplitude*np.array([h.real,h.imag])
    answer = root(residual,guess,tol=1e-12)
    m0 = state(answer.x)
    assert np.linalg.norm(residual(answer.x))<1e-12 and m0[2]>.99
    # Reconstruct a complete lab-frame orbit and Fourier-analyze the torque.
    phases = np.linspace(0,2*np.pi,128,endpoint=False)
    c,s = np.cos(phases),np.sin(phases)
    m = np.array([c*m0[0]-s*m0[1],s*m0[0]+c*m0[1],np.full_like(c,m0[2])]).T
    field = np.array([amplitude*c,amplitude*s,np.full_like(c,bias)]).T
    velocity = np.array([gilbert(mm,bb) for mm,bb in zip(m,field)])
    magnetic = moment*np.cross(m,field)
    spin_rate = -moment*velocity/GAMMA
    mechanical = magnetic-spin_rate
    dc = float(mechanical[:,2].mean()/amplitude**2)
    ac = np.mean((mechanical[:,0]+1j*mechanical[:,1])*np.exp(-1j*phases))/amplitude
    angular_identity = np.max(abs(mechanical-ALPHA*moment*np.cross(m,velocity)/GAMMA))
    field_rate = np.array([-omega*amplitude*s,omega*amplitude*c,np.zeros_like(c)]).T
    absorbed = -moment*np.sum(m*field_rate,axis=1)
    dissipation = ALPHA*moment*np.sum(velocity**2,axis=1)/GAMMA
    heat_balance = np.max(abs(absorbed-dissipation))
    dc_identity = abs(mechanical[:,2].mean()-magnetic[:,2].mean())/amplitude**2
    return np.array([dc,ac.real,ac.imag]),dict(angular_balance=float(angular_identity),
        positive_dissipation_min=float(dissipation.min()),heat_balance=float(heat_balance),
        dc_magnetic_equals_mechanical=float(dc_identity),minimum_mz=float(m[:,2].min()))


def main():
    calibration = [(b,w) for b in [.9,1.1,1.3] for w in [.45,.7,.95,1.2]]
    design = np.array([coefficients(b,w,1.)[0] for b,w in calibration])
    curvature = float(2*design@design)
    fits = []
    for true in [.8,1.07,1.2]:
        fitted = float(design@(true*design)/(design@design))
        fits.append(abs(fitted-true))
    sigma = .002
    noise = np.random.default_rng(318223).normal(0,sigma,(256,len(design)))
    measured = MOMENT*design+noise
    fitted = measured@design/(design@design)
    hidden = [(b,w) for b in [.85,1.1,1.35] for w in [.55,.85,1.15,1.4]]
    checks = []
    maximum_extrapolation = maximum_refinement = maximum_identity = maximum_heat_error = 0.
    for bias,omega in hidden:
        dc,body,field = coefficients(bias,omega)
        exact = np.array([dc,body.real,body.imag])
        values = []
        for amplitude in [.004,.002,.001]:
            value,diagnostics = finite_amplitude_reference(bias,omega,amplitude)
            values.append(value)
            maximum_identity = max(maximum_identity,diagnostics['angular_balance'],diagnostics['dc_magnetic_equals_mechanical'])
            maximum_heat_error = max(maximum_heat_error,diagnostics['heat_balance'])
            assert diagnostics['positive_dissipation_min']>0
        extrapolation = (4*values[1]-values[0])/3
        refined = (4*values[2]-values[1])/3
        maximum_extrapolation = max(maximum_extrapolation,float(abs(refined-exact).max()))
        maximum_refinement = max(maximum_refinement,float(abs(refined-extrapolation).max()))
        errors = [float(np.linalg.norm(v-exact)/np.linalg.norm(exact)) for v in values]
        checks.append(dict(bias=bias,omega=omega,dc=dc,mechanical_ac=[body.real,body.imag],
                           magnetic_ac=[field.real,field.imag],ac_relative_error=float(abs(field-body)/abs(body)),
                           nonlinear_relative_errors=errors,refined_limit_absolute_error=float(abs(refined-exact).max())))
    assert maximum_extrapolation<1e-6
    assert maximum_identity<1e-9 and maximum_heat_error<1e-12
    report = {
        'scope':'Standalone prototype only. No canonical task, calibration, grading, or evaluation changes.',
        'apparatus':'A uniform classical macrospin is embedded in an isotropic insulating body held at zero angular velocity by an ideal torque clamp. Its moment is mu=M*m and spin angular momentum S=-M*m/gamma. The only spin damping is a specified local spin-lattice Gilbert channel. All angular momentum lost through it reaches the measured body/clamp; no electronic spin pumping, external spin bath, radiative angular-momentum loss or other dissipative torque is included. Thermal fluctuations are neglected as part of the deterministic ideal model.',
        'sign_and_readout':'The sensor measures torque exerted by the body on the holding clamp, not the opposite clamp-on-body reaction. B=B0*z+b*(cos(omega*t),sin(omega*t),0), gamma>0. Reach the stable periodic state continuously connected to m=z as b tends to zero.',
        'readout_order':'Define the DC signal as lim_b->0 <tau_z>/b^2 and the complex AC signal as lim_b->0 <(tau_x+i*tau_y)*exp(-i*omega*t)>/b. These are measured response coefficients; their definitions make the retained orders exact and avoid an unspecified small-field truncation.',
        'known_constants':dict(gamma=GAMMA,alpha=ALPHA),
        'proposed_ranges':dict(moment=[.8,1.2],bias=[.8,1.4],omega=[.4,1.4]),
        'private_derivation':{
            'magnetic_response':'m_x+i*m_y = b*h*exp(i*omega*t)+O(b^3), h=gamma/(gamma*B0-omega+i*alpha*omega).',
            'balance':'tau_body=mu cross B-dS/dt=(alpha*M/gamma)*m cross dm/dt. Clamping gives zero lattice angular velocity but does not suppress spin storage.',
            'dc':'Both models give M*alpha*omega*|h|^2/gamma because periodic spin storage averages to zero.',
            'mechanical_ac':'-alpha*M*omega*h/gamma',
            'shortcut_ac':'i*M*(1-B0*h)',
            'ac_separation':'The full complex AC relative error equals1/alpha; it comes from the omitted spin-storage torque, while susceptibility and DC torque remain exact.',
            'power':'RF absorbed power is omega*<tau_z>; it equals alpha*M*|dm/dt|^2/gamma for the periodic circular orbit.'},
        'calibration':dict(controls=calibration,minimum_nonzero_design_coefficient=float(design.min()),
                           strictly_convex_loss_curvature=curvature,noiseless_absolute_fit_error=max(fits),
                           fixed_sigma=sigma,noise_draws=256,maximum_noise_relative_moment_error=float(abs(fitted/MOMENT-1).max()),
                           fitted_moment_standard_deviation=float(sigma/np.sqrt(design@design))),
        'independent_reference':'Nonlinear Gilbert equation solved in the rotating frame, followed by lab-frame orbit reconstruction, total angular momentum balance and Fourier integration. Three finite field amplitudes plus b^2 Richardson extrapolation check the specified response coefficients.',
        'maximum_refined_nonlinear_reference_error':maximum_extrapolation,
        'maximum_extrapolation_refinement_change':maximum_refinement,
        'maximum_angular_or_dc_balance_error':maximum_identity,
        'maximum_heat_balance_error':maximum_heat_error,
        'hidden_examples':checks,
        'primary_source':{
            'title':'Keshtgar, Streib, Kamra, Blanter and Bauer, Magnetomechanical coupling and ferromagnetic resonance in magnetic nanoparticles, Phys. Rev. B95,134447(2017)',
            'url':'https://arxiv.org/html/1610.01072v2',
            'support':'Section IV, Eqs18–21, states total angular momentum balance, mechanical torque and Gilbert transfer to the lattice. Section III defines S=-VMs*m/gamma. The prototype is its zero-body-rotation, isotropic-effective-field case.'},
        'caveats':[
            'A generic Gilbert coefficient alone does not identify the destination of dissipated angular momentum; the apparatus must state the entire channel belongs to the measured lattice/clamp.',
            'A torque on the whole spin-plus-body system differs from the torque transmitted by the lattice to the clamp. Define the latter and its sign explicitly.',
            'Finite temperature would require stochastic Gilbert fields and an ensemble model; omit thermal noise explicitly for this ideal deterministic task.',
            'No anisotropy, demagnetizing torque, orbital contribution or magnetic-field-gradient force is included; a spherical isotropic single-domain idealization can make these assumptions clear.',
            'Readout response coefficients, or exact perturbative orders, must be stated. Finite-amplitude nonlinear signals require a separately specified branch/preparation.',
            'This is a new spin-mechanical angular-momentum-storage observable, but the simple response formula could still be readily solved by the target model. No difficulty result is claimed.'
        ]}
    destination = Path('results/spin-body-torque-prototype.json')
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='hidden_examples'},indent=2))


if __name__=='__main__':
    main()
