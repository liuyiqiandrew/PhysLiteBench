# Emitter lifetime, revision 1

Author-only. The public task measures excited-state population near a passive planar medium. The completed shortcut computes the propagating part of the electromagnetic decay rate correctly, including complex Fresnel amplitudes, interference, polarization and height. It interprets that rate as the total population decay. The full response also contains energy transfer through vacuum-evanescent components.

The physical connection between population decay and the retarded electromagnetic Green tensor follows the weak-coupling electric-dipole treatment of absorbing media; see [Crosse and Scheel, Atomic multipole relaxation rates near surfaces](https://arxiv.org/abs/0901.3084). The finite lifetime is fixed by total electromagnetic energy loss, whereas the propagating-sector approximation excludes part of the material coupling. The public specification supplies the passive local medium, population detector, temperature and limiting order without giving a corrective formula.

## Equations and exact calibration equivalence

Let h=k0*d, u=k_parallel/k0, w=sqrt(1-u²), q=sqrt(epsilon-u²), taking outgoing/decaying roots. With exp(-i*omega*t),

    r_s=(w-q)/(w+q), r_p=(epsilon*w-q)/(epsilon*w+q).

The normalized vertical and horizontal dipole decay factors are

    F_z = 1 + (3/2) Re integral_0^infinity [u³ r_p exp(2i*h*w)/w] du,
    F_x = 1 + (3/4) Re integral_0^infinity [u (r_s-w²*r_p) exp(2i*h*w)/w] du.

The tilted factor is F_z*cos(tilt)²+F_x*sin(tilt)² and the survival probability is exp(-vacuum_rate*F*time). The oracle changes variables separately to real w in [0,1] and v=sqrt(u²-1) in [0,infinity), removing the light-line square-root singularity. Both integrals use adaptive quadrature.

Calibration has real epsilon in {.35,.5,.7,1}. Above the vacuum light line, both normal wavevectors are imaginary and r_s,r_p are real. Thus the evanescent contribution to the rate vanishes exactly. Calibration nevertheless varies height and orientation, testing nontrivial propagating interference rather than only the vacuum law. The same scalar rate is identified by strictly decreasing survival functions. The completed source and oracle coincide on all calibration inputs for every allowed rate.

The unknown vacuum rate is .8/ns, allowed .4–1.2/ns. The fixed 600 nm transition and local-response height interval correspond to about 19–143 nm. The model explicitly takes fixed-material weak coupling; it does not extrapolate a finite coupling through a narrowing material resonance. The small test rate compared with the optical frequency also supports the stated lifetime scale. This is an ideal local continuum task, not a fit to a named material.

## Independent reference

The private reference constructs TE/TM incident, reflected and transmitted electric fields for a unit oscillating dipole. It integrates upward Poynting flux. For absorbing media it also integrates epsilon_imag*|E|²/2 throughout the material; the depth integral is analytic because each plane wave decays exponentially. For transparent calibration media it instead integrates transmitted Poynting flux. Four azimuths integrate the quadratic polarization dependence exactly. Dividing total power by the vacuum dipole power produces the lifetime factor independently of a self-field Green formula.

The independent field calculation agrees with the oracle to 5.69e-14 over hidden inputs and allowed corners. Tightening reference quadrature changes results by at most 3.56e-15. Transmitted Poynting flux and depth-integrated Joule heat agree to 7.22e-16. The small-height vertical/horizontal asymptotes, proportional to 3*Im[(epsilon-1)/(epsilon+1)]/(8*h³) and half that value, are checked outside the scored domain to 7.03e-6 relative error. All sampled decay factors and component powers are positive. Vacuum gives the exact free-space factor.

## Data, thresholds and controls

There are 144 transparent-medium records: four permittivities, three heights, three orientations and four times. Their fixed independent Gaussian sigma=.002 does not encode the unknown rate or noiseless response. Generation seed is 946071; 256 independent noise draws use seed946073. Public and private calibration copies are byte-identical. Regeneration is explicit through `--generate`; normal validation preserves frozen data.

The checked-in fit is .79997428054/ns with reduced chi-square .984384. Oracle normalized errors are 1.16e-5, 1.42e-5 and 1.27e-5. Shortcut errors are .4560, 1.0339 and .5011. The standard .03 group threshold was selected before evaluation. All 256 noisy calibrations pass the parameter, calibration and oracle gates; no completed shortcut passes. Worst oracle hidden error is .001087, and the smallest shortcut group error is .4554. This is calibration-noise sensitivity at fixed apparatus inputs, not an exhaustive hidden-domain Monte Carlo test.

The actual private harness passes 7/7 for the oracle and reports 4 passes with the 3 intended hidden failures for the completed shortcut. Both runs take less than one second locally. No agent evaluation result is claimed here.

Reports: `results/emitter-lifetime-validation.json`, `results/emitter-lifetime-local-controls.json`, `results/emitter-lifetime-source-provenance.json`; peer review is recorded in `results/materials-hardening-physics-review.json`. The task uses the byte-identical neutral instruction, the standard pinned NumPy/SciPy/pytest image, 600-second agent and 60-second verifier limits. AUTHOR, hint, reference and hidden tests are not copied into the agent image.
