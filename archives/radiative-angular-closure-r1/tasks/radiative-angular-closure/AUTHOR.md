# Radiative angular closure, revision 1

Staged and scientifically validated; no model evaluations have been run. Empirical difficulty is unknown. The exact beam solution is simple once the angular closure is questioned, so this task must earn its difficulty through fresh trials.

The apparatus fixes independent straight photon propagation with uniform removal. The observable is the local weak total energy-density response, with no Fourier factor of one half. The oracle evolves the individual angular Fourier amplitudes by a diagonal generator. The independent verifier translates each initial real-space intensity profile and applies survival probability.

The completed shortcut solves the linearized four-moment M1 equations exactly. It retains all supplied directions through the initial energy and flux, uses the full realizable Eddington tensor and its analytic Jacobian, and propagates by a matrix exponential. Writing f=|F|/E, the closure is

    P = E[(1-chi)I/2 + (3chi-1)nn^T/2],
    chi = (3+4f^2)/(5+2 sqrt(4-3f^2)).

At zero flux its linear pressure perturbation is I*dE/3. Uniform absorption scales both background moments by the same factor, so f and the flux Jacobian remain constant. Thus the starter is a completed, coherent angular approximation; it contains no deliberate numerical error. Incoherent beams need not have its pressure tensor. In particular, equal opposed x beams have an energy response proportional to cos(k*t), while this moment closure gives cos(k*t/sqrt(3)).

All calibration modes have k=0, so both controls exactly predict A*exp(-kappa*t). Every A and time is positive. Each response decreases strictly throughout the permitted parameter interval; the least-squares loss for noiseless data has a unique zero. There are 288 independent measurements, from 12 settings repeated 24 times, with constant instrument sigma=.001. The true rate .27, data seed 109371 and noise-validation seed 109379 are private. No uncertainty depends on the hidden response or unknown parameter.

Three hidden groups test opposed, oblique and three-direction preparations. A fourth checks uniform and co-propagating limits, which both controls pass. The grading metric is group RMS error divided by group RMS physical signal, with tolerance .04. This admits errors far larger than numerical and fitting uncertainty. The verifier requires the documented array shape and finite values; it imposes no solver or model form.

Validation is recorded in `staging/radiative-angular-closure-r1/results/radiative-angular-closure-validation.json` from the repository root:

- Both controls fit kappa=.27008318, reduced chi-square 1.06071. Oracle hidden errors are below .000143; shortcut diagnostic errors are 1.012,1.589,1.415. The exact anchor group passes.
- All 256 noise realizations pass calibration, parameter recovery and oracle prediction; all 256 reject the shortcut for the three intended groups. Worst oracle error .000442; best shortcut diagnostic error1.0116.
- Spectral oracle and independent characteristic reference agree within 2.78e-16 over 96 random allowed experiments. Analytic M1 Jacobian agrees with finite differences within 2.27e-10. Zero-flux and co-propagating limits, initial values, direction/position reversal, positive prepared intensity and weak-signal normalization pass.
- A separate conservative real-space finite-volume evolution converges from 4.43e-3 absolute error at 64 cells to 1.64e-5 at 1024 cells. The first draft check stopped at 512 cells; refinement resolved the validation threshold without changing task physics or grading.
- Full-range noiseless recovery error is below 1.02e-8. The shortcut gap remains above 1.0076 over the entire absorption range. Local oracle pytest 9/9 passes; shortcut 6 pass/3 intended hidden failures, each under one second.

Reproduce from the repository root:

```bash
uv run --no-project --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python staging/radiative-angular-closure-r1/scripts/validate_radiative_angular_closure.py
```

Use `--generate` only for deliberate data regeneration. Docker controls and all fresh model evaluations are left to the root runner. Task limits remain 600s agent/60s verifier. The neutral instruction explicitly permits replacing any implementation. Public input supplies the propagation law, preparation and detector definition, without naming M1, a preferred solution method or the missing pressure information. Only public environment files enter the image.

The original stopped partial package is preserved with its verified 19-file manifest in `archives/stopped-partial-stage/` under the stage. No canonical tasks or past trial evidence were changed.

This differs from archived phonon transport: there the collision operator incorrectly removes conserved moments; here there is no scattering, and the mistake is a closure of the known angular distribution. Radiative-plates and film-transmission address boundary/emission/flux conventions rather than angular transport.

Primary support: [Melon Fuksman et al., Multidimensional half-moment multigroup radiative transfer (2025)](https://arxiv.org/html/2504.13999v2), equations 12–14 and section 5.1, states the M1 tensor and demonstrates artificial interactions of crossing fluxes. Its appendix E.1 also treats a purely absorbing medium. The present periodic weak-response apparatus and validation are independently derived. The standard closure originates in [Levermore, Relating Eddington factors to flux limiters (1984)](https://doi.org/10.1016/0022-4073(84)90112-2).
