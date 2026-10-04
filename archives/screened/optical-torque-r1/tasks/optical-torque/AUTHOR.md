# Optical torque, r1

The supplied model has the full passive anisotropic electric polarizability and the correct plane-wave force. Its torque is the field-induced term alone. Fitting a radiation-dressed response does not automatically include the angular momentum carried away by its radiation in the mechanical torque observable.

For the specified exp(-i t) convention and vacuum units, p=alpha E and the torque about the particle center is

    torque = Re(p cross E*)/2 - Im(p* cross p)/(12*pi).

The starter keeps the first term. The independent reference instead integrates r cross (Maxwell stress dot n) over a closed sphere, using the full incident and outgoing dipole fields and subtracting the incident-only stress. It separately checks the outgoing field's angular-momentum flux. Its lab-frame tensor is obtained by matrix inversion, independently of the oracle's dressed principal values.

The tensor is alpha=R diag(a_j/(1-i a_j/(6*pi))) R^T, with a_j=s*[1,1.7,2.6]. It satisfies Im(alpha)=alpha†alpha/(6*pi), so it is lossless and passive for every allowed orientation and strength. The radiated power equals the incident-field work. A lossless anisotropic particle may still receive torque by changing the polarization of scattered radiation. For a normalized circular wave in principal plane i,j the true axial torque is |alpha_i-alpha_j|²/(24*pi), whereas the source gives (|alpha_i|²+|alpha_j|²)/(24*pi). Equal principal values cancel the total torque, but all scored cases use unequal values and have nonzero true torques.

Calibration contains144 principal-axis linear-polarization force records, spanning three principal responses, three field amplitudes, four orientations and four independent repetitions. Force is nonzero, both torque expressions vanish there, and the common positive strength is identifiable from the force. The calibration objective is unimodal over[.6,1.6]; noiseless fits across that full range recover the parameter. Sigma=5e-5 is an instrument constant, not a function of response or strength. Public and private copies are identical. Generation uses seed949101;256 independent calibration-noise draws use seed949103.

At the checked-in fit s=1.0999836597, chi-square is1.08751. Oracle hidden group errors are below2.94e-5. The source errors are6.9787,2.0491 and11.0775, against the unchanged4% group-relative gate. The smallest true hidden torque magnitude is.0069162. All256 noise draws pass the oracle/calibration/parameter checks; no source control passes. Worst noisy oracle error is.0001267 and minimum source error is2.04885. These are local scientific controls, not agent-run outcomes.

Stress torque agrees with the oracle within2.99e-15 over hidden cases and general strength/rotation/polarization probes. The separate self-field angular flux agrees within3.48e-15. Sphere-radius changes and angular quadrature refinement are below1.01e-14. Tests also cover force, tensor passivity, energy balance, principal-axis calibration torque, rotational covariance and the anisotropic circular identity. Actual local verifier: oracle7/7, source4passed and3intended hidden failures, each below a second.

Public inputs define the body axes, complete point response, illumination and measured support force/torque about the center. They do not state the torque formula or name the omitted term. The neutral instruction permits replacing every implementation helper. The force candidate remains a separate family: it concerns linear momentum under interfering illumination; this task concerns angular momentum radiated by an anisotropic dipole even in a single plane wave.

Run scientific validation with:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python scripts/validate_optical_torque.py
```

Use `--generate` only to intentionally regenerate both calibration copies with the stated seed. Reports are `results/optical-torque-validation.json`, `results/optical-torque-local-controls.json` and `results/optical-torque-source-provenance.json`. Root and materials_candidates independently reviewed the source and off-grid Maxwell-stress checks. No agent evaluation had run at science freeze.

Primary sources: Nieto-Vesperinas, [Optical torque: Electromagnetic spin and orbital-angular-momentum conservation laws and their significance](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.92.043843), Phys. Rev. A92,043843(2015); [Optical torque on small bi-isotropic particles](https://arxiv.org/abs/1505.05357), Optics Letters40,3021(2015). The concrete anisotropic controls and independent stress quadrature here are constructed directly from the stated fields.
