# Porous-solid candidate, revision 1

This author file is outside the Docker build context. Agent difficulty has not
yet been measured. Target: three GPT-5.6 Luna high-effort trials without a hint,
then three with `hint.md` appended under the same agent configuration.

The apparatus is a uniform sealed linear Biot solid. The public README specifies
the signs, tensor shear convention, reference state, boundary condition and
definitions of the known coupling constants. The only unknown is drained E;
traceless shear calibration identifies it through G=E/[2(1+nu)].

The drained shortcut obeys sigma=2G*epsilon+lambda*tr(epsilon)*I. With no fluid
mass exchange, zeta=alpha*tr(epsilon)+p/M=0, and total stress adds -alpha*p*I.
Consequently K_u=E/[3(1-2nu)]+alpha^2*M. Calibration has zero trace and cannot
distinguish these laws. Hidden compression, constrained extension and mixed
strains have a nonzero trace. The independent reference solves six stress
equations plus the fluid-content equation without eliminating pressure; the
oracle uses the deviatoric/bulk decomposition. These are the standard
[Biot constitutive relations](https://doc.comsol.com/6.3/doc/com.comsol.help.porous/porous_ug_multiphysics.10.02.html).

The verifier uses private calibration and requires reduced chi-squared below
1.5 (256 scalar records minus one fitted parameter), E within 3%, and each hidden
case's stress RMSE divided by its reference RMS below 0.04. Experimental strains
are exact. Measurement sigma is 0.006 times the largest noiseless calibration
stress. The 4% prediction tolerance is much larger than the oracle's maximum
0.086% hidden error across 256 noise draws and much smaller than the shortcut's
minimum 55.5% error. Calibration seed: 9310; noise-validation seed: 19310.

Run from the repository root:

```bash
uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python scripts/validate_poroelastic_solid.py
python3 scripts/run_science.py tasks/poroelastic-solid --agent oracle --trials 1
python3 scripts/run_science.py tasks/poroelastic-solid --agent oracle --trials 1 --solution-model scripts/poroelastic_solid_baseline.py
```

The validator reads checked-in data unless `--generate` is explicit. It checks
both controls on all hidden cases for all 256 noisy calibrations; zero strain,
zero coupling, pure shear and hydrostatic limits; tensor rotation; fluid-content
conservation; and the positive additional fluid-compression energy. Reports go
to `jobs/poroelastic-solid-validation/summary.json`. Harbor controls and agent
trials are separate evaluation steps managed by the parent task.
