# Entropy anomaly — revision 3

Revision 3 adds a static Lorentz force to the two-dimensional anisotropic calorimeter. The known field changes the rotation of heat-carrying velocity moments even though it does no mechanical work. Calibration includes unequal drag and temperature gradients at zero field, so it already verifies the complete revision 2 heat coefficient. Revision 2 is preserved with its sources, controls, reviews and six trials in `archives/entropy-anomaly-r2`: plain 1/3, hint 3/3. Its two failures were repaired by changing only the kinetic coefficient at fixed fitted friction. Revision 1 is separately archived.

## Physical model

The public underdamped equations specify two velocity components, constant diagonal drag, independent local thermal noise, and Lorentz force `(B*vy,-B*vx)`. The measured quantity is the stationary local Stratonovich bath heat divided by temperature, followed by the mass-to-zero limit. Background heat maintaining the temperature profile is excluded. This is physical calorimetry, so no choice of magnetic reversal convention for an inferred positional path entropy enters the target.

Write gx=gamma, gy=lambda*gamma and

    Gamma = [[gx,-B],[B,gy]].

For the marginal position density rho(x), normalized on one period, the physical currents integrated over y obey

    Gamma*J = F*rho - ex*(T*rho)'.

The starter and oracle both solve this equation. Eliminating Jy gives effective force Fx+B*Fy/gy and effective x drag gx+B^2/gy. They retain the full transverse current Jy=(Fy*rho-B*Jx)/gy. Both compute the positional contribution `integral J^T diag(gx,gy) J/(T*rho) dx`, equal to `integral F.J/T dx` on the periodic domain.

The completed shortcut retains the exact zero-field kinetic coefficient

    C0 = (1/2) * [1/gx + 1/(gx+2*gy)]

for all fields. This is the assumption that a nonworking magnetic force leaves the fast thermal heat contribution unchanged. It uses the magnetic field substantively in the density, both currents and positional dissipation. It returns positive entropy and fits every calibration input. No missing field input or numerical failure is needed for the hidden error.

## Cubic-moment physics

Use normalized Gaussian Hermites in the order H30,H21,H12,H03. Their dissipative relaxation matrix has diagonal `(3gx,2gx+gy,gx+2gy,3gy)`. Lorentz rotation adds antisymmetric neighboring entries `A[i,i+1]=-B*sqrt((3-i)*(i+1))`, with the opposite sign below the diagonal. The temperature-gradient source and kinetic-energy-flux readout use `a=(sqrt(6),0,sqrt(2),0)`. The limiting coefficient is

    C(B) = (1/4) * a^T A^-1 a,
    entropy = positional contribution + C(B)*integral rho*(T')^2/T dx.

The antisymmetric rotation mixes all four cubic moments. Its absence of mechanical work does not imply that their response to a temperature gradient is unchanged. The positive symmetric part of A makes C positive. At B=0 it reduces to C0. At equal drag, the heat-flux combination has angular order one, giving the independent formula `C(B)=6gamma/(9gamma^2+B^2)`.

The stationary finite-mass energy balance provides a separate identity:

    entropy = integral (Fx*Jx+Fy*Jy)/T dx
              - (m/2)*integral (M30+M12)*T'/T^2 dx.

Magnetic work cancels exactly. Uniform temperature gives `F^T Gamma^-1 F/T`; at zero forces the entropy is `C(B)*T0*k^2*(1-sqrt(1-contrast^2))`.

## Independent reference and validation

The private verifier does not use C(B). It solves the stationary finite-mass Kramers equation in Fourier modes of the temperature phase and Gaussian-weighted Hermites of both scaled velocities `wi=sqrt(m/T0)*vi`. Lorentz coupling preserves total Hermite degree but mixes the two components, so the revision 2 triangular y closure is not used. Spatial transport, external forces, both thermal diffusion terms and Lorentz mixing enter the full sparse generator.

The default reference uses 41 Fourier modes and all two-velocity Hermites through total degree 14. It evaluates the bath heat directly as the sum of `gamma_i/m * integral [T0*(c00+sqrt(2)*c_2i)/T-c00] dx`. The subtraction is the mean Stratonovich noise work. Three positive masses h,h/2,h/4 are extrapolated quadratically, with `h=.006*min(gx,gy)^2/(T0*k^2)`.

Calibration has 192 measurements, covering both unequal and equal drag, nonzero forces, thermal contrasts .25 through .6 and wave numbers 1 through 3, all at B=0. The response is a known coefficient divided by the one unknown gamma, so weighted regression identifies inverse friction. Regeneration requires `--generate`; seed 9332.

Run the validator from the repository root:

    OPENBLAS_NUM_THREADS=1 uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 scripts/validate_entropy_anomaly.py

The validator checks 256 calibration-noise realizations, then hidden sensitivity at their observed fitted-parameter extrema. Independent checks halve the mass scale, increase Fourier/Hermite resolution, check probability normalization and positive conditional velocity covariance, and compare direct heat against the third-moment identity. It also checks uniform baths, zero-field agreement, isotropic magnetic response, field/reflected-force symmetry, force-work equality and allowed-range corners. Tiny negative velocity-tail oscillations from Hermite truncation are measured under increasing degree; the heat observable is independently converged.

Reports are `jobs/entropy-anomaly-r3-validation/summary.json` and `results/entropy-anomaly-r3-science.json`. Physics peer reviews are recorded with source hashes in `results/constraint-hardening-physics-review.json` and `results/materials-entropy-r3-kinetic-review.json`. Local and Harbor controls are separate checks; scientific validation does not establish a Luna failure rate.

The first Docker oracle control timed out at 60 seconds before any agent trials ran. Its Dockerfile and failure evidence are preserved in `archives/entropy-anomaly-r3-control-timeout`. The retry limits BLAS/OpenMP threads in Docker; the equations, data, verifier and 60-second limit are unchanged. Local single-thread controls took about 5 seconds.

## Reviewed revision 3 evaluation

The completed threaded retry at `jobs/matrix-hardening-r3-threads-20261002-073658/index.json` produced **0/3 unhinted passes and 3/3 hinted passes** for gpt-5.6-luna at high reasoning effort. All six trials completed without infrastructure exceptions. The oracle passed and completed shortcut failed only the three hidden groups. All paired environments, private tests and task limits match; the hinted instruction is exactly the ordinary instruction plus `hint.md`. Native session provenance confirms the requested model, effort and CLI.

All three unhinted submissions fitted friction accurately while retaining the exact zero-field anisotropic heat coefficient at nonzero magnetic field. They therefore missed Lorentz mixing of the cubic velocity moments; their errors were physical rather than coding or numerical failures. Replacing only that coefficient at each submitted fitted friction repaired all hidden groups, with maximum normalized error below 0.000286 and unchanged calibration. The three hinted submissions independently derived raw or residual cubic-moment balances, with one also recovering calorimetry through the second-moment correction. Their predictions agree with the physical solution on 40 off-grid preparations at their own fitted frictions to within 2.2e-15.

Evidence: `results/brownian-hardening-trial-reviews.json`, `results/entropy-anomaly-r3-fixed-friction-repair.json`, `results/entropy-anomaly-r3-hinted-cross-check.json`, and `results/brownian-retained-provenance-audit.json`. Earlier revisions and the pre-agent control timeout remain separate historical evidence.
