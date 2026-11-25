# Gyroscopic noise, revision 1

The task measures thermal displacement noise of a charged, harmonically trapped classical particle. The supplied code already solves its complete complex, anisotropic, magnetic susceptibility. It constructs a positive reciprocal Gaussian noise spectrum from the symmetric part of that response. This retains every scalar spectrum measured with real instantaneous coordinate weights, including nonzero magnetic fields. A delayed combination of the two detector signals distinguishes the actual equilibrium process from this completed time-symmetric approximation.

The apparatus specifies one classical equilibrium Markov Langevin bath, fixed trap and magnetic field throughout equilibration and measurement, and the two-sided PSD of Q(t)=x(t)+a*y(t-delay). It gives the Fourier convention and the sign of the Lorentz force. It does not give the tensor fluctuation relation, identify the missing correlations, mandate a numerical method, or restrict changes to calibration. The shared neutral instruction allows replacing all implementation details while preserving the API. The same PSD contraction is used for every experiment; only the continuous zero-frequency limit has a separate numerical expression.

## Physics and calibration equivalence

Let J=[[0,1],[-1,0]], K be the rotated positive stiffness matrix, and use exp(-i*omega*t). The exact response is

    chi = [K-omega² I-i*omega*(gamma I-B J)]^-1.

For the specified classical equilibrium reservoir the noise spectrum is

    S = (T/(i*omega))*(chi-chi^dagger)
      = 2*gamma*T*chi*chi^dagger.

The source instead computes

    S0 = (2*T/omega)*Im[(chi+chi^T)/2] = Re(S).

This source is mathematically consistent with the stationary Gaussian process whose matrix time correlation is [C(t)+C(-t)]/2. It is positive: the real part of a Hermitian positive matrix is a real symmetric positive matrix. It preserves the exact auto spectra, real cross spectrum, equal-time positional covariance, field-dependent poles and equilibrium scalar projected spectra. It is a reciprocal correlation closure, not a broken conjugation operation producing a non-Hermitian or negative covariance.

For every real sensor row h, h*S*h^T=h*S0*h^T. Calibration uses delay=0 with varied nonzero signed fields, frequencies, trap angles, temperatures and signed sensor weights. Both controls therefore give exactly the same calibration predictions for any allowed gamma. Two zero-frequency measurements provide responses 2*gamma*T*h*K^-2*h^T with strictly positive slope, proving global identifiability. Full objective scans and noiseless fits at gamma=.401,.67,1.099 confirm it numerically.

A finite y-channel delay gives h=(1,a*exp(i*omega*delay)). Its PSD h*S*h^dagger includes the imaginary antisymmetric cross spectrum that real instantaneous readouts cannot observe. This is a physical time-ordering effect at fixed magnetic bias. All hidden delays and other controls are within the public ranges. Reciprocal-field, instantaneous and DC hidden anchors remain exact for the source, so its failure is confined to the discriminating joint readout.

This apparatus differs from the archived optical cavity's emitted-versus-absorbed power and from spatial modal-noise closures. Here the complete equal-time and instantaneous projected equilibrium statistics remain correct, while operationally delaying a detector makes an equilibrium time-antisymmetric correlation visible. It is still a correlation-closure candidate; source validity does not establish that Luna will miss the distinction.

## Independent reference and checks

The verifier builds the four-dimensional Langevin drift for (x,y,vx,vy), inserts independent reservoir force increments of covariance 2*gamma*T, solves its stationary Lyapunov equation, and integrates exp(A*t)*C over positive time with a resolvent. Adding the negative-time transpose gives the full positional spectrum. It expands the delayed detector contraction independently. This reference never calls the susceptibility or fluctuation-relation helpers.

The author validator additionally integrates the scalar delayed-detector autocorrelation directly in time, using C_xy(t+delay) and C_yx(t-delay). Agreement is below 1e-9 for the tested signed-field cases. This checks the delay sign without inserting a complex sensor weight. Across hidden parameter bounds and 128 random off-grid controls, oracle and verifier matrices differ by at most 8.9e-15; canonical Gibbs covariance and the Lyapunov balance agree within 7.6e-15. Both spectra are positive, all drifts stable, field reversal transposes the exact spectrum, and coordinate reflection leaves the physical readout unchanged. The source becomes exact at B=0, zero delay and zero frequency.

Root's separate check tests 24 random phase-space systems and compares integrated PSD with direct delayed equal-time variance at four settings, agreeing within 1.1e-12. The independent review is in results/root-hardening-physics-review.json, key gyroscopic-noise-r1.

A primary reference for the general tensor fluctuation relation is [Milton and colleagues, Phys. Rev. A 108, 022809 (2023)](https://link.aps.org/accepted/10.1103/PhysRevA.108.022809), equations 2.3–2.5. It distinguishes the anti-Hermitian response from its componentwise imaginary part. The present classical mechanical result follows directly from the stated Langevin apparatus and the derivation above.

## Controls and reproduction

Calibration consists of 288 independent measurements: 24 repeats of 12 settings. True drag is .67 within [.4,1.1]. The fixed PSD instrument uncertainty is .0015 and contains no response or unknown-parameter dependence. Calibration seed is 640913; the 256-realization noise seed is 640917. Public and private calibration copies are identical.

The actual fit is gamma=.6699777460 with reduced chi-square 1.01381. Oracle hidden group errors are at most .00001792, while the source's three discriminating errors are 1.380, .4418 and .9795. The instantaneous/reciprocal anchor group passes for both. Prediction error is group RMS difference divided by group RMS truth, with threshold .04. Calibration chi-square must be below 1.5 and drag error below3%.

All 256 independent fits pass calibration and parameter recovery for both controls. The oracle passes every prediction group; the shortcut fails all three discriminating groups in every realization. Maximum chi-square is 1.30458, maximum relative drag error .00007021, maximum oracle prediction error .00003787, and the minimum discriminating source error is .44173. Fitted drag is identical for both controls in every draw.

The isolated local harness gives eight oracle passes in under one second, and five source passes with exactly the three intended physics failures, also under one second. The 600-second agent and 60-second verifier limits remain unchanged. Full evidence is in results/gyroscopic-noise-validation.json. Run `uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python scripts/validate_gyroscopic_noise.py`; add `--generate` only for deliberate regeneration of both calibration copies. The completed source control is scripts/gyroscopic_noise_baseline.py.

Scientific validity and source separation are established. Model evaluation is scheduled separately by the root runner and no agent outcome is claimed here.

The completed neutral-instruction screen produced 3/3 genuine physical passes: initial ea9CNiK and conditional follow-ups sXHEzjR and 4hSSRxx. All three independently diagnose the missing magnetic cross spectrum and use full Langevin noise propagation. Maximum hidden error is below .000018. The original initial success is included. Native jobs are indexed by jobs/matrix-neutral-r1-20261003-014857/index.json, and final sources, tool trajectories and metrics are reviewed in results/neutrality-brownian-trial-reviews.json. This family does not meet the requested difficulty target. No frozen agent-visible or scientific files were changed during evaluation.
