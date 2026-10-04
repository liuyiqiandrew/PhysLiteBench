# Three candidate outlines

Status: author-only assessment. No task package or model-agent evaluation. The first outline has a small deterministic prototype; the other two are retained as lower-priority alternatives, not proposed implementations.

## 1. Collisionless screening — strongest

Apparatus: a homogeneous one-dimensional collisionless electron distribution in a rigid neutralizing background, with an externally imposed longitudinal electric Fourier mode. Use mass, charge magnitude and permittivity equal to one. The prepared velocity density is n0 F(v/v0)/v0, where F(u)=15(1−u²)²/16 on |u|<1 and zero elsewhere. The known width v0 is set separately in each preparation; n0 is the sole unknown. The measured total electric field includes the self-consistent induced field. A specified exponentially slow switch-on from the remote past fixes the response, with the linear-amplitude limit first and the switch rate taken to zero afterward. This is an ideal collisionless mean-field apparatus; no thermal bath is claimed.

Completed source: solve the real, time-symmetric standing-response kinetic equation using its exact principal-value velocity integral, then solve Poisson self-consistency. This is a mathematically complete nonabsorptive response approximation. It must not be presented privately as the correct solution of the stated causal preparation. Every density, width, wavenumber and drive frequency participates. It is not implemented by discarding the imaginary part of an already computed causal response.

Calibration: static screening plus several nonzero frequencies whose phase velocities lie outside the occupied velocity interval. There are then no resonant particles, so the approximation is exactly equivalent to the causal response, not merely accurate to a chosen tolerance. Static transfer is 1/[1+5n0/(k v0)²], strictly decreasing in n0; it identifies n0 globally. Proposed n0 range [.6,1.4], k in [1.5,2.2], v0 in [.9,1.2], and nonzero calibration phase-speed ratios at least 1.5 keep the real dielectric denominator at least .57086. A future package would use fixed instrument sigma, e.g. .0012 in normalized field units, rather than response-shaped errors.

Discriminator: phase speeds within the occupied velocities. Even the in-phase field amplitude changes after self-consistency; hidden predictions need not be identically zero in the source. The 18-case prototype has exact calibration equivalence and minimum in-phase relative discrepancy .16078. The complex discrepancy is at least .40098. At ratios up to .65, the proposed source remains away from its own real dielectric pole.

Independent reference: derive the causal response from the time-domain Duhamel memory kernel, integrate at finite switch rates, and extrapolate those rates to zero. This does not use the pole jump or the principal-value logarithm. The maximum reference discrepancy is 9.55e−6, falling to 1.21e−6 when switch rates are halved. This is a prototype convergence result, not final task validation.

Overlap: 235 archive/stage/current AUTHOR or assessment files were searched. No Vlasov, Landau-damping or plasma-dispersion task was found. The closest actual authors read were collisionless-trap (adiabatic individual actions), plasma-compression (CGL anisotropic pressure/actions), qubit r8 (retarded quantum bath memory), and finite-band-reservoir (bound-mode occupations after a contact quench). None uses the kinetic resonant-response preparation. Shared collisionless dynamics or retarded-response vocabulary is acknowledged.

Risks: a strong model may immediately recognize Landau causality. A trajectory that selects the correct causal response but miscodes its sign or logarithm is a mathematical/convention failure, not the target. Compact-support distributions can have undamped collective modes outside support; the calibration must avoid their zeros and the public preparation must retain the remote-past switch-on. One must not assume arbitrary finite-time transients decay. No empirical difficulty is claimed.

Primary background: Landau, *On the vibrations of the electronic plasma* (1946), [primary English PDF](https://www.princeton.edu/~vnd/Landau1946.pdf), [original-author reprint record](https://www.ufn.ru/ru/articles/1967/11/m/). The indexed PDF passage on page 27 and publisher abstract were accessible; direct full-PDF retrieval timed out. The task's compact-support distribution and Volterra derivation are independent constructions, not attributed numerical results from that paper.

## 2. Fixed-reference optical speckle exposure — lower priority

Apparatus: a single detected optical mode contains a fixed coherent reference amplitude E0 and independent circular complex Gaussian scattered light e(t), with known intensity and a specified exponential field correlation. A camera integrates intensity over a known exposure; repeated exposures use the same reference field. Photon shot noise is independently subtracted. The unknown decay rate can be identified by reference-free exposures of varied duration and intensity.

Completed source: a zero-mean Gaussian ensemble model with the exact total field first-order correlation, including its constant component. It applies the ordinary intensity-correlation relation and computes the exposure integral exactly. This is the correct statistic for an ensemble of randomly varying reference-speckle amplitudes, but not for exposures conditioned on one fixed reference.

For dynamic field intensity Id and decay rate lambda, the physical intensity covariance is Id² exp(−2lambda|t|)+2|E0|² Id exp(−lambda|t|). The source additionally retains |E0|⁴. This gives exact calibration at E0=0 and an analytically finite hidden discrepancy. Independent verification can integrate the four real Gaussian quadratures directly before exposure averaging.

Overlap/rank: no speckle author was found, but resonance-fluorescence, dressed-photodetection, and boson-transfer-noise already test optical or quantum counting interpretations. The fixed-versus-resampled component also approaches the just-retired fixed-trap preparation theme. The simple missing constant may make this much easier than its apparatus suggests. Do not build before reconsidering that weakness.

Primary background: Pusey and van Megen, [Dynamic light scattering by non-ergodic media](https://doi.org/10.1016/0378-4371(89)90063-0), Physica A 157,705 (1989). The primary abstract explicitly distinguishes time and ensemble intensity correlations; only its accessible abstract was read.

## 3. Nuclear-spin rotational thermodynamics — not preferred

Apparatus: an ideal dilute gas of identical-nucleus rigid diatomics, with specified nuclear spin and electronic/vibrational exchange symmetry. A preparation catalyst allows nuclear-spin conversion during canonical equilibration. Spectroscopic transition energies at varied weak fields identify the one unknown rotational constant. Hidden rotational heat capacity or energy uses the same complete rotor spectrum but its allowed joint rotational/nuclear-spin states.

Completed source: an exact distinguishable-nucleus rotor Gibbs sum with correct energies, degeneracy 2J+1 and field response, but a constant nuclear-spin multiplier. Spectroscopy is exactly insensitive to that multiplier; low-temperature thermodynamics is not. An independent finite product Hilbert-space exchange projector can compute the physical trace.

Overlap/rank: no ortho/para or spin-isomer author was found. However geometric-rotor, hard-core ring/coherence and paired-spin tasks already exercise allowed quantum sectors or statistics. This is also a familiar textbook correction. The calibration is rich spectroscopy but not the same thermodynamic readout, making it a weaker recommendation. No numerical margin or empirical difficulty is claimed.

Primary background: [Partition functions I](https://doi.org/10.1051/0004-6361/201527209), Astronomy & Astrophysics (2016), discusses normal/equilibrium/ortho/para hydrogen. The indexed primary text was read; the full publisher open returned HTTP 403. An exactly defined rigid-rotor task would not claim this approximation is quantitatively exact for real hydrogen at all temperatures.
