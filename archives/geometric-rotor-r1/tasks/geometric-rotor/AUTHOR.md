# Geometric rotor, revision 1

This staged candidate has not been evaluated by a model. It preserves the exact neutral instruction, broad implementation permission, and the standard 600-second agent / 60-second verifier limits.

The apparatus gives the full periodic laboratory-basis spinor Hamiltonian and an unconstrained global Gibbs preparation. It measures rotor kinetic energy, equilibrating at finite internal gap before taking that gap to infinity at fixed inertia and temperature. The reservoir can exchange angular momentum, so a fixed angular sector is not implied. This limit order is part of the physical experiment. Kinetic energy and total energy plus Delta agree only in the stated limit, not at finite Delta.

## Physical approximation

The completed shortcut retains the exact local lower internal energy and the diagonal geometric kinetic term. It models the remaining motion by an ordinary periodic scalar rotor with energies

    E_source(n) = n^2/(2*I) + q^2*sin(theta)^2/(8*I),  n integer.

This is a positive scalar Hamiltonian with a correctly computed canonical mean. Its omission is the phase acquired by the position-dependent internal state around the guide. The full periodic spinor fixes the effective angular sectors. In one periodic gauge the lower internal eigenvector is

    chi(phi) = (−sin(theta/2)*exp(−i*q*phi), cos(theta/2)).

Projecting the kinetic operator gives the connection A=q*(1−cos(theta))/2 and scalar Phi=q^2*sin(theta)^2/(8*I). The physical limiting energies and kinetic eigenvalues are

    E(n) = (n−A)^2/(2*I) + Phi.

Integer changes of A amount to relabeling n. At theta=pi/2 and even q, A is integer, so every calibration curve is exactly identical in both models while Phi remains nonzero. Thus all supplied controls affect the starter and the calibration includes a genuine internal texture. Odd windings or tilted textures generally have noninteger A and distinguish the models. The hidden readout is a positive kinetic energy, not a gauge-dependent angular label.

This mechanism was not found in the archive scan. The archived boson-Hall family uses Berry curvature but misses energy magnetization in transport. The rigid-linkage family tests a classical canonical measure. The superconducting-ring family preserves the winding sector correctly and omits carrier kinetic stress. Those are related geometric or constrained systems, but they do not test the global boundary condition induced by adiabatic internal-state transport.

## Independent reference

The verifier uses the original finite-gap Hamiltonian in laboratory Fourier states |up,n−q> and |down,n>. Each block is

    [(n−q)^2/(2I)+Delta*cos(theta), Delta*sin(theta)]
    [Delta*sin(theta), n^2/(2I)−Delta*cos(theta)].

It diagonalizes each block, forms the complete thermal population, and evaluates the original diagonal kinetic operator through the eigenvectors. It does not use a Berry connection, scalar geometric term or effective angular-sector formula. Values at Delta=1024,2048,4096 are extrapolated in inverse gap. Doubling all three gaps tests this limiting calculation. An additional reference check diagonalizes an unsplit laboratory Fourier matrix.

Across 164 public-domain corner and off-grid controls, the oracle and extrapolated kinetic reference differ by at most 1.04e−8. Gap refinement changes predictions by at most 9.10e−9; raising the Fourier cutoff changes them by 4.5e−16. Eight full-matrix checks agree with the block trace to 2.28e−14. The zero-texture, integer-gauge, low-temperature half-flux and high-temperature limits also pass. These errors are far below the prediction threshold and instrument uncertainty.

## Calibration and controls

The unknown inertia lies in [.8,1.2], with private value 1.07. There are 288 independent energy records: 36 repetitions of eight controls. Sigma=.0005 is a constant instrument uncertainty, independent of the clean response and inertia. Data seed931471 and independent noise-validation seed931479 are fixed. The scalar calibration objective has one minimum across the entire allowed range at each of three separated true inertias; all calibration energies decrease monotonically over that range. Noiseless recovery agrees to1.1e−9, and oracle/shortcut calibration equivalence is4.5e−16.

Both controls fit inertia1.0700355233 and reduced chi-square.95134 on the checked-in data. The oracle's largest hidden normalized RMS error is3.27e−5; the shortcut's three diagnostic group errors are.4265,.2139,.2693. Both pass extra even-winding anchors. Across256 independent noise draws, every calibration, parameter and oracle check passes and every shortcut fails. The worst oracle error is6.44e−5, and the smallest shortcut diagnostic error is.21383, against the.04 group threshold. No tolerance was tightened to force separation.

Local isolated pytest: oracle8/8; shortcut5passed and3intended hidden failures. Both take less than.4seconds. These are scientific/local controls, not evidence of Luna difficulty. Docker controls and model trials remain the parent controller's responsibility after freeze.

Reproduce the staged checks from the repository root:

```bash
uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python staging/geometric-rotor-r1/scripts/validate_geometric_rotor.py
```

Use `--generate` only for intentional regeneration of both calibration copies. After promotion, use the validator under `scripts/`. Reports: [science and local controls](../../results/geometric-rotor-validation.json), [provenance](../../results/geometric-rotor-provenance.json), and separate source/physics peer review. The earlier prototype remains in repository `scripts/prototype_geometric_rotor.py` and `results/geometric-rotor-prototype.json`.

## Sources and remaining uncertainty

The role of a vector potential in adiabatic nuclear motion is established by Mead and Truhlar, [On the determination of Born–Oppenheimer nuclear motion wave functions including complications due to conical intersections and identical nuclei](https://doi.org/10.1063/1.437734), Journal of Chemical Physics70,2284(1979). Mead's [The geometric phase in molecular systems](https://doi.org/10.1103/RevModPhys.64.51), Reviews of Modern Physics64,51(1992), reviews the geometric connection and its global consequences. The particular two-level guide, calibration and kinetic detector here are independently specified ideal models; a molecular conical intersection is not assumed.

A solver may recognize the geometric phase or diagonalize the finite-gap spinor directly. Scientific separation does not establish a low empirical pass rate. No difficulty claim is made before evaluation.

## Evaluated outcome

The neutral-r1 conditional screen completed with3/3 genuine physical passes: initial `hyrm2gy`, followed by `AYLKmco` and `dN75fBo`. All three submissions derive the parity-dependent rotating-spinor boundary condition and the tilt-dependent angular momentum shift. Their spectra are equivalent to the correct geometric connection plus scalar term. Each passes all8 verifier checks, with fitted inertia about1.070035523 and maximum hidden normalized RMS error3.27e−5. Interim code and patch errors were fixed before final submissions.

The task missed the difficulty target and was archived. Both Docker controls, all three model trials, complete native transcripts, staging/prototype, source reviews and frozen provenance are retained. Earlier unevaluated labels describe the original freeze; this outcome paragraph was added only to the archived task AUTHOR. All job and staging sources are unchanged.
