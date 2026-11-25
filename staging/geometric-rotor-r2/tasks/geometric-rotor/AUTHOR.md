# Geometric rotor, revision 2

This revision replaces the two-level internal texture with a four-level Hamiltonian whose low-energy internal space is a degenerate doublet. It keeps the operational measurement of rotational kinetic energy and the finite-gap-equilibration-then-infinite-gap limit. Revision1 scored3/3 physical successes and is preserved separately; this is a new Hamiltonian, calibration, completed shortcut and hidden experiment set, not a retry of unchanged inputs. The exact neutral instruction and standard runtime limits remain unchanged.

The source is a complete local geometric-energy approximation. It constructs the specified four-dimensional frame U(phi), differentiates it, projects its derivative perpendicular to the lower doublet, and computes the full two-by-two positive local matrix. It solves the resulting Fourier Hamiltonian, retaining the correct periodic or antiperiodic endpoint condition of that specified frame. All apparatus controls affect the calculation. The omission is transport within the degenerate doublet around the guide. That transport is generally noncommuting at different positions and changes the global rotor sectors.

Let V be the first two columns of the supplied U. Direct projection gives the Hermitian connection

    A(phi) = [q cos(r phi) sigma_x - q cos(theta) sin(r phi) sigma_y
              + r cos(theta) sigma_z]/2

and local geometric energy

    Phi(phi) = sin(theta)^2 [q^2 sin(r phi)^2 + r^2]/(8 I) * identity_2.

The projected kinetic Hamiltonian is (-i*d/dphi-A)^2/(2I)+Phi. The source keeps Phi and omits A. It is an exact implementation of a positive scalar-matrix Hamiltonian with a valid canonical ensemble. The full source matrix agrees with an independently implemented scalar Fourier potential within2.64e-13 across the validated domain.

The frame satisfies U(2pi)=(-1)^(q+r)U(0). Removing A locally with W'=i A W leaves the scalar wavefunction boundary condition

    chi(2pi)=(-1)^(q+r) W(2pi)† chi(0).

The two SU(2) eigenphases are conjugates. Since Phi is a real scalar potential, both phases produce the same spectrum; the doublet degeneracy cancels from the canonical mean. The oracle integrates the two-by-two transport equation and diagonalizes the scalar Fourier Hamiltonian with its resulting twist. In the strict gap limit the lower internal energy is zero, and the projected total energy equals the limiting physical kinetic energy. This equality does not hold at finite gap; the independent verifier evaluates the original kinetic operator there.

At calibration theta=pi/2, A=(q/2)cos(r phi)sigma_x, so its ordered transport over a full circle is identity. The remaining central parity sign is already included in the source. Thus calibration covers nonzero spatial textures and both endpoint parities while the two models agree exactly. The unknown inertia is identified from the varied canonical energy curves. Intermediate tilts expose the missing doublet transport.

The source approximation is tied to the supplied frame and is not generally gauge invariant. In particular, at theta=0 the physical projector is constant. The correct model reduces to a free periodic rotor, whereas the endpoint-only source can retain an antiperiodic condition when q+r is odd. Validation records this defect as a consequence of its closure; endpoint parity alone is not presented as a physical invariant. The zero-texture diagnostic is outside the scored interval and is not used as a hidden near-zero-signal test.

The public apparatus gives the original laboratory Hamiltonian, the complete periodic four-component wavefunction, unrestricted canonical preparation and the kinetic detector. It does not prescribe a projection, transport calculation, gauge, or numerical method. It defines the order of gap and thermal limits because that order determines the measured state. A direct laboratory calculation remains a valid solution.

The private reference uses matrix exponentials of the original four-dimensional rotations to sample U Q0 U†, then builds the full laboratory Fourier Hamiltonian. It diagonalizes at gaps1024,2048,4096, evaluates kinetic energy with the full eigenvectors, and extrapolates in inverse gap. It shares no connection or projected-potential formula with the oracle. Across108 public-domain parameter corners, reference error is below6.34e-7. Doubling the gap triplet changes predictions by at most1.50e-7, and Fourier cutoff18→24 by1.37e-11. Direct frame differentiation verifies A and Phi within4.45e-16. The finite commutator diagnostic is nonzero; both completed Hamiltonians remain positive.

There are288 independent calibration records with fixed sigma0.0005, true inertia1.07, data seed993041 and noise seed993043. Constant instrument uncertainty carries no noiseless-response information. Calibration curves decrease throughout the allowed inertia interval [.8,1.2]. A161-point global objective check has one minimum at each of three separated true inertias, and noiseless recovery agrees within8.91e-11, including both endpoints. Oracle and source calibration agree within1.88e-13.

Both completed controls fit inertia1.06990728 and reduced chi-square1.07172. Prediction error is group RMSE divided by the group's true RMS kinetic energy, with the unchanged0.04 threshold. Scored energies range from0.0686 to0.2454. Nominal oracle error is below7.87e-5; shortcut diagnostic errors are0.6530,0.6683 and0.5918. Both pass the additional central-holonomy anchors. Across256 independent noise draws, every calibration/parameter/oracle test passes, every shortcut fails, oracle error stays below0.0001326, and shortcut diagnostic error remains above0.5914.

Isolated local verification gives oracle8/8 and shortcut5passes plus3intended hidden failures, each in about0.6seconds. The first validator run completed the scientific assertions but failed while serializing a NumPy integer; correcting only that report conversion allowed the report to be saved. No source, data, or grading changes were needed. These are scientific controls, not evidence of Luna difficulty.

Reports: [scientific validation](../../results/geometric-rotor-r2-validation.json), [local controls](../../results/geometric-rotor-r2-local-controls.json), [source provenance](../../results/geometric-rotor-r2-source-provenance.json). Both preliminary prototypes and their results are preserved under the staging bundle's `prototype/` and `results/`. The preliminary same-parity-only design had weak signal separation; the final source explicitly retains arbitrary frame endpoint parity and gives the larger, independently verified transport effect.

The physical mechanism extends [Wilczek and Zee, Appearance of Gauge Structure in Simple Dynamical Systems, Physical Review Letters52,2111(1984)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.52.2111), which establishes non-Abelian adiabatic transport in degenerate internal spaces. The particular four-level guide, calibration and measured kinetic energy are defined by the task and independently derived here. It is a hardening of the geometric-rotor family, not a claim of an unrelated physical principle.

```bash
uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python scripts/validate_geometric_rotor.py
```

Use `--generate` only to intentionally regenerate both calibration copies. Root owns Docker and model evaluation after promotion; no model trial is part of this source freeze.
