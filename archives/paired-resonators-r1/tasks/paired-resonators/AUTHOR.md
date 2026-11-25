# Paired resonators, revision 1

This task separates solving the linear equations of motion from quantizing their normal modes. The supplied code has the full correct dynamical matrix, all positive eigenfrequencies, both components of every eigenvector, and exact Bose populations. It treats Euclidean-normalized dynamical eigenvectors as canonical bosonic mode amplitudes when reconstructing local thermal fluctuations. That physical identification fails although the eigenvalue problem is solved correctly.

The public apparatus is an exact passive, time-independent three-mode quadratic network. Its fixed physical quadratures obey canonical commutators, its two real quadratic forms are positive, and its equilibrium ensemble is explicitly Gibbs. The output is a difference of local quadrature variances measured on separate equilibrium copies, with the same interacting Hamiltonian at finite and zero temperature. It is not defined through a weak optical absorber, an implicit drive frame, or a change in coupling during detection. The standard neutral instruction permits any implementation preserving the API. No canonical normalization formula or diagnosis appears in agent-visible files.

## Physical distinction

In the fixed local creation-operator basis, the Hamiltonian is energy_scale times

    a^dagger A a + pairing/2*(a^dagger B a^dagger + a B a)

up to a constant. Its exact first-order dynamical matrix is

    D = [[A, pairing*B], [-pairing*B, -A]].

The positive eigenvalues give resonance frequencies after multiplication by energy_scale. Their values do not depend on eigenvector normalization. Frequency spectroscopy therefore calibrates the scale exactly even when pairing is active, and cannot expose the supplied interpretation of eigenvector amplitudes.

For a positive-frequency right eigenvector w=(u,v), preserving the bosonic commutators requires w^dagger Sigma_z w=1, where Sigma_z=diag(I,-I). The columns form a canonical transformation, with U^dagger U-V^dagger V=I and the corresponding mixed-block relations. The local thermal-excess variance is

    sum_l (|U_jl|²+|V_jl|²) n_B(energy_scale*omega_l/T).

The source inserts eigenvectors returned with ordinary Euclidean norm one into this last expression. Its amplitudes are positive and finite and its frequencies are exact, but its proposed ladder modes are not canonical operators of the physical network. At zero pairing the two normalizations coincide. Subtracting the interacting ground-state measurement removes the zero-point contribution in both controls, so no simple nonzero-ground-offset issue identifies the failure.

This is a physical model error when an agent treats classical dynamical mode amplitudes as canonically quantized boson amplitudes. It should not be recorded as a physics failure when an agent derives the canonical transformation correctly and only implements that normalization incorrectly. The archived Bogoliubov-momentum task instead used correctly normalized excitations and confused particle depletion with conserved total momentum; the observable and missing principle here are different.

## Independent verification

The verifier stays in real physical quadratures. It makes the kinetic quadratic form P=A-pairing*B into a unit matrix through q=P^(1/2)y, p=P^(-1/2)pi, then diagonalizes P^(1/2)Q P^(1/2), where Q=A+pairing*B. It reconstructs the position and momentum thermal variances independently, using n_B/omega and n_B*omega respectively. It never constructs the complex dynamical matrix or normalizes its eigenvectors.

The author validator also constructs the full three-mode Hamiltonian directly in a finite bare Fock basis, using creation and annihilation matrix elements. It splits total-number parity to reduce the sparse matrices, diagonalizes each block, and takes thermal traces of each local occupation followed by ground-state subtraction. At pairing .6, total-number cutoffs 18 and30 converge to the covariance reference within 1.5e-14 at temperatures .25 and .45. At maximal pairing, cutoffs30,44,60 reduce the largest discrepancy from .0002193 to .000002091 to 8.13e-9. Increasing retained eigenstates from120 to180 per parity block changes the final result by 1.16e-12. These Fock checks are author-side and cover moderate/maximal pairing at the specified two temperatures; the independent exact quadrature verifier covers the entire public temperature and scale range.

Across the supported corners and256 random settings, the oracle/reference discrepancy is at most 7.11e-15 absolute and3.40e-14 relative. Canonical commutators and diagonalized Hamiltonian relations hold within1.1e-14. The smallest quadratic eigenvalue before applying energy_scale is .21103, the smallest frequency .83632, and the smallest positive signature norm .47698. Thermal excess is nonnegative, zero-temperature output is exact, and the unpaired limit agrees with the source. The high-temperature slope agrees with the classical quadratic Gibbs covariance within1.2e-9.

Root's separate6x6 matrix-function Gaussian reference agrees with the oracle to2.3e-15 and the verifier to2.2e-16 over24 random settings. It confirms that the raw source eigenvectors solve their equations within4.7e-15 while failing the canonical commutators by as much as.474. This review is saved in results/root-hardening-physics-review.json, key paired-resonators-r1.

The canonical nature of bosonic diagonalization is described in [Nam, Napiórkowski and Solovej, Diagonalization of bosonic quadratic Hamiltonians by Bogoliubov transformations](https://arxiv.org/abs/1508.07321), especially the definition of transformations preserving the commutation algebra. The classic finite-mode reference is Colpa, Physica A93,327–353(1978), DOI10.1016/0378-4371(78)90160-7. The task's formulas follow directly from its explicitly specified finite Hamiltonian.

## Calibration, controls and reproducibility

There are288 independent spectroscopy measurements:16 repeats of all three branches at six pairing settings, including strong pairing. Instrument sigma=.001 is constant and independent of response or unknown scale. The true scale is1.04 within[.8,1.3]. Calibration seed is870621 and noise seed870629. Both controls give exactly identical calibration predictions for every scale. The fit is a one-parameter strictly convex weighted quadratic with curvature2.390016e9, proving identifiability; noiseless fits recover both bounds and the nominal parameter to roundoff.

The actual fitted scale is1.0399905803 with reduced chi-square1.00047. Oracle errors are at most.00002123. Source errors are.4134,.2314,.4379, with spectroscopy/unpaired/ground anchors passing. The hidden tolerance is group RMS error divided by group RMS reference below.04. Parameter error must be below3% and reduced calibration chi-square below1.5.

All256 noisy fits pass calibration, parameter recovery and the oracle predictions. The source fails every discriminating group in every realization. Maximum calibration chi-square is1.20927, maximum relative parameter error.00008724, worst oracle hidden error.0002044 and best discriminating source error.23127. Both controls return identical fitted scales across all noise draws. Local verification gives oracle8/8 and source5passes/3intended physics failures in under one second each. Agent/verifier limits remain600/60seconds.

Evidence is in results/paired-resonators-validation.json. Run `uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python scripts/validate_paired_resonators.py`; `--generate` intentionally updates both calibration copies, and `--skip-fock` omits the extra author-side Fock convergence check. The source control is scripts/paired_resonators_baseline.py; the independent Fock helper is scripts/paired_resonators_fock.py. Frozen provenance is recorded in results/paired-resonators-source-provenance.json. No model difficulty outcome is claimed before evaluation.

## Evaluated outcome

The neutral-instruction conditional screen produced three genuine physical successes: initial trial paired-resonators__VP2bdUj and follow-ups paired-resonators__XgJtmKU and paired-resonators__xXac84x. Two solutions reconstructed the canonical real-quadrature covariance; one normalized the bosonic dynamical eigenvectors with the commutator metric. All three fit energy_scale=1.0399905803 and pass every hidden group, with maximum group error below2.13e-5. Oracle and completed shortcut controls gave1/1 and0/1 respectively. There were no infrastructure exceptions.

The complete native evidence and all three reviews are preserved under archives/paired-resonators-r1. This family is archived because the observed3/3 success rate misses the difficulty target. The initial success is included in that aggregate. No agent-visible or scientific file changed after evaluation; only this author outcome note was appended.
