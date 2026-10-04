# Quantum calorimetry, revision 1

The supplied implementation computes the complete position and momentum Gibbs covariances of a Drude-damped quantum oscillator. It differentiates the oscillator's bare energy with respect to temperature. The apparatus measures a different quantity: the heat-capacity difference between the full attached oscillator and reservoir, and the same bare reservoir. Both quantities agree exactly at zero coupling, which is the calibration setting.

The public Hamiltonian includes the counterterm, the spectral-density normalization, full Gibbs preparation and fixed controls. The matched blank and finite-device subtraction order define the measured quantity. The instruction permits changing any implementation. No public text prescribes a spectral-shift formula or a solver. Calibration uncertainty is a fixed instrument value, independent of the unknown frequency and noiseless response.

## Physical derivation

Use hbar=k_B=mass=1, oscillator frequency w0, damping gamma and cutoff Omega. The retarded susceptibility is

    chi(w) = [w0^2-w^2-i*w*gamma*Omega/(Omega-i*w)]^(-1).

The exact local variances are integrals of Im(chi)/pi times coth(w/(2T)) and w^2*coth(w/(2T)). Their temperature derivatives are also evaluated directly in the supplied source. The completed shortcut reports

    C_local = (d<p^2>/dT + w0^2*d<q^2>/dT)/2.

For each finite reservoir, the measured capacity is the sum of oscillator heat capacities of all coupled normal modes minus the corresponding sum for the bare reservoir. In the continuum limit the change in mode density is

    xi(w) = Im[d log(chi(w))/dw]/pi,
    C_excess = integral_0^infinity xi(w)*C_osc(w,T) dw,
    C_osc(w,T) = (w/T)^2 exp(-w/T)/(1-exp(-w/T))^2.

This mode-density shift includes changes to the reservoir spectrum. The independent private reference builds a finite positive arrowhead stiffness matrix from the specified spectral density, diagonalizes it and subtracts the bare-mode capacities. It does not use the susceptibility or the continuum density-shift formula.

At gamma=0 both readouts reduce to C_osc(w0,T). This is strictly decreasing with positive w0 at every fixed positive T, so calibration identifies the sole unknown. The low-temperature continuum result is pi*gamma*T/(3*w0^2), and the high-temperature limit is one.

The primary background is Ingold, Hänggi and Talkner, [Specific heat anomalies of open quantum systems](https://doi.org/10.1103/PhysRevE.79.061105), Physical Review E 79, 061105 (2009), with [author manuscript](https://arxiv.org/abs/0811.3509). The apparatus and finite-bath reference above fix the observable independently of terminology for subsystem thermodynamics.

## Validation and limits

Calibration contains 144 records with true frequency 1.04 and fixed sigma=0.0004. The calibration seed is 380197 and the independent noise seed is 380201. Public and private calibration copies match exactly. The checked-in fit is 1.04001609675 with reduced chi-square 0.999807.

The four hidden groups vary temperature, cutoff, coupling, and uncoupled anchor preparations. Group error is sqrt(mean((prediction-reference)^2)/mean(reference^2)); its limit is 0.04. The parameter limit is 3%, and reduced calibration chi-square must be below 1.5. The nominal oracle's maximum group error is 0.0000166; the shortcut errors are 0.38513, 0.29564 and 0.31288 in the three coupled groups. Both controls pass the uncoupled anchors.

All 256 noise realizations pass calibration and parameter recovery. Every oracle passes all hidden groups and every shortcut fails the three coupled groups. Worst oracle error is 0.0002084 and the smallest shortcut error is 0.29546. The fitted frequencies range from 1.03981169 to 1.04020080. These checks examine calibration noise for the fixed hidden experiments, rather than an exhaustive random sample of every allowed preparation.

The private finite-bath reference agrees with the oracle within 1.13e-7 on all hidden points. Increasing its quadrature order changes results by at most 4.24e-11; extending the reservoir tail reduces the maximum continuum error to 1.42e-8. Public-domain corner comparisons give error below 1.72e-8. The finite-bath projection onto the original oscillator independently verifies the shortcut's local-energy derivative within 6.37e-8. Susceptibility passivity, covariance uncertainty, density-shift normalization, exact calibration equivalence, endpoint parameter recovery and the low/high-temperature limits also pass.

Independent teammate checks use both refined finite Hamiltonians and a Matsubara partition product differentiated through complex Drude roots. The latter agrees with the continuum oracle within 1.22e-15. The finite-Hamiltonian check independently differentiates its thermal energy and checks spectral normalization.

Local isolated pytest gives oracle 8/8 and shortcut 5 passes plus the three intended hidden failures, in about 0.5 seconds per control. Docker controls and model trials are separate and pending at this source handoff. Scientific separation alone is not evidence of Luna difficulty.

One pre-freeze numerical edit briefly placed a scaled integration variable in the wrong helper. A concurrent root review caught a discrepancy. The edit was immediately corrected, restoring the unscaled covariance integral and using w=T*x only in the heat-capacity integral. The unchanged independent root check then passed. Calibration generation, the 256 validation and local controls used the corrected source; no model evaluation used the intermediate file. This resolved diagnostic is retained in the source-provenance report.

Reports: [science](../../results/quantum-calorimetry-validation.json), [local controls](../../results/quantum-calorimetry-local-controls.json), [source provenance](../../results/quantum-calorimetry-source-provenance.json), [root review](../../results/root-hardening-physics-review.json), and [materials review](../../results/materials-hardening-physics-review.json).

Reproduce with `uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python scripts/validate_quantum_calorimetry.py`. Add `--generate` only to intentionally regenerate both calibration copies.
