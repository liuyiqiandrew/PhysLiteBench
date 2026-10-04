# Dressed photodetection, r1

This candidate tests the physical meaning of a photon count after the coupled equilibrium state has already been solved correctly. The supplied model computes the exact Gaussian Gibbs covariances and subtracts the exact interacting-ground-state bare occupation. It is positive, produces no counts at zero temperature, and fits every uncoupled calibration experiment. Those properties do not make it the count rate of the specified absorber.

With first bare frequency one, the coordinate stiffness matrix is K=[[1,2g sqrt(w)],[2g sqrt(w),w²]]. Write K=O diag(Omega_j²) O^T and n_j=(exp(Omega_j/T)-1)^-1. The full position and momentum covariances are O diag((n_j+1/2)/Omega_j) O^T and O diag((n_j+1/2)Omega_j) O^T. The starter predicts

    R_short / detection_rate = sum_j O_aj² * (1/Omega_j + Omega_j) * n_j / 2.

The detector couples through X=a+a†. For each full-system downward transition, the amplitude comes from the complete X matrix element. Its count rate is

    R / detection_rate = sum_j O_aj² * n_j / Omega_j.

The two contributions to X interfere in each transition. Vacuum subtraction removes a constant occupation but does not change those transition strengths. The apparatus fixes the operator, absorber ground preparation, spectral normalization, measured number of events, and weak-measurement order. No removal of the internal coupling or adiabatic conversion of dressed excitations into bare photons is performed. The public description does not supply the positive-frequency decomposition or either rate formula.

The Hamiltonian is the specified ideal quadratic two-mode system, not a claim that arbitrary electromagnetic materials lack additional self-energy terms. Its determinant is w²-4g²w, positive over |g|<=.4sqrt(w). The allowed normal-mode frequencies are approximately .4677 to1.9861, strictly within the detector's [.2,2.5] flat-response band. A finite band avoids an unnecessary ultraviolet detector model. Rates are obtained at leading order before population depletion, so the detached preparation bath need not sustain a nonequilibrium steady emission state.

Calibration has144 independent records at g=0, six temperatures, three second-mode frequencies and eight repetitions. Both models then give R=detection_rate/(exp(1/T)-1). The single parameter is identifiable by weighted linear fitting throughout [.008,.02]. Gaussian instrument sigma=2e-6 is fixed independently of the parameter and output; seed948101 intentionally generates the public/private identical data. Noise validation uses seed948103 and256 independent realizations. Hidden groups vary interaction, temperature and second frequency, with the same3% relative group error gate used for this task. Their source-control errors are .2931, .3201 and .3212; fitted-oracle errors are .0001645. Across256 fits, all calibration, parameter and oracle gates pass; no shortcut passes. The minimum shortcut group error is .2927 and maximum oracle group error is .000474. These are controlled simulation checks, not agent measurements.

The private reference constructs the Hamiltonian in a tensor product of two bare Fock bases, diagonalizes it, constructs every X matrix element, and directly sums the Gibbs-weighted energy-lowering transitions within the detector band. It does not use the analytic rate or a Bogoliubov transformation. All scored cases at cutoff36 agree with the oracle within5.92e-13 before the fitted scale, and36→44 changes at most5.94e-13. Full-range temperature/coupling/frequency corners checked at44→52 differ by at most1.80e-10; cutoff52 differs from the oracle by at most1.65e-12. The same finite-Fock states independently reproduce the starter's vacuum-subtracted occupation, confirming that its mathematical state calculation is correct. Ground-state absorption, interaction sign symmetry, positivity, stability, band coverage and parameter recovery are also checked.

Commands:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python scripts/validate_dressed_photodetection.py
```

Add `--generate` only to intentionally replace both calibration copies using the fixed seed. Reports are `results/dressed-photodetection-validation.json`, `results/dressed-photodetection-local-controls.json` and `results/dressed-photodetection-source-provenance.json`. No Luna evaluation had been launched at science freeze. The standard neutral instruction permits replacing every implementation helper.

The microscopic absorption interpretation follows the primary derivation in Di Stefano et al., [Photodetection probability in quantum systems with arbitrarily strong light–matter interaction](https://www.nature.com/articles/s41598-018-36056-1), Scientific Reports8,17825(2018), [arXiv:1711.10698](https://arxiv.org/abs/1711.10698). The concrete harmonic apparatus and controls here are independently constructed.
