# Poroelastic relaxation, revision 1

The supplied model solves pressure relaxation using the exact stress-free uniform storage and full anisotropic Darcy conductivity. Its physical approximation is to permit every local volume element to relax independently to zero total stress. Uniform calibration cannot distinguish that local-compliance model from the actual affine-plus-periodic elastic solid. Nonuniform pressure requires strains from one continuous displacement field, and the spatial compatibility changes fluid storage.

The public energy gives `p=M(zeta-alpha*q^T e)` and total stress `sigma=C e-alpha*p*q`, with `q=(1,1,0)` and engineering shear. The freely adjusting affine strain enforces zero mean total stress, so a uniform pressure has `S0=1/M+alpha²*q^T C^-1 q`. This is exactly the supplied storage. For nonzero wavevector `k`, a sine displacement amplitude `u` produces cosine strain `e=B(k)u`, where `B=[[kx,0],[0,ky],[ky,kx]]`. Mechanical equilibrium gives `B^T C B u=alpha*B^T q*p`; hence `S(k)=1/M+alpha²*(B^T q)^T(B^T C B)^-1(B^T q)`. Darcy flow and the distributed reservoir connections give pressure decay rate `[k^T K k+g]/[viscosity*S(k)]`.

The private reference uses a different elimination: at fixed fluid content it minimizes the original stored energy over displacement. Its elastic system is `B^T C B+M alpha² d d^T`, with `d=B^T q`, and its pressure follows from `M(zeta-alpha*q^T e)`. It evolves fluid content using drainage, then obtains the measured pressure. For the uniform mode it minimizes over the three freely adjusting affine strains. Thus the reference does not copy the oracle's pressure-controlled acoustic inverse.

Calibration uses 216 uniform-pressure readings over three initial pressures and six times. Their responses are strictly increasing in positive viscosity and determine the shared parameter. All 256 noisy fits pass calibration and parameter checks; endpoint recovery and a full bounded-objective grid are also checked. Fixed 1 Pa measurement uncertainty is independent of response. Both completed controls share calibration and uniform anchors exactly.

The final known elastic tensor has normal–shear entries 0.8 and 0.6 MPa. Every independent elastic entry affects the source's local compliance; this is checked numerically. An earlier orthotropic prototype and its unmeasured scientific controls are preserved under `prototypes/`. The final matrix, calibration and scientific source were selected before any model evaluation. The physical closure, parameter, interface and prediction tolerance are unchanged through this authoring refinement.

For the final matrix the correct storage ranges from 1.56e-7 to 3.04e-7 Pa^-1. Hidden pressures are healthy signals of 251–1190 Pa. Shortcut prediction errors are 38–59%, against a 4% threshold. Correct errors are about 0.018–0.022%; their maximum over 256 noise draws is 0.049%. Local oracle tests pass 7/7, while the shortcut passes calibration, parameter and uniform-anchor checks and fails the three nonuniform groups.

The validator checks all 25 allowed modes and 600 mode/time/amplitude/viscosity combinations, reference agreement below 8e-13 Pa, force balance, strain compatibility, initial conditions, simultaneous mode reversal, and exact Darcy-plus-leakage energy dissipation. The largest strain at any public pressure amplitude is below 0.0007, consistent with the small-strain apparatus. The source also has positive storage and dissipative decay; global fluid balance alone cannot reject it. Its nonuniform local strain violates the compatibility condition, which is the intended physical distinction.

This is related to the archived sealed homogeneous poroelastic task but is a new spatial mechanical constraint, not another version of pore-fluid bulk stiffening. It is also related in broad structure to the retained shared-field transport tasks: local constitutive laws are insufficient without mechanical equilibrium and compatibility. No model difficulty outcome is claimed before evaluation. Retaining local stress-free storage is a physical failure; a correct compatible-displacement derivation followed by a matrix or normalization error must be classified separately.

The exact neutral instruction is unchanged. The public README specifies the energy, mechanical cell, plane-strain/shear convention, preparation, fluxes and pressure detector without supplying a storage correction or requiring a solver. The starter includes a complete fit and positive pressure dynamics. `hint.md` remains private to hinted evaluation.

Primary background: M. A. Biot, *General Theory of Three-Dimensional Consolidation*, Journal of Applied Physics 12, 155–164 (1941), [DOI](https://doi.org/10.1063/1.1712886), [primary paper](https://www.math.purdue.edu/~santos/research/biot_papers/biot_1941.pdf). The specific anisotropic cell, reservoir connections and independent energy minimization are defined and verified here.

From the staging root:

```bash
uv run --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python scripts/validate_poroelastic_relaxation.py
```

Add `--generate` only to intentionally regenerate both calibration copies. Final scientific and local reports are in `results/poroelastic-relaxation-r1-validation.json` and `results/poroelastic-relaxation-r1-local-controls.json`. Root controls all frozen model evaluations.
