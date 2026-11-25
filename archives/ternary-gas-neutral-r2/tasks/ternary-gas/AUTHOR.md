# Ternary-gas revision 2

Revision 2 tests whether local collision loads can be replaced by those of the conserved mean mixture. The completed shortcut now solves the full coupled Maxwell–Stefan friction system and enforces zero total molar flux. Its approximation is to evaluate the composition factors in that matrix at the global mean composition, rather than at the evolving face composition. This is a linear-response approximation about a homogeneous mixture, not the mixture-averaged scalar diffusion model from revision 1.

For normalized fluxes j_i, the physical equations are

    -grad(x_i) = sum(j != i) (x_j*j_i - x_i*j_j)/D_ij,
    sum_i j_i = 0.

The binary diffusion constants are material constants, but the numbers of local collision partners depend on local mole fractions. Both models conserve every species inventory, keep the fraction sum equal to one, and have the same uniform equilibrium. In a binary system, the constrained equations reduce to Fick diffusion with its binary coefficient for every composition. Binary calibration therefore remains exactly equivalent under both closures.

The frozen-background model has positive dissipation in the quadratic ideal-mixture free energy around its reference composition: one-half integral sum_i (x_i-mean_i)²/mean_i. Its Onsager metric is the diagonal matrix of inverse background fractions. It approaches the full model to first order in small perturbations. At finite amplitudes it need not preserve componentwise positivity for every conceivable preparation; the selected hidden profiles and fitted-parameter extrema remain positive, so the measured separation is not a positivity failure.

The public apparatus, neutral instruction, complete calibration data, oracle equations, and grading thresholds are unchanged. The new hidden profiles have stronger positive spatial composition variation, including unequal spatial harmonics, so the local-versus-mean friction distinction exceeds the same normalized error threshold of 0.04. Errors are normalized by RMS departure from the conserved uniform composition.

The physical oracle solves a constrained three-by-three system at each face. The independent verifier eliminates the third flux analytically and evolves two fractions on a fourfold refined grid. The reconstructed third fraction enforces the same total-density condition by a different implementation. Eightfold reference refinement changes outputs by at most 1.992e-5; the oracle/reference difference at the true diffusivity is at most 0.000399 in mole fraction.

The private diffusivity remains 0.035 mm²/s. The original fixed calibration uncertainty is 0.002, calibration seed 9400, and independent-noise seed 19400. Both controls fit 0.0351489263, with reduced chi-squared 0.985461. All 256 noise realizations pass calibration and parameter checks; maximum diffusivity error is 0.646%. Fixed uncertainty contains no noiseless-response side channel.

Nominal oracle hidden errors are 0.001235–0.001385; shortcut errors are 0.07689, 0.08278, and 0.11768. Across fitted-parameter extrema, oracle error stays below 0.002339 and shortcut error above 0.07689. Both controls conserve each species to 3.34e-16 and total fraction to 1.34e-15. Minimum checked fraction is 0.0394. The physical mixing free energy decreases, and the shortcut's background quadratic free energy also decreases. For perturbations scaled down by 0.001, the controls agree within 2.54e-8, validating the interpretation as a linear-response closure. Uniform stationarity and spatial reversal checks pass.

Local isolated controls give oracle 6/6 passing tests in 0.85 seconds. The shortcut passes the three public/private calibration and parameter checks and fails all three intended hidden groups in 0.65 seconds. See [the scientific and local-control report](../../results/ternary-gas-r2-validation.json) and [independent source review](../../results/brownian-cross-review-ternary-r2.json). Revision-2 agent evaluation is pending; no difficulty claim follows from these controls alone.

The complete preceding task, controls, validator, science and trial reviews are preserved in [the neutral-v1 archive](../../archives/ternary-gas-neutral-v1/README.md). Revision 1 scored 0/3 plain and 3/3 hinted with the original instruction, then 3/3 plain with the neutral instruction. Those distinct batches and every initial pass remain preserved.

Reproduce with:

```bash
uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python scripts/validate_ternary_gas.py
```

Ordinary validation leaves data unchanged; `--generate` intentionally rewrites both calibration copies. Primary background: [Duncan and Toor's ternary diffusion experiment](https://aiche.onlinelibrary.wiley.com/doi/10.1002/aic.690080112) and [Boudin, Grec, and Salvarani's Maxwell–Stefan analysis](https://mate.unipv.it/~salvaran/BoudinGrecSalvarani.pdf). This task uses its own ideal apparatus and generated measurements.

The completed neutral-instruction revision-2 screen passed 3/3 unhinted Luna-high trials: the initial success and both required follow-ups. All three submissions restore locally composition-dependent Maxwell–Stefan friction, rather than retaining the fixed-background approximation. These are substantive physical solutions and exceed the requested pass-rate target. Every trial is retained in [the reviewed evidence](../../results/neutrality-materials-trial-reviews.json).
