# Prestrained solid, revision 1

The experiment measures bulk wave speeds in a finitely deformed compressible elastic solid. The supplied model retains the complete material Green-strain tangent, finite deformation, current density, directional projection and eigenmode calculation. Its physical approximation is to use that pushed-forward material tangent as the whole incremental acoustic operator. This omits the restoring force associated with initial stress.

Let `E = (F^T F - I)/2`, `S = dW/dE`, and `C = dS/dE`. A displacement-gradient perturbation changes both factors of `P = F S`. Consequently the Cartesian Hessian `dP/dF` includes `delta_ij S_AB` in addition to the material-tangent term. After spatial projection, the correct acoustic matrix is the supplied matrix plus `(n^T sigma n) I`. The oracle differentiates the nominal stress with respect to the full deformation gradient. The private reference instead evaluates directional second variations of the scalar stored energy for three displacement polarizations and their sums; fourth-order differences and Richardson extrapolation avoid sharing the oracle's tensor derivatives.

The energy's infinitesimal Lamé constants are `mu = modulus` and `lambda = 2.8 modulus` in MPa. The reference density is 1000 kg/m³; the current density is `rho0/det(F)`. The unit wavevector is spatial, so its material counterpart is `F^T n`. These conventions are public apparatus definitions.

Calibration uses six uniaxial tensile stretches, three transverse directions, all three branches and four orientations. The transverse stretches solve zero transverse Cauchy stress. Thus all 216 readings exactly share the correct and supplied wave speeds, even though the largest axial initial stress is about 1.02 times the modulus. The positive modulus is uniquely identified: every speed is proportional to its square root. The fixed 0.02 m/s instrument uncertainty is independent of the noiseless response and carries no response side channel.

Hidden preparations use axial or oblique propagation and other finite deformations inside the public domain. The completed shortcut has 17.4–20.4% relative root-mean-square error, while the correct model has about 0.007% error from the noisy calibration. The 3% speed threshold is well above numerical and calibration uncertainty. All 256 independent noise realizations pass parameter/calibration checks; all physical controls pass and all shortcut controls fail the three prediction groups.

Both operators remain positive. For the shortcut, with `B=F F^T` and `a=1.4-2 log J`, its reference-density-scaled matrix is `a I + (2+a) nn^T + 0.4[(Bn)(Bn)^T-(n^T B n)B]`. Its minimum current acoustic eigenvalue is bounded below over the complete public domain by `[1.4-2 log(1.18)-0.4*1.18^2/0.85^2]/1.18 > 0.2526`. The validator also checks 237 allowed strains/orientations against the independent energy Hessian, refinement, frame covariance, direction reversal, the unstressed limit, exact stress correction, rigid-rotation energy invariance and its second-variation balance. Maximum independent speed disagreement is below 2e-7 m/s.

This is distinct from archived piezoelectric wave and coherent-alloy tasks: the missing contribution comes from finite-deformation geometry and initial mechanical stress, not an electrical elimination or a nonlocal elastic compatibility constraint. No difficulty claim has yet been measured. A retained material-tangent approximation is a physical model failure; an incorrect implementation after deriving the full displacement-gradient Hessian must be classified separately.

The neutral task instruction is unchanged. The public source is a completed, calibrated shortcut, with no unused physical-control inputs or corrective comments. Private `hint.md` is excluded from ordinary agent images.

Primary background: Shams, Destrade and Ogden, *Initial stresses in elastic solids: constitutive laws and acoustoelasticity*, Wave Motion 48 (2011), 552–567, particularly the incremental nominal stress and push-forward in section 3 ([author paper](https://maths.nuigalway.ie/~destrade/Publis/destrade_69.pdf), [arXiv](https://arxiv.org/abs/1302.1961)). The present scalar-energy reference independently establishes the task's specific constitutive law and normalization.

Author reproduction from the staging root:

```bash
uv run --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python scripts/validate_prestrained_solid.py
```

Use `--generate` only to intentionally recreate public/private calibration together. Scientific and isolated local verifier reports are in `results/prestrained-solid-r1-validation.json` and `results/prestrained-solid-r1-local-controls.json`. Evaluation remains pending root-controlled frozen runs.
