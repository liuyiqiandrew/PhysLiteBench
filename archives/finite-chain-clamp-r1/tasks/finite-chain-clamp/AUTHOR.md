# Finite-chain clamp, revision 1

This task tests whether a force-controlled mean-response relation can be inverted to predict the mean mechanical force from a clamp on a finite thermal chain. The supplied source solves the force-controlled preparation exactly. Its inverse is a coherent finite-chain approximation to the clamp experiment, but thermal fluctuations make the two readouts different.

No model evaluation has been run. Scientific validation and isolated local controls are complete; empirical difficulty remains unknown.

## Apparatus and measure

The public preparation uses Cartesian canonical equilibrium of beads connected by central radial springs. Transforming bead positions to bond vectors has constant Jacobian. In each bond's spherical coordinates, the radial factor is `r^2 exp[-k_s(r-b)^2/(2T)]`; as stiffness tends to infinity its normalized angular distribution is uniform. Thus each axial projection is uniform on `[-b,b]`, independently of the other bonds. This is a physically specified stiff-spring preparation, not a choice of rigid constrained Hamiltonian measure.

The clamp constrains only the endpoint x projection. Its transverse coordinates remain free. At finite clamp stiffness K, the partition factor is

    Z_K(X) = integral p_N(x) exp[-K(x-X)^2/(2T)] dx,

and the actual clamp-to-chain force is `K*(X-mean(x)) = -T*d_X log Z_K`. Taking the stated spring limit before K tends to infinity gives `-T*d_X log p_N(X)`. There is no radial `4*pi*R^2` factor for this planar constraint. The public README gives the potentials, Cartesian equilibration, ordered limits, transverse freedom and signed readout; it supplies no distribution or correction formula.

## Oracle and coherent shortcut

The exact force-mode mean is `N*b*L(F*b/T)`, with `L(z)=coth(z)-1/z`. Both source and oracle use this response for every allowed force-mode input. The source numerically inverts this monotone relation in clamp mode. Its forward functions are AST-identical to the separately completed shortcut, whose only addition is the fit.

For the physical clamp output, put `s=(N-|X|/b)/2`. The projected density is an Irwin–Hall polynomial sum proportional to `sum_j (-1)^j choose(N,j)*(s-j)_+^(N-1)`. The oracle takes its logarithmic derivative, with the sign of X. `math.fsum` controls cancellation over the stated N range. This is a closed expression for the specified finite-N ensemble, without fitting a force correction.

The wrong closure is physically meaningful: it identifies the clamp force with the force producing the same *unconstrained mean* extension. It becomes accurate for long chains. A trial that derives the correct constrained density but makes a polynomial, derivative or sign mistake is not a clean instance of the intended physical failure.

## Independent reference and checks

The grading reference directly inverts the independent bond characteristic function `sinc(q)^N`; its density and spatial derivative are separate oscillatory quadratures. It contains neither the oracle polynomial sum nor the inverse-Langevin curve. Composite Gauss–Legendre panels use 256 intervals of length pi with order 32; refinement doubles the intervals and increases order to 48. Force-mode calibration is instead generated with an independent single-bond angular Boltzmann quadrature.

For finite K, multiplying the characteristic function by `exp[-T*q^2/(2*K*b^2)]` gives the Gaussian trap convolution. Its force verifies the operational mechanical-clamp interpretation and converges to the readout in the stated limit. The strongest finite-K extrapolation is accurate to `1.46e-5`; it is an author check, not the grading reference.

The final validator covers 666 full-domain cases: every integer N from 4 through 12, 11 b values over the entire interval, control endpoints and random interiors. Maximum oracle/Fourier difference is `2.92e-8`; maximum refinement change is `2.59e-8`. All physical forces and projected densities are positive. Reflection, length/temperature scaling, density normalization and variance, the analytic last-support-interval force, and large-N equivalence pass. The ungraded single-link limit has a flat interior axial density and zero holding force.

The minimum pointwise source gap in the domain is 5.24% for N=12, which is close to the 4% grading threshold. This limitation is retained. The three diagnostic groups use ordinary N=4,6,8 controls; their minimum group gaps over 41 b values are 18.9%, 13.0%, and 9.54%. A separate force-mode anchor must pass for both models. Larger-N author limits show the expected decay of the difference, reaching about 0.485% at N=128.

## Calibration and grading

The unknown common length is b in `[0.8,1.2]`. The private value is 1.06. Every public constructor initializes it to `None`; the private value is absent from the image sources. The calibration has 27 distinct settings, independently repeated 12 times (324 records), varying N, temperature and positive applied force. Every record has fixed instrument sigma `0.006 L0`, independent of b or the predicted response. Data generation uses seed 161051; the separate 256-noise check uses seed 161057.

Global identifiability follows from

    d mean_x / db = N*(coth(z)-z*csch(z)^2) > 0,  z=F*b/T>0,

because `sinh(z)*cosh(z)>z`. Every individual nonzero-force setting is injective in b. The 41 noiseless full-interval recoveries have maximum absolute error `7.43e-9`. The minimum calibration derivative is 0.4554. Independent calibration quadrature differs by at most `2.67e-14`.

The actual fit is `b=1.0600717911`, reduced chi square 0.93495, and relative parameter error `6.78e-5`. Actual oracle hidden errors are at most 0.000166; shortcut diagnostic errors are 0.22480, 0.13807, and 0.10156. All 256 noise draws pass calibration and parameter gates for both controls; all oracle predictions pass, all three shortcut clamp groups fail, and every shortcut force anchor passes. Across these draws the worst oracle prediction error is 0.000325 and the smallest shortcut diagnostic error is 0.10138.

The ordinary gates are unchanged: reduced chi square below 1.5, parameter relative error below 3%, and each prediction-group normalized RMS error below 4%. The grader enforces numeric output shape and finiteness; it imposes no implementation, formula, dependency or solver restriction. Local pytest: oracle 7/7; shortcut 4 passed and 3 intended prediction failures, each under 0.5 seconds. The author validator ran once successfully, including data generation; no scientific failure or rejected numerical revision is hidden.

## Public boundary and history

`instruction.md` is the exact neutral instruction permitting replacement of all implementation helpers. The Dockerfile copies only the README, starter, public test and calibration data. The hint, private reference, metadata, oracle and this document remain outside the agent image. Source imports are used. The public measurement settings are physical controls, not an instruction to keep any starter formula.

`../../prototype/` preserves the bounded feasibility files, its original `author-manifest.json`, the independent Materials review, and the wording followup. The final README omits the prototype's interpretive rigid-Hamiltonian sentence and redundant alternative-constraint sentence. The unknown b bounds are no longer listed as known controls, and X/N units are explicit. These are pre-evaluation clarifications; no prior task or model outcome was changed.

The archive audit inventories 342 prior AUTHOR/assessment paths and directly reads 12 closest files; it does not claim to have fully read all 342. The rigid-linkage task concerns the canonical momentum metric of an exactly constrained system, whereas this experiment fixes the Cartesian spring measure and tests finite force/clamp ensembles. FENE concerns nonlinear driven constitutive averaging. Fixed-number-band shares the broad thermodynamic-constraint theme but measures thermal capacity, not a fluctuating mechanical force. Electric–magnetic dipole recoil was rejected as an exact archived duplicate. Generic detector backaction and odd-viscosity stress ideas were not developed into numerical packages. Those decisions and source hashes remain in the prototype audit.

## Primary-source scope

- [Sinha and Samuel, *Inequivalence of Statistical Ensembles in Single Molecule Measurements*](https://arxiv.org/pdf/cond-mat/0412504), Sections II–III, was read by author and root. Its transverse-free trap, stiff longitudinal/constant-force preparations and single-rod example directly support the operational distinction. The finite-N freely jointed convolution here is independently derived.
- [Keller, Swigon and Bustamante (2003)](https://pubmed.ncbi.nlm.nih.gov/12547757/): primary abstract read; supports different finite-molecule force/extension preparations. Full PMC access hit a challenge; no full-paper claim is made.
- [Süzen, Sega and Holm](https://arxiv.org/html/0810.3407v3): abstract and theory sections read for context. Their vector/distance conventions and limiting claims are not substituted for this axial-clamp experiment.
- [Fiasconaro and Falo](https://arxiv.org/abs/1805.01499): primary abstract read for the central-spring freely jointed realization. The present stiff-limit measure is derived above.

Final science and local evidence are `../../results/finite-chain-clamp-r1-validation.json` and `../../results/finite-chain-clamp-r1-local-controls.json`; source provenance and independent final peer are kept beside them. No Docker or model-agent evaluation was launched by the author.
