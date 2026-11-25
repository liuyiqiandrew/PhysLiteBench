# Finite-bath gravitating gas, revision 1

The finite reservoir changes the allowed entropy variation of a self-gravitating gas. The completed shortcut uses the correct hydrostatic profiles but applies the ideal-thermostat stability threshold to every reservoir. Ordinary profile temperatures and infinite-bath critical temperatures cannot distinguish it from the physical model. Hidden finite-bath thresholds can.

For the regular isothermal Lane–Emden solution, write u=alpha^2 exp(-psi)/v and v=alpha psi'. The specific temperature is theta=GM/(Rv). The wall pressure gives gas energy E_g=(GM^2/R)(u-3/2)/v. Adding the reservoir gives E_total=(GM^2/R)(u+b-3/2)/v. The first finite-bath energy turning point after the canonical point satisfies

    4u^2 + 2uv + (2b-11)u + 3-2b = 0.

The canonical condition is u=1. As the branch proceeds from dilute gas, the canonical temperature minimum occurs at contrast 32.1249. An isolated gas remains locally stable until contrast 708.6083. Finite positive reservoirs place the first instability between these limits. The gas temperature need not decrease throughout that continuation. The public instructions explicitly distinguish branch order, local stability, finite composite energy, and externally imposed temperature.

The private reference does not use that turning-point criterion. It integrates the equilibrium equations in logarithmic radius, constructs the reduced entropy Hessian for radial mass perturbations q with q(0)=q(R)=0, and finds its first eigenvalue crossing. At G=M=R=1 its quadratic form is

    Q = -1/2 integral q'^2/(4*pi*rho*r^2) dr
        +1/(2*theta) integral q^2/r^2 dr
        -(integral M(r)*q/r^2 dr)^2 / ((3+2b)*theta^2).

The final term vanishes for the infinite reservoir. It follows by maximizing the gas kinetic plus reservoir entropy over their energy split at fixed composite energy. A regular mass-conserving basis q=r^3(1-r^2)P_j(2r^2-1) gives a generalized curvature matrix. The reference uses 96 modes and 256-point quadrature, checked with 128 modes and 384 points. It also checks positive curvature before, and negative curvature after, each critical point. The oracle independently integrates the ordinary Lane–Emden equation and solves the turning-point condition.

Calibration has 240 independent records at 30 settings, with sigma=0.001 in temperature units and seed 616001. G=1.08 is globally identified because every calibration mean is a positive known coefficient times G. Both completed models fit G=1.079964207650113, with reduced chi-square 0.87056. Shared anchor predictions agree; finite-bath shortcut group errors are 18.21%, 13.08%, and 7.99%. The oracle errors are about 0.0033%. All groups use relative RMS error below 4%, with calibration chi-square below 1.5 and parameter relative error below 3%. These margins survive 256 noise realizations with seed 616009. The largest oracle/reference difference across 60 domain cases is 3.46e-10 in output units.

The initial finite-bath concept was reviewed before packaging. Its predecessor, a binary stability-flag proposal, was dropped before implementation because constant stable labels and an ancillary parameter fit gave a weak test. The archive's finite-heat-reservoir task shares an ensemble theme but concerns oscillator equilibrium moments, not the nonuniform gravothermal instability or its entropy mode.

Run `python scripts/validate_finite_bath_gravitating_gas.py` in the pinned numerical environment. Add `--generate` only to intentionally replace calibration and private truth. Run the physical oracle and completed `scripts/finite_bath_gravitating_gas_baseline.py` through Docker separately. Evaluate a single frozen batch of three gpt-5.6-luna High trials with built-in web search disabled; shell network access remains available. Qualification requires all three failures to survive source and trajectory review as physical errors. Preserve every outcome and do not resample an unchanged completed revision.
