# Binary gravitating gas, revision 1

The completed shortcut replaces both species by a monodisperse auxiliary gas with the same total mass and particle number. Its mean particle mass is m/(1-q+q/chi). This is exact for the calibration families chi=1 or q=0, where every critical temperature is a positive known coefficient times G. Calibration therefore identifies G globally while leaving the mixture assumption untested.

In the actual binary gas, a common temperature gives species densities proportional to exp(-Phi/theta) and exp(-chi*Phi/theta). Heavy particles concentrate near the centre. Their self-gravity changes the first canonical instability; averaging particle masses before finding the equilibrium and its unstable mode loses this effect. The [two-species equilibrium and stability framework](https://arxiv.org/abs/cond-mat/0503251) is distinct from finite-particle dynamical equipartition or collapse-time estimates.

Let h be the central heavy mass fraction and alpha^2=4*pi*G*rho_total(0)*R^2/theta. The regular field obeys

    psi'' + 2 psi'/x = (1-h) exp(-psi) + h exp(-chi psi).

Integrate both species' enclosed masses and choose h at each alpha so that the GLOBAL heavy mass fraction is q. Then eta=alpha*psi'(alpha)=GM/(R*theta). The physical oracle follows that fixed-composition branch from dilute states and locates the first eta maximum. The central composition must change during this continuation. The code uses total central density to parameterize the branch; using only light central density can introduce folds. It scans for the first peak before optimizing locally.

The private reference independently integrates equilibrium in logarithmic radius, then checks the canonical free-energy Hessian. For radial enclosed-mass perturbations a(r) and b(r), each regular at zero and zero at the wall,

    2*delta2 F/theta = integral [a'^2/(4*pi*rho_l*r^2)
                               + b'^2/(4*pi*chi*rho_h*r^2)] dr
                       - (G/theta) integral (a+b)^2/r^2 dr.

It uses separate species entropy blocks and the common gravity block, so each species mass is conserved. The first lowest-eigenvalue crossing supplies frozen truth without using the oracle's turning-point criterion. An absent species is removed. The regular basis is r^3(1-r^2)P_j(2r^2-1), with 64 modes per species and 256-point quadrature, refined to 96 modes and 384 points. Separate checks cover the first crossing, mass normalization, heavy central enrichment, equal-mass and absent-heavy limits, and random/boundary domain cases.

Calibration has 240 independent records at 30 settings, sigma=0.001, seed 617001, and true G=1.07. The validator also checks 256 independent noise draws (seed 617009) and 41 noiseless parameter values. Three hidden mixture groups compare moderate, unequal, and heavier particle masses; shared anchors retain the exact calibration limits. Predictions use relative RMS error below 4%, parameter relative error below 3%, and reduced calibration chi-square below 1.5. The saved validation report records actual margins and numerical errors.

The prior finite-bath gravitating-gas candidate passed all three model trials and remains archived unchanged. It tests a reservoir-energy constraint. This candidate fixes the canonical reservoir and instead tests species-dependent spatial redistribution. The separate concept review found no direct archive duplicate.

Run `python scripts/validate_binary_gravitating_gas.py` in the pinned numerical environment. `--generate` intentionally replaces both calibration copies and hidden truth. Run the oracle and completed `scripts/binary_gravitating_gas_baseline.py` in separate fresh Docker jobs. Evaluate one frozen batch of three gpt-5.6-luna High trials, with built-in web search disabled and shell network access still available. Review every final source and trajectory; numerical or implementation failures do not establish the intended physical omission. Preserve every outcome and do not resample an unchanged completed revision.
