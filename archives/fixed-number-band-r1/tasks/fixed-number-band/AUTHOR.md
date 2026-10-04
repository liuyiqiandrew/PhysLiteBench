# Fixed-number band calorimetry, revision 1

This staged candidate has completed scientific validation and local completed controls. No model-agent evaluation has run, so empirical difficulty is unknown. The public task retains the neutral instruction permitting full implementation replacement, an unfinished fit with width=None, fixed instrument uncertainty and standard600/60-second limits. It does not prescribe a response formula or solver.

## Exact state, different thermal response

There are M independent spinless fermion states at each energy -W/2,+W/2. Divide all extensive readings by2M. The public apparatus fixes the exact total particle number while a weak energy-only contact equilibrates both bands. Equilibration precedes neglect of contact energy; no conserved band-population preparation is implied. Geometry and levels stay fixed during the reversible temperature increment.

In the thermodynamic limit the equilibrium occupations are exactly

    f_s=1/[1+exp((E_s-mu)/T)],
    (f_-+f_+)/2=n.

Both completed models solve that constraint correctly at every requested state. Define M_j as the average over the two bands of (E_s-mu)^j*f_s*(1-f_s). The supplied source computes M_2/T^2. This is the positive open-system heat response T*(dS/dT)_mu, or dU-mu*dN under fixed chemical potential. It is not simply (dU/dT)_mu, which contains chemical-work bookkeeping. The source therefore has a coherent thermodynamic interpretation with exact populations; its error concerns the infinitesimal particle-exchange constraint of the measured calorimeter.

For the physically closed sample, differentiating its conserved filling gives dmu/dT=-M_1/(T*M_0). Its capacity is

    C_N=(M_2-M_1^2/M_0)/T^2.

The subtraction is nonnegative. For exactly two levels, setting a=f_-(1-f_-) and b=f_+(1-f_+) reduces the physical answer to

    C_N=W^2*a*b/[2*T^2*(a+b)].

The oracle uses this stable weighted-variance identity, which avoids cancellation when the capacity is exponentially small. None of these diagnostic formulas appears in the public README or instruction. The source uses both temperature and filling fully and has no calibration-specific branch.

## Calibration and global identification

At half filling symmetry fixes mu=0 and M_1=0. Both thermal responses are exactly C=x^2*sech(x)^2 with x=W/(4T). Calibration uses nine temperatures between.10 and.15, each measured32 times independently. Throughout W in[.8,1.2], x>=4/3 and x*tanh(x)>1, so

    dC/dW=2*C/W*[1-x*tanh(x)] < 0.

Every calibrated curve is globally injective on the entire allowed interval. This proof does not depend on optimizer behavior. The sampled least-negative derivative is-.172884. Forty-one noiseless widths, including both endpoints, recover within6.86e-9. Exact calibration equivalence holds to1.67e-16.

The true gap is1.07, distinct from the interval midpoint and absent from the public image. Calibration has288 records, independent fixed sigma=.0003, generation seed152031 and noise seed152037. Public/private data are byte-identical. Finite canonical reference generation has a maximum numerical bias of3.41e-5 of one instrument sigma.

## Independent canonical reference

For a finite sample with exactly N particles, let k be the upper-band occupation. Its multiplicity is choose(M,k)*choose(M,N-k). Apart from a k-independent energy factor, the canonical probability is proportional to

    choose(M,k)*choose(M,N-k)*exp(-W*k/T).

The private reference evaluates these weights with log-gamma arithmetic, computes the centered variance of k directly and returns W^2*Var(k)/(2M*T^2). It does not solve a chemical potential, use Fermi occupations, or perform the susceptibility subtraction. It then removes the first two inverse-M corrections using M=8000,16000,32000. All calibration, scored and canonical validation fillings are rational and yield exactly integer N at every selected size. No integer rounding is silently treated as finite-size bias. Arbitrary public fillings remain valid for the thermodynamic oracle; the reference's controlled finite-size comparisons explicitly use exact fixed filling.

For30 public-domain endpoint/intermediate cases, doubling the size window to16000,32000,64000 changes the extrapolation by at most4.39e-8. The base extrapolation differs from the thermodynamic oracle by at most4.98e-8 absolute. Raw finite-size errors decrease over the four sizes from2.883e-5 to3.604e-6. A fourth-order temperature derivative of the independently computed finite canonical mean energy agrees with its fluctuation capacity within1.61e-8. On the scored inputs, the maximum reference discrepancy is1.15e-9 and the larger-size change is1.01e-9. Both are negligible relative to the ordinary .04 normalized group RMS gate.

A separate60-case corner/interior check differentiates the re-equilibrated thermodynamic mean energy at fixed filling. It agrees with the oracle within1.42e-12; halving its derivative step changes results by at most3.62e-12. Differentiating entropy while actually holding the solved chemical potential constant independently verifies the completed source's own open heat response within1.18e-12. This distinguishes a mathematically valid alternate thermal process from an incorrectly coded energy derivative.

Further checks cover positive physical capacity, nonnegative source excess, particle-hole symmetry, number normalization, and high/low-temperature limits. In the high-temperature limit C_N*T^2 tends to n(1-n)*W^2/4, while the open source tends to n(1-n)*logit(n)^2 away from half filling. At low positive temperature the closed capacity tends to zero even for a partially occupied flat band. These limits outside the scored temperature range are author checks only. The exact example n=.3 and W/T=log9 gives physical9*log(9)^2/272 and source9*log(9)^2/200, a36% discrepancy; numerical error is below5.56e-17.

## Completed controls and noise robustness

Both controls fit width=1.07003260478 with reduced chi-square1.163685. Oracle hidden group errors are at most3.49e-5. Shortcut diagnostic errors are.988515,.988515,.584447; its half-filled anchor passes. The smallest scored heat capacity is.0837359, so normalization does not rely on a vanishing signal.

All256 independent noisy calibrations pass parameter and calibration gates for both controls. Every oracle prediction passes; every completed shortcut fails all three doped groups and passes the half-filled anchor. Maximum reduced chi-square is1.23422, maximum relative width error.000112971, and maximum oracle hidden error.000129118. The minimum shortcut diagnostic error over noise is.584440. Across41 widths spanning the full interval, the smallest fixed diagnostic group gap is.434723. These are explicit sampled checks, not a universal minimum-gap assertion for all possible doped inputs.

Actual local isolated pytest controls give oracle7/7 in.39seconds and shortcut4passes/3intended prediction failures in.48seconds. Science plus local controls takes2.34seconds. The grader checks only numerical outputs, the documented API, calibration and fitted parameter; it does not inspect code or constrain methods. Its standard gates are prediction NRMSE<.04, parameter relative error<.03, and reduced calibration chi-square<1.5.

Reproduce from the repository root:

    uv run --no-project --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python staging/fixed-number-band-r1/scripts/validate_fixed_number_band.py

Only `--generate` intentionally replaces both calibration copies. Scientific and local reports are `results/fixed-number-band-r1-validation.json` and `results/fixed-number-band-r1-local-controls.json` under the stage. Root owns any Docker controls or model evaluations.

## History, overlap and source scope

The complete original prototype and its independent Materials review are preserved under `prototype/`, including the initial screen. That screen showed that some colder doped preparations have accidental near-equality of the two responses, and that an insufficient finite-size window was inaccurate at the coldest half-filled case. Before task construction or any model run, the prototype chose diagnostic temperatures.4–.6 and larger canonical sizes. The complete public temperature domain remains.1–.6. Both exploratory reports are preserved rather than overwritten. Final package validation independently uses its actual source and specified reference sizes.

The archive-overlap audit directly read quantum-calorimetry, finite-band-reservoir, finite-heat-reservoir and spin-strain-response author files, plus the originating outline. Quantum calorimetry changes the subsystem-plus-bath energy being measured; finite-band reservoir changes reachable quench populations; finite heat reservoir changes the total-energy ensemble; spin-strain distinguishes a quantum response from equal-time variance. Here populations are correct and the infinitesimal particle-exchange constraint is wrong. Related ensemble and calorimetric themes are acknowledged; no claim of an unrelated general principle or exhaustive new archive search is made.

Primary background: Kuzmenko and Mikhajlov, [The canonical heat capacity of normal mesoscopic fermion systems](https://arxiv.org/abs/0811.2078). Its primary abstract was read and supports canonical independent-fermion calorimetry as context. The full paper was not read here, and no exact two-flat-band formula is attributed to the abstract. The model, derivative identity, normalization and canonical partition reference above are derived explicitly for this task.

The physical correction is compact and familiar; strong scientific separation does not establish model difficulty. No empirical claim precedes the frozen evaluation.
