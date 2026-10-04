# Chemical-route power prototype

This is an author-only numerical prototype. No task package, calibration dataset or model-agent evaluation has been created. The root authorized a bounded prototype after the three-outline assessment. Empirical difficulty is unknown.

## Physical distinction

The conformation process is exactly Markovian even after parallel reaction routes are aggregated: a state transition i to j has rate K_ji=sum_a k_ji,a. The source solves this full three-state generator without approximating its kinetics. With stationary p, it computes

    P_aggregate=sum_edges (p_i K_ji-p_j K_ij)
                          log[(p_i K_ji)/(p_j K_ij)].

This is the nonnegative entropy-production rate of the exact aggregate Markov chain at kB*T=1. It is a coherent effective-single-reservoir thermodynamic model. It uses every energy, attempt factor, fuel drop and the common rate scale; no rate or state control is ignored.

The stated chemical apparatus has two distinct fuel conversions. Its reversible regeneration work rate is

    P_chemical=sum_edges,a (p_i k_ji,a-p_j k_ij,a)*A_a.

Local detailed balance gives log(k_ji,a/k_ij,a)=A_a-(E_j-E_i). At stationarity the mean internal energy derivative and Shannon entropy derivative vanish. Consequently the chemical work also equals the sum of channel-resolved entropy currents. This identification concerns stationary chemical free-energy consumption. It is not a claim that a molecule's chemical-potential difference is always its bare enthalpy or that a simple calorimeter measures it.

The distinction is physical: aggregation keeps all conformation dynamics but removes the identity of the regenerated reservoirs. A two-edge cycle that goes forward by one route and backward by the other changes no conformation while converting one fuel species into the other. The source's effective reservoir cannot record that cycle.

For one edge let f_a=p_i k_ji,a, b_a=p_j k_ij,a, F=sum f_a and B=sum b_a. The excess entropy is

    F*KL(f/F || b/B)+B*KL(b/B || f/F) >= 0.

It vanishes exactly when all route forward/backward ratios on that edge coincide. This supplies an analytic nonnegative-gap check independent of numerical fitting.

## Exact calibration and identifiability

Both routes remain active. Calibration uses A_0=A_1 with three nonzero common affinities, three energy patterns and two actual barrier patterns:18 distinct settings. For any unknown scale and any energies/barriers, all route ratios on each edge coincide, so the two predictors agree exactly. This is not a calibration with zero power or a removed channel. The common scale multiplies all rates, leaves the stationary state unchanged and multiplies power. Every calibration coefficient is positive, so the least-squares objective is a strictly convex quadratic globally on [.6,1.4]. The smallest calibrated derivative is.295597. Forty-one noiseless scales recover within4.45e-16. The fixed instrument sigma proposed for a future package is.003, independent of the scale and response; no data were generated here.

The first prototype used a channel-swap barrier setting that did not change aggregate kinetics on equal-affinity calibration. That run and source are preserved under initial-calibration/. The final check replaces it with a different total barrier pattern. This change predates any task construction or model run.

## Independent reference and numerical evidence

The oracle prototype solves stationary currents. The independent reference instead constructs a route-marked characteristic generator: each physical jump carries exp(i*chi*A_a) or its inverse, while escape rates stay unchanged. The eigenvalue continuing zero gives the long-time work cumulant rate; its first counting derivative is extracted at two step sizes and extrapolated. It has no stationary-probability solve or aggregate-affinity readout.

The final report contains:

-1024 corners of the proposed unequal-affinity diagnostic subset, with aggregate underestimates between11.94% and57.80%. Both chemical drops are positive. At unit common scale the smallest true power is1.67759 and the smallest aggregate power is.817216, so separation does not depend on a nearly zero signal. Individual weak-route currents can run backward, which is physical; no assertion that every local reaction current is forward is made.
-128 random interior settings plus18 calibration settings checked with the counting reference; maximum error divided by max(1,abs(power)) is1.405e-11. Halving the counting step changes the derivative by at most2.97e-10 on the same scale. These samples are numerical evidence, not a rigorous continuum minimum-gap proof.
-Positive stationary probabilities, conserved probability and stationary energy, agreement of chemical power with route entropy, energy-origin invariance, exchange of route labels, linear rate scaling and zero power at equilibrium.
-The uniform-ring identity: channels with forward/reverse rates(4,1),(2,1) both have forward net current. True power is3log4+log2, while aggregate power is4log3, a9.43% difference. This directly solvable example uses no matrix or counting solver. A larger opposed-route example is retained only as supplementary evidence; it is not needed for the declared positive-affinity margins.

Reproduce with the pinned NumPy2.3.3/SciPy1.16.3 environment using python staging/chemical-route-power-prototype/check.py. Runtime is less than a second after environment setup. These are author science checks, not model evaluations. No256-noise study or Docker control is claimed at the prototype stage.

## Archive overlap and primary evidence

The prior assessment scans235 AUTHOR/ASSESSMENT files and hashes20 actual files read at staging/physics-outlines-20261004-next-brownian/archive-audit.json. The closest predecessor is archived Floquet thermalization revision2. Its full public source, README and AUTHOR were read: it computes correct sideband population rates but counts fixed quasienergy per jump instead of the physical reservoir's sideband energy. This is a real thematic overlap in lost transition-channel thermodynamics. The proposed chemical source uses the mathematically correct entropy production of an exact aggregate Markov process, rather than a fixed jump-energy assignment. No identical parallel chemical-route regeneration apparatus was found, but the difference does not justify claiming an unrelated general principle.

Reactive-piston and membrane tasks alter state dynamics through mechanical or electrochemical constraints. Soret–Dufour omits reciprocal transport. Structured-reservoir thermal bodies changes noise correlations; rotating-reservoir and magnetic-bath-transfer misidentify moving/local thermal heat. Terminal-current noise changes the physical detector combination. Entropy-anomaly integrates out fast velocity modes. None has exact finite-state conformation kinetics with hidden chemical cycles between parallel elementary routes. No existing source or trial was edited.

Primary background: Schmiedl and Seifert, [Stochastic thermodynamics of chemical reaction networks](https://arxiv.org/abs/cond-mat/0605080), describes trajectory-level chemical work and entropy. Esposito, [Stochastic thermodynamics under coarse-graining](https://arxiv.org/abs/1112.5410), supplies related reservoir/coarse-graining context; its state-partition discussion is not claimed to be precisely this route aggregation. These primary abstracts were read; attempted full-text web access failed. The explicit rates, log-sum identity and independent counting construction above establish the present prototype without relying on an uninspected paper formula.
