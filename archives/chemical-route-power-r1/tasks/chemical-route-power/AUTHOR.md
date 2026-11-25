# Chemical-route regeneration power, revision 1

This candidate is scientifically validated and has passed its local completed controls. No model-agent evaluation has run. It keeps the neutral instruction permitting replacement of the entire implementation, an unfinished fit with `rate_scale=None`, fixed measurement uncertainty, and the standard 600/60-second limits. Empirical difficulty is unknown.

The supplied predictor has the exact conformation kinetics. Its completed physical approximation is the entropy production of the aggregate Markov process, interpreted as regeneration power at kB*T=1. That process is mathematically valid and preserves every conformation probability and transition statistic. The stated reservoirs are instead restored separately, so the measured chemical work depends on which elementary reaction made each state transition.

## Physical derivation and calibration

For oriented edge i to j=(i+1) mod3 and reaction route a, write the forward/backward rates as k_ia^+, k_ia^-. The public stoichiometry consumes one F_a and produces one P_a in the forward direction. Its known chemical drop is A_a. The rates obey

    log(k_ia^+/k_ia^-)=A_a-(E_j-E_i).

The full conformation generator sums routes; its stationary state p is exact in both controls. Define f_ia=p_i*k_ia^+, b_ia=p_j*k_ia^- and J_ia=f_ia-b_ia. Reversible restoration of the chemical reservoirs requires stationary power

    P=sum_i,a J_ia*A_a.

This is chemical free-energy consumption. A chemical-potential difference is not generally a bare reaction enthalpy; the task therefore specifies reversible reservoir regeneration, not a simple bath calorimeter. Stationarity makes the mean internal-energy change and Shannon-entropy change vanish. Using local detailed balance, P also equals sum J_ia*log(f_ia/b_ia).

The source first sums f_ia and b_ia over a, then uses

    P_aggregate=sum_i (F_i-B_i)*log(F_i/B_i).

Each term is nonnegative. This is the complete entropy production of the exact aggregate Markov chain, corresponding to an effective single-reservoir description of each conformation edge. It retains all supplied rate controls; its defect is not an incorrect stationary solve or a missing parameter. A forward event in one route followed by a reverse event in the other can change fuel composition without changing conformation. Such a chemical cycle is absent from aggregate thermodynamics.

The per-edge difference is exactly

    F_i*KL(f_i/F_i || b_i/B_i)+B_i*KL(b_i/B_i || f_i/F_i) >= 0.

Equal affinities A_0=A_1 make each route ratio the same on every edge, for any state energies, barriers or rate scale. Calibration therefore has exact agreement while both routes are active and power is nonzero. There are 18 distinct settings spanning three nonzero affinities, three energy patterns and two total barrier patterns, with 16 independent replicates each. The source does not contain a branch for calibration or hidden settings.

The sole unknown rate_scale multiplies every rate and leaves p unchanged. Every calibrated power is a strictly positive known coefficient times this scale. The calibration objective is a globally strictly convex quadratic; the smallest coefficient is .295597. Forty-one noiseless scales across [.6,1.4] recover within4.45e-16. This is global identification, not evidence from a successful local fit alone.

## Independent reference and validation

The oracle solves stationary channel currents. The verifier independently builds a characteristic generator for marked reaction paths. A forward elementary event receives exp(i*chi*A_a), its reverse receives exp(-i*chi*A_a), and the escape diagonal remains unchanged. The eigenvalue continuing zero gives the long-time cumulant rate. Its first derivative is evaluated at two counting steps and extrapolated. The reference has no stationary probability solve and no aggregate-affinity expression. Its common-scale dependence is exact linearity of the entire generator.

The checked hidden cases agree with the reference to4.03e-11 absolute; halving the counting step changes them by at most9.35e-11. Reference generation biases calibration by less than9.74e-9 of one instrument sigma. The smallest hidden reference signal is2.55393. All hidden errors use ordinary group RMS normalized by the physical group's RMS; the gate is .04, without special sign or near-zero handling.

The validator checks 1024 corners of the full public energy, attempt [.5,1.5] and affinity [.2,2.4] ranges. Stationary probabilities remain positive, with minimum .110637. Probability and stationary energy balances are below1.34e-15; the chemical-work/entropy identity and log-sum identity are below4.01e-15. At 64 corner selections and64 independent interior settings, including both rate-scale endpoints, the independent counting reference agrees within1.59e-11 after scaling by max(1,abs(power)); its counting-step change is below4.11e-11. This extends beyond the prototype's narrower diagnostic attempt range. Equal-fuel equilibrium at zero affinity, energy-origin invariance, route exchange and common rate scaling also pass. The zero-affinity equilibrium and shifted energy origin are author-only limits outside the fixed public coordinate convention.

At the supplied calibration, both controls fit rate_scale=1.07001158784 and reduced chi-square .928481. The true scale is1.07. Calibration seed148031, noise seed148037, and fixed absolute sigma=.003 are recorded privately. Sigma has no dependence on the hidden parameter or response. Public and private calibration files are byte-identical.

All256 additional noise realizations pass calibration and parameter gates for both controls. Every physical oracle passes all hidden predictions. Every completed aggregate shortcut passes the equal-affinity anchor and fails all three unequal-affinity groups. Maximum reduced chi-square is1.31741; maximum relative parameter/oracle error is.00018344. The smallest shortcut diagnostic error over noise is.177070. Actual shortcut errors are .363592, .177212 and .291987. Their normalized separation is independent of the common scale, and the 41-point scale check confirms a minimum group gap .177221. The parent prototype separately records 1024 positive-affinity diagnostic corners with gaps11.94–57.80%; that sampled box is not asserted to prove an analytic minimum over every interior point.

Actual isolated local pytest controls give oracle7/7 in.27seconds and shortcut4passes/3intended prediction failures in.28seconds. The science and local checks together take8.64seconds. These runs are author controls; Docker controls and model evaluations are separate root-owned steps.

Reproduce from the repository root:

    uv run --no-project --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python staging/chemical-route-power-r1/scripts/validate_chemical_route_power.py

Only `--generate` intentionally replaces both calibration copies. Reports are `results/chemical-route-power-r1-validation.json` and `results/chemical-route-power-r1-local-controls.json` under the staged root. Source hashes and historical prototype files are preserved by the separate provenance report.

## Scope and archive comparison

This is not claimed to be an unrelated thermodynamic principle. The closest predecessor is archived Floquet thermalization revision2, whose actual public source, README and author derivation were read. It kept sideband population rates but assigned a fixed quasienergy per jump when reporting physical reservoir heat. This candidate instead uses the exact aggregate Markov entropy production, with chemical cycles between parallel elementary routes. Both lose physical transition-channel thermodynamics. The apparatus, coherent source model and calibration nullspace differ, but that thematic overlap is material and preserved in `outlines/archive-audit.json`.

The audit covers235 AUTHOR/ASSESSMENT files and20 directly read files. Reactive-piston changes kinetics through volume coupling; membrane partition and Soret–Dufour omit common electrochemical/reciprocal constraints. Mechanical parallel-bath candidates alter noise correlations or the definition of thermal heat. Terminal-current noise tests the detector's current combination. Entropy-anomaly eliminates fast velocities. No identical parallel chemical-route regeneration task was found. This is an archive assessment, not a novelty claim or a prediction of model failure.

The original prototype, independent peer and both calibration explorations are preserved under `prototype/`. The first exploration swapped two barrier columns while equal affinities made aggregate kinetics unchanged; it was replaced before packaging by a genuinely different total barrier setting. No model outcome prompted that change. All earlier task sources and scored outcomes remain untouched.

Primary context: Schmiedl and Seifert, [Stochastic thermodynamics of chemical reaction networks](https://arxiv.org/abs/cond-mat/0605080), defines reaction-event chemical work and entropy bookkeeping. Esposito, [Stochastic thermodynamics under coarse-graining](https://arxiv.org/abs/1112.5410), treats related multiple-reservoir and coarse-graining questions. Their primary abstracts were read; attempted full-text web access failed. The latter's state-partition theory is not asserted to be this exact route-aggregation apparatus. The specified rates, explicit stationary identities and independent marked-path reference above establish the present construction.
