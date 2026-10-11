# Two-population spin activity concept review

Independent review, 2026-10-11, of the proposed apparatus and `staging/gpt6-expansion/spin-activity/prototype.py`. **The concept is scientifically defensible. Keep the equilibrium-field independent-spin baseline.** Final task sources, domain validation, Docker controls, and model trials remain to be reviewed. The intended target is three frozen, unhinted `gpt-5.6-luna` High trials.

## Apparatus and observable

Use an even number `N` of classical Ising spins, with exactly `N/2` permanently assigned to each field population. Their local fields are `h+H` and `h-H`. These are fixed labels, not spins exchanging populations or fields redrawn during a trajectory. A Hamiltonian consistent with the proposed rates is

    energy = -J*(sum_i sigma_i)**2/(2*N)
             - sum_i (h + label_i*H)*sigma_i,

with thermal energy one. The self-interaction contributes only a constant. The exact flip rate is

    kappa/2 * [1 - sigma_i*tanh(J*sum_(j != i) sigma_j/N + h + label_i*H)].

This obeys finite-N Gibbs detailed balance. State the division by `N`, including in the excluded-spin field, explicitly. Define `K_plus` and `K_minus` as actual flip counts, counting both spin directions positively. The requested joint function is

    psi(s_plus,s_minus) = lim_(N -> infinity, N even) lim_(t -> infinity)
        log E[exp(-s_plus*K_plus-s_minus*K_minus)]/(N*t).

Its units are inverse time, and the counting fields do not change the apparatus rates. A finite-N Gibbs initial preparation and this order of limits remove phase-selection and metastability ambiguities. They do not prescribe a solver. A well-trained statistical physicist can derive the answer from the finite count process without knowing a specialized closed formula.

## Correct macroscopic law

Let `z_plus,z_minus` be magnetization per spin within each population and `M=(z_plus+z_minus)/2`. For each population `alpha`, put

    f_alpha = J*M + h_alpha
    u_alpha = kappa/4*(1-z_alpha)*(1+tanh(f_alpha))
    d_alpha = kappa/4*(1+z_alpha)*(1-tanh(f_alpha)).

These rates are per spin of that population. The limiting answer is

    max_(z_plus,z_minus in [-1,1])
        (1/2)*sum_alpha [2*exp(-s_alpha)*sqrt(u_alpha*d_alpha)
                        -u_alpha-d_alpha].

The population factor must not be applied twice. The prototype's angle representation `z_alpha=sin(theta_alpha)` and factor `kappa/4` outside the two-term sum are correct.

This maximum follows from the reversible finite-N tilted generator. Its symmetric off-diagonal entries are the geometric means of opposite physical transition rates, multiplied by `exp(-s_alpha)`; its diagonal is the unmodified negative escape rate. The maximal row sum bounds its largest eigenvalue. A positive test vector localized over many adjacent count states near the optimal macroscopic state gives the matching limiting lower bound. For finite fields and finite counting biases the optimizing densities can be approached from the interior. No additional ansatz about stationary equilibrium magnetization is required.

The core physics is that atypical activity selects collective magnetizations that change both populations' local fields. Even if single-spin counting at a fixed field is solved exactly, retaining the ordinary equilibrium molecular field loses that response. Requiring the fields to be self-consistent with independently tilted one-spin occupations also does not, by itself, extremize the correct objective: varying either population changes both local fields.

## Baseline choice

Keep the baseline that finds the ordinary equilibrium mean-field magnetization and then uses exact independent-spin counting functions in its fixed local fields. Choose the equilibrium root by the minimum of the ordinary free energy, as in the prototype. Restrict or explicitly handle exact equilibrium phase coexistence, where the finite-N Gibbs state can have several macroscopic branches. A small nonzero `h` and domain checks can avoid an arbitrary baseline branch choice without hiding a condition from the solver.

This baseline defines one definite auxiliary stochastic process. Its rates do not change when a different statistic is requested; its joint generating function is convex and has the correct counting normalization. It is nonzero and uses every apparatus input.

Do not use the proposed alternative that solves biased one-spin self-consistency and then selects whichever root maximizes the sum of single-spin generating functions as the required control. That rule changes the auxiliary dynamics with the counting fields and lacks the variational contribution for changing the collective field. Root selection is not justified by the underlying apparatus or by one fixed alternative Markov process. It may be useful as an exploratory approximation or as a diagnostic of an agent's reasoning, but its additional branch and consistency issues would muddy the intended physical failure.

## Calibration and independent reference

At `J=0`, every spin is independent in its specified field. Oracle and baseline therefore agree exactly at arbitrary `h,H,s_plus,s_minus`. Their common answer is

    kappa/4 * sum_alpha
        [-1 + sqrt(tanh(h_alpha)**2
                   + exp(-2*s_alpha)/cosh(h_alpha)**2)].

Vary nonzero counting fields and local fields; these readings identify the common `kappa` by linear weighted least squares. Fixed instrument uncertainty should not encode an unknown noiseless answer.

The prototype's independent reference is appropriate. It uses counts `n_plus,n_minus` from zero to `N/2`, keeping the population fraction exactly fixed at every size. With `S=2*(n_plus+n_minus)-N`, an up jump sees excluded-spin magnetization `(S+1)/N`, and a down jump sees `(S-1)/N`. Its finite-N rates and symmetric sparse matrix have these factors correctly. Dividing the matrix by `N` gives the required per-spin eigenvalue. This reference does not maximize the oracle's macroscopic objective.

Before packaging, compare at least three increasing sizes and a shifted size window. An ordinary `1/N` extrapolation is appropriate only where convergence supports it; dynamical critical points can have different corrections. The reported example at `J=1.4,h=.1,H=.8,s_plus=-.6,s_minus=.8` has useful finite signal, and the supplied extrapolation differs from the variational value by about `1.2e-5`, far below a 4% relative gate. This one example is feasibility evidence, not domain validation.

Also check: the exact `J=0` law; zero bias; scaling with `kappa`; nonincreasing dependence on each bias and joint convexity; field/population reflection; agreement of refined global-search grids; and the smallest hidden group signal. Use multiple maximizing branches and independent starts so an optimizer failure is not mistaken for a physical prediction. Check both calibration and hidden sensitivity over 256 noise realizations.

## Overlap and classification

No exact Glauber/activity or bimodal-field task was found in the inspected current/archive AUTHOR files. The closest qualified task is ring-current-fluctuations: both study collective changes under trajectory conditioning. The ring has a conserved density and a spatially varying profile; this apparatus has nonconserved magnetization and two permanently distinct field populations. This is a distinct apparatus and reduction problem, not an unrelated foundational principle. Hierarchical-glass-overlaps instead concerns equilibrium replica correlations in a quenched tree.

Primary context is [Guioth and Jack, *Dynamical phase transitions for the activity biased Ising model in a magnetic field*](https://arxiv.org/abs/2002.00905), whose sections 2–3 treat the tilted flip operator, reversible variational principle, and mean-field activity phases. Their energy convention uses `-N*J*m**2`, so their `2J` corresponds to the `J` above. The fixed two-population extension and exact normalization here are derived for this proposed apparatus.

A trial that freezes the equilibrium molecular field when calculating the biased activity law exhibits the intended physical approximation. A trial that selects the correct collective tilted generator or variational law but mishandles a factor of two, misses a numerical maximum, or extrapolates a finite matrix incorrectly does not establish that physical failure. Difficulty remains an empirical question.
