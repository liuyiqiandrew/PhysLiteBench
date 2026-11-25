# Superconducting heat revision 4: stationary spin feedback

This revision retains the superconducting heat family, the circular magnetic
barrier, both driven anomalous coherence signs, and all transfer and pair
creation/annihilation channels from revision 3. The new physical constraint is
finite spin relaxation in two thermostatted proximity islands. Their internal
spin populations feed back through Pauli blocking. All apparatus quantities
except the common barrier conductance are public. The shortcut is the coherent
fast-spin-relaxation limit: it implements the entire correct revision 3 heat
model, taking both spin potentials to zero.

Revision 3 passed science controls but had one genuine physical pass in three
unhinted Luna High trials. Its task, runnable validator and controls, all
reviewed reports, and all three native jobs are preserved under
`archives/superconducting-heat-r3`. No result from that batch qualifies as a
revision 4 result. No revision 4 agent result is assumed by these documents.

## Pinned pairing and population preparation

A locally self-consistent intrinsic BCS gap would be suppressed by the proposed
strong spin imbalance. The public apparatus instead uses intrinsically normal,
noninteracting islands coupled to rigid infinite-gap singlet parents. The
static anomalous self-energy fixes the induced gap. There is no local pairing
interaction to make that gap depend on occupation, no propagating parent
quasiparticle channel, and no spin leak. Proximity couplings are prepared to
follow the supplied induced-gap law, then held fixed. The parents fix charge
chemical potential and phase, with zero steady electrical power. This
construction is essential: screened spin potentials can exceed the induced
gaps. It is stated explicitly, rather than applying an equilibrium gap formula
to a spin-suppressed intrinsic superconductor.

Fast spin-conserving thermalization and opposite-spin recombination leave a
maximum-entropy quasiparticle population with chemical potentials +mu_j and
-mu_j. They fix the common quasiparticle chemical potential to zero but preserve
N_up-N_down. The spin potentials are freely evolving internal state variables;
there is no external spin battery. Spin-conserving local thermostats remove
energy and maintain the input temperatures. The measured port is only H_T's
electronic energy loss from L relative to the common charge chemical potential.
It does not subtract an internal spin potential as external chemical work.

## Transition normalization and signs

Express excitation energies E, gaps d, and spin potentials mu in kelvin.
For Fourier convention exp(-i*n*Omega*t), T_0=I, T_+=a*sigma_+, and
T_-=a*sigma_-. With singlet tensor S=i*sigma_y, the hole amplitude is
R_n=S*T_-n^* *S†. Thus R_0=T_0 and R_±=-T_±. The transfer and creation matrices
for each signed dispersion branch are

    A_n = u_L*u_R*T_n - v_L*v_R*exp(i*phi)*R_n
    B_n = [u_L*v_R*T_n + v_L*u_R*exp(i*phi)*R_n]*S.

Sum |matrix_element|² over the four xi_L,xi_R branches and divide by four.
The reduced static transfer kernel is
(E_L*E_R-d_L*d_R*cos(phi))/sqrt((E_L²-d_L²)*(E_R²-d_R²));
the driven transfer kernel has the plus anomalous sign. Each static spin
channel has weight 1/2. Each driven spin channel has weight a²/2.
Pair creation through the positive harmonic has the minus anomalous sign.

For transfer E_R=E_L+n*w:

- n=0: same-spin up and down channels; net f_Ls-f_Rs; spin changes (-s,+s).
- n=+1: L down to R up absorption; spin changes (+1,+1).
- n=-1: L up to R down emission; spin changes (-1,-1).

For pair creation E_L+E_R=w, B_+ creates an up quasiparticle in each island.
The net forward-minus-reverse occupation is 1-f_Lup-f_Rup, with spin changes
(+1,+1). This counts creation/annihilation once. The loss vector for a net
transfer is (E_L,-E_R), and for a created pair (-E_L,-E_R). Gate work is n*w
and w respectively. In physical number rates the common conversion per
microsiemens is 2e-6*k_B/e²; in pW heat it is 2e6*(k_B/e)².
The normal electrical G includes both electron spins. These factors are checked
against a separate spin-resolved matrix reference and a closed normal-state
balance, rather than inferred from a heat-only scalar trace.

## Relaxation and stationary balance

Nu=1e6 positive-energy quasiparticle states per spin per kelvin includes both
normal dispersion branches. The individual up/down flip rate is 1/(2*tau(E)),
with tau(E)=tau_n*E/sqrt(E²-d²). Tau_n=1e-5 seconds is a polarization decay
time, not an individual flip time. Multiplying the stated collision kernel by
the quasiparticle DOS and taking the up-minus-down number difference cancels
the DOS factor:

    R_j = (nu/tau_n)*integral_d^infinity [f(E-mu_j)-f(E+mu_j)] dE
        = (nu/tau_n)*T_j*[log(1+exp((mu_j-d_j)/T_j))
                          -log(1+exp((-mu_j-d_j)/T_j))].

Set each contact spin injection I_j(mu_L,mu_R)=R_j(mu_j). The oracle integrates
the reduced spin-resolved kernels with 192 Gaussian nodes and solves both
balances using an analytic Jacobian, residual checks, and expanding numerical
search boxes. Cold-gap gradients need Newton polishing after least squares.
The independent reference sums explicit 2x2 Bogoliubov branch matrices,
integrates heat, spin, entropy and Jacobian with adaptive vector quadrature,
and solves the kinetic balances independently. Both use stable, separately
implemented relaxation integrals.

Every transition contributes a negative injection-Jacobian diagonal; each
column's off-diagonal magnitude equals that contribution's diagonal magnitude.
Summing channels preserves weak column diagonal dominance. The strictly
positive relaxation derivative makes the full balance Jacobian strictly
column diagonally dominant with negative diagonal. Integrating that Jacobian
along the line between two hypothetical stationary roots retains strict
column dominance, so the averaged matrix is nonsingular and the roots cannot
differ. This establishes uniqueness globally at positive temperatures.

For large |mu|, the DOS tends to one; finite gap edges and finite drive shifts
supply sublinear corrections. With c=2e-6*k_B*G/e² and r=nu/tau_n, the injection
minus relaxation approaches

    -{c/2*[[1+a²,a²-1],[a²-1,1+a²]] + r*I}*mu + o(|mu|).

The matrix is positive definite because r>0. The balance points inward outside
a sufficiently large sphere, giving a root by continuity, and hence the unique
stationary solution. A numerical cutoff is expanded with the search box. No
physical bound |mu|<=w is assumed: low-drive thermal bias can exceed it, and
the domain screen explicitly finds that behavior.

## Independent physical controls

Each absorption transition raises total N_up-N_down by two. Consequently
P_gate=(k_B*w/2)*(I_L+I_R). Q_L+Q_R+P_gate=0 with both heat losses computed
inside the same device, retaining its actual gaps and temperatures.

Each tunnel channel's net rate times its forward/reverse log ratio is
nonnegative. Its entropy production is
-Q_L/T_L-Q_R/T_R-k_B*mu_L*I_L/T_L-k_B*mu_R*I_R/T_R.
Equal-energy spin flips contribute +k_B*mu_j*R_j/T_j. At stationary balance the
spin-potential terms cancel, yielding nonnegative total entropy
-Q_L/T_L-Q_R/T_R. The internal spin storage therefore needs no unmodeled spin
battery energy subtraction. Energy and spin-work identities, this entropy
identity, chirality reversal, spin-basis covariance, positive transition
probabilities, microscopic collision integration and quadrature refinement
are all included in the validator.

With zero gaps and equal nu,tau on both islands, the stationary spin potentials
are identical, independent of temperature:

    mu_L=mu_R=c*a²*w/[2*(r+c*a²)].

Writing A_j=pi²*T_j²/6+mu_j²/2, the normal left heat is
G*(k_B/e)²*[(1+a²)*(A_L-A_R)-a²*w²/2+a²*w*mu_R]; its gate work is
G*(k_B/e)²*a²*w*(w-mu_L-mu_R). Unit conversions are applied explicitly.
These formulas independently constrain the density convention, spin-flip
factor of two, harmonic weights, heat normalization and spin signs.

## Calibration and evaluation

At a=0, the unique stationary state has mu_L=mu_R=0 even for unequal
T_L,T_R. The completed shortcut and oracle therefore agree exactly on all 144
calibration inputs. The oracle retains the original static calibration
integrator literally, and both calibration JSON files are byte-preserved.
Conductance is still identified by linear weighted least squares. True G=45
microS, sigma=0.0005pW, calibration seed33021, and 256-noise seed43021 are
unchanged. Public and private data remain identical.

The three existing phase groups and their preparations are unchanged. The
fast-spin-relaxation shortcut retains every repaired revision 3 channel and
coherence sign. The finite spin balance changes Pauli blocking, instead of
introducing a further channel omission or phase-sign trick. A weaker known
tau_n=1e-7s prototype gave relative errors only about .006-.008 and was
rejected before any model trial. The 10-microsecond physical relaxation time
makes spin injection and relaxation comparable: c/r is about .48 at G=45.
The successful prototype separates all three groups by roughly .357,.604,.412.

The full validator reads checked-in data by default; generation is explicit.
It checks 256 calibration noise realizations, 191 boundary/random/hidden
settings, additional G=20 and80 controls, matrix and quadrature references,
all physical identities, and local oracle/shortcut pytest. Thresholds remain
chi²<1.5, conductance error<3%, and relative hidden RMSE<.04. Limits remain
600 seconds for the agent and60 seconds for the verifier. The task must be
frozen and Docker oracle/shortcut controls must pass as intended before an
unhinted three-trial gpt-5.6-luna high batch. Science controls are not model
qualification. Preserve and review every native attempt.

Reproduction from the staged root:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/validate_superconducting_heat.py --noise-trials 256

Results are recorded in `results/superconducting-heat-r4-validation.json`.
Prototype scripts and results are preserved with their provenance separately.

## Primary background and provenance

[Yamashita et al., Spin transport and relaxation in superconductors](https://arxiv.org/pdf/cond-mat/0108303),
particularly Eq.5 and its energy-dependent spin relaxation time, supports the
specified collision convention and opposite spin-potential populations.
The apparatus here is a distinct circular-drive kinetic model; the paper's
linear-response intrinsic-superconductor analysis does not justify retaining
an intrinsic gap at large spin accumulation.

[Baran et al., Subgap dynamics of double quantum dot coupled between superconducting and normal leads](https://www.nature.com/articles/s41598-021-90080-2.pdf),
Eq.3, explains the static induced pairing self-energy in the infinite-parent-gap
limit. Its noninteracting construction supports pinned pairing without a
local gap self-consistency equation. The continuum island DOS and rotating
magnetic contact here are explicitly specified idealizations, not a claim
that the paper studied this same driven apparatus.

Independent source review is performed by the root physics reviewer before
freeze. The actual finite-population matrices, heat definition and relaxation
normalization were reviewed against H_T; full validation results and the final
review must accompany any evaluation summary.
