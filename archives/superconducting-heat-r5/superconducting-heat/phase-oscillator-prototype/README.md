# Phase oscillator proposal: scientific derivation

This is a feasibility proposal, not an evaluated task revision. The active
revision 4 task is untouched. `prototype.py` imports constants and the
spin-relaxation law from the immutable revision 4 archive. `reference.py`
independently uses explicit thermal Fock displacement and singlet spin
matrices. `screen.py` saves the complete numerical proposal screen in
`results/superconducting-heat-r5-oscillator-proposal-screen.json`.

## Apparatus and validity

Retain both pinned, noninteracting proximity islands, the circular magnetic
barrier, both finite spin populations, and all revision 4 heat and spin
definitions. Across each weak elementary contact replace its tunnel
amplitude by

```
t0 [I + a(sigma_x cos(Omega t) + sigma_y sin(Omega t))] D,
D = exp[i sqrt(rho) (b + b^dagger)],
H_osc = k_B w0 (b^dagger b + 1/2),  w0 = 24 K.
```

The Hermitian conjugate term uses `D^dagger`; electron and hole tunneling
therefore displace the oscillator in opposite charge directions. The
oscillator is prepared thermally at a known `Tph` between 12 and 24 K;
`rho` is a known input between 0 and 1. No classical voltage or oscillator
chemical battery is applied. Spin-degenerate induced gaps and flat normal
electron density of states remain pinned, including for high-energy
excitations, under the same infinite-parent-gap idealization as revision 4.

For a physical equilibrium-reservoir and sharp-line limit, use many
independent elementary tunnel contacts with identical oscillator cells,
the same stationary thermal bath temperature, drive, and mean phase. Their
normal conductances sum to the one unknown `G`. Each cell's tunneling rate
is much smaller than its restoring bath rate, while that bath rate is much
smaller than the oscillator frequency and relevant spectral separations.
Taking many weak cells at fixed total `G` makes oscillator backaction and
linewidth negligible. This does not require an infinitely fast thermal
reset of a single mode carrying the whole conductance. Contributions from
coherent pair tunneling to bath heat are beyond the retained leading
single-electron transition probabilities in each cell's transparency.

An optional phase-contrast heat measurement prepares two otherwise
identical independent stationary devices, at phases `phi` and `phi+pi`,
allows each device's two spin populations to reach its own steady state,
and reports `[Q_L(phi)-Q_L(phi+pi)]/2`. Ordinary mean heat remains the default
readout. A zero oscillator coupling is exactly revision 4; consequently all
old calibration inputs, numerical bytes, noise, and the inferred `G` can
be retained. Contrast is a physical calorimetric measurement of the same
superconducting contact, not a changed error normalization.

## Quantum weights and scalar kernels

Let `ell>0` denote **oscillator energy gain**, `nb=1/(exp(w0/Tph)-1)`, and
`r=exp(-w0/Tph)`. Direct thermal matrix elements give

```
D_{j,m} = exp(-rho/2) sqrt(min(j,m)!/max(j,m)!)
          (i sqrt(rho))^|j-m| L_min(j,m)^|j-m|(rho),
p_m = (1-r) r^m,
P_ell = sum_m p_m |D_{m+ell,m}|^2,
(D^dagger)_{m+ell,m} = (-1)^ell D_{m+ell,m}.
```

Equivalently, the generating function is

```
sum_ell P_ell exp(-i ell w0 t)
 = exp[-rho(2nb+1)] exp[rho(nb+1)exp(-i w0 t)+rho nb exp(i w0 t)].
```

Thus `sum P_ell=1`, `P_-ell=exp(-ell w0/Tph) P_ell`, and the anomalous
electron/hole covariance in a specified energy sideband is exactly
`(-1)^ell P_ell`. Its sum is the usual static dephasing factor
`DW=exp[-2rho(2nb+1)]`. These are different energy-resolved quantities.

With `S=i sigma_y`, spin harmonics `T_0=I`, `T_+=a sigma_+`,
`T_-=a sigma_-`, and `R_n=S T_-n^* S^dagger`, the transfer matrix uses
`T_-n,R_-n` and the pair-creation matrix uses `T_n,R_n S`, as in revision 4.
The environmental electron and hole pieces use `D` and `D^dagger` in the
corresponding Hermitian-conjugate order. Averaging all four normal-dispersion
branches gives the following nonnegative scalar kernels; `den` is
`sqrt(E_L^2-d_L^2) sqrt(E_R^2-d_R^2)`:

```
transfer n=0: (E_L E_R - (-1)^ell d_L d_R cos(phi))/den,
transfer n=+/-1: (E_L E_R + (-1)^ell d_L d_R cos(phi))/den,
creation n=0: (E_L E_R + (-1)^ell d_L d_R cos(phi))/den,
creation n=+/-1: (E_L E_R - (-1)^ell d_L d_R cos(phi))/den.
```

Static transfer has two equal-spin channels of weight 1/2. Static creation
has two opposite-spin channels of weight 1/2. Driven transfer has
`(sL,sR)=(-n,n)` with weight `a^2/2`. Driven creation has `(n,n)` with weight
`a^2/2`. The common number conversion is `2 G/e^2` times an energy integral.
The Kelvin/microSiemens rate and picoWatt conversions are unchanged from
revision 4. The independent reference derives these weights from 2x2
matrices rather than using any scalar sign table.

## Energy and reverse transitions

For each classical harmonic `n` and oscillator gain `ell`, define
`delta=n*w-ell*w0`. Transfer conserves `E_R=E_L+delta`. Pair creation exists
whenever `E_L+E_R=delta>d_L+d_R`. Therefore oscillator absorption opens
**static opposite-spin creation and n=-1 down/down creation** as well as
the prior n=+1 up/up channel. Pair annihilation is the reverse of each
listed positive-energy creation channel; it must not be enumerated twice.

The net occupations multiply the same nonnegative matrix kernel:

```
transfer: P_ell f_L(1-f_R) - P_-ell f_R(1-f_L),
creation: P_ell (1-f_L)(1-f_R) - P_-ell f_L f_R.
```

The net event contributes left/right electronic heat loss `(E_L,-E_R)`
for transfer or `(-E_L,-E_R)` for creation, classical work `n*w`, and
oscillator energy gain `ell*w0`. Consequently
`Q_L+Q_R+P_gate-Q_env=0`. The spin balance remains `I_j=R_j(mu_j)` and
`P_gate=k_B*w*(I_L+I_R)/2`: every driven event changes total spin by `2n`,
whereas static opposite-spin creation has zero total spin.

Because each reverse weight satisfies thermal detailed balance, every
transition has nonnegative `(forward-reverse)*log(forward/reverse)`.
Spin-flip entropy remains positive. At stationary spin balance their
chemical-potential terms cancel and total reservoir entropy production is
`-Q_L/TL-Q_R/TR+Q_env/Tph`, with electronic heat measured relative to the
fixed charge chemical potential, as in revision 4.

Every individual spin-injection Jacobian has a negative diagonal and equal
magnitude off-diagonal contribution in each column, even when the two
oscillator probabilities differ. Summation gives weak column diagonal
dominance; finite positive local spin relaxation makes it strict. The
integrated Jacobian on the segment between two possible roots is also
strictly dominant, proving uniqueness. Large-polarization slopes retain
the positive definite revision 4 normal-metal matrix because the complete
oscillator probabilities sum to unity; bounded recoil moments shift only
the forcing. An expanding-box solver must check the unconstrained balance,
rather than assuming `|mu|<=w`.

## Independent normal-state limit

With both gaps set to zero and both electronic temperatures equal to `T`,
the spin-symmetric stationary solution has `mu_L=mu_R=mu`. Define
`F_T(x)=x/(1-exp(-x/T))`, with its continuous value `F_T(0)=T`, and

```
J(v) = sum_ell P_ell [F_T(v-ell*w0)-F_T(-v-ell*w0)].
```

The two spin balances reduce to the one scalar monotone equation
`(nu/tau_n) mu = (2G k_B/e^2) (a^2/2) J(w-2mu)`, with conductance in siemens.
At `rho=0`, `J(v)=v` and this is exactly the revision 4 closed normal-state
solution. The proposal numerically checks this scalar signed-electron
phase-space identity independently of the positive-energy BCS quadrature.
Both scalar closures also give identical complete normal-state heat because
their electron/hole anomalous term vanishes. The oscillator moments obey
`mean(ell)=rho` and `var(ell)=rho(2nb+1)`.

## Coherent shortcut

The completed shortcut retains correct `P_ell`, detailed balance, every
transfer and pair channel, all heat reservoirs, and full finite stationary
spin feedback. It assumes independent phase dephasing and energy exchange,
replacing the **conditional** covariance `(-1)^ell P_ell` by `DW P_ell`.
Each channel covariance matrix then has eigenvalues `P_ell(1+/-DW)>=0`;
its rates are nonnegative and it has the same detailed balance and entropy
identities. It is a complete factorized approximation, not an erroneous
normalization or an omitted known channel. It is exact at `rho=0`, in the
normal state where the anomalous factor vanishes, and for the sum of the
environmental anomalous weights, but loses their correlation with exchanged
oscillator energy. The new readout tests that correlation.

## Primary scientific background

[Martinis, quasiparticle P(E) supplement, Eq. 11 and Eq. 21](https://web.physics.ucsb.edu/~martinisgroup/papers/Martinis2009supp.pdf)
derives opposite charge displacement operators and their two environmental
correlators in quasiparticle tunneling.
[Ingold and Nazarov, Eq. 99–101](https://arxiv.org/pdf/cond-mat/0508728)
gives finite-temperature single-oscillator exchange weights and their
Poisson emission/absorption construction.
[Ansari et al., Eq. 14](https://arxiv.org/pdf/1211.4745)
separates ordinary and electron/hole interference environmental channels
nonperturbatively. These works support the mechanism; the driven magnetic
barrier, pinned proximity islands and phase-contrast apparatus above are
our explicitly specified proposed construction, not an apparatus claimed
to be in those papers.
