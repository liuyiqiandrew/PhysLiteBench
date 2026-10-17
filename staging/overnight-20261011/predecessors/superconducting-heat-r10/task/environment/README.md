# Phase-contrast heat exchange through a quantum electromagnetic environment

Model.predict supports three statistics: `mean`, `phase_contrast`, and
`gate_noise`. The last section defines the stochastic spin dynamics and
macroscopic limit required for `gate_noise`. Read all apparatus sections.

Two noninteracting metallic islands L and R have flat normal electronic
state densities, equal charge chemical potentials, and spin-singlet proximity
pairing. Use one fixed laboratory spin basis, with up and down along z.
The quasiparticle spectrum is spin degenerate. There is no Zeeman field,
spin-orbit coupling, charging energy, or quasiparticle lifetime broadening.
There is no separate phonon or photon heat path between the islands. The
local electromagnetic baths below exchange energy only through the specified
electron tunneling Hamiltonian.

Each island has no intrinsic attractive interaction. It is coupled to its own
rigid bulk singlet parent in the infinite-parent-gap limit: integrating out the
parent produces a static anomalous pairing term and no propagating
quasiparticle channel. The parent phase and charge chemical potential are
fixed; no quasiparticle or spin leaks into the parent. Before measurement, the
proximity coupling is set to the specified gap at the prescribed island
temperature, then held fixed. Island quasiparticle populations do not determine
that coupling. A stiff controller holds the stated constant difference of the parent-imposed
pairing phases, with zero applied dc voltage. It does not cancel the quantum
phase operator in the contact Hamiltonian below. Neither parent nor phase
controller supplies steady electrical power.

The island Hamiltonian is
`H_j=sum_(k,s) xi_jk c†_jks c_jks + sum_k [Delta_j exp(i phi_j) c†_jk_up c†_j,-k,down + h.c.]`.
Its positive excitation energy is `E_j=sqrt(xi_jk^2+Delta_j^2)`.
Write `d_j=Delta_j/k_B` and express excitation energies and spin potentials in
kelvin below. The known induced gaps are
`d_j(T)=1.764*T*_j*tanh(1.74*sqrt(T*_j/T-1))`, where
`T*_L=1.2 K` and `T*_R=1.9 K` are supplied coupling scales. This is a prescribed
induced-gap law, including in a spin-polarized stationary state.

Separate spin-conserving thermostats keep the island temperatures fixed.
Fast spin-conserving scattering establishes Fermi quasiparticle distributions.
Fast opposite-spin recombination and excitation establish zero common
quasiparticle chemical potential. They conserve the number difference
`N_up-N_down`. The two populations therefore have the form
`f_js(E)=1/[1+exp((E/k_B-s*mu_j)/T_j)]`, with `s=+1` for up and `s=-1` for down.
The spin potentials `mu_L,mu_R` are internal states, not imposed settings or
external spin batteries. They are fixed by stationary balance of spin injection
and the following spin relaxation. No reservoir fixes them to zero during a
driven measurement. Recombination and thermostat energy flows are local to each
island; the contact detector defined below does not measure those local ports.

The known positive-energy quasiparticle state density, for each spin and each
island, is `nu=1.0e6 states/K` times `N_j(E)=E/sqrt(E^2-Delta_j^2)` above the gap.
Here nu includes both signs of normal dispersion xi; it is twice the normal
electron state density per spin per kelvin. Equal-energy spin flips exchange
angular momentum with the lattice without changing electronic excitation
energy. Their collision kernel is
`(df_js(E)/dt)_flip=-(f_js(E)-f_j,-s(E))/(2*tau_j(E))`, where
`tau_j(E)=tau_n*E/sqrt(E^2-Delta_j^2)` and `tau_n=1.0e-5 s`.
Thus tau_n is the normal-state decay time of spin polarization; the individual
up-to-down and down-to-up flip rates are each `1/(2*tau_j(E))`. There are no
other spin sources or losses. Number and energy equilibration is much faster
than tunneling and spin relaxation and preserves the two Fermi forms above.

The contact consists of many independent weak elementary tunnel cells with
orthogonal transverse electron channels. They connect the same two islands
and share the prescribed classical drive and pairing phase difference. Cell
`i` has its own identical electromagnetic mode and local thermal bath. The
entire interisland coupling is
`H_T(t)=sum_(i,k,q) [c†_Lik T_i(t) D_i c_Riq+h.c.]`, with spinors
`c_jik=(c_jik_up,c_jik_down)^T`,
`T_i(t)=t0_i*[I+a*(sigma_x*cos(Omega*t)+sigma_y*sin(Omega*t))]`, and
`D_i=exp[i*sqrt(rho)*(b_i+b†_i)]`.
Here `I=[[1,0],[0,1]]`, `sigma_x=[[0,1],[1,0]]`,
`sigma_y=[[0,-i],[i,0]]`, real t0_i is energy independent, and
`[b_i,b†_j]=delta_ij`. The oscillator Hamiltonian is
`H_osc=sum_i hbar*omega0*(b†_i*b_i+1/2)`, with the known energy
`hbar*omega0/k_B=24 K`. Rho is a known dimensionless phase coupling. A mode
commutes with all electron operators; Hermitian conjugation in H_T also
conjugates D_i.

Each oscillator is maintained in its stationary thermal state at the known
`oscillator_temperature`, independently of the island temperatures. Its
thermal Fock probabilities are
`p_m=(1-r)*r^m`, `r=exp[-24 K/oscillator_temperature]`. There is no oscillator
chemical battery or classical voltage bias. Use the equilibrium, sharp-line
limit with each cell's electron-transition rate much slower than its local
bath equilibration, and the bath linewidth much smaller than omega0 and all
relevant spectral separations. This hierarchy is possible with many weak
cells: at fixed total conductance, their number can increase while the
per-cell transition rate and linewidth tend to zero, with their ratio tending
to zero. It does not require a single oscillator carrying the full contact
conductance to be reset at an infinite rate. The phases, rho, mode frequency,
and bath temperature are identical across cells; rates add without coherent
interference between different transverse channels.

The externally prescribed rotating magnetic barrier acts only on the
specified tunneling spin matrix. The classical gate supplies or receives
energy and angular momentum. It does not impose a field, voltage, gap change,
or distribution in either island. Use the leading nonzero single-electron
transition probabilities in the small individual cell transparencies, which
remain small even if an instantaneous spin eigenvalue doubles t0_i. Retain
all coherent contributions from the stated singlet pairing. Do not expand in
rho. Coherent condensate energy transfer to oscillator baths beyond this
order in each cell's transparency is neglected. Spin populations vary
negligibly over one gate period. Both the spin kinetic state and the periodic
contact response are stationary before a measurement.

The ordinary `mean` readout is cycle-averaged electronic energy power lost
by L through H_T, relative to the common charge chemical potential. It is in
picowatts; negative means L receives energy. The detector counts the energy
of every contact transition, including exchange with the classical gate and
oscillator baths. It does not subtract an internal spin chemical potential
as external chemical work. Neither gate work nor oscillator-bath heat is a
separately added detector channel. Reversible interaction energy returns to
its initial value after a complete period.

For a `phase_contrast` readout, prepare two independent measurements with the
same temperatures, drive, rho and oscillator baths, at pairing phases phi
and phi+pi respectively. Allow the spin populations in each preparation to
reach their own stationary values; they need not be the same in the two
measurements. The output is half the difference of their ordinary mean
readouts: `[Q_L(phi)-Q_L(phi+pi)]/2`, in pW. The second phase is equivalent
modulo 2pi. This is a superconducting heat-interference measurement; it uses
the same contact energy detector.

Island temperatures are in `[0.25,1.05] K`, phase difference in `[-pi,pi]`
radians, and amplitude a in `[0,1]`. The frequency input
`drive_energy=hbar*Omega/k_B` is in `[0.1,0.5]` or `[7,12] K`; these bands avoid
aligned ideal gap edges over the entire temperature domain. Amplitude zero
switches off the rotating part of the barrier. Oscillator coupling rho is in
`[0,1]`, and oscillator temperature is in `[12,24] K`. Nonzero oscillator
sideband shifts have magnitude at least `24-12=12 K`; neither their transfer
gap edges nor pair thresholds align in these domains. Rho zero switches off
the oscillator coupling. Use `k_B=1.380649e-23 J/K`,
`e=1.602176634e-19 C`, and `hbar=1.054571817e-34 J s`.

The sole unknown is the normal-state electrical conductance of the unmodulated
barrier at rho=0, including both electron spins and the sum of all cell
conductances, `conductance`, in microsiemens between
20 and 80. It is the same in all experiments. Nu, tau_n, and the induced gaps
are known apparatus quantities, as are rho, oscillator frequency and bath
temperature. No spin or oscillator state is a fitted parameter.

## Interface

`Model.fit(records)` returns `self` and sets `conductance`. Each record has
`input`, `value`, and `sigma`, with value and independent Gaussian standard
uncertainty sigma in pW. Each input supplies `left_temperature`,
`right_temperature`, `phase`, and optionally `amplitude` and `drive_energy`.
The additional optional fields are `oscillator_coupling` (rho),
`oscillator_temperature`, and `statistic`. Missing amplitude means zero,
missing drive_energy means 0.3 K, missing oscillator_coupling means zero,
missing oscillator_temperature means 12 K, and missing statistic means
`"mean"`. Statistic is either `"mean"` or `"phase_contrast"`. These defaults
apply to the original unmodulated, oscillator-decoupled calibration records.

`Model.predict(experiments)` accepts a list of input dictionaries and returns a
finite NumPy array of shape `(len(experiments),)` in pW. You may replace the
entire implementation within this API. Calibration is in
`data/calibration.json`. Run `python -m pytest -q test_public.py` for interface
and calibration checks.


## Optional second electromagnetic mode

Each cell may contain an additional independent harmonic mode c_i with hbar*omega1/k_B=48 K. Its optional dimensionless secondary_coupling is in [0,1] and defaults to zero. When this coupling is nonzero, replace D_i in the same tunneling Hamiltonian by

    D_i=exp{i[sqrt(rho)*(b_i+b_i†)
                +sqrt(secondary_coupling)*(c_i+c_i†)]}.

The oscillator Hamiltonian includes hbar*omega1*(c_i†*c_i+1/2) for every cell. Different modes commute. Both modes are in independent thermal states at the same supplied oscillator_temperature. The second mode has thermal Fock probabilities (1-r1)*r1^m, r1=exp[-48 K/oscillator_temperature]. The same weak-cell equilibration and sharp-line hierarchy applies independently to each mode. All joint transitions allowed by this displacement operator contribute at the same leading tunneling order, without a coupling expansion. Final oscillator states are unobserved. There is no additional detector port, drive, spin source, or fitted parameter.

The original mean and phase_contrast readouts include electronic energy transfer through the full contact Hamiltonian, with both oscillator baths present. Each phase preparation again reaches its own stationary spin state. Calibration has both couplings zero. The other ranges, units, phase preparation, and API are unchanged. With mode energies 24 K and 48 K, nonzero sideband energies remain multiples of 24 K, so the stated spectral separation from electronic thresholds still applies.


## Nonlinear phase coupling

The optional real dimensionless quadratic_coupling is in [-.2,.2] and defaults
to zero. It changes the first mode's phase coupling, with X_i=b_i+b_i†, to

    D_i=exp{i[sqrt(rho)*X_i + quadratic_coupling*X_i²
                +sqrt(secondary_coupling)*(c_i+c_i†)]}.

This is the full unitary operator in H_T. Hermitian conjugation conjugates the
entire operator. It is a static nonlinear contact coupling: neither oscillator
Hamiltonian, oscillator thermal state, nor island spectrum is changed. The
phase controller fixes phi=phi_L-phi_R in H_j as written, including when the
contact operator has a nonzero thermal mean phase; it adds no compensating
phase shift. All joint final oscillator states remain unobserved. Use the same
leading single-electron transition order and retain all contributions from
this specified operator without expanding its exponential. There is no new
fitted parameter, heat path, or work source. Every other preparation and
measurement rule, including separate spin equilibration for phase contrast,
continues to apply. Calibration has quadratic_coupling=0.


## Ordered coupling to conjugate oscillator quadratures

An optional dimensionless momentum_coupling chi lies in [-.6,.6] and defaults
to zero. For nonzero chi the first-mode part of the tunnel operator is an
ordered product. Define X_i=b_i+b_i† and P_i=i*(b_i†-b_i), so [X_i,P_i]=2i.
The full operator in H_T is

    D_i = exp{i[sqrt(rho)*X_i + quadratic_coupling*X_i²]}
          exp{i*momentum_coupling*P_i}
          exp{i*sqrt(secondary_coupling)*(c_i+c_i†)}.

Factors act on a ket from right to left exactly in this order. This formula
replaces the preceding single-exponential definition when chi is nonzero
and reduces to it when chi is zero. It is a fixed static contact operator,
with no separate time pulses or classical noisy field. The oscillator
Hamiltonians and their thermal states are unchanged. The contact remains
Hermitian through the stated h.c. term, which conjugates the entire product.
The parent phase convention and controller are unchanged and add no phase
compensation. There is no extra work source, energy path, fitted parameter,
or observation of oscillator final states. All earlier ranges, weak-cell
limits, measurements and kinetic equations apply. Calibration has chi=0.


## Long-time noise of classical gate work

The additional statistic="gate_noise" measures fluctuations of the work
supplied by the prescribed rotating magnetic barrier. At leading tunnel
order, count the work n*hbar*Omega supplied in each Floquet transition with
harmonic n; the reverse transition has opposite work. Equivalently this is
the stochastic energy delivered by the classical gate through H_T. Static
barrier transitions have n=0. Oscillator-bath exchange is separate and is not
added to gate work. Work may be positive or negative event by event. No
electronic or oscillator state is measured or reset by this detector.

For this noise measurement retain fluctuations of the island spin populations.
Conditional on the instantaneous spin populations, the same rapid local
equilibration gives the stated Fermi forms, with mu_j determined by

    S_j=N_up-N_down=nu*integral_(d_j)^infinity
          [E/sqrt(E^2-d_j^2)]*[f_j,+ (E)-f_j,- (E)] dE.

Single-electron tunneling events have the conditional leading-order quantum
transition rates of H_T, including Pauli blocking, both directions and every
allowed oscillator transition. At this weak-cell order their event counts
are independent conditional Poisson processes. Each event changes S_L and
S_R by its actual quasiparticle spin changes. The equal-energy spin flips
already specified are also stochastic transitions with occupation factors
f_j,s(E)*[1-f_j,-s(E)] and per-state rate 1/(2*tau_j(E)); each changes S_j by
minus 2*s. Fast recombination and temperature equilibration conserve S_j.
There are no other spin sources, spin clamps, or detector fluctuations.

The requested result is the leading macroscopic noise. To define it precisely,
multiply both nu and the number of independent tunnel cells by a common size
factor Lambda, so the total conductance is Lambda*conductance. Keep every
temperature, coupling, drive, per-cell transparency and spin-relaxation time
fixed. This scaling leaves the stationary intensive state unchanged. At each
finite Lambda first reach stationarity and take the long observation-time
limit, then report

    limit_(Lambda->infinity) (1/Lambda) limit_(t->infinity)
        Var[W_gate(t)]/t.

The units are pW^2*s (joule^2/second times 10^24). The centering is over the
full stationary ensemble, including tunneling, spin flips and induced spin
population fluctuations. This is zero-frequency noise with no extra factor
of two. It is not the second moment of the mean power. Lambda is a limiting
construction, not an input or new fitted parameter. The existing mean and
phase_contrast readouts retain their preceding definitions. At amplitude=0
gate work and its noise vanish. Every earlier parameter range applies.
Calibration is unchanged and contains only the original mean heat records.
The provided noise predictor uses independent event counts at fixed stationary
spin populations; all helpers may be replaced.
