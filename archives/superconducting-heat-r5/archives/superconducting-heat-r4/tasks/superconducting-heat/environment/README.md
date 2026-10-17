# Heat exchange through a driven magnetic proximity tunnel contact

Two noninteracting metallic islands L and R have flat normal electronic
state densities, equal charge chemical potentials, and spin-singlet proximity
pairing. Use one fixed laboratory spin basis, with up and down along z.
The quasiparticle spectrum is spin degenerate. There is no Zeeman field,
spin-orbit coupling, charging energy, or quasiparticle lifetime broadening.
There is no parallel phonon or photon heat channel across the measured contact.

Each island has no intrinsic attractive interaction. It is coupled to its own
rigid bulk singlet parent in the infinite-parent-gap limit: integrating out the
parent produces a static anomalous pairing term and no propagating
quasiparticle channel. The parent phase and charge chemical potential are
fixed; no quasiparticle or spin leaks into the parent. Before measurement, the
proximity coupling is set to the specified gap at the prescribed island
temperature, then held fixed. Island quasiparticle populations do not determine
that coupling. A stiff phase controller holds the stated constant phase
difference at zero voltage. Neither parent nor phase controller supplies
steady electrical power.

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

The entire interisland coupling is
`H_T(t)=sum_(k,q) [c†_Lk T(t) c_Rq+h.c.]`, with spinors
`c_jk=(c_jk_up,c_jk_down)^T` and
`T(t)=t0*[I+a*(sigma_x*cos(Omega*t)+sigma_y*sin(Omega*t))]`.
Here `I=[[1,0],[0,1]]`, `sigma_x=[[0,1],[1,0]]` and
`sigma_y=[[0,-i],[i,0]]`; real t0 is energy independent. This externally
prescribed rotating magnetic barrier acts only on the tunneling spin matrix.
The classical gate supplies the associated energy and angular momentum. It
does not impose a field, voltage, gap change, or distribution in an island.
Use the leading nonzero order in small individual tunneling transparencies,
which remain small even when an instantaneous eigenvalue of T reaches twice
t0. Coherent transition probabilities retain the specified singlet pairing.
Spin populations vary negligibly over one gate period. The stationary kinetic
state and periodic contact response are established before measurement.

The readout is cycle-averaged electronic energy power lost by L through H_T,
relative to the common charge chemical potential. It is in picowatts; negative
means L receives energy. This detector counts the energy of every contact
transition, including energy received from the gate. It does not subtract a
spin chemical potential as external chemical work. Gate work is not a
separately added detector channel. Reversible interaction energy returns to
its initial value after a complete period.

Island temperatures are in `[0.25,1.05] K`, phase difference in `[-pi,pi]`
radians, and amplitude a in `[0,1]`. The frequency input
`drive_energy=hbar*Omega/k_B` is in `[0.1,0.5]` or `[7,12] K`; these bands avoid
aligned ideal gap edges over the entire temperature domain. Amplitude zero
switches off the rotating part of the barrier. Use `k_B=1.380649e-23 J/K`,
`e=1.602176634e-19 C`, and `hbar=1.054571817e-34 J s`.

The sole unknown is the normal-state electrical conductance of the unmodulated
barrier, including both electron spins, `conductance`, in microsiemens between
20 and 80. It is the same in all experiments. Nu, tau_n, and the induced gaps
are known apparatus quantities.

## Interface

`Model.fit(records)` returns `self` and sets `conductance`. Each record has
`input`, `value`, and `sigma`, with value and independent Gaussian standard
uncertainty sigma in pW. Each input supplies `left_temperature`,
`right_temperature`, `phase`, and optionally `amplitude` and `drive_energy`.
Missing amplitude means zero; missing drive_energy means 0.3 K. These defaults
apply to the unmodulated calibration records.

`Model.predict(experiments)` accepts a list of input dictionaries and returns a
finite NumPy array of shape `(len(experiments),)` in pW. You may replace the
entire implementation within this API. Calibration is in
`data/calibration.json`. Run `python -m pytest -q test_public.py` for interface
and calibration checks.
