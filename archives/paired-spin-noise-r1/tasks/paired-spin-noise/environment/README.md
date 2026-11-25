# Local spin detector

A three-site system has two fermionic spin orbitals per site, with operators `c_(i,s)` for sites `i=0,1,2` and spins up/down. Set hbar=k_B=1. Its defining quadratic Hamiltonian is

`H = sum_(i,j,s) h_ij c†_(i,s)c_(j,s) + sum_i [Delta_i c†_(i,up)c†_(i,down) + h.c.]`.

The real symmetric matrix h has diagonal `[-0.35,0.15,0.70]+offset`; its off-diagonal entries are `h_01=-hopping`, `h_12=-0.8*hopping`, and `h_02=-0.3*hopping`. The pair amplitudes are `Delta_i=pairing*d_i*exp(i*phase*b_i)`, with `d=[1,0.85,1.15]` and `b=[0,1,-0.4]`. Rigid proximity sources impose these amplitudes and phases. This Hamiltonian defines the complete sample model; there are no additional interactions, charging terms, or parity restrictions. Each measurement uses fresh equilibrium preparations `rho=exp(-H/temperature)/Tr(exp(-H/temperature))` on the complete fermionic occupation space.

Independent two-state probes couple weakly to one specified site's spin, `S_z=(c†_up*c_up-c†_down*c_down)/2`. A probe of positive level spacing omega starts in its excited state; its interaction is `lambda(omega)*S_z*(|g><e|+|e><g|)`. Probe spacings form a continuum with number density nu(omega), normalized by

`2*pi*nu(omega)*|lambda(omega)|^2 = gain*omega^2*exp[-(omega-center)^2/(2*width^2)]`, for `omega>0`.

The readout is the summed probe de-excitation rate in the leading weak-coupling regime after the microscopic transient and before appreciable sample or probe depletion. Positive omega is energy gained by the sample when a probe de-excites. The specified spectral weight belongs to the probes; the isolated sample has no lifetime broadening. Time is in inverse energy units. The only unknown is the common positive detector scale `gain`, between 0.6 and 1.6; all sample and probe controls are known.

Inputs have `hopping` in [0.45,1], `pairing` in [0,1.4], `offset` in [-0.25,0.25], `phase` in [-1.5,1.5] radians, `temperature` in [0.2,0.8], `site` equal to 0, 1, or 2, `center` in [0.8,4], and `width` in [0.25,0.75]. Pair amplitudes are externally imposed, rather than determined from a temperature-dependent gap equation.

## Interface

`Model.fit(records)` returns `self` and sets `gain`. Each record has `input`, `value`, and `sigma`; the latter two are the measured rate and its independent Gaussian standard uncertainty. Public calibration is `data/calibration.json`.

`Model.predict(experiments)` accepts a list of input dictionaries and returns a finite NumPy array of shape `(len(experiments),)` containing the rates in the same order.
