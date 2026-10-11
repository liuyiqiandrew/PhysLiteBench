# Electrolyte cell with reversible neutral association

A quiescent, isothermal ideal dilute solution occupies a uniform one-dimensional cell, 0 ≤ x ≤ 0.001 m. Free ions A+, B+, and C- have diffusion coefficients D, 4D, and 2D, with D unknown. They can reversibly associate into a neutral four-ion cluster P through A+ + B+ + 2 C- <-> P. The neutral cluster has diffusion coefficient 0.15D. All species are ideal, with no other interactions or reactions. The cluster transports one A constituent, one B constituent, and two C constituents and carries no electric charge. Both ends block every species and are electrically insulating. No current is applied and there is no fluid flow.

The measurements resolve electroneutral bulk transport; Debye layers and charge-relaxation transients are outside the model. The free-ion concentrations satisfy c_C=c_A+c_B. Each free ion's diffusive and electric drift flux obeys the isothermal Nernst-Planck law with its stated diffusion coefficient and valence; the neutral cluster has only Fickian diffusion. The electric field is the self-consistent field that keeps the total electric current zero. Constituent conservation includes both free and bound species.

Association equilibrates locally much faster than transport. Its ideal mass-action equilibrium is p=K*c_A*c_B*c_C² with K=0.5 (m³/mol)³, where p is the cluster concentration. More precisely, for any finite dissociation rate k_off the net cluster production is k_off*(K*c_A*c_B*c_C²-p); take k_off -> infinity with D, K, geometry, observation times, and total constituent preparations fixed. Every preparation starts in this local chemical equilibrium. Only D is fitted. Binding energy is exchanged with the isothermal thermostat, and neither temperature nor diffusion coefficients depend on reaction heat.

`data/calibration.npz` contains x (m), t (s), initial (mol m⁻³; shape R×X×2), concentration (R×T×X×2), and sigma (same shape). The two measured channels are TOTAL A and TOTAL B constituent concentrations: a_total=c_A+p and b_total=c_B+p. Both initial and concentration use these total concentrations, not just free ions. Measurements have independent Gaussian errors. Initial total concentrations are known and linear between mesh points. Calibration preparations contain only one cation constituent, so they have no clusters; the same apparatus applies to mixed preparations. Total C equals a_total+b_total and is not an independent channel.

TransportModel.fit(data) accepts this dictionary, returns self, and stores positive D in diffusivity (m²/s). predict(t,x,initial) returns a finite NumPy array with shape (len(t),len(x),2), containing the two total constituent concentrations. x is an equally spaced mesh spanning the cell. t contains increasing nonnegative times measured from preparation. Run python -m pytest -q test_public.py. Any implementation preserving this interface may be used.


## Equilibrium concentration fluctuation measurement

The same cell also supports equilibrium fluctuation spectroscopy. Prepare a
spatially uniform mixture with prescribed positive mean total constituent
concentrations totals=(a_total,b_total), each in [.2,2] mol m^-3. The total
number of each constituent in the closed cell is fixed. Local association is
equilibrated as above. Take a macroscopic cross-sectional area A_cs, uniform
along the cell, and retain the leading Gaussian fluctuations as particle
number tends to infinity at fixed concentrations. The mean state remains the
stated ideal dilute equilibrium; there are no additional interactions.

For finite reaction and charge relaxation, the Brownian number flux noises of
the four species are independent and have the standard ideal-particle
amplitudes sqrt(2*D_i*c_i/(N_A*A_cs)). Forward and reverse association events
are independent Poisson events with local mean rates k_off*K*c_A*c_B*c_C^2
and k_off*p per volume. Each event changes species numbers by the stated
stoichiometry, conserving charge and both measured constituents. N_A is
Avogadro's constant. The electric field is self-consistent also for the
fluctuating flux: in the electroneutral bulk limit it cancels the total
instantaneous charge current, including its Brownian contribution. Ends
block all species. Take charge and chemical relaxation much faster than
each resolved diffusion mode, as in the preceding bulk transport model.
No external electric, chemical or stochastic drive is applied.

Let delta c_i^tot be a total-constituent fluctuation. For positive integer h,
define the real mode amplitude

    u_i,h(t)=(1/L)*integral_0^L delta c_i^tot(x,t)*cos(h*pi*x/L) dx.

The detector reports the full two-time equilibrium covariance matrix

    S_ij,h(t)=2*N_A*A_cs*L*E[u_i,h(t)*u_j,h(0)],   i,j in {A,B}.

This normalization removes particle-number and volume factors. Each output
entry has units mol m^-3. Time is a lag t>=0, not an age after a perturbation.
Stationarity and thermal equilibrium are established before either sample.
There is no independent detector noise. The zero spatial mode is excluded
because both global constituent amounts are fixed.

Implement TransportModel.correlate(t,totals,harmonic=1), returning a finite
NumPy array of shape (len(t),2,2), in A,B order for both matrix indices.
harmonic is an integer in {1,...,8}; t is an increasing nonnegative array up
to 600 s and may include zero. This method uses the same fitted diffusivity;
there is no new unknown. Existing fit and mean-profile predict interfaces and
all calibration records are unchanged. The provided correlation predictor
uses independent ideal total-constituent number fluctuations at equal time.
All helpers may be replaced.
