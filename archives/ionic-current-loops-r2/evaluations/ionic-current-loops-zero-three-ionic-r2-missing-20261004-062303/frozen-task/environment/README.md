# Mixing around a closed electrolyte ring

An isothermal dilute ideal electrolyte fills a thin closed circular channel of constant cross section and circumference 2*pi. The coordinate x is arclength along the ring; all concentrations are uniform across its cross section and periodic in x. A stationary porous matrix holds the solvent at rest. Neglect fluid advection, reactions and ion inertia.

Species 0 and 1 are monovalent cations (z0=z1=+1), and species 2 is a monovalent anion (z2=-1). Their ideal Nernst–Planck transport has diffusion coefficients

    D0=D, D1=4*D, D2=D,

and mobilities D_i/(k_B*T). Set k_B*T and the elementary charge to one in the stated concentration, length and time units. The only unknown is the common D in [.5,2]. All three ions are present. Debye length and charge-relaxation time are negligible on the measured scales, so the bulk remains locally electroneutral, c2=c0+c1. The electrically isolated ring has no electrodes, external voltage, imposed current or changing magnetic flux. The electric response is electroquasistatic; inductive effects are negligible. The ideal continuum transport model is assumed exact at these scales.

A new preparation is specified by two entries each in `means`, `amplitudes`, `waves` and `phases`. Initially, for i=0,1,

    c_i(x)=means[i]+amplitudes[i]*cos(waves[i]*x+phases[i]),
    c_2(x)=c_0(x)+c_1(x).

Means lie in [.6,1.4]. Each amplitude has absolute value at most .85 times its mean. Wave numbers are integers 1 or 2 and phases lie in [-pi,pi]. The listed profiles describe the bulk after the fast electric relaxation and before appreciable ionic diffusion.

Species-resolved concentration imaging records the initial rate of a spatial Fourier component:

    d/dt [ (1/pi) integral_0^(2*pi)
           c_species(x,t)*cos(detector*x+detector_phase) dx ] at t=0+.

`species` is 0, 1 or 2, `detector` is an integer 1 through 4, and `detector_phase` is in [-pi,pi]. The signal is signed and has units of concentration per time. Every measurement uses an independently prepared ring. All preparation and detector controls are known exactly.

Calibration records in `data/calibration.json` contain `input`, measured `value` and independent Gaussian standard uncertainty `sigma=.001`, fixed by the instrument.

Implement `Model.fit(records)`, returning self and storing D as `diffusivity`. `Model.predict(experiments)` returns a finite NumPy array of shape `(len(experiments),)` with the specified rates in input order. Run `python -m pytest -q test_public.py`.
