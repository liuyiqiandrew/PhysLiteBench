# Mixing in a periodic electrolyte

An isothermal dilute ideal electrolyte occupies a two-dimensional square periodic cell of side `2*pi`. Both opposite pairs of edges are identified. The concentrations and transport are periodic in x and y and uniform across a constant channel depth. A stationary porous matrix holds the solvent at rest. Neglect fluid advection, reactions and ion inertia.

Species 0 and 1 are monovalent cations and species 2 is a monovalent anion. Their ideal Nernst–Planck transport has diffusion coefficients

    D0=D, D1=4*D, D2=D,

and mobilities D_i/(k_B*T). Set k_B*T and the elementary charge to one in the stated concentration, length and time units. The only unknown is the common D in [.5,2]. Debye length and charge-relaxation time are negligible on the measured scales, so the bulk remains locally electroneutral, c2=c0+c1. The cell is electrically isolated, without electrodes, external voltage or imposed current. No changing magnetic flux threads either periodic cycle. The electric response is electroquasistatic and inductive effects are negligible. The ideal continuum transport model is assumed exact at these scales.

Each new preparation has four known `amplitudes=[a,b,c,d]` and a known `phase`:

    c0(x,y)=1+a*cos(x)+b*cos(y),
    c1(x,y)=1+c*cos(x+phase)+d*cos(x+y),
    c2(x,y)=c0(x,y)+c1(x,y).

The amplitude ranges are a in [.5,.65], b in [.2,.3], c in [.6,.7] and d in [.15,.25]; phase lies in [-pi,pi]. These ranges keep every species concentration positive. The listed profiles describe the bulk after fast electric relaxation and before appreciable ionic diffusion.

Species-resolved concentration imaging records the initial rate

    d/dt [ 2/(2*pi)^2 * integral_cell
           c_species(x,y,t)*cos(mx+ny+pi/4) dx dy ] at t=0+.

`species` is 0, 1 or 2. The known `detector=[m,n]` has integer components from -2 through 2, not both zero. The detector phase is always pi/4. The signal is signed and has units of concentration per time. Every measurement uses an independently prepared cell. All preparation and detector controls are known exactly.

Calibration records in `data/calibration.json` contain `input`, measured `value` and independent Gaussian standard uncertainty `sigma=.001`, fixed by the instrument.

Implement `Model.fit(records)`, returning self and storing D as `self.diffusivity`. `Model.predict(experiments)` returns a finite NumPy array of shape `(len(experiments),)` with the specified rates in input order. Any numerical implementation may be used while preserving this interface.
