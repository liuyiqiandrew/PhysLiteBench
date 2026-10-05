# Candidate apparatus outline — not a task prompt

A slender rod of reference length .05 m and cross-sectional area 1e-4 m² has fixed end separation and free transverse surfaces. Axial mechanical equilibration is fast enough to neglect inertia. Its effective one-dimensional material properties already include transverse relaxation. Temperatures are departures theta=T−300 K.

The material has axial strain epsilon and a local internal viscous strain z. Its Helmholtz free energy per reference volume is

```
psi(epsilon,z,theta;x) = E0*epsilon²/2 + E1*(epsilon-z)²/2
                         - theta*[beta0(x)*epsilon+beta1(x)*(epsilon-z)]
                         - ce*theta²/(2*T0).
```

E0=E1=2e9 Pa, T0=300 K, and ce=1e6 J/(m³ K). The coefficients are beta0(x)=E0*(.002 K^-1)*[1+.65*cos(2*pi*x/L)] and beta1(x)=chi*beta0(x), where chi is a known material setting in [0,1]. The internal strain obeys

```
eta(x)*dz/dt = E1*(epsilon-z)-beta1(x)*theta,
eta(x) = E1*(12 s)*[1+.97*cos(2*pi*x/L)].
```

There are no other constitutive terms. Retain the response to first order in temperature departures and strains about the reference state, including reversible thermal coupling. Internal viscous heat production is second order. Geometry changes are neglected in heat conduction. Thermal conductivity is a common unknown k in [80,220] W/(m K); Fourier's law applies.

The lateral surface and x=L end are insulated. At t=0, the x=0 end is connected to a well-mixed body of heat capacity 6 J/K through a massless contact of known conductance H in [0,1.2] W/K. The heat rate into the rod is H*(T_body−T_surface); H=0 disconnects the body. Neither the rod nor the body exchanges heat elsewhere afterward.

Each experiment is a fresh preparation. The clamps are held fixed while preparation heaters maintain theta(x)=a+b*cos(pi*x/L)+c*cos(2*pi*x/L) until internal relaxation is complete. At t=0 those heaters are removed and the specified contact is made. The clamps remain fixed. The initial profile and initial body departure have magnitude at most 1 K.

The intended API retains the current rod's temperature readouts: spatial mean, unnormalized first and second cosine moments (integral divided by L), and the body temperature departure. Every output is in K. Inputs add the known chi setting to the existing preparation/contact/time fields; time is in [0,80] s. The fitted parameter is conductivity, without a default fitted value.

This outline gives no numerical method or correction formula. A future harness would use the repository's unchanged neutral replacement instruction. Calibration records, noise level, exact interface text, and complete scientific controls have not been packaged.
