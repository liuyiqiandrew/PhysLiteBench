# Viscous Hall channel

A long two-dimensional charged fluid flows along x between stationary walls at y=-W/2 and y=W/2. Use reduced units in which carrier mass, positive charge, equilibrium number density, and equilibrium thermal velocity variance are one. The flow is steady, independent of x, and linear in the applied longitudinal electrochemical field Ex. There is no transverse particle flux. Both velocity components obey no-slip conditions at the walls. The fluid is isothermal and incompressible.

The apparatus is described by the following closed continuum moment model throughout the channel, including at its walls. Momentum density relaxes against the stationary substrate at rate 1/mobility, where the unknown positive mobility lies in [0.05,1.5]. The symmetric kinetic momentum-flux tensor P has equilibrium value I. Its traceless departure from I relaxes at the known rate 1/tau2, with tau2=0.4. The local velocity gradient transports its two tensor indices affinely. The magnetic Lorentz acceleration is (B*v_y,-B*v_x) and acts on each velocity factor in P. The divergence of the third central velocity moment is zero in this closed model. Keep terms to first order in the applied field; products of departures from equilibrium are excluded. Apart from this relaxation and the Lorentz force, momentum is driven only by the electrochemical field and the divergence of P. This is the defining continuum model, not an approximation that requires additional wall kinetic corrections.

The imposed static magnetic field is along z and varies across the channel as

    B(y) = magnetic_field + field_gradient*(2*y/W)
           + field_modulation*cos(2*pi*y/W).

Each of the three coefficients lies in [-3,3]. Width W lies in [0.4,3], and Ex lies in [-0.05,0.05]. There are no extra surface stresses or slip parameters. The isotropic thermodynamic pressure is included in the electrochemical potential. The transverse field measured below is the electric field minus the isotropic pressure gradient per unit charge density; it is not the electrostatic field alone. The momentum-flux departure from I excludes that isotropic pressure.

Each input dictionary contains `width`, `electric_field`, `magnetic_field`, `field_gradient`, `field_modulation`, and `observable`. For `observable="current"`, return the signed longitudinal current density averaged across the full width. For `observable="transverse_voltage"`, it also contains `contacts=[a,b]`, with -1<=a<b<=1 and b-a>=0.2. Return the signed integral of the transverse electrochemical field from y=a*W/2 to y=b*W/2. The contacts are ideal non-invasive probes; the sign of this integral defines the readout. Contacts and fields are exact. Report the steady linear response, with the same fitted mobility for all inputs.

Implement `Model.fit(records)`, returning self and storing the fitted parameter in `mobility`. `predict(experiments)` returns a finite NumPy array of shape (len(experiments),) in input order. Any implementation satisfying this interface may be used.

`data/calibration.json` contains zero-magnetic-field current measurements. Each record has an `input` dictionary, measured `value`, and known independent Gaussian uncertainty `sigma`. The uncertainty is fixed by the instrument independently of the unknown parameter. Run `python -m pytest -q test_public.py` to check calibration and the interface.
