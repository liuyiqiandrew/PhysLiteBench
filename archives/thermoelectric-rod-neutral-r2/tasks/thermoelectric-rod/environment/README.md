# Temperature and voltage of a graded conducting rod

A thermoelectric rod of length L=0.01 m has insulated lateral faces and negligible mechanical expansion. Its end temperatures are held by ideal reservoirs. A current source maintains a spatially uniform electrical current density J, positive along increasing x. There are no other heat paths. Electrical relaxation is faster than the temperature measurements. Local equilibrium and linear reciprocal, isotropic thermoelectric transport apply, with no magnetic field.

The volume heat capacity is 1.5e6 J/(m³ K), electrical resistivity is 2e-5 ohm m, and thermal conductivity at zero current is an unknown constant k in [0.6,2.5] W/(m K). These properties are uniform and temperature independent. The fixed material composition varies smoothly along the rod. Its measured local Seebeck coefficient is

    S(x,T)=2e-4+4e-6*(T-300)+seebeck_span*(x/L-1/2) V/K.

The known preparation control `seebeck_span` is in [-6e-4,6e-4] V/K. The grading is immobile in material coordinates. There is no diffusion of material constituents. The electric-field convention is E=rho*J+S(x,T)*dT/dx. Report the terminal voltage integral of E from left to right.

`data/calibration.npz` contains `x` (m), `t` (s), `initial` (R×X kelvin), `current` (R values in A/m²), `boundary` (R×2 end temperatures), `seebeck_span` (R values in V/K), `temperature` (R×T×X kelvin), `voltage` (R×T volts), and their independent Gaussian uncertainties `sigma_temperature` and `sigma_voltage`. Initial temperatures are linear between mesh nodes and match the reservoirs. Experimental settings are exact. Currents lie in [-2.5e5,2.5e5] A/m², preparation and reservoir temperatures in [270,350] K, and times in [0,160] s.

Implement `Model.fit(data)`, returning self and setting `conductivity`. `Model.predict(t,x,initial,current,boundary,seebeck_span)` returns a dictionary with finite NumPy arrays `temperature` of shape `(len(t),len(x))` and `voltage` of shape `(len(t),)`. The equally spaced mesh includes both ends. Times are nonnegative and increasing from preparation and need not include zero. Use one fitted conductivity for all preparations.
