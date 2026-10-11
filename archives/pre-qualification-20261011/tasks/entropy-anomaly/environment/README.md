# Spatial excess calorimetry with fast alternating thermal contacts

A charged particle moves on a two-dimensional periodic square. Both x and y have period 2*pi. It has exactly two velocity components vx,vy, with the same positive mass m and no other degrees of freedom. Set Boltzmann's constant to one. Use fixed reference length, energy, mass and time units: temperatures are energies, friction and magnetic coefficients have units mass/time, forces energy/length, and the control named switch_rate also has units mass/time.

Two externally maintained local thermal reservoirs are available, but exactly one is connected to the particle at a time. The contact label s(t) is +1 or -1. At each positive mass m its actual switching rate in either direction is switch_rate/m. Waiting times are exponential and both stationary probabilities are one half. The valve process is independent of the thermal Wiener processes and its rates do not depend on the particle. The reservoir temperatures are

    T_s(x) = temperature * (1+s*bath_contrast)
             * (1+contrast*cos(wavenumber*x)).

Both reservoirs have the same drag tensor along the fixed laboratory axes. At finite positive mass the complete equations while state s is connected are

    dx = vx dt,                    dy = vy dt,
    m dvx = (force_x - friction*vx + magnetic*vy) dt
            + sqrt(2*friction*T_s(x)) dWx,
    m dvy = (force_y - drag_ratio*friction*vy - magnetic*vx) dt
            + sqrt(2*drag_ratio*friction*T_s(x)) dWy.

Wx and Wy are independent standard real Wiener processes. Position and velocity stay continuous at every contact change. The valve applies no impulse, changes no particle energy or potential, and performs no particle work at the instant of switching. There is no additional potential, mechanical force or noise. The static signed perpendicular magnetic field does no work. Constant forces may do work as the particle winds around the square. Reservoirs and valve are unaffected by the particle. Heat required to maintain the prescribed temperature profiles or operate the empty valve is separate from the particle calorimetry.

Separate local calorimeters record particle heat delivered to whichever reservoir is connected. Write gamma_x=friction and gamma_y=drag_ratio*friction. On each interval of contact with reservoir s, the recorded bath entropy increment is

    dS_bath = [sum over i=x,y of
               (gamma_i*vi dt-sqrt(2*gamma_i*T_s(x)) dWi) circ vi]
              / T_s(x),

where circ denotes the Stratonovich product. Heat and entropy entering the connected bath are positive. Add the records from both reservoirs, including all contact intervals. The local calorimeters can select a smooth position-dependent detector weight. If the optional input detector_phase is present, use

    w(x) = [1+cos(wavenumber*x-detector_phase)]/2.

If detector_phase is omitted, use w(x)=1. The detector phase is a dimensionless angle in [-pi,pi], referenced to the specified temperature profile. At positive mass the detector records w(x)*dS_bath, using the particle position at the time of the heat transfer. Detection applies no force and changes none of the dynamics. The ideal calorimeters and position measurement add no noise to the predicted rates.

The apparatus measures an excess rate using two preparations at the same finite mass. For the first preparation, use all the experiment settings as given and measure the long-time mean detector-weighted bath entropy rate after the full joint particle/contact process reaches stationarity. At that same mass and in that same preparation, independently record the actual long-time weighted residence R_m, defined by the stationary mean of w(x), equivalently the time integral of w(x) divided by the observation time. For the matched background preparation, set force_x=force_y=0 and contrast=0, keeping the same mass, common friction, drag_ratio, magnetic, temperature, bath_contrast, switch_rate and periodic square. This background has two spatially uniform temperatures temperature*(1+s*bath_contrast), the same valve rate switch_rate/m, and the same finite-mass calorimetry. Independently bring it to joint stationarity and measure its unweighted long-time mean bath entropy rate B_m over all particle positions.

Subtract B_m*R_m from the first weighted mean at that positive mass, using the actual measured residence from the first preparation, then take m -> 0 while holding every input setting, including the control switch_rate, fixed. Return this limiting excess rate per particle. The actual valve rate therefore scales as 1/m throughout the limit. The background subtraction and this order of stationarity, measurement, subtraction and mass limit define the observable. The mass is neither fitted nor supplied as a prediction input. Every requested experiment uses its own matched background preparation and its own actual weighted residence. An excess rate in a detector region may have either sign. When the detector phase is omitted, R_m=1 and this definition reduces to the unweighted excess rate.

Every input contains force_x and force_y, each in [-1.5,1.5]; drag_ratio in [.25,8]; magnetic in [-5,5]; temperature in [.8,1.4]; contrast in [0,.65]; wavenumber, an integer 1,2 or 3; bath_contrast in [0,.65]; and switch_rate in [.2,2]. Settings are exact. The only unknown is the common friction in [.7,1.6], shared by both reservoirs and every preparation. The position distribution is uniform in y and normalized over the full square.

Calibration omits detector_phase and measures the unweighted readout. It uses bath_contrast=0 and magnetic=0, with nonuniform spatial profiles, varied switch_rate controls, unequal as well as equal drag ratios and known forces. Records in data/calibration.json contain input, value and independent Gaussian standard uncertainty sigma. These measurements identify the common friction coefficient.

Implement Model.fit(records), returning self and storing Model.friction. Model.predict(experiments) must return a finite NumPy array of shape (len(experiments),), in input order, with the specified limiting excess entropy rates. Any implementation preserving this interface may be used.
