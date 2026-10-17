# Calorimetry with alternating thermal contacts

A charged particle moves on a two-dimensional periodic square. Both x and y have period 2*pi. It has exactly two velocity components vx,vy, with the same positive mass m and no other degrees of freedom. Set Boltzmann's constant to one. Use fixed reference length, energy and time units: temperatures are energies, friction and magnetic coefficients have units mass/time, forces energy/length, and switch_rate an inverse time.

Two externally maintained local thermal reservoirs are available, but exactly one is connected to the particle at a time. The contact label s(t) is +1 or -1. In either state it switches to the other with the known rate switch_rate, so its waiting times are exponential and both stationary probabilities are one half. The valve process is independent of the thermal Wiener processes and its rates do not depend on the particle. The reservoir temperatures are

    T_s(x) = temperature * (1+s*bath_contrast)
             * (1+contrast*cos(wavenumber*x)).

Both reservoirs have the same drag tensor along the fixed laboratory axes. At finite positive mass the complete equations while state s is connected are

    dx = vx dt,                    dy = vy dt,
    m dvx = (force_x - friction*vx + magnetic*vy) dt
            + sqrt(2*friction*T_s(x)) dWx,
    m dvy = (force_y - drag_ratio*friction*vy - magnetic*vx) dt
            + sqrt(2*drag_ratio*friction*T_s(x)) dWy.

Wx and Wy are independent standard real Wiener processes. Position and velocity stay continuous at every contact change. The valve applies no impulse, changes no particle energy or potential, and performs no particle work at the instant of switching. There is no additional potential, mechanical force or noise. The static signed perpendicular magnetic field does no work. Constant forces may do work as the particle winds around the square. Reservoirs and valve are unaffected by the particle. Heat required to maintain the spatial temperature profiles or operate the empty valve is subtracted as background.

Separate local calorimeters record particle heat delivered to whichever reservoir is connected. Write gamma_x=friction and gamma_y=drag_ratio*friction. On each interval of contact with reservoir s, the recorded bath entropy increment is

    dS_bath = [sum over i=x,y of
               (gamma_i*vi dt-sqrt(2*gamma_i*T_s(x)) dWi) circ vi]
              / T_s(x),

where circ denotes the Stratonovich product. Heat and entropy entering the connected bath are positive. Add the records from both reservoirs, including all contact intervals. Prepare the full joint particle/contact stationary state at fixed positive m, then measure the long-time mean entropy rate, and finally take m -> 0 with every other setting, including switch_rate, fixed. Return this limiting rate per particle. The mass is neither fitted nor supplied as a prediction input. The order of limits and finite-mass calorimetry are part of the apparatus definition. Each experiment is independently brought to joint stationarity.

Every input contains force_x and force_y, each in [-1.5,1.5]; drag_ratio in [.25,8]; magnetic in [-5,5]; temperature in [.8,1.4]; contrast in [0,.65]; wavenumber, an integer 1,2 or3; bath_contrast in [0,.35]; and switch_rate in [.2,2]. Settings are exact. The only unknown is the common friction in [.7,1.6], shared by both reservoirs and every experiment. The position distribution is uniform in y and normalized over the full square.

Calibration uses bath_contrast=0 and magnetic=0, with nonuniform spatial profiles, varied switching rates, unequal as well as equal drag ratios and known forces. Records in data/calibration.json contain input, value and independent Gaussian standard uncertainty sigma. These measurements identify the common friction coefficient.

Implement Model.fit(records), returning self and storing Model.friction. Model.predict(experiments) must return a finite NumPy array of shape (len(experiments),), in input order, with the specified limiting entropy rates. Any implementation preserving this interface may be used.
