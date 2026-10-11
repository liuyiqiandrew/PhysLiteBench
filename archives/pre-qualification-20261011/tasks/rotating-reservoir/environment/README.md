# Thermal calorimetry of a rotor with a finite rotating gas contact

A mass-one rotor is constrained to a fixed circular ring of radius one. Its angular coordinate theta has period 2*pi, its signed tangential speed is v=dtheta/dt, its moment of inertia is one, and its laboratory kinetic energy is v²/2. The ideal radial constraint does no work. Set Boltzmann's constant to one and use fixed reference mass, length and time units. Temperatures are energies, velocities use the reference velocity unit, and all rates use inverse reference time. Velocity is unbounded. There is no potential or additional torque.

The contact contains one persistent effective mass-one tangential gas degree of freedom, with signed velocity g and laboratory kinetic energy g²/2. It is retained throughout a record. Reservoir A thermalizes the rotor in the laboratory frame with a common unknown drag in [.4,1.1]. Reservoir B thermalizes the retained gas contact toward the flow u=s*angular_speed with a known relaxation_rate. Between collisions and motor switches the complete dynamics are

    dtheta = v dt,
    dv = -drag*v dt + sqrt(2*drag*temperature_a) dW_A,
    dg = -relaxation_rate*(g-u) dt
         + sqrt(2*relaxation_rate*temperature_b) dW_B.

W_A and W_B are independent standard real Wiener processes. The independent motor label s takes values +1 and -1 and switches in either direction at flip_rate, with stationary probabilities one half. At a switch theta, v and g remain continuous. The switch changes only the thermostat's flow target; it gives neither particle an impulse or instantaneous mechanical work and deposits no instantaneous calorimeter heat. The motor is externally prescribed and unaffected by the particles.

The entire random collision contact is specified by its instantaneous hazard, evaluated at the current velocities:

    3.6 * [(1-eta) + eta*abs(v-g)/v_star],
    eta = flux_fraction,
    v_star = 1 in the fixed reference velocity units.

Its first channel has a constant clock and its second channel counts relative-speed flux. Conditional collisions are Poisson events according to this current-state hazard. At an event the equal effective masses elastically exchange tangential velocities:

    v_after = g_before,
    g_after = v_before.

Theta remains continuous. This exchange conserves the pair's tangential momentum and laboratory kinetic energy. The gas contact retains its outgoing velocity until subsequent OU evolution, switches and collisions. There is no fresh gas draw at a collision, no reset after an event, no other gas interaction or contact state, and no additional kinetic component. The stated hazard and exchange are the complete ideal contact law, without an extra spatial collision condition. Conditional event clocks, W_A, W_B and the motor are independent except for the stated dependence of the hazard on v and g.

The calorimeter measures heat deposited into the thermal reservoir B by the gas contact's OU thermostat force, in the moving flow frame. On intervals of fixed s its increment is the Stratonovich product

    dQ_B = -(g-u) circ [-relaxation_rate*(g-u) dt
                       + sqrt(2*relaxation_rate*temperature_b) dW_B].

Equivalently the Ito increment, which specifies the measurement jointly with the particle dynamics, is

    dQ_B = relaxation_rate*[(g-u)²-temperature_b] dt
           -sqrt(2*relaxation_rate*temperature_b)*(g-u) dW_B.

The Wiener increment here is the SAME W_B driving g. Only continuous bath-B evolution contributes to Q_B; collisions and motor switches carry no instantaneous Q_B. In particular a collision's gas kinetic-energy change is stored in the retained contact and is not itself a deposit into the thermal reservoir. Flow-maintaining motor work is separate from heat. Its continuous particle work is u times the bath-B thermostat force in the laboratory kinetic-energy balance. There is no background subtraction or other detector contribution.

Records start in the stationary joint rotor/contact/motor ensemble. With statistic="mean", the output is lim(t→infinity) E[Q_B(t)]/t, in energy/time. With statistic="noise", it is lim(t→infinity) Var[Q_B(t)]/t, in energy²/time. This variance is centered about the ensemble mean and includes all stationary rotor, retained contact, motor, collision and Wiener randomness, including their correlations. The ideal calorimeter adds no electronic noise or sampling error to predictions. Statistic is optional and defaults to "mean"; a prediction list may mix both readouts.

Every experiment contains angular_speed in [-.65,.65], flip_rate in [.8,2], temperature_a in [.4,2], temperature_b in [.4,1.4], flux_fraction in [0,1], and relaxation_rate in [.2,3]. All settings are exact, give stationary dynamics and finite stated readouts, and refer to the apparatus above. The only unknown is the common laboratory drag. The finite contact replaces an infinite stream of fresh gas; relaxation_rate is never implicitly infinite.

Calibration uses flux_fraction=0, angular_speed=0, and relaxation_rate=1.3 with varied temperatures and motor rates. Its records omit statistic and measure mean heat. The 288 records in data/calibration.json contain input, measured value and independent Gaussian standard uncertainty sigma=.001 in energy/time.

Implement Model.fit(records), returning self and storing drag. Model.predict(experiments) returns a finite NumPy array of shape (len(experiments),), in input order, containing the selected stationary calorimeter readout. Any implementation preserving this interface may be used.
