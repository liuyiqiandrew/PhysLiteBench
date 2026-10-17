# Calorimetry of a rotor in a rotating collision reservoir

A mass-one tracer is constrained to a fixed circular ring of radius one. Its angular coordinate theta is periodic with period 2*pi, its signed tangential speed is v=dtheta/dt, its moment of inertia is one, and its kinetic energy is v²/2. The ideal radial constraint does no work. Set Boltzmann's constant to one and use fixed reference mass, length and time units. Temperatures are energies, v and gas speeds are velocities, angular_speed is an inverse time (numerically a tangential speed at unit radius), and drag and flip_rate are inverse times. Velocity is unbounded. There is no potential, external torque, radial kinetic degree of freedom or additional force.

A stationary laboratory Langevin thermostat A at temperature temperature_a has the same unknown drag in [.4,1.1] in every experiment. Between gas collisions,

    dtheta = v dt,
    dv = -drag*v dt + sqrt(2*drag*temperature_a) dW_A.

W_A is a standard real Wiener process. Reservoir B supplies fresh mass-one gas particles through an ideal tangential collision contact. All collision kinematics have exactly one tangential degree of freedom; incoming particles have no additional kinetic component contributing to these measurements. Its prescribed mean tangential flow is u=s*angular_speed, with internal thermal temperature temperature_b. The motor label s is an independent symmetric two-state Markov process, switching in either direction at rate flip_rate, with stationary probabilities one half. At a motor reversal tracer theta and v remain continuous; there is no tracer impulse, instantaneous particle work or recorded collision heat. Costs of changing the empty reservoir flow belong to the separate background. The reservoirs and motor are unaffected by the tracer.

The contact combines two specified ideal collision channels. With eta=flux_fraction, the complete differential rate of a collision with incoming signed gas speed g is

    3.6 * [(1-eta) + eta*abs(v-g)/v_star]
        * exp(-(g-u)²/(2*temperature_b))
        / sqrt(2*pi*temperature_b) dg,
    v_star = 1 in the fixed reference velocity units.

This is the rate per time evaluated at the current tracer velocity and current motor state. The first channel has a constant collision clock; the second channel counts incoming relative-speed flux. Their strengths, eta and 1-eta, are known settings. Integrating this differential rate over all real g gives the instantaneous collision hazard. Conditional on an event, the incoming speed distribution is proportional to this differential rate; its thermal Maxwell density is the incoming spatial density before collision selection. There are no other gas interactions, velocity cutoffs, pressure forces, collision normals or contact variables. Fresh particles are used for one collision, then replaced. Conditional collision events and marks, the motor, and W_A are independent according to these stipulated rates.

At each collision equal effective masses exchange tangential velocities:

    v_after = g_before,
    g_after = v_before.

Theta stays continuous. This elastic exchange conserves the two bodies' tangential momentum and laboratory kinetic energy. The fixed ring constraint adds no work in the exchange. Flow-maintaining motor work is separate from heat. The B calorimeter records the change in the outgoing gas particle's kinetic energy relative to the same prescribed flow immediately before and after collision:

    h = [(g_after-u)²-(g_before-u)²]/2.

Let Q_B(t) be the sum of recorded increments in an interval of length t, starting from the stationary joint tracer/motor ensemble. With statistic="mean", the readout is lim(t→∞) E[Q_B(t)]/t, in energy/time. With statistic="noise", it is lim(t→∞) Var[Q_B(t)]/t, in energy²/time. The variance is centered about the ensemble mean and includes all stationary tracer, motor, event-time and gas-mark randomness. The ideal calorimeter adds no electronic or measurement noise. With statistic="scgf", a third measurement sets the real energy bias heat_bias and reports

    lim(t→∞) log E[exp(-heat_bias*Q_B(t))]/t,

in inverse time. This is the logarithmic exponential heat readout of repeated stationary calorimeter records, with the ensemble average inside the logarithm. Each record uses the same tracer, reservoirs, motor rates and heat deposits. Heat_bias is in inverse energy and lies in [-.35,.35]. All these biases lie strictly within abs(heat_bias)<1/max(temperature_a,temperature_b), so the stated stationary exponential readouts are finite. The calorimeter adds no sampling error to model predictions. Statistic is optional and defaults to "mean"; heat_bias is optional and defaults to zero, and affects only the "scgf" readout. A prediction list may mix all three readouts.

Every experiment contains angular_speed in [-.65,.65], flip_rate in [.8,2], temperature_a in [.6,1.6], temperature_b in [.6,1.4], and flux_fraction in [0,1]. All settings are exact and give stationary dynamics with finite stated readouts. The only unknown is the common laboratory drag. Calibration uses flux_fraction=0 and angular_speed=0 with varied temperatures and motor rates; its records omit statistic and measure mean heat. Records in data/calibration.json contain input, measured value and independent Gaussian standard uncertainty sigma=.001 in energy/time.

Implement Model.fit(records), returning self and storing drag. Model.predict(experiments) returns a finite NumPy array of shape (len(experiments),), in input order, containing the selected stationary calorimeter readout. Any implementation preserving this interface may be used.
