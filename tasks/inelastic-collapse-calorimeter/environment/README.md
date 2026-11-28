# Rebound calorimeter

A bead of unit mass moves on the half-line x≥0. Between wall contacts it experiences Gaussian white acceleration. In dimensionless units its position and velocity satisfy

    dX=V dt,       dV=√(2D) dW.

There is no drag or constant force. The acceleration diffusivity D lies in [0.72,1.10]. At a launch the bead starts at x=0 with speed u>0 pointing into the half-line. Follow it until its **first subsequent return** to x=0. Write τ for the duration, and v for the positive magnitude of its incoming velocity. The wall measures all three quantities on this flight.

The wall switches between two restitution settings at the instant of impact. It selects the slow setting if τ>threshold × (u²+v²), and the fast setting otherwise. The input restitution is [fast,slow]. With the selected value r the bead leaves immediately at speed rv. The energy delivered to the wall by this impact is (1−r²)v²/2. The next flight uses fresh Brownian increments and starts the flight clock at zero. Every impact uses the same rule. The wall applies no other impulse or delay. Threshold lies in [0.20,3.0] and both restitutions in [0.014,0.62].

A run sticks when infinitely many impacts accumulate in finite laboratory time. The run is then absorbed at the wall. Define T as the sum of all successive flight durations up to that accumulation and Q as the sum of their actual deposited energies. A run that never reaches such an accumulation contributes zero to the sticking and collection readouts. The apparatus resolves this ideal accumulation without imposing a minimum bounce speed.

The following inputs correspond to independent experiments.

- In mode free, remove the wall and measure the mean squared displacement about the known ballistic displacement after lag in [0.1,1.4]. These measurements calibrate D.
- In mode selector, supply threshold and return the probability that the next impact uses the slow setting. Launch speed can be set to one for this measurement.
- In mode sticking, supply threshold and restitution and return the probability of finite-time sticking from launch speed one.
- In mode timer or calorimeter, also supply launch, clock and dose. Two independent monitors stop a run at a random elapsed time and a random total deposited energy, respectively. Their survival probabilities after elapsed time t and cumulative energy q are exp(−clock × t) and exp(−dose × q). Both monitors start at launch and are independent of the bead. Return the probability of sticking before either monitor stops the run. Equivalently the signal is the mean of exp(−clock × T−dose × Q) over all launches with zero contribution from runs that never stick. Zero rate disables that monitor. Timer shots disable the energy monitor. Calorimeter shots have a positive dose and can use either a positive or zero clock.

For collection shots launch lies in [0.45,0.95], positive clock in [0.25,1.2] and positive dose in [0.16,1.15]. Collection shots use strongly cooling settings. In an ensemble of single flights their mean log(next launch speed / current launch speed) is at most −1.1. The sticking detector can use either side of the transition.

One independently calibrated marginal law is available. With R=v/u the density over all first-return flights is

    p(R) = 3 R^(3/2) / [2π(1+R³)],       R>0.

The same marginal applies at all launch speeds and D. The starter samples this speed law, uses a fixed value of Dτ/u² for each flight and propagates either wall setting through successive rebounds.

For a check at D=0.90 and threshold=0.5 the slow setting is selected with probability 0.7211133. With threshold=0.25, restitutions [0.024,0.27] yield sticking probability zero and [0.62,0.11] yield one. With identical restitutions the transition occurs at r=exp(−π/√3). For a collection example take D=0.90, threshold=0.37, restitution=[0.365,0.020] and launch=0.67. At clock=0.46 with dose=0 the timer reads about 0.16040. Enabling dose=0.47 gives about 0.12681.

Implement Model.fit(records), store the calibrated D in self.diffusivity and return self. Each calibration record has input and value fields. Model.predict(inputs) returns a finite real NumPy array of shape (len(inputs),), in input order, including for an empty list. Maximum absolute error must be below 0.000002 for free calibrations and below 0.004 for each other measurement. The stochastic reference standard errors are below 0.00035.
