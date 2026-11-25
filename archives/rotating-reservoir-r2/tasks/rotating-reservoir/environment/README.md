# Calorimetry with a moving reservoir

A particle of mass one moves in an unbounded laboratory plane. Its position is r=(x,y), velocity is v, and its fixed harmonic potential is U=(stiffness_x*x²+stiffness_y*y²)/2. An overdamped coupler at position q is connected to the particle by an isotropic spring with energy h|r−q|²/2. The known spring constant is h=b/contact_time, where b=0.9. The coupler has no other potential.

Reservoir A is stationary at temperature temperature_a and acts on the inertial particle with drag `drag`. Reservoir B acts on the coupler at temperature temperature_b and has the externally maintained mean velocity u(x)=A*x, where

```
A = [[strain_rate, -angular_speed],
     [angular_speed, -strain_rate]].
```

The complete laboratory stochastic equations are

```
dr = v dt
 dv = [-K r - drag*v - h(r-q)]dt + sqrt(2*drag*temperature_a)dW_A
 dq = [A q + h(r-q)/b]dt + sqrt(2*temperature_b/b)dW_B,
K = diag(stiffness_x, stiffness_y).
```

W_A and W_B are independent two-component standard Wiener processes. Reservoir B's equation is the overdamped force balance at q. These ideal force laws include all mechanical effects; there are no additional pressure forces or hydrodynamic interactions. The particle inertia is retained. All coordinates and reservoirs are two-dimensional. Units are fixed reference mass, length and time, with the particle mass and Boltzmann's constant equal to one; temperatures are energies in these units.

A mechanical actuator maintains reservoir B's prescribed incompressible flow. A separate thermostat maintains its internal thermal temperature. The calorimeter records the long-time mean heat removed by that thermostat from B's internal thermal degrees of freedom, positive for heat entering B from the coupler. The empty-apparatus background is subtracted. Actuator power is a separate mechanical measurement. Reservoir work has the Stratonovich convention. The reservoirs and flow are unaffected by the particle and coupler. All controls are fixed and the full joint state reaches stationarity before recording the calorimeter signal.

Each experiment gives angular_speed in [-0.6,0.6], strain_rate in [-0.4,0.4], contact_time in [0.2,0.8], stiffness_x in [1,1.6], stiffness_y in [2.2,2.8], and both temperatures in [0.8,1.4]. The only unknown is the common drag in [0.4,1.1]. All stated controls give stable stationary dynamics. The controls are exact. Records in data/calibration.json contain input, measured value and independent Gaussian standard uncertainty sigma=0.001.

Implement Model.fit(records), returning self and storing the fitted drag. Model.predict(experiments) must return a finite NumPy array of shape (len(experiments),), in input order, containing the specified heat rates in energy per unit time. Any implementation satisfying this interface may be used.
