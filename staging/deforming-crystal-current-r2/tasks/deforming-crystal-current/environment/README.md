# Current from a deformed polar crystal

A homogeneous polar insulator has one effective +q0/−q0 pair per reference
cubic cell of side a0. Coordinates are in the fixed material basis. The positive
center is at c=(.25,.25,.20), and the negative center is at c+s. There are no
mobile carriers. Under a homogeneous lattice deformation F, a material
coordinate R has laboratory position a0*F*R. The same complete neutral cells
and their surface termination are followed throughout the experiment.

The homogeneous cell free energy at zero macroscopic electric field is

    W0/E0 = K*|x|^2/2 + |x|^4 - h(E) dot x + w0(F),
    x = s - (.08,.06,.22),
    E = (transpose(F)*F - I)/2,
    h(E) = (.60*E13 + .20*E12,
            .50*E23 - .10*E12,
            .45*E11 + .30*E22 + .70*E33 + .20*E12).

Matrix indices run from 1 to 3. The known lattice contribution w0(F) is
independent of s. K is the common unknown stiffness in E0 units, in [.8,1.2].
W0 includes the microscopic zero-field electrostatics. The remaining background
has a known, positive scalar absolute permittivity epsilon_b, independent of
strain and s in the laboratory frame. The macroscopic electric field couples to
the specified cell charges. In the chosen units,
q0^2/(epsilon_b*a0*E0)=1. External mechanical control imposes homogeneous F.
Thermal fluctuations of the macroscopic internal offset are negligible.

The sample is a slab with S reference cells per lateral face and N cells through
its third material direction. It is periodic laterally, with ideal conducting
electrodes following its lower and upper third-direction material faces. There
is no separate surface charge or surface constitutive layer. An ideal
zero-impedance amplifier in the upper crystal lead measures conventional
current entering that electrode and adds no capacitance or leakage.

There are two electrical preparations:

* With `load` omitted or null, both electrodes are held at the same potential.
* With finite `load`, the crystal is connected in parallel to a mechanically
  separate, passive ideal capacitor. The upper electrode and upper capacitor
  plate form one complete isolated metallic node; the lower pair form another.
  Each complete node has zero total free charge, which can redistribute among
  its connected surfaces. There is no voltage source, leakage, extra
  capacitance, or inertia. The load capacitance is
  `C_L = load * epsilon_b*a0*S/N`, with `load` in [.5,2]. C_L is constant during
  mechanical deformation, and the capacitor plates do not follow that motion.

For each preparation, let the internal coordinates and all connected charges
fully equilibrate at the imposed F. Take the macroscopic size limit while
holding `load` fixed for the loaded preparations. Starting from that equilibrium,
impose

    F(t) = F + delta*G*sin(omega*t).

First take the small-amplitude limit and then the quasistatic frequency limit,
with internal and electrical equilibration complete throughout. The output is
the signed cosine coefficient j in

    I_upper(t)/A_ref = delta*omega*(q0/a0^2)*j*cos(omega*t)
                      + o(delta*omega).

A_ref=S*a0^2 is the fixed reference electrode area. F, G, delta and j are
dimensionless; omega is angular frequency and q0 is the positive charge unit.

Each prediction input contains two 3-by-3 nested arrays, `deformation` (F) and
`drive` (G), and optionally `load` as above. Both matrices are upper triangular.
F's diagonal entries lie in [.88,1.12], its upper off-diagonal entries in
[−.12,.12], and each upper-triangular entry of G in [−1,1]. Lower-triangular
entries are zero. Delta is taken small enough that the same material branch and
cell termination are retained. The charge centers remain inside their selected
cells over this domain.

## Data and interface

`data/calibration.json` contains independent measurements with keys `input`,
`value` and `sigma`. Every reading has the fixed instrument standard deviation
.001 in j units. Repetitions are independent. Prediction inputs follow the same
schema and domain.

Implement `Model.fit(records)` to estimate K, store it as `self.stiffness`, and
return `self`. `Model.predict(experiments)` must return a finite NumPy array of
shape `(len(experiments),)` in the input order.
