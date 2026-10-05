# Current from a deformed polar crystal

A homogeneous polar insulator has one effective +q0/−q0 pair per reference
cubic cell of side a0. Coordinates are in the fixed material basis. The positive
center is at c=(.25,.25,.20), and the negative center is at c+s. There are no
mobile carriers. Under a homogeneous lattice deformation F, a material
coordinate R has laboratory position a0*F*R. The same complete neutral cells
and their surface termination are followed throughout the experiment.

The internal offset s fully relaxes at every imposed deformation. Its total
homogeneous cell free energy at zero macroscopic electric field is

    W/E0 = K*|x|^2/2 + |x|^4 - h(E) dot x + w0(F),
    x = s - (.08,.06,.22),
    E = (transpose(F)*F - I)/2,
    h(E) = (.60*E13 + .20*E12,
            .50*E23 - .10*E12,
            .45*E11 + .30*E22 + .70*E33 + .20*E12).

Matrix indices in this formula run from1 to3. The known lattice contribution
w0(F) is independent of s. K is the common unknown internal stiffness in E0
units, in [.8,1.2]. W is the equilibrium material free energy including internal
electrostatic effects. External mechanical control holds the lattice deformation
homogeneous. Thermal fluctuations of the macroscopic internal offset are
negligible.

The sample is a slab, periodic in the two lateral material directions, with
ideal conducting electrodes on its lower and upper third-direction material
faces. Both electrodes follow those faces and are maintained at the same
potential. There is no applied voltage. An ideal amplifier measures conventional
electric current entering the upper electrode. The material is insulating and
has no separate surface charge or surface constitutive layer. Take the
macroscopic-thickness limit before the quasistatic response, so bounded
microscopic surface relaxation does not change the bulk constitutive state.

Starting from the relaxed state at F, the mechanical drive imposes

    F(t) = F + epsilon*G*sin(omega*t).

First take the small-amplitude limit and then the quasistatic frequency limit,
with internal relaxation complete throughout. The output is the signed cosine
coefficient j in

    I_upper(t)/A_ref = epsilon*omega*(q0/a0^2)*j*cos(omega*t)
                      + o(epsilon*omega).

A_ref is the fixed reference electrode area. F, G, epsilon and j are
dimensionless; omega is angular frequency and q0 is the positive charge unit.

Each prediction input contains two3-by-3 nested arrays, `deformation` (F) and
`drive` (G). Both are upper triangular. F's diagonal entries lie in [.88,1.12]
and its three upper off-diagonal entries in [−.12,.12]. Each upper-triangular
entry of G lies in [−1,1]. Lower-triangular entries are zero. Epsilon is taken
small enough that the same material branch and cell termination are retained.
The charge centers remain inside their selected cells over this domain.

## Data and interface

`data/calibration.json` contains independent measurements with keys `input`,
`value` and `sigma`. Every reading has the fixed instrument standard deviation
.001 in j units. Repetitions are independent. Prediction inputs follow the same
schema and domain.

Implement `Model.fit(records)` to estimate K, store it as `self.stiffness`, and
return `self`. `Model.predict(experiments)` must return a finite NumPy array of
shape `(len(experiments),)` in the input order.
