# Counting photons transferred between optical rails

A pulse occupies 2–6 ideal bosonic modes of a common frequency. All optics are lossless. Each rail starts in an independent thermal state with the specified mean occupation n. A degenerate parametric pump on that rail then applies S with

    S† a S = a cosh(r) − exp(i chi) a† sinh(r).

There is no displacement. The rails pass through a preparation mesh, then a nondestructive meter measures the total photon number Q in a selected bank of input rails. The meter writes the integer q on a classical register. Within each eigenspace the measurement is an ideal Lüders projection. It acquires no information about the individual occupations within the bank.

The light passes through a second mesh called `before`. Some experiments then insert a nondestructive parity meter on a selected set of rails. This meter makes a projective measurement of (−1) to the power of their combined photon number. Only shots with the specified parity are retained. It resolves no further information. When the meter is absent all shots are retained. The pulse then passes through the `after` mesh. A final detector records the total photon number q′ in the original bank of rail labels at the exit.

The reading is the average of cos(lambda (q′−q) − phase) on retained shots multiplied by a common visibility V in [0.65,0.95]. All angles are in radians. Each input has the following fields.

* `source` lists [n,r,chi] for each rail in index order, with n in [0,0.9], r in [0,0.85] and chi any phase.
* `prep`, `before` and `after` are lists of up to 15 two-rail mixers in the order encountered. An empty mesh is the identity. For [i,j,t,p], rail indices are distinct and zero based. The outgoing coherent amplitudes are cos(t) a_i − exp(−ip) sin(t) a_j and exp(ip) sin(t) a_i + cos(t) a_j, respectively. All other amplitudes are unchanged.
* `bank` is a nonempty proper subset of rail indices. `gate` is null or an object with `rails` (a nonempty subset) and `parity` (1 for even, −1 for odd). Heralded settings have retention probability at least 0.05. `lambda` and `phase` are readout angles.

For example, use source=[[0,0.7,0],[0,0,0]], bank=[0], before=[[0,1,π/2,0]], empty prep and after, gate=null, lambda=π and phase=0. Every pulse contains an even number of photons. Swapping the rails therefore gives a reading equal to V.

Calibration in `data/calibration.json` consists of {input,value} records, rounded to eight decimal places. These two-rail measurements use bank=[0], empty prep and after, no gate, and one mixer in before that mixes unsqueezed thermal rail 0 with vacuum rail 1. Implement Model.fit(records), returning self with fitted attribute `visibility`, and Model.predict(inputs), returning a finite real NumPy array of shape (len(inputs),) in input order. The maximum absolute error must be below 0.00002 in calibration and below 0.001 on each new reading. The starter treats photons as independently routed rays with thermal number statistics at the mean occupation after pumping.
