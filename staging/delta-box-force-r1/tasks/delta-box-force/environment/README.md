# Force on a quantum cavity wall

Use hbar=k_B=1. One spinless nonrelativistic particle moves in the interval [-length/2,length/2]. The two walls are impenetrable and the wavefunction vanishes at them. The particle has the same unknown positive mass in [.4,.9] in every experiment. A static point defect is pinned at x=0. Its potential is strength*delta(x), where the repulsive strength is a known physical control independent of length, mass and temperature. The exact one-particle Hamiltonian is

    H = -(1/(2*mass))*d²/dx² + strength*delta(x).

There are no other potentials, particles or interactions. For each preparation the particle reaches the canonical quantum Gibbs equilibrium of this Hamiltonian at the specified temperature. The weak preparation contact is removed before reading the detector; the stationary mixture is unchanged. The massive walls and the defect remain held fixed. The detector records the mean outward force exerted by the particle on the right wall, excluding apparatus backgrounds. The defect has its own support. Wall and defect recoil can be neglected.

Each experiment contains length in [.8,1.8], strength in [0,12], and temperature in [.1,2]. All controls are exact. Calibration is performed with strength=0 at varied lengths and temperatures. Records in data/calibration.json contain input, measured value and independent Gaussian standard uncertainty sigma=.001 in force units. The instrument uncertainty is fixed independently of the noiseless response.

Implement Model.fit(records), returning self and storing the fitted mass. Model.predict(experiments) must return a finite NumPy array of shape (len(experiments),), in input order, containing the specified wall forces. Any implementation satisfying this interface may be used.
