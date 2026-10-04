# Diffusion in a coherent elastic alloy

A single coherent solid occupies a three-dimensional periodic cube of side 2*pi in reduced length units. Let c(r,t) be the small dimensionless concentration departure from a uniform reference alloy. Its spatial mean is zero. A compositional change has stress-free strain E0*c, with the fixed laboratory tensor E0=diag(0,0.5,-0.5). The material has isotropic linear elasticity with Lame coefficients lambda=2 and mu=3, in reduced energy-density units. Composition does not change these coefficients. There are no cracks, dislocations, plasticity, body forces, interfaces, or loss of coherence.

The mean strain is held at zero by the periodic cell. Local strain is the symmetric displacement gradient, and the displacement is periodic. Mechanical equilibration is much faster than diffusion, so the elastic field is at mechanical equilibrium at every measured time. Elastic inertia is neglected. Strains are small, and all constitutive equations below are the exact quadratic approximation defining this apparatus.

The free-energy density is

f = 0.4*c^2/2 + 0.2*|grad c|^2/2 + (lambda/2)*[tr(epsilon-E0*c)]^2 + mu*(epsilon-E0*c):(epsilon-E0*c).

Here epsilon=(grad u+grad u^T)/2, and the colon sums both indices of the symmetric tensor (off-diagonal components are counted twice). Chemical diffusion is conserved: its flux is minus one uniform positive mobility times the gradient of the chemical potential obtained from the total free energy. Mechanical work and composition have the thermodynamic coupling defined by this free energy. Ignore material advection, heat flow and thermal fluctuations beyond measurement noise. The only unknown is mobility in [0.03,0.2], in the reduced units of this specification.

Prepare c(r,0)=amplitude*cos(q dot r), with nonzero integer wavevector q=(qx,qy,qz), each component in [-3,3]. Its magnitude and direction are known. The input field `wavevector` is a three-element list, `amplitude` is between -0.03 and0.03, and `time` is between0 and6. The observable is the signed cosine Fourier coefficient of the concentration at the requested time, in the same normalization as amplitude. The quadratic constitutive model preserves this single-mode form.

Calibration prepares wavevectors along the laboratory x axis. Every record has input, value, and independent Gaussian standard deviation sigma; the one mobility is shared by all experiments. Implement Model.fit(records), returning self and setting Model.mobility, and Model.predict(experiments), returning a finite NumPy array with one coefficient per input. Read data/calibration.json.
