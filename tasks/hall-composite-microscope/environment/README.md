# Two-current Hall microscope

A thin film carries two conserved charge currents. Its two materials form an infinite square checkerboard. A unit square with lower left corner at integer coordinates (i,j) is phase 0 when i+j is even and phase 1 otherwise. Both exposures of any local measurement use the same film at the same position. Ignore interfaces of zero area. Coordinates are dimensionless.

Each exposure is stationary. Species α has a curl-free two-dimensional electrochemical field Eα and a divergence-free current Jα. Represent a spatial vector (x,y) by x+iy, so i rotates it counterclockwise. Stack the two species into complex columns e and j. In phase p the constitutive law is

    j = (A_p + i g b H_p) e.

Here b is the signed magnetic bias and g is the common gain. Each material record provides the real symmetric species matrices A_p as longitudinal and H_p as hall. Their off-diagonal elements couple the species. Across interfaces both tangential fields and both normal currents are continuous. The prescribed drive is the mean e over the film. Each row of drive gives [Ex,Ey] for one species. The fluctuations are gradients of 2-periodic potentials in both directions. Angle is measured counterclockwise in radians. Area averages below are over a whole 2×2 cell.

For mode calibration the coupon consists of a single material. For mode bulk, materials lists the two checkerboard phases in label order. Both read the weighted area-averaged current

    Re[exp(−i angle) contactᵀ ⟨j⟩].

For mode local, bias, drives and contacts each have two entries in exposure order. If u and v are the two contacts, the phase-resolved local signal is

    Re[exp(−i angle) ⟨1_phase (uᵀ e_first(x))* (vᵀ e_second(x))⟩].

The star denotes complex conjugation. Both fields refer to the same point x, including for unequal biases. A pixel outside the specified phase contributes zero. The average includes all pixels. With angle zero the local signal measures the dot product. With angle π/2 it measures the signed planar cross product of the first vector with the second.

The gain lies in [0.65,1.15]. The eigenvalues of A lie in [0.25,5] and those of H in [−2.1,2.1]. Each bias has magnitude at most 1.5. Each contact and each row of a drive has Euclidean norm at most one. The calibration records in /app/data/calibration.json have keys input and value. Set Model.gain from these records in fit(records) and return self. predict(inputs) takes a list of queries and returns a finite, real, one-dimensional NumPy array of matching length, including for an empty list. Calibration must achieve maximum absolute error below 0.000002. Bulk, same-exposure field power, two-exposure field correlation and cross-product readings must each achieve maximum absolute error below 0.002.

## Worked readings

Take A₀=diag(1,2), A₁=diag(9/2,2) and zero Hall matrices. Drive species 0 along +x with unit field and leave species 1 undriven. Select species 0 with both contacts. The bulk current along x is c=√(9/2). Two identical exposures give local power c/2 in phase 0 and 1/(2c) in phase 1. Turning the second drive to +y and the detector to π/2 gives c²/(1+c²) in phase 0 and 1/(1+c²) in phase 1.

For a coupled example take A₀=[[1.1,0.38],[0.38,2.1]], H₀=[[0.55,0.43],[0.43,−0.2]], A₁=[[3.2,−0.50],[−0.50,0.7]] and H₁=[[−0.61,0.27],[0.27,1.05]]. For this worked example take gain 0.94 and use biases [1.1,−0.85], drives [[[0.8,0.1],[−0.2,0.5]],[[-0.4,0.7],[0.6,0.15]]] and contacts [[0.75,−0.4],[0.3,0.8]]. At angle zero the readings in phases 0 and 1 are 0.06896977 and 0.05794170. At π/2 they are 0.11159612 and 0.15780388.

The supplied model fits the uniform coupons. For a two-phase film it initially assumes uniform gradients in both phases.
