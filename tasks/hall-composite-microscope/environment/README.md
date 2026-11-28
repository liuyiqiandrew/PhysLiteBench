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

## Capacitor noise experiment

In mode noise the four terminal voltages form the real vector V=[⟨E₀ₓ⟩,⟨E₁ₓ⟩,⟨E₀ᵧ⟩,⟨E₁ᵧ⟩]. The four film current outputs I are the mean J components in the same order. There is no imposed mean field in this experiment. A capacitor network with real positive definite capacitance C and a parallel passive load of real positive definite conductance L sets the terminal dynamics

    C dV/dt + I + L V + η_load = 0.

The load delivers independent white current noise with covariance 2 T_load L δ(t−t′). The film has two independently thermalized Ohmic modes in each phase. Diagonalize Aₚ in a real orthonormal species basis u₀,u₁ ordered by increasing eigenvalue a₀<a₁. At each point and for each Cartesian direction the two modes supply independent white current sources. Mode k has species direction uₖ and covariance strength 2 aₖ Tₚₖ. Thus the local constitutive law is now j=(Aₚ+i g b Hₚ)e+η. The x and y parts of η are independent with the same real species covariance. Different positions and times are independent. The spatial delta is normalized with the cell average measure d²x/4, equivalent to reporting terminal noise per unit active area. The local total currents obey continuity and conservation at every instant. Hall currents are noiseless. The charge relaxation within the film is instantaneous on the scale of C.

The input gives materials, capacitance C and load L. The two entries in bias and baths describe successive stationary settings of the source parameters. Each bath has phase_modes=[[T₀₀,T₀₁],[T₁₀,T₁₁]] and load=T_load. Before time zero the first setting has been held for an arbitrarily long time. At time zero the bias and bath strengths change instantaneously to their second settings. The same capacitor network remains attached and V is continuous. Two real four-component contacts u and v measure the ordered voltage correlation

    ⟨(uᵀ V(wait)) (vᵀ V(wait+lag))⟩.

All voltages have zero mean. All source temperatures lie in [0.05,2.2]. The two longitudinal eigenvalues in each noisy phase differ by at least 0.1. C and L have eigenvalues in [0.5,2] and [0.1,1.2] respectively. The times wait and lag are in [0,0.8]. Each contact has norm at most one. The same 0.002 maximum absolute error applies to the uniform-temperature, split-temperature and switched-bias readings.

As a check, use identical phases A=diag(1,1.9) and H=diag(0.7,−0.45), g=0.94, C=I and L=0.4I. Keep all temperatures at 1.3 and switch b from −0.3 to +1.2. Select the x voltage of species 0 at wait=0.27 and its y voltage at lag=0.4. The result is −1.3 exp(−1.4×0.4) sin(0.94×1.2×0.7×0.4), or −0.23065396.

For a split-temperature check, use the two materials from the coupled worked reading at g=0.94. Set C=[[1.2,0.12,−0.13,0],[0.12,0.84,0.08,0.05],[−0.13,0.08,0.96,0.11],[0,0.05,0.11,1.1]] and L=[[0.4,0.1,0,0],[0.1,0.5,0,0.08],[0,0,0.3,−0.02],[0,0.08,−0.02,0.45]]. Initially phase_modes=[[1.8,0.3],[1.1,0.15]] with load temperature 0.7 and b=1.1. The final phase_modes are [[0.2,1.6],[1.85,0.3]] with load temperature 0.6. At wait=0.18 and lag=0.23, measure u=[0.7,0.3,−0.5,0.2] then v=[−0.3,0.6,0.4,0.45]. Final bias −1.2 gives 0.05473119. Changing only that final bias to +1.2 gives −0.00021862.

The public film.py implements all ordinary checkerboard bulk and phase-probe readings. Its signal(query,gain) function accepts the bulk and local query formats above. For the capacitor experiment the supplied model uses the spatial mean of the final bath temperatures and assumes the capacitors have fully equilibrated at that temperature.
