# Two-current Hall microscope

A stationary two-species conductor in a perpendicular magnetic field obeys a coupled elliptic system. Its longitudinal coefficients are strictly positive and its real Hall blocks do no work. The experiment averages the product of two fields on the same translated checkerboard. Fluctuations inside each phase survive this measurement. The underlying planar Hall and cross-coefficient dualities are described in [Classical Hall effect in two-dimensional composites](https://doi.org/10.1103/PhysRevB.38.11296) and [Linear response of two-phase composites with cross moduli](https://doi.org/10.1103/PhysRevA.40.1568).

## Reduction to ordinary scalar conductors

Write f=ij and z=iA−gbH so f=z e. Both f and e are gradients because each current is divergence-free. Real symplectic changes of coordinates in (e,f) preserve these conditions and symmetric constitutive matrices. Set R to the positive square root of A₀. The normalized fields are e′=Re and f′=R⁻¹(f+gbH₀e). Their phase matrices are iI and Z=R⁻¹[iA₁+gb(H₀−H₁)]R⁻¹. The Cayley transform W=(Z−iI)(Z+iI)⁻¹ is complex symmetric and strictly contractive. Factor it as U diag(s) Uᵀ with U unitary and s nonnegative. With X+iY=Uᵀ, take E=Xe′+Yf′ and F=−Ye′+Xf′. Then F=iKE, with K=I in phase 0 and K=diag(c²) in phase 1, where c²=(1+s)/(1−s) componentwise. Thus each canonical species sees its own ordinary scalar checkerboard. Fourfold rotational symmetry gives isotropy. Phase-interchange symmetry and current-field duality give scalar effective conductivity c. The laboratory e equals R⁻¹(Xᵀ−iYᵀK)E in each phase. Use the same relation with K=diag(c) for the means.

## Recovering a local cross correlation

Consider scalar canonical fields E and G from two exposures, with contrasts c² and d². The two fields share the microscopic phase mask. Write Dₚ for the full-cell contribution of phase p to E·G, and Xₚ for the contribution to E×G. Integration of a periodic gradient dotted into a divergence-free current gives the product of the means. Applied first to one current and then to the other, this yields

    D₀+c²D₁ = c ⟨E⟩·⟨G⟩
    D₀+d²D₁ = d ⟨E⟩·⟨G⟩.

The integral of the determinant of two gradients also equals the determinant of their means. Rotating both currents into gradients preserves their determinant. This gives

    X₀+X₁ = ⟨E⟩×⟨G⟩
    X₀+c²d²X₁ = cd ⟨E⟩×⟨G⟩.

Hence D₁=⟨E⟩·⟨G⟩/(c+d) and X₁=⟨E⟩×⟨G⟩/(1+cd). Phase 0 contributes cd times the corresponding phase 1 result. These formulas extend by continuity to repeated and unit contrasts under uniform ellipticity. The equations apply to arbitrary choices of the two applied mean fields. Combining each cross product with its actual laboratory transformation gives the measured local correlation, including mixed species and unequal magnetic fields.

## Independent data generation

The fixture generator uses scale-separated laminations and real four-dimensional matrices. It forms equal parts of the two materials across a straight interface and solves for the local field from tangential-field and normal-current continuity. It then repeatedly combines equal areas of the preceding composite and its ninety-degree rotation. It propagates one material-response operator per exposure, and a separate bilinear local detector through the very same interfaces at both biases. The nested length scales are fully separated at every step. The sequence tends to an isotropic medium with the same scalar geometric means. The integral identities above show why its phase-resolved bilinear observables converge to the checkerboard values. Nine refinements generate the fixtures and a twelve-refinement check compares them with the analytic checkerboard solution.

A third calculation solves the periodic two-species system directly on the stated checkerboard with conforming triangular finite elements. For the coupled public dot product, errors at 12, 24, 48 and 96 cells per direction are 0.00643, 0.00311, 0.00148 and 0.00070. The validator also checks phase swapping, exchanging exposures, full-cell circulation, unit and repeated contrasts, identical phases and the published scalar examples.

The stronger physical control predicts the exact bulk and both phase mean fields, then samples the two local probes independently within each phase. Its maximum errors on field power, cross-bias correlation and circulation are 0.1240, 0.0410 and 0.0586. The analytic solution differs from the frozen answers by less than 0.000000006. The uniform-gradient starter fits calibration and has respective errors of 0.2631, 0.1855 and 0.0999.

Run `OPENBLAS_NUM_THREADS=1 python scripts/validate_hall_composite_microscope.py` from the repository root. Add `--generate` to regenerate the static records.
