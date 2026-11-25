# Adiabatic capture r1

This is a new staged task, not a evaluated benchmark result. The reviewed bounded prototype and every earlier exploratory report are preserved under `prototype/`. Root alone launches model evaluations after source, science and peer approval.

The public apparatus specifies a classical isolated quartic Hamiltonian, a finite-width action ensemble with independent uniform canonical angle, a prescribed cubic slow ramp, and final total energy at the fixed Hamiltonian zero. The ensemble average precedes the infinite-duration limit. The same preparation scale J0 is unknown in every experiment. Dimensionless units and all input bounds are explicit. No classical/quantum limit interchange, final kinetic-energy phase average, or shifted energy readout is used.

## Completed source and intended physical distinction

The completed source numerically inverts the total enclosed phase volume at one common final energy. It integrates every allowed coordinate interval and the whole initial action band. It uses both physical controls and exactly describes the noncrossing calibration. This is the globally ergodic continuation of an adiabatic phase-volume rule. It is a coherent physical approximation; its numerical area integrals and inversion are complete.

With q=sqrt(s)*x and p=s*P, energy scales as s² and phase area as s^(3/2). At unit scale the two saddle lobes have areas A_L,A_R. Both grow monotonically, with constant fractions alpha_i=A_i/(A_L+A_R). An incoming action J reaches the saddle when (A_L+A_R)*s^(3/2)=2*pi*J. For the stated finite-width classical ensemble, the capture probabilities are alpha_i and the outgoing lobe actions are alpha_i*J. Those separate actions generally imply different final energies. Continuing a single globally mixed energy therefore changes the physical ensemble after the energy shell splits.

The oracle inverts each branch area and averages its energy with the capture probabilities. The action integral is split at the capture threshold, including partial capture. The public endpoint interval remains [.1,2.2]; it is not restricted to the diagnostic regime. Symmetry at delta=0 is a private limiting check because the two energy closures then agree.

The intended failure is retaining the global-volume ensemble despite correct parameter fitting and mathematical phase-space calculations. A trial that recognizes capture but misnormalizes actions, miscomputes branch areas, or fails a trajectory integration is a mathematical/numerical or mixed failure. It must not be counted as a clean intended physical failure merely because its score is zero. A fixed-J0 in-memory response repair and public trajectory review are required for any later causal claim.

## Calibration and grading

The predeclared calibration uses 256 independent readings at delta=.2 and s_final=.15, with fixed Gaussian sigma=.001. J0=.1437 appears only in the private reference and metadata. All model constructors initialize the parameter to None; the public fit is unfinished and a separate completed shortcut is supplied for controls.

The entire J0 interval [.12,.16] remains on connected outer orbits during this calibration. Writing J=J0*z, with fixed z uniform on [1,1.2], gives d<E>/dJ0=<z*omega(J0*z)> >0. Thus one repeated-setting mean identifies J0 globally, rather than relying on a local optimizer or sampled profile alone. The completed control uses this monotone inversion. The validator also checks 33 parameter values, endpoint recovery and all diagnostic groups across the whole parameter interval.

The public and private calibration files are byte-identical. Public calibration and private calibration gates remain1.5, parameter relative error remains.03, and every prediction group uses the ordinary.04 RMS-relative gate. The neutral unrestricted318-byte instruction is copied unchanged from the canonical protocol. Agent and verifier timeouts remain600s and60s. Neither the README nor the public image contains capture formulas, the private hint, an implementation restriction, or a required solver.

Two connected-orbit anchors and six captured cases form the private prediction set. The three diagnostic groups vary delta across.15,.2,.25 and include both endpoint1.8 and2.2. Their total energies have ordinary finite magnitudes; no near-zero barrier energy is used to manufacture a relative gap. The intermediate small-energy/partial-capture cases are scientific limit checks only.

## Independent reference and runtime

The reference constructs the declared initial action distribution using a numerical area inversion, then places phase points uniformly in time along each frozen initial orbit using DOP853. This initial area calculation specifies the preparation only. Final values are obtained by direct propagation of q,p through the actual ramp. There are no final branch-action substitutions, capture probabilities, final area inversions or precomputed energy tables in the reference.

The production reference uses a fourth-order extended-phase-space symplectic splitting with duration512, step.04,32 action nodes and256 angle points. It records the final Hamiltonian and external work independently. Every one of the eight scored settings is checked at duration1024 with64×512 preparation points and separately at half time step. The prototype additionally compares independent DOP853 ensemble propagation, longer durations, phase offsets and ramp shape. These are empirical convergence checks, not a uniform high-precision theorem for every endpoint.

Only exact repeated reference calls are cached inside the verifier. No interpolant is used. Isolated local controls execute the full fresh reference and public/private tests under the unchanged60s limit. The final science report records both the sum of eight actual cold trajectory runtimes and complete subprocess wall times.

## Preserved pre-freeze corrections

The bounded prototype originally mislabeled endpoint.45 as partial capture. Its own results showed zero captured fraction. That source and report were preserved, the label was corrected, and a genuine half-band capture endpoint was added with explicit action-integral splitting. The complete peer-reviewed history is copied unchanged under `prototype/`.

The first package validation completed its256 noise checks, scored refinements and domain trajectories but then failed an unjustified1e-10 symmetric-mean assertion: split versus unsplit action quadratures differ by1.06e-7 at a small-energy partial-capture endpoint. Increasing their order reduces the difference. Final validation checks the physical symmetry pointwise in action and reports the distinct mean-quadrature residual and refinement.

A separate near-threshold check found that a branch root bracket ending1e-10 below the saddle could exclude an arbitrarily small captured segment. The oracle now uses exact minimum and saddle endpoint areas. The initial source, logs, failure and all16 before/after endpoint probes are preserved in `history/initial-validation/` and `results/`. The dataset and grading never changed. Every original256 fit and scored/calibration prediction is compared against the repaired source. Exact unchanged trajectory records can be reused during this targeted validation only after reference-source hash verification; final local verifier subprocesses compute their trajectories afresh. The final report states this execution history explicitly.

## Sources and overlap

- A. Neishtadt, *Averaging method for systems with separatrix crossing*: https://arxiv.org/abs/1705.04347. Sections2.1–2.4 support ensemble averaging, nondegenerate saddle capture and growth-rate probabilities.
- P. Stabel and J. R. Anglin, *Dynamical change under slowly changing conditions: the quantum Kruskal–Neishtadt–Henrard theorem*: https://doi.org/10.1088/1367-2630/aca557. The classical KNH discussion and quartic example support the phase-area construction. The accessible full text read for this prototype is preserved by URL in its feasibility record: https://inspirehep.net/files/6605eec5cbece001798c0edfc5e0c46d.
- R. Bürkle et al., *How to probe the microscopic onset of irreversibility with ultracold atoms*: https://arxiv.org/abs/1903.04834. Its ensemble and limit discussion motivates avoiding a single-orbit or zero-width capture claim. It does not establish the numerical values of this one-way example.

The actual closest archived AUTHOR read was `archives/screened/tasks/collisionless-trap/AUTHOR.md`: noncrossing mode actions versus thermal equilibration. The broad adiabatic/ergodicity theme is shared, while this task changes the topology of the energy shell and redistributes a classical ensemble into disconnected lobes. The prototype contains the overlap audit and source hashes. This package does not overwrite or relabel any earlier task or outcome.
