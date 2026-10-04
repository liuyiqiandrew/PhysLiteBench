# Two-bath magnetic heat leak: prototype only

No task, calibration dataset, grader or evaluation has been created. The small-field limit is physically sound but is not recommended as the next benchmark in its present form.

For a symmetric positive stiffness K, isotropic drag gamma and two independent Cartesian reservoirs with temperatures Tx and Ty, use

    dx = v dt
    m dv = (-K x - gamma v + B J v) dt + sqrt(2 gamma diag(Tx,Ty)) dW
    J = [[0,1],[-1,0]], B=b*sqrt(m).

The stationary heat rate entering the particle from bath x is gamma*(Tx/m - <vx^2>). The definition is at finite mass before the joint limit. The Lorentz force does no total work but transfers energy between the two velocity components.

Write Y=<x v^T>=y J, delta=<vx^2-vy^2>, c=<vx vy>, s=tr K, d=Kxx-Kyy and h=Kxy. Exact stationary covariance balances give

    gamma*delta - 2B*c - 2h*y = gamma*(Tx-Ty)/m
    B*delta + 2gamma*c + d*y = 0
    m*h*delta - m*d*c + gamma*s*y = 0.

The heat is -B*c-h*y. With B=b*sqrt(m), its limiting value is

    Qx = (Tx-Ty)/gamma * [h^2/s + b^2/2], Qy=-Qx.

The limit of the position process is the zero-field OU process. Consequently its exact limiting covariance and lag covariance cannot depend on b. A position-based heat surrogate necessarily drops b altogether. There is no natural way to retain substantive b dependence in that same limiting surrogate without changing the observable or regime. This makes the candidate weaker than the desired complete scaffold using every control. The independent physics reviewer reached the same conclusion.

The numerical prototype solves the full scaled finite-mass four-dimensional Lyapunov problem independently of the three scalar moment equations. In 32 random controls the refined small-mass extrapolation agrees with the stated limit within 3.89e-9; direct Lyapunov versus scalar heat differs by at most 6.97e-10. The largest heat-balance residual is 1.36e-9 and the Lyapunov residual is 8.89e-15. `report.json` preserves all controls, masses and predictions. These are exploratory checks, not a benchmark validation.

A possible stronger alternative keeps a fixed positive mass and the exact full Gaussian response in the supplied solver. It assigns only trap-mediated exchange diag(KY) to bath heat. That is a coherent channel attribution but omits the Lorentz-mediated kinetic exchange (-B<VxVy>, +B<VxVy>). Both channels and all field-dependent covariances are finite and substantive. At B=0 the channel attribution is exactly calibrated. For m=.4, gamma=1.1, K=[[1.2,.5],[.5,1.5]], (Tx,Ty)=(1.7,.7), B=.8, bath-x heat is .5085363514 while trap-mediated exchange is .0412430462. This has not been developed into a task; it would still need a distinctness decision, calibration identifiability and full independent controls.

Archive inspection found no exact magnetic Cartesian two-temperature gyrator task. Nearby mechanisms are retained `entropy-anomaly` (spatial temperature gradients and cubic velocity relaxation), retained `magnetic-tracer` (mechanical area under singular elimination), archived `active-bath-work` (colored active forcing at a joint time-scale limit), and archived `thermal-bodies-r11` (discarded cross-mode reservoir correlations). The small-field version shares enough kinetic-elimination structure that it should not be presented as wholly unrelated.

Primary sources:

- Abdoli et al., *Brownian magneto-gyrator as a tunable microengine*, https://arxiv.org/abs/2110.05284. The paper identifies momentum-mediated heat transfer requiring finite-mass information. Its published form, *Tunable Brownian magneto heat pump*, https://pmc.ncbi.nlm.nih.gov/articles/PMC9352690/, gives the separate Cartesian heat balance and Lorentz velocity-correlation term. The b*sqrt(m) scaling and finite limiting sum above are our explicit derivation, not a quoted result.
- Filliger and Reimann, *Brownian Gyrator: A Minimal Heat Engine on the Nanoscale*, https://doi.org/10.1103/PhysRevLett.99.230602, for the tilted harmonic two-bath apparatus.
