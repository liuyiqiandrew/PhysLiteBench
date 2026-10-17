# Independent finite-bias rotor proposal review

This author-only prototype is a read-only scientific review. It changes no task,
grading, calibration or model trial. Revision 6 is fully preserved under
`../../../archives/rotating-reservoir-r6`.

The same constrained rotor has velocity `v`, OU bath A, and two motor states
`s=±1` switching at the disclosed symmetric rate. In state s the mean incoming
gas velocity is `u=s*Omega`. Equal effective masses exchange velocities at a
collision, `v'=g`, with differential arrival rate
`3.6*((1-eta)+eta*abs(v-g))*Normal(g;u,Tb) dg`.

An outgoing gas particle deposits heat
`h=((v-u)^2-(g-u)^2)/2` in the moving thermal reservoir. The instantaneous work
on the rotor from the motor is `u*(g-v)`, so the individual event obeys
`Delta(v^2/2)=work-h`. A motor-state flip gives no instantaneous rotor impulse
or rotor work. The heat observable is the sum of these gas heat deposits. After
stationary preparation, the proposed optional measurement is
`theta(chi)=lim_(t->infinity) log E[exp(-chi*Qgas(t))]/t`, in inverse time, with
`chi` in inverse energy and `abs(chi)<=.35`. Mean and zero-frequency noise
remain available exactly as in revision 6. Calibration and the unknown drag
are unchanged.

The completed Gaussian current approximation uses the exact physical mean
`mu` and noise `c2`, then predicts `-chi*mu+chi^2*c2/2`. It is the Gaussian
extension of the long-time central fluctuation law to finite exponential
bias. The correct finite-bias observable depends on the biased trajectory
ensemble and higher current cumulants. A future public starter must contain
only this Gaussian readout, with exact revision-6 mean and noise; the full
tilted generator belongs exclusively to private scientific controls.

`spectral.py` independently discretizes the continuous backward generator

`L_chi f_s = gamma*Ta*f_s'' - gamma*v*f_s' + kappa*(f_-s-f_s)
             + integral rate(v,g,s)*(exp(-chi*h)*f_s(g)-f_s(v)) dg`.

Chebyshev differentiation and split Gauss-Legendre collision quadrature are
independent of the positive finite-volume oracle proposal. For numerical
conditioning it uses `f_old=exp[-q*(v-u)^2/2]*f_new`, with `q=min(chi,0)`.
The original reflecting cutoff condition transforms to
`f_new'=q*(v-u)*f_new`. An explicitly separate gauge-Neumann control tests
cutoff sensitivity; it is never claimed to be the same finite-cutoff model.
All principal eigenvalues in the screen are real and spectrally separated.

For every bounded preparation and bias, use the positive test function
`f(v)=exp[a*v^2/2]`, `a=.4-.5*chi`. Then `a>=.225`, `a<.625<=1/Ta`, and
`.225<=a+chi<.7142857<=1/Tb`. The tilted gas gain divided by f has a negative
quadratic velocity exponent, a finite Gaussian gas integral, and at most a
linear prefactor. The OU contribution has negative quadratic leading term.
The motor flips leave f unchanged. Thus `L_chi f/f` has a finite uniform upper
bound. The ordinary stationary process has finite expectation of f by the
same unmarked OU/jump Lyapunov estimate. This establishes finite exponential
growth for the entire public bias box; it does not advertise a maximal strip.
The chosen box lies strictly within `abs(chi)<1/max(Ta,Tb)`.

For zero flow and constant collision clock, the OU interval transform is
`M_chi(t)=[1+(1-exp(-2*gamma*t))*(chi*(Ta-Tb)-chi^2*Ta*Tb)]^(-1/2)`.
The renewal root solves `nu*integral exp[-(nu+theta)*t]*M_chi(t) dt=1`.
The regrouping of event heat into interval heat differs by endpoint energies;
use this as a physical anchor only where endpoint exponential moments are
finite. Independent Laguerre renewal quadrature and direct two-Gaussian
quadrature reproduce this anchor. Zero-flow fluctuation symmetry is checked
only when both `chi` and `1/Tb-1/Ta-chi` are admissible.

`validate_proposal.py` preserves 128 full-box corner checks, 24 random checks,
four cutoff/resolution levels, alternate gauge and boundary controls,
independent finite-volume comparisons, renewal and Gaussian-cycle anchors,
zero-flow fluctuation symmetry and zero-bias recovery of mean and noise.
Results are in `../../../results/rotating-reservoir-r7-brownian-proposal-audit.json`.
The three bounded physical group proposals in
`final-coherent-group-proposal.json` have Gaussian relative RMS gaps of
25.0%, 11.1% and 9.0%. A zero-bias exact-zero control belongs to scientific
validation, avoiding division by a vanishing signal in relative hidden scoring.

Reproduction from the staging root:

```sh
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -B \
  results/prototypes/rotating-reservoir-r7-brownian/validate_proposal.py
```

This proposal depends only on the saved sibling root FV prototype and the
immutable revision-6 archive; it does not depend on temporary external modules.
