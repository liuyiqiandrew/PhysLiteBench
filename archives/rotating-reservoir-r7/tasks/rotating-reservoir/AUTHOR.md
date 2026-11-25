# Rotating reservoir, revision 7: finite exponential heat bias

The retained ring rotor, relative-speed gas collision selection, joint motor dynamics, mean heat, full temporal noise and unknown drag are exactly those of revision 6. This revision adds an optional calorimeter exponential readout at a disclosed finite energy bias. The completed source retains the entire correct revision-6 physical dynamics and both measured cumulants, then extends the central Gaussian heat-fluctuation law to finite exponential bias. The physical distinction is the higher current cumulants and biased trajectory ensemble. Calibration bytes, parameter and prediction gates, and 600/60-second limits remain unchanged.

## Previous evidence and preservation

Revision 6 has three raw verifier reward passes and two ApiUsageLimitError harness exceptions. All three native trajectories nevertheless include task_complete final records. The fixed-drag independent review examines 40 preparations and both mean/noise readouts. N8kWZsn correctly solves the flux generator and full correlations, agreeing within 1.27e-7. JF9FXP7 correctly uses a finite Hermite flux/counting generator, with maximum numerical approximation error .000248. FnqWRaG correctly implements a finite central velocity grid and the full post-jump Poisson correction, with maximum discretization error .00437. A gauge-row-only numerical repair changes the latter discrepancy negligibly; no physical-model defect is claimed. All raw passes, usage interruptions, installed sources and trajectories are preserved and this batch cannot establish 0/3 physical failures.

The complete immutable archives/rotating-reservoir-r6 has 188 manifest-verified files. It contains the task, both completed controls, independent reference, source hashes, validators, all three native/frozen plain jobs, both fresh Docker controls, proposal versions and review. Its manifest SHA is b22d29171dc32a8f42ab6dff7eb78093c9ad9113f69156806945a377d63f738f. Prior full archives remain separate. No earlier task is resampled unchanged.

## Apparatus, calibration and observable

Rotor radius, inertia and gas effective mass are one. Velocity is unbounded and the radial constraint does no work. Between collisions dv=-gamma*v dt+sqrt(2*gamma*Ta)dW. An independent symmetric motor label s switches at kappa and sets u=s*Omega. A flip changes neither rotor velocity nor rotor energy and gives no instantaneous particle work or recorded heat. The literal incoming differential rate is

    W_s(v,dg)=3.6*((1-eta)+eta*abs(v-g))*Normal(g;u,Tb) dg.

Equal masses exchange v'=g and g'=v. Gas heat is h=((v-u)^2-(g-u)^2)/2. Each collision satisfies Delta(v^2/2)+h=u*(g-v); the right side is flow-maintaining motor work. This heat, the motor process and every preparation are public. The retained bounded domain is gamma in [.4,1.1], Ta in [.6,1.6], Tb in [.6,1.4], Omega in [-.65,.65], kappa in [.8,2] and eta in [0,1].

The unchanged 288 calibration records use eta=Omega=0 and default mean, varied temperature contrasts, sigma=.001, seed71063. The exact mean gamma*nu/(2*gamma+nu)*(Ta-Tb), nu=3.6, identifies the only unknown gamma=.67. Fit and uncertainties are unchanged; the public/private records are byte-identical to revision 6. Mean is lim E[Q_B]/t and noise is lim Var(Q_B)/t, with every stationary source of randomness retained.

The optional third readout is

    theta(chi)=lim_(t->infinity) t^-1 log E[exp(-chi*Q_B(t))],

starting from the stationary joint ensemble. Statistic='scgf' selects it; heat_bias=chi is in inverse energy, default zero, and lies in [-.35,.35]. Output units are inverse time. Bias affects only this readout and lists may mix all three statistics. The stated window satisfies abs(chi)<1/max(Ta,Tb) for every public preparation. We advertise this conservative stationary moment window, not a maximal strip from a cycle transform.

## Exact tilted generator and numerical independence

The backward generator for exponential heat is

    L_chi f_s=-gamma*v*f_s'+gamma*Ta*f_s''
      +kappa*(f_-s-f_s)
      +integral W_s(v,dg)*(exp(-chi*h)*f_s(g)-f_s(v)).

The escape rate is unweighted. The dominant eigenvalue gives theta. At zero bias theta=0, theta'(0)=-mean and theta''(0)=noise. This is a finite-bias path ensemble; the central-limit theorem fixes only its small-bias expansion.

The private oracle retains the revision-6 positive Scharfetter-Gummel OU generator and literal midpoint collision quadrature. An exact discrete similarity D_s(v)=exp[-q*(v-u)^2/2], q=min(chi,0), conditions the operator. The OU entries receive D(post)/D(pre), the collision gain receives the remaining exp[-(chi-q)*h] mark while its unweighted escape loss is retained, and motor off-diagonals receive exp(-2*q*s*Omega*v). For negative bias the collision mark cancels; positive bias retains the literal marked gain and avoids a growing similarity gauge. The matrix remains irreducible Metzler. A strictly positive Lyapunov vector supplies a Collatz upper bound, then a shift above that bound and inverse iteration find the Perron eigenvalue. Dense full eigenvalue solves independently check this numerical selection. The oracle uses 161/322 cells and second-order Richardson for SCGF; the unchanged mean/noise implementation retains its 241/482 grids or exact constant-clock polynomial closure. Zero bias returns exactly zero.

The standalone private reference differentiates the continuous OU generator by Chebyshev collocation and integrates the literal marked jump kernel by Gauss-Legendre quadrature split at g=v. Barycentric interpolation evaluates the post-jump function. Its conditioning gauge uses q=min(chi,0), and the original reflecting numerical cutoff condition transforms consistently to y'=q*(v-u)*y. A separate gauge-Neumann sensitivity control is labelled as a different cutoff boundary. The production reference imports no oracle, finite-volume code, archive or prototype. Node91/121/151/181, quadrature100/140/180/220 and extent7/9/11/13 refinements distinguish numerical cutoffs from the physically unbounded apparatus.

The proposed same-family physics and actual generator sources were independently approved by the root teammate. The saved author prototype audit verifies 128 full-box corners, 24 interiors, gauge and cutoff controls, independent finite-volume agreement, renewal anchors, admissible fluctuation symmetry and zero-bias cumulants. Its source and all exploratory group screens remain under results/prototypes/rotating-reservoir-r7-brownian; the sibling root FV prototype imports the immutable r6 archive.

## Finite stationary exponential moments and analytic controls

For every allowed bias choose f(v)=exp(a*v^2/2), a=.4-.5*chi. Then a>=.225, a<.625<=1/Ta and .225<=a+chi<.7142857<=1/Tb. The marked gas gain divided by f has negative quadratic v exponent, a finite Gaussian gas integral and at most a linear relative-speed factor. OU gives a negative quadratic leading term; the motor flips leave f fixed. Thus L_chi f/f has a finite uniform upper bound. The ordinary stationary process has finite E[f] from the corresponding OU/reset Lyapunov estimate. This proves finite exponential growth throughout the public box. The previous full-domain polynomial moment, uniqueness, passivity and microscopic heat/work proofs are preserved unchanged with the r6 source and science report.

For zero flow and constant collision clock nu, an independent regenerative OU interval has g~N(0,Tb) and v(t)=exp(-gamma*t)*g+sqrt(Ta*(1-exp(-2*gamma*t)))*xi. Direct two-Gaussian integration gives

    M_chi(t)=[1+(1-exp(-2*gamma*t))*(chi*(Ta-Tb)-chi^2*Ta*Tb)]^(-1/2).

The renewal eigenvalue solves nu*integral exp[-(nu+theta)*t]*M_chi(t)dt=1. Event heat and grouped interval heat differ only by endpoint energies, so this is used only within finite stationary endpoint moment ranges. Independent Laguerre quadrature checks the root against direct adaptive quadrature and both generator discretizations. At zero flow local detailed balance gives theta(chi)=theta(1/Tb-1/Ta-chi) when both arguments are admissible. No unsupported driven single-current symmetry is assumed. Numerical derivatives at zero bias recover the retained mean and noise.

## Completed Gaussian approximation and experiments

The public starter and scripts/rotating_reservoir_baseline.py are identical completed controls. They contain the full correct collision generator, mean and temporal noise, and only the finite-bias approximation

    theta_G(chi)=-chi*mean+chi^2*noise/2.

There is no exact tilted-generator branch, answer flag, hidden preparation selection or private reference in the image. This is a coherent Gaussian current closure with the correct mean and covariance, not an unfinished implementation or a repeated old Maxwell approximation. It is exact in the small-bias expansion through second order and matches all calibration and retained mean/noise anchors.

Three four-case SCGF groups compare rotating equal-temperature positive bias, cold-A/hot-B rotating positive bias and hot-A/cold-B rotating negative bias. They vary disclosed speed, thermal drive and persistence while keeping eta=1. The true-parameter Gaussian relative RMS gaps are .25031,.11121 and .09000. Each reference group has a nonzero RMS signal; no tightened gate or division by an exact-zero readout supplies the separation. The fourth group measures old constant-clock means and full-flux stationary noise. All groups retain the inherited .04 relative RMS gate; gamma/.03 and reduced chi-square/1.5 are unchanged. Scientific zero-bias identities are separate exact-zero controls.

## Reproduction and evidence boundary

From this candidate root run:

    OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -B scripts/validate_rotating_reservoir.py

The validator reads unchanged calibration, checks AST identity of the retained mean/noise dynamics against immutable r6, independently checks SCGF, all full-box endpoints, microscopic calorimetry and analytic limits, runs 256 newly noisy fits at seed146824, and runs isolated public/private pytest for both completed controls. It records all source hashes, errors, fitting margins and timings in results/rotating-reservoir-r7-validation.json. The source must fit the correct drag and calibration, pass stationary anchors, and fail only the three finite-bias physical groups. Fresh Docker and frozen Luna-high trials are separate parent-owned evidence, with every exception retained. No paid model trial is run by this validator.

The Docker context remains environment/ and copies only README, public starter, public test and unchanged calibration. Solution, independent reference, hidden cases, true drag, author explanation and optional hint remain private. This candidate changes no task resources. Source peer review and passing controls must precede freezing; scientific validity alone is not a claim of achieved 0/3.

The initial full-chi discrete gauge stopped an author motor-reversal check at extreme constant-clock positive-bias corners, with maximum error8.68e-8; dense eigenvalues also showed conditioning sensitivity. The original source, validator, log and independent diagnostic are preserved. The piecewise gauge above changes only exact numerical similarity, retaining the physical tilted generator and all grading. Final validation checks exact motor-label permutation and dense/Perron agreement.
