# Entropy anomaly — revision 2

Revision 2 asks whether an isotropic kinetic-heat correction can be extended to unequal drag by averaging the friction tensor. Calibration already measures entropy in nonuniform-temperature baths with equal drag. The completed shortcut therefore includes the known two-dimensional entropy anomaly and fits every calibration preparation exactly. Hidden experiments change the drag ratio and expose the separate relaxation of the two heat-carrying velocity moments.

Revision 1, including its full task, controls, science reports, fixed-friction repair and six reviewed trials, is preserved in `archives/entropy-anomaly-r1`. Its plain result was 2/3 and hint result 3/3. This is a new physical setup and calibration at revision 2; those outcomes are not evidence for this revision.

## Apparatus and closure

The particle has two periodic position coordinates and two velocity components. Its constant drag tensor is diag(gamma,lambda*gamma), and each component's thermal noise has the corresponding local fluctuation-dissipation amplitude. Temperature varies only with x. Both constant force components are known. The only fitted parameter is gamma; lambda is a known experimental setting.

The public measurement is local bath entropy from finite-mass Stratonovich heat, first averaged in the stationary state and then taken to zero mass. Background heat needed to maintain the bath profile is excluded. This order distinguishes calorimetry from entropy assigned only to an eliminated positional process. The general phenomenon is described in [Celani et al., Anomalous thermodynamics at the micro-scale](https://arxiv.org/abs/1206.1742); the anisotropic coefficient here follows directly from the stated microscopic moments.

The x density rho is normalized over one period; the full position density is rho(x)/(2*pi). The limiting currents, integrated over y, obey

    gamma*Jx = Fx*rho - (T*rho)',
    Jy(x) = Fy*rho(x)/(lambda*gamma).

Both controls compute this state correctly. Their positional entropy contribution is

    integral [gamma*Jx^2 + lambda*gamma*Jy(x)^2]/[T(x)*rho(x)] dx.

The shortcut isotropizes only the fast thermal drag, using gamma_mean=(gamma_x+gamma_y)/2 in the two-dimensional kinetic coefficient 2/(3*gamma_mean). This is a complete mean-drag approximation; lambda still enters both positional current and kinetic heat. It conserves probability, returns positive entropy and correctly reproduces equal-drag baths even when their temperature is nonuniform.

The correct kinetic coefficient is

    C = (1/2) * [1/gamma_x + 1/(gamma_x+2*gamma_y)],
    kinetic entropy = C * integral rho*(T')^2/T dx.

Replacing two distinct velocity relaxation channels by one mean rate is the physical defect. The coefficient reduces to 2/(3*gamma) at equal drag and approaches 1/(2*gamma_x) for arbitrarily fast transverse relaxation. No missing input, coding error or invalid probability density is required for the shortcut failure.

## Moment derivation

Let M_ab(x)=integral vx^a*vy^b*p(x,vx,vy) dvx dvy, after integrating over the uniform y coordinate. The leading local Gaussian second and fourth moments determine the third-moment sources. At stationary small mass,

    m*M30 -> 3*T*Jx - rho*T*T'/gamma_x,
    m*M12 -> T*Jx - rho*T*T'/(gamma_x+2*gamma_y).

The mixed moment loses one power of vx at rate gamma_x and two powers of vy at rate 2*gamma_y. Its denominator cannot be obtained by treating all velocity directions as equally damped. Constant Fy contributes its mechanical work; its term in the leading mixed third-moment balance is lower order in mass.

The stationary finite-mass energy balance gives

    entropy = integral (Fx*Jx + Fy*Jy)/T dx
              - (m/2)*integral (M30+M12)*T'/T^2 dx.

The periodic integral of Jx*T'/T is zero. Substitution yields the coefficient C above and the positional term. At zero force, rho is proportional to 1/T, giving the independent closed form

    entropy = C*T0*k^2*(1-sqrt(1-a^2)).

At uniform temperature the exact finite-mass result is [Fx^2/gamma_x+Fy^2/gamma_y]/T.

## Independent private reference

The verifier never uses the limiting kinetic coefficient. It solves finite-mass Kramers equations in Fourier x and Gaussian-weighted Hermites of wi=sqrt(m/T0)*vi. Since nothing varies with y, the equations are triangular in y-Hermite degree. Degrees 0,1,2 close exactly for the moments entering calorimetry; higher y degrees cannot feed back into them.

Degree 0 is the stationary x Kramers equation. Degree 1 has the exact conditional mean Fy/gamma_y. Degree 2 solves a second sparse linear system with its own 2*gamma_y relaxation and temperature-dependent source. The direct finite-mass entropy is the sum over components of

    gamma_i/m * integral [T0*(c00+sqrt(2)*c_2i)/T - c00] dx.

The subtracted term is the mean Stratonovich noise work. The reference uses 79 spatial nodes and 32 x-velocity modes, then extrapolates three positive masses h,h/2,h/4 quadratically to zero. Here h=.004*min(gamma_x,gamma_y)^2/(T0*k^2). This keeps even the slower velocity relaxation controlled. The combination is rate(h)/3-2*rate(h/2)+8*rate(h/4)/3.

Validation halves the mass scale and increases resolution to 95 spatial nodes and 48 velocity modes. A separate direct-heat versus third-moment heat check confirms the calorimetric sign and normalization. Conditional transverse velocity variance is positive. The x-velocity spectral approximation has tiny tail oscillations whose integrated negative mass decreases from 5.34e-6 at 24 modes to 8.82e-9 at 48 and 9.11e-11 at 64; the heat observable is already converged far below grading tolerance.

## Calibration and validation

All 200 calibration measurements use lambda=1, thermal contrast 0.25 through 0.6, nonzero forces in both directions, and wave numbers 1 through 3. The coefficient of 1/gamma is known for every preparation, so weighted linear regression identifies inverse friction. Both controls share this exact calibration map. Regenerate only with the validator's explicit `--generate` flag; the seed is 9331.

Run from the repository root:

    OPENBLAS_NUM_THREADS=1 uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 scripts/validate_entropy_anomaly.py

The fitted friction is 1.1005442339 versus the private value 1.1; reduced chi-square is 0.92336. Independent finite-mass calibration-reference error is at most 1.20e-8. All 256 calibration-noise realizations pass, with maximum relative friction error 0.00380. Hidden sensitivity is checked at the observed fitted-parameter extrema; this is not described as exhaustive hidden Monte Carlo testing.

Oracle hidden normalized RMS error is at most 0.000495. The mean-drag shortcut scores 0.219, 0.428 and 0.452. At the observed noise extrema, maximum oracle error is 0.00382 and minimum shortcut error is 0.216. Each group uses RMS truth as its scale and a 0.04 threshold, comfortably above calibration and numerical uncertainty.

The kinetic reference disagrees with the oracle by at most 3.91e-9 on hidden cases; halving the mass changes results by 3.46e-9, and increasing the spatial/velocity basis changes them by 3.30e-11. Allowed-range corner disagreement is below 4.67e-8. Direct and third-moment heat agree within 1.84e-12. Checks also cover uniform temperature, exact zero-force entropy, equal drag, probability normalization, force reversal, constant-force current and inverse-friction scaling.

Evidence is in `jobs/entropy-anomaly-r2-validation/summary.json` and `results/entropy-anomaly-r2-science.json`. Peer reviews record current source hashes in `results/constraint-hardening-physics-review.json` and `results/materials-entropy-r2-kinetic-review.json`. Harbor controls and Luna difficulty evaluation are separate root-scheduled checks. No model failure rate is claimed from the scientific controls alone.
