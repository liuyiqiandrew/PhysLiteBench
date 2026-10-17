# Finite recirculating contact proposal, rotating reservoir r8

This is a read-only scientific prototype, not an active task revision. The r7
task, controls and all three passing submissions remain frozen and fully archived.

The rotor has unit inertia and angular velocity v. One finite, unit-inertia gas
contact cell has velocity g and is retained after collisions. Bath A acts on v;
bath B thermalizes g toward u=s*Omega, where the independent symmetric motor
label s flips at kappa. In the same dimensionless energy/time convention as the
parent family,

    dv = -gamma*v*dt + sqrt(2*gamma*Ta)*dWa
    dg = -lambda*(g-s*Omega)*dt + sqrt(2*lambda*Tb)*dWb.

Independent Wiener increments Wa,Wb are independent of the motor and collisions.
The known collision rate is

    nu*((1-eta)+eta*abs(v-g)), nu=3.6,

and an elastic equal-mass collision swaps (v,g). It conserves total kinetic
energy and angular momentum. The gas particle is neither removed nor redrawn at
a collision; it carries the rotor's recoil into its subsequent thermalization
and future collisions. A motor flip changes only the thermostat's flow target:
neither particle receives an impulse, and there is no instantaneous particle
work or heat at that flip.

The thermal calorimeter at port B measures

    dQb = lambda*((g-u)^2-Tb)*dt
          -sqrt(2*lambda*Tb)*(g-u)*dWb.

This is the Ito form of minus (g-u) times the thermostat force in the
Stratonovich convention. Its Wiener increment is the SAME Wb that drives g.
Collisions and motor flips carry no instantaneous Qb. The stationary mean is
lambda*(E[(g-u)^2]-Tb), and the proposed diagnostic is the zero-frequency thermal
heat noise lim(t->infinity) Var(Qb(t))/t, with units energy squared per time.
Mean heat remains the default calibration readout, with units energy per time.
The mean motor work is lambda*(Omega^2-Omega*E[s*g]); stationarity gives

    gamma*(E[v^2]-Ta) + mean(Qb/t) = mean(motor work/time).

Proposed public bounds are gamma in [.4,1.1], Ta in [.4,2], Tb in [.4,1.4],
lambda in [.2,3], Omega in [-.65,.65], kappa in [.8,2], eta in [0,1]. Thus the new
contact remains genuinely finite. The parent gas-refresh limit is a useful
separate lambda->infinity control; it is not silently applied to these inputs.

## Calibration and completed approximation

For eta=Omega=0 the EXACT first/second moments close, although the stationary
joint law is generally nonGaussian. The static mean conductance is

    C = gamma*lambda*nu/[2*gamma*lambda+nu*(gamma+lambda)].

Known lambda and nonzero temperature contrasts identify gamma monotonically.
The same unknown drag bounds, calibration chi-square gate, parameter gate and
.04 relative prediction gate can be retained. The apparatus changes the old
infinite-gas calibration conductance, so calibration would be deliberately
regenerated and documented. No old data would be silently relabelled.

The completed approximate source retains gas recoil and finite lambda. It solves
all exact motor-conditioned first/second moment equations, closing only the
flux integrals by assuming the conditional pair (v,g) has a joint Gaussian law
with its full fitted mean and covariance. At eta=0 those exact mean equations do
not need any Gaussian assumption. For heat fluctuations it projects the exact
kinetic generator onto quadratic observables in that assumed Gaussian law and
uses its Poisson response, INCLUDING the shared-Wiener calorimeter cross term.
This is a completed pair-Gaussian kinetic approximation, not independent shot
noise, a heat-sign error, a missing contact, or an unfinished solver. Its source
contains no exact high-order kinetic solution or switch selecting that answer.

Even at eta=0 the conditional law is nonGaussian: random elastic swaps mix bath
temperatures without replacing the gas cell. Exact energy anticorrelations and
fourth cumulants matter to the noise but are absent from Gaussian factorization.
The source's eta=0 noise is consequently approximate, while its mean calibration
is exact. No discontinuous exact eta=0 noise branch is introduced to conceal this.

## Exact independent controls

For eta=Omega=0, let x=E[v^2],y=E[g^2], A=E[v^4],B=E[g^4],C4=E[v^2*g^2]. Then

    (2*gamma+nu)*x-nu*y = 2*gamma*Ta
    -nu*x+(2*lambda+nu)*y = 2*lambda*Tb
    (4*gamma+nu)*A-nu*B = 12*gamma*Ta*x
    -nu*A+(4*lambda+nu)*B = 12*lambda*Tb*y
    C4 = (gamma*Ta*y+lambda*Tb*x)/(gamma+lambda)
    C4-x*y = -nu*(x-y)^2/[2*(gamma+lambda)].

The exact quadratic Poisson corrector is av*(v^2-x)+ag*(g^2-y), with
D=4*gamma*lambda+2*nu*(gamma+lambda), av=lambda*nu/D and
ag=lambda*(2*gamma+nu)/D. The noise is

    2*lambda*Tb*y
    +2*lambda*[av*(C4-x*y)+ag*(B-y^2)]
    -8*lambda*Tb*ag*y.

The last term is the shared-Wiener term. In equilibrium this reduces to 2*C*T^2;
when gamma=Omega=0 total energy bounds Qb and its asymptotic noise is zero. The
root teammate independently implements all degree-four monomials, including the
motor label, and the degree-two Poisson problem at eta=0 for arbitrary Omega.

The positive finite-volume method uses a Cartesian velocity grid, independent
Scharfetter-Gummel OU transitions, exact grid swaps and motor reflections. It
marks ONLY bath-B OU transitions by minus their relative kinetic-energy changes,
then computes stationary and Poisson heat statistics. The independent weak
Hermite method integrates the continuous OU generator and its calorimeter
Feynman-Kac derivatives. Rotating to relative and total velocity lets
Gauss-Laguerre/Hermite quadrature integrate the nonsmooth flux matrix exactly
within each finite polynomial subspace. It shares neither OU edge marks nor
finite-volume matrices with the first method. Neither method calls the Gaussian
closure.

## Domain and scientific risks

For E=(v^2+g^2)/2, collisions and motor flips leave E unchanged. The generator
satisfies LE <= -min(2*gamma,lambda)*E + gamma*Ta+lambda*Tb+lambda*Omega^2/2.
For exp(a*E), any fixed 0<a<1/max(Ta,Tb), for example a=.2 throughout this box,
gives negative quadratic OU terms at large velocities; the bounded flow term is
absorbed by completing a square. Thus all stationary polynomial energy moments
needed for the noise are finite uniformly in the public domain. Positive,
nondegenerate OU diffusion and the finite-energy collision involution give a
unique stationary law. The symmetric collision intensity is unchanged by the
swap, so collisions do not introduce an undisclosed inelastic heat port.

The initial mild-box mean-only Gaussian separation was too small. Those negative
screens are retained. The proposed diagnostic instead uses genuine finite-cell
energy correlations in thermal heat noise. Temperature contrast is moderate:
Ta1.6,Tb.4,lambda2 produces about 14-17% source gaps even in the independently
closed eta=0 anchors. Partial/full flux groups retain the same mechanism and
require resolving the real velocity-selection correlations, rather than adding
an unrelated interaction. Final grouping, noise robustness, source positivity,
independent convergence and cold verifier runtime must pass before packaging.

Polynomial weak truncation can converge slowly for a cold bath; independent
positive-grid refinement is therefore mandatory. Grid noise has a finite-mesh
calorimetry error, so no single coarse grid is claimed to solve the continuum
model. At the initial selected flux case, FV101/201 Richardson noise agrees with
Hermite34 to about 3.6e-5 relatively; both refinements and their timings are saved.
The source is a moment/response closure rather than an exact positive Markov
process. Its covariance, projected decay, nonnegative noise, stationarity and
energy balance must be checked over the declared box, and that limitation must
be stated in author documentation. No evaluation outcome is promised.
