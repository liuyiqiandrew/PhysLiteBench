# Magnetic tracer R9

R8 contains a completed passing solution that restores the antisymmetric
fast-correlation term. Preserve every R8 result. R9 extends the channel to
two directional thermal fluid reservoirs with unequal temperatures, fixed
orthogonal axes, independent Wiener forces and the same isotropic total drag.
Every microscopic rule and parameter range is public. The supplied completed
approximation now includes correct finite-mass circulation drift and all
prior R8 physics; it drops only correlated longitudinal/detector noise.
Public helper defaults for the unknown friction have also been removed.

For Theta=R diag(1+a,1-a) R^T, the fluid fast-noise covariance in valve state s
is2*gamma_s*T_s*Theta. Correctors are unchanged and conditional covariances
solve the full coupled Lyapunov system. D=sym(sum C_s S_s P^T) now has a
nonzero off-diagonal component. The mechanical detector drift remains
g=a(x)*b_y+a_prime(x)*K_yx. Its noise covariance with x is2*a(x)*D_xy.
Consequently the forward tilted generator has A1=diag(g)-2*dx*diag(a*D_xy)
and A2=2*diag(a^2*D_yy). Ignoring the mixed term preserves density and mean
circulation but changes noise and third cumulants.

The independent full finite-mass Kramers reference adds the off-diagonal
velocity diffusion before tilting the actual mechanical path integral.
Validation covers bath-axis/temperature relabelling, field/force/bath reflection,
energy and reservoir entropy balance, positive covariance, joint numerical
refinement, isotropic regression and256fixed-seed noise fits. Tolerances and
600/60second budgets remain unchanged; evaluate exactly3fresh unhinted
GPT-5.6Luna High attempts after scientific and Docker controls pass.
