# Magnetic tracer R7

R6 had one completed pass, one physical omission, and one timed-out mathematical
implementation. Preserve all results. The new preparation changes only the
passive detector: it records integral a(x) dy along each finite-mass path and
then takes the prescribed stationary long-time and small-mass limits.

The complete shortcut retains R6's correct valve-conditioned spatial process
and computes a Stratonovich path functional of that position diffusion. The
physical finite-mass functional has drift a*b_y+a_prime*K_yx, where
K=sum_s C_s S_s P^T, whereas the shortcut uses its symmetric part D. The
antisymmetric fast velocity correlation remains measurable in the path
integral. Since D_xy=0, the joint effective x/Y tilted generator has scalar
counting terms q*g+q^2*a^2*D_yy. Its first three eigenvalue derivatives give
the requested stationary cumulants. Equilibrium physical mean circulation
vanishes even though the naive position line integral can have a nonzero mean.

The independent reference tilts the full finite-mass two-state Kramers
generator by q*a(x)*v_y before eliminating anything. It uses four-dimensional
Hermite and spatial Fourier modes and extrapolates three positive masses.
The two representations are checked with independent refinement, equilibrium,
field and spatial symmetries, old preparation limits, and calibration noise.
Tolerance remains 2.5%, with 600/60-second limits and exactly three fresh
unhinted GPT-5.6 Luna High trials after all scientific and Docker controls.
