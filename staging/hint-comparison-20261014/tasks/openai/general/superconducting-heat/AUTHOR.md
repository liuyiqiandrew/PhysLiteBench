# Superconducting heat R11

R10 has a completed Luna solution that correctly accounts for linear spin
feedback in zero-frequency work noise. Preserve the entire batch. R11 adds
the leading extensive third cumulant for the same stochastic microscopic
apparatus. No quantum rules, spin rates, mean observables, noise observables,
calibration or tolerance change. The complete shortcut now has correct R10
noise but treats linearly screened work increments as independent stationary
Poisson marks when computing the third cumulant.

Let H(mu,p,q)=sum_e r_e(mu)*(exp(p.dot(ds_e)+q*w_e)-1) include both directions
of every tunnel and spin-flip event. At the stationary saddle, H_p=H_mu=0.
Its first response has p_prime=-J^-T*h_mu and
mu_prime=-J^-1*(Q_sw+Q_ss*p_prime). The third derivative of the saddle value
is the third directional derivative of H along(mu_prime,p_prime,1):
sum r*ell^3+3*r_prime*ell^2+3*r_second*ell, ell=w+p_prime.dot(ds).
Terms with second saddle responses cancel by the first saddle equations.
The oracle differentiates Fermi/Pauli rates analytically through second order;
the shortcut drops both rate-response terms and retains only r*ell^3.

The independent reference uses Fock/Bogoliubov rates, constructs the full
finite counting-field Hamiltonian and solves its four nonlinear stationary
saddle equations at positive and negative counting fields. Third finite
differences with independent step refinement verify the oracle. Controls,
domain tests, numerical refinement and 256 noisy fits precede exactly three
fresh unhinted GPT-5.6 Luna High trials with unchanged600/60second budgets.
