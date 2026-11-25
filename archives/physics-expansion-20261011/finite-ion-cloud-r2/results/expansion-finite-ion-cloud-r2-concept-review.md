# Finite ion cloud, revision 2: concept review

**Verdict: approve for packaging.** This is a substantive finite-cell revision. Revision 1 tested correlations in the number of delocalized ions in a dilution limit. This revision tests angular pair correlations in a finite canonical Gibbs ensemble. No model outcome is implied by this approval.

Keep the genuine two-dimensional logarithmic Hamiltonian, central charge (+Nq), (N=1,2) mobile charges (-q), homogeneous dielectric without images, and uncharged hard outer wall. State that equilibrium is established at finite (R/a=h\in[1.25,12]), with no dilution limit. Define pressure by varying (R) at fixed (a,N,T,\Gamma), where (\Gamma=ug). In particular, pressure is not a derivative at fixed radius ratio. The proposed (u\in[0.1,3]), (g\in[0.7,1.4]) domain has a finite, positive partition function and pressure throughout; repulsive coincident-ion configurations cause no collapse.

For one ion, write (L=\log h), (s=2-\Gamma). Direct radial integration gives

\[
\phi_1=(1-e^{-2L})\frac{s}{2(1-e^{-sL})},\qquad
\phi_1\big|_{s=0}=\frac{1-e^{-2L}}{2L}.
\]

At fixed (h>1,u>0), this is strictly decreasing in (g), so the proposed one-ion calibration identifies the common unknown. Changing absolute (a) at fixed (h) leaves the dimensionless observable unchanged, as required by logarithmic scale invariance.

For two ordered radii, let (d=\log(r_2/r_1)), (b=2-\Gamma), and (c=4-3\Gamma). Integrating the relative angle gives

\[
H(d)={}_2F_1(-\Gamma/2,-\Gamma/2;1;e^{-2d}),
\quad
Z\propto\int_0^L H(d)e^{bd}\frac{e^{c(L-d)}-1}{c}\,\mathrm d d,
\]

\[
Z'=\int_0^L H(d)e^{bd+c(L-d)}\,\mathrm d d,
\qquad \phi_2=\frac{1-e^{-2L}}4\frac{Z'}Z.
\]

These formulas and the annular-area factor are correct. At (c=0), the quotient is (L-d); implement its continuous limit. The one-ion (\Gamma=2) limit also needs stable evaluation. There is no finite-cell condensation branch with exactly zero pressure.

The completed control has a defensible physical interpretation: restrict the Gibbs trial distribution to independent uniform azimuths conditional on the two radii, while optimizing their joint radial distribution without restriction. Its variational free energy contains the uniform angular mean of the pair potential, (\langle\log|\mathbf r_1-\mathbf r_2|\rangle=\log\max(r_1,r_2)), yielding precisely (H=1). Both the exact ensemble and this control have uniform one-particle angles. Only the exact ensemble retains relative-angle correlations. Describe this restricted angular state rather than literal charged rings, which could introduce unwanted self energies. It is exact for all one-ion calibration preparations and has positive, nontrivial two-ion predictions.

The prototype separation is ample: at ((\Gamma,h)=(2,3)), exact/control pressures are (0.2578309392504/0.3019574654); at ((3,5)), they are (0.0464690207960/0.0798634812). Very weak coupling, very narrow annuli, and the eventual dilution limit suppress the distinction, so the proposed intermediate-radius hidden preparations are appropriate. The mechanism differs from the archived membrane charge-averaging task: this is a fixed two-particle positional correlation in a full Gibbs ensemble, not replacement of different chemical species by a mean valence.

The proposed direct positive quadrature over two radial coordinates and relative angle is an adequate independent reference. It evaluates the microscopic Boltzmann factor without the hypergeometric identity or analytic common-radius integral. I additionally integrated directly in the original ordered radii and angle at 96 and 160 Gauss nodes, using the outer contact integral for pressure. Across six cases spanning \(\Gamma=0.07\) to (4.2), (h=1.25) to (12), including (\Gamma=4/3,2), the 160-node maximum absolute disagreement with the prototype was (1.42\times10^{-11}). The independently evaluated boundary integrals satisfied (hZ_h-Z_{\rm inner}=(4-3\Gamma)Z) to (1.77\times10^{-12}). This is the two-ion specialization of the curved-wall contact identity; the uncharged outer-wall relation is (\beta P=n(R)). See [Mallarino, Téllez and Trizac, contact theorem](https://arxiv.org/pdf/1410.6499).

Package convergence checks, the ideal-gas and narrow-annulus limits, and the contact identity. Retention of the angle-independent Gibbs state despite the stated interacting apparatus would be a physical failure; failures to evaluate the angular integral or fit the identifiable parameter require separate classification.

Reviewed prototypes: `staging/gpt6-expansion/finite_ion_cell_prototype.py` and `staging/gpt6-expansion/finite_ion_cell_reference.py`. No task files were changed and no model trials were run.
