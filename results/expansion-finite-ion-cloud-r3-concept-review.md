# Finite ion cloud, revision 3: backup concept review

**Verdict: defensible as a pair-cluster approximation, pending numerical feasibility and pressure-separation checks.** This approves concept development only. It does not change revision 2 or classify its unfinished trials.

The same finite neutral two-dimensional logarithmic cell with N=1,2,3 is unambiguous. Keep the full pairwise Hamiltonian, fixed disk and uncharged outer wall, canonical ensemble, finite radius ratio, and pressure derivative at fixed N, disk radius, temperature and coupling. No three-body force is required: the shared particle positions already couple the three pair interactions. One-ion calibration retains global identification of the coupling; exact two-ion calibration additionally tests the complete pair angular response.

At fixed radii, define

\[
W_d(\theta)=(1+e^{-2d}-2e^{-d}\cos\theta)^{\Gamma/2},\quad
c_m(d)=\int_0^{2\pi}W_d(\theta)e^{-im\theta}\frac{d\theta}{2\pi}.
\]

For ordered logarithmic radii (s,s+x,s+x+y), the exact angular factor is

\[
\mathcal A(x,y)=\int W_x(\theta)W_y(\varphi)
 W_{x+y}(\theta+\varphi)\frac{d\theta\,d\varphi}{(2\pi)^2}
 =\sum_{m\in\mathbb Z}c_m(x)c_m(y)c_m(x+y).
\]

The proposed control keeps only (m=0): (H(x)H(y)H(x+y)). This is precisely the uniform-message Bethe angular partition approximation conditional on the radii. Each normalized pair belief has consistent uniform one-particle marginals; it neglects consistency around the closed triangle. Pair-cluster free energies and omitted loop corrections are standard statistical-mechanical approximations; see [Xiao and Zhou, partition-function loop series](https://arxiv.org/abs/1104.3631). The displayed continuous-angle specialization follows directly from Fourier integration here.

Use that interpretation explicitly in author documentation. It is **not** an exact Gibbs distribution of three planar positions, and it is not the restricted, normalized Gibbs variational trial distribution used for revision 2. Do not claim a variational upper bound for it. A separate auxiliary partition function with three independent pair-angle coordinates is positive and normalized, but those coordinates are a relaxed description, not simultaneously realizable particle angles. This limitation is the physical closure being tested, rather than an incorrect implementation of its equations. Avoid calling it an actual tree obtained by deleting an edge: all three pair energies remain in the approximation.

There is a simple independent loop check. At \(\Gamma=2\), writing (r=e^{-x},t=e^{-y}),

\[
\mathcal A=(1+r^2)(1+t^2)(1+r^2t^2)-2r^2t^2.
\]

The control omits the last term. Equal radii give exact factor 6 versus control 8. At this coupling the full N-ion angular Vandermonde integral also reduces the radial partition function, up to radius-independent constants, to

\[
Z_N\propto\prod_{j=0}^{N-1}\int_a^R r^{2j+1-2N}\,dr,
\]

providing a separate finite-pressure anchor for N=3. The angular-factor difference alone does not establish a useful pressure gap: common factors cancel in a logarithmic derivative, and very narrow annuli suppress the distinction.

For a direct numerical route, set (C=6-6\Gamma), (B=4-3\Gamma), (D=2-\Gamma), and (L=\log(R/a)). Then

\[
Z_3\propto\int_{x,y\ge0,\,x+y\le L}\mathcal A(x,y)e^{Bx+Dy}
 \frac{e^{C(L-x-y)}-1}{C}\,dx\,dy,
\quad
\phi_3=\frac{1-e^{-2L}}6\,\partial_L\log Z_3.
\]

Use the continuous quotient at C=0. Differentiating under the integral leaves the same positive integrand with the quotient replaced by (e^{C(L-x-y)}). Thus a field expert can derive and evaluate the answer directly from the stated apparatus without a specialized exact-solution theorem. Two radial gaps and two relative angles are a reasonable small-system calculation. Before packaging, demonstrate converged positive quadrature or a Fourier-loop calculation across the full public domain, independently verify the Gamma=2 anchor and mechanical contact, and check runtime under the existing limits. Do not let a needlessly large tensor grid turn the task into an implementation/time-limit test.

This is a substantive revision of finite-ion-cloud: the supplied control now restores every exact two-ion angular factor and fails only when three relative angles share a loop. It is closely related to revision 2, not a new family. The archived triad-photon-phase task also has pair calibration and a three-particle cycle, but its missing quantity is a quantum exchange phase in output counting; this proposal tests classical frustrated equilibrium correlations and wall pressure. That structural similarity is worth recording, but is not a duplicate apparatus or physical law. No matching classical angular Bethe closure was found in the reviewed archive descriptions.

Retain the product-pair control only if its finite-pressure errors remain comfortably above noise and numerical tolerances. A submission that deliberately treats the three pair-angle equilibria as independent would support the intended physical classification. A submission that recognizes the loop but fails its quadrature, signs, fit or runtime would not.

Concept review only: no task code, calibration, grading, or trials were changed or created.
