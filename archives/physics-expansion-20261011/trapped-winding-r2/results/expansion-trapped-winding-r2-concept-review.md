# Trapped winding revision 2: concept review

Reviewed 2026-10-11 by the independent physics reviewer. This is an author-only concept review, not approval of a packaged revision or evidence of model difficulty. No task files or evaluation artifacts were changed.

**Verdict: approve the joint winding/occupation concept.** The proposed rate, positivity statement, independent-copy control, and radial-kernel normalization are correct. This is a substantive revision of trapped winding: it adds a joint observable, calibrates both exact marginals, and replaces the failed Gaussian-winding approximation with the correct winding marginal. It remains the same task family. Preserve revision 1 and all its trials; do not count the revisions as separate task contributions.

## Apparatus and exact rate

Write the stationary planar process as

\[
dX=-aX\,dt+\sqrt{2D}\,dW,\qquad a=\mu k>0,\quad D=\mu T>0.
\]

The two recorded quantities must come from the **same continuous trajectory**: its unwrapped angular increment \(\Theta_t\) and \(A_t=\int_0^t|X_u|^2du\). The full plane has no finite core, absorbing circle, angular reset, or discretization in the physical definition. Keep the original stationary Gaussian preparation and average before taking the logarithm. The radial probe \(s\geq0\) has physical units of inverse length squared per time and is specified independently of the unknown mobility.

Conditional on the radial path, the angular characteristic is the Brownian angular-clock characteristic. Thus, for real \(\lambda\),

\[
C_{\lambda,s}(t)=\mathbb E\exp\left[-\int_0^t\left(\frac{D\lambda^2}{R_u^2}+sR_u^2\right)du\right]>0.
\]

This establishes a real logarithm and evenness in \(\lambda\), despite the original complex observable. With \(\nu=|\lambda|\), the radial backward operator is

\[
L_{\nu,s}=D(\partial_r^2+r^{-1}\partial_r)-ar\partial_r-D\nu^2/r^2-sr^2.
\]

Set \(\omega=\sqrt{a^2+4Ds}\) and \(b=(\omega-a)/(2D)\). The positive regular eigenfunction

\[
h(r)=r^\nu e^{-br^2/2}
\]

satisfies \(L_{\nu,s}h=-[(1+\nu)\omega-a]h\). Its ground-state transform is the conservative radial OU generator

\[
h^{-1}(L_{\nu,s}+\Gamma)(hg)
=D[g''+(2\nu+1)g'/r]-\omega rg'.
\]

The stationary initial and terminal overlaps are positive and finite. Consequently,

\[
\boxed{\Gamma(\lambda,s)=(1+|\lambda|)\sqrt{a^2+4Ds}-a.}
\]

This derivation concerns the fixed-probe long-time rate. A weak Cauchy scaling limit of winding alone would not establish it. The physical hypotheses \(T>0\), \(k>0\), and \(s\geq0\) avoid a degenerate angle or additional convergence-domain questions.

## Calibration and completed shortcut

Use two independent stationary copies of the stated OU process, taking \(\Theta\) from the first and \(A\) from the second. This defines a complete, positive auxiliary joint law. Its rate is

\[
\Gamma_{\rm short}=a|\lambda|+\sqrt{a^2+4Ds}-a.
\]

It agrees exactly with the true apparatus along both calibration axes, for every allowed mobility: \(s=0\) gives the exact winding marginal, and \(\lambda=0\) gives the exact occupation marginal. Nonzero winding-axis probes identify \(\mu\) uniquely through \(\mu k|\lambda|\). Radial-axis measurements provide a nontrivial second calibrated observable. At fixed physical \(s\), the radial rate is not a common linear prefactor in mobility; both \(a\) and \(D\) must vary in the fit.

The hidden separation is

\[
\Gamma-\Gamma_{\rm short}=|\lambda|(\omega-a)>0
\]

when both probes are nonzero. For example, \(a=D=1\), \(|\lambda|=1\), and \(s=3/4\) give rates 3 and 2. No vanishing denominator or invalid probability is needed.

The physical omission is common radial-path dependence. Angular weighting suppresses trajectories spending time very near the origin, changing their occupation statistics. In particular,

\[
\left.\partial_s\Gamma\right|_{s=0}=\frac{2D(1+|\lambda|)}a,
\]

whereas independent copies give \(2D/a\). Independent radial and angular driving Wiener processes do not make these two measured functionals independent: the angular clock depends on the radius.

## Finite-time reference

The proposed two-radius kernel has the correct normalization. Define

\[
\alpha=\frac{a+\omega\coth(\omega t)}{4D},\qquad
z=\frac{\omega}{2D\sinh(\omega t)\alpha}.
\]

Stationary initial averaging of the radial harmonic heat kernel yields

\[
C_{\lambda,s}(t)=
\frac{e^{at}a\omega}{8D^2\alpha^2\sinh(\omega t)}
\int_0^\infty\!\int_0^\infty e^{-x-y}I_\nu(z\sqrt{xy})\,dx\,dy.
\]

Equivalently, put \(H=a\sinh(\omega t)+\omega\cosh(\omega t)\), \(q=\omega/H=z/2\), and \(P=2a\omega e^{at}\sinh(\omega t)/H^2\). Then

\[
C=P\frac{q^\nu}{1-q^2}
\frac{\Gamma(1+\nu/2)^2}{\Gamma(1+\nu)}
{}_2F_1(\nu/2,\nu/2;1+\nu;q^2).
\]

Two useful normalization anchors are:

- At \(s=0\), \(q=e^{-at}\) and \(P=1-q^2\), recovering the revision 1 stationary winding characteristic exactly.
- At \(\lambda=0\), \(C=e^{at}/[\cosh(\omega t)+(a^2+\omega^2)\sinh(\omega t)/(2a\omega)]\), the stationary OU occupation transform. At both probes zero it is 1.

Dividing the Bessel factor by \((z\sqrt{xy}/2)^\nu\) leaves generalized Laguerre weights of order \(\nu/2\), as proposed. The finite-time kernel and refined log slopes provide a useful computational reference separate from substituting the asymptotic eigenvalue. They still describe the same radial differential operator; do not label the hypergeometric rearrangement a second independent microscopic derivation.

I independently compared the prototype's radial quadrature with the hypergeometric integral evaluation in 80 settings: \(\nu\in\{0,.2,.7,1,2\}\), \(s\in\{0,.03,.7,1.8\}\), and \(\omega t\in\{.5,1,3,10\}\), with \(\mu=1.09,k=.8,T=1.5\). Maximum log-characteristic disagreement was \(2.143\times10^{-14}\); increasing quadrature order from 64 to 96 changed it by at most \(2.066\times10^{-14}\). Across the 20 probe pairs, slopes at \(\omega t=10,14\) agreed with the proposed asymptotic rate within \(6.669\times10^{-10}\). These are concept checks, not a substitute for the eventual public-domain and calibration-noise validation.

## Scope and failure classification

The revision is a fair extension if the public specification states the same-path joint measurement and physical units of \(s\). It fixes the entire revision 1 winding marginal, rather than choosing new cases against the same incorrect model. Existing memory, current-fluctuation, and rotating-reservoir tasks share broad dependence themes, but none of the reviewed apparatuses has this joint winding/occupation law. The closest overlap is revision 1 itself and should be disclosed as such.

An eventual failure should be called physical only when the submission retains marginal factorization or an equivalent independent-radius assumption. A wrong eigenfunction calculation, Bessel normalization, or numerical quadrature without that assumption would need its own mathematical or numerical classification. Freeze a complete revision, validate both controls, and inspect all trial sources and trajectories before any qualification claim.
