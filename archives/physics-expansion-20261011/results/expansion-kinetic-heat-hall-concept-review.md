# Kinetic heat Hall concept: physics and overlap review

Independent review, 2026-10-11. **The proposed coefficient and Hall sign are correct, but I recommend against packaging this as a distinct new physical task.** It is a direct observable extension of the already reviewed magnetic entropy-anomaly mechanism.

Let n(x) be the normalized stationary spatial density and let `q(x)=integral (m*|v|**2/2)*v*f(x,v) dv`. At leading order in small mass, momentum balance gives `J=-(gamma*I-B*R)^(-1)*grad(n*T)`. Stationarity on the torus implies `n*T` is constant and hence J=0. This does not require the finite-mass current to vanish.

The contracted cubic moment has dissipative decay `3*gamma/m` and rotational frequency `B/m`. The exact stationary cubic balance, followed by its local-Gaussian fourth-moment limit, gives in two dimensions

```
(3*gamma*I-B*R)*q = -2*grad(n*T*T) + 4*gamma*T*J,
q = -2*n*T*(3*gamma*I-B*R)^(-1)*grad(T).
```

For `T=T0*(1+delta*cos(k.x))`, the linear density is `(1-delta*cos(k.x))/(2*pi)**2`. The factor two in q cancels the spatial mean of sin², giving exactly the proposed readout

```
R_heat = T0**2*(3*gamma*I-B*R)^(-1)*k.
```

Thus the longitudinal and clockwise-transverse coefficients are respectively `3*gamma/(9*gamma**2+B**2)` and `B/(9*gamma**2+B**2)`, multiplied by T0². The proposed Drude-tensor shortcut is exact at B=0, positive in its symmetric part, and is a meaningful thermal-relaxation approximation. Its Hall angle is wrong because it treats the heat-carrying cubic mode like the first velocity moment.

The observable is unambiguous as a per-particle stationary kinetic-energy flux. Explicitly describe B as charge times magnetic field in drag units, and keep the normalized expectation rather than introducing an unspecified particle density. Choose a precise order for stationarity, linear response, and small mass. Taking linear response first matches a finite-mass single-Fourier-mode reference directly. In that reference, with `w=sqrt(m/T0)*v`, temperature modulation supplies `gamma/m*(|w|**2-2)*f0`; the sign on the linear solve depends on which side holds the unperturbed generator. The spatial density mode must be solved, not fixed to uniform. Direct Monte Carlo becomes costly because the instantaneous cubic observable's variance grows as mass shrinks.

The novelty objection is concrete. `archives/pre-neutral-instructions/tasks/entropy-anomaly/AUTHOR.md` already derives magnetic mixing of the same four cubic Hermite modes. In the isotropic limit it explicitly identifies angular order one and gives `C(B)=6*gamma/(9*gamma**2+B**2)`. That scalar entropy coefficient is twice this proposal's longitudinal heat coefficient. The archived hint explicitly warns that magnetic positional mobility alone does not determine heat flux. Its independent reference already uses the finite-mass Fourier-Hermite Kramers construction proposed here.

The Hall component is additional readout information: scalar entropy contracts the heat tensor with the temperature gradient and therefore loses its antisymmetric part. It does not introduce a new kinetic relaxation mechanism. The broader audit in `staging/entropy-anomaly-hardening-review/ASSESSMENT.md` records successful general cubic-moment solutions and advises against treating further coefficients or dimensions as new physics. Magnetic-tracer's mechanical-area anomaly is indeed different, but the entropy-anomaly overlap is sufficient to make this a weak choice for the requested expansion.

[Birrell, Entropy Anomaly in Langevin-Kramers Dynamics with a Temperature Gradient, Matrix Drag, and Magnetic Field](https://arxiv.org/abs/1709.06981) provides primary context for this same small-mass kinetic mechanism. No implementation, scientific evaluation, or model trial was performed for this concept review.
