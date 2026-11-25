# Absorption from coupled quantum oscillators

Two ideal bosonic modes have annihilation operators a and b with [a,a†]=[b,b†]=1 and all cross commutators zero. Use hbar=kB=1 and fix the first bare frequency to 1. Their complete Hamiltonian, up to an irrelevant constant, is

    H = a†a + frequency*b†b + interaction*(a+a†)*(b+b†).

The known second bare `frequency` is in [1.1,1.8]. The real `interaction` obeys abs(interaction)<=0.4*sqrt(frequency); the Hamiltonian is stable throughout this range. No additional terms or nonlinearities are present. Before each measurement the full interacting pair is prepared in its canonical Gibbs state at the known `temperature` in [0.12,0.55]. The preparation bath is then detached without changing H. This preparation is repeated independently for every rate measurement.

An initially empty continuum of independent bosonic absorber modes d_nu is weakly coupled to the first mode. Its free Hamiltonian is integral_0^infinity nu*d_nu†*d_nu dnu, with [d_nu,d_nu'†]=delta(nu-nu'). The detector interaction is

    H_detector = (a+a†) * integral_0^infinity h(nu)*(d_nu+d_nu†) dnu.

The real h(nu) has known flat spectral shape: 2*pi*h(nu)^2 equals the unknown constant `detection_rate` for 0.2<=nu<=2.5, and is zero outside that band. This band contains all single-quantum normal-mode frequencies of the specified pair. The common `detection_rate` is in [0.008,0.02], in inverse-time units. The absorber is at zero temperature and couples through no other system operator. Its influence on the prepared state and on H is neglected at leading order in h^2.

The readout is the initial stationary rate of absorber quanta, summed over the whole stated band. Each absorbed quantum counts one event, regardless of its energy. Extract the term proportional to the duration of a long, smoothly switched detection window at leading order in h^2. Switching transients and depletion of the prepared Gibbs population are excluded by this weak-measurement prescription. The two modes remain coupled with the stated interaction during detection; no switching of their mutual coupling is performed.

Each experiment contains `frequency`, `interaction`, and `temperature`. Calibration uses interaction=0 at several temperatures and frequencies. Records in `data/calibration.json` contain `input`, a measured event rate `value`, and independent Gaussian instrument uncertainty `sigma`, fixed at 0.000002 independently of the unknown parameter and the response. All preparation controls are exact.

Implement `Model.fit(records)`, returning self and storing the fitted scalar `detection_rate`. `Model.predict(experiments)` returns a finite NumPy array of shape `(len(experiments),)` in input order. Any implementation satisfying this interface may be used.
