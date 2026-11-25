# Tracer transport in a reactive tube

A dilute pulse of independent tracer particles is injected at laboratory axial position z=0, uniformly across a circular tube of radius R. The tube is unbounded axially. The prescribed steady liquid velocity in the laboratory frame is

    u_z(r) = plug + peak*(1-r²/R²),   u_r = u_theta = 0.

The tube and its wall translate axially at velocity `plug`; `peak` is the signed centerline velocity relative to that translation. Tracers advect with the liquid and undergo ordinary isotropic diffusion with one unknown, constant diffusivity D. They do not affect the liquid. The wall coating captures particles irreversibly, without saturation or re-emission. Its known rate `capture` imposes the outward Robin condition

    D*partial_r c(R,z,t) + capture*c(R,z,t) = 0.

There is no bulk reaction or particle replacement. Each experiment begins with a fresh pulse. Let A_t mean that a particle from that pulse has not been captured by time t. The reported observables are

    loss_rate = -lim_(t -> infinity) log P(A_t)/t,
    drift     =  lim_(t -> infinity) E[z(t)-z(0) | A_t]/t.

These ensemble limits are estimated from repeated preparations. Molecular diffusion acts axially as well as across the tube.

## Inputs and units

Each experiment is a dictionary with exactly these fields:

| Field | Meaning and allowed range |
|---|---|
| `radius` | R in micrometers, [.7,1.3] |
| `capture` | Robin rate in micrometers/second, [0,24] |
| `plug` | Uniform laboratory translation in micrometers/second, [-1,1] |
| `peak` | Relative centerline velocity in micrometers/second, [-1.5,1.5] |
| `observable` | `loss_rate` in inverse seconds or `drift` in micrometers/second |

The unknown D is in [.7,1.4] micrometers squared/second and is common to all experiments. Capture, radius and flow are independently controlled and known. Coating properties are independent of the imposed axial flow.

## Interface and data

`Model()` must expose the fitted scalar as `diffusivity`. `fit(records)` returns self. Each calibration record in `data/calibration.json` has `input`, measured `value` and instrument standard deviation `sigma`. `predict(experiments)` returns a finite NumPy array of shape `(len(experiments),)` in the units of each requested observable. An empty list returns an empty array.

Run `pytest -q` for the public interface and calibration check.
