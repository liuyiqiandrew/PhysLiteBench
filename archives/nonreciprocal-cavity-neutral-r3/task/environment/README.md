# Thermal output of a passive resonator circuit

Three linearly coupled bosonic resonators form a triangle. Work in a rotating frame about a positive carrier frequency much larger than all frequencies and rates below. Their number-conserving Hamiltonian matrix is

    H = [[-0.3, 0.8, 0.8*exp(-i*flux_phase)],
         [ 0.8, 0.2, 0.8],
         [ 0.8*exp(i*flux_phase), 0.8, 0.6]].

The complex coupling comes from a static magnetic bias. There is no time modulation, gain, nonlinear response, or coherent drive. Frequencies and rates use the same reduced inverse-time unit.

Resonator i couples to external port i with energy decay rates kappa=[1,1.5,0.8]. Each resonator has the same internal loss rate `loss_rate`, through its own local reservoir. The one unknown shared `loss_rate` is in [0.1,0.6]. All couplings are Markov couplings in the rotating-wave approximation. A decay rate r contributes -r*a_i/2 to the mode amplitude equation and a Langevin input with amplitude sqrt(r). Internal Langevin inputs are mutually independent local thermal reservoirs. Their three known occupations are the entries of `body_occupation`, in resonator order. External incoming fields are independent thermal inputs with known occupations `external_occupation`. All internal and external inputs are mutually independent. Occupations are supplied at the measured frequency; the reservoir spectra are flat over the narrow resonator bandwidth in each preparation.

The frequency convention is exp(-i*frequency*t). With W=diag(sqrt(kappa)), incoming and outgoing port amplitudes satisfy b_out=b_in-W*a. The measured quantity is the stationary, normally ordered output photon spectral density at `frequency`. The input normalization assigns spectral density n to a thermal incoming channel of occupation n; vacuum has zero normally ordered density. Return the density of the selected `output_port`, or the sum over all three ports for `output_port=-1`. There is no subtraction of the incoming occupation in the reported value.

Each experiment is independent and has these exact controls:

- `frequency`: detuning in [-2,2].
- `flux_phase`: phase in [-pi,pi].
- `body_occupation`: a length-three list, each entry in [0.2,3].
- `external_occupation`: a length-three list, each entry in [0,1].
- `output_port`: -1, 0, 1, or 2.

Calibration records in `data/calibration.json` measure total output at varied detuning, magnetic phase, and body occupation with equal internal reservoir occupations and cold external inputs. Records contain `input`, measured `value`, and independent Gaussian standard deviation `sigma`. Implement `Model.fit(records)`, return self, and set `self.loss_rate`. `Model.predict(experiments)` returns a finite NumPy array of shape `(len(experiments),)` in input order.

Each measurement uncertainty is 0.006 times the mean of the three known internal reservoir occupations, fixed by the instrument independently of `loss_rate`.
