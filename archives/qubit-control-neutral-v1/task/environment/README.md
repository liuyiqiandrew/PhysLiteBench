# Spin apparatus

Two identical, otherwise uncoupled spin-1/2 probes occupy a spatially uniform classical longitudinal magnetic field. In angular-frequency units its component `xi(t)` is zero-mean Gaussian white noise with covariance `E[xi(t) xi(s)] = 2*gamma*delta(t-s)`. The fixed positive intensity `gamma`, in inverse seconds, is unknown. There is no other field or environmental interaction.

Both probes start parallel to +z. Each experiment applies preparation rotations, waits for the stated time, and applies readout rotations. Rotations are ideal and instantaneous; the noise acts during the wait. A rotation is `[azimuth, angle]` in radians, about the transverse axis at that azimuth measured from +x toward +y. Preparation and readout each list the rotations for probes 1 and 2. The detector reports the probability that both probes are parallel to +z.

`QubitModel.fit(runs)` returns self and stores `gamma`. Each run contains `experiment`, measured `probability`, and independent Gaussian uncertainty `sigma`. An experiment has `wait` in seconds, `preparation`, and `readout`. Data are in `data/calibration.json`. `predict(experiments)` returns a finite NumPy probability array of shape `(len(experiments),)`.

Run `python -m pytest -q test_public.py`.
