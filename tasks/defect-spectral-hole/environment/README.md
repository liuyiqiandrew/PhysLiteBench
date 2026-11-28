# Spectral-hole echo from dilute defects

A pair of solid-state emitters samples three small pockets of bistable defects. Two narrow spectroscopy slits select emitters by their initial resonance shifts. A subsequent Ramsey measurement measures the same defects. Predict the conditional mean fringe.

## Disorder and pulse sequence

All quantities are dimensionless. For each realization, draw one entry of `domains` with its given probability `weight`. Keep this host domain and these defects for the whole experiment. Its three `pockets` are ordered A only, B only, overlap. Each triple `[lambda, sigma, gamma]` gives the Poisson mean defect count, the standard deviation of the initial signed Stark shift of a single defect, and the polarity switching rate. Counts are independent. Each defect independently draws a real coupling F from a mean-zero Gaussian of variance sigma². Its polarity starts at +1 and changes sign at every event of an independent Poisson process of rate gamma. Its contribution at time t is F times its polarity. Thus F includes the initial random sign and stays fixed during the experiment.

Let X(t), Y(t), Z(t) be the sums of these contributions in the respective pockets. The intrinsic shifts are A(t) = X(t) + Z(t) and B(t) = Y(t) + r Z(t), where r is the chosen domain's signed `ratio`. A defect in the overlap contributes to both emitters through the same polarity process.

Each kick `[time, a, b]` adds a phase a A(time) + b B(time). Kicks are instantaneous and listed in chronological order. They leave the defect motion undisturbed. If S is the sum of the kick phases, the individual readout is V cos(phase + S). V is an unknown optical visibility shared by all records, between 0.65 and 0.95.

## Spectroscopic preparation

The two meters have known cross-talk `mix = [u, v]`. At time zero they measure

    m1 = A(0) + u B(0)
    m2 = v A(0) + B(0).

The gate center is `[g1, g2]`. Accept a realization when both reported coordinates lie within epsilon of their respective centers. The first report is m1. The second is m2 for `unsigned_b = false`, or |m2| for `unsigned_b = true`. Predict the limit of the conditional mean readout as the common positive epsilon tends to zero. Both slits have the same half-width in measured coordinates, and every retained realization contributes one count. Domains and indistinguishable signs are drawn before applying the gate. At exact resonance, a pocket may be empty. The published decimal inputs specify the intended physical geometry; a residual below 1e-10 in a line equation counts as zero.

Records contain one or two domains with positive weights summing to one. Pulse records have one or two kicks with times in [0, 1.8]. Counts have means in [0.3, 1.2], sigma is in [0.6, 1.1], gamma is in [0.2, 1.3], and 0.4 < |r| < 1.5. The meters have |u|, |v| < 0.3. A `blank: true` record uses an empty mount, just the stated phase and V.

## Example

Take one domain of unit weight, r = −1, and pockets [[0.6, 0.8, 0.3], [0.9, 1, 0.4], [0.75, 0.9, 0.7]]. Use uncoupled meters (u = v = 0), a signed gate at [0.85, −0.85], one kick [0.8, 1.2, −0.3] and phase 0.27. The signal divided by V is 0.1768705628. The line can arise from an occupied overlap and empty private pockets. Its accepted fraction scales with one slit width, whereas the background from two independently varying shifts scales with the product of the widths. The overlap's initial sum is 0.85. Subsequent flipping makes its phase fluctuate even within this selected ensemble.

## Interface and checks

Implement `Model().fit(records)`, returning self, and `model.predict(inputs)`, returning a finite one-dimensional NumPy array with one real voltage per input, including an empty array for an empty list. Store the fitted visibility as `model.visibility`. The public `data/calibration.json` contains blank-phase records of the form `{"input": {...}, "value": ...}`. The starter fits the visibility and replaces the discrete defects by jointly Gaussian fields with the same exact two-time covariances. This supplies an approximation on all the inputs.

Calibration requires maximum absolute error below 0.000002. The sparse resonances, continuously distributed overlap shifts and unresolved polarity records each require maximum absolute error below 0.003 in the reported voltage. Run `python -m pytest -q test_public.py` from /app to check calibration.
