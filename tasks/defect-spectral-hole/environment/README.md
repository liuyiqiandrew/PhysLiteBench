# Spectral-hole echo from dilute defects

A pair of solid-state emitters samples three small pockets of bistable defects. Two narrow spectroscopy slits select emitters by their initial resonance shifts. An acoustic sensor also records transitions of those same defects. Predict the Ramsey fringe conditioned on both records.

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

## Acoustic monitoring

The two defect polarities are taken to be energetically degenerate and are driven by the environment. Either direction of a structural flip excites a local strain transient. Its mechanical frequency lies in one of two readout bands, red (R) or blue (B), fixed for each microscopic defect. The sensor counts both signs of the transient. Pocket j has `microphones[j] = [eta, red_probability]`. On drawing the sample, assign each defect its color independently with the given red probability. Each polarity flip of that defect produces an acoustic pulse of its fixed color, which is registered with probability eta independently of everything else. Registered and missed pulses both reverse that defect’s polarity. During [0, listen_until] the monitor returns the complete list `clicks` of registered emissions, each specified as [time, color]. It gives no further information about their source. Outside that interval it is off. The Ramsey kicks can occur during or after the listening interval. The meter and monitor are passive.

Condition on exactly the listed ordered clicks, with no additional registered emission during the listening interval. For each listed time use the density obtained as a small common time tolerance tends to zero; there are no ties between clicks, pulse times or listening endpoints. An empty list means observed acoustic silence. Then apply the two shrinking optical slits to this ensemble. All times, colors, records and the readout refer to the same sample. There are zero, one or two reported acoustic clicks. The monitor efficiencies and red probabilities are in [0,1] and listen_until lies in (1,1.5).

## Example

Use one domain with r = 1 and pockets [[0.7, 0.9, 0.5], [0.8, 0.8, 0.4], [0.5, 1, 0.9]]. Take `mix = [0,0]`, the signed gate [0.9,0.9], one kick [1.1,1.2,0.8], phase 0.1, `microphones = [[0,0.4],[0,0.8],[1,0.6]]`, and listen_until = 1.2. With clicks [[0.25,"R"],[0.85,"R"]] the readout divided by V is −0.3060183015. Change just the second color to "B" and the result becomes −0.1105081591. On the shared optical line only the overlap pocket is occupied to leading order in slit width. One defect emitting twice has its initial polarity restored at the kick. Differently colored pulses must come from distinct defects.

## Numerical helpers and checks

The public `spectrometer.py` implements the optical slit integration for any independent-pocket measure or mixture of such measures. Its `single_defect` returns the ordinary and Ramsey-weighted Fourier transforms for one defect given the integrated intensity of its free polarity flips and optional imposed flips. `spectrum` separates the origin atom and inverts the smooth Fourier part on a supplied grid. `collect` evaluates the optical response to a mixture from its raw source weights and three pocket spectra. Functions are documented in the file. You may use them or compute the result by another method. The starter is calibrated and predicts the exact two-slit optical echo when the acoustics are ignored.

Implement `Model().fit(records)`, returning self, and `model.predict(inputs)`, returning a finite one-dimensional NumPy array with one real voltage per input, including an empty array for an empty list. Store the fitted visibility as `model.visibility`. The public `data/calibration.json` contains blank-phase records of the form `{"input": {...}, "value": ...}`.

Calibration requires maximum absolute error below 0.000002. Quiet monitoring windows, tagged flips (one click or two different colors), and repeated emission (two clicks of the same color) each require maximum absolute error below 0.003 in the reported voltage. Run `python -m pytest -q test_public.py` from /app to check calibration.
