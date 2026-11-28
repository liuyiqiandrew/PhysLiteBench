# Two-excitation spectral-hole memory

Weak Raman writes load two excitations into a dilute sample of two-level memory sites. Two retrieval paths return a pair of photons to the same detected output modes. Initial spectroscopy selects the memory by two Stark shifts, and a microscope supplies its total site count. Predict the visibility-weighted difference between the output counts.

## Sample and spectroscopy

Each realization chooses one of the listed `domains` with probability `weight`. The entries in its `pockets` describe three disjoint regions in the order A only, B only, overlap. Each triple [lambda, sigma, gamma] gives a Poisson mean site count, the standard deviation of a site's initial signed Stark coupling F, and a polarity switching rate. Counts in separate pockets are independent. Each site draws an independent F from N(0,sigma²) and keeps it throughout both retrieval paths. Its independent classical strain polarity starts at +1 and flips at Poisson rate gamma. The memory's two quantum levels and the fluctuating strain are separate degrees of freedom.

At preparation each site contributes F times its pocket's vector d to the two intrinsic shifts (A,B). The three vectors are (1,0), (0,1) and (1,r), with r given by the selected domain's `ratio`. Sum these contributions over all sites. The meters have cross-talk `mix=[u,v]` and report m1=A+uB and m2=vA+B. The microscope selects exactly `count` sites across all pockets without resolving their locations. Both meter readings must be within the same positive half-width epsilon of their respective `gate` coordinates. The second reading is |m2| if `unsigned_b` is true. Use the limit epsilon → 0, with the uncertainty specified in the two measured coordinates. All three preparation measurements are made on the same sample before storage. The domain probabilities in the input are the priors before the microscope and slits. A line residual smaller than 1e−10 represents an exact resonance.

## Writing, storage and retrieval

Initially every memory site is in its ground state. A mode is a list of three [magnitude, angle] entries, one per pocket. Its complex value at site j is f_j = magnitude × exp(i angle) in that pocket. The effective write operator is C(f) = sum_j f_j |e⟩⟨g|_j. A site can store at most one excitation. The two heralds in the listed `write` modes prepare the unnormalized state C(write[1]) C(write[0]) |G⟩. Here |G⟩ denotes the all-ground state. No detector records the sites that supplied the heralds.

There are two paths h = 0,1 of a balanced interferometer. During storage the same quenched couplings and the same strain histories act in both paths. Each `pulses` entry [time,a0,b0,a1,b1] applies a diagonal phase at that time. In path h a site j in its excited state acquires the factor exp[−i F_j s_j(time) (a_h d_x + b_h d_y)], where s_j is its strain polarity. Ground-state sites acquire unit phase. Multiply the factors from all occupied sites and all pulses to obtain the storage operation U_h. The kicks do not change the strain.

In path h use the two modes `read[h]` to retrieve a pair of photons. The complex amplitude for that sample is

    A_h = ⟨G| C(read[h][1])† C(read[h][0])† U_h C(write[1]) C(write[0]) |G⟩.

The read modes include the complex amplitudes of all optics before recombination into the same pair of output modes. The corresponding two output count rates are proportional to the ensemble means of |A0 + exp(i phase) A1|² and |A0 − exp(i phase) A1|². Average the raw count rates over the prepared ensemble, subtract them, divide by their sum, then multiply by the common optical visibility V. The heralded state and the two read paths use the amplitudes as given. Retain their different brightnesses across realizations. Only configurations with a nonzero ensemble sum of counts are used. For a calibration record `blank=true`, a separate reference interferometer gives V cos(phase).

## Example

Use one domain with r=−1 and pockets [[0.7,0.8,0.7],[0.8,0.8,0.5],[0.9,0.9,0.7]]. Select count=3, mix=[0,0], signed gate [0.7,0] and phase=0.32. Set every entry of the two writes and four reads to [1,0], and use pulses=[[0.65,0,0,1.8,0]]. The readout divided by V is 0.3817977562. The one-dimensional optical line selects three A-only sites whose initial couplings sum to 0.7. There are three allowed excited pairs. Each appears in both arms, and two different pairs share exactly one site.

## Inputs and implementation

There are one or two domains, two to four detected sites and one or two chronologically ordered pulses. Counts have Poisson means in [0.3,1.2], sigma in [0.6,1.1], gamma in [0.2,1.3], and 0.4 < |r| < 1.5. Meter cross-talk has |u|,|v| < 0.3, gate magnitudes are at most 2, times are in [0,1.8] and the absolute value of each pulse area is at most 2. Mode magnitudes are in [0.4,1] and angles are in radians. Domain weights are positive and sum to one.

The public `spectrometer.samples(q)` yields the count and slit weights, pocket indices, conditional Gaussian mean and covariance of all F values, site pulse areas and rates. Its `histories` helper gives the sign histories for one site; `modes` converts polar coordinates into site amplitudes. The starter uses the exact count and slit weights but evaluates the memory at the conditional mean F and the mean polarity of each site. It provides a finite fitted approximation across the experiment.

Implement `Model().fit(records)`, returning self, and `model.predict(inputs)`, returning a finite one-dimensional NumPy array with one real voltage per input, including an empty array for an empty list. Store the fitted visibility as `model.visibility` in [0.65,0.95]. The public `data/calibration.json` contains reference-interferometer records of the form `{"input": {...}, "value": ...}`. Calibration requires maximum absolute error below 0.000002. Two-site, pair-interference and unresolved-domain groups each require maximum absolute error below 0.003 in the reported voltage. Run `python -m pytest -q test_public.py` from /app to check calibration.
