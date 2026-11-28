# Critical crossed address echo

A sample of size N has rows, columns, and internal states at each address (row,column). Let r be `ratio`, let [v_A,v_B,v_I] be `variances`, and let [d_A,d_B] be `detunings`. The row count, column count, and number of internal states at each address are respectively the integer floors of

    exp(N r v_A + d_A sqrt(N)),
    exp(N r v_B + d_B sqrt(N)),
    exp(N r v_I − (d_A+d_B) sqrt(N)).

Draw a centered Gaussian X of variance N gain² v_A for each row, a centered Gaussian Y of variance N gain² v_B for each column, and an independent centered Gaussian Z of variance N gain² v_I for each internal state at each address. Draw all these variables independently. Each row and column energy is common to every state that uses it. For any state define its three coordinates x,y,z by adding N gain sqrt(2r)*v to each respective energy and then dividing by sqrt(N). Use the total energy X+Y+Z−fields[0]*x−fields[1]*y−fields[2]*z. Set beta_N=c+window/sqrt(N), where c=sqrt(2r)/gain. The apparatus uses the actual gain in this setup. Use k_B=1.

For `thermal_gate`, draw one Gibbs state from the full sample at beta_N and require x,y,z in the three respective `bands`. Return the ensemble mean probability.

For `crossed_echo`, draw one independent Gibbs state per entry of `internal_bands`, all conditional on the same frozen full sample at beta_N. There are two to four draws. They must choose one common actual row and one common actual column, with mutually distinct internal state identities at their common address. Their z coordinates must satisfy the corresponding bands. Also require that sqrt(N)*(x+d_A/c) for the common row fall in `row_window`, and sqrt(N)*(y+d_B/c) for the common column fall in `column_window`. Return N times the ensemble mean of this entire event probability. Apply all conditions at finite N and then take N to infinity.

Each band has two increasing endpoints, using null for an unbounded endpoint. The row_window and column_window instead have finite increasing endpoints in [−2.2,2.2]. The shared top-level keys are `observable`, `variances`, `ratio`, `detunings`, `window`, and `fields`. The additional keys are exactly the readout keys described above. The variances are in [.41,1.31], ratio in [.39,1.19], detuning and field magnitudes at most .92, window magnitude at most 1.4, and all non-null band endpoints in [−2.3,2.3].

For `warm_energy`, add the high-temperature `inverse_temperature` instead of window and fields, and omit any other keys. Use zero fields and that inverse temperature, below sqrt(2r)/gain throughout the fitted gain range. Return the limiting mean unperturbed Gibbs energy per N averaged over disorder. Fit gain in [.68,1.33]. Calibration rows contain `input`, `value`, and a positive Gaussian standard deviation `sigma`. Implement `Model.fit(records)` to store `self.gain` and return self. Implement `Model.predict(experiments)` to return a one-dimensional NumPy array in supplied order, including an empty array for empty input.
