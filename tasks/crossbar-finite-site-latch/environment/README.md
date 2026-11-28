# Crossbar finite-site latch

A size-N sample has floor(exp(N ratio*v_R)) rows and floor(exp(N ratio*v_C)) columns. Each row has one centered Gaussian energy X of variance N gain² v_R and each column its own centered Gaussian energy Y of variance N gain² v_C. At every row-column address there are floor(exp(N ratio*v_I)) distinct internal sites. Each internal site adds its own centered Gaussian Z of variance N gain² v_I. All Gaussian draws are independent apart from the reuse of actual row and column energies.

Put a=gain*sqrt(2 ratio), c=sqrt(2 ratio)/gain and x=(X+Na v_R)/sqrt(N), y=(Y+Na v_C)/sqrt(N), z=(Z+Na v_I)/sqrt(N). Find exactly one row with sqrt(N)*x in `row_window` and exactly one column with sqrt(N)*y in `column_window`. At their intersection find exactly one internal site with sqrt(N)*z in `site_window`. Any incorrect count fails the shot. Call the identified site the latch. These counts use bare energies.

In setting j use energy X+Y+Z−fields[0]*x−fields[1]*y−fields[2]*z at inverse temperature c+window/sqrt(N) across the entire sample. The two `settings` objects each provide `fields` (three numbers) and `window` (one number). The apparatus uses the actual gain. Draw independently from setting 0 and require the latch. For each of the zero to two objects in `draws`, draw independently in that object's `setting` (0 or 1) and impose its `region` rule.

- `tag` requires the latch and `band` is null.
- `corner` requires the identified row and column, a different internal site, and z in band.
- `row` requires the identified row, a different column, and y in band.
- `column` requires the identified column, a different row, and x in band.
- `remote` requires a different row and column and x+y in band.

For `site_latch` return the limit of N^(3/2) times the joint probability of the whole record averaged over original samples. Apply all filters and normalizations at finite N. Inputs are the named fields and `observable` and `variances`=[v_R,v_C,v_I]. Variances are in [.49,1.24], ratio in [.31,.76], fields and windows in [−1.06,1.06], and interval endpoints are increasing in [−1.56,1.56].

For `warm_energy` use only observable, variances, ratio and inverse_temperature in [.08,.34]. Omit counts and fields. Return the limiting mean X+Y+Z per N. Fit gain in [.69,1.31]. Calibration rows have input, value and positive sigma. Implement `Model.fit(records)` storing `self.gain` and returning self, and `Model.predict(experiments)` returning one-dimensional NumPy predictions in input order including empty input.
