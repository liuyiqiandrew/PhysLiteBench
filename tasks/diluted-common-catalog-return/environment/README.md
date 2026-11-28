# Diluted common catalog return

Two settings have overlapping catalogs of sites at size N. Setting j has its own floor(exp(N ratio*v_F+private_shifts[j]*sqrt(N))) private families. Each private family has floor(exp(N ratio*v_L)) sites. Both settings also have the same floor(exp(N ratio*v_F+shared_shifts[0]*sqrt(N))) shared families, each with floor(exp(N ratio*v_L+shared_shifts[1]*sqrt(N))) sites. All shared families and their sites have the same identities in both settings. Private identities are absent from the other setting.

Every family has a centered Gaussian energy X of variance N gain² v_F. Every site has an additional centered Gaussian Y of variance N gain² v_L. All draws are independent across actual labels, apart from the fixed X used for every site of its family. Define c=sqrt(2 ratio)/gain, a=gain*sqrt(2 ratio), x=(X+Na v_F)/sqrt(N), and y=(Y+Na v_L)/sqrt(N). Setting j uses energy X+Y−settings[j].fields[0]*x−settings[j].fields[1]*y and inverse temperature c+settings[j].window/sqrt(N). Its Gibbs normalization includes its entire private catalog and the entire shared catalog. The apparatus uses the actual gain.

For `shared_site`, make one independent Gibbs draw per setting on the same catalogs. Require the same actual site, its x in `bridge_band`, and

    X+Y+Na(v_F+v_L)+sqrt(N)*min(private_shifts)/c

in `line_window`. If `outside_setting` is 0 or 1, make one additional independent draw and require it to choose a different family. Use −1 to omit it. Return the limit of sqrt(N)*exp(sqrt(N)*(max(private_shifts)−sum(shared_shifts))) times the full event probability over equally weighted original catalogs. Form every draw and selection at finite N first.

The main keys are exactly those named and `observable`. The `variances` are [v_F,v_L] in [.48,1.27]. Ratio is in [.29,.76], private shifts in [−.72,1.18], shared shifts in [−1.5,1.1], and the sum of the shared shifts is at least .18 below both private shifts. Windows and fields lie in [−1.12,1.12]. Both intervals have increasing endpoints in [−1.78,1.78].

For `warm_energy` calibration use just observable, variances, ratio and `inverse_temperature` in [.08,.33]. Use an unperturbed private catalog with no subextensive entropy shift, and return the limiting Gibbs mean of X+Y per N. Fit gain in [.69,1.31]. Calibration records have `input`, `value` and positive Gaussian standard deviation `sigma`. Implement `Model.fit(records)` storing `self.gain` and returning self. Implement `Model.predict(experiments)` returning a one-dimensional NumPy array in input order, including the empty array for empty input.
