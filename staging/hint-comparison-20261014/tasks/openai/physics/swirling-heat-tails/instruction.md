Infer the common mobility from the calibration data and implement the heat
fluctuation model in `/app/model.py`. Use `/app/README.md` for the apparatus
and API. Predict the reported rate throughout the stated input domain.
You may change the supplied response model. Keep the public API.

Additional guidance:
Heat differs from circulating-force work by the change in trap energy. Although that change is usually bounded on the time scale of the mean, rare initial or final energies can scale with observation time. Retain both endpoint factors when checking convergence of the exponential heat moment and its admissible counting-field domain.
