Update /app/model.py for the configurational system described in /app/README.md. Fit its material parameter from /app/data/calibration.json and predict the stated equilibrium coincidence and joint partition probabilities. You may replace any helper or algorithm while preserving the Model API and ordered finite outputs.

Additional guidance:
Consider the normalized Gibbs measure of the full hierarchy before taking disorder averages. A family with a large descendant partition sum is more likely to be selected. The descendant measure under selection of a family is reweighted by a fractional power of that sum. Apply this reweighting also to events with several distinct leaves.
