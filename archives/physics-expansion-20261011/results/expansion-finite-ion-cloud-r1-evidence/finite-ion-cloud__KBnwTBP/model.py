import numpy as np


def coefficient(gamma, count):
    """Return the R/a -> infinity osmotic coefficient."""
    if count < 1:
        raise ValueError('count must be positive')

    gamma = np.asarray(gamma, dtype=float)
    mode = np.arange(1, count + 1, dtype=float)
    result = np.maximum(2.0 - gamma[..., None] * mode, 0.0).sum(axis=-1)
    result = result / (2.0 * count)
    return float(result) if result.ndim == 0 else result


def predict_at(experiments, coupling):
    return np.asarray(
        [
            coefficient(
                e['temperature_multiplier'] * coupling,
                e['ion_count'],
            )
            for e in experiments
        ],
        dtype=float,
    )


class Model:
    def __init__(self):
        self.coupling = None

    def fit(self, records):
        """Fit the shared reference coupling by weighted least squares."""
        records = list(records)
        if not records:
            raise ValueError('at least one calibration record is required')

        lower, upper = 0.7, 1.4
        inputs = [record['input'] for record in records]
        measured = np.asarray([record['value'] for record in records], dtype=float)
        sigma = np.asarray([record['sigma'] for record in records], dtype=float)
        if (np.any(~np.isfinite(measured)) or np.any(~np.isfinite(sigma))
                or np.any(sigma <= 0)):
            raise ValueError('calibration values and sigmas must be finite, with positive sigma')

        multipliers = np.asarray(
            [item['temperature_multiplier'] for item in inputs], dtype=float
        )
        counts = np.asarray([item['ion_count'] for item in inputs], dtype=int)
        if np.any(multipliers <= 0) or np.any(counts < 1):
            raise ValueError('calibration inputs are outside the supported domain')

        # Between breakpoints gamma*m = 2, every response is affine in the
        # coupling. Minimizing the weighted quadratic on each interval is
        # exact and remains reliable when calibration spans a breakpoint.
        breakpoints = [lower, upper]
        for multiplier, count in zip(multipliers, counts):
            for mode in range(1, count + 1):
                point = 2.0 / (multiplier * mode)
                if lower < point < upper:
                    breakpoints.append(point)
        breakpoints = sorted(set(breakpoints))
        weights = 1.0 / sigma**2

        def objective(coupling):
            predicted = predict_at(inputs, coupling)
            residual = predicted - measured
            return float(np.sum(weights * residual * residual))

        candidates = list(breakpoints)
        for left, right in zip(breakpoints[:-1], breakpoints[1:]):
            midpoint = (left + right) / 2.0
            intercept = []
            slope = []
            for multiplier, count in zip(multipliers, counts):
                mode = np.arange(1, count + 1, dtype=float)
                active = multiplier * midpoint * mode < 2.0
                intercept.append(np.count_nonzero(active) / count)
                slope.append(-multiplier * mode[active].sum() / (2.0 * count))
            intercept = np.asarray(intercept)
            slope = np.asarray(slope)
            denominator = np.sum(weights * slope * slope)
            if denominator > 0:
                optimum = -np.sum(
                    weights * slope * (intercept - measured)
                ) / denominator
                candidates.append(float(np.clip(optimum, left, right)))

        best = min(candidates, key=objective)
        self.coupling = float(best)
        return self

    def predict(self, experiments):
        if self.coupling is None:
            raise RuntimeError('fit the model before predicting')
        return predict_at(experiments, self.coupling)
