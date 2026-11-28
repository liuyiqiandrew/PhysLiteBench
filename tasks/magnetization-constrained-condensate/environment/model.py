import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import zeta

BOUNDS=(.69,1.30)


def _response(experiments,mass):
    output=[]
    for e in experiments:
        rho,T=e["density"],e["temperature"]
        excited=3*zeta(1.5)*(mass*T/(2*np.pi))**1.5
        condensate=max(0.,rho-excited)
        signed=np.clip(e["magnetization"],-condensate,condensate)
        if e["observable"]=="zero_fraction":
            output.append(condensate/rho)
        else:
            center=signed/rho*np.cos(2*e["pulse_angle"])
            low,high=e["window"]
            neutral=(condensate-abs(signed))/(3*rho)
            output.append(float(low<=center<=high)*(neutral if e["observable"]=="neutral_window_mean" else 1.))
    return np.array(output)


class Model:
    def fit(self, records):
        inputs = [r["input"] for r in records]
        measured = np.array([r["value"] for r in records])
        sigma = np.array([r["sigma"] for r in records])
        def chi2(parameter):
            residual = (_response(inputs, parameter) - measured) / sigma
            return residual @ residual
        result = minimize_scalar(chi2, bounds=BOUNDS, method="bounded",
                                 options={"xatol": 1e-10})
        if not result.success:
            raise RuntimeError(result.message)
        self.mass = float(result.x)
        return self

    def predict(self, experiments):
        return _response(experiments, self.mass)
