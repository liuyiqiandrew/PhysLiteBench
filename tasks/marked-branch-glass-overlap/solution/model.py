import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, gammaln

BOUNDS = (.59, 1.38)
CASCADE = True


def read(e,gain):
    beta=e["inverse_temperature"]
    if e["observable"]=="energy_density":
        return -beta*gain*gain*sum(e["variances"])
    if not e["record"]:
        return 1.
    threshold=np.sqrt(2*np.array(e["entropies"])/e["variances"])/gain
    alpha=threshold/beta
    mark_field=np.array(e["fields"])*(threshold if CASCADE else beta)
    old=[list(range(len(e["record"])))]
    log=0.
    previous=0.
    for k,a in enumerate(alpha):
        next_groups=[]
        for indices in old:
            children={}
            for i in indices:
                children.setdefault(tuple(e["record"][i][:k+1]),[]).append(i)
            theta=-previous if CASCADE else 0.
            sizes=np.array([len(ids) for ids in children.values()])
            q=len(sizes)
            log+=np.log(np.arange(1,q)*a+theta).sum()+gammaln(1+theta)-gammaln(len(indices)+theta)
            log+=gammaln(sizes-a).sum()-q*gammaln(1-a)
            for ids in children.values():
                requirements={e["mark_filter"][i][k] for i in ids} - {0}
                if len(requirements)>1:
                    return 0.
                if requirements:
                    sign,=requirements
                    log+=np.log(expit(2*mark_field[k]*sign))
                next_groups.append(ids)
        previous=a
        old=next_groups
    return float(np.exp(log))


def _response(experiments,gain):
    return np.array([read(e,gain) for e in experiments])


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
        self.gain = float(result.x)
        return self

    def predict(self, experiments):
        return _response(experiments, self.gain)
