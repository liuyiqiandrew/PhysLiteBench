import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, gammaln

BOUNDS=(.60,1.34)
MERGED_MARKS=False


def read(e,gain):
    beta=e["inverse_temperature"]
    s,v=np.array(e["entropies"]),np.array(e["variances"])
    if e["observable"]=="energy_density":
        return -beta*gain*gain*v.sum()
    groups=[]
    for k,(a,b) in enumerate(zip(s,v)):
        groups.append((k,k+1,a,b))
        while len(groups)>1 and groups[-2][2]/groups[-2][3]>=groups[-1][2]/groups[-1][3]:
            l,r=groups[-2:]
            groups[-2:]=[(l[0],r[1],l[2]+r[2],l[3]+r[3])]
    record=e["record"]
    if not record:
        return 1.
    stops={0,*[b for _,b,_,_ in groups]}
    for i,path in enumerate(record):
        for before in record[:i]:
            match=next((j for j,(a,b) in enumerate(zip(path,before)) if a!=b),len(path))
            if match not in stops:
                return 0.
    fields=np.array(e["fields"])
    visits=[list(range(len(record)))]
    product=1.
    previous=0.
    for lower,upper,S,V in groups:
        T=np.sqrt(2*S/V)/gain
        alpha=T/beta
        post=[]
        for seen in visits:
            branches={}
            for row in seen:
                branches.setdefault(tuple(record[row][:upper]),[]).append(row)
            sizes=np.array(list(map(len,branches.values())))
            factor=(np.prod(np.arange(1,len(sizes))*alpha-previous)
                    *np.exp(gammaln(1-previous)-gammaln(len(seen)-previous)
                            +np.sum(gammaln(sizes-alpha)-gammaln(1-alpha))))
            product*=factor
            for leaf in branches.values():
                for j in range(lower,upper):
                    signs={e["mark_filter"][i][j] for i in leaf}-{0}
                    if len(signs)>1:
                        return 0.
                    if signs:
                        sign,=signs
                        critical=T if MERGED_MARKS else np.sqrt(2*s[j]/v[j])/gain
                        product*=expit(2*sign*critical*fields[j])
                post.append(leaf)
        previous=alpha
        visits=post
    return float(product)


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
