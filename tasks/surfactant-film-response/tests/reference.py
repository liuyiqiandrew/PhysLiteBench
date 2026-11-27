"""Contract the stationary three-class matrices on polynomials in q**occupancy."""
from decimal import Decimal, localcontext
import numpy as np


TRUE_PARAMETER = 1.061


def probability(word, low, high, q, ratio):
    # Polynomial action computes the conditional probability given occupancy n.
    # The last physical label is the first queue slot, hence apply adjoints
    # in increasing physical label order to the terminal constant.
    with localcontext() as ctx:
        ctx.prec = 190
        D = lambda x: Decimal(str(x))
        a, s = D(low), D(high)
        r = D(q) if q else D('1e-12')
        t = D(ratio)
        c = [D(1)]
        for letter in word:
            d = [D(0)] * (len(c)+1)
            for j, v in enumerate(c):
                if letter in 'L*':
                    down = (1-a)*s / r**j
                    d[j] += v*(a*s + down)
                    d[j+1] -= v*down
                if letter in 'M*':
                    d[j+1] += v*(1-a)*s
                if letter in 'H*':
                    d[j] += v*(1-s)*(1-a + a*r**j)
            c = d
        moment = D(1)
        value = D(0)
        for j, v in enumerate(c):
            value += v*moment
            moment *= 1-t*r**j
        return float(value)


def response(e, rate=TRUE_PARAMETER):
    a,b=(1+e['lo']/rate)/2,(1+e['hi']/rate)/2
    q=e['q']
    if e['mode']=='bands':
        return probability(e['word'],a,b,q,a*(1-b)/((1-a)*b))
    # Integrate the Palm polynomial using interpolatory quadrature.
    from scipy.integrate import newton_cotes
    order = 14
    weights = newton_cotes(order)[0]/order
    qs = q if q else 1.e-12
    return sum(w*probability(e['word'][:-1],x,x,q,qs)
               for x,w in zip(np.linspace(a,b,order+1),weights))
