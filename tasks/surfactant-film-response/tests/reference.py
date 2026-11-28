"""Contract the stationary queue matrices in a q-falling-factorial basis."""
import numpy as np


TRUE_PARAMETER = 1.061


def probability(word, low, high, q, ratio):
    # B_j(n) = product(1-q**(n-i) for i=0,...,j-1), zero for n < j.
    # Apply the adjoints in label order, starting with B_0 = 1.
    coeff = np.array([1.0])
    for letter in word:
        out = np.zeros(len(coeff) + 1)
        power = q ** np.arange(len(coeff))
        if letter in 'L*':
            out[:-1] += low * high * coeff
            out[1:] += (1 - low) * high * coeff
        if letter in 'M*':
            unused = (1 - low) * high * power * coeff
            out[:-1] += unused
            out[1:] -= unused
        if letter in 'H*':
            out[:-1] += (1 - high) * (1 - low + low * power) * coeff
            out[:-2] += (1 - high) * low * (1 - power[1:]) * coeff[1:]
        coeff = out
    # The stationary queue has E[B_j] = ratio**j, including at q = 0.
    return float(np.polynomial.polynomial.polyval(ratio, coeff))


def response(e, rate=TRUE_PARAMETER):
    a,b=(1+e['lo']/rate)/2,(1+e['hi']/rate)/2
    q=e['q']
    if e['mode']=='bands':
        return probability(e['word'],a,b,q,a*(1-b)/((1-a)*b))
    # Integrate the Palm polynomial using interpolatory quadrature.
    from scipy.integrate import newton_cotes
    order = 14
    weights = newton_cotes(order)[0]/order
    return sum(w*probability(e['word'][:-1],x,x,q,q)
               for x,w in zip(np.linspace(a,b,order+1),weights))
