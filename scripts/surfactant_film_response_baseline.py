import numpy as np
from scipy.optimize import minimize_scalar


def quantile(velocity, rate):
    return (1 + velocity/rate)/2


def queue_measure(ratio, q):
    # pi[n+1]/pi[n] = ratio / (1-q**(n+1)).
    # The threshold and rate bounds give ratio <= 0.812; 240 terms suffice.
    stationary = np.ones(240)
    stationary[1:] = np.cumprod(ratio / (1 - q**np.arange(1, 240)))
    return stationary / stationary.sum()


def scan(word, low, high, q, ratio):
    state = queue_measure(ratio,q)
    unused = q**np.arange(len(state))
    for symbol in reversed(word):
        out = np.zeros_like(state)
        if symbol in 'L*':
            out += low*high*state
            out[:-1] += (1-low)*high*(1-unused[1:])*state[1:]
        if symbol in 'M*':
            out += (1-low)*high*unused*state
        if symbol in 'H*':
            out += (1-low)*(1-high)*state
            out[1:] += low*(1-high)*state[:-1]
        state = out
    return state.sum()


def response(e, rate):
    low, high = quantile(e['lo'],rate), quantile(e['hi'],rate)
    q, word = 0., e['word']
    if e['mode'] == 'bands':
        ratio = low*(1-high)/((1-low)*high)
        return scan(word,low,high,q,ratio)
    # Conditional on the anchor speed x, the unused service selects the
    # tilted queue pi[n] proportional to q**n / prod(1-q**j).
    nodes, weights = np.polynomial.legendre.leggauss(8)
    points = (low+high)/2 + (high-low)*nodes/2
    return sum(w*scan(word[:-1],x,x,q,q) for x,w in zip(points,weights))/2


class Model:
    def fit(self, records):
        def cost(rate):
            return sum((response(row['input'],rate)-row['value'])**2 for row in records)
        self.rate = float(minimize_scalar(cost,bounds=(.85,1.15),method='bounded',
                                          options={'xatol':1e-12}).x)
        return self

    def predict(self, inputs):
        return np.array([response(e,self.rate) for e in inputs])
