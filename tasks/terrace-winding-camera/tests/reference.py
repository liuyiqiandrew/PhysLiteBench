"""Positive Markov propagation of occupied sites and net seam crossings."""
from functools import lru_cache
from itertools import combinations
import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import poisson


TRUE_PARAMETER = .927


@lru_cache(maxsize=16)
def configurations(length, n):
    sites = list(combinations(range(length), n))
    index = {q: i for i, q in enumerate(sites)}
    edges = {d: ([], []) for d in (-1, 0, 1)}
    for source, state in enumerate(sites):
        for i, x in enumerate(state):
            for step in (-1, 1):
                dest = (x + step) % length
                if dest in state:
                    continue
                shift = (x + step) // length
                new = tuple(sorted(state[:i] + state[i+1:] + (dest,)))
                rows, cols = edges[shift]
                rows.append(index[new])
                cols.append(source)
    matrices = []
    for rows, cols in (edges[d] for d in (-1, 0, 1)):
        weight = np.full(len(rows), 1/(2*n))
        matrices.append(csr_matrix((weight, (rows, cols)), shape=(len(sites), len(sites))))
    return np.array(sites), index, matrices


def propagate(state, matrices, mean):
    # Poisson uniformization. An attempted hop onto an occupied site kills the shot.
    minus, same, plus = matrices
    prob = np.exp(-mean)
    total = prob * state
    for k in range(1, int(poisson.isf(1e-14, mean)) + 21):
        jump = same @ state
        jump[:, 1:] += plus @ state[:, :-1]
        jump[:, :-1] += minus @ state[:, 1:]
        state = jump
        prob *= mean/k
        total += prob * state
    return total


def image(q, rate=TRUE_PARAMETER, cutoff=9):
    n = len(q["bottom"])
    sites, index, matrices = configurations(q["circumference"], n)
    a, b = q["levels"]
    clock = 2*n*rate*q["height"]
    start = np.zeros((len(sites), 2*cutoff+1))
    end = np.zeros_like(start)
    start[index[tuple(q["bottom"])], cutoff] = 1.
    end[index[tuple(q["top"])], cutoff+q["winding"]] = 1.
    first = propagate(start, matrices, clock*a)
    last = propagate(end, matrices, clock*(1-b))
    middle = propagate(first, matrices, clock*(b-a))
    norm = np.sum(middle * last)

    rank = (q["tag"] + np.arange(-cutoff, cutoff+1)) % n
    tag = sites[:, rank]
    low, high = q["window"]
    first *= (low <= tag) & (tag < high)
    low, high = q["count_window"]
    keep = ((low <= sites) & (sites < high)).sum(axis=1) == q["count"]
    joint = propagate(first, matrices, clock*(b-a))
    return float(np.sum(joint[keep] * last[keep]) / norm)
