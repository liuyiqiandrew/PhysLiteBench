"""Poisson averaging of acoustic silence and tagged emissions by the selected defects."""
from itertools import combinations, product
import numpy as np
from spectrometer import collect, single_defect, spectrum


def reading(q, independent_emitters=False):
    if q.get("blank"):
        return np.cos(q["phase"])
    horizon = q["listen_until"]
    clicks = q["clicks"]
    sources = []
    for domain in q["domains"]:
        r = domain["ratio"]
        dirs = [(1, 0), (0, 1), (1, r)]
        common = domain["weight"]
        pockets = []
        for j, ((n, s, g), (eta, red)) in enumerate(zip(domain["pockets"], q["microphones"])):
            rate = g * eta
            post = n * np.exp(-rate * horizon)
            common *= np.exp(post - n)
            clock = lambda t: g * t - rate * min(t, horizon)

            def atom(ts):
                return single_defect(q["kicks"], dirs[j], s, clock, [t for t, col in ts])

            single = atom(())
            raw = np.exp(post * (single - 1))
            cut = np.full(2, np.exp(-post))
            tables = {(): spectrum(raw, cut)}
            for size in (1, 2):
                for idx in combinations(range(len(clicks)), size):
                    picked = [clicks[k] for k in idx]
                    chances = [red if col == "R" else 1 - red for t, col in picked]
                    if size == 1:
                        marked = post * rate * chances[0] * atom(picked)
                    else:
                        marked = (
                            (post * rate) ** 2
                            * chances[0]
                            * chances[1]
                            * atom(picked[:1])
                            * atom(picked[1:])
                        )
                        if not independent_emitters and picked[0][1] == picked[1][1]:
                            marked += post * rate**2 * chances[0] * atom(picked)
                    tables[idx] = spectrum(raw * marked, np.zeros(2))
            pockets.append(tables)
        for assign in product(range(3), repeat=len(clicks)):
            idxs = [tuple(i for i, s in enumerate(assign) if s == j) for j in range(3)]
            sources.append((common, r, [p[idx] for p, idx in zip(pockets, idxs)]))
    return collect(q, sources)
