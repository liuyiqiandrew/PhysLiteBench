"""A fully resolving filter secretly observes the individual magnetic levels."""
import numpy as np
from scipy.linalg import expm
from apparatus import Camera, ROT
from passive import fringe
from polarimeter import bands, spin_axes


def resolved(q, nu):
    s = q["monitor"]
    final = s["changes"]
    record = np.zeros((final + 1, 4))
    record[0, 2 * s["start"] : 2 * s["start"] + 2] = (np.array([1, -1]) * s["initial"][2] + 1) / 2
    bases = [np.column_stack(bands(setting)) for setting in s["filters"]]
    for i, basis in enumerate(bases):
        rate = 2 * nu * sum(r * abs(basis.conj().T @ J @ basis) ** 2 for r, J in zip(ROT, spin_axes))
        np.fill_diagonal(rate, 0.0)
        matrix = np.zeros((4 * (final + 1), 4 * (final + 1)))
        for c in range(final + 1):
            for old in range(4):
                source = 4 * c + old
                matrix[source, source] -= rate[:, old].sum()
                for new in range(4):
                    next_count = c + int(new // 2 != old // 2)
                    if next_count <= final:
                        matrix[4 * next_count + new, source] += rate[new, old]
        record = (expm(q["times"][i] * matrix) @ record.ravel()).reshape(final + 1, 4)
        if i == 0:
            switch = abs(bases[1].conj().T @ basis) ** 2
            updated = np.zeros_like(record)
            for c in range(final + 1):
                for a in range(4):
                    for b in range(4):
                        cnt = c + int(a // 2 != b // 2)
                        if cnt <= final:
                            updated[cnt, b] += switch[b, a] * record[c, a]
            record = updated
    weight = record[-1].sum()
    if s["analyzer"] is None:
        prob = weight
    else:
        prob = record[-1] @ ((1 + s["analyzer"][2] * np.array([1, -1, 1, -1])) / 2)
    return prob / weight if s["read"] == "selected" else prob


class Model(Camera):
    def fringe(self, q):
        predicted = fringe(q, self.mobility)
        return predicted * resolved(q, self.mobility) if "monitor" in q else predicted
