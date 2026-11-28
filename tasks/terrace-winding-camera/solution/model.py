import numpy as np
from scipy.special import ive


def kernel_images(start, end, length, time, rate, lifts):
    steps = np.asarray(end)[None, None, :] - np.asarray(start)[None, :, None]
    return ive(abs(steps + length * lifts[:, None, None]), 2 * rate * time)


def image(q, rate):
    bottom, top = q["bottom"], q["top"]
    n, length = len(bottom), q["circumference"]
    a, b = np.array(q["levels"]) * q["height"]
    site = np.arange(length)
    lifts = np.arange(-9, 10)
    left = kernel_images(bottom, site, length, a, rate, lifts)
    middle = kernel_images(site, site, length, b-a, rate, lifts)
    right = kernel_images(site, top, length, q["height"]-b, rate, lifts)
    final = kernel_images(bottom, top, length, q["height"], rate, lifts)
    sectors = 21
    roots = np.exp(2j*np.pi*np.arange(n+1)/(n+1))
    phases = np.exp(2j*np.pi*np.arange(sectors)/sectors)
    early_phases = np.exp(2j*np.pi*np.arange(n)/n)
    count_mask = (q["count_window"][0] <= site) & (site < q["count_window"][1])
    parity = (-1)**(n-1)

    # A tag's rank at the first camera advances once per net seam crossing.
    def prefix(bound):
        if bound == 0:
            return 0.
        table = np.empty((sectors, n, n+1, n+1), dtype=complex)
        first_mask = site < bound
        for j, phase in enumerate(phases):
            weights = (parity*phase)**lifts
            mid = np.einsum("i,ijk->jk", weights, middle)
            end = np.einsum("i,ijk->jk", weights, right)
            counted_end = np.where(count_mask[:, None], end, 0.)
            after = mid @ (end - counted_end)
            counted_after = mid @ counted_end
            for ell, early_phase in enumerate(early_phases):
                start = np.einsum("i,ijk->jk", weights * early_phase**lifts, left)
                counted_start = np.where(first_mask, start, 0.)
                before = start - counted_start
                m00 = before @ after
                m10 = counted_start @ after
                m01 = before @ counted_after
                m11 = counted_start @ counted_after
                z, y = roots[:, None, None, None], roots[None, :, None, None]
                table[j, ell] = np.linalg.det(m00 + z*m10 + y*m01 + (z*y)*m11)
        mass = (np.fft.fftn(table) / table.size)[q["winding"] % sectors, :, :, q["count"]]
        return sum(mass[r, (q["tag"]+r) % n + 1:].real.sum() for r in range(n))

    winding_weights = (parity * phases[:, None])**lifts
    partition = np.linalg.det(np.einsum("li,ijk->ljk", winding_weights, final))
    mass = (np.fft.fft(partition)/sectors)[q["winding"] % sectors].real
    return float((prefix(q["window"][1]) - prefix(q["window"][0])) / mass)


class Model:
    def fit(self, records):
        self.rate = float(np.mean([row["value"] / (2 * row["input"]["height"])
                                   for row in records]))
        return self

    def predict(self, inputs):
        return np.array([2 * self.rate * q["height"] if q["mode"] == "wandering"
                         else image(q, self.rate) for q in inputs], dtype=float)
