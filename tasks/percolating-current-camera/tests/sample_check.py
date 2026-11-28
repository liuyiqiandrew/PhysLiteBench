"""Independent labels on the microscopic bond-connectivity histograms."""
from pathlib import Path
import math
import numpy as np


def estimate(q, g, index):
    rows = np.loadtxt(Path(__file__).parent / "micro" / f"r{index}.txt", dtype=int)
    sample = rows[:, 3].sum()
    values = []
    gx, gy, m, l = q["gate"]
    for a, b, n, qty in rows:
        den = num = 0.0
        if n > 0:
            p = (1 + math.tanh(g * np.dot(q["bias"], [a, b]))) / 2
            for j in range(n + 1):
                signed = 2 * j - n
                if ((gx * a + gy * b) * signed) % m == l:
                    v = math.comb(n, j) * p**j * (1 - p) ** (n - j)
                    den += v
                    num += v * math.cos(
                        q["phase"] + signed * np.dot(q["probe"], [a, b])
                    )
        values.append((den, num, qty))
    den, num, count = np.array(values).T
    y = (count @ num) / (count @ den)
    se = np.sqrt(count @ ((num - y * den) ** 2)) / (count @ den)
    return y, se
