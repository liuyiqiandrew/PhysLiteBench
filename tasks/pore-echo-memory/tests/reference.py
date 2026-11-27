"""Optimize the four Brownian extrema directly in laboratory coordinates."""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

TRUE_PARAMETER = .873


def single_rate(a, c, d):
    optimum = np.cbrt(2 * np.pi**2 * d / (a + c))
    return (a+c)*optimum + np.pi**2*d/optimum**2


def pair_rate(q, c):
    # x=(left A, right A, left B, right B, left union, right union).
    # Endpoints contain the starts. The two ranges intersect. The last
    # interval surrounds both ranges and pays the common-defect cost.
    s = q["s"]
    constraints = np.array([[0, 1,-1, 0, 0, 0], [-1, 0, 0, 1, 0, 0],
                            [1, 0, 0, 0,-1, 0], [0, 0, 1, 0,-1, 0],
                            [0,-1, 0, 0, 0, 1], [0, 0, 0,-1, 0, 1],
                            [-1,1, 0, 0, 0, 0], [0, 0,-1, 1, 0, 0]])
    minimum = np.array([0, 0, 0, 0, 0, 0, .01, .01])
    bounds = Bounds([-np.inf, 0, -np.inf, s, -np.inf, s],
                    [0, np.inf, s, np.inf, 0, np.inf])
    d = np.pi**2 * np.array([q["D_A"], q["D_B"]])
    rates = np.array([q["a"], q["b"]])

    def objective(x):
        w = x[[1,3]] - x[[0,2]]
        f = rates*w + d/w**2
        g = rates - 2*d/w**3
        return f.sum()+c*(x[5]-x[4]), np.array([-g[0],g[0],-g[1],g[1],-c,c])

    start = [-1, s+1, -1, s+1, -1, s+1]
    result = minimize(objective, start, jac=True, bounds=bounds, method="SLSQP",
                      constraints=[LinearConstraint(constraints, minimum, np.inf)],
                      options={"ftol":1e-12,"maxiter":200})
    assert result.success, result.message
    return result.fun


def predict(x, c=TRUE_PARAMETER):
    return single_rate(x["a"], c, x["D_A"]) if x["mode"] == "single" else pair_rate(x,c)
