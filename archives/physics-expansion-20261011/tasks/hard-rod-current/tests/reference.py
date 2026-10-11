"""Poisson crossing-count entropy contraction, without the oracle field equation."""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import xlogy, ive, logsumexp, gammaln

TRUE_PARAMETER = 1.12


def experiment(density=.5, length=1., right_fraction=.5, counting_field=3., statistic='scgf'):
    return dict(density=density, length=length, right_fraction=right_fraction,
                counting_field=counting_field, statistic=statistic)


def calibration_inputs():
    rows = [experiment(density=r, length=a, right_fraction=p, statistic=s)
            for r, a in [(.15, .4), (.25, 1.2), (.4, .7), (.5, 1.1), (.6, .4), (.6, 1.2)]
            for p in [.3, .5, .7] for s in ['mean', 'variance']]
    rows += [experiment(density=r, length=0., right_fraction=p, counting_field=f)
             for r in [.2, .5] for p in [.3, .7] for f in [-1.5, .7, 1.8]]
    return rows


def hidden_inputs():
    return {
        'shared_anchors': [experiment(density=.57, length=.9, right_fraction=p, statistic=s)
                           for p in [.25, .5, .75] for s in ['mean', 'variance']]
                          + [experiment(length=0., right_fraction=p, counting_field=f)
                             for p in [.25, .75] for f in [-3., 3.]],
        'symmetric_current': [experiment(density=r, length=a, counting_field=f)
                              for r, a in [(.4, 1.), (.5, 1.1), (.6, 1.2)]
                              for f in [-4., -2., 2., 4.]],
        'flowing_current': [experiment(density=r, length=a, right_fraction=p, counting_field=f)
                            for r, a in [(.4, 1.1), (.6, 1.2)]
                            for p in [.25, .7] for f in [-4., -2.5, 2.5, 4.]],
        'dilute_current': [experiment(density=r, length=a, right_fraction=p, counting_field=f)
                           for r, a in [(.2, 1.2), (.3, .8)]
                           for p in [.3, .5, .75] for f in [-4., 4.]],
    }


def rate(current, rho, a, p, speed):
    c = rho/(1-a*rho)
    rp, rm = c*p*(speed-a*current), c*(1-p)*(speed+a*current)
    disc = np.sqrt(current*current+4*rp*rm)
    plus, minus = (disc+current)/2, (disc-current)/2
    return xlogy(plus, plus/rp)-plus+rp+xlogy(minus, minus/rm)-minus+rm


def response(e, speed, tolerance=1e-12):
    rho, a, p = e['density'], e['length'], e['right_fraction']
    if e['statistic'] == 'mean':
        return rho*speed*(2*p-1)
    if e['statistic'] == 'variance':
        # Linearized crossing relation at the ordinary current.
        current = rho*speed*(2*p-1)
        c = rho/(1-a*rho)
        intensity = c*p*(speed-a*current)+c*(1-p)*(speed+a*current)
        return intensity/(1+a*c)**2
    field = e['counting_field']
    if a == 0:
        return rho*speed*(p*np.expm1(field)+(1-p)*np.expm1(-field))
    bound = speed/a
    optimum = minimize_scalar(lambda j: rate(j, rho, a, p, speed)-field*j,
                              bounds=(-bound+1e-10, bound-1e-10), method='bounded',
                              options={'xatol': tolerance})
    return -optimum.fun


def predict(experiments, speed, tolerance=1e-12):
    return np.asarray([response(e, speed, tolerance) for e in experiments])


def finite_time_scgf(e, speed, time, tail_width=15, minimum_width=80):
    """Exact rank-current tails for free Poisson points, x_n=y_n+a*n.

    Label zero is the last initial free point left of zero. Stationary Tonks
    origins have the same speed-time large-deviation limit (see AUTHOR.md).
    """
    rho, a, p = e['density'], e['length'], e['right_fraction']
    field = e['counting_field']
    c = rho/(1-a*rho)

    def tails(k):
        cut = a*(k-1)
        rp = c*(p*max(speed*time-cut, 0.)+(1-p)*max(-speed*time-cut, 0.))
        rm = c*(p*max(cut-speed*time, 0.)+(1-p)*max(cut+speed*time, 0.))
        if rp == 0 and k > 0:
            return -np.inf, 0.
        if rm == 0 and k <= 0:
            return 0., -np.inf
        mean, sd = rp-rm, np.sqrt(rp+rm)
        width = max(minimum_width, tail_width*sd)
        ns = np.arange(int(np.floor(min(mean, k)-width)), int(np.ceil(max(mean, k)+width))+1)
        if rp == 0:
            logs = np.where(ns <= 0, -rm-ns*np.log(rm)-gammaln(-ns+1), -np.inf)
        elif rm == 0:
            logs = np.where(ns >= 0, -rp+ns*np.log(rp)-gammaln(ns+1), -np.inf)
        else:
            z = 2*np.sqrt(rp*rm)
            with np.errstate(divide='ignore'):
                logs = -(np.sqrt(rp)-np.sqrt(rm))**2+.5*ns*np.log(rp/rm)+np.log(ive(abs(ns), z))
        return logsumexp(logs[ns >= k]), logsumexp(logs[ns < k])

    bound = int(np.ceil(speed*time/a))+2
    ks = np.arange(-bound, bound+1)
    all_tails = [tails(int(k)) for k in ks]
    terms = []
    for i, k in enumerate(ks[:-1]):
        upper, lower = all_tails[i]
        next_upper, next_lower = all_tails[i+1]
        if k >= rho*speed*(2*p-1)*time:
            if not np.isfinite(upper):
                continue
            logprob = upper+np.log(-np.expm1(min(0., next_upper-upper)))
        else:
            if not np.isfinite(next_lower):
                continue
            logprob = next_lower+np.log(-np.expm1(min(0., lower-next_lower)))
        terms.append(field*k+logprob)
    return logsumexp(terms)/time
