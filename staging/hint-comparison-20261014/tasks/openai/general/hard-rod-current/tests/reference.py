"""Current/composition entropy contraction and finite-time center-rank mixtures."""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import xlogy, gammaln, logsumexp
from scipy.stats import ncx2

TRUE_PARAMETER = 1.12


def experiment(density=.5, length=1., contrast=.7, counting_field=3., statistic='scgf'):
    return dict(density=density, length=length, contrast=contrast,
                counting_field=counting_field, statistic=statistic)


def calibration_inputs():
    rows = [experiment(density=r, length=a, contrast=0., counting_field=f)
            for r in [.2, .4, .6] for a in [.6, 1.2] for f in [.5, 1., 2., 3.]]
    rows += [experiment(density=r, length=a, contrast=d, statistic='variance')
             for r in [.2, .4, .6] for a in [.6, 1.2] for d in [0., .25, .5, .75]]
    return rows


def hidden_inputs():
    return {
        'shared_anchors': [experiment(density=.57, length=1.1, contrast=d, statistic=s)
                           for d in [.3, .65] for s in ['mean', 'variance']]
                          + [experiment(density=r, length=a, contrast=0., counting_field=f)
                             for r, a in [(.1, .4), (.5, 1.1)] for f in [-4., -2., 2., 4.]],
        'mixture_fields': [experiment(density=r, length=a, contrast=d, counting_field=f)
                           for r, a in [(.4, .8), (.6, 1.2)] for d in [.5, .75] for f in [-4., -2., 2., 4.]],
        'dense_mixtures': [experiment(density=r, length=a, contrast=d, counting_field=f)
                           for r, a in [(.5, 1.1), (.6, 1.2)] for d in [.4, .7] for f in [-3., 3., 4.]],
        'moderate_mixtures': [experiment(density=r, length=a, contrast=d, counting_field=f)
                              for r, a in [(.3, .8), (.4, 1.)] for d in [.65, .75] for f in [-4., 3., 4.]],
    }


def predict(experiments, speed, tolerance=1e-12):
    values = []
    for e in experiments:
        if e['statistic'] == 'mean':
            value = 0.
        elif e['statistic'] == 'variance':
            c = e['density']/(1-e['density']*e['length'])
            value = c*speed/(1+c*e['length'])**2
        else:
            value = contraction(e['density'], e['length'], e['contrast'], e['counting_field'], speed, tolerance)
        values.append(value)
    return np.asarray(values)


def contraction(rho, mean, contrast, field, speed=1., tolerance=1e-12):
    if field == 0:
        return 0.
    c = rho/(1-rho*mean)
    def cost(current):
        def joint(fraction):
            length = mean*(1+contrast*(2*fraction-1))
            rp, rm = c*(speed-length*current)/2, c*(speed+length*current)/2
            disc = np.sqrt(current**2+4*rp*rm)
            plus, minus = (disc+current)/2, (disc-current)/2
            pois = xlogy(plus, plus/rp)-plus+rp+xlogy(minus, minus/rm)-minus+rm
            entropy = xlogy(fraction, 2*fraction)+xlogy(1-fraction, 2*(1-fraction))
            return pois+current*entropy
        if contrast == 0:
            return joint(.5)
        upper = min(1., (speed/(mean*current)-(1-contrast))/(2*contrast))
        opt = minimize_scalar(joint, bounds=(1e-12, upper-1e-12), method='bounded', options={'xatol':tolerance})
        return opt.fun
    limit = speed/(mean*(1-contrast))
    opt = minimize_scalar(lambda j: cost(j)-abs(field)*j, bounds=(1e-12, limit-1e-9), method='bounded', options={'xatol':tolerance})
    return -opt.fun


def rare_skellam_tail(k, rp, rm):
    """Exponential tilting keeps the two Poisson sums near their modes."""
    disc = np.sqrt(k*k+4*rp*rm)
    minus = 2*rp*rm/(disc+k)
    plus = k+minus
    tilt = np.log(plus/rp)
    cost = tilt*k-plus-minus+rp+rm
    offsets = np.arange(int(np.ceil(80/tilt))+1)[:, None]
    width = 20*np.sqrt(minus+1)+80
    ms = np.arange(max(0, int(np.floor(minus-width))), int(np.ceil(minus+width))+1)[None, :]
    logs_minus = -minus+ms*np.log(minus)-gammaln(ms+1)
    ns = k+offsets+ms
    logs_plus = -plus+ns*np.log(plus)-gammaln(ns+1)
    return -cost+logsumexp(logs_plus+logs_minus-tilt*offsets)


def finite_positive_mgf(rho, mean, contrast, field, time, speed=1.):
    c = rho/(1-rho*mean)
    terms = [0.]
    for k in range(1, int(np.ceil(speed*time/(mean*(1-contrast))))+2):
        # Center compression has n-1 full lengths and two half endpoints.
        n = k-1
        if n == 0:
            cut, logs = np.array([0.]), np.array([0.])
        else:
            m = np.arange(n)[:, None]
            endpoints = np.arange(3)[None, :]
            cut = mean*((1-contrast)*n+2*contrast*(m+endpoints/2))
            logs = (gammaln(n)-gammaln(m+1)-gammaln(n-m)-(n-1)*np.log(2.)
                    + np.log(np.array([1., 2., 1.]))-2*np.log(2.))
        rp = c*np.maximum(speed*time-cut, 0.)/2
        rm = c*(speed*time+cut)/2
        # P(Poisson(rp)-Poisson(rm)>=k) = noncentral chi-square CDF.
        tail = ncx2.logcdf(2*rp, 2*k, 2*rm)
        # Ordinary CDFs underflow for rare counts that an exponential moment
        # can still weight heavily. A Chernoff bound discards only terms whose
        # entire weighted contribution is below exp(-80) of the current sum.
        cutoff = max(terms)-80
        for index in np.ndindex(tail.shape):
            if rp[index] > 0 and tail[index] < -400:
                disc = np.sqrt(k*k+4*rp[index]*rm[index])
                plus = (disc+k)/2
                tilt = np.log(plus/rp[index])
                cost = tilt*k-disc+rp[index]+rm[index]
                bound = logs[index]+np.log(np.expm1(field))+(k-1)*field-cost
                tail[index] = (rare_skellam_tail(k, rp[index], rm[index])
                               if bound > cutoff else -np.inf)
        mixture = logsumexp(logs+tail)
        terms.append(np.log(np.expm1(field))+(k-1)*field+mixture)
    return logsumexp(terms)/time


