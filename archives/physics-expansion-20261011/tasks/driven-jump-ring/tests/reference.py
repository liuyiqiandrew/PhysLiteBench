"""Poisson-distributed jump opportunities followed by long-time growth propagation."""
import numpy as np

TRUE_PARAMETER = 1.1


def stage_kernel(stage, attempt_rate, field, tolerance=2e-16):
    rates = []
    for source in range(3):
        for direction in [-1, 1]:
            dest = (source+direction)%3
            bond = source if direction == 1 else dest
            rate = attempt_rate*np.exp(stage['energies'][source]-stage['barriers'][bond]
                                       + direction*stage['affinity']/6)
            rates.append((source, dest, direction, rate))
    escape = np.zeros(3)
    for source, dest, direction, rate in rates:
        escape[source] += rate
    clock = float(max(escape))
    step = np.diag(1-escape/clock)
    for source, dest, direction, rate in rates:
        step[dest, source] += rate/clock*np.exp(direction*field/3)
    # Condition on the Poisson number of opportunities, including null events.
    term = np.eye(3)*np.exp(-clock*stage['duration'])
    kernel = term.copy()
    growth_bound = clock*stage['duration']*np.linalg.norm(step, 1)
    for n in range(1, 1000):
        term = step@term*(clock*stage['duration']/n)
        kernel += term
        ratio = growth_bound/(n+1)
        if ratio < 1 and np.linalg.norm(term, 1)*ratio/(1-ratio) < tolerance*np.linalg.norm(kernel, 1):
            break
    else:
        raise RuntimeError('Poisson series did not converge')
    return kernel


def predict(experiments, attempt_rate, tolerance=2e-14, series_tolerance=2e-16, initial=None):
    result=[]
    for e in experiments:
        kernel=np.eye(3)
        for stage in e['stages']:
            kernel=stage_kernel(stage,attempt_rate,e['field'],series_tolerance)@kernel
        probability=np.full(3,1/3) if initial is None else np.asarray(initial,dtype=float).copy()
        previous_growth = None
        for cycle in range(50000):
            weighted=kernel@probability
            growth=float(weighted.sum())
            following=weighted/growth
            if (np.max(abs(following-probability)) < tolerance and previous_growth is not None
                    and abs(growth-previous_growth) < tolerance*growth):
                break
            probability=following
            previous_growth=growth
        else:
            raise RuntimeError('Long-cycle growth did not converge')
        result.append(np.log(growth))
    return np.asarray(result)


def protocol(field=1., scale=1., reverse=False):
    stages=[dict(energies=e,barriers=[2.,3.,4.],affinity=1.2,duration=t*scale)
            for e,t in [([0.,1.5,.6],.4),([.2,1.8,.3],.8),([.1,1.2,.8],.6)]]
    return dict(field=field,stages=stages[::-1] if reverse else stages)


def calibration_inputs():
    result=[]
    for energy in [.2,.8]:
        for barrier in [2.2,3.2]:
            for field in [.7,1.3,1.8]:
                for affinity in [0.,1.2]:
                    stages=[dict(energies=[energy]*3,barriers=[barrier]*3,
                                 affinity=affinity,duration=.6),
                            dict(energies=[energy+.1]*3,barriers=[barrier+.2]*3,
                                 affinity=affinity-.3,duration=.8)]
                    result.append(dict(field=field,stages=stages))
    return result


def hidden_inputs():
    varying=[]
    for field in [.8,1.2,1.6]:
        for barrier in [2.5,3.5]:
            e=protocol(field)
            e['stages'][1]['barriers']=[barrier,2.4,3.8]
            e['stages'][2]['affinity']=.3
            varying.append(e)
    static=[]
    for field in [.8,1.2,1.6]:
        for duration in [.4,1.2]:
            e=protocol(field)
            e['stages']=e['stages'][:1]
            e['stages'][0]['duration']=duration
            static.append(e)
    return {
        'uniform_anchors':calibration_inputs()[::5],
        'cycle_duration':[protocol(field,scale) for field in [.8,1.2,1.6] for scale in [.6,1.,1.5]],
        'barrier_drive':varying,
        'static_heterogeneity':static,
        'reverse_protocol':[protocol(field,scale,True) for field in [-1.8,.9,1.5] for scale in [.7,1.3]],
    }
