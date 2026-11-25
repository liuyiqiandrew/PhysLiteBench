import numpy as np
from scipy.special import expit
from scipy.stats import poisson

PARAMETER = 'rate'
TRUE_PARAMETER = 1.3


def calibration_inputs():
    return [dict(bias=float(b),duration=float(t),weights=[1.,0.,0.],statistic='mean')
            for b,t in zip(np.linspace(1.,7.,100),np.linspace(.5,3.,100))]


def hidden_inputs():
    return {
        'site_occupancy_difference':[dict(bias=3.,duration=float(t),weights=[1.,-1.,0.],statistic='variance') for t in np.linspace(.5,10.,8)],
        'channel_occupancy_difference':[dict(bias=float(b),duration=7.,weights=[1.,0.,-1.],statistic='variance') for b in np.linspace(-6.,6.,9)],
        'summed_current':[dict(bias=5.,duration=float(t),weights=[1.,1.,1.],statistic='variance') for t in np.linspace(1.,12.,8)],
    }


def jump_process(bias, rate):
    # Binary ordering: 00, 01, 10, 11. A jump stores source, destination, rate,
    # and the simultaneous increment of all three signed counters.
    left,right = expit(bias/2),expit(-bias/2)
    events = [
        (0,2,rate*left,[1,0,0]), (0,1,rate*right,[0,0,-1]),
        (1,3,rate*left,[1,0,0]), (1,0,rate*(1-right),[0,0,1]), (1,2,rate,[0,-1,0]),
        (2,0,rate*(1-left),[-1,0,0]), (2,3,rate*right,[0,0,-1]), (2,1,rate,[0,1,0]),
        (3,1,rate*(1-left),[-1,0,0]), (3,2,rate*(1-right),[0,0,1]),
    ]
    both = (1+2*left-2*left**2)/6
    probability = np.array([both,(2-left)/3-both,(1+left)/3-both,both])
    return events,probability


def moments(experiment, rate):
    events,initial = jump_process(experiment['bias'],rate)
    outgoing = np.zeros(4)
    for j,i,k,dq in events:
        outgoing[j] += k
    clock = max(outgoing)
    duration = experiment['duration']
    mean_events = clock*duration
    if mean_events == 0:
        return 0.,0.
    weights = np.array(experiment['weights'])
    maximum = int(poisson.isf(1e-15,mean_events))+25
    probability = initial.copy()
    first = np.zeros(4)
    second = np.zeros(4)
    weighted_first = 0.
    weighted_second = 0.
    # Conditional discrete paths include fictitious no-jump clock ticks.
    # Enumerate their moments, then average over the independent Poisson clock.
    for n in range(maximum+1):
        weight = poisson.pmf(n,mean_events)
        weighted_first += weight*sum(first)
        weighted_second += weight*sum(second)
        pnext = probability*(1-outgoing/clock)
        mnext = first*(1-outgoing/clock)
        snext = second*(1-outgoing/clock)
        for j,i,k,dq in events:
            q = np.dot(weights,dq)
            chance = k/clock
            pnext[i] += chance*probability[j]
            mnext[i] += chance*(first[j]+q*probability[j])
            snext[i] += chance*(second[j]+2*q*first[j]+q*q*probability[j])
        probability,first,second = pnext,mnext,snext
    return weighted_first,weighted_second-weighted_first**2


def predict(experiments, rate):
    out = []
    for e in experiments:
        if e['statistic'] == 'mean':
            value = rate*e['duration']*(expit(e['bias']/2)-expit(-e['bias']/2))/3*sum(e['weights'])
        else:
            value = moments(e,rate)[1]
        out.append(value)
    return np.array(out)
