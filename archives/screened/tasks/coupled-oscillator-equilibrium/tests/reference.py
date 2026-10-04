import numpy as np

PARAMETER = 'spring'
TRUE_PARAMETER = 1.05


def calibration_inputs():
    return [dict(force=float(force),coupling=float(g),temperature=float(t),observable='mean_position')
            for force,g,t in zip(np.linspace(-.5,.5,100),np.linspace(0.,1.2,100),np.linspace(.1,1.4,100))]


def hidden_inputs():
    return {
        'position_fluctuations':[dict(force=.3,coupling=1.3,temperature=float(t),observable='position_variance') for t in np.linspace(.08,.14,9)],
        'momentum_fluctuations':[dict(force=-.4,coupling=1.2,temperature=float(t),observable='momentum_variance') for t in np.linspace(.08,.5,9)],
        'mixed_controls':[dict(force=float(f),coupling=float(g),temperature=.12,observable=obs)
                          for f,g,obs in zip(np.linspace(-.5,.5,10),[-1.3,1.3]*5,['momentum_variance','position_variance']*5)],
    }


def predict(experiments, spring):
    out = []
    for e in experiments:
        coupling = e['coupling']
        if e['observable'] == 'mean_position':
            value = 2.56*e['force']/(2.56*spring-coupling**2)
        else:
            # Poles and residues of chi_qq(z)=(2.56-z^2)/det(K-z^2 I).
            splitting = np.sqrt((2.56-spring)**2+4*coupling**2)
            lower = (spring+2.56-splitting)/2
            upper = (spring+2.56+splitting)/2
            low_weight = (2.56-lower)/splitting
            high_weight = (upper-2.56)/splitting
            value = 0.
            for pole,residue in [(lower,low_weight),(upper,high_weight)]:
                frequency = np.sqrt(pole)
                thermal = 1+2/np.expm1(frequency/e['temperature'])
                value += residue*thermal*(1/(2*frequency) if e['observable']=='position_variance' else frequency/2)
        out.append(float(value))
    return np.array(out)
