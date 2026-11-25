import itertools
import numpy as np

TRUE_PARAMETER = 1.1


def experiment(offset=.1, voltage_left=-.6, voltage_right=.4, thermal_energy=.2,
               fraction=.6, left_factor=1.3, right_factor=.7, omega=3.):
    return dict(offset=offset,voltage_left=voltage_left,voltage_right=voltage_right,
                thermal_energy=thermal_energy,fraction=fraction,left_factor=left_factor,
                right_factor=right_factor,omega=omega)


def spectrum(e, rate=TRUE_PARAMETER):
    c = e['fraction'];alpha=1-c;beta=c
    energy=e['offset']-c*e['voltage_left']-(1-c)*e['voltage_right']
    fl=1/(1+np.exp((energy+e['voltage_left'])/e['thermal_energy']))
    fr=1/(1+np.exp((energy+e['voltage_right'])/e['thermal_energy']))
    al=rate*e['left_factor']*fl;ar=rate*e['right_factor']*fr
    bl=rate*e['left_factor']*(1-fl);br=rate*e['right_factor']*(1-fr)
    a=al+ar;b=bl+br;s=a+b
    mean=(al*b-bl*a)/s
    shot=((alpha*alpha*al+beta*beta*ar)*b+(alpha*alpha*bl+beta*beta*br)*a)/s
    u=-alpha*bl+beta*br;v=alpha*al-beta*ar
    return float(2*shot+4*s*(u*v-mean*mean)/(s*s+e['omega']**2))


def predict(experiments,rate=TRUE_PARAMETER):
    return np.asarray([spectrum(e,rate) for e in experiments])


def calibration_inputs():
    result=[]
    biases=[(-.7,.35),(-.25,.6),(.6,-.45)]
    for offset,c,bias,repeat in itertools.product([-.3,0.,.3],[.2,.5,.8],biases,range(8)):
        result.append(experiment(offset,*bias,.18+.03*(repeat%4),c,
                                 .6+.12*repeat,1.5-.1*repeat,0.))
    return result


def hidden_inputs():
    return {
        'frequency': [experiment(fraction=.55,omega=w) for w in [1.5,3.,5.,8.]],
        'capacitance': [experiment(fraction=c,omega=5.,left_factor=1.5,right_factor=.6) for c in [.25,.45,.65,.8]],
        'thermal_bias': [experiment(*x) for x in [(-.15,.5,-.6,.3,.65,1.2,.8,6.),(.2,-.5,.65,.15,.5,.7,1.4,4.),
                                                (0.,.0,.0,.25,.75,1.4,.6,5.),(.1,-.3,.4,.4,.4,1.5,.8,7.)]],
        'zero_frequency': [experiment(omega=0.,fraction=.33),experiment(offset=-.2,voltage_left=.2,voltage_right=-.7,fraction=.72,omega=0.)]
    }
