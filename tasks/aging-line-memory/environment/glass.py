import numpy as np
from scipy.special import betainc

class Readout:
    def fit(self, records):
        e=np.array([r['input']['threshold'] for r in records])
        attenuation=-np.log([r['value'] for r in records])
        self.depth_scale=float(np.clip((e@e)/(e@attenuation),.9,1.1))
        return self

    def predict(self, inputs):
        return np.asarray([np.exp(-q['threshold']/self.depth_scale) if 'threshold' in q
                           else self.memory(q) for q in inputs],dtype=float)

def effective_age(q,tg):
    old=q['temperatures'][0];age=sum(q['write']);f_age=None
    for i,((dt,dwell),temp) in enumerate(zip(q['pulses'],q['temperatures'][1:])):
        power=old/temp
        if f_age is not None:
            f_age=f_age**power
        age=age**power
        if i==q['read'][0]:
            f_age=age+q['read'][1]*dt
        age+=dt+dwell
        old=temp
    final=q['temperatures'][0]
    power=old/final
    x=final/tg
    # Assign a single Arrhenius age to the ensemble at each temperature.
    age=age**power;start=f_age**power
    return float(betainc(x,1-x,start/(age+q['recovery'])) /
                 betainc(x,1-x,age/(age+q['recovery'])))
