import numpy as np

CHARGE_DENSITY = 1e4


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        raise NotImplementedError

    def predict(self, experiments):
        out=[]
        for e in experiments:
            f=e['fraction'];angle=e['normal_angle'];c,s=np.cos(angle),np.sin(angle)
            rotation=np.array([[c,-s],[s,c]])
            r1=1/(CHARGE_DENSITY*self.mobility)
            r2=1/(CHARGE_DENSITY*2*self.mobility)
            h1=-e['field']/CHARGE_DENSITY
            h2=e['field']/CHARGE_DENSITY
            tangential_conductance=f/r1+(1-f)/r2
            normal_resistance=f*r1+(1-f)*r2
            tangential_resistance=1/tangential_conductance
            hall_resistance=(f*h1/r1+(1-f)*h2/r2)/tangential_conductance
            local=np.array([[normal_resistance,hall_resistance],[-hall_resistance,tangential_resistance]])
            electric=rotation@local@rotation.T@np.array([1.,0.])
            out.append(electric[0 if e['observable']=='longitudinal_resistivity' else 1])
        return np.array(out)
