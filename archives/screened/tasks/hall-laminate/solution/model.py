import numpy as np

CHARGE_DENSITY = 1e4


def conductivity(mobility,field,sign):
    beta = mobility*field
    return CHARGE_DENSITY*mobility/(1+beta*beta)*np.array([[1.,sign*beta],[-sign*beta,1.]])


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        design=[]
        for record in records:
            e=record['input'];f=e['fraction'];c,s=np.cos(e['normal_angle']),np.sin(e['normal_angle'])
            design.append((c*c*(f+(1-f)/2)+s*s/(f+2*(1-f)))/CHARGE_DENSITY)
        design=np.array(design)
        values=np.array([r['value'] for r in records])
        sigma=np.array([r['sigma'] for r in records])
        inverse=np.sum(design*values/sigma**2)/np.sum(design**2/sigma**2)
        self.mobility=float(np.clip(1/inverse,.2,1.5))
        return self

    def predict(self, experiments):
        out = []
        for e in experiments:
            tensors = [conductivity(self.mobility,e['field'],1),conductivity(2*self.mobility,e['field'],-1)]
            fractions = [e['fraction'],1-e['fraction']]
            angle=e['normal_angle'];c,s=np.cos(angle),np.sin(angle)
            rotation=np.array([[c,-s],[s,c]])
            local=[rotation.T@tensor@rotation for tensor in tensors]
            A=sum(f/t[0,0] for f,t in zip(fractions,local))
            B=sum(f*t[0,1]/t[0,0] for f,t in zip(fractions,local))
            C=sum(f*t[1,0]/t[0,0] for f,t in zip(fractions,local))
            D=sum(f*(t[1,1]-t[1,0]*t[0,1]/t[0,0]) for f,t in zip(fractions,local))
            effective=np.array([[1/A,B/A],[C/A,D+C*B/A]])
            electric=rotation@np.linalg.solve(effective,rotation.T@np.array([1.,0.]))
            out.append(electric[0 if e['observable']=='longitudinal_resistivity' else 1])
        return np.array(out)
