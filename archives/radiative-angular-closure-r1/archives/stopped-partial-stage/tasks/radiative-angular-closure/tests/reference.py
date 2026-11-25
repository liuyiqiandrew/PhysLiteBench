"""Real-space transport of the prepared intensity profiles."""
import numpy as np
TRUE_PARAMETER=.27


def predict(experiments, absorption=TRUE_PARAMETER):
    result=[]
    for e in experiments:
        total=0.0
        for direction,weight,modulation in zip(e["directions"],e["weights"],e["modulations"]):
            origin=e["position"]-direction[0]*e["time"]
            initial_change=weight*modulation*np.cos(e["wavenumber"]*origin)
            total+=initial_change*np.exp(-absorption*e["time"])
        result.append(total)
    return np.asarray(result,dtype=float)


def calibration_inputs():
    rays=[([[1,0,0],[-1,0,0]],[.5,.5]),
          ([[.8,.6,0],[-.8,.6,0]],[.6,.4]),
          ([[1,0,0],[0,1,0],[-.6,0,.8]],[.35,.3,.35])]
    unique=[]
    for i in range(12):
        directions,weights=rays[i%3]
        unique.append(dict(directions=directions,weights=weights,
            modulations=[.35+.035*i]*len(weights),wavenumber=0,time=.4+.22*i,position=.3+.31*i))
    return [dict(e) for _ in range(24) for e in unique]


def hidden_inputs():
    groups={"opposed_directions":[],"oblique_directions":[],"three_directions":[]}
    for i in range(12):
        k=1+i%3;time=(2.6+.035*i)/k;x=.06+.026*i
        groups["opposed_directions"].append(dict(directions=[[1,0,0],[-1,0,0]],
            weights=[.5+.01*i,.5-.01*i],modulations=[.7,.7],wavenumber=k,time=time,position=x))
        u=.73+.01*i;v=float(np.sqrt(1-u*u))
        groups["oblique_directions"].append(dict(directions=[[u,v,0],[-u,v,0]],
            weights=[.5+.005*i,.5-.005*i],modulations=[.7,.65],wavenumber=k,time=time,position=x))
        groups["three_directions"].append(dict(directions=[[1,0,0],[0,1,0],[-.6,0,.8]],
            weights=[.35+.004*i,.3-.005*i,.35+.001*i],modulations=[.8,.5,.6],
            wavenumber=k,time=time,position=x))
    return groups
