"""Static changing-tilt capture screen, preserving every attempted setting."""
import json,time
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
import r1_geometry as old

HERE=Path(__file__).resolve().parent


def tilt(s, final, slope, endpoint):
    return final+slope*np.log(s/endpoint)


def area_derivative(delta):
    c=old.geometry(float(delta))[0][1]
    half=np.sqrt(2*(1-c*c))
    angle=np.arcsin(2*c/half)
    return 2*np.sqrt(2)*np.array([angle+np.pi/2,angle-np.pi/2])


def growth(s, final, slope, endpoint):
    delta=tilt(s,final,slope,endpoint)
    areas=old.geometry(float(delta))[2]
    return np.sqrt(s)*(1.5*areas+slope*area_derivative(delta))


def one_action(action, final, slope, endpoint):
    total=lambda s:s**1.5*old.geometry(float(tilt(s,final,slope,endpoint)))[2].sum()
    target=2*np.pi*action
    if total(.05)>=target:
        raise ValueError('Initial orbit is not connected')
    if total(endpoint)<=target:
        e=endpoint**2*old.energy_for_volume(target/endpoint**1.5,final)
        return dict(physical=e,instant_area=e,endpoint_area=e,crossing=None)
    crossing=brentq(lambda s:total(s)-target,.05,endpoint,xtol=1e-13)
    delta=tilt(crossing,final,slope,endpoint)
    areas=old.geometry(float(delta))[2]
    branch_actions=crossing**1.5*areas/(2*np.pi)
    probabilities=growth(crossing,final,slope,endpoint)
    probabilities/=probabilities.sum()
    instant=areas/areas.sum()
    energies=np.array([endpoint**2*old.branch_energy(2*np.pi*j/endpoint**1.5,final,i)
                       for i,j in enumerate(branch_actions)])
    final_areas=old.geometry(float(final))[2];end_fraction=final_areas/final_areas.sum()
    endpoint_energies=np.array([endpoint**2*old.branch_energy(2*np.pi*action*f/endpoint**1.5,final,i)
                               for i,f in enumerate(end_fraction)])
    return dict(physical=float(probabilities@energies),instant_area=float(instant@energies),
                endpoint_area=float(end_fraction@endpoint_energies),crossing=float(crossing),
                crossing_tilt=float(delta),probabilities=probabilities.tolist(),
                area_fractions=instant.tolist(),branch_actions=branch_actions.tolist())


def predict(j0, final, slope, endpoint, order=24):
    z,w=old.quadrature(order)
    rows=[one_action(float(j0*(1.1+.1*x)),final,slope,endpoint) for x in z]
    values={key:float(w@np.array([r[key] for r in rows])/2)
            for key in ['physical','instant_area','endpoint_area']}
    values['instant_relative_gap']=abs(values['instant_area']/values['physical']-1)
    values['endpoint_relative_gap']=abs(values['endpoint_area']/values['physical']-1)
    values['min_crossing']=min(r['crossing'] for r in rows if r['crossing'] is not None) if any(r['crossing'] is not None for r in rows) else None
    values['max_crossing']=max(r['crossing'] for r in rows if r['crossing'] is not None) if any(r['crossing'] is not None for r in rows) else None
    values['details']=rows
    return values


def main():
    start=time.perf_counter();rows=[]
    for j0 in [.12,.1437,.16]:
        for final in [.15,.2,.22]:
            for slope in [-.025,0.,.03,.06,.10,.15]:
                for endpoint in [1.8,2.2]:
                    parameters=dict(j0=j0,delta_final=final,slope=slope,s_final=endpoint)
                    try:
                        samples=np.geomspace(.05,endpoint,129)
                        tilts=np.array([tilt(s,final,slope,endpoint) for s in samples])
                        if np.max(abs(tilts))>=2/(3*np.sqrt(3)):
                            rows.append(dict(**parameters,status='rejected_lost_double_well',tilt_min=float(tilts.min()),tilt_max=float(tilts.max())))
                            continue
                        minimum=float(min(growth(s,final,slope,endpoint).min() for s in samples))
                        if minimum<=0:
                            rows.append(dict(**parameters,status='rejected_nonmonotone_lobe',minimum_growth=minimum));continue
                        values=predict(j0,final,slope,endpoint)
                        rows.append(dict(**parameters,status='evaluated',minimum_growth=minimum,**values))
                    except Exception as exc:
                        rows.append(dict(**parameters,status='author_error',error=repr(exc)))
    report={'runtime_seconds':time.perf_counter()-start,'rows':rows,'count':len(rows),'model_evaluations':0}
    (HERE/'screen-report.json').write_text(json.dumps(report,indent=2)+'\n')
    good=[r for r in rows if r['status']=='evaluated' and abs(r['physical'])>.5]
    print(json.dumps({'seconds':report['runtime_seconds'],'count':len(rows),'status_counts':{s:sum(r['status']==s for r in rows) for s in set(r['status'] for r in rows)},'top':[{k:v for k,v in r.items() if k!='details'} for r in sorted(good,key=lambda r:r['instant_relative_gap'],reverse=True)[:12]]},indent=2))

if __name__=='__main__':main()
