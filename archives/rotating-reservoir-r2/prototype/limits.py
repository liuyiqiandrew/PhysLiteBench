"""Additional bounded limiting and parameter-range checks; initial report preserved."""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import solve_continuous_lyapunov
from check import evaluate, B, R

rows=[]
for g in np.linspace(.4,1.1,41):
    for w,s,t,kx,ky,ta,tb in [(.3,.4,.6,1.2,2.5,1.,1.),(-.45,-.35,.8,1.5,2.4,1.2,.9),(.2,.3,.7,1.,2.8,.9,1.1)]:
        r=evaluate(float(g),w,s,t,kx,ky,ta,tb);rows.append(r)
limits=[]
for w,s in [(.3,.4),(-.5,-.3)]:
    g=.67;kx=1.2;ky=2.5;ta=1.2;tb=.9;A=w*R+np.diag([s,-s])
    drift=np.block([[np.zeros((2,2)),np.eye(2)],[-np.diag([kx,ky])+B*A,-(g+B)*np.eye(2)]])
    noise=np.diag([0.,0.,2*(g*ta+B*tb),2*(g*ta+B*tb)])
    C=solve_continuous_lyapunov(drift,-noise);L=np.column_stack([-A,np.eye(2)])
    markov=float(B*np.trace(L@C@L.T)-2*B*tb)
    r=[]
    for tau in [.1,.01,.001,.0001]:
        value=evaluate(g,w,s,tau,kx,ky,ta,tb)
        r.append({'tau':tau,'heat':value['heat'],'relative_particle_work':value['relative_particle_work'],'markov_heat_error':abs(value['heat']-markov),'finite_compliance_gap':value['absolute_gap']})
    limits.append({'omega':w,'strain':s,'markov_heat':markov,'refinements':r})
gibbs=[]
for g,t,kx,ky,T in [(g,t,kx,ky,T) for g in [.4,1.1] for t in [.2,.8] for kx,ky in [(1.,2.2),(1.6,2.8)] for T in [.8,1.4]]:
    r=evaluate(g,0.,0.,t,kx,ky,T,T)
    target=np.diag([T/kx,T/ky,T,T,B*T/t,B*T/t])
    gibbs.append({'controls':r['controls'],'covariance_error':float(abs(np.array(r['covariance'])-target).max()),'heat_error':abs(r['heat'])})
summary={'parameter_scan_cases':len(rows),'minimum_heat':min(r['heat'] for r in rows),'minimum_relative_particle_work':min(r['relative_particle_work'] for r in rows),'minimum_relative_gap':min(r['relative_gap'] for r in rows),'maximum_relative_gap':max(r['relative_gap'] for r in rows),'max_rest_gibbs_error':max(r['covariance_error'] for r in gibbs),'max_rest_equilibrium_heat':max(r['heat_error'] for r in gibbs)}
out={'status':'bounded followup, no task or model runs','parameter_scan':rows,'zero_compliance_limit':limits,'equilibrium_gibbs':gibbs,'summary':summary}
Path(__file__).with_name('limits-report.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(summary,indent=2))
assert summary['minimum_heat']>.1 and summary['minimum_relative_particle_work']>.05
assert summary['minimum_relative_gap']>.3
assert summary['max_rest_gibbs_error']<1e-11
for r in limits:
    errors=[x['markov_heat_error'] for x in r['refinements']]
    assert all(a>b for a,b in zip(errors,errors[1:]))
