"""Unevaluated prototype; no benchmark source or grading generated."""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import solve_continuous_lyapunov

J=np.array([[0.,1.],[-1.,0.]])

def covariance(m,gamma,K,T,B):
    # Time-rescaled finite-mass OU for (x,sqrt(m)*v).
    A=np.block([[np.zeros((2,2)),np.sqrt(m)*np.eye(2)],[-np.sqrt(m)*K,-gamma*np.eye(2)+B*J]])
    Q=np.zeros((4,4)); Q[2:,2:]=2*gamma*np.diag(T)
    C=solve_continuous_lyapunov(A,-Q)
    C=(C+C.T)/2
    heat=gamma*(T-np.diag(C)[2:])/m
    # Independent stationary mechanical balance splits trap and Lorentz exchange.
    Y=C[:2,2:]/np.sqrt(m); V=C[2:,2:]/m
    trap=np.diag(K@Y); magnetic=-B*np.array([V[0,1],-V[0,1]])
    return C,heat,trap,magnetic,float(np.max(np.abs(A@C+C@A.T+Q)))

def moment_heat(m,gamma,K,T,B):
    s=np.trace(K); d=K[0,0]-K[1,1]; h=K[0,1]
    # Exact stationary equations for delta=<vx²-vy²>, c=<vx vy>, y=<x vy>.
    A=np.array([[gamma,-2*B,-2*h],[B,2*gamma,d],[m*h,-m*d,gamma*s]])
    delta,c,y=np.linalg.solve(A,np.array([gamma*(T[0]-T[1])/m,0.,0.]))
    return -B*c-h*y

def limit(gamma,K,T,b):
    dt=T[0]-T[1]
    trap=dt*K[0,1]**2/(gamma*np.trace(K)); kinetic=dt*b*b/(2*gamma)
    return trap+kinetic,trap,kinetic

rng=np.random.default_rng(53801)
rows=[]; max_first_law=0.; max_moment=0.; max_residual=0.; max_pos_error=0.
for _ in range(32):
    gamma=rng.uniform(.7,1.6); k1,k2=rng.uniform(.8,1.8,2)
    h=rng.uniform(-.55,.55)*np.sqrt(k1*k2)
    K=np.array([[k1,h],[h,k2]]); T=rng.uniform(.5,2.,2); b=rng.uniform(-1.5,1.5)
    target,trap,kinetic=limit(gamma,K,T,b)
    X0=solve_continuous_lyapunov(-K/gamma,-2*np.diag(T)/gamma)
    values=[]
    for m in [2.**-12,2.**-14,2.**-16,2.**-18]:
        B=b*np.sqrt(m); C,heat,qt,qb,res=covariance(m,gamma,K,T,B)
        analytic=moment_heat(m,gamma,K,T,B)
        values.append(float(heat[0])); max_moment=max(max_moment,abs(heat[0]-analytic))
        max_first_law=max(max_first_law,abs(sum(heat)))
        max_residual=max(max_residual,res)
        if m==2.**-18:max_pos_error=max(max_pos_error,float(np.max(abs(C[:2,:2]-X0))))
    ext=np.dot([1/3,-2,8/3],values[:3]); refined=np.dot([1/3,-2,8/3],values[1:])
    rows.append(dict(gamma=gamma,K=K.tolist(),temperatures=T.tolist(),b=b,
                     predicted_limit=target,trap_only_limit=trap,kinetic_limit=kinetic,
                     finite_mass_heat=values,extrapolated=float(ext),refined=float(refined),
                     refined_error=float(abs(refined-target))))

# A fixed-mass alternative uses the field in the exact positional covariance and
# trap-power channel; only identifying that channel as reservoir heat is wrong.
finite=[]
K=np.array([[1.2,.5],[.5,1.5]]); T=np.array([1.7,.7]); gamma=1.1
for m in [.15,.4,.8]:
 for B in [0.,.4,.8,1.2]:
    C,heat,trap,magnetic,res=covariance(m,gamma,K,T,B)
    finite.append(dict(mass=m,field=B,heat=float(heat[0]),trap_only=float(trap[0]),
                       kinetic_exchange=float(magnetic[0]),balance_error=float(abs(heat[0]-trap[0]-magnetic[0]))))
report=dict(status='prototype_only_no_task_or_evaluation',samples=rows,
            maximum_refined_limit_error=max(x['refined_error'] for x in rows),
            maximum_lyapunov_vs_independent_moment_heat_error=max_moment,
            maximum_first_law_error=max_first_law,maximum_lyapunov_residual=max_residual,
            maximum_position_covariance_error_at_smallest_mass=max_pos_error,
            fixed_mass_alternative=finite)
Path(__file__).with_name('report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('samples','fixed_mass_alternative')},indent=2))
print(json.dumps(finite,indent=2))
