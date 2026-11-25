from pathlib import Path
import importlib.util,json,time
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import expit
from scipy.optimize import root, least_squares
from scipy.integrate import quad
STAGE=Path('/private/tmp/physlitebench-hardening-20261005/staging/luna-high-hardening-20261005')
def load(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
r3=load(STAGE/'archives/superconducting-heat-r3/tasks/superconducting-heat/solution/model.py','r3')
ref3=load(STAGE/'archives/superconducting-heat-r3/tasks/superconducting-heat/tests/reference.py','ref3')
NU=1e6 # positive-energy quasiparticle states per spin per kelvin at zero gap
TAU=1e-5 # normal-state spin-polarization relaxation time, seconds
RATE=2e-6*r3.KB/r3.ELECTRON_CHARGE**2
HEAT=2e6*(r3.KB/r3.ELECTRON_CHARGE)**2

def preparation(tl,tr,phase,a,w,nodes=192,normal=False,mu_window=20.):
 dl,dr=(0.,0.) if normal else (r3.gap(tl,1.2),r3.gap(tr,1.9))
 x,weights=leggauss(nodes);channels=[];upper=np.sqrt((60*max(tl,tr)) if not a else (80*max(tl,tr)+mu_window))
 for n in (-1,0,1):
  if n and not a:continue
  edge=max(dl,dr-n*w);z=(x+1)*upper/2;el=edge+z*z;er=el+n*w
  den=np.sqrt((el-dl)*(el+dl)*(er-dr)*(er+dr));sign=-1 if n==0 else 1
  spectral=(el*er+sign*dl*dr*np.cos(phase))/den
  base=weights*upper/2*2*z*spectral
  spins=((1,1,.5),(-1,-1,.5)) if n==0 else ((-n,n,a*a/2),)
  for sl,sr,weight in spins:channels.append((el,er,base*weight,sl,sr,False,n*w))
 if a and w>dl+dr:
  ang=(x+1)*np.pi/4;el=dl+(w-dl-dr)*np.sin(ang)**2;er=w-el
  base=weights*np.pi/4*2*(el*er-dl*dr*np.cos(phase))/np.sqrt((el+dl)*(er+dr))*a*a/2
  channels.append((el,er,base,1,1,True,w))
 return (tl,tr,dl,dr,channels,(phase,a,w,nodes,normal,mu_window))

def evaluate(prep,mu):
 tl,tr,dl,dr,channels=prep[:5];heat=np.zeros(3);spin=np.zeros(2);jac=np.zeros((2,2));entropy=0.
 for el,er,weight,sl,sr,pair,work in channels:
  fl=expit((sl*mu[0]-el)/tl);fr=expit((sr*mu[1]-er)/tr)
  dlmu=fl*(1-fl)/tl;drmu=fr*(1-fr)/tr
  if pair:
   F=1-fl-fr;curr=np.array([sl,sr]);left_loss=-el;right_loss=-er
   df=np.stack([-sl*dlmu,-sr*drmu],axis=0)
   affinity=(el-sl*mu[0])/tl+(er-sr*mu[1])/tr
  else:
   F=fl-fr;curr=np.array([-sl,sr]);left_loss=el;right_loss=-er
   df=np.stack([sl*dlmu,-sr*drmu],axis=0)
   affinity=-(el-sl*mu[0])/tl+(er-sr*mu[1])/tr
  integral=weight@F
  heat+=np.array([weight@(F*left_loss),weight@(F*right_loss),work*integral])
  spin+=curr*integral
  jac+=np.outer(curr,df@weight)
  entropy+=weight@(F*affinity)
 return heat,spin,jac,float(entropy)

def relaxation(t,delta,mu):
 x=(mu-delta)/t;y=(-mu-delta)/t
 magnitude=abs(mu)
 if magnitude/t<50:
  value=np.sign(mu)*t*np.log1p(expit((-magnitude-delta)/t)*np.expm1(2*magnitude/t))
 else:
  value=t*(np.logaddexp(0,x)-np.logaddexp(0,y))
 derivative=expit(x)+expit(y)
 return value,derivative

def steady(prep,g=45.,tau=TAU,start=(0.,0.)):
 tl,tr,dl,dr,channels=prep[:5];lam=NU/(tau*RATE*g)
 def residual(mu):
  _,spin,_,_=evaluate(prep,mu)
  loss=np.array([relaxation(t,d,m)[0] for t,d,m in zip((tl,tr),(dl,dr),mu)])
  return spin-lam*loss
 def derivative(mu):
  _,_,jac,_=evaluate(prep,mu)
  loss=np.array([relaxation(t,d,m)[1] for t,d,m in zip((tl,tr),(dl,dr),mu)])
  return jac-lam*np.diag(loss)
 bound=20.
 guess=np.array(start,dtype=float)
 for expansion in range(8):
  ans=least_squares(residual,guess,jac=derivative,bounds=(-bound,bound),x_scale="jac",ftol=1e-14,xtol=1e-14,gtol=1e-14,max_nfev=200)
  for _ in range(10):
   value=residual(ans.x)
   if max(abs(value))<1e-13:break
   correction=np.linalg.solve(derivative(ans.x),value)
   step=1.
   while step>1e-5:
    candidate=ans.x-step*correction
    if max(abs(candidate))<bound and max(abs(residual(candidate)))<max(abs(value)):
     ans.x=candidate;break
    step/=2
   if step<=1e-5:break
  if max(abs(residual(ans.x)))<1e-10 and max(abs(ans.x))<bound*.8:break
  bound*=2
  phase,a,w,nodes,normal,_=prep[5]
  prep=preparation(tl,tr,phase,a,w,nodes,normal,mu_window=bound)
  guess=ans.x
 else:raise RuntimeError((ans.message,ans.x,residual(ans.x)))
 rawheat,spin,jac,sigma=evaluate(prep,ans.x)
 loss=np.array([relaxation(t,d,m)[0] for t,d,m in zip((tl,tr),(dl,dr),ans.x)])
 return dict(mu_K=ans.x,heat_pW=HEAT*g*rawheat,spin_per_s=RATE*g*spin,stationarity_residual_per_s=RATE*g*residual(ans.x),
  spin_work_identity_pW=HEAT*g*abs(rawheat[2]-(spin.sum()/2)*next((c[-1] for c in channels if c[-1]>0),0.)),
  energy_work_residual_pW=HEAT*g*abs(rawheat.sum()),tunneling_entropy=kB_scale(g)*sigma,
  spinflip_entropy=kB_scale(g)*lam*sum(ans.x*loss/np.array([tl,tr])),jacobian=derivative(ans.x))

def kB_scale(g):return RATE*g*r3.KB

def matrix_probability(el,er,dl,dr,phase,n,a,pair):
 mats=ref3.spin_matrices(a);S=ref3.SINGLET;T=mats[n];R=S@mats[-n].conj()@S.conj().T;prob=np.zeros((2,2))
 for sl in (-1,1):
  ul=np.sqrt((1+sl*np.sqrt(el*el-dl*dl)/el)/2);vl=np.sqrt((1-sl*np.sqrt(el*el-dl*dl)/el)/2)
  for sr in (-1,1):
   ur=np.sqrt((1+sr*np.sqrt(er*er-dr*dr)/er)/2);vr=np.sqrt((1-sr*np.sqrt(er*er-dr*dr)/er)/2)
   A=(ul*vr*T+vl*ur*np.exp(1j*phase)*R)@S if pair else ul*ur*T-vl*vr*np.exp(1j*phase)*R
   prob+=abs(A)**2/4
 return prob

def independent(prep,phase,a,w,mu,g=45.):
 tl,tr,dl,dr,_=prep[:5];values=np.zeros(5)
 for n in (-1,0,1):
  if n and not a:continue
  edge=max(dl,dr-n*w);upper=np.sqrt(80*max(tl,tr)+max(abs(mu))+w)
  def integrand(z,index):
   el=edge+z*z;er=el+n*w;J=el*er/np.sqrt((el-dl)*(el+dl)*(er-dr)*(er+dr));P=matrix_probability(el,er,dl,dr,phase,-n,a,False)
   values=np.zeros(5)
   for i,sl in enumerate((1,-1)):
    for j,sr in enumerate((1,-1)):
     F=expit((sl*mu[0]-el)/tl)-expit((sr*mu[1]-er)/tr)
     values+=P[i,j]*F*np.array([el,-er,n*w,-sl,sr])
   return 2*z*J*values[index]
  values+=np.array([quad(integrand,0,upper,args=(i,),epsabs=1e-10,epsrel=1e-10)[0] for i in range(5)])
 if a and w>dl+dr:
  interval=w-dl-dr
  def integrand(angle,index):
   el=dl+interval*np.sin(angle)**2;er=w-el;P=matrix_probability(el,er,dl,dr,phase,1,a,True);J=2*el*er/np.sqrt((el+dl)*(er+dr));values=np.zeros(5)
   for i,sl in enumerate((1,-1)):
    for j,sr in enumerate((1,-1)):
     F=1-expit((sl*mu[0]-el)/tl)-expit((sr*mu[1]-er)/tr)
     values+=P[i,j]*F*np.array([-el,-er,w,sl,sr])
   return J*values[index]
  values+=np.array([quad(integrand,0,np.pi/2,args=(i,),epsabs=1e-10,epsrel=1e-10)[0] for i in range(5)])
 return values*np.r_[np.full(3,HEAT*g),np.full(2,RATE*g)]

start=time.time();report={'nu_positive_QP_per_spin_per_K':NU,'normal_spin_relaxation_seconds':TAU,'hidden':{},'normal':[],'independent':[]}
for group,es in ref3.hidden_inputs().items():
 outputs=[];old=[];mus=[];residuals=[];energy=[];spinwork=[];entropy=[]
 for e in es:
  tl,tr,p,a,w=[e[k] for k in ('left_temperature','right_temperature','phase','amplitude','drive_energy')];prep=preparation(tl,tr,p,a,w);value=steady(prep)
  outputs.append(value['heat_pW'][0]);old.append(r3.predict_at([e],45.)[0]);mus.append(value['mu_K'].tolist());residuals.append(max(abs(value['stationarity_residual_per_s'])));energy.append(value['energy_work_residual_pW']);spinwork.append(value['spin_work_identity_pW']);entropy.append(value['tunneling_entropy']+value['spinflip_entropy'])
 report['hidden'][group]={'relative_RMSE_shortcut':float(np.sqrt(np.mean((np.array(outputs)-old)**2)/np.mean(np.array(outputs)**2))),'mu_K':mus,'correct_pW':outputs,'mu0_shortcut_pW':old,'stationarity_per_s_max':max(residuals),'energy_residual_pW_max':max(energy),'spin_work_identity_pW_max':max(spinwork),'entropy_W_per_K_min':min(entropy)}
for tl,tr,a,w in ((.25,.25,1.,8.),(.4,.8,.7,7.),(.8,.4,.3,10.),(.25,1.05,1.,.3)):
 prep=preparation(tl,tr,.7,a,w,normal=True);v=steady(prep);c=RATE*45.;r=NU/TAU;mu=c*a*a*w/(2*(r+c*a*a));A=lambda T,m:np.pi**2*T*T/6+m*m/2
 Q=1e6*(r3.KB/r3.ELECTRON_CHARGE)**2*45.*((1+a*a)*(A(tl,mu)-A(tr,mu))-a*a*w*w/2+a*a*w*mu)
 report['normal'].append({'mu_closed_K':mu,'mu_numerical_K':v['mu_K'].tolist(),'mu_error_K':float(max(abs(v['mu_K']-mu))),'heat_error_pW':float(abs(v['heat_pW'][0]-Q))})
for tl,tr,p,a,w in ((.25,.25,0.,.6,7.),(.45,.8,np.pi,.8,8.),(.75,.5,.7,1.,10.),(.5,.9,-.8,.8,.3)):
 prep=preparation(tl,tr,p,a,w);v=steady(prep);ind=independent(prep,p,a,w,v['mu_K']);allstarts=[steady(prep,start=s)['mu_K'] for s in ((w/4,w/4),(-w/4,-w/4),(w/2,0.))];refined=steady(preparation(tl,tr,p,a,w,nodes=288));J=v['jacobian']
 report['independent'].append({'input':[tl,tr,p,a,w],'mu_K':v['mu_K'].tolist(),'matrix_heat_agreement_pW':float(max(abs(ind[:3]-v['heat_pW']))),'matrix_spin_agreement_per_s':float(max(abs(ind[3:]-v['spin_per_s']))),'node_refinement_heat_pW':float(max(abs(v['heat_pW']-refined['heat_pW']))),'multiple_start_mu_error_K':float(max(abs(np.array(allstarts)-v['mu_K']).ravel())),'column_diagonal_dominance_margin':float(min(-J[0,0]-abs(J[1,0]),-J[1,1]-abs(J[0,1]))),'jacobian_max_real_eigenvalue':float(max(np.linalg.eigvals(J).real))})
report['seconds']=time.time()-start
out=Path('/private/tmp/physlite-super-spin-accumulation-prototype.json');out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
