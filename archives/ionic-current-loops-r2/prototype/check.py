import numpy as np
import json
from pathlib import Path
D=np.array([1.,.08,3.]);Z=np.array([1.,1.,-1.])

def compute(means,amplitudes,waves,phases,n=1024):
 x=np.arange(n)*2*np.pi/n;k=np.fft.fftfreq(n,1/n)
 c=np.array([means[i]+amplitudes[i]*np.cos(waves[i]*x+phases[i]) for i in [0,1]])
 c=np.vstack((c,c.sum(axis=0)))
 dc=np.fft.ifft(1j*k*np.fft.fft(c,axis=1),axis=1).real
 sigma=D@c;gprime=(Z*D)@dc
 loop=-np.mean(gprime/sigma)/np.mean(1/sigma)
 result=[]
 for current in [loop,0.]:
  field=(current+gprime)/sigma
  flux=-D[:,None]*dc+(Z*D)[:,None]*c*field
  rate=-np.fft.ifft(1j*k*np.fft.fft(flux,axis=1),axis=1).real
  result.append(dict(rate=rate,field=field,flux=flux))
 return x,c,loop,result

if __name__=='__main__':
    rows=[]
    for amplitude in [.3,.5,.7,.85]:
     for phase in [.7,1.3,2.1]:
      x,c,j,models=compute([1,.8],[amplitude,.8*amplitude],[1,1],[0,phase])
      a,b=models;error=np.sqrt(np.mean((a['rate']-b['rate'])**2)/np.mean(a['rate']**2))
      rows.append(dict(relative_amplitude=amplitude,phase=phase,current=float(j),rate_relative_gap=float(error),minimum_concentration=float(c.min()),mean_electric_field=float(np.mean(a['field'])),shortcut_mean_field=float(np.mean(b['field'])),max_local_charge_rate=float(np.max(abs(Z@a['rate']))),mass_rate_max=float(np.max(abs(np.mean(a['rate'],axis=1)))),entropy_rate=float(np.sum(a['flux']**2/(D[:,None]*c))*2*np.pi/len(x))))
    report=dict(status='prototype_only',examples=rows)
    p=Path(__file__).parent/'report.json';p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
