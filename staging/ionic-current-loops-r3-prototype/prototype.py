"""Closed periodic electrolyte versus zero-mean-current elliptic closure.

Prototype only. No calibration dataset, task package or model evaluations.
"""
from pathlib import Path
import json
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import LinearOperator, cg


def profiles(n, amplitudes, phase):
    q = np.arange(n)*2*np.pi/n
    x, y = np.meshgrid(q, q, indexing='ij')
    a, b, c = amplitudes[:3]
    c1 = 1+a*np.cos(x)+b*np.cos(y)
    if len(amplitudes)==3:
        c2 = 1+c*np.cos(x+y+phase)
    else:
        c2 = 1+c*np.cos(x+phase)+amplitudes[3]*np.cos(x+y)
    concentrations = np.array([c1,c2,c1+c2])
    return x,y,concentrations


def spectral(amplitudes=(.4,.4,.75), phase=1., n=63):
    x,y,c = profiles(n,amplitudes,phase)
    d = np.array([1.,4.,1.])
    z = np.array([1.,1.,-1.])
    conductivity = np.einsum('i,ijk->jk',d,c)
    g = np.einsum('i,ijk->jk',z*d,c)
    k = np.fft.fftfreq(n,1/n)
    kx,ky = np.meshgrid(k,k,indexing='ij')
    k2 = kx*kx+ky*ky
    def grad(f):
        ft = np.fft.fft2(f)
        return np.array([np.fft.ifft2(1j*kx*ft).real,np.fft.ifft2(1j*ky*ft).real])
    def div(v):
        return np.fft.ifft2(1j*kx*np.fft.fft2(v[0])+1j*ky*np.fft.fft2(v[1])).real
    def lap(f):
        return np.fft.ifft2(-k2*np.fft.fft2(f)).real
    def apply(v):
        v = v.reshape(n,n)
        return (-div(conductivity*grad(v))+v.mean()).ravel()
    denominator = conductivity.mean()*k2
    denominator[0,0] = 1.
    def precondition(v):
        return np.fft.ifft2(np.fft.fft2(v.reshape(n,n))/denominator).real.ravel()
    op = LinearOperator((n*n,n*n),matvec=apply,dtype=float)
    pre = LinearOperator(op.shape,matvec=precondition,dtype=float)
    def solve(rhs):
        phi,info = cg(op,rhs.ravel(),M=pre,rtol=3e-13,atol=1e-14,maxiter=1000)
        assert info==0,info
        return phi.reshape(n,n)
    phi = solve(lap(g))
    electric = -grad(phi)
    current = -grad(g)+conductivity*electric
    corrections = []
    for a in range(2):
        basis = np.zeros((2,n,n));basis[a] = 1.
        basis -= grad(solve(-grad(conductivity)[a]))
        corrections.append(basis)
    corrections = np.array(corrections)
    # Columns are mean current responses to each imposed harmonic field.
    conductance = np.stack([(conductivity*basis).mean(axis=(1,2)) for basis in corrections],axis=1)
    harmonic = np.linalg.solve(conductance,-current.mean(axis=(1,2)))
    source_electric = electric+np.einsum('i,ijab->jab',harmonic,corrections)
    local_electric = grad(g)/conductivity
    grad_c = np.array([grad(ci) for ci in c])
    fluxes = {}
    for name,field in [('oracle',electric),('shortcut',source_electric),('old_local',local_electric)]:
        fluxes[name] = -d[:,None,None,None]*grad_c+(d*z)[:,None,None,None]*c[:,None,:,:]*field
    return dict(x=x,y=y,c=c,fluxes=fluxes,electric=electric,source_electric=source_electric,
                current=current,harmonic=harmonic,conductance=conductance,
                current_divergence=float(abs(div(current)).max()),
                shortcut_current_mean=np.einsum('i,ijab->jab',z,fluxes['shortcut']).mean(axis=(1,2)),
                current_rms=float(np.sqrt(np.mean(current**2))),
                mean_current=current.mean(axis=(1,2)))


def rate(fields, species, mode, phase, model='oracle'):
    m,n = mode
    test = m*fields['x']+n*fields['y']+phase
    flux = fields['fluxes'][model][species]
    return float(-2*np.mean((m*flux[0]+n*flux[1])*np.sin(test)))


def finite_volume(amplitudes, phase, species, mode, readout_phase, n=96, closed=True):
    x,y,c = profiles(n,amplitudes,phase)
    h=2*np.pi/n;d=np.array([1.,4.,1.]);z=np.array([1.,1.,-1.])
    sigma=np.einsum('i,ijk->jk',d,c);g=np.einsum('i,ijk->jk',d*z,c)
    face_sigma=np.array([.5*(sigma+np.roll(sigma,-1,a)) for a in range(2)])
    ids=np.arange(n*n).reshape(n,n);rows=[];cols=[];values=[]
    for a in range(2):
        i=ids.ravel();j=np.roll(ids,-1,a).ravel();w=face_sigma[a].ravel()/h**2
        rows.extend([i,i,j,j]);cols.extend([i,j,i,j]);values.extend([w,-w,-w,w])
    A=coo_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(cols))),shape=(n*n,n*n)).tocsr()
    op=LinearOperator(A.shape,matvec=lambda v:A@v+v.mean(),dtype=float)
    k=np.fft.fftfreq(n,1/n);kx,ky=np.meshgrid(k,k,indexing='ij')
    denom=sigma.mean()*4/h**2*(np.sin(np.pi*kx/n)**2+np.sin(np.pi*ky/n)**2);denom[0,0]=1
    pre=LinearOperator(A.shape,matvec=lambda v:np.fft.ifft2(np.fft.fft2(v.reshape(n,n))/denom).real.ravel(),dtype=float)
    def grad(f):return np.array([(np.roll(f,-1,a)-f)/h for a in range(2)])
    def div(v):return sum((v[a]-np.roll(v[a],1,a))/h for a in range(2))
    def solve(rhs):
        phi,info=cg(op,rhs.ravel(),M=pre,rtol=1e-12,atol=1e-14,maxiter=1000);assert info==0
        return phi.reshape(n,n)
    field=-grad(solve(div(grad(g))))
    current=-grad(g)+face_sigma*field
    if not closed:
        corrections=[]
        for a in range(2):
            e=np.zeros((2,n,n));e[a]=1
            corrections.append(e-grad(solve(-div(face_sigma*e))))
        response=np.stack([(face_sigma*e).mean(axis=(1,2)) for e in corrections],axis=1)
        harmonic=np.linalg.solve(response,-current.mean(axis=(1,2)))
        field+=np.einsum('i,ijab->jab',harmonic,corrections)
    face_c=np.array([.5*(c[species]+np.roll(c[species],-1,a)) for a in range(2)])
    flux=-d[species]*grad(c[species])+z[species]*d[species]*face_c*field
    test=np.cos(mode[0]*x+mode[1]*y+readout_phase)
    return float(2*np.mean(np.sum(flux*grad(test),axis=0)))


def screen():
    cases=[]
    for phase in [.4,.8,1.2,1.6,2.]:
        f=spectral(phase=phase)
        rates=[]
        for species in [0,1]:
            for mode in [(1,0),(0,1),(1,1),(1,-1),(2,1),(2,0)]:
                for angle in [0.,np.pi/4,np.pi/2]:
                    a=rate(f,species,mode,angle);b=rate(f,species,mode,angle,'shortcut')
                    rates.append(dict(species=species,mode=mode,readout_phase=angle,oracle=a,shortcut=b,
                                      difference=b-a,relative_error=abs(b-a)/max(abs(a),1e-12)))
        cases.append(dict(phase=phase,mean_current=f['mean_current'].tolist(),harmonic=f['harmonic'].tolist(),
                          current_rms=f['current_rms'],current_divergence=f['current_divergence'],rates=rates))
    cal=spectral(phase=0.)
    cal_rows=[]
    for species in [0,1]:
        for mode in [(1,0),(0,1),(1,1),(1,-1),(2,1),(2,0)]:
            cal_rows.append(dict(species=species,mode=mode,oracle=rate(cal,species,mode,0.),
                                 shortcut=rate(cal,species,mode,0.,'shortcut'),old_local=rate(cal,species,mode,0.,'old_local')))
    return dict(status='prototype_only',amplitudes=[.4,.4,.75],calibration=dict(mean_current=cal['mean_current'].tolist(),
                current_rms=cal['current_rms'],rates=cal_rows),screen=cases)


if __name__=='__main__':
    report=screen()
    p=Path(__file__).with_name('report.json');p.write_text(json.dumps(report,indent=2)+'\n')
    for case in report['screen']:
        rows=[r for r in case['rates'] if abs(r['oracle'])>.05]
        print(case['phase'],case['mean_current'],sorted(rows,key=lambda r:r['relative_error'],reverse=True)[:3])
    print('calibration',report['calibration'])
