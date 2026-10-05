"""Bounded author-only dispersion prototype. No task or model evaluation."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import json,time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve,expm_multiply,eigs


def velocity_modes(field):
    out=[]
    for comp,axis,amps in [(0,1,(field['a'],field['c'])),(1,0,(field['b'],field['d']))]:
        for wave,amp in enumerate(amps,1):
            if amp==0: continue
            for sign in [-1,1]:
                q=[0,0];q[axis]=sign*wave
                vec=np.zeros(2,complex);vec[comp]=amp*sign/(2j)
                out.append((tuple(q),vec))
    return out


def source(field,D):
    return np.diag([D+(field['a']**2+field['c']**2/4)/(2*D),D+(field['b']**2+field['d']**2/4)/(2*D)])


def fourier(field,D,cutoff=18):
    modes=[(i,j) for i in range(-cutoff,cutoff+1) for j in range(-cutoff,cutoff+1) if (i,j)!=(0,0)]
    lookup={p:i for i,p in enumerate(modes)}; count=len(modes)
    rows=list(range(count));cols=list(range(count));k2=np.array([i*i+j*j for i,j in modes])
    vals=list(D*k2+0j);U=np.zeros((count,2),complex)
    vel=velocity_modes(field)
    for q,v in vel:
        U[lookup[q]]+=v
        for col,p in enumerate(modes):
            row=lookup.get((p[0]+q[0],p[1]+q[1]))
            if row is not None:
                rows.append(row);cols.append(col);vals.append(-1j*np.dot(p,v))
    A=sparse.coo_matrix((vals,(rows,cols)),shape=(count,count)).tocsc()
    chi=spsolve(A,U)
    H=np.real(U.conj().T@chi)
    K=D*np.eye(2)+(H+H.T)/2
    energy=D*np.eye(2)+D*np.real(chi.conj().T@(k2[:,None]*chi))
    return K,{'residual':float(np.max(abs(A@chi-U))),'energy_identity_error':float(np.max(abs(K-energy))),'minimum_eigenvalue':float(np.min(np.linalg.eigvalsh(K))),'source_upper_bound_min_eigenvalue':float(np.min(np.linalg.eigvalsh(source(field,D)-K)))}


def jump_generator(field,D,n):
    h=2*np.pi/n; xy=np.indices((n,n)).reshape(2,-1).T*h; size=n*n
    ux=field['a']*np.sin(xy[:,1])+field['c']*np.sin(2*xy[:,1])
    uy=field['b']*np.sin(xy[:,0])+field['d']*np.sin(2*xy[:,0]); v=np.array([ux,uy]).T
    rows=[];cols=[];rates=[];incs=[]
    ids=np.arange(size).reshape(n,n)
    for axis in range(2):
        for sign in [-1,1]:
            rows.extend(range(size));cols.extend(np.roll(ids,-sign,axis=axis).ravel())
            rates.extend(D/h**2+sign*v[:,axis]/(2*h))
            delta=np.zeros((size,2));delta[:,axis]=sign*h;incs.extend(delta)
    rows=np.array(rows);cols=np.array(cols);rates=np.array(rates);incs=np.array(incs)
    assert rates.min()>0,(field,D,n,rates.min())
    off=sparse.coo_matrix((rates,(rows,cols)),shape=(size,size)).tocsc()
    L=off-sparse.diags(np.asarray(off.sum(axis=1)).ravel())
    L1=[sparse.coo_matrix((rates*incs[:,j],(rows,cols)),shape=L.shape).tocsc() for j in range(2)]
    U=np.column_stack([np.asarray(Q.sum(axis=1)).ravel() for Q in L1])
    bare=np.zeros((2,2))
    for i in range(2):
        for j in range(2):bare[i,j]=np.sum(rates*incs[:,i]*incs[:,j])/size
    return L,L1,U,bare,(rows,cols,rates,incs)


def reference(field,D,n=64):
    L,L1,U,bare,_=jump_generator(field,D,n);size=L.shape[0]
    chi=np.zeros_like(U);chi[1:]=spsolve(-L[1:,1:],U[1:]);chi-=chi.mean(axis=0)
    H=np.array([np.mean(Q@chi,axis=0) for Q in L1]);K=(bare+H+H.T)/2
    return K,{'generator_row_sum':float(np.max(abs(L@np.ones(size)))),'uniform_stationary_residual':float(np.max(abs(np.asarray(L.sum(axis=0)).ravel()))),'poisson_residual':float(np.max(abs(L@chi+U))),'minimum_rate':float(np.min(jump_generator(field,D,n)[4][2]))}


def finite_time(field,D,n=32):
    L,L1,U,bare,_=jump_generator(field,D,n);size=L.shape[0]
    avg=np.asarray(L1[0].sum(axis=0)).ravel()/size
    aug=sparse.bmat([[L,None,sparse.csc_matrix(U[:,0,None])],[sparse.csc_matrix(2*avg[None,:]),sparse.csc_matrix((1,1)),sparse.csc_matrix([[bare[0,0]]])],[None,None,sparse.csc_matrix((1,1))]],format='csc')
    y=np.zeros(size+2);y[-1]=1
    T=8/D
    vals=expm_multiply(aug,y,start=0,stop=2*T,num=3,traceA=float(aug.diagonal().sum()))
    vars=vals[:,-2];slope=(vars[2]-vars[1])/(2*T)
    K,_=reference(field,D,n)
    return {'n':n,'times':[0,T,2*T],'displacement_variances':vars.tolist(),'late_interval_diffusivity':float(slope),'stationary_jump_diffusivity':float(K[0,0]),'relative_time_error':float(abs(slope-K[0,0])/K[0,0]),'initial_mean_phase':'uniform','monte_carlo':False}


def counting_check(field,D,n=32):
    L,_,_,_,(rows,cols,rates,delta)=jump_generator(field,D,n)
    results=[]
    for s in [.02,.01]:
        lam=[]
        for sign in [-1,1]:
            off=sparse.coo_matrix((rates*np.exp(sign*s*delta[:,0]),(rows,cols)),shape=L.shape).tocsc()
            Q=off+sparse.diags(L.diagonal())
            ev=eigs(Q,k=1,sigma=0,return_eigenvectors=False)[0]
            assert abs(ev.imag)<1e-9
            lam.append(float(ev.real))
        results.append({'s':s,'eigenvalues':lam,'diffusivity':sum(lam)/(2*s*s)})
    extrap=(4*results[1]['diffusivity']-results[0]['diffusivity'])/3
    K,_=reference(field,D,n)
    return {'rows':results,'extrapolated':extrap,'jump_reference':float(K[0,0]),'relative_error':float(abs(extrap-K[0,0])/K[0,0])}


def main():
    start=time.monotonic()
    fields=[]
    for a in [0,.1,.5,1,2,4,6,8]: fields.append({'name':f'equal_{a}','a':a,'b':a,'c':0.,'d':0.})
    fields += [dict(name='unequal_1_4',a=1.,b=4.,c=0.,d=0.),dict(name='unequal_4_1',a=4.,b=1.,c=0.,d=0.),dict(name='harmonic_equal',a=4.,b=4.,c=.6,d=-.4),dict(name='harmonic_unequal',a=6.,b=3.,c=-.7,d=.5)]
    rows=[]
    for field in fields:
        for D in [.8,1.,1.2]:
            K,checks=fourier(field,D,18);Kr,_=fourier(field,D,24);S=source(field,D)
            rows.append({'field':field,'diffusivity':D,'physical':K.tolist(),'source':S.tolist(),'relative_errors_per_axis':(abs(np.diag(S-K))/np.diag(K)).tolist(),'fourier_refinement':float(np.max(abs(K-Kr))),'checks':checks})
    refs=[]
    select=[rows[i] for i in [1,4,10,16,18,20,21,22,25,28,31,34]]
    for row in select:
        field=row['field'];D=row['diffusivity'];K=np.array(row['physical'])
        k48,c48=reference(field,D,48);k96,c96=reference(field,D,96);ext=(4*k96-k48)/3
        r={'field':field,'diffusivity':D,'grid48':k48.tolist(),'grid96':k96.tolist(),'extrapolated':ext.tolist(),'relative_error':float(np.max(abs(ext-K))/np.max(abs(K))),'checks48':c48,'checks96':c96}
        if field['name'] in ['equal_6','equal_8','harmonic_equal']:
            k192,_=reference(field,D,192);ex2=(4*k192-k96)/3
            r.update(grid192=k192.tolist(),refined_relative_error=float(np.max(abs(ex2-K))/np.max(abs(K))),relative_refinement_change=float(np.max(abs(ex2-ext))/np.max(abs(K))))
        refs.append(r)
    cal=[];calmax=0
    for D in np.linspace(.8,1.2,33):
        values=[]
        for a in [.6,1.2,2.0,3.0]:
            f=dict(a=a,b=0.,c=.25*a,d=0.)
            K,_=fourier(f,D,10);S=source(f,D);calmax=max(calmax,float(np.max(abs(K-S))))
            values.append(float(K[0,0]))
        C=np.array([a*a+(.25*a)**2/4 for a in [.6,1.2,2.0,3.0]])/2
        recovered=(C[-1]-C[0])/(values[-1]-values[0])
        cal.append({'D':float(D),'longitudinal_values':values,'recovered':float(recovered)})
    temporal=[finite_time(dict(a=a,b=a,c=0.,d=0.),D) for a,D in [(1.,1.),(4.,.8),(6.,1.2)]]
    count=counting_check(dict(a=4.,b=4.,c=0.,d=0.),1.)
    reversals=[]
    for f in [fields[5],fields[-1]]:
        k,_=fourier(f,1.)
        reverse={key:(-value if key!='name' else value+'_reversed') for key,value in f.items()}
        kr,_=fourier(reverse,1.)
        swapped=dict(name='axis_swapped',a=f['b'],b=f['a'],c=f['d'],d=f['c'])
        ks,_=fourier(swapped,1.)
        reversals.append({'field':f,'flow_reversal_error':float(np.max(abs(k-kr))),'axis_exchange_error':float(np.max(abs(k-ks[::-1,::-1])))})
    summary={'max_fourier_refinement':max(r['fourier_refinement'] for r in rows),'max_reference_relative_error':max(r['relative_error'] for r in refs),'max_refined_reference_relative_error':max(r.get('refined_relative_error',0) for r in refs),'calibration_exact_error':calmax,'max_parameter_recovery_error':max(abs(r['D']-r['recovered']) for r in cal),'max_finite_time_relative_error':max(r['relative_time_error'] for r in temporal),'min_physical_eigenvalue':min(r['checks']['minimum_eigenvalue'] for r in rows),'max_energy_identity_error':max(r['checks']['energy_identity_error'] for r in rows),'min_source_upper_bound_eigenvalue':min(r['checks']['source_upper_bound_min_eigenvalue'] for r in rows),'strong_equal_min_relative_gap':min(min(r['relative_errors_per_axis']) for r in rows if r['field']['name'] in ['equal_4','equal_6','equal_8']),'run_seconds':time.monotonic()-start}
    out={'status':'bounded_feasibility','model_runs':0,'field_cases':rows,'independent_jump_reference':refs,'calibration':cal,'finite_time_displacement_moments':temporal,'counting_eigenvalue_check':count,'symmetries':reversals,'summary':summary,'scope':'Fourier stationary generator versus positive spatial jump generator; finite-time exact moment propagation. No particle Monte Carlo, task harness, Docker or model calls. All tested fields, including weak gaps, retained.'}
    Path(__file__).with_name('report.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    assert summary['max_fourier_refinement']<2e-7
    assert summary['max_reference_relative_error']<2e-4
    assert summary['max_refined_reference_relative_error']<2e-5
    assert summary['calibration_exact_error']<1e-11
    assert summary['max_parameter_recovery_error']<1e-12
    assert summary['max_finite_time_relative_error']<5e-4
    assert summary['strong_equal_min_relative_gap']>.2
    assert count['relative_error']<1e-5

if __name__=='__main__':main()
