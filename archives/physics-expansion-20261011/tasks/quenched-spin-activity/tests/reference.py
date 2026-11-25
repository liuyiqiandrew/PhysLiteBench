"""Finite-population symmetric jump generator, extrapolated in inverse size."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

TRUE_PARAMETER = 1.13


def finite(e,N,kappa=1.):
    n=N//2; dim=(n+1)**2
    x,y=np.meshgrid(np.arange(n+1),np.arange(n+1),indexing='ij');mag=(x+y-n)/n
    up=[];down=[]
    for pop,field in zip([x,y],[e['field']+e['disorder'],e['field']-e['disorder']]):
        up.append((n-pop)*kappa/2*(1+np.tanh(e['coupling']*(mag+1/N)+field)))
        down.append(pop*kappa/2*(1-np.tanh(e['coupling']*(mag-1/N)+field)))
    diagonal=-(up[0]+up[1]+down[0]+down[1]).ravel()/N
    ids=np.arange(dim).reshape(n+1,n+1)
    rows=[ids[:-1,:].ravel(),ids[:,:-1].ravel()]
    cols=[ids[1:,:].ravel(),ids[:,1:].ravel()]
    val=[np.exp(-e['bias_plus'])*np.sqrt(up[0][:-1,:]*down[0][1:,:]).ravel()/N,np.exp(-e['bias_minus'])*np.sqrt(up[1][:,:-1]*down[1][:,1:]).ravel()/N]
    a=np.concatenate(rows);b=np.concatenate(cols);v=np.concatenate(val)
    matrix=coo_matrix((np.concatenate([diagonal,v,v]),(np.concatenate([np.arange(dim),a,b]),np.concatenate([np.arange(dim),b,a]))),shape=(dim,dim)).tocsr()
    return float(eigsh(matrix,k=1,which='LA',tol=1e-11,v0=np.ones(dim),return_eigenvectors=False)[0])


def predict(experiments, attempt_rate, base_size=80):
    values=[]
    for e in experiments:
        sizes=np.array([base_size,2*base_size,4*base_size])
        finite_values=[finite(e,int(n),attempt_rate) for n in sizes]
        values.append(float(np.polynomial.polynomial.polyfit(1/sizes,finite_values,2)[0]))
    return np.asarray(values)


def experiment(**changes):
    row=dict(coupling=1.2,field=.15,disorder=.5,bias_plus=.6,bias_minus=1.)
    row.update(changes)
    return row


def calibration_inputs():
    return [experiment(coupling=0.,field=h,disorder=d,bias_plus=a,bias_minus=b)
            for h in [0.,.25] for d in [.2,.6,1.] for a,b in [(.4,.7),(-.4,.5),(.8,-.3)]]


def hidden_inputs():
    return {
        'uncoupled_anchors':[experiment(coupling=0.,field=h,disorder=.4,bias_plus=a,bias_minus=.9)
                             for h in [-.2,.4] for a in [-.7,.3,.9]],
        'quiet_populations':[experiment(coupling=j,disorder=d,bias_plus=.7,bias_minus=.9)
                              for j in [.7,1.1,1.4] for d in [.3,.7]],
        'opposed_biases':[experiment(coupling=j,field=.1,disorder=d,bias_plus=a,bias_minus=.8)
                          for j in [1.1,1.4] for d in [.5,.8] for a in [-.6,-.9]],
        'field_response':[experiment(coupling=1.2,disorder=.6,field=h,bias_plus=.5,bias_minus=1.2)
                          for h in [-.4,-.15,.15,.4]],
        'moderate_coupling':[experiment(coupling=j,disorder=.5,field=.2,bias_plus=.8,bias_minus=.3)
                             for j in [.3,.5,.8]],
    }

