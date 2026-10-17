"""Conservative finite-volume momentum and electrical-current balances."""
from functools import lru_cache
import numpy as np
from scipy.sparse import coo_matrix, diags, bmat, eye
from scipy.sparse.linalg import splu

PARAMETER = 'viscosity'
TRUE_PARAMETER = .01
BOUNDS = (.006,.018)


@lru_cache(16)
def operators(nx,ny):
    dx,dy=.012/nx,.008/ny
    cells=nx*ny
    gr,gc,gd,cr,cc,cd,yr,yc,yd=[],[],[],[],[],[],[],[],[]
    row=0
    for j in range(ny):
        for i in range(nx-1):
            a=j*nx+i;b=a+1
            gr.extend([row,row]);gc.extend([a,b]);gd.extend([-1/dx,1/dx])
            cr.extend([row,row]);cc.extend([a,b]);cd.extend([.5,.5]);row+=1
    for j in range(ny-1):
        for i in range(nx):
            a=j*nx+i;b=a+nx
            gr.extend([row,row]);gc.extend([a,b]);gd.extend([-1/dy,1/dy])
            yr.extend([row,row]);yc.extend([a,b]);yd.extend([.5,.5]);row+=1
    gradient=coo_matrix((gd,(gr,gc)),shape=(row,cells)).tocsc()
    motional_x=coo_matrix((cd,(cr,cc)),shape=(row,cells)).tocsc()
    motional_y=coo_matrix((yd,(yr,yc)),shape=(row,cells)).tocsc()
    boundary=np.zeros((ny,nx))
    boundary[:,0]+=2/dx**2;boundary[:,-1]+=2/dx**2
    boundary[0,:]+=2/dy**2;boundary[-1,:]+=2/dy**2
    viscous=gradient.T@gradient+diags(boundary.ravel())
    # Pin one potential; the omitted charge equation follows from global balance.
    electric_gradient=gradient[:,1:]
    return viscous,motional_x,motional_y,electric_gradient


@lru_cache(12)
def trajectory(viscosity,bx,by,nx=48,ny=32,steps=2400):
    viscous,cx,cy,g=operators(nx,ny)
    c=by*cx-bx*cy
    cells=nx*ny;dt=12./steps
    electric=g.T@g
    velocity=viscosity/1000.*viscous+100.*(c.T@c)
    coupling=100.*(c.T@g)
    mass=eye(cells,format='csc')
    left=bmat([[mass+dt/2*velocity,dt/2*coupling],[g.T@c,electric]],format='csc')
    right=bmat([[mass-dt/2*velocity,-dt/2*coupling],[None,None]],format='csc')
    factor=splu(left)
    old=np.zeros(2*cells-1)
    history=[old[:cells].copy()]
    forcing=np.r_[np.full(cells,dt/1000.),np.zeros(cells-1)]
    # The lower block is always the exact instantaneous charge constraint.
    top=right[:cells]
    for _ in range(steps):
        rhs=forcing.copy();rhs[:cells]+=top@old
        old=factor.solve(rhs)
        history.append(old[:cells].copy())
    return np.array(history)


def finite_volume_predict(experiments,viscosity,nx=96,ny=64,steps=2400):
    out=[]
    modes=np.arange(1,256,2,dtype=float)
    rates=viscosity/1000*((modes[:,None]*np.pi/.012)**2+(modes[None,:]*np.pi/.008)**2)
    mean=2*np.sqrt(2)/(np.pi*modes)
    for e in experiments:
        if e['magnetic_x']==0 and e['magnetic_y']==0:
            amplitude=-np.expm1(-rates*e['time'])/rates*mean[:,None]*mean[None,:]
            weight=mean[:,None]*mean[None,:] if e['observable']=='mean' else 2*np.sin(modes[:,None]*np.pi*e['x'])*np.sin(modes[None,:]*np.pi*e['y'])
            out.append(e['pressure_gradient']/1000*np.sum(weight*amplitude))
            continue
        history=trajectory(viscosity,e['magnetic_x'],e['magnetic_y'],nx,ny,steps)
        index=e['time']/12*steps;i=min(int(index),steps-1);f=index-i
        u=((1-f)*history[i]+f*history[i+1]).reshape(ny,nx)
        if e['observable']=='mean':value=u.mean()
        else:
            x=e['x']*nx-.5;y=e['y']*ny-.5
            a=int(x);b=int(y);fx=x-a;fy=y-b
            value=(1-fx)*(1-fy)*u[b,a]+fx*(1-fy)*u[b,a+1]+(1-fx)*fy*u[b+1,a]+fx*fy*u[b+1,a+1]
        out.append(e['pressure_gradient']*value)
    return np.array(out)


def predict(experiments,viscosity,nx=96,ny=64,steps=2400):
    fine=finite_volume_predict(experiments,viscosity,nx,ny,steps)
    coarse=finite_volume_predict(experiments,viscosity,nx//2,ny//2,steps)
    return (4*fine-coarse)/3


def calibration_inputs():
    return [dict(magnetic_x=bx,magnetic_y=by,pressure_gradient=g,time=float(t),observable=obs,x=.38,y=.42)
            for bx,by in [(0.,.3),(.3,0.)] for g in [.06,.15]
            for obs in ['mean','point'] for t in np.geomspace(.08,7.,10)]


def hidden_inputs():
    return {name:[dict(magnetic_x=bx,magnetic_y=by,pressure_gradient=.12,time=float(t),observable=obs,x=x,y=y)
            for obs,x,y in [('mean',.5,.5),('point',.23,.31),('point',.23,.69),('point',.72,.27)]
            for t in np.geomspace(.12,10.,14)]
            for name,bx,by in [('oblique_positive',.45,.45),('oblique_negative',.45,-.45),('unequal_components',.6,.2)]}
