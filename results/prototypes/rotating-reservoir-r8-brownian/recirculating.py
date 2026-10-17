"""Read-only proposal: finite thermalized gas contact with elastic velocity swaps.

Independent positive finite-volume and normalized Hermite weak generators,
plus a completed conditional-Gaussian flux moment closure. No task imports.
"""
from functools import lru_cache
from math import erf
import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.laguerre import laggauss
from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_ivp
from scipy.linalg import solve
from scipy.sparse import csr_matrix, diags, eye, kron
from scipy.sparse.linalg import spsolve

RATE = 3.6


def bernoulli(x):
    x = np.asarray(x)
    out = np.empty_like(x)
    small = abs(x)<1e-5
    out[small]=1-x[small]/2+x[small]**2/12-x[small]**4/720
    out[~small]=x[~small]/np.expm1(x[~small])
    return out


def ou_grid(edges, drag, temperature, flow=0.):
    h=edges[1]-edges[0]
    force=-drag*(edges[1:-1]-flow)
    diffusion=drag*temperature
    up=diffusion/h**2*bernoulli(-force*h/diffusion)
    down=diffusion/h**2*bernoulli(force*h/diffusion)
    a=diags([up,down],[1,-1],shape=(len(edges)-1,len(edges)-1),format='csr')
    return a-diags(np.asarray(a.sum(axis=1)).ravel())


def finite_volume(e, gamma=.67, points=61, extent=7., details=False, noise=False):
    """Conditional +motor generator; -motor density is its velocity reflection."""
    ta=e['temperature_a'];tb=e['temperature_b'];omega=e['angular_speed']
    lam=e['relaxation_rate'];eta=e['flux_fraction'];flip=e['flip_rate']
    width=extent*np.sqrt(max(ta,tb))+abs(omega)
    edges=np.linspace(-width,width,points+1);x=(edges[1:]+edges[:-1])/2
    identity=eye(points,format='csr');n=points**2
    rows=np.arange(n);swap=rows.reshape(points,points).T.ravel();reverse=rows[::-1]
    permutation=csr_matrix((np.ones(n),(rows,swap)),shape=(n,n))
    reflection=csr_matrix((np.ones(n),(rows,reverse)),shape=(n,n))
    v,g=np.meshgrid(x,x,indexing='ij');v=v.ravel();g=g.ravel()
    av=kron(ou_grid(edges,gamma,ta),identity,format='csr')
    ag=kron(identity,ou_grid(edges,lam,tb,omega),format='csr')
    rate=RATE*((1-eta)+eta*abs(v-g))
    matrix=av+ag+diags(rate)@(permutation-eye(n,format='csr'))+flip*(reflection-eye(n,format='csr'))
    system=matrix.T.tolil();system[n//2,:]=np.ones(n);target=np.zeros(n);target[n//2]=1.
    p=spsolve(system.tocsc(),target)
    vmean=float(p@v);gmean=float(p@g);v2=float(p@(v*v));g2=float(p@(g*g));vg=float(p@(v*g))
    mean=lam*(g2-2*omega*gmean+omega**2-tb)
    result=dict(mean=mean,vmean=vmean,gmean=gmean,v2=v2,g2=g2,vg=vg,
                energy_covariance=float(.25*(p@(v*v*g*g)-v2*g2)),
                bath_a=gamma*(v2-ta),motor=lam*(omega**2-omega*gmean),
                minimum_probability=float(p.min()),normalization_error=float(abs(p.sum()-1)),
                stationary_residual=float(max(abs(matrix.T@p))))
    if details:
        qa=-float(p@(av@(.5*v*v)));qb=-float(p@(ag@(.5*(g-omega)**2)))
        power=float(p@(omega*(ag@g)))
        result.update(discrete_bath_a=qa,discrete_bath_b=qb,discrete_motor=power,
                      discrete_first_law_residual=abs(qa+qb-power),
                      continuum_first_law_residual=abs(result['bath_a']+mean-result['motor']),
                      edge_probability=float(p.reshape(points,points)[[0,-1],:].sum()+p.reshape(points,points)[:,[0,-1]].sum()))
    if noise:
        agcoo=ag.tocoo();energy=.5*(g-omega)**2
        marks=energy[agcoo.row]-energy[agcoo.col]
        first=csr_matrix((agcoo.data*marks,(agcoo.row,agcoo.col)),shape=(n,n))
        second=csr_matrix((agcoo.data*marks**2,(agcoo.row,agcoo.col)),shape=(n,n))
        drift=np.asarray(first.sum(axis=1)).ravel();mean_grid=float(p@drift)
        poisson=matrix.tolil();pivot=int(np.argmax(p));poisson[pivot,:]=p
        rhs=mean_grid-drift;rhs[pivot]=0.
        phi=spsolve(poisson.tocsc(),rhs)
        shot=float(p@np.asarray(second.sum(axis=1)).ravel())
        correlation=float(2*p@(first@phi))
        result.update(noise=shot+correlation,noise_shot=shot,noise_correlation=correlation,
                      noise_poisson_residual=float(max(abs(matrix@phi-(mean_grid-drift)))))
    return result


def normalized_hermites(z, degree):
    z=np.asarray(z);h=np.zeros((z.size,degree+1));h[:,0]=1.
    if degree:h[:,1]=z
    for n in range(1,degree):h[:,n+1]=(z*h[:,n]-np.sqrt(n)*h[:,n-1])/np.sqrt(n+1)
    return h


@lru_cache(16)
def collision_weak_matrix(degree, variance=1.6, quadrature=None):
    """Exact Gaussian polynomial integral for the nonsmooth collision rate.

    Rotate to z=(v-g)/sqrt(2),t=(v+g)/sqrt(2). Gauss-Laguerre integrates
    |z| times its Gaussian exactly after z=+/-sqrt(2*variance*y).
    Summing both signs cancels odd z powers; Hermite integrates the t part.
    """
    q=quadrature or (2*degree+4)
    th,tw=hermgauss(q);yl,yw=laggauss(q)
    t=np.sqrt(2*variance)*th[None,:]
    blocks=[];swapped=[];weights=[]
    for sign in [-1.,1.]:
        z=sign*np.sqrt(2*variance*yl[:,None])
        v=(t+z)/np.sqrt(2);g=(t-z)/np.sqrt(2)
        hv=normalized_hermites(v.ravel()/np.sqrt(variance),degree)
        hg=normalized_hermites(g.ravel()/np.sqrt(variance),degree)
        blocks.append((hv[:,:,None]*hg[:,None,:]).reshape(q*q,-1))
        swapped.append((hg[:,:,None]*hv[:,None,:]).reshape(q*q,-1))
        weights.append((np.sqrt(variance)/np.pi*yw[:,None]*tw[None,:]).ravel())
    b=np.concatenate(blocks);jump=np.concatenate(swapped)-b;w=np.concatenate(weights)
    matrix=b.T@(w[:,None]*jump)
    # The reversible swap generator is symmetric and dissipative under the
    # isotropic Gaussian integration weight, including its |v-g| rate.
    assert np.max(abs(matrix-matrix.T))<1e-9
    return matrix


def hermite(e, gamma=.67, degree=12, variance=1.6, quadrature=None, total_degree=False, noise=False):
    """Independent polynomial weak generator; no FV or Gaussian closure calls."""
    ta=e['temperature_a'];tb=e['temperature_b'];omega=e['angular_speed']
    lam=e['relaxation_rate'];eta=e['flux_fraction'];flip=e['flip_rate']
    width=degree+1;size=width**2;matrix=np.zeros((size,size))
    index=lambda n,m:n*width+m
    for n in range(width):
        for m in range(width):
            j=index(n,m)
            matrix[j,j]-=gamma*n+lam*m+flip*(1-(-1)**(n+m))+RATE*(1-eta)
            matrix[index(m,n),j]+=RATE*(1-eta)
            if n>=2:matrix[index(n-2,m),j]+=gamma*(ta/variance-1)*np.sqrt(n*(n-1))
            if m>=2:matrix[index(n,m-2),j]+=lam*(tb/variance-1)*np.sqrt(m*(m-1))
            if m>=1:matrix[index(n,m-1),j]+=lam*omega/np.sqrt(variance)*np.sqrt(m)
    matrix+=RATE*eta*collision_weak_matrix(degree,variance,quadrature)
    retained=np.array([index(n,m) for n in range(width) for m in range(width) if not total_degree or n+m<=degree])
    lookup={j:i for i,j in enumerate(retained)}
    matrix=matrix[np.ix_(retained,retained)];size=len(retained)
    index_full=index
    index=lambda n,m:lookup[index_full(n,m)]
    forward=matrix.T.copy();forward[0]=0.;forward[0,0]=1.;rhs=np.zeros(size);rhs[0]=1.
    moments=solve(forward,rhs,check_finite=False)
    mv=np.sqrt(variance)*moments[index(1,0)];mg=np.sqrt(variance)*moments[index(0,1)]
    v2=variance*(1+np.sqrt(2)*moments[index(2,0)]);g2=variance*(1+np.sqrt(2)*moments[index(0,2)])
    vg=variance*moments[index(1,1)]
    mean=lam*(g2-2*omega*mg+omega*omega-tb)
    v2g2=variance**2*(1+np.sqrt(2)*(moments[index(2,0)]+moments[index(0,2)])+2*moments[index(2,2)])
    result=dict(mean=float(mean),vmean=float(mv),gmean=float(mg),v2=float(v2),g2=float(g2),vg=float(vg),
                energy_covariance=float(.25*(v2g2-v2*g2)),
                bath_a=float(gamma*(v2-ta)),motor=float(lam*(omega*omega-omega*mg)),
                first_law_residual=float(abs(gamma*(v2-ta)+mean-lam*(omega*omega-omega*mg))),
                stationary_residual=float(max(abs(matrix.T@moments))))
    if noise:
        # Independent continuous diffusion Feynman--Kac derivatives. No
        # finite-volume heat marks, stochastic paths or shot-noise closure.
        # dQ=lam[(g-u)^2-Tb]dt-sqrt(2lamTb)(g-u)dW_b.
        derivative=np.zeros((width**2,width**2));multiply=np.zeros_like(derivative);square=np.zeros_like(derivative)
        for n in range(width):
            for m in range(width):
                j=index_full(n,m)
                square[j,j]=variance*(2*m+1)
                if m>=1:
                    derivative[index_full(n,m-1),j]=np.sqrt(m/variance)
                    multiply[index_full(n,m-1),j]=np.sqrt(variance*m)
                if m+1<width:multiply[index_full(n,m+1),j]=np.sqrt(variance*(m+1))
                if m>=2:square[index_full(n,m-2),j]=variance*np.sqrt(m*(m-1))
                if m+2<width:square[index_full(n,m+2),j]=variance*np.sqrt((m+1)*(m+2))
        identity=np.eye(width**2);relative_square=square-2*omega*multiply+omega**2*identity
        first=2*lam*tb*(multiply-omega*identity)@derivative-lam*(relative_square-tb*identity)
        second=2*lam*tb*relative_square
        first=first[np.ix_(retained,retained)];second=second[np.ix_(retained,retained)]
        drift=-first[:,0];constant=np.zeros(size);constant[0]=1.
        tilted_mean=float(moments@first[:,0]);system=matrix.copy();system[0]=moments
        rhs=tilted_mean*constant-first[:,0];rhs[0]=0.
        r1=solve(system,rhs,check_finite=False)
        shot=float(moments@second[:,0]);correlation=float(2*moments@first@r1)
        result.update(noise=shot+correlation,noise_shot=shot,noise_correlation=correlation,
                      noise_poisson_residual=float(max(abs(matrix@r1-(tilted_mean*constant-first[:,0])))))
    return result


def folded(mu, variance):
    sigma=np.sqrt(max(variance,1e-14));gaussian=np.sqrt(2/np.pi)*np.exp(-mu*mu/(2*sigma*sigma))
    sign=erf(mu/(np.sqrt(2)*sigma))
    absolute=sigma*gaussian+mu*sign
    signed_square=(mu*mu+sigma*sigma)*sign+mu*sigma*gaussian
    absolute_cube=(mu**3+3*mu*sigma*sigma)*sign+(mu*mu+2*sigma*sigma)*sigma*gaussian
    return absolute,signed_square,absolute_cube


def gaussian_closure(e, gamma=.67, noise=False):
    """All exact five first/second equations, Gaussian assumption only in flux.

    The conditional covariance is retained; the gas is not refreshed at
    collisions, and finite gas relaxation and rotor backaction remain.
    At eta=0 the Gaussian assumption disappears from these equations.
    """
    ta=e['temperature_a'];tb=e['temperature_b'];omega=e['angular_speed']
    lam=e['relaxation_rate'];eta=e['flux_fraction'];flip=e['flip_rate']
    def derivative(t,y):
        mv,mg,v2,g2,vg=y;mu=mv-mg;total=mv+mg
        vv=v2-mv*mv;gg=g2-mg*mg;cov=vg-mv*mg;variance=vv+gg-2*cov
        absolute,signed_square,absolute_cube=folded(mu,variance)
        flux_mean=RATE*((1-eta)*mu+eta*signed_square)
        mixed=total*signed_square+(vv-gg)/max(variance,1e-14)*(absolute_cube-mu*signed_square)
        exchange=RATE*((1-eta)*(g2-v2)-eta*mixed)
        return np.array([-gamma*mv-flux_mean-2*flip*mv,
            -lam*(mg-omega)+flux_mean-2*flip*mg,
            -2*gamma*(v2-ta)+exchange,
            -2*lam*(g2-omega*mg-tb)-exchange,
            -(gamma+lam)*vg+lam*omega*mv])
    initial=[0.,0.,ta,tb,0.]
    solution=solve_ivp(derivative,[0,200],initial,method='BDF',rtol=2e-10,atol=2e-12)
    assert solution.success
    mv,mg,v2,g2,vg=solution.y[:,-1]
    covariance=np.array([[v2-mv*mv,vg-mv*mg],[vg-mv*mg,g2-mg*mg]])
    mean=lam*(g2-2*omega*mg+omega**2-tb)
    cov=vg-mv*mg
    result=dict(mean=float(mean),vmean=float(mv),gmean=float(mg),v2=float(v2),g2=float(g2),vg=float(vg),
                energy_covariance=float(mv*mg*cov+.5*cov**2),
                minimum_covariance=float(np.linalg.eigvalsh(covariance).min()),
                derivative_residual=float(max(abs(derivative(200,solution.y[:,-1])))),
                first_law_residual=float(abs(gamma*(v2-ta)+mean-lam*(omega**2-omega*mg))))
    if noise:
        # Conditional-Gaussian quadratic response. Project the exact kinetic
        # generator in this assumed Gaussian law, then retain its quadratic
        # Poisson response. This is an approximation even at eta=0 for heat
        # noise: eta=0 calibrates mean heat, not the nonGaussian joint law.
        vv=v2-mv*mv;gg=g2-mg*mg;cds=vv-gg
        vd=vv+gg-2*cov;vs=vv+gg+2*cov
        md=mv-mg;ms=mv+mg;conditional=max(vs-cds*cds/vd,0.)
        nodes,weights=leggauss(80);gh,gw=hermgauss(5)
        ds=[];ws=[]
        limits=[md-10*np.sqrt(vd),0.,md+10*np.sqrt(vd)]
        for left,right in zip(limits[:-1],limits[1:]):
            d=(right-left)/2*nodes+(right+left)/2
            w=(right-left)/2*weights*np.exp(-(d-md)**2/(2*vd))/np.sqrt(2*np.pi*vd)
            ds.append(d);ws.append(w)
        d=np.concatenate(ds)[:,None];w=np.concatenate(ws)[:,None]*gw[None,:]/np.sqrt(np.pi)
        s=ms+cds/vd*(d-md)+np.sqrt(2*conditional)*gh[None,:]
        v=(s+d)/2;g=(s-d)/2;v=v.ravel();g=g.ravel();w=w.ravel()
        basis=lambda v,g:np.array([np.ones_like(v),v,g,v*v,v*g,g*g]).T
        b=basis(v,g);swapped=basis(g,v)
        local=np.array([np.zeros_like(v),-(gamma+2*flip)*v,
            -lam*(g-omega)-2*flip*g,-2*gamma*(v*v-ta),
            -(gamma+lam)*v*g+lam*omega*v,-2*lam*(g*g-omega*g-tb)]).T
        local+=RATE*((1-eta)+eta*abs(v-g))[:,None]*(swapped-b)
        gram=b.T@(w[:,None]*b);generator=solve(gram,b.T@(w[:,None]*local))
        moments=w@b
        a=np.array([lam*(omega**2-tb),0.,-2*lam*omega,0.,0.,lam])
        system=generator.copy();system[0]=moments;constant=np.zeros(6);constant[0]=1.
        rhs=mean*constant-a;rhs[0]=0.;phi=solve(system,rhs)
        gradient=phi[2]+phi[4]*v+2*phi[5]*g
        shot=2*lam*tb*float(w@((g-omega)**2))
        correlation=2*float(a@gram@phi-2*lam*tb*(w@((g-omega)*gradient)))
        result.update(noise=shot+correlation,noise_shot=shot,noise_correlation=correlation,
                      projected_stationary_residual=float(max(abs(moments@generator))),
                      poisson_residual=float(max(abs(generator@phi-(mean*constant-a)))),
                      projected_nonconstant_decay=float(min(-value.real for value in np.linalg.eigvals(generator) if abs(value)>1e-8)))
    return result


def calibration_mean(e, gamma=.67):
    lam=e['relaxation_rate']
    return gamma*lam*RATE/(2*gamma*lam+RATE*(gamma+lam))*(e['temperature_a']-e['temperature_b'])


def constant_clock_anchor(e, gamma=.67):
    """Independent closed raw-moment anchor, restricted to zero circulation."""
    assert e['angular_speed']==0. and e['flux_fraction']==0.
    ta=e['temperature_a'];tb=e['temperature_b'];lam=e['relaxation_rate'];nu=RATE
    x,y=np.linalg.solve([[2*gamma+nu,-nu],[-nu,2*lam+nu]],[2*gamma*ta,2*lam*tb])
    av4,bg4=np.linalg.solve([[4*gamma+nu,-nu],[-nu,4*lam+nu]],[12*gamma*ta*x,12*lam*tb*y])
    cross=(gamma*ta*y+lam*tb*x)/(gamma+lam)
    denominator=4*gamma*lam+2*nu*(gamma+lam)
    pv=lam*nu/denominator;pg=lam*(2*gamma+nu)/denominator
    mean=lam*(y-tb)
    noise=2*lam*tb*y+2*lam*(pv*(cross-x*y)+pg*(bg4-y*y))-8*lam*tb*pg*y
    gaussian_noise=2*lam*tb*y+4*lam*pg*y*y-8*lam*tb*pg*y
    return dict(mean=float(mean),noise=float(noise),gaussian_noise=float(gaussian_noise),
                v2=float(x),g2=float(y),v4=float(av4),g4=float(bg4),v2g2=float(cross),
                energy_covariance=float(.25*(cross-x*y)),
                negative_energy_correlation_identity=float(-nu*(x-y)**2/(8*(gamma+lam))))
