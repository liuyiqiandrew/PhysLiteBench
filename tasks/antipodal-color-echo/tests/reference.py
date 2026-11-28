import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.special import roots_legendre

PARAMETER="gain"
TRUE_PARAMETER=1.062
CALIBRATION_SEED=28313839
NOISE_FRACTION=.004
S2P=np.sqrt(2*np.pi)
R,A=roots_legendre(30)


def box(bounds):
    lo,hi=bounds
    return (hi+lo)/2+(hi-lo)*R/2,(hi-lo)*A/2


def normal(x,scale):
    return np.exp(-.5*(x/scale)**2)/(S2P*scale)


def evaluate(e,gain):
    if e["observable"]=="coil":
        return gain*e["current"]*e["turns"]
    vf,vl=e["variances"]
    s,t=gain*np.sqrt([vf,vl])
    c=np.sqrt(2*e["ratio"])/gain
    k=c*(vf+vl)/vl
    z=[]
    for q,leak,offset in zip(e["windows"],e["leakage"],e["offsets"]):
        scale=np.exp(.5*q*q*(s*s+t*t))
        reduced=quad(lambda x:normal(x+q*s*s,s)*ndtr((x-q*t*t)/t),
                     0,np.inf,epsabs=4e-12)[0]*scale
        z.append(reduced*(1+np.exp(-c*offset) if leak==0 else 1))
    bounds=e["root_band"]
    slab=e["observable"]=="slab_echo"
    if slab:
        lo=max(bounds[0],-.5*e["leakage"][0])
        hi=min(bounds[1],.5*e["leakage"][1])
        if hi<=lo:
            return 0.
        bounds=lo,hi
    u,wu=box(bounds)
    p,wp=box(e["line_windows"][0])
    q,wq=box(e["line_windows"][1])
    U=u[:,None,None]
    P=p[None,:,None]
    Q=q[None,None,:]
    m0=np.exp(-c*P)
    m1=np.exp(-c*Q)
    d0=z[0]+m0
    d1=z[1]+m1
    if not slab:
        d0=d0+np.exp(-c*(Q+2*U+e["offsets"][0]))
        d1=d1+np.exp(-c*(P-2*U+e["offsets"][1]))
    values=(m0/d0)*(m1/d1)*np.exp(k*(P+Q))
    j=e["outside_setting"]
    if j!=-1:
        values=values*z[j]/(d0 if j==0 else d1)
    root=normal(u,s)*np.exp(-(u/t)**2) if slab else normal(0,s)
    return float(np.sum(wu*root*(values@wq@wp))/(2*np.pi*t*t))


def predict(experiments,gain):
    return np.array([evaluate(e,gain) for e in experiments],dtype=float)


def calibration_inputs():
    return [dict(observable="coil",current=float(x),turns=.5+(i%17)/8)
            for i,x in enumerate(np.linspace(-2.1,2.2,112))]


def case(slab=False,**kw):
    values=dict(observable="slab_echo" if slab else "balanced_echo",ratio=.31,
                variances=[.82,.67],windows=[-.7,.4],line_windows=[[-.7,.15],[-.5,.38]],
                root_band=[-1.,.8],leakage=[.55,.85] if slab else [0.,0.],offsets=[.15,.39],
                outside_setting=-1)
    values.update(kw)
    return values


def hidden_inputs():
    rng=np.random.default_rng(531807)
    groups={"finite_crossed_lines":[case(),case(outside_setting=1),case(offsets=[1.5,0]),
                                  case(root_band=[.8,1.5]),case(root_band=[-1.5,-.8])],
            "leakage_selected_slab":[case(True),case(True,leakage=[0,1.2]),
                                   case(True,leakage=[1.4,0]),case(True,root_band=[.8,1.1])],
            "counterfamily_marks":[case(outside_setting=0),case(True,outside_setting=1)],
            "symmetry_and_calibration":[case(),case(windows=[.4,-.7],root_band=[-.8,1.],
                   line_windows=[[-.5,.38],[-.7,.15]],offsets=[.39,.15])]}
    for i in range(10):
        extra=dict(ratio=float(rng.uniform(.24,.42)),variances=rng.uniform(.5,1.22,2).tolist(),
                   windows=rng.uniform(-1.1,1.06,2).tolist(),
                   line_windows=[[float(l),float(h)] for l,h in rng.uniform(-1.3,-.3,2)[:,None]
                                 +np.array([0.,1.05])],
                   root_band=[-.89-i*.055,.6+.063*i],offsets=rng.uniform(.05,1.6,2).tolist())
        groups["finite_crossed_lines"].append(case(**extra))
        leak=[0.,1.36] if i%3==0 else [1.11,0.] if i%3==1 else [.62,.9]
        groups["leakage_selected_slab"].append(case(True,leakage=leak,**extra))
        groups["counterfamily_marks"].append(case(i%2==0,outside_setting=i%2,**extra))
    groups["symmetry_and_calibration"]+=calibration_inputs()[::12]
    return groups


def checks():
    first,other=hidden_inputs()["symmetry_and_calibration"][:2]
    pair=predict([first,other],TRUE_PARAMETER)
    assert abs(pair[0]-pair[1])<1e-13
    null=evaluate(case(True,leakage=[0,1.2],root_band=[-1,-.2]),TRUE_PARAMETER)
    assert null==0
    return dict(exchange=pair.tolist(),outside_slab=null)
