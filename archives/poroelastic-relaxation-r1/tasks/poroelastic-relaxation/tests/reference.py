import itertools
import numpy as np

TRUE_PARAMETER=.0011
C=np.array([[6.,2.,.8],[2.,4.,.6],[.8,.6,3.]])*1e6
Q=np.array([1.,1.,0.]);M=20e6;A=.9
K=np.array([[1.,.2],[.2,.6]])*1e-15


def experiment(mode=(1,1),time=.2,initial_pressure=2000.):
    return dict(mode=list(mode),time=time,initial_pressure=initial_pressure)


def fluid_stiffness(e):
    k=2*np.pi*np.array(e['mode'])/.01
    if np.linalg.norm(k)==0:
        b=np.eye(3)
    else:
        b=np.array([[k[0],0.],[0.,k[1]],[k[1],k[0]]])
    d=b.T@Q
    # Minimize the supplied energy at fixed unit fluid content.
    coupled=b.T@C@b+M*A*A*np.outer(d,d)
    u=np.linalg.solve(coupled,M*A*d)
    strain=b@u
    pressure=M*(1-A*Q@strain)
    stored=.5*strain@C@strain+.5*M*(1-A*Q@strain)**2
    return float(pressure),float(stored),strain


def predict(experiments,viscosity=TRUE_PARAMETER):
    result=[]
    for e in experiments:
        pressure,energy,strain=fluid_stiffness(e)
        k=2*np.pi*np.array(e['mode'])/.01
        drain=(k@K@k+1e-10)/viscosity
        initial_content=e['initial_pressure']/pressure
        content=initial_content*np.exp(-drain*pressure*e['time'])
        result.append(pressure*content)
    return np.asarray(result)


def calibration_inputs():
    return [experiment((0,0),t,a) for t,a,repeat in itertools.product([.25,.5,.8,1.2,1.8,2.6],[1200.,1800.,2400.],range(12))]


def hidden_inputs():
    return {
        'axial_pressure': [experiment((1,0),t,a) for t,a in [(.2,1800.),(.35,2400.),(.5,1500.),(.7,2200.)]],
        'oblique_pressure': [experiment(mode,t,a) for mode,t,a in [((1,1),.18,2400.),((1,-1),.3,1800.),((2,1),.12,2600.),((-1,2),.25,1600.)]],
        'relaxation': [experiment((1,1),t,2000.) for t in [.1,.2,.3,.4]],
        'uniform_anchors': [experiment((0,0),.4,1500.),experiment((0,0),2.,2500.)]
    }
