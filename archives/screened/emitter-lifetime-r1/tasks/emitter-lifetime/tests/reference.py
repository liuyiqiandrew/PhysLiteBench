"""Independent outgoing-flux and material-heating calculation."""
from functools import lru_cache
import numpy as np
from scipy.integrate import quad

TRUE_PARAMETER=.8


def experiment(height,real,imag,tilt,time):
    return dict(height=float(height),epsilon_real=float(real),epsilon_imag=float(imag),tilt=float(tilt),time=float(time))


def calibration_inputs():
    return [experiment(d,eps,0.,angle,t) for eps in [.35,.5,.7,1.] for d in [.25,.7,1.3] for angle in [0.,.65,1.3] for t in [.3,.8,1.4,2.2]]


def hidden_inputs():
    cases={
        'dielectric':[(.25,2.2,.6,0.),(.45,3.,1.1,.4),(.8,1.5,.4,.9)],
        'absorbing_metal':[(.3,-2.,.8,.2),(.5,-3.,1.2,.6),(.9,-1.4,1.5,1.2)],
        'orientation':[(.32,2.7,.9,angle) for angle in [0.,.7,np.pi/2]]}
    return {name:[experiment(*case,t) for case in group for t in [.04,.12,.35,.8]] for name,group in cases.items()}


def angular_power(u,height,epsilon,vertical):
    """Azimuth-averaged upward flux and depth-integrated absorption density."""
    w=np.sqrt(1-u*u+0j);q=np.sqrt(epsilon-u*u+0j)
    rs=(w-q)/(w+q);rp=(epsilon*w-q)/(epsilon*w+q)
    dipole=np.array([0.,0.,1.]) if vertical else np.array([1.,0.,0.])
    upward=material=0.
    # Quadratic polarization intensities have no azimuthal harmonics above order two.
    for angle in [0.,np.pi/2,np.pi,3*np.pi/2]:
        c,s=np.cos(angle),np.sin(angle)
        te=np.array([-s,c,0.]);tm_down=np.array([w*c,w*s,u]);tm_up=np.array([-w*c,-w*s,u])
        tm_inside=np.array([q*c,q*s,u])/np.sqrt(epsilon)
        incident=np.exp(1j*w*height)*1j/(2*w)
        transmitted=incident*((1+rs)*te*(te@dipole)+2*np.sqrt(epsilon)*w/(epsilon*w+q)*tm_inside*(tm_down@dipole))
        if epsilon.imag>0:
            # Integrating exp(2 Im(q) z) over z<0 gives 1/(2 Im(q)).
            material+=1.5*epsilon.imag/q.imag*float(np.vdot(transmitted,transmitted).real)/4
        else:
            magnetic=np.cross(np.array([u*c,u*s,-q]),transmitted)
            material+=-3*float(np.cross(transmitted,magnetic.conj())[2].real)/4
        if u<1:
            outgoing=1j/(2*w)*((te*(te@dipole)+tm_up*(tm_up@dipole))*np.exp(-1j*w*height)+(rs*te*(te@dipole)+rp*tm_up*(tm_down@dipole))*np.exp(1j*w*height))
            upward+=3*w.real*float(np.vdot(outgoing,outgoing).real)/4
    return upward,material


@lru_cache(maxsize=1024)
def power_components(height,real,imag,vertical,tolerance=2e-10):
    epsilon=complex(real,imag)
    if epsilon==1:return .5,.5
    def propagating(theta,index):
        u=np.sin(theta)
        return angular_power(u,height,epsilon,vertical)[index]*u*np.cos(theta)
    points=[float(np.arcsin(np.sqrt(real)))] if imag==0 and real<1 else None
    upward=quad(lambda theta:propagating(theta,0),0.,np.pi/2,points=points,epsabs=tolerance,epsrel=tolerance)[0]
    material=quad(lambda theta:propagating(theta,1),0.,np.pi/2,points=points,epsabs=tolerance,epsrel=tolerance)[0]
    if imag>0:
        material+=quad(lambda v:angular_power(np.sqrt(1+v*v),height,epsilon,vertical)[1]*v,0.,40/height,epsabs=tolerance,epsrel=tolerance,limit=200)[0]
    return upward,material


def factor(e,tolerance=2e-10):
    components=np.array([sum(power_components(e['height'],e['epsilon_real'],e['epsilon_imag'],vertical,tolerance)) for vertical in [True,False]])
    return float(components@np.array([np.cos(e['tilt'])**2,np.sin(e['tilt'])**2]))


def predict(experiments,vacuum_rate):
    return np.array([np.exp(-vacuum_rate*factor(e)*e['time']) for e in experiments])
