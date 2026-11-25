"""Independent heat-flux divergence plus electrical work on a refined mesh."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags

TRUE_CONDUCTIVITY=1.25
CAPACITY=1.5e6
RHO=2e-5


def predict(t,x,initial,current,boundary,seebeck_span,conductivity,refinement=4):
    t,x,initial=np.asarray(t),np.asarray(x),np.asarray(initial)
    fine=np.linspace(x[0],x[-1],refinement*(len(x)-1)+1)
    dx=fine[1]-fine[0];start=np.interp(fine,x,initial)
    face_x=(fine[1:]+fine[:-1])/2
    def derivative(time,interior):
        temp=np.r_[boundary[0],interior,boundary[1]]
        face=(temp[:-1]+temp[1:])/2;gradient=np.diff(temp)/dx
        seebeck_face=2e-4+4e-6*(face-300.)+seebeck_span*(face_x/.01-.5)
        heat_flux=current*seebeck_face*face-conductivity*gradient
        seebeck_center=2e-4+4e-6*(interior-300.)+seebeck_span*(fine[1:-1]/.01-.5)
        field=RHO*current+seebeck_center*(temp[2:]-temp[:-2])/(2*dx)
        return (-np.diff(heat_flux)/dx+current*field)/CAPACITY
    if t[-1]==0:
        temperature=start[None].copy()
    else:
        n=len(start)-2;pattern=diags([np.ones(n-abs(k)) for k in [-1,0,1]],[-1,0,1])
        result=solve_ivp(derivative,(0.,t[-1]),start[1:-1],t_eval=t,method='BDF',
                         rtol=2e-10,atol=1e-8,jac_sparsity=pattern)
        assert result.success,result.message
        temperature=np.column_stack([np.full(len(t),boundary[0]),result.y.T,np.full(len(t),boundary[1])])
    face=.5*(temperature[:,1:]+temperature[:,:-1])
    s=2e-4+4e-6*(face-300.)+seebeck_span*(face_x/.01-.5)
    voltage=RHO*current*.01+np.sum(s*np.diff(temperature),axis=1)
    return {'temperature':temperature[:,::refinement],'voltage':voltage}


def calibration_settings():
    x=np.linspace(0,.01,31);t=np.array([2.,5.,10.,18.,32.,55.,85.])
    boundaries=np.array([[290.,330.],[310.,285.]])
    profiles=[np.linspace(*b,len(x))+a*np.sin(np.pi*x/.01) for b,a in zip(boundaries,[8.,-7.])]
    return {'x':x,'t':t,'initial':np.array(profiles),'boundary':boundaries,
            'current':np.array([1.2e5,-1.2e5]),'seebeck_span':np.zeros(2)}


def hidden_inputs():
    x=np.linspace(0,.01,51);t=np.array([5.,15.,30.,60.,100.,160.])
    result=[]
    for current,boundary,amplitude,span in [(2e5,(290.,330.),8.,5e-4),
                                           (-2e5,(290.,330.),8.,5e-4),
                                           (1.5e5,(315.,285.),-6.,-6e-4)]:
        initial=np.linspace(*boundary,len(x))+amplitude*np.sin(np.pi*x/.01)
        result.append((t,x,initial,current,boundary,span))
    return result


def metrics(model,data):
    residuals=[]
    for i,initial in enumerate(data['initial']):
        output=model.predict(data['t'],data['x'],initial,data['current'][i],data['boundary'][i],data['seebeck_span'][i])
        for field in ['temperature','voltage']:
            residuals.extend(((output[field]-data[field][i])/data['sigma_'+field][i]).ravel())
    result={'parameter':float(model.conductivity),'parameter_relative_error':abs(float(model.conductivity)/TRUE_CONDUCTIVITY-1),
            'calibration_chi2':float(np.sum(np.square(residuals))/(len(residuals)-1)),'hidden':{}}
    for i,args in enumerate(hidden_inputs()):
        truth=predict(*args,TRUE_CONDUCTIVITY);actual=model.predict(*args)
        scale=np.sqrt(np.mean((truth['temperature']-np.linspace(*args[4],len(args[1])))**2))
        assert actual['temperature'].shape==truth['temperature'].shape and actual['voltage'].shape==truth['voltage'].shape
        assert np.isfinite(actual['temperature']).all() and np.isfinite(actual['voltage']).all()
        result['hidden'][f'current_{i}']=float(np.sqrt(np.mean((actual['temperature']-truth['temperature'])**2))/scale)
        result.setdefault('voltage_absolute_error',[]).append(float(np.max(abs(actual['voltage']-truth['voltage']))))
    return result
