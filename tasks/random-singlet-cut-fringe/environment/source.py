"""Sharp first-line measures of a common cut, before spectrometer dither."""
import numpy as np

P={'x':np.array([[0.,1.],[1.,0.]],complex),'y':np.array([[0.,-1j],[1j,0.]]),'z':np.diag([1.,-1.])}
singlet=np.array([0.,1.,-1.,0.])/np.sqrt(2)
slow=(3-np.sqrt(5))/2;fast=3-slow

def rotate(d,t):return np.cos(t/2)*np.eye(2)-1j*np.sin(t/2)*P[d]

class Sharp:
    """Return [raw acceptance density, raw complex Ramsey density] in the physical line coordinates."""
    def __init__(self,q):
        self.edges=np.r_[0.,np.cumsum([b['width'] for b in q['bands']])]
        self.start=np.array([1.,0.]) if q['start']=='renewed' else np.array([fast/3,slow/3])
        self.tables=[]
        for b in q['bands']:
            arms=[]
            for seq in b['arms']:
                ket=singlet
                for side,axis,t in seq:
                    u=rotate(axis,t);ket=(np.kron(u,np.eye(2)) if side=='L' else np.kron(np.eye(2),u))@ket
                arms.append(ket)
            m=np.vdot(*arms)
            projection=[];eff=[]
            for eta,d,t in b['detectors']:
                u=rotate(d,t);projection.append(u.conj().T@np.diag([1.,0])@u);eff.append(eta)
            p11=eff[0]*eff[1]*np.vdot(singlet,np.kron(*projection)@singlet).real
            if abs(p11)<1e-15:p11=0.
            pl,pr=np.array(eff)/2
            p=np.array([1-pl-pr+p11,pl-p11,pr-p11,p11])
            table={}
            for active in range(4):
                for seen in range(4):
                    weight=p[(np.arange(4)&active)==seen].sum()*np.array([1.,m])
                    if seen:table[active,seen]=weight*fast
                    else:
                        # For a 2x2 matrix, exp(tA) = exp(-3t/2) [cosh(dt/2)I +
                        # 2 sinh(dt/2)(A+3I/2)/d] with d=sqrt(5+4z).
                        table[active,seen]=(np.sqrt(5+4*weight),weight)
            self.tables.append(table)

    def raw(self,left,right,diagonal=False,order=None):
        """On diagonal return density per single common frequency, elsewhere per pair.

        To request the continuous limit at left==right, pass diagonal=False
        and order='LR' or 'RL' for the side from which the limit is taken.
        """
        if min(left,right)<0 or max(left,right)>self.edges[-1]:return np.zeros(2,complex)
        events=([(left,3)] if diagonal else
                [(left,1),(right,2)] if left<right or (left==right and order!='RL') else [(right,2),(left,1)])
        state=np.tile(self.start,(2,1)).astype(complex);active=3;next_event=0
        for j,table in enumerate(self.tables):
            l,r=self.edges[j:j+2]
            def flow(dt):
                if dt==0:return
                d,z=table[active,0]
                A=state[:,0].copy();B=state[:,1].copy()
                base=np.exp(-1.5*dt);co=np.cosh(.5*d*dt);si=2*np.sinh(.5*d*dt)/d
                state[:,0]=base*(co*A+si*((1.5-slow)*A+fast*z*B))
                state[:,1]=base*(co*B+si*(slow*A+(1.5-fast)*B))
            while next_event<len(events) and events[next_event][0]<=r:
                t,mark=events[next_event];flow(t-l)
                state[:,0]=table[active,mark]*state[:,1];state[:,1]=0.
                active &= ~mark;next_event+=1;l=t
            flow(r-l)
        return state.sum(axis=1)
