"""Keep the shared singlet through both passes over the carrier."""
import numpy as np

swap=np.eye(4).reshape(2,2,2,2).transpose(1,0,2,3).reshape(4,4)
state=np.array([0.,1.,-1.,0.])/np.sqrt(2)
singlet=np.outer(state,state).reshape(2,2,2,2)


def partner_map(early,late,resolve=False):
    def joint(t):
        return (np.cos(t)*np.eye(4)-1j*np.sin(t)*swap).reshape(2,2,2,2)
    a,A=map(joint,early);b,B=map(joint,late)
    rho=np.diag(np.diag(singlet.reshape(4,4))).reshape(2,2,2,2) if resolve else singlet
    # Ket and bra carrier indices before and after the central channel are k,K and l,L.
    channel=np.einsum('iwls,IwLt,kzjx,KzJy,xsyt->iIjJlLkK',
                      B,b.conj(),A,a.conj(),rho,optimize=True)
    return channel.reshape(16,16)
