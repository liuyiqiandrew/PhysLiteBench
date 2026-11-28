"""The two island-arm trace matrices before coupling an optional external detector."""
import numpy as np
from dot_core import Island, normalized


def history_table(q):
    """Return tag strings and their (bra,ket) contributions to retention and complex Ramsey coherence."""
    island=Island(q)
    initial=normalized(q['counter']['prepare'])
    final=normalized(q['counter']['analyze'])
    size=len(initial)
    branches={((),k):amp*island.identity for k,amp in enumerate(initial) if amp!=0}
    for encounter in q['encounters']:
        tag=normalized(encounter['tag'])
        eraser=normalized(encounter['erase'])
        gates={}
        for k in {k for _,k in branches}:
            outgoing=island.outgoing(encounter,k)
            yes=np.vdot(eraser,tag)*np.einsum('i,ijk->jk',tag.conj(),outgoing)
            total=np.einsum('i,ijk->jk',eraser.conj(),outgoing)
            gates[k]=(total-yes,yes)
        next_branches={}
        for (history,k),operator in branches.items():
            for bit in (0,1):
                r=(k+bit)%size
                phase=np.exp(-.5j*encounter['dwell']*r*(r-1))
                next_branches[history+(bit,),r]=phase*gates[k][bit]@operator
        branches=next_branches
    projected={}
    for (history,k),operator in branches.items():
        projected[history]=projected.get(history,0)+final[k].conjugate()*operator
    histories=list(projected)
    matter=np.array(list(projected.values()))
    marker=np.ones((len(histories),len(histories)),complex)
    for j,encounter in enumerate(q['encounters']):
        bits=np.array([h[j] for h in histories])
        m=complex(*encounter['marker'])
        marker*=np.array([[1,m],[m.conjugate(),1]])[bits[:,None],bits]
    def gram(left,right):
        return marker*np.einsum('aij,bji->ab',left.conj().transpose(0,2,1),right@island.rho)
    arms=[]
    for arm in q['probe']:
        op=island.identity
        for a,b,turn in arm:
            pair=island.sites[a]@island.sites[b]
            op=(np.cos(turn/2)*island.identity-np.sin(turn/2)*pair)@op
        arms.append(op@matter)
    return histories,gram(matter,matter),gram(*arms)


def measurement(q,detector=None):
    histories,rate,coherence=history_table(q)
    gram=detector(q['lead'],histories) if detector is not None and 'lead' in q else 1.
    probability=np.sum(gram*rate).real
    fringe=(np.exp(-1j*q['phase'])*np.sum(gram*coherence)).real
    return probability,fringe/probability
