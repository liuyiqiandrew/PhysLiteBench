import numpy as np
C=np.array([[6.,2.,0.],[2.,4.,0.],[0.,0.,3.]])*1e6
q=np.array([1.,1.,0.]);alpha=.9;M=20e6
S0=1/M+alpha**2*q@np.linalg.solve(C,q)
K=np.array([[1.,.2],[.2,.6]])*1e-15;leak=1e-10
for mode in [[1,0],[0,1],[1,1],[1,-1],[2,1],[1,2]]:
 k=2*np.pi/.01*np.array(mode);g=np.array([[k[0],0.],[0.,k[1]],[k[1],k[0]]]);storage=1/M+alpha**2*(g.T@q)@np.linalg.solve(g.T@C@g,g.T@q)
 conductance=(k@K@k+leak)/.0011
 for t in [.2,.4]:
  correct=np.exp(-conductance/storage*t);source=np.exp(-conductance/S0*t)
  print(mode,'storage',storage,'source',S0,'t',t,'signals',correct,source,'rel',source/correct-1)
