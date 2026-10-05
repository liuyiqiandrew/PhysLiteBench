"""Exact rational verification of a rounded common Lyapunov certificate."""
from pathlib import Path
from fractions import Fraction as F
from itertools import product
import json
HERE=Path(__file__).parent
report=json.loads((HERE/'stability-report.json').read_text())
P=[[F(f'{x:.8f}') for x in row] for row in report['P']]
b=F('0.9');margin=F('0.001');n=6

def matrix(v):
    g,kx,ky,h,w,s=v;A=[[F(0) for _ in range(n)] for _ in range(n)]
    for i,k in enumerate([kx,ky]):
        A[i][2+i]=1;A[2+i][i]=-k-h;A[2+i][2+i]=-g;A[2+i][4+i]=h
        A[4+i][i]=h/b;A[4+i][4+i]=(-h/b)+(s if i==0 else -s)
    A[4][5]=-w;A[5][4]=w
    return A

def positive_pivots(M):
    L=[[F(int(i==j)) for j in range(n)] for i in range(n)];D=[]
    for i in range(n):
        d=M[i][i]-sum(L[i][k]*L[i][k]*D[k] for k in range(i));assert d>0
        D.append(d)
        for j in range(i+1,n):L[j][i]=(M[j][i]-sum(L[j][k]*L[i][k]*D[k] for k in range(i)))/d
    return [str(d) for d in D]

def subtract_margin(M):return [[M[i][j]-(margin if i==j else 0) for j in range(n)] for i in range(n)]

vertices=[]
for v in product(*[[F(str(x)) for x in vals] for vals in report['bounds'].values()]):
    A=matrix(v)
    Q=[[-sum(A[k][i]*P[k][j]+P[i][k]*A[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
    vertices.append({'controls':dict(zip(report['bounds'],map(str,v))),'positive_LDL_pivots':positive_pivots(subtract_margin(Q))})
out={'status':'exact rational certificate verified','coordinates':['x','y','vx','vy','qx','qy'],'P_decimal':[[str(float(x)) for x in row] for row in P],
 'uniform_margin':str(margin),'P_minus_margin_positive_pivots':positive_pivots(subtract_margin(P)),'vertices':vertices,
 'proof':'P−I/1000 and −A^T P−P A−I/1000 have exact positive rational LDL pivots at every vertex. A is affine in all six independent controls. Each interior matrix is their convex combination, so P>0 and A^T P+P A<0 throughout the complete box. Temperatures do not enter A.'}
(HERE/'exact-stability-certificate.json').write_text(json.dumps(out,indent=2)+'\n')
print('Exact rational certificate verified at',len(vertices),'vertices; uniform margin',margin)
