import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la

seed=0
rng=np.random.default_rng(seed)


m=32
#Tridiagonal matrix
A=sp.diags([np.ones(m),-1.1*np.ones(m),np.ones(m)],[-1,0,1],shape=(m,m)).toarray()

print(la.eigvals(A)),rng.normal(0,1,size=m)
print(np.linalg.cond(A))

p=[0]

s=set(range(1,m))

pdiff=list(set(range(0,m)).intersection(set(p)))

it=0
while not all(e > 0 for e in la.eigvalsh(A[np.ix_(pdiff,pdiff)])):
    it=it+1
    if it>100:
        break
    for i in s:
        pi=p+[i]
        pd=list(set(range(0,m)).intersection(set(pi)))
        eigs=la.eigvalsh(A[np.ix_(pi,pi)])
        if all(e<0 for e in eigs):
            p.append(i)
            s.remove(i)
            pdiff=list(set(range(0,m)).intersection(set(p)))
            break


print(la.eigvalsh(A[np.ix_(p,p)]))
pdiff=list(set(range(0,m)).difference(set(p)))
print(la.eigvalsh(A[np.ix_(pdiff,pdiff)]))
print(p)
