import argparse
import scipy.linalg as la
from functools import partial
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt






#parser = argparse.ArgumentParser()
#parser.add_argument("--diag",type=float)
#parser.add_argument("--gmres_restart",type=int)
#parser.add_argument("--topk",type=int)
#parser.add_argument("--res_mode",choices=["min","max"])
#parser.add_argument("--trigger_tol",type=float)

#args=parser.parse_args()




def arnoldi_dgks_fr(A,M,v,k):
    norm=np.linalg.norm
    dot=np.dot
    eta=1.0/np.sqrt(2.0)

    m=len(v)
    Z=np.zeros((m,k+1))
    V=np.zeros((m,k+1))
    H=np.zeros((k+1,k))
    V[:,0]=v/norm(v)
    for j in range(0,k):
        Z[:,j]=M(V[:,j])
        w=A(Z[:,j])
        h=V[:,0:j+1].T @ w
        f=w-V[:,0:j+1] @ h
        s = V[:,0:j+1].T @ f
        f = f - V[:,0:j+1] @ s
        h = h + s
        beta=norm(f)
        H[0:j+1,j]=h
        H[j+1,j]=beta
        V[:,j+1]=f/beta
    return Z,V,H




def arnoldi_dgks(A,v,k):
    norm=np.linalg.norm
    dot=np.dot
    eta=1.0/np.sqrt(2.0)

    m=len(v)
    V=np.zeros((m,k+1))
    H=np.zeros((k+1,k))
    V[:,0]=v/norm(v)
    for j in range(0,k):
        w=A(V[:,j])
        h=V[:,0:j+1].T @ w
        f=w-V[:,0:j+1] @ h
        s = V[:,0:j+1].T @ f
        f = f - V[:,0:j+1] @ s
        h = h + s
        beta=norm(f)
        H[0:j+1,j]=h
        H[j+1,j]=beta
        V[:,j+1]=f/beta
    return V,H


def gmres_update(A,b,x,k):
    r=b-A@x
    V,H=arnoldi_dgks(lambda y : A@y,r,k)
    e=np.zeros(k+1)
    e[0]=np.linalg.norm(r)
    y,_,_,_=np.linalg.lstsq(H,e,rcond=None)
    return x+V[:,0:k]@y



def fgmres_update(A,M,b,x,k):
    r=b-A@x
    Z,V,H=arnoldi_dgks_fr(lambda y : A@y,M,r,k)
    e=np.zeros(k+1)
    e[0]=np.linalg.norm(r)
    y,_,_,_=np.linalg.lstsq(H,e,rcond=None)
    return x+Z[:,0:k]@y

