import sys
sys.path.append("/home/reidatcheson/clones/nd_gauss_seidel")
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la
import arnoldi
import util

seed=0
rng=np.random.default_rng(seed)

eps = 1e-4
k = 10
mx=64
my=64
m=mx*my
bands = [1,mx]
A = sp.diags([rng.uniform(-1,1,size=m) for _ in bands], bands,shape=(m,m))
A = A+A.T + eps * sp.eye(m)
A = sp.coo_matrix(A)
ndtree = util.nested_dissection(A,128)
M = util.assemble_gs_precon(ndtree,A,nparents=4)
util.spy_huge(M,256,"M.svg")

A = sp.bmat([[None, A.T],[A, None]])
M = sp.bmat([[M.T,None],[None,M]])




x=rng.uniform(-1,1,size=(2*m,))
b=A@x

luM = spla.splu(M)
luA = spla.splu(A)
print(A.nnz,M.nnz,luA.nnz,luM.nnz)




xh=np.zeros((2*m,))
maxiter=10
for it in range(maxiter):
    xh=arnoldi.fgmres_update(A,lambda x : x.copy(),b,xh,k)
    print(f"GMRES({k}) iteration {it}, residual: {np.linalg.norm(b-A@xh)}, err: {np.linalg.norm(x-xh,ord=np.inf)}")

print("=====PRECON========")
xh=np.zeros((2*m,))
maxiter=10
for it in range(maxiter):
    xh=arnoldi.fgmres_update(A,lambda z : luM.solve(z) ,b,xh,k)
    print(f"PRECON GMRES({k}) iteration {it}, residual: {np.linalg.norm(b-A@xh)}, err: {np.linalg.norm(x-xh,ord=np.inf)}")



#def M(z):
#    out,_ = spla.minres(A,z,maxiter=1000)
#    return out
#xh=np.zeros((m,))
#maxiter=10
#for it in range(maxiter):
#    xh=arnoldi.fgmres_update(A,lambda x : M(x),b,xh,k)
#    print(f"GMRES({k}) iteration {it}, residual: {np.linalg.norm(b-A@xh)}, err: {np.linalg.norm(x-xh,ord=np.inf)}")


