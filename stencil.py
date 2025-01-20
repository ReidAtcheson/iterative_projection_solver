import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la

seed=0
rng=np.random.default_rng(seed)

restart=1
k=restart

#2D 5 point stencil blocked up
mx=64
my=64
bx=16
by=16
m=mx*my
maxiter=50
#Make empty sparse matrix
A=sp.lil_matrix((mx*my,mx*my))
def idx(ix,iy):
    return ix+mx*iy
for ix in range(0,mx):
    for iy in range(0,my):
        A[idx(ix,iy),idx(ix,iy)]=4
        if ix>0:
            A[idx(ix,iy),idx(ix-1,iy)]=-1
        if ix<mx-1:
            A[idx(ix,iy),idx(ix+1,iy)]=-1
        if iy>0:
            A[idx(ix,iy),idx(ix,iy-1)]=-1
        if iy<my-1:
            A[idx(ix,iy),idx(ix,iy+1)]=-1

#Now form blocks by list of permutations
perms=[]
for ix in range(0,mx,bx):
    for iy in range(0,my,by):
        ixbeg=ix
        ixend=min(ix+bx,mx)
        iybeg=iy
        iyend=min(iy+by,my)
        p=[]
        for i in range(ixbeg,ixend):
            for j in range(iybeg,iyend):
                p.append(idx(i,j))
        perms.append(p)

restart=5
k=len(perms)





#print("cond(A)=,",np.linalg.cond(A))
#print("min(real(eig(A)))=",np.min(np.real(la.eigvals(A))))
#print("max(real(eig(A)))=",np.max(np.real(la.eigvals(A))))

#Manufacture a solution
x=rng.uniform(-1,1,size=(m,))
b=A@x

it=0
def gmres_callback(xk):
    global it
    print(f"GMRES({restart}) iteration {it}, residual: {np.linalg.norm(b-A@xk)}")
    it+=1

spla.gmres(A,b,callback=gmres_callback,callback_type='x',restart=restart,maxiter=maxiter)



#Now suppose we want instead to solve A*(D*r)=r
#where D is a diagonal matrix containing `k` values along the diagonal
#The operator (alpha_0,...,alpha_k) -> (A*D(alpha_0,...,alpha_k)*r) is linear

#Stat with initial guess of x0=0
xh=np.zeros((m,))



for it in range(maxiter):

    r=b-A@xh

    def makeD(alphas):
        d=np.zeros((m,))
        for j,p in enumerate(perms):
            d[p]=alphas[j]
        #return sparse matrix diagonal matrix containing `d` on diagonal
        return sp.diags(d)



    def matvec(alphas):
        D=makeD(alphas)
        return A@(D@r)

    #Just as a test get the columns of `matvec` directly by applying it to identity columns
    #If this actually works then probably a simple matrix-free iterative method could
    #solve this least-squares system without much memory overheads
    M=np.zeros((m,k))
    I=np.zeros((k,1))
    for i in range(k):
        I[i]=1.0
        res=matvec(I)
        M[:,i]=res
        I[i]=0.0

    #Get numerical rank of M
    #print(sum(la.svdvals(M)>1e-10))


    #Solve Mv=r by least squares
    v,_,_,_=la.lstsq(M,r)
    D=makeD(v)

    #print(f"norm of update: {np.linalg.norm(U@Vt@r)}")

    #Make new solution x
    xh=xh + D@r

    #print before/after residual
    #print(f"Before: {np.linalg.norm(b)}")
    print(f"k={k},  iteration={it}, residual:  {np.linalg.norm(b-A@xh)}")
    #print("cond(A)=",np.linalg.cond(A))

    #U,_=la.qr(AU,mode='economic')


