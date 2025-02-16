import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la

seed=0
rng=np.random.default_rng(seed)

m=64
restart=5
k=restart

#Random uniform matrix + alpha*identity
#simple hard case for iterative solvers - has good conditioning
#but spectrum is terrible for krylov methods
A=rng.uniform(-1,1,size=(m,m))  + 2*np.eye(m)


#Tridiagonal matrix
A=sp.diags([rng.normal(0,1,size=m),rng.normal(1,0.1,size=m),rng.normal(0,1,size=m)],[-1,0,1],shape=(m,m)).toarray()

#Tridiagonal matrix with spectrum negated on half
A=sp.diags([rng.normal(0,1,size=m),rng.normal(3,0.1,size=m),rng.normal(0,1,size=m)],[-1,0,1],shape=(m,m)).toarray()
A[m//2:,:]=-A[m//2:,:]






print("cond(A)=,",np.linalg.cond(A))
print("min(real(eig(A)))=",np.min(np.real(la.eigvals(A))))
print("max(real(eig(A)))=",np.max(np.real(la.eigvals(A))))

#Manufacture a solution
x=rng.uniform(-1,1,size=(m,))
b=A@x

it=0
def gmres_callback(xk):
    global it
    print(f"GMRES({restart}) iteration {it}, residual: {np.linalg.norm(b-A@xk)}")
    it+=1

spla.gmres(A,b,callback=gmres_callback,callback_type='x',restart=restart,maxiter=20)



#Now suppose we want instead to solve A*(D*r)=r
#where D is a diagonal matrix containing `k` values along the diagonal
#The operator (alpha_0,...,alpha_k) -> (A*D(alpha_0,...,alpha_k)*r) is linear

#Stat with initial guess of x0=0
xh=np.zeros((m,))



for it in range(20):

    r=b-A@xh

    def makeD(alphas):
        d=np.zeros((m,))
        for j,i in enumerate(range(0,m,m//k)):
            ibeg=i
            iend=min(i+m//k,m)
            d[ibeg:iend]=alphas[j]
        return np.diag(d)



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


