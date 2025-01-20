import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la
from scipy.optimize import least_squares

seed=0
rng=np.random.default_rng(seed)

m=64
restart=4
k=4
#This is a fixed function with 4 parameters
assert(k==4)

#Random uniform matrix + alpha*identity
#simple hard case for iterative solvers - has good conditioning
#but spectrum is terrible for krylov methods
#A=rng.uniform(-1,1,size=(m,m))  + 2*np.eye(m)


#Tridiagonal matrix
#A=sp.diags([rng.normal(0,1,size=m),rng.normal(1,0.1,size=m),rng.normal(0,1,size=m)],[-1,0,1],shape=(m,m)).toarray()

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



#Now suppose we want instead to solve A*(D(alphas)*r)=r
#where D is a diagonal matrix whose diagonal entries are defined
#by alpha_0 + alpha_1 * tanh(alpha_2 * x + alpha_3)

#Then after finding the optimal alphas for D(alphas) the update for `x` becomes
#x=x + D(alphas)*r

#Start with initial guess of x0=0
xh=np.zeros((m,))



for it in range(20):

    r=b-A@xh

    def makeD(alphas):
        xs=np.linspace(-1.0,1.0,m)
        d=alphas[0]+alphas[1]*np.tanh(alphas[2]*xs+alphas[3])
        return np.diag(d)

    #nonlinear residual
    def nres(alphas):
        D=makeD(alphas)
        rh=r-A@(D@r)
        return rh

    #Find optimal `alphas` via nonlinear least squares
    result=least_squares(nres,np.zeros(k),jac='3-point',verbose=2)
    D=makeD(result.x)


    #print(f"norm of update: {np.linalg.norm(U@Vt@r)}")

    #Make new solution x
    xh=xh + D@r

    #print before/after residual
    #print(f"Before: {np.linalg.norm(b)}")
    print(f"k={k},  iteration={it}, residual:  {np.linalg.norm(b-A@xh)}")
    #print("cond(A)=",np.linalg.cond(A))

    #U,_=la.qr(AU,mode='economic')


