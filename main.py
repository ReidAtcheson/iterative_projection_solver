import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la

seed=0
rng=np.random.default_rng(seed)

m=64
A=rng.uniform(-1,1,size=(m,m))  + 10*np.eye(m)

#Manufacture a solution
x=rng.uniform(-1,1,size=(m,))
b=A@x

it=0
def gmres_callback(xk):
    global it
    print(f"GMRES iteration {it}, residual: {np.linalg.norm(b-A@xk)}")
    it+=1

spla.gmres(A,b,callback=gmres_callback,callback_type='x')



#Now suppose we want instead to solve (AU*(VT*r))=r given a guess x0 and
#residual r=b-A*x0
#Suppose U is a random orthogonal matrix, then we want to solve for VT.
#The function mapping VT --> AU*(VT*r) is linear so we can do 
#a least-squares fit

k=4

#Stat with initial guess of x0=0
xh=np.zeros((m,))

#Make random columns for U and then orthogonalize
U=rng.uniform(-1,1,size=(m,k))
U,_=la.qr(U,mode='economic')


for _ in range(200):

    r=b-A@xh
    AU=A@U

    def matvec(v):
        #Compute AU*(VT*v)
        Vt=np.reshape(v,(k,m))
        return AU@(Vt@r)

    #Just as a test get the columns of `matvec` directly by applying it to identity columns
    #If this actually works then probably a simple matrix-free iterative method could
    #solve this least-squares system without much memory overheads
    M=np.zeros((m,m*k))
    I=np.zeros((m*k,1))
    for i in range(m*k):
        I[i]=1.0
        res=matvec(I)
        M[:,i]=res
        I[i]=0.0

    #Get numerical rank of M
    #print(sum(la.svdvals(M)>1e-10))


    #Solve Mv=r by least squares
    v,_,_,_=la.lstsq(M,r)
    #Create Vt out of v
    Vt=np.reshape(v,(k,m))

    #print(f"norm of update: {np.linalg.norm(U@Vt@r)}")

    #Make new solution x
    xh=xh + U@Vt@r

    #print before/after residual
    #print(f"Before: {np.linalg.norm(b)}")
    print(f"After: {np.linalg.norm(b-A@xh)}")
    #print("cond(A)=",np.linalg.cond(A))

    #U,_=la.qr(AU,mode='economic')

    U=rng.uniform(-1,1,size=(m,k))
    U,_=la.qr(U,mode='economic')

