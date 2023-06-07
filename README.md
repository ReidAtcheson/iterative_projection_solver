# iterative_projection_solver
experiment with a unique solver concept


The idea basically is to form an iterative solver by starting
with an initial guess `x` and residual `r=b-Ax0` then compute
the update `x=x+e` by setting `e=Mr` and solving for `M`  in

```
A(Mr) = r
```

here `M` can be any matrix but in practice we want `M` to be a very simple and structured
matrix dependent on only a few paramters (otherwise solving `A(Mr)=r` becomes just as hard
as solving `Ax=b`)

The function `F(M)=A(Mr)` is linear in its argument `M` so if we can make `M` linearly dependent
on just a few parameters like say `alpha_0,...alpha_k` then we get a least squares problem in
`k` parameters for the update of `x` and can repeat the process.


# Choices of M

## Low rank factorization with a fixed factor

First I tried (in `main.py`) to set `M=UV^T` where `U` is a random orthogonal matrix with `k` columns
and then solve for `V^T`. This had very mixed results and I think largely unsuccesful.


## Diagonal matrix with just a few values

Another possibility is to make `M` diagonal but with repeating entries. For example

`M = diag([rho,rho,rho,rho,rho,alpha,alpha,alpha,alpha])`

and then solve for `rho,alpha` to form the update of `x` This was much more successful than the first try (with low rank factors)
and had the expected improvement in convergence rates as `k` increases (note that higher values of `k` means we need to store more columns
in this formulation, like with GMRES).

Notably this worked on a matrix where I took a definite matrix and negated half of its rows so that it became indefinite, these commonly
cause restarted GMRES to stagnate, however when we set the diagonal entries of `M` to correspond to the same partition of rows used for the 
negation then convergence is very rapid.

