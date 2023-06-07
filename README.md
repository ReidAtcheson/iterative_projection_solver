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

```
cond(A)=, 4.5780129345667016
min(real(eig(A)))= -4.8235254007989905
max(real(eig(A)))= 4.198723018018987
GMRES(2) iteration 0, residual: 14.80967025124617
GMRES(2) iteration 1, residual: 8.462324480481689
GMRES(2) iteration 2, residual: 6.580761287293522
GMRES(2) iteration 3, residual: 5.048427936980048
GMRES(2) iteration 4, residual: 3.9079254502377285
GMRES(2) iteration 5, residual: 3.16137411250473
GMRES(2) iteration 6, residual: 2.768665193759367
GMRES(2) iteration 7, residual: 2.4306334624521355
GMRES(2) iteration 8, residual: 1.9573268439904057
GMRES(2) iteration 9, residual: 1.5089075690726834
GMRES(2) iteration 10, residual: 1.3231600231618093
GMRES(2) iteration 11, residual: 1.2218208357336249
GMRES(2) iteration 12, residual: 1.114597852213136
GMRES(2) iteration 13, residual: 0.9411285330367135
GMRES(2) iteration 14, residual: 0.734272733056416
GMRES(2) iteration 15, residual: 0.6506857609622219
GMRES(2) iteration 16, residual: 0.5999054619097547
GMRES(2) iteration 17, residual: 0.5388520461731874
GMRES(2) iteration 18, residual: 0.43955967508156607
GMRES(2) iteration 19, residual: 0.34960675299580274
GMRES(2) iteration 20, residual: 0.3128276511581906
k=2,  iteration=0, residual:  4.923878696653819
k=2,  iteration=1, residual:  2.1532931057880114
k=2,  iteration=2, residual:  0.9694324886826715
k=2,  iteration=3, residual:  0.4868056737178905
k=2,  iteration=4, residual:  0.23354880956925206
k=2,  iteration=5, residual:  0.10634202932775515
k=2,  iteration=6, residual:  0.05885311746794666
k=2,  iteration=7, residual:  0.02764386526588386
k=2,  iteration=8, residual:  0.014161637674199574
k=2,  iteration=9, residual:  0.007836975579725734
k=2,  iteration=10, residual:  0.0033558426340838106
k=2,  iteration=11, residual:  0.0019900272445765037
k=2,  iteration=12, residual:  0.0009523943154272405
k=2,  iteration=13, residual:  0.0005365141054099278
k=2,  iteration=14, residual:  0.00027222887456405183
k=2,  iteration=15, residual:  0.00013573925409474908
k=2,  iteration=16, residual:  7.876964812573646e-05
k=2,  iteration=17, residual:  3.709306418118373e-05
k=2,  iteration=18, residual:  2.2147943831110797e-05
k=2,  iteration=19, residual:  1.0880586599023498e-05
```

