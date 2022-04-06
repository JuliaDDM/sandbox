using Random
using SparseArrays


#################################################
#
#     premiers tests
#
#################################################


sdiff1(m) = spdiagm(-1 => -ones(m - 1), 0 => ones(m))
# make the discrete -Laplacian in 2d, with Dirichlet boundaries
# adapted from https://math.mit.edu/~stevenj/18.303/lecture-10.html
function Laplacian2d(Nx, Ny, Lx, Ly)
    dx = Lx / (Nx + 1)
    dy = Ly / (Ny + 1)
    Dx = sdiff1(Nx) / dx
    Dy = sdiff1(Ny) / dy
    Ax = Dx' * Dx
    Ay = Dy' * Dy
    return kron(spdiagm(0 => ones(Ny)), Ax) + kron(Ay, spdiagm(0 => ones(Nx)))
end

 m = 100
 n = 40
 npart = 8


# A = spdiagm(-1 => -ones(m - 1), 0 => 2.0 * ones(m), 1 => -ones(m - 1))
# Omega = Domain(1:m)
A = Laplacian2d(m, n, 1, 1);


nt=n*m
newnumbering=shuffle(1:nt)

PermatationMatrix=sparse(1:nt,newnumbering,ones(nt));
invPermatationMatrix=sparse(newnumbering,1:nt,ones(nt));

matcheck=PermatationMatrix*invPermatationMatrix-sparse(1:nt,1:nt,ones(nt))

APm1=A*invPermatationMatrix;
x=rand(1:nt);
y = A*x;

Px=PermatationMatrix*x;

APm1*Px-y
