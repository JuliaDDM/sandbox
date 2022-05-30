#################################################
#
# 2022-05-30  Frederic Nataf
#
#################################################
using SparseArrays, LightGraphs, GraphPlot, Plots , Metis, LinearAlgebra, ThreadsX, Test,  FLoops , ThreadSafeDicts , BenchmarkTools

include("ddmUtilities.jl")
include("decomposition.jl")
include("ddomain.jl")



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
Omega = Domain(1:m*n)

g = Graph(A)
(initial_partition, decomposition) = create_partition(g, npart)
g_adj = abs.(A)
inflated_indices = Vector{Vector{Int64}}();
map(sub_id -> push!(inflated_indices, inflate_indices(g_adj, sub_id)), initial_partition)

DomDecPartition = create_partition_DDomain(Omega, g, npart)

SetSubdomains = Vector{Domain}()# createurs pas tops

map(indic -> push!(SetSubdomains, Domain(Omega, indic)), inflated_indices)



my_very_first_DDomain = DDomain(Omega, SetSubdomains)
