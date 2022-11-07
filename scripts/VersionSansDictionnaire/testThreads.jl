
#################################################
#
# 2022-07-24  Frederic Nataf
#
#################################################
using SparseArrays, LightGraphs, GraphPlot, Plots , Metis, LinearAlgebra, ThreadsX, Test,  FLoops , ThreadSafeDicts , BenchmarkTools

# Why ThreadsX : dynamic load balancing, task based and not thread based => composability w.r.t. to nested parallelism (Pierre)

BLAS.set_num_threads(1)

include("ddmUtilities.jl")
include("decomposition.jl")
include("ddomain.jl")
include("dvector.jl")
include("doperator.jl")



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

 m = 50
 n = 50
 npart = 1000


# A = spdiagm(-1 => -ones(m - 1), 0 => 2.0 * ones(m), 1 => -ones(m - 1))
# Omega = Domain(1:m)
A = Laplacian2d(m, n, 1, 1);
Omega = Domain(1:m*n);

g = Graph(A);
(initial_partition, decomposition) = create_partition(g, npart);
g_adj = abs.(A);
inflated_indices = Vector{Vector{Int64}}();
map(sub_id -> push!(inflated_indices, inflate_indices(g_adj, sub_id)), initial_partition);

DomDecPartition = create_partition_DDomain(Omega, g, npart);

SetSubdomains = Vector{Domain}();# createurs pas tops

map(indic -> push!(SetSubdomains, Domain(Omega, indic)), inflated_indices);

my_very_first_DDomain = DDomain(Omega, SetSubdomains);


#dvector tests with real numbers

my_very_first_DVect = DVector(my_very_first_DDomain, 3.14);

#############################################################################
#
#  Tests operateurs
#
#############################################################################

DA = DOperator(my_very_first_DDomain , A);


# preconditioneur decompose
# chronometrable
# chronometrable



Avec = fill( spzeros(size(A)) , length(my_very_first_DDomain) )
for i in 1:length(my_very_first_DDomain)
    Avec[i] =  deepcopy(A)
end


# function DOperatorBlockJacobiForTest(DDomD, Avec)
#     ThreadsX.foreach( enumerate(subdomains(DDomD)) ) do ( i , sdi )
#     cholesky(Avec[i]);
#     end
# end


function DOperatorBlockJacobiForTest(DDomD, Avec)
#    Threads.@threads for i in 1:length(DDomD)
ThreadsX.foreach( 1:length(DDomD) ) do i
#    cholesky(Avec[i]);
    lu(Avec[i]);
#    Avec[i]*Avec[i];
    end
end


function DOperatorBlockJacobiForTestSolve(DDomD, Avec)
#    ThreadsX.foreach( enumerate(subdomains(DDomD)) ) do ( i , sdi )
        ThreadsX.foreach( enumerate(subdomains(DDomD)) ) do ( i , sdi )
#    ones(size(Avec[i])[1])\Avec[i];
    sum( ones(size(Avec[i])[1]) );
    end
end

DOperatorBlockJacobiForTest(my_very_first_DDomain , Avec);
DOperatorBlockJacobiForTestSolve(my_very_first_DDomain , Avec);

println("Running test with $( Threads.nthreads()) threads")

@time DOperatorBlockJacobiForTest(my_very_first_DDomain , Avec);

@time DOperatorBlockJacobiForTestSolve(my_very_first_DDomain , Avec);

# for T in 1 2 4 8 12 24 ; do
# echo "# Using $T thread(s): "
# julia -t $T ./testThreads.jl
# echo ""
# done
