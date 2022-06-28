
#################################################
#
# 2022-05-30  Frederic Nataf
#
#################################################
using SparseArrays, LightGraphs, GraphPlot, Plots , Metis, LinearAlgebra, ThreadsX, Test,  FLoops , ThreadSafeDicts , BenchmarkTools

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

 m = 10
 n = 40
 npart = 12


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


#dvector tests with real numbers

my_very_first_DVect = DVector(my_very_first_DDomain, 3.14)

MakeCoherent(my_very_first_DVect)

vuesur(MakeCoherent(Diboolean(my_very_first_DDomain)))

ones(my_very_first_DDomain)

vuesur(DVector( my_very_first_DDomain , ones(length(my_very_first_DDomain.up)) ))

vuesur(noncoherentrandDVector(my_very_first_DDomain))

vuesur(MakeCoherent(noncoherentrandDVector(my_very_first_DDomain)))

vuesur(Dimultiplicity(my_very_first_DDomain))

vuesur(similar(my_very_first_DVect))

vuesur(copy(my_very_first_DVect))

dot(ones(my_very_first_DDomain),ones(my_very_first_DDomain))

fill!(my_very_first_DVect, 6.)

vuesur(my_very_first_DVect+my_very_first_DVect)

 DVector2Vector(my_very_first_DVect)

vuesur(DVector(my_very_first_DVect.domain , DVector2Vector(my_very_first_DVect)))


#dvector tests with complex numbers
#problems with partition of unity and dot_op
# MethodError: no method matching dot_op(::DVector{Float64}, ::DVector{ComplexF64}, ::Base.Broadcast.BroadcastFunction{typeof(*)}) which is real and a solution must not interfere with transposition

my_very_first_Complex_DVect = DVector(my_very_first_DDomain, 1. + 1.0im)

my_very_first_Complex_DVect+my_very_first_Complex_DVect

#############################################################################
#
#  Tests operateurs
#
#############################################################################

DA = DOperator(my_very_first_DDomain , A)


#vuesur(DA.matvec(my_very_first_DVect))
vuesur(DA.matvec(my_very_first_DVect))


function test_mat_vec( A , v , domain )
    Dv = DVector(domain,v)
    DA = DOperator(domain , A)
    res = norm(DVector2Vector(DA.matvec(Dv))-A*v)/norm(A*v)
    println("res $res")
    return res
end



@test test_mat_vec(A,rand(length(my_very_first_DDomain.up)),my_very_first_DDomain) < 1.e-6


# preconditioneur decompose

Am1=DOperatorBlockJacobi(my_very_first_DDomain , A)
####### RAS iteratif twice in a row for reproductability tests  ###################
b = ones(length(Omega))
@time solex=A\b
sol = zeros(length(Omega))
itmax = 20
dsol = DVector(my_very_first_DDomain,sol)
dres = zeros(my_very_first_DDomain)
db = DVector(my_very_first_DDomain,b)

for it in 1:itmax
    global dsol , dres
    dres = dot_op( db , DA.matvec(dsol) , (-))
    println("Norme du vrai residu " , norm( b-A*DVector2Vector(dsol) ) , " at iteration " , it )
    # correction
    dcor = Am1.matvec(dres)
    #MakeCoherent!(dcor)
    dtmp = MakeCoherent(dcor)
#    dsol = dot_op(dsol , dcor , (+) )
    dsol = dot_op(dsol , dtmp , (+) )
#    plot!(DVector2Vector(dsol))
end
#################################################################################@
b = ones(length(Omega))
@time solex=A\b
sol = zeros(length(Omega))
itmax = 20
dsol = DVector(my_very_first_DDomain,sol)
dres = zeros(my_very_first_DDomain)
db = DVector(my_very_first_DDomain,b)

for it in 1:itmax
    global dsol , dres
    dres = dot_op( db , DA.matvec(dsol) , (-))
    println("Norme du vrai residu deuxieme run " , norm( b-A*DVector2Vector(dsol) ) , " at iteration " , it )
    # correction
    dcor = Am1.matvec(dres)
    #MakeCoherent!(dcor)
    dtmp = MakeCoherent(dcor)
#    dsol = dot_op(dsol , dcor , (+) )
    dsol = dot_op(dsol , dtmp , (+) )
#    plot!(DVector2Vector(dsol))
end
#################################################################################@
