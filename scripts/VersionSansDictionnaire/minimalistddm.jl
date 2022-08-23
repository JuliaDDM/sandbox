
#################################################
#
# 2022-05-30  Frederic Nataf
#
#################################################
using SparseArrays, LightGraphs, GraphPlot, Plots , Metis, LinearAlgebra, ThreadsX, Test,  FLoops , ThreadSafeDicts , BenchmarkTools

# Why ThreadsX : dynamic load balancing, task based and not thread based => composability w.r.t. to nested parallelism (Pierre)

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

 m = 600
 n = 600
 npart = 24


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

SetSubdomains = Vector{Domain}()# createurs pas tops

map(indic -> push!(SetSubdomains, Domain(Omega, indic)), inflated_indices);

my_very_first_DDomain = DDomain(Omega, SetSubdomains);


#dvector tests with real numbers

my_very_first_DVect = DVector(my_very_first_DDomain, 3.14);

MakeCoherent(my_very_first_DVect);

vuesur(MakeCoherent(Diboolean(my_very_first_DDomain)));

ones(my_very_first_DDomain);

vuesur(DVector( my_very_first_DDomain , ones(length(my_very_first_DDomain.up)) ));

vuesur(noncoherentrandDVector(my_very_first_DDomain));

vuesur(MakeCoherent(noncoherentrandDVector(my_very_first_DDomain)));

vuesur(Dimultiplicity(my_very_first_DDomain));

vuesur(similar(my_very_first_DVect));

vuesur(copy(my_very_first_DVect));

dot(ones(my_very_first_DDomain),ones(my_very_first_DDomain));

fill!(my_very_first_DVect, 6.);

vuesur(my_very_first_DVect+my_very_first_DVect);

 DVector2Vector(my_very_first_DVect);

vuesur(DVector(my_very_first_DVect.domain , DVector2Vector(my_very_first_DVect)));

mul!( my_very_first_DVect , ones(my_very_first_DDomain) , 2. );

rmul!( my_very_first_DVect , 2.71 );

axpy!(3. , ones(my_very_first_DDomain) , my_very_first_DVect );

axpby!(3. , ones(my_very_first_DDomain) , 5. , my_very_first_DVect );

norm(2. * ones(my_very_first_DDomain));


#dvector tests with complex numbers
my_very_first_Complex_DVect = DVector(my_very_first_DDomain, 1. + 1.0im);

zdecvec = my_very_first_Complex_DVect+my_very_first_Complex_DVect;

vuesur(zdecvec);

vuesur(dot_op(my_very_first_Complex_DVect, zdecvec  , (.*) ));

#############################################################################
#
#  Tests operateurs
#
#############################################################################

DA = DOperator(my_very_first_DDomain , A);


#vuesur(DA.matvec(my_very_first_DVect))
vuesur(DA.matvec(my_very_first_DVect));


function test_mat_vec( A , v , domain )
    Dv = DVector(domain,v)
    DA = DOperator(domain , A)
    res = norm(DVector2Vector(DA.matvec(Dv))-A*v)/norm(A*v)
    println("res $res")
    return res
end



@test test_mat_vec(A,rand(length(my_very_first_DDomain.up)),my_very_first_DDomain) < 1.e-6


# preconditioneur decompose
# chronometrable
@time Am1=DOperatorBlockJacobi(my_very_first_DDomain , A);
# chronometrable
function RASiteration( rhs , Am1 )
    dcor = Am1.matvec(rhs)
    res = MakeCoherent(dcor)
    return res
end

function ASMiteration( rhs , Am1 )
    dcor = Am1.matvec(rhs)
    res = Update(dcor)
    return res
end




#
# ####### RAS iteratif  ###################
# b = ones(length(Omega));
# @time solex=A\b;
# sol = zeros(length(Omega));
# itmax = 20
# dsol = DVector(my_very_first_DDomain,sol);
# dres = zeros(my_very_first_DDomain);
# db = DVector(my_very_first_DDomain,b);
# for it in 1:itmax
#     global dsol , dres;
#     dres = dot_op( db , DA.matvec(dsol) , (-));
#     # correction
#     dtmp = RASiteration( dres , Am1 );
# #    dsol = dot_op(dsol , dcor , (+) )
#     dsol = dot_op(dsol , dtmp , (+) );
#     println("Norme du vrai residu  " , norm( b-A*DVector2Vector(dsol) ) , " at iteration " , it )
# #    plot!(DVector2Vector(dsol))
# end

#for T in 1 2 4 8 12 24 ; do
#echo "# Using $T thread(s): "
#julia -t $T ./minimalistddm.jl
#echo ""
#done
