#################################################
#
# 2021-12-12  Frederic Nataf
#
#################################################
# essai de partir du point de vue utilisateur
# il existe qq part une numérotation globale
using SparseArrays, LightGraphs, GraphPlot, Plots , Metis, LinearAlgebra, ThreadsX, Test,  FLoops , ThreadSafeDicts , BenchmarkTools



include("ddmUtilities.jl")
include("decomposition.jl")
include("ddomain.jl")
include("dvector.jl")
include("doperator.jl")



function DOperatorBlockJacobiThreadedTest(DDomD, A)
    DA_lu = ThreadSafeDict()
    ThreadsX.foreach(subdomains( DDomD ))  do sdi
        #        for sdi ∈ subdomains(DDomD)
        DA_lu[sdi] = factorize(A[global_indices(sdi), global_indices(sdi)]  )
    end
end

function DOperatorBlockJacobiThreadedTestuseless(DDomD, A)
    ThreadsX.foreach(subdomains( DDomD ))  do sdi
        #        for sdi ∈ subdomains(DDomD)
         factorize(A[global_indices(sdi), global_indices(sdi)]  )
    end
end


function DOperatorBlockJacobiTest(DDomD, A)
    #DA_lu = ThreadSafeDict()
    DA_lu = Dict()
#    ThreadsX.foreach(subdomains( DDomD ))  do sdi
    for sdi ∈ subdomains(DDomD)
        DA_lu[sdi] = factorize(A[global_indices(sdi), global_indices(sdi)]  )
    end
end

function DOperatorBlockJacobiThreadedTestFloops(DDomD, A)
#    DA_lu = Dict()
    DA_lu = ThreadSafeDict()
    @floop for sdi ∈ subdomains(DDomD)
         DA_lu[sdi] =
         factorize(A[global_indices(sdi), global_indices(sdi)]  )
    end
end


#    Di  # la partition de l'unité locale au sous domaine vue comme un operateur local verifiant une certaine propriété
#  Di est définie sur un DDomain (constructeur) et on l'interroge en luio donnant un sous domaine et il renvoie le vecteur des poids D(Omage_i)
mutable struct DPOU
    Ddomain::DDomain
    #  constructeur(DDomain,options)
    #  DPOU(DDomain) renvoie le vecteur des poids sur le domaine en question
end
#RAS: constructeur(DDomain, A , POU) , la POU est associée à un algorithme plus qu'à une décomposition si argument POU absent, on prend les Di implicite à ???



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

SetSubdomains = Set{Domain}()# createurs pas tops

map(indic -> push!(SetSubdomains, Domain(Omega, indic)), inflated_indices)

my_very_first_DDomain = DDomain(Omega, SetSubdomains)


my_very_first_DVect = DVector(my_very_first_DDomain, 1.0)

aa = Update(my_very_first_DVect)
bb = DVector(my_very_first_DDomain, 3.0)

dot_op(aa, bb, (.*))
# aa .* bb
# Vincent: a ne pas faire mais plutôt ..* car en fait broadcast à deux niveaux

my_very_first_Di = Di(my_very_first_DDomain)

zzz = Update(dot_op(my_very_first_Di, my_very_first_DVect, (.*)))


#vuesur(zzz)

#vuesur(Diboolean(my_very_first_DDomain))

#vuesur(Update(dot_op(Diboolean(my_very_first_DDomain), my_very_first_DVect, (.*))))

aaa = 1.0 * collect(1:length(Omega))

daaa = DVector(my_very_first_DDomain, aaa)

#vuesur(Update(dot_op(Diboolean(my_very_first_DDomain), daaa, (.*))))

DVector2Vector(daaa)

vtest = rand(length(Omega))
norm(vtest .- DVector2Vector(DVector(my_very_first_DDomain, vtest)))

DA = DOperator(my_very_first_DDomain , A)

DAseq = DOperatorSequential(my_very_first_DDomain , A)

DA.matvec(daaa)

function test_mat_vec( A , v , domain )
    Dv = DVector(domain,v)
    DA = DOperator(domain , A)
    norm(DVector2Vector(DA.matvec(Dv))-A*v)/norm(A*v)
end

function compare_mat_vecSeqPar( A , v , domain )
    Dv = DVector(domain,v)
    DA = DOperator(domain , A)
    DASeq = DOperatorSequential(domain , A)
    (norm(DVector2Vector(DA.matvec(Dv))-A*v)/norm(A*v) , norm(DVector2Vector(DASeq.matvec(Dv))-A*v)/norm(A*v) )
end


function test_mat_vecreproductible( A , v , domain )
    Dv = DVector(domain,v)
    Dv2 = DVector(domain,v)
    DA = DOperator(domain , A)
    DVector2Vector(DA.matvec(Dv))-DVector2Vector(DA.matvec(Dv2))
end




test_mat_vec(A,aaa,my_very_first_DDomain)
test_mat_vec(A,rand(length(my_very_first_DDomain.up)),my_very_first_DDomain)

# a regler pour etre vraiment zero ==> PRIORITAIRE
# en fait en l'absence de parallélisme la fonction est forcément reproductible
# par contre elle ne coincide pas forcément avec la version séquentielle
@test norm(test_mat_vec(A,rand(length(my_very_first_DDomain.up)),my_very_first_DDomain)) < 1.e-6

Am1=DOperatorBlockJacobi(my_very_first_DDomain , A)
#Am1.matvec(daaa)


####### RAS iteratif  ###################
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

if (0>1)
@show npart
 @show @btime DOperatorBlockJacobiTest(my_very_first_DDomain , A);

@show @btime DOperatorBlockJacobiThreadedTest(my_very_first_DDomain , A);

 @show @btime DOperatorBlockJacobiThreadedTestuseless(my_very_first_DDomain , A);

 @show @btime DOperatorBlockJacobiThreadedTestFloops(my_very_first_DDomain , A);
end



# faire DOperator ok
# faire des tests ok
### séparer en modules pour avoir un code principal plus léger, voir julialang modules => what if a file is modified and is included
# paralleliser le produit matrice vecteur
### être cohérent au niveau des encapsulations
### Questions: ThreadSafeDict est intrusif. ThreadSafeDict une fois, ThreadSafeDict toujours => type Union??? dans la definition de DVector????
###            algorithme d'inflation pas en N^2, cf PETSc???
# compatibilité avec les librairies de méthodes de Krylov : cf LinearSolve -> ::invPrecondintioner ou on definit le prod mat vec au lieu de ldiv
# deux niveaux - trois niveaux
# commenter - unit test dossier test de la documentation de Julia
# a nettoyer,
# a encapsuler, trop de references aux membres des structures???
# Passer de Dict à vector pour éviter les problèmes de race condition en parallèle => si sous domaines petits, cela risque c'etre penalisant, regarder enumerate pour avoir une syntaxe  ??

# Dictionnaire et //
# melanger Dict() et floops => risque de conflit en écriture
#        ThreadSafeDict
#   s'avouer vaincu et transferer le dictionnair en deux vecteurs
#  /!\  Floops (basé sur Thread statique ) vs ThreadsX (Thread dynamique)
#  restons sur ThreadsX et regardons Floops
# SUPPRIMER les ThreadSafeDict car inutiles ou bien quand utilisés en lecture passer au Dict normaux?? ??
## REFAIRE le point sur les dictionnaires


# npart = 8
# julia> @btime DOperatorBlockJacobiTest(my_very_first_DDomain , A);
#   895.837 ms (473 allocations: 938.20 MiB)
#
# julia>  @btime DOperatorBlockJacobiThreadedTest(my_very_first_DDomain , A);
#   4.794 s (626 allocations: 938.21 MiB)
#
# julia> @btime DOperatorBlockJacobiThreadedTestuseless(my_very_first_DDomain , A);
#   4.062 s (576 allocations: 938.21 MiB)


# npart = 80
# 912.769 ms (4700 allocations: 1.36 GiB)
# 2.761 s (5215 allocations: 1.36 GiB)
# 2.757 s (5118 allocations: 1.36 GiB)


# npart = 400
#   951.615 ms (22583 allocations: 3.55 GiB)
#   134.000 ms (23837 allocations: 3.55 GiB)
#   178.366 ms (23732 allocations: 3.55 GiB)

noncoherentrandDVector( my_very_first_DDomain );

       compare_mat_vecSeqPar(A,rand(length(my_very_first_DDomain.up)),my_very_first_DDomain)

 @test norm(test_mat_vec(A,rand(length(my_very_first_DDomain.up)),my_very_first_DDomain)) < 1.e-9

 @test norm(test_mat_vecreproductible(A,rand(length(my_very_first_DDomain.up)),my_very_first_DDomain)) == 0.
