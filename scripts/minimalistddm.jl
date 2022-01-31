#################################################
#
# 2021-12-12  Frederic Nataf
#
#################################################
# essai de partir du point de vue utilisateur
# il existe qq part une numérotation globale
using SparseArrays, LightGraphs, GraphPlot, Metis, LinearAlgebra, ThreadsX, Test, Plots

"""
intersectalamatlab( a , b )

Returns the intersection of the values of a and b as well their positions
# Arguments
- 'a' and 'b' are vectors
# Example
a =[3 , 45 , 123 , 12]
b = [12 , 19 , 46 , 56 , 123]
intersectalamatlab( a , b )
([123, 12], [3, 4], [5, 1])
"""
function intersectalamatlab(a, b)
    function findindices!(resa, ab, a)
        for (i, el) ∈ enumerate(ab)
            resa[i] = findfirst(x -> x == el, a)
        end
    end
    ab = intersect(a, b)
    resa = Vector{Int64}(undef, length(ab))
    findindices!(resa, ab, a)
    # comment lines resa and resb
    resa
    resb = similar(resa)
    findindices!(resb, ab, b)
    resb

    return (ab, resa, resb)
end

"""
create_partition( g , npart )

Returns a pair: (vector of indices making up a partition of 1:size(g) into npart subdomains , the coloring of the dofs)
# Arguments
- 'g' : the graph connections of the degrees of freedom
- 'npart'    : Number of subsets of indices
# Example
m=9
npart = 3
A = spdiagm(-1 => -ones(m-1) , 0 => 2. *ones(m) , 1 => -ones(m-1))
g = Graph(A)
(initial_partition  , decomposition) = create_partition( g , npart )
"""
function create_partition(g, npart)
    decomposition = Metis.partition(g, npart)
    partition_indices = map(i -> findall(x -> (x == i), decomposition), 1:npart)
    return (partition_indices, decomposition)
end





"""
create_partition_DDomain( Domain , g , npart )

Returns a decomposed domain that forms a partition of 1:size(g) into npart subdomains
# Arguments
- 'g' : the graph connections of the degrees of freedom
- 'npart'    : Number of subdomains
# Example
m=9
npart = 3
A = spdiagm(-1 => -ones(m-1) , 0 => 2. *ones(m) , 1 => -ones(m-1))
g = Graph(A)
initial_partition = create_partition_DDomain( g , npart )
"""
function create_partition_DDomain(domain, g, npart)
    # il manque un contrôle sur la cohérence entre les indices de domain et ceux de g et surtout l'expression à donner à ce contrôle.
    (initial_partition, decomposition) = create_partition(g, npart)
    res_up = domain
    res_subdomains = Set{typeof(domain)}()
    for indices ∈ initial_partition
        newsd = Domain(domain, indices)
        push!(res_subdomains, newsd)
    end
    return DDomain(res_up, res_subdomains)
end


"""
inflate_indices( g_adj , indices )

Returns the vector of the indices inflated by its direct neighbors as defined by the adjacency matrix 'g'.
# Arguments
- 'g_adj' : the adjacency matrix of the degrees of freedom with non zero on the diagonal (a square matrix)
- 'indices'    : a set of indices
# Example on initial_partition (a vector of vector of indices)
g_adj =
(inflated_indices , decomposition) = map(sub_id->inflate_indices( g_adj , sub_id ) ,  initial_partition)
"""
function inflate_indices(g_adj, indices::Vector{Int64})
    #trouver les voisins
    (n, m) = size(g_adj)
    vi = zeros(Float64, m)
    vi[indices] .= 1.
    vi = g_adj * vi
    inflated_indices = findall(x -> x > 0, vi)
    return inflated_indices
end



"""
inflate_subdomain!( g_adj , subdomain )

Inflate a 'subdomain' and update the overlap of itself and of its neighbors
# Arguments
- 'g_adj' : the adjacency matrix of some matrix
- 'subdomain'   :  subdomain to be inflated.
"""
function inflate_subdomain!(g_adj, subdomain)
    # up.subdomain == domain ??? A FAIRE
    # g_adj a bien la taille de up.subdomain???
    # que se passe t il avec la partition de l'unité de domain???
    inflated_indices = inflate_subdomain(g_adj, global_indices(subdomain))
    new_indices = filter(x -> !(x in global_indices(subdomain)), inflated_indices)
    append!(global_indices(subdomain), new_indices)#loctoglob est mis a jour, danger creer un nouveau sous domaine plutot?
end



#     POU::DPOU# , en fait c'est plutôt un Shared_vector (cohérent qui en a besoin). En fait, on peut repousser la question POUM à plus tard.
# la chose principale est de pouvoir coder ∑_i R_i^T R_i
###
# vecteur global -> vecteur decompose coherent ( scattered "eclate" mais pas Shared_vector , decomposed DVector ) ->  vecteur decommpose incoherent
# |                   |
# |                   |
# |                   |
# alpha*u ---------> alpha*Du
# A*u -------------> DA*Du
# (u,v) -----------> (Du,Dv)= ∑_i (Ui,Di V_i)
# ASM independant de la partition de l'unite
# vecteur global <- vecteur partage coherent
#
# vecteur partage pas forcement coherent : les rendre cohérent , (cf. RAS)
# somme compensée pour etre plus stable vis a vis des erreurs d'arrondi  -->
#       -->  CF MakeCoherent si partition de l'unite non Booleenne ou Gradient conjugue (aussi???) ?


#################################################
#
#       struct Domain
#
#################################################


#https://docs.julialang.org/en/v1/manual/constructors/
mutable struct Domain# sous domaine aussi
    up::Domain # le (i.e. un seul??) surdomaine éventuellement lui-même
    loctoglob::AbstractVector{Int64} # vecteur d'indices de up qui sont Domain, Int64 pourrait être un paramètre cf indices cartésiens ...
    Domain(loctoglob::AbstractVector{Int64}) = (D = new(); D.loctoglob = copy(loctoglob); D.up = D; return D)
    Domain(up, loctoglob) = issubset(loctoglob, up.loctoglob) ? new(up, loctoglob) : error("indices $loctoglob have to be a subset of the superdomain")
end

# potentiellement dangereux
function global_indices(sd::Domain)
    return sd.loctoglob
end

import Base.length
function length(sd::Domain)
    return length(sd.loctoglob)
end

#################################################
#
#       struct DDomain
#
#################################################


mutable struct DDomain
    up::Domain # le domaine décomposé
    subdomains::Set{Domain} # ensemble des sous domaines
    overlaps::Dict{Domain,Dict{Domain,Tuple{Vector{Int64},Vector{Int64}}}}
    # sd --> (subdomain_vois --> vecteur ( k_loc , k_vois ))
    DDomain(up::Domain, subdomains::Set{Domain}) = (
        res_overlaps = Dict{Domain,Dict{Domain,Tuple{Vector{Int64},Vector{Int64}}}}();
        for sdi ∈ subdomains
            res_overlaps[sdi] = (Dict{Domain,Tuple{Vector{Int64},Vector{Int64}}})()
            for sdj ∈ subdomains
                if (sdi !== sdj)
                    @show (sdisdj, kloc, kvois) = intersectalamatlab(global_indices(sdi), global_indices(sdj))
                    if (!isempty(sdisdj))
                        res_overlaps[sdi][sdj] = (kloc, kvois)
                    end
                end
            end
        end;
        res = new(up, subdomains, res_overlaps);
        return res
    )
end

function subdomains(domain::DDomain)
    return domain.subdomains
end



#################################################
#
#       struct DVector
#
#################################################
mutable struct DVector
    domain::DDomain
    data::Dict{Domain,Vector{Float64}}
    # + , - , a* , .* , similar etc ... si on peut automatiquement hériter de ce qui vient de vecteur, on a gagné voir comment faire en Julia
    # boucles sur eval ??
    # ce qui est lié à l'aspect cohérent : prodscal et donc demande une partition de l'unite qulle quelle soit
    # relation avec les vecteurs "habituels" : import_from_global(!) , export_to_global(!)
end

function subdomains(DVect::DVector)
    return subdomains(DVect.domain)
end

import Base.values
function values(DVect::DVector, sd::Domain)
    return DVect.data[sd]
end


function DVector(ddomain::DDomain, initial_value::Float64)
    data_res = Dict{Domain,Vector{Float64}}()
    for sd ∈ ddomain.subdomains
        data_res[sd] = zeros(Float64, length(global_indices(sd)))
        data_res[sd] .= initial_value
    end
    return DVector(ddomain, data_res)
end

import Base.ones, Base.zeros , Base.rand
for sym in [ :ones , :zeros , :rand ]
    @eval function $(Symbol(string(sym)))(ddomain::DDomain)
        # Float64 should be inferred automatically
        data_res = Dict{Domain,Vector{Float64}}()
        for sd ∈ ddomain.subdomains
            #here as well
            data_res[sd] = $sym( length(global_indices(sd)) )
        end
        return MakeCoherent(DVector(ddomain, data_res))
    end
end

"""
DVector( domain , vecsrc )

Returns a decomposed vector built from a vector
# Arguments
- 'domain' : the decomposed domain
- 'vecsrc' : the classical vector
"""
function DVector(ddomain::DDomain, Usrc)
    if !(length(ddomain.up) == length(Usrc))
        error("Lengthes of decomposed domain and vector must match: $(length(ddomain.up)) is not $(length(Usrc)) ")
    end
    res = DVector(ddomain, 0.0)
    #what if Usrc is already a decomposed vector??
    for sd ∈ ddomain.subdomains
        values(res, sd) .= Usrc[global_indices(sd)]
    end
    return res
end

import  Base.similar
# first try of Metaprogramming
for sym in [ :similar   ]
    @eval function $(Symbol(string(sym)))(a::DVector)
        res = DVector(a.domain, 0.0)
        for sd ∈ subdomains(res)
            values(res, sd) .=  $sym( values(a,sd) )
        end
        return res
    end
end

import Base.copy
function copy(DVec::DVector)
    res = DVector(DVec.domain, 0.0)
    for sdi ∈ subdomains(DVec)
        res.data[sdi] .= values(DVec,sdi)
    end
    return res
end


function dot_op(x::DVector, y::DVector, dot_op)
    if !(x.domain == y.domain)
        error("Domains of both decomposed vectors must be the same")
    end
    res = DVector(x.domain, 0.0)
    for sd ∈ subdomains(res)
        res.data[sd] .= dot_op(x.data[sd], y.data[sd])
    end
    return res
end

# DV1 .* DV2 , iterable venant d'un abstractvector , risque de perdre le //
# Vincent --> ou surcharger broadcast


"""
MakeCoherent( dvector )

Returns a coherent decomposed vector
Using a Boolean partition of unity ensures that the result is roundoff error free and execution order independent
# Argument
- dvector : a decomposed vector
"""
function MakeCoherent(DVect::DVector)
    #Diboolean ensures that the result is roundoff error free
    return Update(dot_op(Diboolean(DVect.domain), DVect, (.*)))
end


"""
MakeCoherent!( dvector )

Makes a decomposed vector coherent
Using a Boolean partition of unity ensures that the result is roundoff error free and execution order independent
# Argument
- dvector : a decomposed vector
"""
function MakeCoherent!(DVect::DVector)
    #Diboolean ensures that the result is roundoff error free
    tmp=copy(DVect)
    Dvect = MakeCoherent(tmp)
end



function DVector2Vector(DVect::DVector)
    Dres = MakeCoherent(DVect)
    ddomain = DVect.domain
    res = zeros(Float64, length(ddomain.up))
    #peu compatible avec une parallelisation
    #il faudrait differencier selon que Diboolean est zero ou non
    for (sd, val) ∈ Dres.data
        res[global_indices(sd)] .= val
    end
    return res
end

"""
returns a decomposed vector R_i ∑_j R_j^T U_j
"""
function Update(DVec::DVector)
    res = DVector(DVec.domain, 0.0)
    for sd ∈ DVec.domain.subdomains
        res.data[sd] .= DVec.data[sd]
    end
    for sd ∈ DVec.domain.subdomains
        for sdvois ∈ DVec.domain.overlaps[sd]
            res.data[sd][sdvois.second[1]] .+= DVec.data[sdvois.first][sdvois.second[2]]
        end
    end
    return res
end


"""
Di( domain )

returns a decomposed vector that corresponds to a multiplicity based partition of unity function
# Arguments
- 'domain' : the support domain
"""
function Di(domain::DDomain)
    tmp = DVector(domain, 1.0)
    multiplicity = Update(tmp)
    res = dot_op(tmp, multiplicity, (./))
    return res
end


"""
Diboolean( domain )

returns a decomposed vector that corresponds to a Boolean partition of unity function
# Arguments
- 'domain' : the support domain
"""
function Diboolean(domain::DDomain)
    res = DVector(domain, 1.0)
    vector_of_subdomains = collect(subdomains(domain))
    for (i, sd) ∈ enumerate(vector_of_subdomains)
        for sdvois ∈ intersect(vector_of_subdomains[i+1:end], collect(keys(domain.overlaps[sd])))
            res.data[sdvois][domain.overlaps[sd][sdvois][2]] .= 0.0
        end
    end
    return res
end


function vuesur(U::DVector)
    for sd ∈ subdomains(U)
        println(values(U, sd))
    end
end



#################################################
#
#       struct DOperator
#
#################################################

# pour effectuer le produit matrice vecteur V_i =  ∑_j R'_i A R_j^T D_j U_j  (V = A U)
mutable struct DOperator
    DDomD::DDomain # domaine de départ décomposé
    DDomA::DDomain # domaine d'arrivée décomposé
    matvec # collection of local operators Aij = R'_i A R_j^T, i ∈ DDomA, j ∈ DDomD
    # ou plutot le produit matrice vecteur avec un DVector qui vit sur DDomD? ??
end



"""
# Arguments
- 'DDomD'
- 'A' : a square matrix given by its entries
"""
function DOperator(DDomD, A)
    DA = Dict{Tuple{Domain,Domain},SparseMatrixCSC{Float64,Int64}}()
    for sdi ∈ subdomains(DDomD)
        for sdj ∈ subdomains(DDomD)
            DA[(sdi, sdj)] = A[global_indices(sdi), global_indices(sdj)]
        end
    end
    function shared_mat_vec( x )
        dom = x.domain
        res = zeros(dom)
        y = zeros(dom)
        #Diboolean pour avoir plus de reproductibilité?
        di = Di( dom )
        for sdi ∈ subdomains( dom )
            values( y , sdi ) .= values(di,sdi) .* values(x,sdi)
        end
        # boucle parallelisable
        for sdi ∈ subdomains( res )
           values(res , sdi) .= DA[(sdi,sdi)]*values(y , sdi)
        end
     # boucle exterieur parallelisable
        for sdi ∈ subdomains( y )
           # boucle sequentiel
           for sdj ∈ subdomains( y )# ou plus efficace et plus clair , passer par les cles avec premier element fixe et deuxieme sdj
              if ( haskey(DA , (sdi,sdj) ) &&  !( sdi == sdj))
                 values(res , sdi) .+= DA[(sdi,sdj)]*values(y , sdj)
              end
           end
        end
        # 1) qu'en est il de la cohérence du vecteur resultat (à faire avec une partition booleenne Dib
        # 2) de l'indépendance vis à vis de l'ordre d'éxecution en // (reproductibilité en cause) 3) vis à vis du séquentiel?
        # Solutions: faire les opérations dans le même ordre ? via un surcoût garantir l'indépendance des résultats vis à vis de l'ordre de la somme (arithmétique compensée)
        return MakeCoherent(res)#manque la reproductibilité
     end
     return DOperator( DDomD , DDomD , shared_mat_vec )
end

#################################################
#
#       struct DOperatorBlockJacobi
#
#################################################

struct DOperatorBlockJacobi
    DDomD::DDomain # domaine de départ et d'arrivée décomposé
    DDomA::DDomain # domaine de départ et d'arrivée décomposé
    matvec # collection of local solvers
end

"""
DOperatorBlockJacobi(DDomD, A)

Returns direct local solvers for the Dirichlet matrices of a global matrix A
# Arguments
- 'DDomD'
- 'A' : a square matrix given by its entries
"""
function DOperatorBlockJacobi(DDomD, A)
    DA_lu = Dict()
    for sdi ∈ subdomains(DDomD)
        DA_lu[sdi] = factorize(A[global_indices(sdi), global_indices(sdi)]  )
    end

    function shared_mat_vec( x )
        dom = x.domain
        res = DVector( dom , 0. )
        # boucle parallelisable , Di a ajouter somewhere
        for sdi ∈ subdomains( res )
            @show values(x , sdi)
           values(res , sdi) .= DA_lu[sdi]\ values(x , sdi)
           @show values(res , sdi)
        end
        return res
     end
     return DOperatorBlockJacobi( DDomD , DDomD , shared_mat_vec )
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

 m = 12
 n = 95
 npart = 2


A = spdiagm(-1 => -ones(m - 1), 0 => 2.0 * ones(m), 1 => -ones(m - 1))
Omega = Domain(1:m)
# A = Laplacian2d(m, n, 1, 1);
# Omega = Domain(1:m*n)

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

my_very_first_Di = Di(my_very_first_DDomain)

zzz = Update(dot_op(my_very_first_Di, my_very_first_DVect, (.*)))


vuesur(zzz)

vuesur(Diboolean(my_very_first_DDomain))

vuesur(Update(dot_op(Diboolean(my_very_first_DDomain), my_very_first_DVect, (.*))))

aaa = collect(1:length(Omega))

daaa = DVector(my_very_first_DDomain, aaa)

vuesur(Update(dot_op(Diboolean(my_very_first_DDomain), daaa, (.*))))

DVector2Vector(daaa)

vtest = rand(length(Omega))
norm(vtest .- DVector2Vector(DVector(my_very_first_DDomain, vtest)))

DA = DOperator(my_very_first_DDomain , A)

DA.matvec(daaa)

function test_mat_vec( A , v , domain )
    Dv = DVector(domain,v)
    DA = DOperator(domain , A)
    DVector2Vector(DA.matvec(Dv))-A*v
end


test_mat_vec(A,aaa,my_very_first_DDomain)
test_mat_vec(A,rand(length(my_very_first_DDomain.up)),my_very_first_DDomain)

@test norm(test_mat_vec(A,rand(length(my_very_first_DDomain.up)),my_very_first_DDomain)) < 1.e-11

Am1=DOperatorBlockJacobi(my_very_first_DDomain , A)
#Am1.matvec(daaa)


####### RAS iteratif  ###################
# erreur qq part
b = ones(length(Omega))
solex=A\b
sol = zeros(length(Omega))
itmax = 100
dsol = DVector(my_very_first_DDomain,sol)
dres = zeros(my_very_first_DDomain)
db = DVector(my_very_first_DDomain,b)

for it in 1:itmax
    global dsol , dres
    dres = dot_op( db , DA.matvec(dsol) , (-))
    println("Norme du vrai residu " , norm( b-A*DVector2Vector(dsol) ) )
    # correction
    dcor = Am1.matvec(dres)
    vuesur(dcor)
    MakeCoherent!(dcor)
    dsol = dot_op(dsol , dcor , (+) )
#    plot!(DVector2Vector(dsol))
end



# faire DOperator
# faire des tests
# paralleliser
# commenter
# a nettoyer,
# a encapsuler, trop de references aux membres des structures???
