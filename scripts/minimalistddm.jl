#################################################
#
# 2021-12-12  Frederic Nataf
#
#################################################
# essai de partir du point de vue utilisateur
# voir les notes manuscrites
# il existe qq part une numérotation globale
# Analogie avec MPI (MPICommWorld , Communicators , subcommunicators , etc ... ) serait utile????
# processus -> un sous ensemble de degres de liberté
# Communicators (des processus et des moyens de communiquer) ->  des degrés de liberté == sous domaine
# MPICommWorld -> l'ensemble des degrés de liberté => vision rigide car le nombre de dof (processus) varie au cours du calcul


using SparseArrays , LightGraphs , GraphPlot , Metis , LinearAlgebra, ThreadsX





"""
create_partition_DDomain( Domain , g , npart )

Returns decomposed domain that form a partition of 1:size(g) into npart subdomains
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
function create_partition_DDomain( domain , g , npart )
    # il manque un contrôle sur la cohérence entre les indices de domain et ceux de g et surtout l'expression à donner à ce contrôle.
    ( initial_partition , decomposition ) = create_partition( g , npart )
    res_up = domain
    res_subdomains = Set{typeof(domain)}()
    res_neighborhood = Dict{typeof(domain),Vector{Tuple{Int64,Int64}}}()
    for indices ∈ initial_partition
        newsd = Domain( domain , indices )
        push!( res_subdomains , newsd )
    end
    return DDomain( res_up , res_subdomains , res_neighborhood )
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
function create_partition( g , npart )
    decomposition = Metis.partition(g, npart)
    partition_indices = map( i-> findall( x-> (x==i) , decomposition ) , 1:npart )
    return ( partition_indices , decomposition );
end



"""
inflate_indices( g_adj , indices )

Returns the vector of the indices inflated by its direct neighbors as defined by the adjacency matrix 'g'.
# Arguments
- 'g_adj' : the adjacency matrix of the degrees of freedom with non zero on the diagonal (a square matrix)
- 'indices'    : a set of indices
# Example on initial_partition (a vector of vector of indices)
g_adj = adjacency_matrix( g ,  Int64 )
(inflated_indices , decomposition) = map(sub_id->inflate_indices( g_adj , sub_id ) ,  initial_partition)
"""
function inflate_indices( g_adj , indices::Vector{Int64} )
    #trouver les voisins
    (n,m) = size(g_adj)
    vi=zeros(Int64,m)
    vi[indices] .= 1
    vi = g_adj*vi
    inflated_indices = findall(x->x>0,vi)
    return inflated_indices
end



"""
inflate_subdomain!( g_adj , subdomain , ddomain )

Inflate a 'subdomain' and updates the overlap of itself and of its neighbors
# Arguments
- 'g_adj' : the adjacency matrix of some matrix
- 'subdomain'   :  subdomain to be inflated
- 'ddomain'  :  a 'decomposed domain' that contains 'subdomain'
"""
function inflate_subdomain!( g_adj , subdomain , ddomain )
    # up.subdomain == domain ??? A FAIRE
    # ici, on ne suppose pas partir d'une partition probablement lourd
    # si on veut faire une version qui ressemble à celle dans decompositionparallel.jl, il
    # faut garder une centralisation de qui est à qui.
    # on a pour une decomposition donnée un vecteur global qui à chaque degre de liberte, donne l'ensemble des sous domaines qui le possedent. C'est stocké chez DDomain qui doit donc être adapté. 
    inflated_indices = inflate_subdomain( g_adj , global_indices(subdomain) )
    new_indices = filter(x -> !(x in global_indices(subdomain)), inflated_indices)
    append!( global_indices(subdomain) ,  new_indices )#loctoglob est mis a jour What if range??
    append!( decomposition.not_responsible_for_indices(subdomain) ,  new_indices )
    # il reste mettre à jour not_responsible_for chez soi
    # et responsible_for_others chez les responsables
    for kglob ∈ new_indices
        kloc = decomposition.glob_to_loc( subdomain , kglob  )
        sdrespo = subdomains[who_is_responsible_for_who(subdomain)[kglob]]
        kvois = decomposition.glob_to_loc( sdrespo , kglob  )
        # mise a jour du responsable
        if haskey( not_responsible_for( subdomain ) , sdrespo )
            push!( not_responsible_for( subdomain )[sdrespo] , ( kloc , kvois ) )
        else
            not_responsible_for( subdomain )[sdrespo] = [( kloc , kvois )]
        end
        #      responsible_for_others
        push!( responsible_for_others( sdrespo ) , ( kvois , subdomain , kloc ) )
    end
end


mutable struct Shared_vector
    domain::DDomain
    vectors::Dict{DDomain, Vector{Float64}}
    # + , - , a* , .* , similar etc ... si on peut automatiquement hériter de ce qui vient de vecteur, on a gagné voir comment faire en Julia
    # ce qui est lié à l'aspect cohérent : prodscal et donc demande une partition de l'unite qqsoit
    # relation avec les vecteurs "habituels" : import_from_global(!) , export_to_global(!)

    #
end

#     POU::DPOU# , en fait c'est plutôt un Shared_vector (cohérent qui en a besoin). En fait, on peut repousser la question POUM à plus tard. la chose principale est de pouvoir coder ∑_i R_i^T R_i


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
# vecteur partage pas forcement coherent : les rendre cohérent , (RAS)
# somme compensée pour etre plus stable vis a vis des erreurs d'arrondi  --> CF MakeCoherent si partition de l'unite non Booleenne ou Gradient conjugue (aussi???) ?


#https://docs.julialang.org/en/v1/manual/constructors/
mutable struct Domain# sous domaine aussi
    up::Domain # le (i.e. un seul??) surdomain éventuellement lui-même
    loctoglob::AbstractVector{Int64} # vecteur d'indices de up qui sont Domain, Int64 pourrait être un paramètre cf indices cartésiens ...
    Domain(loctoglob::AbstractVector{Int64}) = ( D = new(); D.loctoglob = copy(loctoglob) ; D.up = D; return D; )
    Domain(up,loctoglob) =  issubset(loctoglob,up.loctoglob) ?  new(up,loctoglob) : error("indices $loctoglob have to be a subset of the superdomain")
end

function global_indices( sd::Domain)
    return sd.loctoglob
end


# pour effectuer le produit matrice vecteur V_i =  ∑_j R'_i A R_j^T D_j U_j  (V = A U)
mutable struct DDomain
    up::Domain # le domaine que l'on décompose
    subdomains::Set{Domain} # ensemble des sous domaines
    neighborhood::Dict{Domain,Vector{Tuple{Int64,Int64}}}
    # subdomain_vois --> vecteur ( k_loc , k_vois )
end

mutable struct DOperator
    DDomD::DDomain # domaine de départ décomposé
    DDomA::DDomain # domaine d'arrivée décomposé
    matvec # collection of local operators Aij = R'_i A R_j^T, i ∈ DDomA, j ∈ DDomD
end


#    Di  # la partition de l'unité locale au sous domaine vue comme un operateur local verifiant une certaine propriété
#  Di est définie sur un DDomain (constructeur) et on l'interroge en luio donnant un sous domaine et il renvoie le vecteur des poids D(Omage_i)
mutable struct DPOU
   Ddomain::DDomain
 #  constructeur(DDomain,options)
 #  DPOU(DDomain) renvoie le vecteur des poids sur le domaine en question
end
#RAS: constructeur(DDomain, A , POU) , la POU est associée à un algorithme plus qu'à une décomposition si argument POU absent, on prend les Di implicite à ???



#
#   Create domain initial_decomposition => DDomain
#
#   Create parition of unity  => DDomain aussi ???
#
#   Create decomposed Operator  => DA
#
#
# premiers tests
m=9
Omega = Domain(1:m)
npart = 3
A = spdiagm(-1 => -ones(m-1) , 0 => 2. *ones(m) , 1 => -ones(m-1))
g = Graph(A)
(initial_partition  , decomposition) = create_partition( g , npart )
g_adj = adjacency_matrix( g ,  Int64 )
(inflated_indices , decomposition) = map(sub_id->inflate_indices( g_adj , sub_id ) ,  initial_partition)

DomDecPartition = create_partition_DDomain( Omega , g , npart )
