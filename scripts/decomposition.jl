#################################################
#
# 2021-08-18  Frederic Nataf
#
#################################################

module decomposition
export create_partition , inflate_subdomain , Subdomain , ndof , not_responsible_for ,
responsible_for_others , global_indices , create_partition_subdomain , inflate_subdomain!

using SparseArrays , LightGraphs , GraphPlot , Metis , LinearAlgebra



"""
create_partition( g , npart )

Returns a vector of the subdomain indices of a partition of 1:size(g) into npart subdomains
# Arguments
- 'g' : the graph connections of the degrees of freedom
- 'npart'    : Number of subdomains
# Example
m=9
npart = 3
A = spdiagm(-1 => -ones(m-1) , 0 => 2. *ones(m) , 1 => -ones(m-1))
g = Graph(A)
initial_partition = create_partition( g , npart )
"""
function create_partition( g , npart )
    decomposition = Metis.partition(g, npart)
    subdomain_partition_indices = map( i-> findall( x-> (x==i) , decomposition ) , 1:npart )
    return subdomain_partition_indices
end


"""
inflate_subdomain( g_adj , subdomain_indices )

Returns the vector of the indices of the subdomain inflated by its direct neighbors as defined by the adjacency matrix 'g'.
# Arguments
- 'g_adj' : the adjacency matrix of the degrees of freedom with non zero on the diagonal (a square matrix)
- 'subdomain_indices'    : Indices of a subdomain
# Example
g_adj = adjacency_matrix(g ,  Int64 )
inflated_subdomains = map(sd->inflate_subdomain( g_adj , sd ) ,  initial_partition)
 voir inflate_subdomain2( sd ) pour une version composable
"""
function inflate_subdomain( g_adj , subdomain_indices::Vector{Int64} )
    #trouver les voisins
    (n,m) = size(g_adj)
    vi=zeros(Int64,m)
    vi[subdomain_indices] .= 1
    vi = g_adj*vi
    inflated_subdomain_indices = findall(x->x>0,vi)
    return inflated_subdomain_indices
end

mutable struct Subdomain
    loctoglob::Vector{Int64}# utile seulement pour la création du recouvrement (semble t il)
    not_responsible_for::Dict{Subdomain, Vector{Tuple{Int64, Int64}}} # sdvois du responsable -> vecteur de pairs (local number , distant number in sdvois)
    responsible_for_others::Dict{Int64, Vector{Tuple{Subdomain, Int64}}}  # k -> vecteur de pairs ( subdomain_vois , k_loc_chezvois )  dupliquant le degré de liberté k, more or less imposes the way to iterate in function Update.
end

function ndof( sbd::Subdomain )
    return length(sbd.loctoglob)
end

function not_responsible_for( sbd::Subdomain )
    return sbd.not_responsible_for
end

function responsible_for_others( sbd::Subdomain )
    return sbd.responsible_for_others
end

function global_indices( sbd::Subdomain )
    return sbd.loctoglob
end

# le plus simple est de faire une structure domaine qui soit assez fourre-tout pour être souple et créer après les sous structures pratiques pour les mises à jour.
# le vecteur who_is_responsible_for sera partagé par tous les sous domaines et permettra de créer les fameuses sous structures pratiques pour les mises à jour.
# cela devrait permettre de garder les code Update! qui a déjà été écrit. 

# proposition de version passant par la numérotation globale
mutable struct Subdomain2
    loctoglob::Vector{Int64}
    globtoloc::Vector{Int64}#the reciprocal of loctoglob
    not_responsible_for::Dict{Subdomain, Vector{Int64}} # sdvois d'un responsable -> vecteur de global numberings (one way to indicate who is responsible)
    responsible_for_others::Dict{Int64, Vector{Subdomain}}  # k_glob -> vecteur de subdomain_vois dupliquant le degré de liberté k
end


"""
create_partition_subdomain( g , npart )

Returns a vector of subdomains that form a partition of 1:size(g) into npart subdomains
# Arguments
- 'g' : the graph connections of the degrees of freedom
- 'npart'    : Number of subdomains
# Example
m=9
npart = 3
A = spdiagm(-1 => -ones(m-1) , 0 => 2. *ones(m) , 1 => -ones(m-1))
g = Graph(A)
initial_partition = create_partition_subdomain( g , npart )
"""
function create_partition_subdomain( g , npart )
    initial_partition = create_partition( g , npart )
    res = Subdomain[]
    for indices ∈ initial_partition
        push!( res , Subdomain( indices , Dict() , Dict() ) )
    end
    return res
end

"""
inflate_subdomain!( g_adj , subdomain , subdomains )

Returns the vector of the subdomains where subdomain was inflated by its direct neighbors as defined by the adjacency matrix 'g'.
# Arguments
- 'g_adj' : the adjacency matrix of the degrees of freedom with non zero on the diagonal (a square matrix)
- 'subdomain'    : the subdomain to be inflated
- 'subdomains' : the set of all the subdomains
# Example
g_adj = adjacency_matrix(g ,  Int64 )
Subdomain ?????
inflated_subdomain!( g_adj , subdomain , create_partition_subdomain( g , npart ) )
"""

function inflate_subdomain!( g_adj , subdomain , subdomains )
# indices of the inflated subdomain, the new ones will be put last (otherwise numberings in neighbors have to be modified ??)
# ici, on est numérotation globale
    inflated_indices = inflate_subdomain( g_adj , global_indices(subdomain) )
    new_indices = filter(x -> !(x in indices(subdomain)), inflated_indices)
    append!( indices(subdomain) ,  new_indices )
# il reste mettre à jour not_responsible_for chez soi
# et responsible_for_others chez les responsables
#   for k ∈ new_indices
#       trouver le sous domaine sdvois responsable de k
#       chez sdvois:  agrandir le vecteur responsible_for_others avec la paire (sdloc, )

end

# puisqu'il n'y a qu'un seul et unique responsable et que les charges ont peu de chande de changer, cela fait sens de stocker le vecteur correspondant global_number -> son responsable

end
