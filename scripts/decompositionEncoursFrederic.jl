#################################################
#
# 2021-08-18  Frederic Nataf
#
#################################################

module decomposition
export create_partition , inflate_subdomain , Subdomain , ndof , not_responsible_for ,
responsible_for_others , global_indices , create_partition_subdomain , who_is_responsible_for_who
#, inflate_subdomain!

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
(initial_partition  , decomposition) = create_partition( g , npart )
"""
function create_partition( g , npart )
    decomposition = Metis.partition(g, npart)
    subdomain_partition_indices = map( i-> findall( x-> (x==i) , decomposition ) , 1:npart )
    return ( subdomain_partition_indices , decomposition)
end


"""
inflate_subdomain( g_adj , subdomain_indices )

Returns the vector of the indices of the subdomain inflated by its direct neighbors as defined by the adjacency matrix 'g'.
# Arguments
- 'g_adj' : the adjacency matrix of the degrees of freedom with non zero on the diagonal (a square matrix)
- 'subdomain_indices'    : Indices of a subdomain
# Example
g_adj = adjacency_matrix(g ,  Int64 )
(inflated_subdomains , decomposition) = map(sd->inflate_subdomain( g_adj , sd ) ,  initial_partition)
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

# est un vecteur global commun à une decomposition et qui dit qui est responsable de qui
# à partir de cela et de la liste des indices du sous domaine, on peut construire
# les champs qui existaient déjà.
mutable struct Subdomain
    who_is_responsible_for_who::Vector{Int64}# vecteur global commun à une decomposition et qui dit qui est responsable de qui stocke en numérotation globale. A la place de Int64 ne faudrait il pas mettre Subdomain????
    not_responsible_for_indices::Vector{Int64}
    loctoglob::Vector{Int64}# utile seulement pour la création du recouvrement (semble t il)
    not_responsible_for::Dict{Subdomain, Vector{Tuple{Int64, Int64}}} # sdrespo du responsable -> vecteur de pairs (local number , distant number in sdrespo)
    responsible_for_others::Dict{Int64, Vector{Tuple{Subdomain, Int64}}}  # k_loc -> vecteur de pairs ( subdomain_vois , k_loc_chezvois )  dupliquant le degré de liberté k_loc, more or less imposes the way to iterate in function Update.
end

function ndof( sbd::Subdomain )
    return length(sbd.loctoglob)
end

function not_responsible_for_indices( sbd::Subdomain )
    return sbd.not_responsible_for_indices
end

function who_is_responsible_for_who( sbd::Subdomain )
    return sbd.who_is_responsible_for_who
end

function glob_to_loc( sbd::Subdomain , kglob )
    findfirst( x->x==kglob , global_indices( sbd ) )
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
    ( initial_partition , decomposition ) = create_partition( g , npart )
    res = Subdomain[]
    i = 1
    for indices ∈ initial_partition
        newsd = Subdomain( decomposition , Int64[] , indices , Dict() , Dict() )
        push!( res , newsd )
    end
    return res
end




end# du module
