#################################################
#
# 2021-08-18  Frederic Nataf
#
#################################################

module decomposition
export  Subdomain , ndof , not_responsible_for ,
responsible_for_others , global_indices , create_partition_subdomain , who_is_responsible_for_who , ndof_responsible_for ,
neighborhood ,  buffer_responsible_for_others , values ,
inflate_subdomain! , Shared_vector , subdomains , MakeCoherent! , import_from_global! , import_from_global , export_to_global , export_to_global! , vuesur ,
Update_wo_partition_of_unity! , Update_wi_partition_of_unity! , create_buffers_communication! ,
Domain

using SparseArrays , LightGraphs , GraphPlot , Metis , LinearAlgebra, ThreadsX

"""
create_partition( g , npart )

Returns a pair: vector of the subdomain indices of a partition of 1:size(g) into npart subdomains AND the coloring of the dofs
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
    return ( subdomain_partition_indices , decomposition );
end


"""
inflate_subdomain( g_adj , subdomain_indices )

Returns the vector of the indices of the subdomain inflated by its direct neighbors as defined by the adjacency matrix 'g'.
# Arguments
- 'g_adj' : the adjacency matrix of the degrees of freedom with non zero on the diagonal (a square matrix)
- 'subdomain_indices'    : Indices of a subdomain
# Example on a vector of subdomains
g_adj = adjacency_matrix( g ,  Int64 )
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


#################################################
#
#       struct Subdomain
#
#################################################

mutable struct Subdomain
    who_is_responsible_for_who::Vector{Int64}
    # vecteur global commun à une decomposition et qui dit qui est responsable de qui stocke en numérotation globale. A la place de Int64 ne faudrait il pas mettre Subdomain????
    not_responsible_for_indices::Vector{Int64}
    # utile seulement pour la création du recouvrement (semble t il) i.e. ghost zone
    loctoglob::Vector{Int64}
    # auto explanatory
    not_responsible_for::Dict{Subdomain, Vector{Tuple{Int64 , Int64}}}# a changer après en Vector(triplet comme responsible_for_others)
    # sdrespo du responsable -> vecteur de pairs (local number , distant number in sdrespo)
    responsible_for_others::Vector{Tuple{ Int64 , Subdomain , Int64}}
    # ajoute t on un champ responsible_for??
    # Vector  ( k_loc , subdomain_vois , k_loc_chezvois )
    #responsible_for_others::Dict{Int64, Vector{Tuple{Subdomain, Int64}}}  # k_loc -> vecteur de pairs ( subdomain_vois , k_loc_chezvois )  dupliquant le degré de liberté k_loc, more or less imposes the way to iterate in function Update.
    buffer_responsible_for_others::Dict{Subdomain,Vector{Float64}}# ne sert plus à rien en fait A supprimer??
    #subdomain_vois -> vecteur ( value )
    neighborhood::Dict{Subdomain,Vector{Tuple{Int64,Int64}}}
    # subdomain_vois > vecteur ( k_loc , k_loc ) donne la manière de parcourir les buffers de communications??? A cOMPLETER
end

"""
ndof( subdomain )

Returns the number of degrees of freedom of a 'subdomain' including its overlap
"""
function ndof( sbd::Subdomain )
    return length(sbd.loctoglob)
end

"""
ndof_responsible_for( subdomain )

Returns informations on the dofs the subdomain is not responsible for
"""
function ndof_responsible_for( sbd::Subdomain )
    return ndof(sbd) - length( not_responsible_for_indices(sbd) )
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

function buffer_responsible_for_others( sbd::Subdomain )
    return sbd.buffer_responsible_for_others
end

function neighborhood( sbd::Subdomain )
    return sbd.neighborhood
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
    for indices ∈ initial_partition
        newsd = Subdomain( decomposition , Int64[] , indices , Dict() , Vector[] , Dict{Subdomain,Vector{Float64}}() , Dict{Subdomain,Vector{Tuple{Int64,Int64}}}() )
        push!( res , newsd )
    end
    return res
end


"""
inflate_subdomain!( g_adj , subdomain , subdomains )

Inflate a 'subdomain' and updates the data structure of itself and of its neighbors but not the communicatin buffers(!)
# Arguments
- 'g_adj' : the adjacency matrix of some matrix
- 'subdomain'   :  subdomain to be inflated
- 'subdomains'  :  a group of 'subdomains' that contains 'subdomain'
"""
function inflate_subdomain!( g_adj , subdomain , subdomains )
    # indices of the inflated subdomain, the new ones will be put last (otherwise numberings in neighbors have to be modified ??)
    # ici, on est en numérotation globale
    inflated_indices = inflate_subdomain( g_adj , global_indices(subdomain) )
    new_indices = filter(x -> !(x in global_indices(subdomain)), inflated_indices)
    append!( global_indices(subdomain) ,  new_indices )#loctoglob est mis a jour
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

function create_buffers_communication!( sbd::Subdomain )
    # liste des sousdomaines voisins dependants
    # pour combien de points
#    sdvois_size_ovlp = Dict{Subdomain,Int64}()
    for (k_loc , sdvois , k_loc_chezvois) ∈ responsible_for_others( sbd )
        if haskey( neighborhood( sbd ) , sdvois )
#            sdvois_size_ovlp[sdvois]  += 1
            push!( decomposition.neighborhood( sbd )[sdvois]  , ( k_loc , k_loc_chezvois ) )
        else
#            sdvois_size_ovlp[sdvois]  = 1
            decomposition.neighborhood( sbd )[sdvois]  = [(k_loc , k_loc_chezvois)]
        end
    end
end


function create_buffers_communication2!( sbd::Subdomain )
    # liste des sousdomaines voisins dependants
    # pour combien de points
    sdvois_size_ovlp = Dict{Subdomain,Int64}()
    for (k_loc , sdvois , k_loc_chezvois) ∈ responsible_for_others( sbd )
        if haskey( sdvois_size_ovlp , sdvois )
            sdvois_size_ovlp[sdvois]  += 1
            push!( decomposition.neighborhood( sbd )[sdvois]  , ( k_loc , k_loc_chezvois ) )
        else
            sdvois_size_ovlp[sdvois]  = 1
            decomposition.neighborhood( sbd )[sdvois]  = [(k_loc , k_loc_chezvois)]
        end
    end
    for ( sdvois , ndof_vois_ovlp ) ∈ sdvois_size_ovlp
        buffer_responsible_for_others( sbd )[ sdvois ] = zeros( ndof_vois_ovlp )
    end
end


#################################################
#
#       struct Domain
#
#################################################
mutable struct Domain
    subdomains::Vector{Subdomain}#Vector, Set ???
end

"""
ndof( U::Domain )

Returns the number of dofs in the domain, duplicated unknowns count for one only
"""
function ndof( U::Domain )
    res = 0
    for sd ∈ subdomains( U )
        res += ndof_responsible_for( sd )
    end
    return res
end


"""
subdomains( U::Domain )

Returns the subdomains that compose the domain 'U'
"""
function subdomains(U::Domain)
    return U.subdomains
end




#################################################
#
#       struct Shared_vector
#
#################################################


mutable struct Shared_vector
    values::Dict{Subdomain, Vector{Float64}}
end

"""
ndof( U::Shared_Vector )

Returns the number of dofs in the shared vector, duplicated unknowns count for one only
"""
function ndof( U::Shared_vector )
    res = 0
    for sd ∈ subdomains( U )
        res += ndof_responsible_for( sd )
    end
    return res
end


"""
subdomains( U::Shared_vector )

Returns the subdomains that support the shared vector 'U'
"""
function subdomains(U::Shared_vector)
    return keys(U.values)
end

"""
values( U::Shared_vector , subdomain )

Returns the local vector of 'subdomain'
"""
function values(  U::Shared_vector , sd::Subdomain  )
    return U.values[sd]
end


# return what???
"""
Update_wo_partition_of_unity2!( U::Shared_vector )

Performs the operation ``(U_i)_{1 ≤ i ≤ N} -> (U_i + R_i ∑_j R_j^T U_j)_{1 ≤ i ≤ N}``
"""
function Update_wo_partition_of_unity2!( U::Shared_vector )
    # Est on vraiment obligé de faire trois opérations. Fusionner Fetch et Update pour éviter les buffers à stocker.
    Fetch_not_responsible_for2!( U )
    Update_responsible_for_others2!( U )
    # @sync en parallèle ou bien dépendance de tâches , cf starpu.jl
    MakeCoherent!( U )
end


"""
Update_wo_partition_of_unity!( U::Shared_vector )

Performs the operation ``(U_i)_{1 ≤ i ≤ N} -> (R_i ∑_j R_j^T U_j)_{1 ≤ i ≤ N}``
"""
function Update_wo_partition_of_unity!( U::Shared_vector )
    Update_responsible_for_others!( U )
    # @sync en parallèle ou bien dépendance de tâches , cf starpu.jl
    MakeCoherent!( U )
end


# return what???
"""
Update_wi_partition_of_unity!( U::Shared_vector )

Performs the operation ``(U_i)_{1 ≤ i ≤ N} -> (R_i ∑_j R_j^T D_j U_j)_{1 ≤ i ≤ N}``
"""
function Update_wi_partition_of_unity!( U::Shared_vector )
    MakeCoherent!( U )
end

# le responsable va lire les valeurs venant des d.d.l. dupliquées
function Fetch_not_responsible_for2!( U::Shared_vector )
    # remplissage des buffers
    ThreadsX.foreach(subdomains( U )) do sd  #for sd ∈ subdomains( U )
        for ( sdvois , fecthed_values ) ∈ buffer_responsible_for_others( sd )
            for ( k , ( k_loc , k_loc_chezvois ) ) ∈ enumerate( decomposition.neighborhood( sd )[ sdvois ])
                fecthed_values[k] = decomposition.values(U,sdvois)[k_loc_chezvois]
            end
        end
    end
end


# le responsable accumule les valeurs venant des d.d.l. dupliquées
function Update_responsible_for_others2!( U::Shared_vector )
    ThreadsX.foreach(subdomains( U )) do sd  #for sd ∈ subdomains( U )
        for sdvois ∈ keys( buffer_responsible_for_others( sd ) )
            for ( val , ( kloc , klocchezvois ) ) ∈ zip( buffer_responsible_for_others( sd )[ sdvois ] , decomposition.neighborhood( sd )[ sdvois ] )
                decomposition.values(U,sd)[kloc] += val
            end
        end
    end
end

# Phase 1 de update
function Update_responsible_for_others!( U::Shared_vector )
    ThreadsX.foreach(subdomains( U )) do sd #   for sd ∈ subdomains( U )
        for sdvois ∈ keys( decomposition.neighborhood( sd ) )
            for ( kloc , klocchezvois ) ∈ decomposition.neighborhood( sd )[ sdvois ]
                decomposition.values(U,sd)[kloc] += decomposition.values(U,sdvois)[klocchezvois]
            end
        end
    end
end



# Phase 2 de update, les non responsables vont lire les valeurs chez le responsable
function MakeCoherent!( U::Shared_vector )
    ThreadsX.foreach(subdomains( U )) do sd  #     for sd ∈ subdomains( U )
        for ( sdneigh , numbering ) ∈ not_responsible_for( sd )
            #            MakeCoherent( U , sd , sdneigh , numbering )
            for (k,l) ∈ numbering
                values(U,sd)[k] = values(U,sdneigh)[l]
            end
        end
    end
end


"""
import_from_global!( U::Shared_vector , uglob::Vector{Float64} )

Performs the operation ``(U_i = R_i uglob)_{1 ≤ i ≤ N}``
"""
function import_from_global!( U::Shared_vector , uglob::Vector{Float64} )
    ThreadsX.foreach(subdomains( U )) do sd  #for sd ∈ subdomains( U )
        values( U , sd )  .=  uglob[ global_indices(sd) ]
    end
end

"""
import_from_global( domain::Domain , uglob::Vector{Float64} )

Returns a shared vector ``(R_i uglob)_{1 ≤ i ≤ N}``
"""
function import_from_global( domain::Domain , uglob::Vector{Float64} )
    values = Dict{Subdomain, Vector{Float64}}()
    for sd ∈ subdomains( domain )
        values[sd] = zeros(ndof(sd))
        values[sd] .= uglob[ global_indices(sd ) ]
    end
    return Shared_vector( values )
end



# faire make coherent avant ou dedans??
"""
export_to_global( U::Shared_vector )

Returns the global vector uglob such that  ``(R_i uglob = U_i)_{1 ≤ i ≤ N}``
"""
function export_to_global( U::Shared_vector )
    # pour clarifier la fonction n'exporter que les données dont on est responsable
    res = zeros( ndof( U ) )
    ThreadsX.foreach(subdomains( U )) do sd  #for sd ∈ subdomains( U )
        #        for sd ∈ subdomains( U )
        res[ global_indices( sd ) ] .= values( U , sd )
    end
    return res
end

"""
export_to_global!( u::Vector{Float64} , U::Shared_vector )

Performs the operation ``(u = U_i)_{1 ≤ i ≤ N}`` for the indices i is responsible for
"""
function export_to_global!( u::Vector{Float64} , U::Shared_vector )
    # pour clarifier la fonction n'exporter que les données dont on est responsable
    # TBD check the sizes
    ThreadsX.foreach(subdomains( U )) do sd  #for sd ∈ subdomains( U )
        #        for sd ∈ subdomains( U )
        u[ global_indices( sd ) ] .= values( U , sd )
    end
end





function vuesur( U::Shared_vector )
    for sd ∈ subdomains( U )
        println(values( U , sd ))
    end
end





end# du module
