include("decompositionEncoursFrederic.jl")
# faire deux fois include("decompositionEncoursFrederic.jl") semble poser problème
using SparseArrays , LightGraphs , GraphPlot , Metis , LinearAlgebra , .decomposition


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
        if haskey( responsible_for_others( sdrespo ) , kvois )
            push!( responsible_for_others( sdrespo )[kvois] , ( subdomain , kloc ) )
        else
            responsible_for_others( sdrespo )[kvois] = [( subdomain , kloc )]
        end
    end
end




######################################################
mutable struct Shared_vector
    values::Dict{Subdomain, Vector{Float64}}
end

function ndof( U::Shared_vector )
    res = 0
    for sd ∈ subdomains( U )
        res += ndof_responsible_for( sd )
    end
    return res
end

# faire make coherent avant ou dedans??
function export_to_global( U::Shared_vector )
    res = zeros( ndof( U ) )
    for sd ∈ subdomains( U )
        res[ global_indices( sd ) ] .= values( U , sd )
    end
    return res
end

function import_from_global!( U::Shared_vector , uglob::Vector{Float64} )
    for sd ∈ subdomains( U )
        values( U , sd )  .=  uglob[ global_indices(sd) ]
    end
end


function subdomains(U::Shared_vector)
    return keys(U.values)
end

function values(  U::Shared_vector , sd::Subdomain  )
    return U.values[sd]
end


# Phase 2 de update, les non responsables vont lire les valeurs chez le responsable
function MakeCoherent!( U::Shared_vector )
    for sd ∈ subdomains( U )
        for ( sdneigh , numbering ) ∈ not_responsible_for( sd )
            #            MakeCoherent( U , sd , sdneigh , numbering )
            for (k,l) ∈ numbering
                values(U,sd)[k] = values(U,sdneigh)[l]
            end
        end
    end
end

# Phase 1 de update
# return what???
function Update_responsible_for_others!( U::Shared_vector )
    for sd ∈ subdomains( U )
        for k ∈ keys( responsible_for_others( sd ) )
            for (sdvois , kvois) ∈ responsible_for_others( sd )[k]
                values(U,sd)[k] += values(U,sdvois)[kvois]
            end
        end
    end
end

# return what???
function Update_wo_partition_of_unity!( U::Shared_vector )
    Update_responsible_for_others!( U )
    MakeCoherent!( U::Shared_vector )
end






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
        if haskey( responsible_for_others( sdrespo ) , kvois )
            push!( responsible_for_others( sdrespo )[kvois] , ( subdomain , kloc ) )
        else
            responsible_for_others( sdrespo )[kvois] = [( subdomain , kloc )]
        end
    end
end


function vuesur( U::Shared_vector )
    for sd ∈ subdomains( U )
@show        println(values( U , sd ))
    end
end


###############@ TESTS BASIQUES ####################################


# necesaire mais incomprehensible ???? 
function ndof( sbd::Subdomain )
    return length(sbd.loctoglob)
end

m=9
npart = 3
A = spdiagm(-1 => -ones(m-1) , 0 => 2. *ones(m) , 1 => -ones(m-1))
g = Graph(A)
initial_decomposition = create_partition_subdomain( g , npart )
g_adj = adjacency_matrix(g ,  Int64 )


initial_decomposition = create_partition_subdomain( g , npart )

inflate_subdomain!( g_adj , initial_decomposition[1] , initial_decomposition )

inflate_subdomain!( g_adj , initial_decomposition[1] , initial_decomposition )

initial_decomposition



Vshareddict = Dict{Subdomain,Vector{Float64}}()
for sd ∈   initial_decomposition
    Vshareddict[sd] = ones(ndof(sd))
end

Vshared = Shared_vector(Vshareddict)


Update_wo_partition_of_unity!(Vshared)

vuesur( Vshared )


# ecrire test de validation
# changer la forme du dict
# passer les dict en vecteur??
# faire une structure plus allégée??

for sd ∈ subdomains(Vshared)
    println("ndof = ", ndof(sd))
    values( Vshared , sd ) .= ones(ndof(sd))
end



uglob = 4. * ones(m)

import_from_global!( Vshared , uglob )

vuesur( Vshared )


Update_wo_partition_of_unity!(Vshared)
