include("decompositionEncoursFrederic.jl")
using SparseArrays , LightGraphs , GraphPlot , Metis , LinearAlgebra , .decomposition



m=9
npart = 3
A = spdiagm(-1 => -ones(m-1) , 0 => 2. *ones(m) , 1 => -ones(m-1))

# test des fonctions de decomposition de domaines

g = Graph(A)
initial_decomposition = create_partition_subdomain( g , npart )
g_adj = adjacency_matrix(g ,  Int64 )


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
        kvois = decomposition.glob_to_loc( sdrespo , kloc  )
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

initial_decomposition = create_partition_subdomain( g , npart )

inflate_subdomain!( g_adj , initial_decomposition[1] , initial_decomposition )

initial_decomposition
