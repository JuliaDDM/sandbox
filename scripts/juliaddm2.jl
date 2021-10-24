include("decomposition2.jl")
# faire deux fois include("decompositionEncoursFrederic.jl") semble poser problème
using SparseArrays , LightGraphs , GraphPlot , Metis , LinearAlgebra , .decomposition



######################################################


# Phase 1 de update
# return what???
# function Update_responsible_for_others!( U::Shared_vector )# not correct for triple intersection or more
#     for sd ∈ subdomains( U )
#         for k ∈ keys( responsible_for_others( sd ) )
#             for (sdvois , kvois) ∈ responsible_for_others( sd )[k]
#                 values(U,sd)[k] += values(U,sdvois)[kvois]
#             end
#         end
#     end
# end


function Update_responsible_for_others!( U::Shared_vector )# not correct for triple intersection or more
    for sd ∈ subdomains( U )
        for sdvois ∈ keys( buffer_responsible_for_others( sd ) )
            for ( val , ( kloc , klocchezvois ) ) ∈ zip( buffer_responsible_for_others( sd )[ sdvois ] , decomposition.neighborhood( sd )[ sdvois ] )
                @show decomposition.values(U,sd)[kloc]
                @show val
                decomposition.values(U,sd)[kloc] += val
            end
        end
    end
end


function Fetch_not_responsible_for!( U::Shared_vector )
    # remplissage des buffers
    for sd ∈ subdomains( U )
        for ( sdvois , fecthed_values ) ∈ buffer_responsible_for_others( sd )
            for ( k , ( k_loc , k_loc_chezvois ) ) ∈ enumerate( decomposition.neighborhood( sd )[ sdvois ])
                @show                 fecthed_values[k] = decomposition.values(U,sdvois)[k_loc_chezvois]
            end
        end
    end
end


# return what???
function Update_wo_partition_of_unity!( U::Shared_vector )
    Fetch_not_responsible_for!( U )
    Update_responsible_for_others!( U )
    MakeCoherent!( U )
end


###############@ TESTS BASIQUES ####################################


m=9
npart = 3
A = spdiagm(-1 => -ones(m-1) , 0 => 2. *ones(m) , 1 => -ones(m-1))
g = Graph(A)
initial_decomposition = create_partition_subdomain( g , npart )
g_adj = adjacency_matrix(g ,  Int64 )

inflate_subdomain!( g_adj , initial_decomposition[1] , initial_decomposition )

inflate_subdomain!( g_adj , initial_decomposition[1] , initial_decomposition )



initial_decomposition

Vshareddict = Dict{Subdomain,Vector{Float64}}()
for sd ∈   initial_decomposition
    Vshareddict[sd] = ones(ndof(sd))
end

Vshared = Shared_vector(Vshareddict)


function create_buffers_communication!( sbd::Subdomain )
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



for sd ∈ subdomains( Vshared )
    create_buffers_communication!( sd )
end



Update_wo_partition_of_unity!(Vshared)

vuesur( Vshared )


# ecrire test de validation
# changer la forme du dict
# passer les dict en vecteur??
# faire une structure plus allégée??

uglob = 4. * ones(m)

import_from_global!( Vshared , uglob )

vuesur( Vshared )

Update_wo_partition_of_unity!(Vshared)

vuesur( Vshared )
