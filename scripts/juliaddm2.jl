include("decomposition2.jl")
# faire deux fois include("decompositionEncoursFrederic.jl") semble poser problème
using SparseArrays , LightGraphs , GraphPlot , Metis , LinearAlgebra , .decomposition



######################################################


# Phase 1 de update
# return what???
function Update_responsible_for_others!( U::Shared_vector )# not correct for triple intersection or more
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
    Fethch_not_responsible_for( U )
    Update_responsible_for_others!( U )# not correct for triple intersection or more
    MakeCoherent!( U::Shared_vector )
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

vuesur( Vshared )


#
# function Update_wo_partition_of_unity!( U::Shared_vector )
#     Fetch_remote_values!( U )
#     Update_responsible_for_others!( U )
#     MakeCoherent!( U::Shared_vector )
# end
