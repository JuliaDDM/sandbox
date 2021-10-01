#################################################
#
# 2021-07-30  Frederic Nataf
#
#################################################

include("decomposition.jl")

using .decomposition

mutable struct Subdomain
    loctoglob::Vector{Int64}
end

function ndof(sbd::Subdomain)
    return length(sbd.loctoglob)
end


# il faut généraliser le Float64, avoir plus de généralité dans le choix du conteneur de la liste des sous domaines, vecteurs, etc ...
# tout mettre sous forme de fonction d'accès
#faire des versions localisees qui seront appelees par les fonctions globales
#multihreader le tout

"""
    collectiveRi!( Ui , Omegai , U )

Adds the local restriction of the global vector U to the local vector Ui
  Ui += R_i U
 # Arguments
 - 'Ui'::Vector{Vector{Float64}} : vector of local vectors
 - 'Omegai::Vector{Subdomain}'    : vector of subdomains (renumbering information)
 - 'U::Vector{Float64}'           : global vector
"""
function collectiveRi!( Ui::Vector{Vector{Float64}} , Omegai::Vector{Subdomain} ,  U::Vector{Float64})
    for ( subd , Uloc ) in  zip( Omegai , Ui )
        for k in 1:ndof(subd)
            Uloc[k] += U[subd.loctoglob[k]]
        end
    end
end


"""
    collectiveUpdate!( U , Ui , Omegai )

Computes U +=  ∑_i R_i^T Ui
# Arguments
- 'U::Vector{Float64}'           : global vector
- 'Ui'::Vector{Vector{Float64}} : vector of local vectors
- 'Omegai::Vector{Subdomain}'    : vector of subdomains (renumbering information)
"""
function collectiveUpdate!( U::Vector{Float64} , Ui::Vector{Vector{Float64}} , Omegai::Vector{Subdomain} )
    for ( subd , Uloc ) in  zip( Omegai , Ui )
        for k in 1:ndof(subd)
            U[subd.loctoglob[k]] += Uloc[k]
        end
    end
end

"""
    collectiveDi!( Ui , Di )
Computes Uloc[i] .*= D[i]
# Arguments
- 'Ui'::Vector{Vector{Float64}} : vector of local vectors
- 'Di'::Vector{Vector{Float64}} : vector of the partition of unity vectors
"""
function collectiveDi!( Ui::Vector{Vector{Float64}} , Di::Vector{Vector{Float64}} )
    for ( Uloc , Dloc ) in  zip( Ui , Di )
        Uloc .*= Dloc
    end
end

###########################################################################
#
#
#      MAIN
#
#
###########################################################################

####  tests a partir de decoupage METIS

using SparseArrays , LightGraphs , GraphPlot , Metis , LinearAlgebra , .decomposition


m=9
npart = 3
A = spdiagm(-1 => -ones(m-1) , 0 => 2. *ones(m) , 1 => -ones(m-1))

# test des fonctions de decomposition de domaines

g = Graph(A)
initial_partition = create_partition( g , npart )
g_adj = adjacency_matrix(g ,  Int64 )

inflated_subdomains_indices = map(sd->inflate_subdomain( g_adj , sd ) ,  initial_partition)
inflated_subdomains = Subdomain.(inflated_subdomains_indices )

# test de la version plus composable

g2 = Graph(A)
initial_partition2 = create_partition( g2 , npart )
inflate_with_g2 = inflate_subdomain2( g2 )
inflated_subdomains_indices2 = map(sd->inflate_with_g2( sd ) ,  initial_partition)
inflated_subdomains2 = Subdomain.( inflated_subdomains_indices2 )

## comment choisir entre les deux version ci dessus ?? 

# test des fonctions collectiveR_i, D_i etc ...
println("tests de base")
subd1 = Subdomain(  [1 ; 2 ; 3 ; 4])
V1 = zeros(ndof(subd1))
subd2 = Subdomain( [4 ; 5 ; 6 ; 7])
V2=zeros(ndof(subd2))
U = ones(7)
Vi = Vector{Vector{Float64}}(undef,2)
Vi[1] = V1
Vi[2] = V2


collectiveRi!( Vi , [subd1 ; subd2] , U  )
U .= 0.
collectiveUpdate!( U , Vi , [subd1 ; subd2] )
Vi[1] .= 0.
Vi[2] .= 0.
collectiveRi!( Vi , [subd1 ; subd2] , U  )
Di = Vector{Vector{Float64}}(undef,2)
Di[1] = zeros(length(Vi[1]))
Di[1] .=  1. ./ Vi[1]
Di[2] = zeros(length(Vi[2]))
Di[2] .=  1. ./ Vi[2]
Vi[1] .= 1.
Vi[2] .= 1.
collectiveDi!( Vi , Di )

# Contrainte: les dofs locales et de recouvrement doivent être contigues
#  On peut garder le stockage distribué des vecteurs sans passer par un vecteur global
#  Paralléliser le passage au vecteur global pour se faire la main puis
#
#  MPI3 fournit les R_i R_j^T
#
#
#
