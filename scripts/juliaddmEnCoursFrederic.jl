#################################################
#
# 2021-07-30  Frederic Nataf
#
#################################################
include("decomposition.jl")
using .decomposition

mutable struct Subdomain
    loctoglob::Vector{Int64}
    not_responsible_for::Dict{Subdomain, Vector{Tuple{Int64, Int64}}} # sdvois -> vecteur de pairs (local number , distant number in sdvois)
    responsible_for_others::Dict{Int64, Vector{Tuple{Subdomain, Int64}}}  # k -> vecteur de pairs ( subdomain_vois , k_loc_chezvois ), more or less imposes the way to iterate in function Update.
end


# function subdomains_from_initial_partition( initial_partition , g_adj , overlaps )
#      ressd = sd
#     for (sd , ovrlp) ∈ zip(initial_partition , overlaps)
#         for i ∈ 1:ovrlp
#         inflate_subdomain( g_adj , sd )
#
#     return subdomains
# end



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


function ndof( sbd::Subdomain )
    return length(sbd.loctoglob)
end

function not_responsible_for( sbd::Subdomain )
    return sbd.not_responsible_for
end

function responsible_for_others( sbd::Subdomain )
    return sbd.responsible_for_others
end
######################################################
mutable struct Shared_vector
    values::Dict{Subdomain, Vector{Float64}}
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

# not used yet. To be inserted in MakeCoherent( U::Shared_vector )???
# return what???
function MakeCoherent!( U , sd , sdneigh , numbering )
    for (k,l) ∈ numbering
        values(U,sd)[k] = values(U,sdneigh)[l]
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


# il faut reecrire a la main ce cas test avec les nouvelles structures
# puis tester MakeCoherent
#


# test des fonctions collectiveR_i, D_i etc ...
println("tests de base")

not_responsible_1 = Dict{Subdomain, Vector{Tuple{Int64, Int64}}}( )
responsible_for_others_1 = Dict{Int64, Vector{Tuple{Subdomain, Int64}}}()

not_responsible_2 = Dict{Subdomain, Vector{Tuple{Int64, Int64}}}( )
responsible_for_others_2 = Dict{Int64, Vector{Tuple{Int64, Int64}}}(1 => [] )

subd1 = Subdomain(  [1 ; 2 ; 3 ; 4]  , not_responsible_1 , responsible_for_others_1 )
subd2 = Subdomain( [4 ; 5 ; 6 ; 7]  , not_responsible_2 , responsible_for_others_2 )

subd2.responsible_for_others[1] = [( subd1 , 4 )]
subd1.not_responsible_for[subd2] = [(4 , 1)]


V1 = zeros(ndof(subd1))
V2=ones(ndof(subd2))
U = ones(7)
Vi = Vector{Vector{Float64}}(undef,2)
Vi[1] = V1
Vi[2] = V2

Vshared=   Shared_vector(Dict( subd1 => V1 , subd2 => V2 ))



Update_wo_partition_of_unity!(Vshared)

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
