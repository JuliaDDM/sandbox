####################################################@
#
#
#       R_i V = ∑_j R_i A R_j^T D_j U_j
#
#
####################################################@
include("decompositionparallel.jl")
using SparseArrays , LightGraphs , GraphPlot , Metis , LinearAlgebra , BenchmarkTools , .decomposition, ThreadsX , ThreadSafeDicts


struct Shared_matrix
    domain::Domain
    original_matrix
    sub_matrices::Dict{Tuple{Subdomain,Subdomain},Any}
end

function create_sub_matrix
    sub_matrices = Dict{Tuple{Subdomain,Subdomain},Any}()
    for sdi ∈ subdomains(domain)
        for sdj ∈ subdomains(domain)
            sub_matrices[(sdi,sdj)] = A[ global_indices( sdi ) , global_indices( sdj ) ] * POU( domain , sdj )
        end
    end
    return sub_matrices
end

Shared_matrix(domain::Domain,A) = Shared_matrix( domain , A ,  Dict{Tuple{Subdomain,Subdomain},Any}() )

function matvec( A , U )
    AU = similar( U )
    matvec!( AU , A , U )
    return AU
end


function matvec!( V , A , U )
    V =


end
