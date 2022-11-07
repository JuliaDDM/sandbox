#################################################
#
#       struct DVector
#
#################################################
mutable struct DVector{T,A<:BlockDenseVector{T}} <: AbstractVector{T}
    domain::DDomain
    # pour devenir compatible avec BlockArrays.jl, il faut passer à une paire de vecteurs au lieu d'un vecteur de paires
    data::A
#    data::Vector{Pair{Domain,Vector{T}}}# cf. collect(Dict{Domain,Vector{T}}())
    function DVector(domain,data)
        # equals blocks numbers, ajouter un message plus explicite
        isequal(length(domain) , length(parent(data))) || throw(DomainError())
        new{eltype(data),typeof(data)}( domain , data )
    end
end

function Base.size(v::DVector)
end

function DVector{T}(domain::DDomain) where T
    data = map(domain.subdomains) do sub
        Vector{T}(undef, length(sub))
    end |> blockarray
    DVector(domain, data)
end
