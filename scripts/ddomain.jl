
#################################################
#
#       struct Domain
#
#################################################


#https://docs.julialang.org/en/v1/manual/constructors/
mutable struct Domain# sous domaine aussi
    up::Domain # le (i.e. un seul??) surdomaine éventuellement lui-même
    loctoglob::AbstractVector{Int64} # vecteur d'indices de up qui sont Domain, Int64 pourrait être un paramètre cf indices cartésiens ...
    Domain(loctoglob::AbstractVector{Int64}) = (D = new(); D.loctoglob = copy(loctoglob); D.up = D; return D)
    Domain(up, loctoglob) = issubset(loctoglob, up.loctoglob) ? new(up, loctoglob) : error("indices $loctoglob have to be a subset of the superdomain")
end

# potentiellement dangereux
function global_indices(sd::Domain)
    return sd.loctoglob
end

import Base.length
function length(sd::Domain)
    return length(sd.loctoglob)
end

#################################################
#
#       struct DDomain
#
#################################################


mutable struct DDomain
    up::Domain # le domaine décomposé
    subdomains::Set{Domain} # ensemble des sous domaines
    overlaps::Dict{Domain,Dict{Domain,Tuple{Vector{Int64},Vector{Int64}}}}
    # sd --> (subdomain_vois --> vecteur ( k_loc , k_vois ))
    DDomain(up::Domain, subdomains::Set{Domain}) = (
        res_overlaps = Dict{Domain,Dict{Domain,Tuple{Vector{Int64},Vector{Int64}}}}();
        for sdi ∈ subdomains # algo en N^2 faire mieux avec des boundinng box (mais lesquelles?)
            res_overlaps[sdi] = (Dict{Domain,Tuple{Vector{Int64},Vector{Int64}}})()
            for sdj ∈ subdomains
                if (sdi !== sdj)
                    (sdisdj, kloc, kvois) = intersectalamatlab(global_indices(sdi), global_indices(sdj))
                    if (!isempty(sdisdj))
                        res_overlaps[sdi][sdj] = (kloc, kvois)
                    end
                end
            end
        end;
        res = new(up, subdomains, res_overlaps);
        return res
    )
end

function subdomains(domain::DDomain)
    return domain.subdomains
end
