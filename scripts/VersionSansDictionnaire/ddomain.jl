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
    subdomains::Vector{Domain} # ensemble des sous domaines
    overlaps::Dict{Int64,Dict{Int64,Tuple{Vector{Int64},Vector{Int64}}}}
#    overlaps::SparseMatrixCSC{Tuple{Vector{Int64},Vector{Int64}},Int64}
# Essayer Union type(nothing,mon truc )
    # sd --> (subdomain_vois --> vecteur ( k_loc , k_vois ))
    DDomain(up::Domain, subdomains::Vector{Domain}) = (
        res_overlaps = Dict{Int64,Dict{Int64,Tuple{Vector{Int64},Vector{Int64}}}}();
        for ( i , sdi) ∈ enumerate(subdomains) # algo en N^2 faire mieux avec des bounding box (exemple: index_min,index_max )
            res_overlaps[i] = (Dict{Int64,Tuple{Vector{Int64},Vector{Int64}}})()
            for ( j , sdj) ∈ enumerate(subdomains)
                if (sdi !== sdj)
                    (sdisdj, kloc, kvois) = intersectalamatlab(global_indices(sdi), global_indices(sdj))
                    if (!isempty(sdisdj))
                        res_overlaps[i][j] = (kloc, kvois)
                    end
                end
            end
        end;
        res = new(up, subdomains, res_overlaps );
        return res
    )
end

# pour eviter le N^2,
    # l'intersection est symétrique !!!--> /2
    # accélérer le ci dessus avec une boudning box basée sur le min et max des indices
    # plus imbriquer les choses via par exemple l'algorithme d'inflation (cf. matrice adjacence)
    # ne parcourir que les points ajoutés à un sous domaine (N^2 reste mais moins de taches par couple de sous domaines)

function subdomains(domain::DDomain)
    return domain.subdomains
end


function length(ddomain::DDomain)
    return length(ddomain.subdomains)
end
