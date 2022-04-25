

#################################################
#
#       struct DVector
#
#################################################
mutable struct DVector{T}
    domain::DDomain
    data::MyDict{Domain,Vector{T}}
    # ATTENTION : le choix ThreadSafeDict vs. Dict influe les autres fonctions
    # + , - , a* , .* , similar etc ... si on peut automatiquement hériter de ce qui vient de vecteur, on a gagné voir comment faire en Julia
    # boucles sur eval ??
    # ce qui est lié à l'aspect cohérent : prodscal et donc demande une partition de l'unite qulle quelle soit
    # relation avec les vecteurs "habituels" : import_from_global(!) , export_to_global(!)
end

function subdomains(DVect::DVector)
    return subdomains(DVect.domain)
end

import Base.values
function values(DVect::DVector, sd::Domain)
    return DVect.data[sd]
end


function DVector(ddomain::DDomain, initial_value::T) where {T<:Number}
    datatype = typeof(initial_value)
    data_res = MyDict{Domain,Vector{datatype}}()
    #data_res = Dict{Domain,Vector{datatype}}()
    ThreadsX.foreach( ddomain.subdomains )  do sd
    #for sd ∈ ddomain.subdomains
        data_res[sd] = zeros(datatype, length(global_indices(sd)))
    end
    #is actually lockfree since the value of the dictionnary have been allocated just above
    ThreadsX.foreach( ddomain.subdomains )  do sd
        data_res[sd] .= initial_value
    end
    return DVector(ddomain, data_res)
end

import Base.ones, Base.zeros , Base.rand
for sym in [ :ones , :zeros , :rand ]
    @eval function $(Symbol(string(sym)))(ddomain::DDomain)
        # Float64 should be inferred automatically
        data_res = MyDict{Domain,Vector{Float64}}()
        ThreadsX.foreach( ddomain.subdomains )  do sd
#        for sd ∈ ddomain.subdomains
            #here as well
            data_res[sd] = $sym( length(global_indices(sd)) )
        end
        return MakeCoherent(DVector(ddomain, data_res))
    end
end

"""
DVector( domain , vecsrc )

Returns a decomposed vector built from a vector
# Arguments
- 'domain' : the decomposed domain
- 'vecsrc' : the classical vector
"""
function DVector(ddomain::DDomain, Usrc::Vector{T}) where {T<:Number}
    if !(length(ddomain.up) == length(Usrc))
        error("Lengthes of decomposed domain and vector must match: $(length(ddomain.up)) is not $(length(Usrc)) ")
    end
    res = DVector(ddomain, 0.0)
    #what if Usrc is already a decomposed vector??
    ThreadsX.foreach( ddomain.subdomains )  do sd
#    for sd ∈ ddomain.subdomains
        values(res, sd) .= Usrc[global_indices(sd)]
    end
    return res
end

import  Base.similar
# first try of Metaprogramming
for sym in [ :similar   ]
    @eval function $(Symbol(string(sym)))(a::DVector)
        res = DVector(a.domain, 0.0)
        ThreadsX.foreach( subdomains(res) )  do sd
#        for sd ∈ subdomains(res)
            values(res, sd) .=  $sym( values(a,sd) )
        end
        return res
    end
end

import Base.copy
function copy(DVec::DVector)
    res = DVector(DVec.domain, 0.0)
    ThreadsX.foreach(subdomains( Dvec ))  do sdi
    # for sdi ∈ subdomains(DVec)
        res.data[sdi] .= values(DVec,sdi)
    end
    return res
end


function dot_op(x::DVector, y::DVector, dot_op)
    if !(x.domain == y.domain)
        error("Domains of both decomposed vectors must be the same")
    end
    res = DVector(x.domain, 0.0)
    ThreadsX.foreach(subdomains( res ))  do sd
#    for sd ∈ subdomains(res)
        res.data[sd] .= dot_op(x.data[sd], y.data[sd])
    end
    return res
end

# DV1 .* DV2 , iterable venant d'un abstractvector , risque de perdre le //
# Vincent --> ou surcharger broadcast


"""
MakeCoherent( dvector )

Returns a coherent decomposed vector
Using a Boolean partition of unity ensures that the result is roundoff error free and execution order independent
R_i ∑_j R_j^T D_j U_j
# Argument
- dvector : a decomposed vector
"""
function MakeCoherent(DVect::DVector)
    #Diboolean ensures that the result is roundoff error free
    return Update(dot_op(Diboolean(DVect.domain), DVect, (.*)))
end


# """
# MakeCoherent!( dvector )
#
# Makes a decomposed vector coherent BOGUE
# Using a Boolean partition of unity ensures that the result is roundoff error free and execution order independent
# # Argument
# - dvector : a decomposed vector
# """
# function MakeCoherent!(DVect::DVector)
#     #Diboolean ensures that the result is roundoff error free
#     tmp=copy(DVect)
#     Dvect = copy(MakeCoherent(tmp))
# end



function DVector2Vector(DVect::DVector)
    Dres = MakeCoherent(DVect)
    ddomain = DVect.domain
    res = zeros(Float64, length(ddomain.up))
    #peu compatible avec une parallelisation
    #il faudrait differencier selon que Diboolean est zero ou non
    # si je mets ThreadSafeDict ici (c.a.d.????), j'ai une erreur à l'execution. => strange ??
    for (sd, val) ∈ Dres.data
        res[global_indices(sd)] .= val
    end
    return res
end

"""
returns a decomposed vector R_i ∑_j R_j^T U_j
"""
function Update(DVec::DVector)
    res = DVector(DVec.domain, 0.0)
    ThreadsX.foreach( DVec.domain.subdomains )  do sd
#    for sd ∈ DVec.domain.subdomains
        res.data[sd] .= DVec.data[sd]
    end
    ThreadsX.foreach( DVec.domain.subdomains )  do sd
#    for sd ∈ DVec.domain.subdomains
        for sdvois ∈ DVec.domain.overlaps[sd]
            res.data[sd][sdvois.second[1]] .+= DVec.data[sdvois.first][sdvois.second[2]]
        end
    end
    return res
end


"""
Di( domain )

returns a decomposed vector that corresponds to a multiplicity based partition of unity function
# Arguments
- 'domain' : the support domain
"""
function Di(domain::DDomain)
    tmp = DVector(domain, 1.0)
    multiplicity = Update(tmp)
    res = dot_op(tmp, multiplicity, (./))
    return res
end


"""
Diboolean( domain )

returns a decomposed vector that corresponds to a Boolean partition of unity function
# Arguments
- 'domain' : the support domain
"""
function Diboolean(domain::DDomain)
    res = DVector(domain, 1.0)
    vector_of_subdomains = collect(subdomains(domain))
    ThreadsX.foreach( enumerate(vector_of_subdomains) )  do (i, sd)
#    for (i, sd) ∈ enumerate(vector_of_subdomains)
        for sdvois ∈ intersect(vector_of_subdomains[i+1:end], collect(keys(domain.overlaps[sd])))
            res.data[sdvois][domain.overlaps[sd][sdvois][2]] .= 0.0
        end
    end
    return res
end


function vuesur(U::DVector)
    for sd ∈ subdomains(U)
        println(values(U, sd))
    end
end


function noncoherentrandDVector(ddomain::DDomain)
    data_res = MyDict{Domain,Vector{Float64}}()
    ThreadsX.foreach( ddomain.subdomains )  do sd
#    for sd ∈ ddomain.subdomains
        data_res[sd] = floor.(10. *rand( length(global_indices(sd)) ))
    end
    return DVector(ddomain, data_res)
end
