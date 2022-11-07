#################################################
#
#       struct DVector
#
#################################################
mutable struct DVector{T}
    domain::DDomain
    # pour devenir compatible avec BlockArrays.jl, il faut passer à une paire de vecteurs au lieu d'un vecteur de paires
    data::Vector{Pair{Domain,Vector{T}}}# cf. collect(Dict{Domain,Vector{T}}())
    # + , - , a* , .* , similar etc ... si on peut automatiquement hériter de ce qui vient de vecteur, on a gagné voir comment faire en Julia
    # boucles sur eval ??
    # ce qui est lié à l'aspect cohérent : prodscal et donc demande une partition de l'unite qulle quelle soit
    # relation avec les vecteurs "habituels" : DVector2Vector
end

function subdomains(DVect::DVector)
    return subdomains(DVect.domain)
end

import Base.values
function values(DVect::DVector, sd::Domain)
    pos =  findfirst( x -> isequal(x.first,sd) , DVect.data)
    return DVect.data[pos].second
end


function DVector(ddomain::DDomain, initial_value::T) where {T<:Number}
    datatype = typeof(initial_value)
    data_res = Vector{Pair{Domain,Vector{datatype}}}(undef,length(ddomain))
    ThreadsX.foreach( enumerate(ddomain.subdomains) )  do ( i , sdi )
        data_res[i] = ( sdi => fill( initial_value , length(sdi) ) )
    end
    return DVector(ddomain, data_res)
end


"""
basic printing of a decomposed vector
"""
function vuesur(U::DVector)
    for ( i , sdvec) ∈ U.data
        println(i,"   ",sdvec)
    end
end


"""
Update(U)

returns the decomposed vector R_i ∑_j R_j^T U_j
# Arguments
- 'U' : a decomposed vector
"""
function Update(DVec::DVector)
    res = DVector(DVec.domain, 0.0)
    ThreadsX.foreach( zip(res.data,DVec.data) )  do ( (sdres , vecres) , (sdsrc , vecsrc) )
        vecres .= vecsrc
    end
    ThreadsX.foreach( enumerate(DVec.domain.subdomains) )  do ( i , sdi )
        for sdvois ∈ DVec.domain.overlaps[i]
            res.data[i].second[sdvois.second[1]] .+= DVec.data[ sdvois.first ].second[sdvois.second[2]]
        end
    end
    return res
end



import Base.* , Base.fill!
function *( a::T , x::DVector{T} )  where {T<:Number}
    res = DVector(x.domain, zero(T))
    ThreadsX.foreach( zip( res.data , x.data ) )  do ( (sdres , vecres) , (sdsrc , vecsrc) )
        vecres .= a * vecsrc
    end
    return res
end

function dot_op(x::DVector{T}, y::DVector{T}, dot_op) where {T<:Number}
    if !(x.domain == y.domain)
        error("Domains of both decomposed vectors must be the same")
    end
    res = DVector( x.domain , zero(T) )
    ThreadsX.foreach( zip( res.data , x.data , y.data ) )  do ( (sdres , vecres) , (sdx ,vecx) , (sdy , vecy) )
        vecres .= dot_op( vecx , vecy )
    end
    return res
end


"""
Dibooleannaive( domain )

returns a decomposed vector that corresponds to a Boolean partition of unity function
# Arguments
- 'domain' : the support domain
"""
function Dibooleannaive(ddomain::DDomain)
    res = DVector(ddomain, 1.0)
    ThreadsX.foreach( enumerate(subdomains(res))  ) do ( i , sdi )
         for jvois ∈ intersect( collect(i+1:length(ddomain)) , keys(ddomain.overlaps[i])  )
           res.data[jvois].second[ddomain.overlaps[i][jvois][2]] .= 0.0
         end
    end
    return res
end

# premiere version
function my_memoize(f)
    mem = Dict()
    function memf(x)
        if x ∈ keys(mem)
#            println("I know you")
            return mem[x]
        end
        mem[x] = f(x)
        return mem[x]
    end
    return memf
end

Diboolean = my_memoize(Dibooleannaive)


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
#    return Update(dot_op(Diboolean(DVect.domain), DVect, (.*)))
    Di = Diboolean(DVect.domain)
    return MakeCoherent(DVect,Di)
end


"""
MakeCoherent( dvector , Di )

Returns a coherent decomposed vector
R_i ∑_j R_j^T D_j U_j
# Argument
- dvector : a decomposed vector
- Di      : a partition of unity vector
"""
function MakeCoherent(DVect::DVector , Di::DVector)
    if !(DVect.domain == Di.domain)
        error("Domains of both decomposed vectors must be the same")
    end
    return Update(dot_op(Di, DVect, (.*)))
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


import Base.ones, Base.zeros , Base.rand
for sym in [ :ones , :zeros , :rand ]
    @eval function $(Symbol(string(sym)))(ddomain::DDomain)
        # Float64 should be inferred automatically
        data_res = Vector{Pair{Domain,Vector{Float64}}}(undef,length(ddomain));
        ThreadsX.foreach( enumerate(ddomain.subdomains) )  do ( i , sdi )
            data_res[i] = ( sdi => $sym( length(sdi) ) );
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
    res = DVector( ddomain, zero(T) )
    #what if Usrc is already a decomposed vector??
    ThreadsX.foreach( enumerate(ddomain.subdomains) )  do  ( i , sdi )
    #    values(res, sd) .= Usrc[global_indices(sd)]
        res.data[i].second .= Usrc[global_indices(sdi)]
    end
    return res
end


"""
Dimultiplicity( domain )

returns a decomposed vector that corresponds to a multiplicity based partition of unity function
# Arguments
- 'domain' : the support domain
"""
function Dimultiplicity(domain::DDomain)
    tmp = DVector(domain, 1.0)
    multiplicity =

    (tmp)
    res = dot_op(tmp, multiplicity, (./))
    return res
end


function noncoherentrandDVector(ddomain::DDomain)
    data_res = Vector{Pair{Domain,Vector{Float64}}}(undef,length(ddomain))
    ThreadsX.foreach( enumerate(ddomain.subdomains) )  do ( i , sdi )
        data_res[i] = ( sdi => rand(length(sdi)))
    end
    return DVector(ddomain, data_res)
end



import  Base.similar
# first try of Metaprogramming
for sym in [ :similar   ]
    @eval function $(Symbol(string(sym)))(a::DVector)
        res = DVector(a.domain, 0.0)
        ThreadsX.foreach( enumerate(subdomains(res)) )  do ( i , sdi )
            res.data[i].second .=  $sym( a.data[i].second )
        end
        return res
    end
end

import Base.copy
function copy(DVec::DVector)
    res = DVector(DVec.domain, 0.0)
    ThreadsX.foreach( enumerate(subdomains( DVec )) )  do ( i , sdi )
    res.data[i].second .=  DVec.data[i].second
    end
    return res
end

import LinearAlgebra.dot
"""
dot( U , V )

returns the scalar product of two coherent decomposed vectors
 ∑_i ( U , Diboolean(V) )
# Arguments
- 'U and V' : two decomposed vectors
"""
function dot(Da::DVector,Db::DVector)
    tmp = dot_op( Diboolean(Db.domain)  , Db , (.*) )
# time and allocation could be saved by not creating tmp but making a .* method with three vectors:
    res = ThreadsX.sum(   dot( sdveca.second ,  sdvectmp.second ) for ( sdveca , sdvectmp ) ∈ zip(Da.data,tmp.data) )
    return res
end

import LinearAlgebra.norm
"""
norm(v)

compute the 2-norm of a vector
"""
function norm( Da::DVector)
    return sqrt( dot( Da , Da ) )
end


function fill!(x::DVector{T} , a::T )  where {T<:Number}
    ThreadsX.foreach( x.data )  do sdvec
#    for sd ∈ subdomains(res)
        fill!( sdvec.second , a)
    end
end


import Base.+
function +( x::DVector , y::DVector )
    return dot_op( x , y , +)
end


import LinearAlgebra.mul!
"""
 Out of place scalar multiplication; multiply vector v with scalar α and store the result in w

"""
function mul!(w::DVector , v::DVector , α )
    ThreadsX.foreach( zip( w.data , v.data ) ) do ( sdvec_w , sdvec_v )
        LinearAlgebra.mul!( sdvec_w.second , sdvec_v.second , α )
    end
end

import LinearAlgebra.rmul!
"""
    rmul!(v, α): in-place scalar multiplication of v with α;
"""
function rmul!(v::DVector , α )
    ThreadsX.foreach( v.data ) do sdvec_v
        LinearAlgebra.rmul!( sdvec_v.second , α )
    end
end

import LinearAlgebra.axpy!
"""
    axpy!(α, v, w): store in w the result of α*v + w
"""
function axpy!(α, v::DVector , w::DVector )
    ThreadsX.foreach( zip( v.data , w.data ) ) do ( sdvec_v , sdvec_w )
        axpy!( α , sdvec_v.second , sdvec_w.second )
    end
end

import LinearAlgebra.axpby!
"""
    axpby!(α, v, β, w): store in w the result of α*v + β*w
"""
function axpby!( α , v::DVector , β , w::DVector )
    ThreadsX.foreach( zip( v.data , w.data ) ) do ( sdvec_v , sdvec_w )
        axpby!( α , sdvec_v.second , β , sdvec_w.second )
    end
end



# DV1 .* DV2 , iterable venant d'un abstractvector , risque de perdre le //
# Vincent --> ou surcharger broadcast



function DVector2Vector(DVect::DVector{T}) where {T<:Number}
    Dres = MakeCoherent(DVect)
    ddomain = DVect.domain
    res = zeros( T , length(ddomain.up) )
    #peu compatible avec une parallelisation
    #il faudrait differencier selon que Diboolean est zero ou non
    # si je mets ThreadSafeDict ici (c.a.d.????), j'ai une erreur à l'execution. => strange ??
    for (sd, val) ∈ Dres.data
        res[global_indices(sd)] .= val
    end
    return res
end
