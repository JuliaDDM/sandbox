# 2022.02.18 Vincent Le Chenadec
import Base: parent, size, getindex

"""
Lazy array for storing finite differencing formulas.

"""
struct FDMArray{F,P,T,N,AA<:AbstractArray{T,N}} <: AbstractArray{T,N}
    op::F
    param::P
    data::AA
end

parent(x::FDMArray) = x.data
size(x::FDMArray) = size(parent(x))

function getindex(x::FDMArray, index::Int...)
    (; op, param, data) = x
    op(param, data, index...)
end

islast(dim, data, index...) =
    index[dim] == last(axes(data, dim))
isfirst(dim, data, index...) =
    index[dim] == first(axes(data, dim))

"""
Differentiation.

"""
δ(y, x) = y - x

"""
Forward differentiation.

"""
function δ⁺(dim, data, index...)
    x = getindex(data, index...)
    if !islast(dim, data, index...)
        shifted = map(enumerate(index)) do (d, i)
            i + (d == dim)
        end
        y = getindex(data, shifted...)
    else
        y = zero(eltype(data))
    end
    δ(y, x)
end

"""
Backward differentiation.

"""
function δ⁻(dim, data, index...)
    if !isfirst(dim, data, index...)
        shifted = map(enumerate(index)) do (d, i)
            i - (d == dim)
        end
        x = getindex(data, shifted...)
    else
        x = zero(eltype(data))
    end
    y = getindex(data, index...)
    δ(y, x)
end

"""
Interpolation.

"""
σ(x, y) = x + y

"""
Forward interpolation.

"""
function σ⁺(dim, data, index...)
    x = getindex(data, index...)
    if !islast(dim, data, index...)
        shifted = map(enumerate(index)) do (d, i)
            i + (d == dim)
        end
        y = getindex(data, shifted...)
    else
        y = zero(eltype(data))
    end
    σ(y, x)
end

"""
Backward interpolation.

"""
function σ⁻(dim, data, index...)
    if !isfirst(dim, data, index...)
        shifted = map(enumerate(index)) do (d, i)
            i - (d == dim)
        end
        x = getindex(data, shifted...)
    else
        x = zero(eltype(data))
    end
    y = getindex(data, index...)
    σ(y, x)
end

"""
Boundary conditions for face capacities.

"""
function μ(dim, data, index...)
    isfirst(dim, data, index...) && return zero(eltype(data))
    any(enumerate(index)) do (d, i)
        i == last(axes(data, d))
    end && return zero(eltype(data))
    getindex(data, index...)
end

###
for op in (:δ⁻, :δ⁺, :σ⁻, :σ⁺, :μ)
    @eval $op(dim, data) = FDMArray($op, dim, data)
end

###
using Symbolics
import Symbolics: Arr, scalarize

A = @variables A₁[1:6, 1:6] A₂[1:6, 1:6]
@variables T[1:6, 1:6]
@variables D[1:6, 1:6]
@variables V[1:6, 1:6]

@register_symbolic δ(x::Real, y::Real)::Real
@register_symbolic σ(x::Real, y::Real)::Real

@register_symbolic δ⁻(i, X::Arr{Num, 2})::Arr{Num, 2}
@register_symbolic δ⁺(i, X::Arr{Num, 2})::Arr{Num, 2}
@register_symbolic σ⁻(i, X::Arr{Num, 2})::Arr{Num, 2}
@register_symbolic σ⁺(i, X::Arr{Num, 2})::Arr{Num, 2}
@register_symbolic  μ(i, X::Arr{Num, 2})::Arr{Num, 2}

# set capacities to 0 on boundaries
A̅ = map(eachindex(A)) do i
    μ(i, A[i])
end

# 2 x volume-weighted gradient (Dirichlet)
G = map(eachindex(A̅)) do i
    δ⁻(i, σ⁺(i, A̅[i]) .* T) .- σ⁻(i, δ⁺(i, A̅[i]) .* D)
end

# 2 x volume-weighted laplacian with Dirichlet B.C.
Δ = mapreduce(+, eachindex(G)) do i
    σ⁺(i, A̅[i]) .* δ⁺(i, G[i] ./ σ⁻(i, V))
end

# generate all possible formulas
formula = scalarize.(Δ)

using Symbolics.Rewriters

@variables i::Int j::Int

fdm = map(CartesianIndices(formula)) do outer
    u, v = Tuple(outer)
    rules = map(reshape(CartesianIndices((u-1:u+2, v-1:v+2)), :)) do inner
        i, j = Tuple(inner)
        @rule getindex(~x, $i, $j) => :($x[i+$(i-u), j+$(j-v)])
    end
    push!(rules, @rule(δ(~x...) => :((-)($(x...)))))
    push!(rules, @rule(σ(~x...) => :((+)($(x...)))))
    Meta.parse(string(formula[outer])) |>
    Postwalk(Chain(rules)) |>
    eval
end

#=
fdm = substitute.(fdm, Ref(Dict(δ => -, σ => +)))

fdm = map(fdm) do el
    eval(rewrite(Meta.parse(string(el)), []))
end
=#

