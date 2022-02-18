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

###
islast(dim, data, index...) =
    index[dim] == last(axes(data, dim))
isfirst(dim, data, index...) =
    index[dim] == first(axes(data, dim))

"""
Forward differencing.

"""
function δ⁺(dim, data, index...)
    res = -getindex(data, index...)
    if !islast(dim, data, index...)
        shifted = map(enumerate(index)) do (d, i)
            i + (d == dim)
        end
        res += getindex(data, shifted...)
    end
    res
end

"""
Backward differencing.

"""
function δ⁻(dim, data, index...)
    res = getindex(data, index...)
    if !isfirst(dim, data, index...)
        shifted = map(enumerate(index)) do (d, i)
            i - (d == dim)
        end
        res -= getindex(data, shifted...)
    end
    res
end

###

"""
Forward interpolation.

!!! note

    Missing a factor ``1/2``.

"""
function σ⁺(dim, data, index...)
    res = getindex(data, index...)
    if !islast(dim, data, index...)
        shifted = map(enumerate(index)) do (d, i)
            i + (d == dim)
        end
        res += getindex(data, shifted...)
    end
    res
end

"""
Backward interpolation.

!!! note

    Missing a factor ``1/2``.

"""
function σ⁻(dim, data, index...)
    res = getindex(data, index...)
    if !isfirst(dim, data, index...)
        shifted = map(enumerate(index)) do (d, i)
            i - (d == dim)
        end
        res += getindex(data, shifted...)
    end
    res
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

@register_symbolic δ⁻(i, X::Arr{Num, 2})::Arr{Num, 2}
@register_symbolic δ⁺(i, X::Arr{Num, 2})::Arr{Num, 2}
@register_symbolic σ⁻(i, X::Arr{Num, 2})::Arr{Num, 2}
@register_symbolic σ⁺(i, X::Arr{Num, 2})::Arr{Num, 2}
@register_symbolic  μ(i, X::Arr{Num, 2})::Arr{Num, 2}

# set capacities to 0 on boundaries
A̅ = map(enumerate(A)) do (i, Aᵢ)
    μ(i, Aᵢ)
end

# volume-weighted gradient with Dirichlet B.C.
G = map(enumerate(A̅)) do (i, C)
    δ⁻(i, σ⁺(i, C) .* T) .- σ⁻(i, δ⁺(i, C) .* D)
end

# volume-weighted laplacian with Dirichlet B.C.
Δ = mapreduce(+, enumerate(G)) do (i, C)
    σ⁺(i, A̅[i]) .* δ⁺(i, C ./ σ⁻(i, V))
end

# generate all possible formulas
formula = scalarize(Δ)

