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
M = @variables M₁[1:6, 1:6] M₂[1:6, 1:6]
@variables T[1:6, 1:6]
@variables D[1:6, 1:6]
@variables G[1:6, 1:6]
@variables H[1:6, 1:6]
@variables V[1:6, 1:6]

@register_symbolic δ(x::Real, y::Real)::Real
@register_symbolic σ(x::Real, y::Real)::Real
@register_symbolic not(x::Real)::Real

@register_symbolic δ⁻(i, X::Arr{Num, 2})::Arr{Num, 2}
@register_symbolic δ⁺(i, X::Arr{Num, 2})::Arr{Num, 2}
@register_symbolic σ⁻(i, X::Arr{Num, 2})::Arr{Num, 2}
@register_symbolic σ⁺(i, X::Arr{Num, 2})::Arr{Num, 2}
@register_symbolic  μ(i, X::Arr{Num, 2})::Arr{Num, 2}

# set capacities to 0 on boundaries
A̅ = map(eachindex(A)) do i
    μ(i, A[i])
end

# 2 x volume-weighted gradient (Dirichlet B. C.)
Gᴰ = map(eachindex(A̅)) do i
    (δ⁻(i, σ⁺(i, A̅[i]) .* T) .- σ⁻(i, δ⁺(i, A̅[i]) .* D)) ./ σ⁻(i, V)
end .|> scalarize

# 2 x volume-weighted gradient (Neumann B. C.)
Gᴺ = map(eachindex(A̅)) do i
    (A̅[i] .* δ⁻(i, T) - σ⁻(i, δ⁺(i, A̅[i]) .* G .* H)) ./ σ⁻(i, V)
end .|> scalarize

# 2 x volume-weighted laplacian (Neumann B. C.)
Δᴺ = mapreduce(+, eachindex(A̅)) do i
    δ⁺(i, A̅[i] .* Gᴺ[i]) .- δ⁺(i, A̅[i]) .* G
end |> scalarize

# 2 x volume-weighted laplacian (Dirichlet B. C.)
Δᴰ = mapreduce(+, eachindex(A̅)) do i
    σ⁺(i, A̅[i]) .* δ⁺(i, Gᴰ[i])
end |> scalarize

# Mixed B. C.
Δ = mapreduce(+, eachindex(A̅)) do i
    δ⁺(i, A̅[i] .* (M[i] .* Gᴺ[i] .+ not.(M[i]) .* Gᴰ[i])) .-
    δ⁺(i, A̅[i]) .* (M[i] .* G .+ not.(M[i]) .* σ⁺(i, Gᴰ[i]))
end |> scalarize

# generate all possible formulas
#formula = scalarize.(Δᴰ)
#formula = scalarize(Δᴰ)
formula = Δᴰ

using Symbolics.Rewriters

@variables i::Int j::Int

fdm0 = map(CartesianIndices(formula)) do outer
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
fdm0 = substitute.(fdm0, Ref(Dict(δ => -, σ => +)))

fdm0 = map(fdm0) do el
    eval(rewrite(Meta.parse(string(el)), []))
end
=#

function removezeros(x::AbstractMatrix)
    low, up = Vector{Int}(undef, 2), Vector{Int}(undef, 2)
    #
    low[1] = all(iszero, x[begin, :])
    up[1] = all(iszero, x[end, :])
    #
    low[2] = all(iszero, x[:, begin])
    up[2] = all(iszero, x[:, end])
    #
    inds = map(axes(x), low, up) do axis, l, u
        axis[begin + l:end - u]
    end
    #
    x[inds...], low, up
end

function foo(x, low, up)
    all(isone, size(x)) && return x, low, up

    inner = map(axes(x)) do axis
        axis[begin + 1:end - 1]
    end
    low .+= 1
    up .+= 1
    foo(x[inner...], low, up)
end

fdm, low, up = foo(removezeros(fdm0)...)

function bar(x::AbstractMatrix)
    block = Expr(:block)

    y, low, up = removezeros(fdm0)

    block
end

function baz(low::Int, up::Int, expr)
    lower = Expr(:call, :+, :begin, low)
    upper = Expr(:call, :-, :end, up)
    rng = Expr(:call, :(:), lower, upper)
    Expr(:ref, expr, rng)
end

baz(1, 1, :(Base.OneTo(32)))

baz(::typeof(+)) = :begin
baz(::typeof(-)) = :end

function baz(shift::Int, dir::Function, expr)
    (shift == 0 && dir == +) && return Expr(:call, :first, expr)
    (shift == 0 && dir == -) && return Expr(:call, :last, expr)
    index = Expr(:call, Symbol(dir), baz(dir), shift)
    Expr(:ref, expr, index)
end

function toexpr(n, low, up, axis)
    n == 1 && return [baz(low, up, axis)]

    vcat(baz(low, +, axis),
         toexpr(n-2, low+1, up+1, axis),
         baz(up, -, axis))
end

toexpr(5, 0, 1, :(axes(a, 1)))

using TiledIteration

outerrange = CartesianIndices((1:6, 1:6))
innerrange = CartesianIndices((3:3, 3:3))

for I in EdgeIterator(outerrange, innerrange)
    @show I
end

left(expr, iter) = map(iter) do i
    i == 0 && return :(first($expr))
    :($expr[begin + $i])
end

right(expr, iter) = map(reverse(iter)) do i
    i == 0 && return :(last($expr))
    :($expr[end - $i])
end
#
#function middle(expr, i, j)
#
#end

vcat(left(:x, 0:1), right(:x, 0:2))

