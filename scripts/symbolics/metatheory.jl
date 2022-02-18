# 1D
using Metatheory
using Metatheory.Rewriters

const op = (:δ⁻, :δ⁺, :σ⁻, :σ⁺)
const sub = ("\u2081", "\u2082", "\u2083")
#const sym = (:i,)# :j, :k)

n = (8, 8, 8)

# width of boundary conditions on faces
const width = (# set 1:n to zero
               left = (2, 1), # 1:n
               # set n:4
               right = (4, 3, 2)) # n:4

### expr[1]
expr = [:(sum(σ⁺(A[i], i) * δ⁺((δ⁻(σ⁺(A[i], i) * T, i) - σ⁻(δ⁺(A[i], i) * Tᵧ, i)) / σ⁻(V, i), i), i))]

### expr[2]
# inner or outer
# j ≠ i
function einstein(sym, num)
    [@rule(getindex(~x, $sym) => :($(Symbol(x, sub[num])))),
     @rule((~f::in(op))(~x, $sym) => :($f($x, $num)))]
end

einstein(expr) = rewrite(expr, [@rule sum(~x, ~i) =>
    :((+)($(map(j -> rewrite(x, einstein(i, j)), first(axes(n)))...)))])

# :(sum(δ⁺(A[i], j), j ≠ i)
push!(expr, einstein(last(expr)))

### expr[3]
function morinishi(::Type{Val{2}})
    [@rule(δ⁻(~x, ~i) => :($x - θ($x, $i, -1))),
     @rule(δ⁺(~x, ~i) => :(θ($x, $i, +1) - $x)),
     @rule(σ⁻(~x, ~i) => :($x + θ($x, $i, -1))),
     @rule(σ⁺(~x, ~i) => :(θ($x, $i, +1) + $x))]
end

morinishi(expr, order = Val{2}) =
    rewrite(expr, morinishi(order))

push!(expr, morinishi(last(expr)))

### expr[4]
function stencil(expr)
    rules = [@rule(θ((~f::!=(:θ))(~x...), ~y...) =>
                   :($f($(map(z -> :(θ($z, $(y...))), x)...)))),
             @rule(θ(θ(~x, ~d, ~i), ~d, ~j) =>
                   :(θ($x, $d, $(i + j))))]
    rewrite(expr, rules)
end

push!(expr, stencil(last(expr)))

### expr[5]
center(x...) = 2one(Int)
isasym = Base.Fix2(isa, Symbol)
index = [@rule(~x::isasym => :($x[$(ntuple(center, length(n))...)]))]

push!(expr, Postwalk(Chain(index))(last(expr)))

### expr[6]
rule = @rule θ(getindex(~x, ~i...), ~d, ~s) =>
    :($x[$(map(((j, k),) -> k + (j == d ? s : 0), enumerate(i))...)])

push!(expr, rewrite(last(expr), [rule]))

#    :($x[$(map(((j, k),) -> k + sym[j], enumerate(i))...)])

#=
### expr[7]
bc(sym) = @rule getindex($sym, ~i, -1, ~k...) --> 0
push!(expr, rewrite(last(expr), bc.(Symbol.(:A, sub))))
=#

using Base.Iterators

#=
function tiles(n)
    iters = map(n) do _
        left, right = width
        UnitRange.(1, left)..., UnitRange(4, 1), UnitRange.(right, 4)...
    end
    product(iters...)
end
=#

_boundary(::Type{Val{1}}, sym::Symbol, i) =
    @rule getindex($sym, $i, ~j...) --> 0
_boundary(::Type{Val{2}}, sym::Symbol, j) =
    @rule getindex($sym, ~i, $j, ~k...) --> 0
_boundary(::Type{Val{3}}, sym::Symbol, k) =
    @rule getindex($sym, ~i, ~j, $k, ~l...) --> 0

function _boundary(::Type{Val{N}}, ::Type{Val{D}}, sym::Symbol) where {N,D}
    left, right = width
    iter = UnitRange.(1, left)...,
           UnitRange(4, 1),
           UnitRange.(right, 4)...
    orth = map(iter) do el
        map(el) do i
            _boundary(Val{D}, Symbol(sym, sub[D]), i)
        end
    end
    para = map(d for d in 1:N if d ≠ D) do d
        _boundary(Val{D}, Symbol(sym, sub[d]), 2)
    end
    for el in para
        push!(last(orth), el)
    end
    orth
end

function boundary(n::NTuple{N}, sym) where {N}
    iters = ntuple(length(n)) do D
        _boundary(Val{N}, Val{D}, sym)
    end
    map(product(iters...)) do el
        vcat(el...)
    end
end

using Symbolics

if n isa NTuple{1}
    @variables V[1:4]
    @variables A₁[1:4]
    @variables T[1:4]
    @variables Tᵧ[1:4]
elseif n isa NTuple{2}
    @variables V[1:4, 1:4]
    @variables A₁[1:4, 1:4]
    @variables A₂[1:4, 1:4]
    @variables T[1:4, 1:4]
    @variables Tᵧ[1:4, 1:4]
elseif n isa NTuple{3}
    @variables V[1:4, 1:4, 1:4]
    @variables A₁[1:4, 1:4, 1:4]
    @variables A₂[1:4, 1:4, 1:4]
    @variables A₃[1:4, 1:4, 1:4]
    @variables T[1:4, 1:4, 1:4]
    @variables Tᵧ[1:4, 1:4, 1:4]
end

tiles = map(boundary(n, :A)) do rules
    Meta.parse(string(eval(rewrite(last(expr), rules))))
end

#face(sym, i) = map(i) do el
#    @rule getindex($sym, $el) --> 0
#end
#@variables n::Int

#=
exprbc = rewrite(last(expr),
                 face(:A₁, (1, 2))) |>
         eval |>
         string |>
         Meta.parse
         =#

### expr[7]
#=
shift = [@rule getindex(~x, ~i...) =>
         :($x[$(map(((j, k),) -> Expr(:call, :+, k, sym[j]), enumerate(i))...)])]

push!(expr, Postwalk(Chain(shift))(last(expr)))
=#

#=
using Base.Iterators

### expr[8] : 0 + 0
quote
    function interior(V, A₁, A₂, A₃, T)
        map(product(UnitRange(first(axes(T, 1)) + 2, last(axes(T, 1)) - 2),
                    UnitRange(first(axes(T, 2)) + 2, last(axes(T, 2)) - 2),
                    UnitRange(first(axes(T, 3)) + 2, last(axes(T, 3)) - 2))) do (i, j, k)
            $(last(expr))
        end
    end
end |> eval

=#

#n = (8, 8, 8)

#=
V = rand(n...)
A₁, A₂, A₃ = map(_ -> rand(n...), n)
T = rand(n...)
R = similar(V)

using BenchmarkTools

@btime interior($V, $A₁, $A₂, $A₃, $T)

=#

#=
using Symbolics

@variables V[1:n[1], 1:n[2], 1:n[3]]
@variables A₁[1:n[1], 1:n[2], 1:n[3]]
@variables A₂[1:n[1], 1:n[2], 1:n[3]]
@variables A₃[1:n[1], 1:n[2], 1:n[3]]
@variables T[1:n[1], 1:n[2], 1:n[3]]
@variables Tᵧ[1:n[1], 1:n[2], 1:n[3]]
@variables i::Int j::Int k::Int
=#

#=
symexpr = eval(last(expr))
D = Differential(T[i, j, k])
jacexpr = Meta.parse(string(expand_derivatives(D(symexpr))))
=#

#push!(expr, Meta.parse(string(eval(last(expr)))))

#=
dT = interior(V, A₁, A₂, A₃, T)

(fastinterior, fastinterior!) =
    build_function(dT, V, A₁, A₂, A₃, T, expression = Val{false})

(multiinterior, multiinterior!) =
    build_function(dT, V, A₁, A₂, A₃, T,
                   expression = Val{false},
                   parallel = Symbolics.MultithreadedForm())

V = rand(n...)
A₁, A₂, A₃ = map(_ -> rand(n...), n)
T = rand(n...)
R = similar(V)

@btime $fastinterior($V, $A₁, $A₂, $A₃, $T)
@btime $fastinterior!($R, $V, $A₁, $A₂, $A₃, $T)
@btime $multiinterior($V, $A₁, $A₂, $A₃, $T)
@btime $multiinterior!($R, $V, $A₁, $A₂, $A₃, $T)
=#

nothing

