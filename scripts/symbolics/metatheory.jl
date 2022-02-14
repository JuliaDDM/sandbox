# 1D
using Metatheory
using Metatheory.Rewriters

const op = (:δ⁻, :δ⁺, :σ⁻, :σ⁺)
const dims = 1:3
const sub = ("\u2081", "\u2082", "\u2083")

### expr[1]
expr = [:(sum(σ⁺(A[i], i) * δ⁺(δ⁻(σ⁺(A[i], i) * T, i) / σ⁻(V, i), i), i))]

### expr[2]
function einstein(sym, num)
    [@rule(getindex(~x, $sym) => :($(Symbol(x, sub[num])))),
     @rule((~f::in(op))(~x, $sym) => :($f($x, $num)))]
end

einstein(expr) = rewrite(expr, [@rule sum(~x, ~i) =>
    :((+)($(map(j -> rewrite(x, einstein(i, j)), dims)...)))])

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
isasym = Base.Fix2(isa, Symbol)
index = [@rule(~x::isasym => :($x[$(ntuple(zero, length(dims))...)]))]

push!(expr, Postwalk(Chain(index))(last(expr)))

### expr[6]
rule = @rule θ(getindex(~x, ~i...), ~d, ~s) =>
    :($x[$(map(((j, k),) -> k + j == d ? s : 0, enumerate(i))...)])

push!(expr, rewrite(last(expr), [rule]))

### expr[7]
bc(sym) = @rule getindex($sym, ~i, -1, ~k...) --> 0
push!(expr, rewrite(last(expr), bc.(Symbol.(:A, sub))))

### expr[8] : 0 + 0

