using Symbolics
import Symbolics: Arr, scalarize

const subscripts = ("\u2081", "\u2082", "\u2083")

n = (6, 6, 6)

components(sym::Symbol, n::NTuple{N,Int}) where {N} = 
    ntuple(N) do i
        Symbol(sym, subscripts[i])
    end

variables(sym::Symbol, n::NTuple{N,Int}) where {N} =
    @variables($sym[Base.OneTo.(n)...]) |> only

variables(sym::NTuple{N,Symbol}, n::NTuple{N,Int}) where {N} =
    map(sym) do el
        @variables($el[Base.OneTo.(n)...]) |> only
    end

function gradient(n; T = :T, A = :A)
    T̂ = variables(T, n)
    Â = variables(components(A, n), n)

    map(Â) do B
        B .* T̂
    end
end

G = gradient(n)

# D = divergence(n; U = :U, ...) should work whether U is
# - a Symbol ;
# - an Arr{Num}.

