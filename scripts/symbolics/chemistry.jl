rhs(u, t, param) =
    [param * u[1] * √u[3] * exp(-1 / u[3]),
     param * u[2] ^ 2 * u[3] * exp(-1 / u[3]),
     param * u[1] * u[3] ^ 2* exp(-1 / u[3]) +
     param * u[2] ^ 2 * u[3] * exp(-1 / u[3])]

function rhs!(du, u, t, param)
    du[1] = param * u[1] * √u[3] * exp(-1 / u[3])
    du[2] = param * u[2] ^ 2 * u[3] * exp(-1 / u[3])
    du[3] = param * u[1] * u[3] ^ 2* exp(-1 / u[3]) +
            param * u[2] ^ 2 * u[3] * exp(-1 / u[3])
    nothing
end

param = -1

using Symbolics

@variables t u[1:3]
du = rhs(u, t, param)

(f, f!) = build_function(du, u, t, expression = Val{false})

# array
jac = Symbolics.jacobian(du, collect(u))
(df, df!) = build_function(jac, u, expression = Val{false})

# sparse
spjac = Symbolics.sparsejacobian(du, collect(u))
(spdf, spdf!) = build_function(spjac, u, expression = Val{false})

t = 0.0
u = rand(3)
du = similar(u)

using BenchmarkTools

@btime rhs!($du, $u, $t, $param)
@btime $f!($du, $u, $t)

u = rand(3)

jac = Matrix{Float64}(undef, 3, 3)
@btime $df!($jac, $u)

jac = similar(spjac, Float64)
@btime $spdf!($jac, $u)

"""
Some idea of how this could look like
```julia
map(chemical::AbstractVector) do (A, b, E, n, νᵣ, νₚ)
    res = reduce(*, enumerate(νᵣ)) do i, ν
        Y[i] ^ ν
    end
    res * Y[19] ^ n * exp(-E / R / Y[19])
end
```

"""
nothing

