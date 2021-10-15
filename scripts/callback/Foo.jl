# https://julialang.org/blog/2013/05/callback/
module Foo

export bar

const libfoo = joinpath(pwd(), "libfoo")

function wrap(a, b, thunk::Ptr{Cvoid})
    func = unsafe_pointer_to_objref(thunk)::Function
    func(a, b)
end

function bar(func, a, b)
    cwrap = @cfunction(wrap, Cdouble, (Cdouble, Cdouble, Ptr{Cvoid}))
    @ccall libfoo.bar(a::Cdouble, b::Cdouble, func::Any, cwrap::Ptr{Cvoid})::Cdouble
end

end

using .Foo

# example with top-level function
function func(x, y)
    return x + y
end
z = bar(func, 1, π)
println("Top-level : 1 + π = $z")

# example with anonymous function
z = bar(√2, -1) do x, y
    x - y
end
println("Anonymous : √2 - -1 = $z")

