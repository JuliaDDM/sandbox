# https://julialang.org/blog/2013/05/callback/
const libfoo = joinpath(pwd(), "libfoo")

function func(a, b)
    return a + b
end

function wrap(a, b, thunk::Ptr{Cvoid})
    func = unsafe_pointer_to_objref(thunk)::Function
    func(a, b)
end

function bar(func, a, b)
    cwrap = @cfunction(wrap, Cdouble, (Cdouble, Cdouble, Ptr{Cvoid}))
    @ccall libfoo.bar(a::Cdouble, b::Cdouble, func::Any, cwrap::Ptr{Cvoid})::Cdouble
end

a, b = 1, 3.0
c = bar(func, a, b)
println("$a + $b = $c")

