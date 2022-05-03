

const MyDict{K,V} = ThreadSafeDict{K,V}
#const MyDict{K,V} = Dict{K,V}  # ERROR: LoadError: TaskFailedException Stacktrace: [1] wait @ ./task.jl:334 [inlined]


"""
intersectalamatlab( a , b )

Returns the intersection of the values of a and b as well their positions
# Arguments
- 'a' and 'b' are vectors
# Example
a =[3 , 45 , 123 , 12]
b = [12 , 19 , 46 , 56 , 123]
intersectalamatlab( a , b )
([123, 12], [3, 4], [5, 1])
"""
function intersectalamatlab(a, b)
    function findindices!(resa, ab, a)
        for (i, el) ∈ enumerate(ab)
            resa[i] = findfirst(x -> x == el, a)
        end
    end
    ab = intersect(a, b)
    resa = Vector{Int64}(undef, length(ab))
    findindices!(resa, ab, a)
    # comment lines resa and resb
    resa
    resb = similar(resa)
    findindices!(resb, ab, b)
    resb

    return (ab, resa, resb)
end