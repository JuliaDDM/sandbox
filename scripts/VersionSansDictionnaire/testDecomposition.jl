
include("ddmUtilities.jl")
include("decomposition.jl")
using SparseArrays

m=901
npart = 3
A = spdiagm(-1 => -ones(m-1) , 0 => 2. *ones(m) , 1 => -ones(m-1))
g = Graph(A)
(initial_partition  , decomposition) = create_partition( g , npart )
