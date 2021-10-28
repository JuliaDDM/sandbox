include("decomposition2.jl")
# faire deux fois include("decompositionEncoursFrederic.jl") semble poser problème
using SparseArrays , LightGraphs , GraphPlot , Metis , LinearAlgebra , .decomposition


# m=9
# A = spdiagm(-1 => -ones(m-1) , 0 => 2. *ones(m) , 1 => -ones(m-1))


npart = 3
sdiff1(m) = spdiagm(-1 => -ones(m-1) , 0 => ones(m) )
# make the discrete -Laplacian in 2d, with Dirichlet boundaries
# adapted from https://math.mit.edu/~stevenj/18.303/lecture-10.html
function Laplacian2d(Nx, Ny, Lx, Ly)
   dx = Lx / (Nx+1)
   dy = Ly / (Ny+1)
   Dx = sdiff1(Nx) / dx
   Dy = sdiff1(Ny) / dy
   Ax = Dx' * Dx
   Ay = Dy' * Dy
   return kron( spdiagm( 0 => ones(Ny) ) , Ax) + kron(Ay, spdiagm( 0 => ones(Nx) ))
end

 m=5
 n=6

A = Laplacian2d(m,n,1,1)

g = Graph(A)
initial_decomposition = create_partition_subdomain( g , npart )
g_adj = adjacency_matrix(g ,  Int64 )
# will work iff npart >= 3
inflate_subdomain!( g_adj , initial_decomposition[1] , initial_decomposition );

inflate_subdomain!( g_adj , initial_decomposition[1] , initial_decomposition );

inflate_subdomain!( g_adj , initial_decomposition[1] , initial_decomposition );

inflate_subdomain!( g_adj , initial_decomposition[1] , initial_decomposition );

inflate_subdomain!( g_adj , initial_decomposition[2] , initial_decomposition );

inflate_subdomain!( g_adj , initial_decomposition[3] , initial_decomposition );

#initial_decomposition

Vshareddict = Dict{Subdomain,Vector{Float64}}();
for sd ∈   initial_decomposition
    Vshareddict[sd] = ones(ndof(sd));
end

Vshared = Shared_vector(Vshareddict);

# Mettre cet appel avant???
for sd ∈ subdomains( Vshared )
    create_buffers_communication!( sd );
end

Update_wo_partition_of_unity!(Vshared)

vuesur( Vshared )

uglob = 4. * ones(ndof(Vshared))

import_from_global!( Vshared , uglob );

vuesur( Vshared )

Update_wo_partition_of_unity!(Vshared)

vuesur( Vshared )

println(export_to_global(Vshared))

# test à ajouter
# convergence de RAS
# divergence de ASM
