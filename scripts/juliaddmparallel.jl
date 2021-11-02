include("decompositionparallel.jl")
using SparseArrays , LightGraphs , GraphPlot , Metis , LinearAlgebra , BenchmarkTools , .decomposition, ThreadsX



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


npart = 3
m=9
n=9

A = spdiagm(-1 => -ones(m-1) , 0 => 2. *ones(m) , 1 => -ones(m-1))
# A = Laplacian2d(m,n,1,1)

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
# construcution d'un vecteur partagé
Vshareddict = Dict{Subdomain,Vector{Float64}}();
for sd ∈   initial_decomposition
    Vshareddict[sd] = ones(ndof(sd));
end
Vshared = Shared_vector(Vshareddict);

# Mettre cet appel avant???
for sd ∈ subdomains( Vshared )
    create_buffers_communication!( sd );
end


#Update_wo_partition_of_unity!(Vshared)

#vuesur( Vshared )

uglob = 4. * ones(ndof(Vshared));

import_from_global!( Vshared , uglob );

#vuesur( Vshared )

Update_wo_partition_of_unity!(Vshared);

#vuesur( Vshared )

println(export_to_global(Vshared))

# pour motiver le parallelisme programmer RAS

domain = Domain( initial_decomposition );

rhs = ones(ndof(Vshared))
u = similar(rhs)
###############################################################
#
#    RAS premier jet sans le produit matrice vecteur parallele
#    U_{n+1} = U_n + ∑_i  R_i^T D_i Ai^{-1} R_i (b - A*U_n)
#    U_{n+1} =  ∑_i [ R_i^T D_i ( R_i U_n + Ai^{-1} R_i ](b - A*U_n)
#
###############################################################
# factorisation des matrices locales
Ai_lu = Dict()
#ThreadsX.foreach(subdomains( U )) do sd # surement faux pour l'instant à cause du dictionaire.
   for sd ∈ subdomains( domain )
   Ai_lu[sd] = factorize(A[ global_indices( sd ) , global_indices( sd ) ])
end
u .= 0.
itmax = 50
Riu = import_from_global( domain , u );
Rirhs = import_from_global( domain , rhs );
for i ∈ 1:itmax
   residual = rhs - A*u
   import_from_global!( Rirhs , residual )
   import_from_global!( Riu , u )
ThreadsX.foreach(subdomains( domain )) do sd  #  for ( sd , facteur ) ∈ Ai_lu
      decomposition.values( Riu , sd ) .+= Ai_lu[sd]\ decomposition.values( Rirhs , sd )
   end
   Update_wi_partition_of_unity!( Riu )
   export_to_global!( u , Riu )
end

# test à ajouter
# convergence de RAS
# divergence de ASM
# tests de non régression ?? : ndofs, Nsd (test en dur ATTENTION le test devra prendre une partition metis en argument )
# tests en partant de uglobal == 1 puis update et resultat ≤ N
