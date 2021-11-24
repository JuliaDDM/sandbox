# essai de partir du point de vue utilisateur
# voir les notes manuscrites
# il existe qq part une numérotation globale
# Analogie avec MPI (MPICommWorld , Communicators , subcommunicators , etc ... ) serait utile????
# processus -> un sous ensemble de degres de liberté
# Communicators (des processus et des moyens de communiquer) ->  des degrés de liberté == sous domaine
# MPICommWorld -> l'ensemble des degrés de liberté


# pour effectuer le produit matrice vecteur V_i =  ∑_j R'_i A R_j^T D_j U_j  (V = A U)
mutable struct DOperator
    DDomD::DDomain # domaine de départ décomposé
    DDomA::DDomain # domaine d'arrivée décomposé
    matvec # collection of local operators Aij = R'_i A R_j^T, i ∈ DDomA, j ∈ DDomD
end

#https://docs.julialang.org/en/v1/manual/constructors/
struct Domain# sous domaine aussi
    up::Domain # le (un??) surdomain éventuellement lui-même ou Nothing???
    loctoglob::AbstractVector{Int64} # vecteur d'indices de up qui sont Domain
    Domain(loctoglob::AbstractVector{Int64}) = ( D = new(); D.loctoglob = copy(loctoglob) ; D.up = D; return D; )
    Domain(up,loctoglob) =  issubset(loctoglob,up.loctoglob) ?  new(up,loctoglob) : error("indices $loctoglob have to be a subset of the superdomain")
end

mutable struct POU
    domain::Domain # un sous domaine
    Di  # la partition de l'unité locale au sous domaine
end


mutable struct DDomain
    up::Domain # le domain que l'on décompose
    subdomains::Set{POU} # ensemble de (sous domaines , Partition_of_unity) decomposant up
end

#
#   Create domain initial_decomposition => DDomain
#
#   Create parition of unity  => DDomain aussi ??? 
#
#   Create decomposed Operator  => DA
#
#
