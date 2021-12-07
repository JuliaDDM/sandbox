# essai de partir du point de vue utilisateur
# voir les notes manuscrites
# il existe qq part une numérotation globale
# Analogie avec MPI (MPICommWorld , Communicators , subcommunicators , etc ... ) serait utile????
# processus -> un sous ensemble de degres de liberté
# Communicators (des processus et des moyens de communiquer) ->  des degrés de liberté == sous domaine
# MPICommWorld -> l'ensemble des degrés de liberté => vision rigide car le nombre de dof (processus) varie au cours du calcul

# pour effectuer le produit matrice vecteur V_i =  ∑_j R'_i A R_j^T D_j U_j  (V = A U)
mutable struct DOperator
    DDomD::DDomain # domaine de départ décomposé
    DDomA::DDomain # domaine d'arrivée décomposé
    matvec # collection of local operators Aij = R'_i A R_j^T, i ∈ DDomA, j ∈ DDomD
end

mutable struct DDomain
    up::Domain # le domaine que l'on décompose
    subdomains::Set{Domain} # ensemble des sous domaines
    neighborhood::Dict{Subdomain,Vector{Tuple{Int64,Int64}}}
    # subdomain_vois > vecteur ( k_loc , k_vois )
    POU::DPOU# douteux a enlever

end

mutable struct Shared_vector
    domain::DDomain
    vectors::Dict{Subdomain, Vector{Float64}}
    # + , - , a* , .* , similar etc ... si on peut automatiquement hériter de ce qui vient de vecteur, on a gagné voir comment faire en Julia
    # ce qui est lié à l'aspect cohérent : prodscal et donc demande une partition de l'unite qqsoit
    # relation avec les vecteurs "habituels" : import_from_global(!) , export_to_global(!)

    #
end

###
# vecteur global -> vecteur decompose coherent (nom malheureux , scattered "eclate" , decomposed DVector ) ->  vecteur decommpose incoherent
# |                   |
# |                   |
# |                   |
# alpha*u ---------> alpha*Du
# A*u -------------> DA*Du
# (u,v) -----------> (Du,Dv)= ∑_i (Ui,Di V_i)
# ASM independant de la partition de l'unite
# vecteur global <- vecteur partage coherent
#
# vecteur partage pas forcement coherent : les rendre cohérent , (RAS)
# somme compensée pour etre plus stable vis a vis des erreurs d'arrondi  --> CF MakeCoherent si partition de l'unite non Booleenne ou Gradient conjugue (aussi???) ?


#https://docs.julialang.org/en/v1/manual/constructors/
struct Domain# sous domaine aussi
    up::Domain # le (un??) surdomain éventuellement lui-même ou Nothing???
    loctoglob::AbstractVector{Int64} # vecteur d'indices de up qui sont Domain, Int64 pourrait être un paramètre cf indices cartésiens ...
    Domain(loctoglob::AbstractVector{Int64}) = ( D = new(); D.loctoglob = copy(loctoglob) ; D.up = D; return D; )
    Domain(up,loctoglob) =  issubset(loctoglob,up.loctoglob) ?  new(up,loctoglob) : error("indices $loctoglob have to be a subset of the superdomain")
end

#    Di  # la partition de l'unité locale au sous domaine vue comme un operateur local verifiant une certaine propriété
#  Di est définie sur un DDomain (constructeur) et on l'interroge en luio donnant un sous domaine et il renvoie le vecteur des poids D(Omage_i)
 mutable struct DPOU
   Ddomain::DDomain
 #  constructeur(DDomain,options)
 #  DPOU(DDomain) renvoie le vecteur des poids sur le domaine en question
end
#RAS: constructeur(DDomain, A , POU) , la POU est associée à un algorithme plus qu'à une décomposition si argument POU absent, on prend les Di implicite à ???



#
#   Create domain initial_decomposition => DDomain
#
#   Create parition of unity  => DDomain aussi ???
#
#   Create decomposed Operator  => DA
#
#
