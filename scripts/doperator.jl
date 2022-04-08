
#     POU::DPOU# , en fait c'est plutôt un Shared_vector (cohérent qui en a besoin). En fait, on peut repousser la question POUM à plus tard.
# la chose principale est de pouvoir coder ∑_i R_i^T R_i
###
# vecteur global -> vecteur decompose coherent ( scattered "eclate" mais pas Shared_vector , decomposed DVector ) ->  vecteur decommpose incoherent
# |                   |
# |                   |
# |                   |
# alpha*u ---------> alpha*Du (a faire)
# A*u -------------> DA*Du (fait)
# (u,v) -----------> (Du,Dv)= ∑_i (Ui,Di V_i) (a faire)
# ASM independant de la partition de l'unite
# vecteur global <- vecteur partage coherent
#
# vecteur partage pas forcement coherent : les rendre cohérent , (cf. RAS)
# somme compensée pour etre plus stable vis a vis des erreurs d'arrondi  -->
#       -->  CF MakeCoherent si partition de l'unite non Booleenne ou Gradient conjugue (aussi???) ?

# # Vincent--> pour changer facilement de dictionnaire
# struct MyDict{K,V,T<:AbstractDict{K,V}} <: AbstractDict{K, V}
#     dict::T
# end
# # commenter une des deux lignes
# MyDict{K,V}() where {K,V} = MyDict(Dict{K,V}())
#
# MyDict{K,V}() where {K,V} = MyDict(ThreadSafeDict{K,V}())
#
# # Dict --> MyDict dans le code
# # pour eviter de redefinir toutes les fcts de Dict
# Base.parent(d::MyDict) = d.dict

##ou plus simplement, a la typedef C:
# const MyDict{K,V} = Dict{K,V}
# const MyDict{K,V} = ThreadSafeDict{K,V}

#################################################
#
#       struct DOperator
#
#################################################

# pour effectuer le produit matrice vecteur V_i =  ∑_j R'_i A R_j^T D_j U_j  (V = A U)
mutable struct DOperator
    DDomD::DDomain # domaine de départ décomposé
    DDomA::DDomain # domaine d'arrivée décomposé
    matvec # collection of local operators Aij = R'_i A R_j^T, i ∈ DDomA, j ∈ DDomD
    # ou plutot le produit matrice vecteur avec un DVector qui vit sur DDomD? ??
end



"""
# Arguments
- 'DDomD'
- 'A' : a square matrix given by its entries
"""
function DOperator(DDomD, A)
    DA =  ThreadSafeDict{Tuple{Domain,Domain},SparseMatrixCSC{Float64,Int64}}()
#    DA = Dict{Tuple{Domain,Domain},SparseMatrixCSC{Float64,Int64}}()
    ThreadsX.foreach(subdomains( DDomD ))  do sdi
#    for sdi ∈ subdomains(DDomD)
        for sdj ∈ subdomains(DDomD)
            DA[(sdi, sdj)] = A[global_indices(sdi), global_indices(sdj)]
        end
    end
    function shared_mat_vec( x )
        dom = x.domain
        res = zeros(dom)
        y = zeros(dom)
        #Diboolean pour avoir plus de reproductibilité?
#        di = Di( dom )
        di = Diboolean( dom )
        ThreadsX.foreach(subdomains( dom ))  do sdi
#        for sdi ∈ subdomains( dom )
            values( y , sdi ) .= values(di,sdi) .* values(x,sdi)
        end
        # boucle parallelisable
        ThreadsX.foreach(subdomains( res ))  do sdi
        # for sdi ∈ subdomains( res )
           values(res , sdi) .= DA[(sdi,sdi)]*values(y , sdi)
        end
     # boucle exterieur parallelisable
     ThreadsX.foreach(subdomains( y ))  do sdi
        # for sdi ∈ subdomains( y )
           # boucle sequentiel
          # ThreadsX.foreach(subdomains( y ))  do sdj essayé pour voir et effectivement non parallelisable sous cette forme.
           for sdj ∈ subdomains( y )# ou plus efficace et plus clair , passer par les cles avec premier element fixe et deuxieme sdj
              if ( haskey(DA , (sdi,sdj) ) &&  !( sdi == sdj))
                 values(res , sdi) .+= DA[(sdi,sdj)]*values(y , sdj)
              end
           end
        end
        return MakeCoherent(res)
     end
     return DOperator( DDomD , DDomD , shared_mat_vec )
end


"""
# Arguments
- 'DDomD'
- 'A' : a square matrix given by its entries
"""
function DOperatorSequential(DDomD, A)
    DA = Dict{Tuple{Domain,Domain},SparseMatrixCSC{Float64,Int64}}()
    for sdi ∈ subdomains(DDomD)
        for sdj ∈ subdomains(DDomD)
            DA[(sdi, sdj)] = A[global_indices(sdi), global_indices(sdj)]
        end
    end
    function shared_mat_vec( x )
        dom = x.domain
        res = zeros(dom)
        y = zeros(dom)
        #Diboolean pour avoir plus de reproductibilité?
#        di = Di( dom )
        di = Diboolean( dom )
        for sdi ∈ subdomains( dom )
            values( y , sdi ) .= values(di,sdi) .* values(x,sdi)
        end
        # boucle parallelisable
        for sdi ∈ subdomains( res )
           values(res , sdi) .= DA[(sdi,sdi)]*values(y , sdi)
        end
     # boucle exterieur parallelisable
        for sdi ∈ subdomains( y )
           # boucle sequentiel
           for sdj ∈ subdomains( y )# ou plus efficace et plus clair , passer par les cles avec premier element fixe et deuxieme sdj
              if ( haskey(DA , (sdi,sdj) ) &&  !( sdi == sdj))
                 values(res , sdi) .+= DA[(sdi,sdj)]*values(y , sdj)
              end
           end
        end
        return MakeCoherent(res)
     end
     return DOperator( DDomD , DDomD , shared_mat_vec )
end



#################################################
#
#       struct DOperatorBlockJacobi
#
#################################################

struct DOperatorBlockJacobi
    DDomD::DDomain # domaine de départ et d'arrivée décomposé
    DDomA::DDomain # domaine de départ et d'arrivée décomposé
    matvec # collection of local solvers
end

"""
DOperatorBlockJacobi(DDomD, A)

Returns direct local solvers for the Dirichlet matrices of a global matrix A
# Arguments
- 'DDomD'
- 'A' : a square matrix given by its entries
"""
function DOperatorBlockJacobi(DDomD, A)
    # DA_lu = ThreadSafeDict()
    # ThreadsX.foreach(subdomains( DDomD ))  do sdi
    #     #        for sdi ∈ subdomains(DDomD)
    #     DA_lu[sdi] = factorize(A[global_indices(sdi), global_indices(sdi)]  )
    # end

    DA_lu = Dict()
#    ThreadsX.foreach(subdomains( DDomD ))  do sdi
    for sdi ∈ subdomains(DDomD)
        DA_lu[sdi] = factorize(A[global_indices(sdi), global_indices(sdi)]  )
    end

    function shared_mat_vec( x )
        dom = x.domain
        res = DVector( dom , 0. )
        # boucle parallelisable , Di a ajouter somewhere
        for sdi ∈ subdomains( res )
            values(res , sdi) .= DA_lu[sdi]\ values(x , sdi)
        end
        return res
    end
    return DOperatorBlockJacobi( DDomD , DDomD , shared_mat_vec )
end
