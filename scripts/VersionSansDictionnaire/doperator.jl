
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
##ou plus simplement, a la typedef C:
# const MyDict{K,V} = Dict{K,V}
# const MyDict{K,V} = ThreadSafeDict{K,V}
# la vraie perte avec les ThreadSafeDict est éventuellement "aux allocations des clès"

#################################################
#
#       struct DOperator
#
#################################################

# pour effectuer le produit matrice vecteur V_i =  ∑_j R'_i A R_j^T D_j U_j  (V = A U)
mutable struct DOperator
    DDomD::DDomain # domaine de départ décomposé
    DDomA::DDomain # domaine d'arrivée décomposé
    matvec # the matrix vector product here collection of local operators Aij = R'_i A R_j^T, i ∈ DDomA, j ∈ DDomD
    # à "encapsuler" en un produit matrice vecteur avec un DVector qui vit sur DDomD? ??
end

"""
# Arguments
- 'DDomD'
- 'A' : a square matrix given by its entries
"""
function DOperator(DDomD, A)
    DA = Vector{Tuple{Domain,Domain,SparseMatrixCSC{Float64,Int64},Int64,Int64}}()
    # not safe for parallelism since the vector has yet to be allocated
    for ( i , sdi ) ∈ enumerate(subdomains(DDomD))
        for ( j, sdj ) ∈ enumerate(subdomains(DDomD))
            Aij = A[global_indices(sdi), global_indices(sdj)]
            if !(iszero(Aij))
                push!(DA, ( sdi , sdj , Aij , i , j )  )
            end
        end
    end
    function shared_mat_vec( x )
        dom = x.domain
        res = zeros(dom)
        #y = zeros(dom)
        #Diboolean pour avoir plus de reproductibilité
        di = Diboolean( dom )
        #for ( yvec ,  divec , xvec ) ∈ zip( y.data , di.data , x.data )
        #    yvec[2] .= divec[2] .* xvec[2]
        #end
        y = dot_op( di , x , (.*))
        # buffers to store Aij*x[j] , reallocated at each call to ensure safe parallelism
        Aijvj = Vector{Union{Tuple{Domain,Domain,Vector,Int64,Int64},Nothing}}( nothing , length(DA) )
        ThreadsX.foreach(  enumerate(zip( Aijvj , DA )) ) do ( i, (ddv , ddop) )
    #    for ( i, (ddv , ddop) ) ∈  enumerate(zip( Aijvj , DA ))
            Aijvj[i] = ( ddop[1] , ddop[2] , ddop[3] * y.data[ ddop[5] ].second , ddop[4] , ddop[5] )
        end
# https://sparsearrays.juliasparse.org/dev/
# to have something efficient, should look at
# and install version 1.8 of Julia
# with sparse arrays
# with sparse block matrices
#
        for inc ∈ Aijvj
            res.data[ inc[4] ][2] .+= inc[3]
        end
        return MakeCoherent(res)
    end
    return DOperator( DDomD , DDomD , shared_mat_vec )
end



import Base.*

function (*)( A::DOperator , v)
    return A.matvec( v)
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
    woDA_lu = Vector{typeof(factorize(A))}( undef , length(DDomD) )
    ThreadsX.foreach( enumerate(subdomains(DDomD)) ) do ( i , sdi )
    #for ( i , sdi )  ∈ enumerate(subdomains(DDomD))
        woDA_lu[i] = factorize(A[global_indices(sdi), global_indices(sdi)]  )
    end
    function shared_mat_vec( x )
        dom = x.domain
        res = DVector( dom , 0. )
        ThreadsX.foreach( enumerate(subdomains(dom)) ) do ( i , sdi )
        #for ( i , sdi ) ∈ enumerate(subdomains(dom))
            res.data[i].second .= woDA_lu[i] \ x.data[i].second
        end

        return res
    end
    return DOperatorBlockJacobi( DDomD , DDomD , shared_mat_vec )
end
