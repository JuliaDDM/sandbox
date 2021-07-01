using Distributed
using SparseArrays
using LinearAlgebra
using IterativeSolvers
using Plots
# interval [1:120] into three parts
# as a dictionnray of subdomains
Ni = Dict()
# TODO ajouter le recouvrement comme un parametre
push!(Ni, 1 => collect(1:70)  )
push!(Ni, 2 => collect(51:90)  )
push!(Ni, 3 => collect(70:120) )
#global matrix
m=120
A = spdiagm(-1 => -ones(m-1) , 0 => 2. *ones(m) , 1 => -ones(m-1))
#global sol
rhs = ones(m)
#Direct solve
A_lu = factorize(A)
udirect = A_lu \ rhs
uexact = udirect
# Additive Schwarz Method (ASM) preconditioner
interrupt()
nbworkers = 2
addprocs(nbworkers)
@everywhere using LinearAlgebra

Ai_lu = Dict()

#pmap
# map( i -> factorize(A[ i , i ])     ,  values(Ni)   )       is OK
# AND  pmap( i -> A[ i , i ]*A[ i , i ]     ,  values(Ni)   ) is OK
# BUT  pmap( i -> factorize(A[ i , i ])     ,  values(Ni)   ) is not !!!!
# message d'erreur :
#3-element Array{SuiteSparse.CHOLMOD.Factor{Float64},1}:
#Error showing value of type Array{SuiteSparse.CHOLMOD.Factor{Float64},1}:
#ERROR: ArgumentError: pointer to the SuiteSparse.CHOLMOD.C_Factor{Float64} object is null.
##This can happen if the object has been serialized.

