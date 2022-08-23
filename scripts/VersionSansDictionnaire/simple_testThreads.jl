
#################################################
#
# 2022-08-07  Frederic Nataf
#
#################################################
using SparseArrays,  LinearAlgebra, ThreadsX,   FLoops ,  BenchmarkTools


BLAS.set_num_threads(1)

m = 50;
n = 1000;
A = sprand(Float64,n,n,0.1);
A= sparse((A+A')+Matrix(1000.0I, n, n));

Avec = fill( spzeros(size(A)) , m )
for i in 1:length(Avec)
    Avec[i] =  deepcopy(A)
end


function DOperatorBlockJacobiForTest( Avec)
#    ThreadsX.foreach( Avec ) do ( mat )
        ThreadsX.foreach( 1:length(Avec) ) do ( i )
     cholesky(Avec[i]);
 end
end


DOperatorBlockJacobiForTest( Avec);

println("Running test with $( Threads.nthreads()) threads")

@time DOperatorBlockJacobiForTest( Avec);

# for T in 1 2 4 8 12 24 ; do echo "# Using $T thread(s): "; julia -t $T ./simple_testThreads.jl; echo ""; done
