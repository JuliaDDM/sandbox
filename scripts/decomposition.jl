#################################################
#
# 2021-08-18  Frederic Nataf
#
#################################################

module decomposition
export create_partition , inflate_subdomain  , inflate_subdomain2

using SparseArrays , LightGraphs , GraphPlot , Metis , LinearAlgebra



"""
create_partition( g , npart )

Returns a vector of the subdomain indices of a partition of 1:size(g) into npart subdomains
# Arguments
- 'g' : the graph connections of the degrees of freedom
- 'npart'    : Number of subdomains
# Example
m=9
npart = 3
A = spdiagm(-1 => -ones(m-1) , 0 => 2. *ones(m) , 1 => -ones(m-1))
g = Graph(A)
initial_partition = create_partition( g , npart )
"""
function create_partition( g , npart )
    decomposition = Metis.partition(g, npart)
    subdomain_partition_indices = map( i-> findall( x-> (x==i) , decomposition ) , 1:npart )
    return subdomain_partition_indices
end


"""
inflate_subdomain( g_adj , subdomain_indices )

Returns the vector of the indices of the subdomain inflated by its direct neighbors as defined by the adjacency matrix 'g'.
# Arguments
- 'g_adj' : the adjacency matrix of the degrees of freedom with non zero on the diagonal (a square matrix)
- 'subdomain_indices'    : Indices of a subdomain
# Example
g_adj = adjacency_matrix(g ,  Int64 )
inflated_subdomains = map(sd->inflate_subdomain( g_adj , sd ) ,  initial_partition)
 voir inflate_subdomain2( sd ) pour une version composable
"""
function inflate_subdomain( g_adj , subdomain_indices )
    #trouver les voisins
    (n,m) = size(g_adj)
    vi=zeros(Int64,m)
    vi[subdomain_indices] .= 1
    vi = g_adj*vi
    inflated_subdomain_indices = findall(x->x>0,vi)
    return inflated_subdomain_indices
end


"""
inflate_subdomain2( g_adj )

Returns the function that enables to inflate a subdomain with its direct neighbors as defined by the adjacency matrix 'g_adj'.
# Arguments
- 'g_adj' : the adjacency matrix of the degrees of freedom with non zero on the diagonal (a square matrix)
# Example
g_adj = adjacency_matrix(g ,  Int64 )
inflate_subdomain_with_g_adj = inflate_subdomain2( g_adj )
inflated_subdomains = map(sd->inflate_subdomain_with_g_adj( sd ) ,  initial_partition)
 voir aussi  inflate_subdomain( sd , g_adj )
"""
function inflate_subdomain2( g_adj )
    function inflate( subdomain_indices )
        return inflate_subdomain( g_adj , subdomain_indices )
    end
    return inflate
end

end
