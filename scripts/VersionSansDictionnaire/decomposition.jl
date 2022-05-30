#using LightGraphs, GraphPlot, Plots , Metis
using LightGraphs, Metis



"""
create_partition( g , npart )

Returns a pair: (vector of indices making up a partition of 1:size(g) into npart subdomains , the coloring of the dofs)
# Arguments
- 'g' : the graph connections of the degrees of freedom
- 'npart'    : Number of subsets of indices
# Example
"""
function create_partition(g, npart)
    decomposition = Metis.partition(g, npart)
    partition_indices = map(i -> findall(x -> (x == i), decomposition), 1:npart)
    return (partition_indices, decomposition)
end


"""
create_partition_DDomain( Domain , g , npart )

Returns a decomposed domain that forms a partition of 1:size(g) into npart subdomains
# Arguments
- 'g' : the graph connections of the degrees of freedom
- 'npart'    : Number of subdomains
# Example
m=9
npart = 3
A = spdiagm(-1 => -ones(m-1) , 0 => 2. *ones(m) , 1 => -ones(m-1))
g = Graph(A)
initial_partition = create_partition_DDomain( g , npart )
"""
function create_partition_DDomain(domain, g, npart)
    # il manque un contrôle sur la cohérence entre les indices de domain et ceux de g et surtout l'expression à donner à ce contrôle.
    (initial_partition, decomposition) = create_partition(g, npart)
    res_up = domain
    res_subdomains = Vector{typeof(domain)}()
    for indices ∈ initial_partition
        newsd = Domain(domain, indices)
        push!(res_subdomains, newsd)
    end
    return DDomain(res_up, res_subdomains)
end


"""
inflate_indices( g_adj , indices )

Returns the vector of the indices inflated by its direct neighbors as defined by the adjacency matrix 'g'.
# Arguments
- 'g_adj' : the adjacency matrix of the degrees of freedom with non zero on the diagonal (a square matrix)
- 'indices'    : a set of indices
# Example on initial_partition (a vector of vector of indices)
g_adj =
(inflated_indices , decomposition) = map(sub_id->inflate_indices( g_adj , sub_id ) ,  initial_partition)
"""
function inflate_indices(g_adj, indices::Vector{Int64})
    #trouver les voisins
    (n, m) = size(g_adj)
    vi = zeros(Float64, m)
    vi[indices] .= 1.
    vi = g_adj * vi
    inflated_indices = findall(x -> x > 0, vi)
    return inflated_indices
end



"""
inflate_subdomain!( g_adj , subdomain )

Inflate a 'subdomain' and update the overlap of itself and of its neighbors
# Arguments
- 'g_adj' : the adjacency matrix of some matrix
- 'subdomain'   :  subdomain to be inflated.
"""
function inflate_subdomain!(g_adj, subdomain)
    # up.subdomain == domain ??? A FAIRE
    # g_adj a bien la taille de up.subdomain???
    # que se passe t il avec la partition de l'unité de domain???
    inflated_indices = inflate_subdomain(g_adj, global_indices(subdomain))
    new_indices = filter(x -> !(x in global_indices(subdomain)), inflated_indices)
    append!(global_indices(subdomain), new_indices)#loctoglob est mis a jour, danger creer un nouveau sous domaine plutot?
end
