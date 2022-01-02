# Domains and their decomposition

## Domains and subdomains

From a domain $\Omega$ having $n$ d.o.fs, we can create a subdomain $\Omega_i$ having $n_i$ d.o.f.s. This subdomain is seen as a domain which is characterized by two things:
1. its parent domain, here $\Omega$, in the code it is referred to as **up**. 
2. an injection $f_i : [1:n_i] \mapsto {\mathcal N}_i$ where ${\mathcal N}_i$ is the set of degrees of freedom of the subdomain. In the code this injection is stored as a vector called **loctoglob**, that is $$\text{loctoglob} = [f_i(1),\,f_i(2) ,\, \ldots,\, f_i(n_i) ]\,.$$  


**Note** When a domain is not a subdomain, its parent **up** is itself and the vector **loctoglob** refers to a set of degrees of freedom, usually simply the range $[1:n]$. 

This way, it makes sense to create a subdomain of a subdomain.


**Unclear topics** 
- what about the "confusion" between a subdomain and a domain??
- seemingly related question, what about indexing with something different from integers e.g. Cartesian indices? 
- these questions seem related to the numbering of the d.o.f's of a domain which is demanded by interactions with linear algebra. 

## Domain decomposition

It is defined as:
1.  a domain $\Omega$ (which may be in practice a subdomain of some sub/domain)
2. a set (i.e. unordered) of subdomains $\{\Omega_1 ,\, \Omega_2 ,\, \ldots ,\, \Omega_N\}$

For making it useful for a decomposed vector, see [Decomposed vector](#Decomposed-vector) below, we build its overlaps $({\mathcal N}_i \cap {\mathcal N}_j)_{i \neq j}$ which is stored a dictionary of dictionary of the intersection (à la matlab, i.e. we keep track of the location in the original subdomains) 

A FAIRE POUR CLARIFIER:
- [ ] ajouter la possibilité d'indexer avec des indices cartésiens pour débloquer les confusions liées à la numérotation



# Decomposed vector


**Unclear topics** 
- la gestion des informations nécessaires à l'opération $$R_i \sum_j R_j^T\,U_j \longrightarrow V_i$$ sera très différente selon que l'on est en MPI, OpenMP ou autre. 