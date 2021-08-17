module MultiwayNumberPartitioning

using JuMP, LinearAlgebra

export partition, partition_min_largest!, partition_max_smallest!, partition_min_range!

"""
    partition([objective! = partition_min_range!], S::AbstractVector, k::Int; optimizer) -> Vector{Int}

Find a partition of `S` into `k` subsets such the sums of the subsets are "as near as possible".

Inputs:
* (optional): an objective, such as [`partition_min_largest!`](@ref), [`partition_max_smallest!`](@ref), or [`partition_min_range!`](@ref). Defaults to `partition_min_range!`.
* `S` a collection of values to partition,
* `k` a number of subsets to partition into
* `optimizer`: a JuMP-compatible mixed-integer optimizer, such as GLPk, HiGHS, Cbc, Gurobi, etc.

Returns a vector of indices `v` such that `v[i] == j` if `S[i]` is assigned to subset `j`. 
"""
partition(group_sizes, k; optimizer) = partition(partition_min_range!, group_sizes, k; optimizer)

function partition(objective!, group_sizes::AbstractVector, k::Int; group_labels = zeros(Int, length(group_sizes), 0), optimizer)
    model = Model(optimizer)
    group_partitions = populate_model!(model, group_sizes, k; group_labels)
    objective!(model)
    JuMP.optimize!(model)
    # Now we've got a sort of one-hot encoded partitioning:
    onehot_partition = value.(group_partitions)

    # We wish to convert it to a vector of indexes so that the `i`th element of the
    # vector corresponds to the subset `j` that element `i` belongs to:
    partition = dropdims(mapslices(row -> only(findall(>(0.5), row)), onehot_partition; dims=2); dims=2)

    return partition
end

# Populates an empty `model` with variables and objectives
function populate_model!(model, S, k; group_labels, α=0)
    N = length(S)

    # does element i belong to set j
    @variable(model, group_membership[i=1:N, j=1:k], binary=true)

    # Element `i` belongs to exactly one set j
    @constraint(model, [i=1:length(S)], sum(group_membership[i, :]) == 1)

    # The sum of the `j`th subset can be expressed as a dot product between the `j`th column of `element` and `S`.
    # Why? Because the `j`th column of `element` is a boolean vector expressing which elements are in set `j`.
    # If element `i` is in set `j`, then `element[i,j]==1` and we add `element[i,j]*S[i] = S[i]` to our total.
    # If element `i` is *not* in set `j`, then `element[i,j]=0` and we add nothing to our total.
    # Thus, the total is the sum of subset `j`.
    @expression(model, subset_sum[j=1:k], dot(group_membership[:, j], S))


    # group_labels[i, l]

    # How many times does subset `j` have label `l`
    L = size(group_labels, 2)
    weights = [ sum(group_labels[:, l]) for l = 1:L]
    @expression(model, subset_labels[j=1:k, l=1:L], dot(group_membership[:,j], group_labels[:, l]) / weights[l])


    # want `subset_labels[:, l]` to be uniform: for each label `l`, each subset has approximately the same number. So we will maximize the entropy.
    @variable(model, t[j=1:k, l=1:L])

    @constraint(model, entropy_con[j=1:k, l=1:L], [t[j, l], subset_labels[j, l], 1] in MOI.ExponentialCone())
    @expression(model, sum_of_entropies, sum(t))
    
    # We can remove some non-uniqueness by ordering our subset sums.
    # This is also convenient for getting the smallest and largest elements.
    @constraint(model, [j=1:(k-1)], subset_sum[j] <= subset_sum[j+1])

    return group_membership
end

"""
    partition_min_largest!(model)

Adds an objective function to the model to achieve the following:

* Given a partition of `S` into `k` subsets, consider the sum of each subset
* Minimize the *largest* such sum over all subsets

"""
partition_min_largest!(model, α=0) = @objective(model, Min, last(model[:subset_sum]) - α*model[:sum_of_entropies])

"""
    partition_max_smallest!(model)

Adds an objective function to the model to achieve the following:

* Given a partition of `S` into `k` subsets, consider the sum of each subset
* Maximize the *smallest* such sum over all subsets

"""
partition_max_smallest!(model, α=0) = @objective(model, Max, first(model[:subset_sum]) - α*model[:sum_of_entropies])

"""
    partition_min_range!(model)

Adds an objective function to the model to achieve the following:

* Given a partition of `S` into `k` subsets, consider the sum of each subset
* Minimize the difference between the largest and smallest such sum over all subsets

"""
partition_min_range!(model, α=0) = @objective(model, Min, last(model[:subset_sum]) - first(model[:subset_sum]) + α*model[:sum_of_entropies])


end # module
