module MultiwayNumberPartitioning

using JuMP, LinearAlgebra

export partition, partition_min_largest!, partition_max_smallest!, partition_min_range!

"""
    partition([objective! = partition_min_range!], S::AbstractVector, k::Int; optimizer, strict::Bool=true) -> Vector{Int}

Find a partition of `S` into `k` subsets such the sums of the subsets are "as near as possible".

Arguments:
* (optional): an objective, such as [`partition_min_largest!`](@ref), [`partition_max_smallest!`](@ref), or [`partition_min_range!`](@ref). Defaults to `partition_min_range!`.
* `S` a collection of values to partition,
* `k` a number of subsets to partition into

Keyword arguments:
* `optimizer` (required): a JuMP-compatible mixed-integer optimizer, such as GLPK, HiGHS, Cbc, Gurobi, etc.
* `strict` (optional): if `true`, an error will be thrown if the model was not solved optimally. If `false`, only a warning will be issued instead. Defaults to `true`.

Returns a vector of indices `v` such that `v[i] == j` if `S[i]` is assigned to subset `j`. 
"""
partition(S, k; optimizer, strict::Bool=true) = partition(partition_min_range!, S, k; optimizer, strict)

function partition(objective!, S::AbstractVector, k::Int; optimizer, strict::Bool=true)
    model = Model(optimizer)
    element = populate_model!(model, S, k)
    objective!(model)
    JuMP.optimize!(model)

    # Check that we've solved it optimally
    if !(termination_status(model) == MOI.OPTIMAL && primal_status(model) == MOI.FEASIBLE_POINT)
        msg = "Problem was not solved optimally or solution was not feasible. Termination status: $(termination_status(model)). Primal status: $(primal_status(model))."
        if strict
            error(mg)
        else
            @warn msg
        end
    end
    
    # Now we've got a sort of one-hot encoded partitioning:
    onehot_partition = value.(element)

    # We wish to convert it to a vector of indexes so that the `i`th element of the
    # vector corresponds to the subset `j` that element `i` belongs to:
    partition = dropdims(mapslices(row -> only(findall(>(0.5), row)), onehot_partition; dims=2); dims=2)

    return partition
end

# Populates an empty `model` with variables and objectives
function populate_model!(model, S, k)
    # does element i belong to set j
    @variable(model, element[i=1:length(S), j=1:k], binary=true)

    # Element `i` belongs to exactly one set j
    @constraint(model, [i=1:length(S)], sum(element[i, :]) == 1)

    # The sum of the `j`th subset can be expressed as a dot product between the `j`th column of `element` and `S`.
    # Why? Because the `j`th column of `element` is a boolean vector expressing which elements are in set `j`.
    # If element `i` is in set `j`, then `element[i,j]==1` and we add `element[i,j]*S[i] = S[i]` to our total.
    # If element `i` is *not* in set `j`, then `element[i,j]=0` and we add nothing to our total.
    # Thus, the total is the sum of subset `j`.
    @expression(model, subset_sum[j=1:k], dot(element[:, j], S))

    # We can remove some non-uniqueness by ordering our subset sums.
    # This is also convenient for getting the smallest and largest elements.
    @constraint(model, [j=1:(k-1)], subset_sum[j] <= subset_sum[j+1])

    return element
end

"""
    partition_min_largest!(model)

Adds an objective function to the model to achieve the following:

* Given a partition of `S` into `k` subsets, consider the sum of each subset
* Minimize the *largest* such sum over all subsets

"""
partition_min_largest!(model) = @objective(model, Min, last(model[:subset_sum]))

"""
    partition_max_smallest!(model)

Adds an objective function to the model to achieve the following:

* Given a partition of `S` into `k` subsets, consider the sum of each subset
* Maximize the *smallest* such sum over all subsets

"""
partition_max_smallest!(model) = @objective(model, Max, first(model[:subset_sum]))

"""
    partition_min_range!(model)

Adds an objective function to the model to achieve the following:

* Given a partition of `S` into `k` subsets, consider the sum of each subset
* Minimize the difference between the largest and smallest such sum over all subsets

"""
partition_min_range!(model) = @objective(model, Min, last(model[:subset_sum]) - first(model[:subset_sum]))


end # module
