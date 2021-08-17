# MultiwayNumberPartitioning

A simple Julia package to optimally solve the [multiway number partitioning](https://en.wikipedia.org/wiki/Multiway_number_partitioning) problem
using a JuMP model with mixed-integer programming.

There is one main function `partition` which tries to accomplish the following task:
given a collection of numbers `S` and a number `k`, try to partition `S` into `k` subsets of roughly equal size.

For example:
```julia
julia> using MultiwayNumberPartitioning, HiGHS

julia> S =  [1, 1, 1, 3, 2, 1];

julia> inds = partition(S, 3; optimizer = HiGHS.Optimizer)
6-element Vector{Int64}:
 2
 2
 3
 1
 3
 2

julia> S[inds .== 1] # group 1
1-element Vector{Int64}:
 3

julia> S[inds .== 2] # group 2
3-element Vector{Int64}:
 1
 1
 1

julia> S[inds .== 3] # group 3
2-element Vector{Int64}:
 1
 2
```

We can see all three groups here have equal sum.

See the [example](./example/example.jl) for a more detailed usage example.


## Choice of objective function

In the case where the problem can not be perfectly solved (unlike the previous example), we choose an objective to optimize.

MultiwayNumberPartitioning.jl provides three objective functions:

* `partition_min_largest!`: minimize sum of the largest subset
* `partition_max_smallest!`: maximize the sum of the smallest subset
* `partition_min_range!`: minimize the difference between the sum of the largest subset and the smallest

(where here "largest" and "smallest" refer to the sums of the subsets).
