using MultiwayNumberPartitioning, HiGHS, JuMP
using DataFrames, UUIDs, StableRNGs, Chain

# In this example, we have a collection of individual animals, identified by UUID, each with a species.
# We wish to obtain partition our animals into 4 groups,
n_partitions = 4
# such that:
# * the groups are of roughly equal size
# * all animals of the same species are in the same group

# First, let's generate our data:
function make_table(n_animals, label_set)
    rng = StableRNG(324)
    possible_species = ["Aardvark", "Albatross", "Alligator", "Alpaca", "Anole", "Ant",
                        "Anteater"]
    return DataFrame(:species => [rand(rng, possible_species) for i in 1:n_animals],
                     :id => [uuid4(rng) for i in 1:n_animals],
                     :label => [rand(rng, label_set) for i in 1:n_animals])
end

n_animals = 100
label_set = 1:5
df = make_table(n_animals, label_set)

function count_labels(v, label_set)
    d = Dict{eltype(label_set),Int}()
    sizehint!(d, length(label_set))
    for l in label_set
        d[l] = 0
    end
    for elt in v
        d[elt] += 1
    end
    return [d[l] for l in label_set]
end

# Now, we will group them by species and count how many individuals we have:
count_by_species = combine(groupby(df, :species), nrow => :n_individuals,
                           :label => (v -> Ref(count_labels(v, label_set))) => :labels)

group_labels = reduce(vcat, permutedims.(count_by_species.labels))
# Now, we will partition them.
# First, we choose our optimizer. We will use the new MIT licensed HiGHS solver (https://github.com/ERGO-Code/HiGHS)
optimizer = optimizer_with_attributes(HiGHS.Optimizer, MOI.Silent() => true) # quiet, please!

# Now, perform the partitioning:
transform!(count_by_species,
           :n_individuals => (v -> partition(partition_min_range!, v, n_partitions; optimizer, group_labels)) => :partition)

# Already here we can look at `count_by_species` to see the results. However, we can also
# `leftjoin!` these partitions back onto our original DataFrame so that we can see which individual is in which partition:
partitioned_df = leftjoin(df, count_by_species; on=:species)

#####
##### Comparing objective functions
#####

# We can also explore trying different objective functions. Let's start again with our `count_by_species` DataFrame:
count_by_species = combine(groupby(df, :species), nrow => :n_individuals)

# Now, let us try all three algorithms supported by the package:
for alg in (partition_min_largest!, partition_max_smallest!, partition_min_range!)
    alg_name = repr(alg; context=:compact => true)
    transform!(count_by_species,
               :n_individuals => (v -> partition(alg, v, n_partitions; optimizer)) => alg_name)
end
# Let's also generate two random partitions to compare to:
count_by_species.partition_random_1 = rand(StableRNG(431), 1:n_partitions,
                                           nrow(count_by_species))
count_by_species.partition_random_2 = rand(StableRNG(891), 1:n_partitions,
                                           nrow(count_by_species))

# We can summarize the results with a little DataFrames manipulation.
# See <https://bkamins.github.io/julialang/2021/05/28/pivot.html> for more on `stack` and `unstack`.
summary_df = @chain count_by_species begin
    stack(r"partition", :n_individuals; variable_name=:algorithm, value_name=:group)
    groupby([:algorithm, :group])
    combine(:n_individuals => sum => :group_size)
    unstack(:group, :algorithm, :group_size)
end

# We can see the random partitions looks much worse in terms of having groups of equal size; e.g. the first
# has a group of size 12 and one of size 47, while the other partitions have groups of closer sizes.

# We can do some simple checks:
@testset "Partitions add up to the whole" begin
    for col in names(summary_df, r"partition")
        @test sum(summary_df[:, col]) == n_animals
    end
end

@testset "Partitions are optimal (rough check)" begin
    for col in names(summary_df, r"partition")
        # "max smallest" really holds, compared to the other algorithms and to the random partition:
        @test minimum(summary_df[:, :partition_max_smallest!]) >=
              minimum(summary_df[:, col])

        # "min largest" really holds, compared to the other algorithms and to the random partition:
        @test maximum(summary_df[:, :partition_min_largest!]) <= maximum(summary_df[:, col])

        # "min range" really holds:
        r = v -> maximum(v) - minimum(v)
        @test r(summary_df[:, :partition_min_range!]) <= r(summary_df[:, col])
    end
end
