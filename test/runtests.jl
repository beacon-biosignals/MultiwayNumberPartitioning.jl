using MultiwayNumberPartitioning, HiGHS
using Test


@testset "MultiwayNumberPartitioning.jl" begin
    @testset "Example" begin
        include("../example/example.jl")
    end
end
