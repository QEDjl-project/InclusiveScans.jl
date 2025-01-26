using Test
using Random
using CUDA
using InclusiveScans

@testset "InclusiveScans.jl Tests" begin
    sizes = (1, 100, 1000, 25_000, 100_000)

    for N in sizes
        @testset "Test with N = $N" begin
            rng = Xoshiro(137)  # Fixed seed

            # Generate random input
            bit_input = BitVector(rand(rng, Bool, N))
            float_input = Float32.(bit_input)

            d_in = CuArray(float_input)
            d_out = CUDA.zeros(Float32, N)

            # Run inclusive scan on GPU
            InclusiveScans.largeArrayScanInclusive!(d_out, d_in, Int32(N))

            h_out = Array(d_out)
            h_check = Base.accumulate(+, float_input)

            # Check if GPU result matches the CPU reference
            @test isapprox(h_out, h_check; atol=eps(Float32))
        end
    end
end
