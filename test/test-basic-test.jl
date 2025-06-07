using Test
using Random
using CUDA
using InclusiveScans

RNG = Xoshiro(137)  # Fixed seed
SIZES = (1, 100, 1000, 25_000, 100_000, 2^24)
TYPES = (Float16, Float32, Float64, Int32, Int64, ComplexF32, ComplexF64)

_custom_eps(::Type{T}) where {T <: AbstractFloat} = sqrt(eps(T))
_custom_eps(::Type{T}) where {T <: Integer} = zero(T)
_custom_eps(::Type{Complex{T}}) where {T <: AbstractFloat} = sqrt(eps(T))

@testset "InclusiveScans.jl Tests" begin
    @testset "Test with T = $T" for T in TYPES
        @testset "Test with N = $N" for N in SIZES
            if (T == Float16 && N >= 10_000)
                # precision of F16 is too little for large tests
                continue
            end

            # Generate random input
            input = rand(RNG, T, N)

            d_in = CuArray(input)
            d_out = similar(d_in)

            # Run inclusive scan on GPU
            InclusiveScans.largeArrayScanInclusive!(d_out, d_in, Int32(N))

            h_out = Array(d_out)
            h_check = Base.accumulate(+, input)

            # Check if GPU result matches the CPU reference
            @test isapprox(h_out, h_check, rtol = _custom_eps(T))

            # Run exclusive scan on GPU
            InclusiveScans.largeArrayScanExclusive!(d_out, d_in, Int32(N))

            h_out = Array(d_out)
            h_check .-= input

            @test isapprox(h_out, h_check, rtol = _custom_eps(T))
        end
    end
end
