using Test
using Random
using CUDA
using InclusiveScans

@testset "InclusiveScans.jl" begin
    ATOL = eps(Float32)
    RTOL = sqrt(eps(Float32))

    N = 25000
    h_in = rand(Float32, N)
    d_in = CuArray(h_in)
    d_out = CUDA.zeros(Float32, N)

    InclusiveScans.largeArrayScanInclusive!(d_out, d_in, N)
    h_out = Array(d_out)

    # using CUDA.accumulate
    d_out_cuda = CUDA.accumulate(+, d_in)
    h_out_cuda = Array(d_out_cuda)

    # CPU cumsum check
    h_check = accumulate(+, h_in)

    @test isapprox(h_out, h_check, atol = ATOL, rtol = RTOL)
    @test isapprox(h_out_cuda, h_check, atol = ATOL, rtol = RTOL)
end
