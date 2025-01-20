using InclusiveScans
using CUDA

@testset "InclusiveScans.jl" begin
    function _cpu_inclusive_cumsum(x::Vector{Float32})
        s = 0.0f0
        out = similar(x)
        for i = 1:length(x)
            s += x[i]
            out[i] = s
        end
        return out
    end
    eps = 0.1
    N = 25000
    h_in = rand(Float32, N)
    d_in = CuArray(h_in)
    d_out = CUDA.zeros(Float32, N)

    InclusiveScans.largeArrayScanInclusive!(d_out, d_in, N)
    h_out = Array(d_out)

    # CPU cumsum check
    h_check = _cpu_inclusive_cumsum(h_in)

    maxdiff = maximum(abs.(h_out .- h_check))
    @test maxdiff < eps
end
