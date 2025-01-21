using CUDA
using BenchmarkTools
using InclusiveScans

CUDA.versioninfo()
CUDA.device()

const SUITE = BenchmarkGroup()

SUITE["InclusiveScans.jl"] = BenchmarkGroup()
SUITE["Base.accumulate!"] = BenchmarkGroup()
SUITE["CUDA.accumulate!"] = BenchmarkGroup()


PROBLEM_SIZES = Int.(2 .^ (10:10:30))
_build_label(i::Int) = "2^$(Int(log2(i)))"
@show PROBLEM_SIZES
@show _build_label.(PROBLEM_SIZES)

for PROBLEM_SIZE in PROBLEM_SIZES
    h_in = convert.(Float32, rand(Bool,PROBLEM_SIZE))
    
    d_in_ic = CuArray(h_in)
    d_out_ic = CUDA.zeros(Float32, PROBLEM_SIZE)
    SUITE["Base.accumulate!"][_build_label(PROBLEM_SIZE)] =
        @benchmarkable CUDA.@sync InclusiveScans.largeArrayScanInclusive!(
            $d_out_ic,
            $d_in_ic,
            $PROBLEM_SIZE,
        )

    h_out = zeros(PROBLEM_SIZE)
    SUITE["CUDA.accumulate!"][_build_label(PROBLEM_SIZE)] =
        @benchmarkable Base.accumulate!(+, $h_in, $h_out)

    d_in_cu= CuArray(h_in)
    d_out_cu = CUDA.zeros(Float32, PROBLEM_SIZE)
    SUITE["InclusiveScans.jl"][_build_label(PROBLEM_SIZE)]=
        @benchmarkable CUDA.@sync CUDA.accumulate!(+, $d_in_cu, $d_out_cu)

end

tune!(SUITE)
result = run(SUITE; verbose = true)

BenchmarkTools.save("bench.json", result)