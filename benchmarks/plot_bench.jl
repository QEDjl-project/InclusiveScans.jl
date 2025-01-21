using BenchmarkTools
using BenchmarkPlots, StatsPlots
using StatsPlots.PlotMeasures

include("plot_utils.jl")

results = BenchmarkTools.load("bench.json")[1]

bench_plots = []
for bench in keys(results)
    @show results[bench]
    push!(bench_plots,plot(
        results[bench],
        title=bench,
        xlab = "problem size",
        ylab = "time taken (ns)",
        yscale = :log10,
        ylim = (1e3, 1e10),
        framestyle = :box,
        minorticks = 10,
        legend = false,
        size = (1600, 1000),
        margin = 14mm,
    ));
end

P = plot(bench_plots...,layout=(1,3));

savefig(P,"bench_results.png");