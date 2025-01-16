module ConstrainedGaussians


using Distributions
using DomainSets
using Random
using QuadGK
using StatsPlots
#using InclusiveScans

export ConstrainedGaussian1D
export target_dist, filter, norm, normalize
export endpoints
export generate_CPU
export plot_compare

include("interface.jl")
include("impl.jl")
include("generate.jl")
include("plot_compare.jl")

end
