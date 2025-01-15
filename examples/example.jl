
# some imports

using BenchmarkTools
using Random
using Plots
RNG = MersenneTwister(1234)

# import the example
using ConstrainedGaussians

# setup

mu = rand(RNG)                  # central value
sig = rand(RNG)                 # variance
dom = (mu+0.5*sig,mu+3.0*sig)   # support/domain

## example distribution to be sampled
d = ConstrainedGaussian1D(mu,sig,dom)

## number of samples to be generated
N = Int(1e6)

## generation of samples
samples = generate_CPU(RNG,d,N ; batch_size=1000)

## plotting: histogram vs target_dist
P = plot_compare(samples,d)

## save plot
savefig(P,"example_compare.pdf")
