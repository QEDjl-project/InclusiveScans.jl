
function generate_batch_CPU(rng::AbstractRNG,batch_size::Int,dist::ConstrainedGaussian1D)
    rand_args = rand(rng,Uniform(DomainSets.endpoints(dist.domain)...),batch_size)

    # opt this block out to be evaluated on a single element
    # and broadcast over everything
    # -------
    rand_vals = target_dist.(dist,rand_args)
    rel_rand_vals = normalize(dist,rand_vals)
    rand_probs = rand(rng,batch_size)
    mask = filter.(rel_rand_vals,rand_probs)
    # -------

    accepted_args = rand_args[mask] # this one with InclusiveScan.jl

    return accepted_args
end

function generate_CPU(rng::AbstractRNG,dist::ConstrainedGaussian1D{T},N::Int; batch_size::Int = 1000) where {T}
    accepted_args = T[]
    sizehint!(accepted_args,N+batch_size-1)

    nrun = 0
    while nrun <= N
       accepted_args_batch = generate_batch_CPU(rng,batch_size,dist)
       append!(accepted_args,accepted_args_batch)
       nrun += length(accepted_args_batch)
    end

    return accepted_args
end
