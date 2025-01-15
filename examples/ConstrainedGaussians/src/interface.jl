

# general abstract test-distribution

abstract type AbstractTestDist{T} end

Base.broadcastable(dist::AbstractTestDist) = Ref(dist)
target_dist(d::AbstractTestDist{T}, x) where {T} = x in d.domain ? pdf(d.dist, x) : zero(T)

function normalize(d, x)
    return x / norm(d)
end

function filter(rel_rand_val, rand_prob)
    return rel_rand_val >= rand_prob
end
