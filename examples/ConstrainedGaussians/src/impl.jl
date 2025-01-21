
# constrained Gaussian

struct ConstrainedGaussian1D{T<:Real,DIST,DOM<:AbstractInterval} <: AbstractTestDist{T}
    mu::T
    sig::T
    dist::DIST
    domain::DOM

    function ConstrainedGaussian1D(
        mu::T,
        sig::T,
        domain::DOM = Interval(-5 * sig, 5 * sig),
    ) where {T<:Real,DOM<:AbstractInterval}
        dist = Normal(mu, sig)
        return new{T,typeof(dist),DOM}(mu, sig, dist, domain)
    end
end

function ConstrainedGaussian1D(mu::T, sig::T, domain::D) where {T<:Real,D<:Tuple}
    return ConstrainedGaussian1D(mu, sig, Interval(domain...))
end

endpoints(dist::ConstrainedGaussian1D) = DomainSets.endpoints(dist.domain)

function norm(d::ConstrainedGaussian1D{T}) where {T}
    mu = d.mu
    dom = d.domain

    if mu in dom
        return target_dist(d, mu)
    end

    if mu <= minimum(dom)
        return target_dist(d, minimum(dom))
    end

    return target_dist(d, maximum(dom))
end
