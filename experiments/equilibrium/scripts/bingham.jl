module ExactBingham

using LinearAlgebra: norm
using Random: AbstractRNG, randn!

export BinghamSetup, setup_bingham, rand_bingham, log_acceptance

"""Angular-central-Gaussian rejection envelope for a diagonal Bingham law."""
struct BinghamSetup
    penalties::Vector{Float64}
    inv_sqrt_omega::Vector{Float64}
    b::Float64
    log_bound::Float64
end

"""
    setup_bingham(eigenvalues) -> BinghamSetup

Prepare exact rejection sampling of unit vectors `y` with density proportional
to `exp((N/2) * sum(eigenvalues .* y.^2))`, where `N=length(eigenvalues)`.
The input is the spectrum of the physical coupling matrix W, in any order.
Physical spins on the radius-sqrt(N) sphere are `x = sqrt(N) * V * y`, where
the columns of V are W's eigenvectors in the same order.

This is the angular-central-Gaussian envelope of Kent, Ganeiber and Mardia,
https://arxiv.org/abs/1310.8110. No saddle approximation or numerical partition
function enters the sampled density. The root below optimizes acceptance;
the envelope inequality holds for every `0 < b <= N`.
"""
function setup_bingham(eigenvalues::AbstractVector{<:Real})
    n = length(eigenvalues)
    n > 0 || throw(ArgumentError("the spectrum must be nonempty"))
    all(isfinite, eigenvalues) || throw(ArgumentError("spectrum must be finite"))
    values = Float64.(eigenvalues)
    penalties = (n / 2) .* (maximum(values) .- values)
    all(isfinite, penalties) || throw(ArgumentError("scaled spectral gaps overflow"))

    # At least one penalty is zero, so the unique root of
    # sum(1 / (b + 2*penalties[i])) = 1 lies in [1,N].
    lo, hi = 1.0, Float64(n)
    for _ in 1:64
        mid = (lo + hi) / 2
        if sum(p -> inv(mid + 2p), penalties) > 1
            lo = mid
        else
            hi = mid
        end
    end
    b = (lo + hi) / 2
    inv_sqrt_omega = @. inv(sqrt(1 + 2 * penalties / b))
    log_bound = (b - n) / 2 + (n / 2) * log(n / b)
    return BinghamSetup(penalties, inv_sqrt_omega, b, log_bound)
end

"""
    log_acceptance(setup, quadratic_penalty)

Log rejection probability for a unit-sphere ACG proposal with
`quadratic_penalty = sum(setup.penalties .* y.^2)`. The identity
`log(1+v) <= v` proves the envelope globally, including coalesced spectra.
"""
function log_acceptance(setup::BinghamSetup, quadratic_penalty::Real)
    n = length(setup.penalties)
    v = (setup.b + 2quadratic_penalty - n) / n
    # Algebraically equivalent to -t + (N/2)log(1+2t/b) - log_bound,
    # but accurate near the maximum of the acceptance ratio.
    return min(0.0, (n / 2) * (log1p(v) - v))
end

"""
    rand_bingham(rng, setup, nsamples) -> (samples, proposed)

Return independent exact Bingham draws as columns of an `N x nsamples`
matrix, in W's eigenbasis, and the total number of ACG proposals attempted.
All randomness uses the supplied RNG. Returned vectors have unit norm.
"""
function rand_bingham(rng::AbstractRNG, setup::BinghamSetup, nsamples::Integer)
    nsamples >= 0 || throw(ArgumentError("nsamples must be nonnegative"))
    n = length(setup.penalties)
    samples = Matrix{Float64}(undef, n, nsamples)
    candidate = Vector{Float64}(undef, n)
    accepted = 0
    proposed = 0
    while accepted < nsamples
        proposed += 1
        randn!(rng, candidate)
        candidate .*= setup.inv_sqrt_omega
        candidate ./= norm(candidate)
        penalty = 0.0
        @inbounds @simd for i in eachindex(candidate)
            penalty += setup.penalties[i] * candidate[i]^2
        end
        if log(rand(rng)) <= log_acceptance(setup, penalty)
            accepted += 1
            @views samples[:, accepted] .= candidate
        end
    end
    return samples, proposed
end

end
