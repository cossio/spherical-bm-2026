# Analytic RMT solution in physical parameters (γ, η, c1, c2).
#
# Maps to internal RMT parameters via:
#   a₁ = η·c1,  a₂ = η·c2,  b = K·η,  c = γ·η
# where K = c1 + c2 (= 2 for our model).
#
# Wraps RMT_Solution.solve and re-expresses all results in physical units.

module RMT_Solution_v2

import ..RMT_Solution

"""
    solve(c, γ, η) -> NamedTuple
    solve(c1, c2, γ, η) -> NamedTuple

Solve the RMT saddle-point equations in physical parameters.

# Arguments
- `c`: vector of signal strengths [c₁, c₂, …, c_K] (sorted descending, cₖ > 0)
  or two positional arguments `c1, c2` for the K=2 case.
- `γ`: inverse temperature ratio
- `η`: signal-to-noise ratio

Mapping to internal RMT parameters: aₖ = η·cₖ,  b = K·η,  c = γ·η.

# Returns
Named tuple with fields:
- `μ, M, g, u, ξ, λ, o, d, phase`: from the RMT saddle-point solution
- `edge`: semicircle edge = 2/√(γη)
- `γ, η, c`: input parameters
"""
function solve(c::AbstractVector{<:Real}, γ::Real, η::Real)
    @assert issorted(c; rev=true) "c must be sorted in descending order"
    @assert all(>(0), c)
    @assert γ > 0 && η > 0

    a = η .* c
    K = sum(c)
    b = K * η
    c_rmt = γ * η

    sol = RMT_Solution.solve(collect(a), b, c_rmt)
    edge = 2 / sqrt(c_rmt)

    return (; sol.μ, sol.M, sol.g, sol.u, sol.ξ, sol.λ, sol.o, sol.d, sol.phase,
              edge, γ, η, c)
end

# Convenience method for K=2
solve(c1::Real, c2::Real, γ::Real, η::Real) = solve([c1, c2], γ, η)

end
