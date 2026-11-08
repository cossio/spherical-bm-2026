# Analytic solution

module RMT_Solution

wigner_density(λ::Real, c::Real) = (c / (2π)) * sqrt(max(0, 4/c - λ^2))

function wigner_cdf(λ::Real, c::Real)
    l = 2 / sqrt(c)
    if λ ≤ -l
        return 0.0
    elseif λ ≥ l
        return 1.0
    else
        return 0.5 + (λ * sqrt(l^2 - λ^2)) / (π * l^2) + asin(λ / l) / π
    end
end

function stieltjes(z::Real, c::Real)
    l = 2 / sqrt(c)
    @assert z ≥ l - 1e-10 * l "stieltjes: z=$z is well below edge l=$l"
    return c/2 * (z - sqrt(max(0, (z - l) * (z + l))))
end

function stieltjes_inverse(a::Real, c::Real)
    @assert a ≤ sqrt(c)
    return 1 / a + a / c
end

function log_potential(z::Real, c::Real)
    g = stieltjes(z, c)
    return 1/(2c) * g^2 - log(g)
end

function solve(a::AbstractVector, b::Real, c::Real)
    # u here means the vector of overlaps squared
    @assert issorted(a; rev=true)

    λ_paramagnetic = paramagnetic_λ_position.(a, c)
    g_paramagnetic = paramagnetic_g_position.(a, c)
    u_paramagnetic = max.(0, 1 .- c ./ a.^2)
    ξ_paramagnetic = a .* stieltjes_inverse.(min.(g_paramagnetic, a), c)

    o = count(a.^2 .> c)

    @assert all(isapprox.(g_paramagnetic, stieltjes.(λ_paramagnetic, c); rtol=1e-6))
    @assert (a[1] ≤ c ≤ a[1]^2 || c ≥ max(a[1]^2, 1)) == (g_paramagnetic[1] ≥ 1)

    if g_paramagnetic[1] ≥ 1 # confirms paramagnetic phase
        μ = stieltjes_inverse(1, c)
        M = 0.0
        return (; μ, M, g = g_paramagnetic, u = u_paramagnetic, ξ = ξ_paramagnetic, λ = λ_paramagnetic, o, d = 0, phase = :paramagnetic)
    else
        for d = 1:length(a)
            g1 = collided_g(view(a, 1:d), b, c)
            @assert g1 ≥ g_paramagnetic[d] # the chemical potential pushes eigenvalues towards the bulk
            if d < length(a) && g1 < min(1, g_paramagnetic[d + 1]) || d == length(a) && g1 < min(1, sqrt(c))
                μ = stieltjes_inverse(g1, c)
                u = [1 .- g1 ./ a[1:d]; u_paramagnetic[d+1:end]]
                g = [fill(g1, d); g_paramagnetic[d+1:end]]
                ξ = a .* stieltjes_inverse.(min.(g, a), c)
                λ = [fill(μ, d); λ_paramagnetic[d+1:end]]
                M = 1 - g1
                @assert M ≥ 0 # we already ruled out paramagnetic
                return (; μ, M, g, u, ξ, λ, o, d, phase = :ferromagnetic_outlier)
            end
        end
        # all eigenvalues must be at the edge
        λ = fill(2 / sqrt(c), length(a))
        g = fill(sqrt(c), length(a))
        u = max.(0, 1 .- g ./ a)
        ξ = a .* stieltjes_inverse.(min.(g, a), c)
        μ = 2 / sqrt(c)
        M = 1 - sqrt(c)
        @assert M ≥ 0 # we already ruled out paramagnetic
        return (; μ, M, g, u, ξ, λ, o, d=max(o, 1), phase = :ferromagnetic_sticky)
    end
end

function collided_g(a, b::Real, c::Real)
    d = length(a)
    t = b - sum(a)
    return (t + sqrt(t^2 + 4 * b * d * c)) / (2b)
end

function paramagnetic_λ_position(a::Real, c::Real)
    if a ≤ sqrt(c)
        return 2 / sqrt(c)
    else
        return 1/a + a / c
    end
end

paramagnetic_g_position(a::Real, c::Real) = c / max(a, sqrt(c))

end
