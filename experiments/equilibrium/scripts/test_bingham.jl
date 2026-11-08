using Random: Xoshiro
using Statistics: mean, var
using Test: @test, @testset, @test_throws

include("bingham.jl")
using .ExactBingham: setup_bingham, rand_bingham, log_acceptance

function simpson(f, a, b; intervals=16_384)
    iseven(intervals) || throw(ArgumentError("Simpson intervals must be even"))
    h = (b - a) / intervals
    total = f(a) + f(b)
    for i in 1:intervals-1
        total += (isodd(i) ? 4 : 2) * f(a + h * i)
    end
    return total * h / 3
end

function check_mean(samples, exact; z=6.0)
    return abs(mean(samples) - exact) <= z * sqrt(var(samples) / length(samples))
end

@testset "Exact Bingham sampler" begin
    @testset "Input checks and empty draws" begin
        @test_throws ArgumentError setup_bingham(Float64[])
        @test_throws ArgumentError setup_bingham([0.0, NaN])
        setup = setup_bingham(zeros(3))
        @test_throws ArgumentError rand_bingham(Xoshiro(1), setup, -1)
        samples, proposed = rand_bingham(Xoshiro(1), setup, 0)
        @test size(samples) == (3, 0)
        @test proposed == 0
    end

    @testset "Uniform sphere Dirichlet moments" begin
        n, nsamples = 8, 200_000
        setup = setup_bingham(zeros(n))
        samples, proposed = rand_bingham(Xoshiro(2026110601), setup, nsamples)
        @test proposed == nsamples
        @test maximum(abs.(sum(abs2, samples; dims=1) .- 1)) < 1e-14
        squares = samples .^ 2
        for i in 1:n
            @test check_mean(view(samples, i, :), 0.0)
            @test check_mean(view(squares, i, :), 1 / n)
            @test check_mean(view(squares, i, :) .^ 2, 3 / (n * (n + 2)))
        end
        @test check_mean(squares[1, :] .* squares[2, :], 1 / (n * (n + 2)))
    end

    @testset "N=2 independent quadrature reference" begin
        for gap in (0.1, 2.0, 20.0)
            # N=2: p(theta) proportional to exp(-gap*sin(theta)^2).
            weight(theta) = exp(-gap * sin(theta)^2)
            normalizer = simpson(weight, 0.0, 2pi)
            exact = simpson(theta -> cos(theta)^2 * weight(theta), 0.0, 2pi) / normalizer
            exact_fourth = simpson(theta -> cos(theta)^4 * weight(theta), 0.0, 2pi) / normalizer
            setup = setup_bingham([gap, 0.0])
            samples, proposed = rand_bingham(Xoshiro(2026110602), setup, 150_000)
            @test proposed >= size(samples, 2)
            @test check_mean(samples[1, :] .^ 2, exact)
            @test check_mean(samples[1, :] .^ 4, exact_fourth)
            @test check_mean(samples[1, :] .* samples[2, :], 0.0)
        end
    end

    @testset "Coalesced spectra, shifts, reproducibility, envelope" begin
        for values in ([3.0, 3.0, 0.0, -2.0], [10.0, 0.0, -10.0, -20.0], zeros(4))
            setup = setup_bingham(values)
            shifted = setup_bingham(values .+ 8.0)
            @test setup.penalties == shifted.penalties
            @test setup.inv_sqrt_omega == shifted.inv_sqrt_omega
            samples, proposed = rand_bingham(Xoshiro(88), setup, 10_000)
            shifted_samples, shifted_proposed = rand_bingham(Xoshiro(88), shifted, 10_000)
            @test samples == shifted_samples
            @test proposed == shifted_proposed
            @test maximum(abs.(sum(abs2, samples; dims=1) .- 1)) < 1e-14
            @test abs(sum(p -> inv(setup.b + 2p), setup.penalties) - 1) < 1e-13
            for t in range(0.0, maximum(setup.penalties); length=1001)
                direct = -t + (length(values) / 2) * log1p(2t / setup.b) - setup.log_bound
                @test direct <= 2e-13
                @test log_acceptance(setup, t) <= 0
                @test isapprox(log_acceptance(setup, t), direct; atol=2e-13)
            end
            if values[1] == values[2]
                @test check_mean(samples[1, :] .^ 2 - samples[2, :] .^ 2, 0.0)
            end
        end
    end

    @testset "Moderate N and strong condensation" begin
        n = 64
        values = vcat([20.0, 20.0], zeros(n-2))
        setup = setup_bingham(values)
        samples, proposed = rand_bingham(Xoshiro(123), setup, 20_000)
        @test all(isfinite, samples)
        @test maximum(abs.(sum(abs2, samples; dims=1) .- 1)) < 1e-14
        @test proposed >= size(samples, 2)
        @test check_mean(samples[1, :] .^ 2 - samples[2, :] .^ 2, 0.0)
        @test check_mean(samples[1, :], 0.0)
        # Independent 1D reference: s=y1^2+y2^2 is Beta(1,(N-2)/2)
        # under the uniform sphere, exponentially tilted by (N/2)*20*s.
        beta_b = (n - 2) / 2
        tilt = n * 20 / 2
        mode = 1 - (beta_b - 1) / tilt
        peak = tilt * mode + (beta_b - 1) * log1p(-mode)
        weight(s) = s == 1 ? 0.0 : exp(tilt * s + (beta_b - 1) * log1p(-s) - peak)
        normalizer = simpson(weight, 0.0, 1.0)
        exact_mass = simpson(s -> s * weight(s), 0.0, 1.0) / normalizer
        exact_mass2 = simpson(s -> s^2 * weight(s), 0.0, 1.0) / normalizer
        mass = samples[1, :] .^ 2 + samples[2, :] .^ 2
        @test check_mean(mass, exact_mass)
        @test check_mean(mass .^ 2, exact_mass2)
    end
end
