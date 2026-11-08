# FFT-based stationary K=1 DMFT solver. Picard iteration on (R, Q, μ, s)
# with FFTW for the causal convolutions and cross-correlations.

module StationaryFFT

using FFTW

export solve_stationary_fft

@inline φ₁(z) = abs(z) < 1e-4 ?
    @evalpoly(z, 1.0, 0.5, 1/6, 1/24, 1/120) : expm1(z) / z

# Trapezoidal weight for index j over an inclusive range [lo, hi]:
#   tw(j, lo, hi) = 0   if lo > hi
#                 = 0.5 if j == lo or j == hi
#                 = 1.0 otherwise
@inline tw(j, lo, hi) = lo >= hi ? 0.0 : (j == lo || j == hi) ? 0.5 : 1.0

"""
    Buffers(N, M)

Pre-allocated FFT scratch space for problem size N (real signal length) and
padded transform length M = nextpow(2, 2N).
"""
struct Buffers
    N::Int
    M::Int
    fbuf::Vector{Float64}   # real input buffer (length M)
    gbuf::Vector{Float64}
    Fhat::Vector{ComplexF64}
    Ghat::Vector{ComplexF64}
    out::Vector{Float64}    # real output buffer (length M)
    P::FFTW.rFFTWPlan{Float64,-1,false,1,Tuple{Int}}
    Pi::AbstractFFTs.ScaledPlan
end

function Buffers(N::Int)
    M = nextpow(2, 2N)
    fbuf = zeros(Float64, M)
    gbuf = zeros(Float64, M)
    Fhat = zeros(ComplexF64, M ÷ 2 + 1)
    Ghat = zeros(ComplexF64, M ÷ 2 + 1)
    out  = zeros(Float64, M)
    P  = plan_rfft(fbuf)
    Pi = plan_irfft(Fhat, M)
    return Buffers(N, M, fbuf, gbuf, Fhat, Ghat, out, P, Pi)
end

"""
    causal_conv!(C, f, g, B; correct_endpoints=true) -> C

Compute the causal convolution

    C[k] = Σ_{j=1}^{k-1} w_j(k) f[j] g[k-j],   k = 1..N

with trapezoidal weights w_1(k) = w_{k-1}(k) = 0.5, interior w_j = 1.0.
For k = 1, C[1] = 0; for k = 2, the inner sum is a single 0.5·f[1] g[1].

Implementation: linear convolution of f with a left-shifted g (g'[1] = 0,
g'[m] = g[m-1] for m ≥ 2) gives the unweighted sum
    C_full[k] = Σ_{j=1}^{k-1} f[j] g[k-j].
Endpoint correction subtracts 0.5·(f[1] g[k-1] + f[k-1] g[1]) for k ≥ 3
to drop the boundary pair to weight 0.5; for k = 2 we keep the single
0.5·f[1] g[1] term.
"""
function causal_conv!(C::Vector{Float64}, f::Vector{Float64}, g::Vector{Float64},
                      B::Buffers)
    N = B.N
    @assert length(C) == N
    @assert length(f) == N
    @assert length(g) == N
    M = B.M
    # zero pad
    fill!(B.fbuf, 0.0); fill!(B.gbuf, 0.0)
    @inbounds for i in 1:N
        B.fbuf[i] = f[i]
        # g'[1] = 0, g'[m] = g[m-1] for m = 2..N+1
        B.gbuf[i + 1] = g[i]
    end
    # FFT
    mul!(B.Fhat, B.P, B.fbuf)
    mul!(B.Ghat, B.P, B.gbuf)
    @inbounds for i in eachindex(B.Fhat)
        B.Fhat[i] = B.Fhat[i] * B.Ghat[i]
    end
    mul!(B.out, B.Pi, B.Fhat)
    # extract first N samples; out[k] = Σ_{j=1}^{k-1} f[j] g[k-j]
    @inbounds for k in 1:N
        C[k] = B.out[k]
    end
    # Trapezoidal endpoint correction: degenerate range for k ≤ 2 gives 0;
    # otherwise downweight the two endpoints (j = 1 and j = k−1) to 0.5.
    @inbounds begin
        C[1] = 0.0
        if N >= 2
            C[2] = 0.0
        end
        for k in 3:N
            C[k] -= 0.5 * (f[1] * g[k-1] + f[k-1] * g[1])
        end
    end
    return C
end

"""
    cross_corr!(C, f, g, B) -> C

Cross-correlation

    C[k] = Σ_{j=1}^{N-k+1} w_j(k) f[j+k-1] g[j],   k = 1..N

with trapezoidal weights w_1(k) = w_{N-k+1}(k) = 0.5 (endpoints), interior
1.0. For k = N, the sum has length 1 and is degenerate (set to 0).

Implementation via correlation theorem: rfft(f) ⋅ conj(rfft(g)) → ifft
gives `Σ_j f[j+k-1] g[j]` as a function of the lag k-1 = 0..N-1, modulo
the cyclic wrap (avoided by zero-padding to M ≥ 2N).
"""
function cross_corr!(C::Vector{Float64}, f::Vector{Float64}, g::Vector{Float64},
                     B::Buffers)
    N = B.N
    @assert length(C) == N
    M = B.M
    fill!(B.fbuf, 0.0); fill!(B.gbuf, 0.0)
    @inbounds for i in 1:N
        B.fbuf[i] = f[i]
        B.gbuf[i] = g[i]
    end
    mul!(B.Fhat, B.P, B.fbuf)
    mul!(B.Ghat, B.P, B.gbuf)
    # cross-corr at lag ℓ: F̂ · conj(Ĝ) → ifft → out[ℓ+1] = Σ_j f[j+ℓ] g[j]
    @inbounds for i in eachindex(B.Fhat)
        B.Fhat[i] = B.Fhat[i] * conj(B.Ghat[i])
    end
    mul!(B.out, B.Pi, B.Fhat)
    @inbounds for k in 1:N
        # k corresponds to lag ℓ = k - 1
        C[k] = B.out[k]
    end
    # Trapezoidal endpoint correction with Jm = N − k + 1; if Jm ≤ 1
    # (i.e. k = N) the sum is degenerate and set to 0.
    @inbounds for k in 1:N
        Jm = N - k + 1
        if Jm == 1
            # k = N: degenerate (single point, lo >= hi) → 0
            C[k] = 0.0
        elseif Jm >= 2
            # subtract 0.5·(boundary contributions)
            # at j=1: f[k]·g[1]; at j=Jm: f[N]·g[Jm]
            C[k] -= 0.5 * (f[k] * g[1] + f[N] * g[Jm])
        end
    end
    return C
end

# In-place mul! shim for FFTW plans (rfft / irfft)
import LinearAlgebra: mul!

"""
    solve_stationary_fft(γ, η, β, ν; N_τ, Δτ, s_scan, maxiter, tol)
        -> (; s_inf, μ_inf, F0)

FFT-accelerated stationary K=1 DMFT solver.

Keyword arguments: `N_τ` (number of grid points), `Δτ` (grid spacing),
`s_scan` (scan of order parameters for the F(s) sign-change),
`maxiter` (Picard cap), `tol` (relative residual tol).
"""
function solve_stationary_fft(γ, η, β, ν;
        N_τ::Int = 600,
        Δτ::Union{Nothing,Float64} = nothing,
        maxiter::Int = 1000,
        tol::Float64 = 1e-12,
        s_scan = range(0.0, 0.999, length=80))

    Δτv = Δτ === nothing ? clamp(sqrt(η * γ) / (5 * max(ν, 0.01)), 0.005, 0.2) :
                            Δτ
    c_Q = -ν / 2
    c_R = ν^2 / (η * γ)
    EXP = [exp(-γ * k * Δτv / 2) for k in 0:2N_τ]

    B = Buffers(N_τ)
    # scratch
    Σ        = Vector{Float64}(undef, N_τ)
    Dreg     = Vector{Float64}(undef, N_τ)  # D_st^reg[k] = c_R · EXP[k] · Q[k]
    convΣR   = Vector{Float64}(undef, N_τ)
    convΣQ   = Vector{Float64}(undef, N_τ)
    ccΣQ     = Vector{Float64}(undef, N_τ)  # cross-corr (M_st ⋆ Q)
    ccDR     = Vector{Float64}(undef, N_τ)  # cross-corr (D_reg ⋆ R)
    Qn       = Vector{Float64}(undef, N_τ)
    Rn       = Vector{Float64}(undef, N_τ)

    function solve_bath(s; Q0=nothing, R0=nothing, μ0=NaN)
        Q = Q0 !== nothing ? copy(Q0) :
            [s^2 + (1 - s^2) * EXP[k] for k in 1:N_τ]
        R = R0 !== nothing ? copy(R0) :
            Float64[EXP[k] for k in 1:N_τ]
        μ = isnan(μ0) ? ν / β + (ν / γ) * s^2 + 1.0 : μ0
        src = (ν / γ) * s^2

        for it in 1:maxiter
            # Σ[k] = M_st(τ_k) on grid
            @inbounds for k in 1:N_τ
                Σ[k] = EXP[k] * (c_Q * Q[k] + c_R * R[k])
                Dreg[k] = c_R * EXP[k] * Q[k]
            end

            # μ update: integrals of M_st(σ) Q_st(σ) and D_reg(σ) R_st(σ)
            iμ = 0.0
            @inbounds for k in 1:N_τ
                w = tw(k, 1, N_τ)
                iμ += w * (Σ[k] * Q[k] + Dreg[k] * R[k])
            end
            μn = src + ν / β + Δτv * iμ
            eμ = exp(-μn * Δτv); pμ = φ₁(-μn * Δτv)

            # ── Inner R solve via fixed-point iteration on (Σ *_c R):
            # Σ is frozen for the outer iter; iterate Rn until self-consistent
            # with the convolution. Each pass = 1 FFT pair → cheap.
            # Initialize Rn from R (previous outer)
            Rn .= R
            for _inner in 1:50
                causal_conv!(convΣR, Σ, Rn, B)
                # Build new Rn forward
                Rn[1] = 1.0
                rerr = 0.0
                @inbounds for k in 2:N_τ
                    new_Rn_k = eμ * Rn[k-1] + pμ * Δτv * (Δτv * convΣR[k])
                    rerr = max(rerr, abs(new_Rn_k - Rn[k]))
                    Rn[k] = new_Rn_k
                end
                rerr < tol * 10 && break
            end

            # ── Cross-correlations using updated Rn (D_reg ⋆ Rn, Σ ⋆ Q)
            cross_corr!(ccDR, Dreg, Rn, B)
            cross_corr!(ccΣQ, Σ, Q, B)

            # ── Inner Q solve via fixed-point iteration on (Σ *_c Q):
            # Inc[k] = Δτ·(ccΣQ[k] + ccDR[k]); the Q step uses Inc[k-1].
            Qn .= Q
            for _inner in 1:50
                causal_conv!(convΣQ, Σ, Qn, B)
                Qn[1] = 1.0
                qerr = 0.0
                @inbounds for k in 2:N_τ
                    Inc_km1 = Δτv * (ccΣQ[k-1] + ccDR[k-1])
                    new_Qn_k = eμ * Qn[k-1] + pμ * Δτv * (src + Δτv * convΣQ[k] + Inc_km1)
                    qerr = max(qerr, abs(new_Qn_k - Qn[k]))
                    Qn[k] = new_Qn_k
                end
                qerr < tol * 10 && break
            end

            # ── Convergence check
            err = abs(μn - μ)
            @inbounds for k in 1:N_τ
                err = max(err, abs(Qn[k] - Q[k]), abs(Rn[k] - R[k]))
            end
            Q .= Qn; R .= Rn; μ = μn
            err < tol && break
        end
        # F(s) closure
        iΣ = 0.0
        @inbounds for k in 1:N_τ
            iΣ += tw(k, 1, N_τ) * EXP[k] * (c_Q * Q[k] + c_R * R[k])
        end
        return (; F = ν / γ - μ + Δτv * iΣ, Q, R, μ)
    end

    # Scan F(s) and find equilibrium. Use COLD starts (no warm start across
    # s_scan) so that each F(s) sample reflects the fixed point selected at
    # that s independently — warm-starting from a previous s can collapse
    # genuinely distinct stable branches of F(s) into a single visible
    # sign-change, masking the multi-root structure that exists at
    # γ ∈ {0.7, 0.8} small ν.
    F_vals = zeros(length(s_scan))
    Qw = nothing; Rw = nothing; μw = NaN
    for (i, s) in enumerate(s_scan)
        r = solve_bath(s)   # cold start
        F_vals[i] = r.F
        # Keep last warm start for downstream bisections (largest s_scan).
        Qw = copy(r.Q); Rw = copy(r.R); μw = r.μ
    end

    # Multi-root selection: scan ALL stable sign changes (F: + → −), bisect
    # each, and return the LARGEST root (deepest condensation = global
    # thermodynamic minimum among FM solutions). This fixes the multi-root
    # bug at γ ∈ {0.7, 0.8} small ν where multiple FM stable solutions exist.
    s_inf = 0.0
    μ_inf = solve_bath(0.0).μ
    n_stable_roots = 0
    roots_found = Float64[]
    for i in 1:length(s_scan)-1
        if F_vals[i] > 0 && F_vals[i+1] < 0
            n_stable_roots += 1
            slo, shi = Float64(s_scan[i]), Float64(s_scan[i+1])
            # Use local Q/R/μ scratch for this bisection so we don't poison
            # subsequent root searches with warm starts at the wrong s.
            Qb = Qw === nothing ? nothing : copy(Qw)
            Rb = Rw === nothing ? nothing : copy(Rw)
            μb = μw
            for _ in 1:60
                sm = (slo + shi) / 2
                rm = solve_bath(sm; Q0=Qb, R0=Rb, μ0=μb)
                Qb = copy(rm.Q); Rb = copy(rm.R); μb = rm.μ
                rm.F > 0 ? (slo = sm) : (shi = sm)
                (shi - slo) < 1e-12 && break
            end
            s_root = (slo + shi) / 2
            push!(roots_found, s_root)
            # Update s_inf/μ_inf to the largest root found so far.
            if s_root > s_inf
                s_inf = s_root
                μ_inf = solve_bath(s_root; Q0=Qb, R0=Rb, μ0=μb).μ
            end
        end
    end
    if n_stable_roots > 1
        println("  [stationary_fft] ν=", round(ν; sigdigits=4),
                " γ=", γ, ": stable roots found: ", n_stable_roots,
                " at s ≈ ", round.(roots_found; digits=4),
                " → picked largest = ", round(s_inf; digits=4))
    end

    return (; s_inf, μ_inf, F0=F_vals[1])
end

end # module
