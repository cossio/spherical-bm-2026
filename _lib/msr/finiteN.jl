# Finite-N simulation of the arbitrary-K uSBM training dynamics. Couples
# Ẇ = ½C − K/(2N) xx^T − γ/2 W + Ω/√(ηN) and ẋ = ν W x − κ x + √(2ν/β) ξ
# (with x projected onto |x|² = N), via exact-OU on W and Heun on x. Uses
# c_a = √N · e_a so C = diag(c_1, …, c_K, 0, …, 0), s_a = x_a/√N.
#
# Usage: julia finiteN.jl [--gamma --eta --beta --nu --c=c1,...,cK
#                          --N --Tmax=auto --s0=0.05 --nsave=300
#                          --seed=42 --neigen=0 --outfile=auto]
# `--neigen>0` diagonalises W at save times (O(N³) per save).

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "_julia_env"))

using LinearAlgebra, Random, JLD2
BLAS.set_num_threads(1)

# ── Parse command-line arguments ─────────────────────────────────────
function parse_args(args)
    scalars = Dict(
        "gamma" => 0.5,
        "eta"   => 3.0,
        "beta"  => 1.0,
        "nu"    => 1.0,
        "N"     => 2000.0,
        "dt"    => NaN,
        "Tmax"  => NaN,
        "nsave" => 300.0,
        "seed"  => 42.0,
        # Optionally diagonalize W at save times and store top-M
        # eigenvalues + overlaps with planted directions c_a. Off by
        # default because eigen(W) is O(N³).
        "neigen" => 0.0,
    )
    c_str  = "1.0"
    s0_str = "0.05"
    outfile = ""
    for arg in args
        if startswith(arg, "--")
            kv = split(arg[3:end], '='; limit=2)
            length(kv) == 2 || error("Bad argument: $arg (expected --key=value)")
            key, val = kv
            if key == "outfile"
                outfile = val
            elseif key == "c"
                c_str = val
            elseif key == "s0"
                s0_str = val
            elseif haskey(scalars, key)
                scalars[key] = parse(Float64, val)
            else
                error("Unknown argument: --$key")
            end
        else
            error("Unexpected positional argument: $arg (use --key=value)")
        end
    end
    return scalars, c_str, s0_str, outfile
end

parsed, c_str, s0_str, outfile_arg = parse_args(ARGS)

const γ_val = parsed["gamma"]
const η_val = parsed["eta"]
const β_val = parsed["beta"]
const ν_val = parsed["nu"]
const N     = round(Int, parsed["N"])

const c_vec = parse.(Float64, split(c_str, ','))
const K     = length(c_vec)
N >= K || error("N=$N must be ≥ K=$K")

s0_vec = let toks = split(s0_str, ',')
    if length(toks) == 1
        fill(parse(Float64, toks[1]), K)
    elseif length(toks) == K
        parse.(Float64, toks)
    else
        error("--s0 must be a scalar or CSV of length K=$K")
    end
end
sum(abs2, s0_vec) < 1 || error("Σ s0_a² = $(sum(abs2, s0_vec)) must be < 1")

const σ² = 1 / (γ_val * η_val)
const edge_val = 2sqrt(σ²)
const n_save = round(Int, parsed["nsave"])

const dt    = isnan(parsed["dt"])   ? min(0.02, 0.5 / max(ν_val, 1.0))       : parsed["dt"]
const T_max = isnan(parsed["Tmax"]) ? clamp(30.0 / ν_val, 5.0, 500.0)        : parsed["Tmax"]
const n_steps = round(Int, T_max / dt)
const save_interval = clamp(T_max / n_save, dt, 1.0)
const save_every = max(1, round(Int, save_interval / dt))

Random.seed!(round(Int, parsed["seed"]))

# ── Eigen-tracking (direct eigendecomposition of W at save times) ───
const M_eig   = round(Int, parsed["neigen"])
const use_eig = M_eig > 0
M_eig >= 0 || error("--neigen must be ≥ 0")
M_eig <= N || error("--neigen=$M_eig must be ≤ N=$N")

println("Finite-N simulation: K=$K, c=$c_vec, N=$N, γ=$γ_val, η=$η_val, β=$β_val, ν=$ν_val")
println("  dt=$dt, T=$T_max, n_steps=$n_steps, save_every=$save_every")
println("  σ²=$(round(σ²; digits=4)), edge=$(round(edge_val; digits=4))")
println("  s₀ = $s0_vec")
println("  convention: c_a = √N · e_a (first K basis vectors)")
println("  method: exact OU (W) + Heun predictor-corrector (x)")
if use_eig
    println("  eigen-tracking: top-$M_eig eigenvalues of W + overlaps with c_a at save times")
end

# ── Initial conditions ───────────────────────────────────────────────
# GOE stationary W (no spike)
σ_off = sqrt(1 / (γ_val * η_val * N))
W = zeros(N, N)
for j in 1:N
    W[j,j] = σ_off * sqrt(2) * randn()
    for i in 1:j-1
        z = σ_off * randn()
        W[i,j] = z
        W[j,i] = z
    end
end

# x on sphere with specified overlaps s0_a with c_a (c_a = √N e_a):
# s_a = x_a / √N, so x_a = √N s0_a for a=1..K, and the remaining N−K
# components carry the perpendicular mass √(N(1 − Σ s0_a²)).
x = zeros(N)
s2_sum = sum(abs2, s0_vec)
for a in 1:K
    x[a] = sqrt(N) * s0_vec[a]
end
if N > K
    perp = randn(N - K)
    perp .*= sqrt(N * (1 - s2_sum)) / norm(perp)
    @views x[K+1:end] .= perp
end
@assert abs(norm(x)^2 - N) < 1e-8 "||x||² = $(norm(x)^2) ≠ $N"
for a in 1:K
    s_check_a = x[a] / sqrt(N)
    println("  Initial s_$a = $(round(s_check_a; digits=5)) (target: $(s0_vec[a]))")
end

x0 = copy(x)

println("  n_saves ≈ $(div(n_steps, save_every) + 1)")

# ── Preallocate ─────────────────────────────────────────────────────
Wx = zeros(N)
ξ = zeros(N)
x_pred = zeros(N)
Wx_pred = zeros(N)

n_saves = div(n_steps, save_every) + 1
t_saved  = zeros(n_saves)
s_saved  = zeros(K, n_saves)                   # K channels
U_saved  = zeros(n_saves)
μ_saved  = zeros(n_saves)
C_saved  = zeros(n_saves)
S_saved  = zeros(K, n_saves)                   # per-channel signal block c_a^T W c_a / N

# ── Eigen-tracking storage ───────────────────────────────────────────
# At each save time, store top-M_eig eigenvalues of W and the overlaps
# u_{r,a} = (v_r · c_a/√N)² = v_r[a]² (since c_a = √N e_a).
λ_eig_saved = use_eig ? zeros(M_eig, n_saves) : zeros(0, n_saves)
u_eig_saved = use_eig ? zeros(M_eig, K, n_saves) : zeros(0, K, n_saves)  # u[r,a,t] = overlap² of top-r eigvec with c_a direction

# ── Helpers ──────────────────────────────────────────────────────────
function add_goe_noise!(W, N, scale)
    @inbounds for j in 1:N
        W[j,j] += scale * sqrt(2) * randn()
        for i in 1:j-1
            z = scale * randn()
            W[i,j] += z
            W[j,i] += z
        end
    end
end

function extract_observables!(s_out, S_out, W, x, x0, c_vec, K, Wx_buf, N, β_val, ν_val)
    # Per-channel overlap: c_a = √N e_a  →  s_a = x_a/√N
    for a in 1:K
        s_out[a] = x[a] / sqrt(N)
    end
    mul!(Wx_buf, W, x)
    U = dot(x, Wx_buf) / N
    μ = ν_val * U + ν_val * (N - 1) / (N * β_val)
    Ct = dot(x, x0) / N
    # Signal block S_a = c_a^T W c_a / N = W[a,a] (since c_a = √N e_a)
    for a in 1:K
        S_out[a] = W[a, a]
    end
    return (U=U, μ=μ, C=Ct)
end

# ── Top-M eigenvalue / eigenvector-overlap extraction ───────────────
# Compute top M eigenvalues of W (symmetric N×N). For each top eigenvector
# v_r, store u[r,a] = (c_a · v_r / √N)² = v_r[a]² since c_a = √N e_a.
# Uses full eigen(Symmetric(W)); O(N³) per call — call sparingly.
function extract_eig_observables!(λ_out, u_out, W, K, N, M_eig)
    F = eigen(Symmetric(W))
    # eigenvalues ascending; take top M_eig
    @views λ_top = F.values[end:-1:end-M_eig+1]
    @views V_top = F.vectors[:, end:-1:end-M_eig+1]
    λ_out .= λ_top
    @inbounds for a in 1:K, r in 1:M_eig
        u_out[r, a] = V_top[a, r]^2
    end
    return nothing
end

# Save initial state
s_tmp = zeros(K); S_tmp = zeros(K)
obs = extract_observables!(s_tmp, S_tmp, W, x, x0, c_vec, K, Wx, N, β_val, ν_val)
t_saved[1] = 0.0
s_saved[:, 1] = s_tmp
S_saved[:, 1] = S_tmp
U_saved[1] = obs.U; μ_saved[1] = obs.μ; C_saved[1] = obs.C
if use_eig
    λeig_tmp = zeros(M_eig)
    ueig_tmp = zeros(M_eig, K)
    extract_eig_observables!(λeig_tmp, ueig_tmp, W, K, N, M_eig)
    λ_eig_saved[:, 1] .= λeig_tmp
    u_eig_saved[:, :, 1] .= ueig_tmp
end

# ── Main simulation loop ────────────────────────────────────────────
function run_simulation!(W, x, Wx, ξ, x_pred, Wx_pred, c_vec, K, x0,
                         t_saved, s_saved, U_saved, μ_saved, C_saved, S_saved,
                         n_steps, save_every, n_saves,
                         λ_eig_saved, u_eig_saved, use_eig, M_eig)
    println("\nRunning finite-N simulation...")
    t_start = time()
    save_idx = 2
    s_tmp = zeros(K); S_tmp = zeros(K)
    λeig_tmp = use_eig ? zeros(M_eig)       : zeros(0)
    ueig_tmp = use_eig ? zeros(M_eig, K)    : zeros(0, K)

    # Exact OU transition kernel constants for W
    w_decay = exp(-γ_val * dt / 2)
    w_drive = (1 - w_decay) / (γ_val / 2)                         # ∫₀^dt e^{-γ(dt-s)/2} ds
    noise_W_scale = sqrt((1 - w_decay^2) / (γ_val * η_val * N))
    noise_x_scale = sqrt(2 * ν_val * dt / β_val)
    # Signal drive: (1/2) C integrated → w_drive · c_a / 2 added to W[a,a]
    # (with c_a = √N e_a, C is diagonal with entries c_a on first K diag entries)
    sampler_coeff = -w_drive * K / (2 * N)        # xx^T feedback with K factor
    half_dt_ν = (dt / 2) * ν_val

    for step in 1:n_steps
        # ── W update: exact OU exponential integrator ────────
        rmul!(W, w_decay)
        # Signal: integrated C drive (diagonal spikes)
        @inbounds for a in 1:K
            W[a, a] += w_drive * c_vec[a] / 2
        end
        # Sampler: integrated xx^T feedback with K factor
        BLAS.syr!('U', sampler_coeff, x, W)
        @inbounds for j in 2:N, i in 1:j-1
            W[j,i] = W[i,j]
        end
        # GOE noise with exact OU kernel variance
        add_goe_noise!(W, N, noise_W_scale)

        # ── x update: Heun predictor-corrector ───────────────
        mul!(Wx, W, x)
        U₁ = dot(x, Wx) / N
        μ₁ = U₁ + (N - 1) / (N * β_val)
        inv₁ = 1.0 / (1.0 + dt * ν_val * μ₁)
        @inbounds for i in 1:N
            x_pred[i] = (x[i] + dt * ν_val * Wx[i]) * inv₁
        end

        randn!(ξ)
        x_dot_ξ = dot(x, ξ) / N
        @inbounds for i in 1:N
            ξ[i] = noise_x_scale * (ξ[i] - x_dot_ξ * x[i])
            x_pred[i] += ξ[i]
        end
        x_pred .*= sqrt(N) / norm(x_pred)

        mul!(Wx_pred, W, x_pred)
        U₂ = dot(x_pred, Wx_pred) / N
        μ₂ = U₂ + (N - 1) / (N * β_val)

        μ_avg = (μ₁ + μ₂) / 2
        inv_avg = 1.0 / (1.0 + dt * ν_val * μ_avg)
        @inbounds for i in 1:N
            x[i] = (x[i] + half_dt_ν * (Wx[i] + Wx_pred[i])) * inv_avg + ξ[i]
        end
        x .*= sqrt(N) / norm(x)

        # ── Save observables ─────────────────────────────────────
        if step % save_every == 0 && save_idx <= n_saves
            obs = extract_observables!(s_tmp, S_tmp, W, x, x0, c_vec, K, Wx, N, β_val, ν_val)
            t_saved[save_idx] = step * dt
            s_saved[:, save_idx] = s_tmp
            S_saved[:, save_idx] = S_tmp
            U_saved[save_idx] = obs.U
            μ_saved[save_idx] = obs.μ
            C_saved[save_idx] = obs.C
            if use_eig
                extract_eig_observables!(λeig_tmp, ueig_tmp, W, K, N, M_eig)
                λ_eig_saved[:, save_idx] .= λeig_tmp
                u_eig_saved[:, :, save_idx] .= ueig_tmp
            end
            save_idx += 1

            if step % (50 * save_every) == 0
                elapsed = time() - t_start
                progress = step / n_steps * 100
                pr_info = use_eig ? "  λ=$(round.(λeig_tmp; digits=3))" : ""
                println("  t=$(round(step*dt; digits=1))  " *
                        "s=$(round.(s_tmp; digits=4))$pr_info  " *
                        "$(round(Int, progress))%  " *
                        "($(round(elapsed; digits=1))s)")
                flush(stdout)
            end
        end
    end

    elapsed_total = time() - t_start
    println("Done in $(round(elapsed_total; digits=1))s")
    return save_idx - 1
end

n_actual = run_simulation!(W, x, Wx, ξ, x_pred, Wx_pred, c_vec, K, x0,
                           t_saved, s_saved, U_saved, μ_saved, C_saved, S_saved,
                           n_steps, save_every, n_saves,
                           λ_eig_saved, u_eig_saved, use_eig, M_eig)

# ── Save results ─────────────────────────────────────────────────────
if isempty(outfile_arg)
    datadir = joinpath(@__DIR__, "..", "..", "data", "MSR")
    mkpath(datadir)
    ν_tag = replace(string(ν_val), "." => "p")
    c_tag = replace(join(c_vec, "_"), "." => "p")
    outfile = joinpath(datadir, "20260406_finiteN_K$(K)_c$(c_tag)_nu$(ν_tag).jld2")
else
    outfile = outfile_arg
    mkpath(dirname(outfile))
end
eig_payload = use_eig ? Dict(
    "M_eig"  => M_eig,
    "lambda" => λ_eig_saved[:, 1:n_actual],   # (M_eig, nsave)
    "u"      => u_eig_saved[:, :, 1:n_actual], # (M_eig, K, nsave)
) : Dict{String,Any}()

jldsave(outfile;
    t = t_saved[1:n_actual],
    s = s_saved[:, 1:n_actual],
    U = U_saved[1:n_actual],
    μ = μ_saved[1:n_actual],
    C = C_saved[1:n_actual],
    S = S_saved[:, 1:n_actual],
    c = c_vec,
    K = K,
    N = N,
    use_eig = use_eig,
    eig = eig_payload,
    params = Dict("γ"=>γ_val, "η"=>η_val, "β"=>β_val, "ν"=>ν_val,
                   "c"=>c_vec, "K"=>K,
                   "σ²"=>σ², "edge"=>edge_val, "N"=>N, "dt"=>dt, "s₀"=>s0_vec,
                   "M_eig"=>(use_eig ? M_eig : 0))
)
println("\nSaved to $outfile")
println("Final state (t=$(t_saved[n_actual])):")
for a in 1:K
    println("  s_$a = $(round(s_saved[a, n_actual]; digits=5))")
end
println("  μ  = $(round(μ_saved[n_actual]; digits=4))")
if use_eig
    println("  top-$M_eig eigenvalues of W = $(round.(λ_eig_saved[:, n_actual]; digits=4))")
    println("  overlaps |v_r · c_a/√N|² =")
    for r in 1:M_eig
        println("    v_$r: $(round.(u_eig_saved[r, :, n_actual]; digits=4))")
    end
end
