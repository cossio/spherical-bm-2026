# Gaussian-visible Bernoulli-hidden RBM on Ken French standardized returns.
# Same SWAG / γ infrastructure as 02_train_rbm.jl but uses continuous data.

import RestrictedBoltzmannMachines as RBMs
using Optimisers: Adam, Descent
using HDF5
using LinearAlgebra: svdvals
using LogExpFunctions: logsumexp
using Statistics: mean, std
using Random: Random, MersenneTwister

function parse_args(args)
    d = Dict{String, String}()
    i = 1
    while i <= length(args)
        k = args[i]
        startswith(k, "--") || error("Expected flag at $i, got $k")
        d[k[3:end]] = args[i + 1]
        i += 2
    end
    return d
end

const AP = parse_args(ARGS)
const γ          = parse(Float64, get(AP, "gamma", "0.0"))
const M          = parse(Int,     get(AP, "M", "20"))
const SEED       = parse(Int,     get(AP, "seed", "0"))
const ITERS      = parse(Int,     get(AP, "iters", "8000"))
const BATCH      = parse(Int,     get(AP, "batch", "256"))
const LR         = parse(Float64, get(AP, "lr", "1e-3"))
const STEPS      = parse(Int,     get(AP, "steps", "1"))
const N_BETAS    = parse(Int,     get(AP, "nbetas", "10000"))
const N_AIS      = parse(Int,     get(AP, "nais", "20"))
const MTRAIN     = parse(Int, get(AP, "mtrain", "-1"))
# Optional seed for the data subsample, decoupled from RBM-init SEED.
# Setting this to a fixed value (across all sweeps) lets all RBM-init seeds
# train on the SAME M_train subset, isolating training+eval noise.
const MTRAIN_SEED = parse(Int, get(AP, "mtrain_seed", "-1"))
const SWAG       = parse(Bool, get(AP, "swag", "false"))
const SWAG_START = parse(Int, get(AP, "swag_start", "6000"))
const SWAG_EVERY = parse(Int, get(AP, "swag_every", "100"))
const SWAG_LR    = parse(Float64, get(AP, "swag_lr", "5e-3"))
const SWAG_NBETAS = parse(Int, get(AP, "swag_nbetas", "5000"))
const SWAG_NAIS   = parse(Int, get(AP, "swag_nais", "10"))
const LANGEVIN_T = parse(Float64, get(AP, "langevin_T", "0.0"))
const DATA_H5    = get(AP, "data", "data/kenfrench49_daily.h5")
const OUT        = get(AP, "out", "results/run.h5")

println("=== Gaussian-RBM finance run ===")
println("  γ=$γ  M=$M  seed=$SEED  iters=$ITERS  swag=$SWAG  T=$LANGEVIN_T")
println("  data=$DATA_H5  out=$OUT")

Random.seed!(SEED)

# ── Data ────────────────────────────────────────────────────────────

h5 = h5open(DATA_H5, "r")
Xtr = Float32.(read(h5, "train_x"))   # shape (N_vis, N_train) after F-order swap
Xte = Float32.(read(h5, "test_x"))
close(h5)
N_vis = size(Xtr, 1)
N_train_total = size(Xtr, 2)
println("  train: $(size(Xtr))  test: $(size(Xte))  N_vis=$N_vis")

if MTRAIN > 0 && MTRAIN < N_train_total
    split_seed = MTRAIN_SEED >= 0 ? MTRAIN_SEED : SEED
    rng_split = MersenneTwister(split_seed)
    idx = Random.randperm(rng_split, N_train_total)[1:MTRAIN]
    Xtr = Xtr[:, idx]
end

# ── Build Gaussian-Binary RBM ───────────────────────────────────────

function make_rbm(N, M)
    gauss  = RBMs.Gaussian(; θ = zeros(Float32, N), γ = ones(Float32, N))
    hidden = RBMs.Binary(; θ = zeros(Float32, M))
    w = (randn(Float32, N, M) ./ sqrt(Float32(N)))
    return RBMs.RBM(gauss, hidden, w)
end

rbm = make_rbm(N_vis, M)
RBMs.initialize!(rbm, Xtr)

# ── Logging callback ────────────────────────────────────────────────

log_iters, log_Ftr, log_Fte, log_wnorm = Int[], Float64[], Float64[], Float64[]
log_every = max(1, div(ITERS, 50))

cb_log = function(; rbm, iter, kwargs...)
    if iter % log_every == 0 || iter == ITERS
        n_probe = min(1024, size(Xtr, 2))
        Ftr = mean(RBMs.free_energy(rbm, Xtr[:, rand(1:size(Xtr, 2), n_probe)]))
        Fte = mean(RBMs.free_energy(rbm, Xte[:, rand(1:size(Xte, 2), min(n_probe, size(Xte, 2)))]))
        wn  = sqrt(sum(abs2, rbm.w))
        push!(log_iters, iter); push!(log_Ftr, Float64(Ftr))
        push!(log_Fte, Float64(Fte)); push!(log_wnorm, Float64(wn))
    end
end

# ── Training ────────────────────────────────────────────────────────

if !SWAG
    println("Single-phase Adam, $ITERS iters …")
    t0 = time()
    RBMs.pcd!(rbm, Xtr;
        batchsize = BATCH, iters = ITERS, steps = STEPS,
        optim = Adam(Float32(LR)),
        l2_weights = Float32(γ),
        callback = cb_log,
    )
    t_train = time() - t0
    snapshots_w = typeof(rbm.w)[]
    snapshots_iter = Int[]
    snapshots_a = typeof(rbm.visible.par)[]
    snapshots_b = typeof(rbm.hidden.par)[]
else
    println("Phase 1 (burn-in): $SWAG_START Adam iters …")
    t0 = time()
    RBMs.pcd!(rbm, Xtr;
        batchsize = BATCH, iters = SWAG_START, steps = STEPS,
        optim = Adam(Float32(LR)),
        l2_weights = Float32(γ),
        callback = cb_log,
    )
    t_phase1 = time() - t0
    println("  done in $(round(t_phase1, digits=1)) s")

    snapshots_a = typeof(rbm.visible.par)[]
    snapshots_b = typeof(rbm.hidden.par)[]
    snapshots_w = typeof(rbm.w)[]
    snapshots_iter = Int[]
    noise_scale = LANGEVIN_T > 0 ? Float32(sqrt(2 * SWAG_LR * LANGEVIN_T)) : 0f0

    cb_swag = function(; rbm, iter, kwargs...)
        if noise_scale > 0
            rbm.visible.par .+= noise_scale .* randn(Float32, size(rbm.visible.par))
            rbm.hidden.par  .+= noise_scale .* randn(Float32, size(rbm.hidden.par))
            rbm.w           .+= noise_scale .* randn(Float32, size(rbm.w))
        end
        cb_log(; rbm, iter = iter + SWAG_START, kwargs...)
        if iter % SWAG_EVERY == 0
            push!(snapshots_a, copy(rbm.visible.par))
            push!(snapshots_b, copy(rbm.hidden.par))
            push!(snapshots_w, copy(rbm.w))
            push!(snapshots_iter, iter + SWAG_START)
        end
    end

    swag_iters = ITERS - SWAG_START
    println("Phase 2 (SWAG): $swag_iters SGD iters at lr=$SWAG_LR …")
    t0 = time()
    RBMs.pcd!(rbm, Xtr;
        batchsize = BATCH, iters = swag_iters, steps = STEPS,
        optim = Descent(Float32(SWAG_LR)),
        l2_weights = Float32(γ),
        callback = cb_swag,
    )
    t_phase2 = time() - t0
    println("  done in $(round(t_phase2, digits=1)) s, K=$(length(snapshots_a))")
    t_train = t_phase1 + t_phase2
end

# ── MAP eval ────────────────────────────────────────────────────────

println("MAP AIS …")
t0 = time()
F_ais_map = RBMs.aise(rbm; nbetas = N_BETAS, nsamples = N_AIS)
logZ_map = Float64(RBMs.logmeanexp(F_ais_map))
logZ_map_std = Float64(std(F_ais_map))
Ftr_full = RBMs.free_energy(rbm, Xtr)
Fte_full = RBMs.free_energy(rbm, Xte)
LL_tr_map = mean(-Ftr_full .- logZ_map)
LL_te_map = mean(-Fte_full .- logZ_map)
t_ais_map = time() - t0
println("  logZ_MAP=$(round(logZ_map, digits=3))±$(round(logZ_map_std, digits=3))")
println("  <logP>_train(MAP)=$(round(Float64(LL_tr_map), digits=3))")
println("  <logP>_test (MAP)=$(round(Float64(LL_te_map), digits=3))  [$(round(t_ais_map, digits=1))s]")

# ── SWAG eval ───────────────────────────────────────────────────────

K = length(snapshots_a)
LL_tr_snap_mean = 0.0; LL_te_snap_mean = 0.0
LL_tr_pp = 0.0;        LL_te_pp = 0.0
snap_logZ = Float64[]
t_swag_ais = 0.0

if SWAG && K > 0
    println("Per-snapshot AIS for K=$K …")
    t0 = time()
    tmp = make_rbm(N_vis, M)
    snap_F_tr = zeros(Float64, size(Xtr, 2), K)
    snap_F_te = zeros(Float64, size(Xte, 2), K)
    snap_logZ = zeros(Float64, K)
    for k in 1:K
        tmp.visible.par .= snapshots_a[k]
        tmp.hidden.par .= snapshots_b[k]
        tmp.w .= snapshots_w[k]
        F_ais_k = RBMs.aise(tmp; nbetas = SWAG_NBETAS, nsamples = SWAG_NAIS)
        snap_logZ[k] = Float64(RBMs.logmeanexp(F_ais_k))
        snap_F_tr[:, k] = Float64.(RBMs.free_energy(tmp, Xtr))
        snap_F_te[:, k] = Float64.(RBMs.free_energy(tmp, Xte))
    end
    t_swag_ais = time() - t0

    LL_te_per = -snap_F_te .- snap_logZ'
    LL_tr_per = -snap_F_tr .- snap_logZ'
    LL_te_snap_mean = mean(mean(LL_te_per; dims=1))
    LL_tr_snap_mean = mean(mean(LL_tr_per; dims=1))
    logK = log(K)
    LL_te_pp = mean([logsumexp(view(LL_te_per, i, :)) - logK for i in 1:size(Xte, 2)])
    LL_tr_pp = mean([logsumexp(view(LL_tr_per, i, :)) - logK for i in 1:size(Xtr, 2)])
    println("  <logP>_test(snap)=$(round(LL_te_snap_mean, digits=3))  (PP)=$(round(LL_te_pp, digits=3))")
end

# ── W spectrum ──────────────────────────────────────────────────────

W_map = reshape(Array(rbm.w), N_vis, M)
σ_map = Float64.(svdvals(W_map))
println("  top-5 σ(W_MAP): $(round.(σ_map[1:min(end, 5)], digits=3))")

# ── Save ────────────────────────────────────────────────────────────

isdir(dirname(OUT)) || mkpath(dirname(OUT))
h5open(OUT, "w") do h
    write(h, "gamma", γ); write(h, "M", M); write(h, "seed", SEED)
    write(h, "iters", ITERS); write(h, "swag_enabled", Int(SWAG))
    write(h, "N_vis", N_vis); write(h, "n_train", size(Xtr, 2))
    write(h, "n_test", size(Xte, 2)); write(h, "t_train", t_train)
    write(h, "t_ais_map", t_ais_map); write(h, "langevin_T", LANGEVIN_T)
    write(h, "log/iters", log_iters); write(h, "log/Ftr", log_Ftr)
    write(h, "log/Fte", log_Fte); write(h, "log/wnorm", log_wnorm)
    # MAP
    write(h, "map/logZ", logZ_map); write(h, "map/logZ_std", logZ_map_std)
    write(h, "map/LL_train", Float64(LL_tr_map)); write(h, "map/LL_test", Float64(LL_te_map))
    write(h, "map/W_svdvals", σ_map)
    # SWAG
    write(h, "swag/K", K)
    write(h, "swag/LL_train_snap_mean", LL_tr_snap_mean)
    write(h, "swag/LL_test_snap_mean", LL_te_snap_mean)
    write(h, "swag/LL_train_pp", LL_tr_pp); write(h, "swag/LL_test_pp", LL_te_pp)
    write(h, "swag/W_svdvals_mean", σ_map)  # placeholder
    write(h, "swag/t_ais", t_swag_ais)
    if !isempty(snap_logZ); write(h, "swag/logZ", snap_logZ); end
end
println("Wrote $OUT")
