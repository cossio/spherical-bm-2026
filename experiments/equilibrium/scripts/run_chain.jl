#!/usr/bin/env julia
# Full Sym_N posterior, with its independent scalar trace integrated analytically.
# Only Julia standard libraries; exact exchange acceptance, no saddle in target.
using LinearAlgebra, Random, Statistics, TOML, SHA, Dates, Printf
include("bingham.jl")
using .ExactBingham
BLAS.set_num_threads(1)

function options(args)
    iseven(length(args)) || error("Expected --key value pairs")
    Dict(replace(args[i], "--" => "") => args[i+1] for i in 1:2:length(args))
end

function centered_goe(rng, n, c)
    z = randn(rng, n, n)
    a = (z + z') / sqrt(2n*c)
    t = tr(a)/n
    for i in 1:n
        a[i,i] -= t
    end
    a
end

function initial_matrix(rng, n, c, spectrum, gamma, chain)
    a = centered_goe(rng, n, c)
    if chain == 2 # deliberately strongly data aligned
        for k in eachindex(spectrum)
            a[k,k] += spectrum[k]/gamma
        end
    elseif chain == 3 # two equally strong, randomly oriented competing modes
        q = Matrix(qr(randn(rng,n,length(spectrum))).Q)[:,1:length(spectrum)]
        a .+= (3/sqrt(c)) .* (q*q')
    elseif chain == 4 # broad prior spectrum
        a .*= 2.5
    end
    a -= (tr(a)/n)*I
    a
end

const OBS = ["lambda1", "lambda2", "lambda3", "gap12", "top2_mean",
             "u11", "u22", "projector_c1", "projector_c2", "h1", "h2", "h_top2",
             "sample_c1", "sample_c2", "second_moment", "w11", "w22", "trace_shift",
             "centered_lambda1", "accepted", "rho", "aux_efficiency"]

function measure(rng, a, eig, n, c, nspin, accepted, rho, efficiency)
    y, _ = rand_bingham(rng, setup_bingham(eig.values), nspin)
    x = eig.vectors * y
    t = sqrt(2/c)*randn(rng)/n # exact independent trace component
    lam = reverse(eig.values) .+ t
    v = eig.vectors
    h1, h2 = mean(abs2, y[end,:]), mean(abs2, y[end-1,:])
    [lam[1], lam[2], n >= 3 ? lam[3] : NaN, lam[1]-lam[2], (lam[1]+lam[2])/2,
     v[1,end]^2, v[2,end-1]^2, sum(abs2,v[1,end-1:end]), sum(abs2,v[2,end-1:end]),
     h1, h2, h1+h2, mean(abs2,x[1,:]), mean(abs2,x[2,:]),
     sum(abs2,eig.values)/n+t^2, a[1,1]+t, a[2,2]+t, t, eig.values[end],
     accepted, rho, efficiency]
end

function main(args)
    opts = options(args)
    configpath = get(opts,"config",joinpath(@__DIR__,"..","config.toml"))
    config = TOML.parsefile(configpath)
    name = get(opts,"case","deep_coalesced")
    cases = vcat(config["cases"],config["controls"])
    case = only(filter(x->x["name"]==name,cases))
    n = parse(Int,get(opts,"n","64"))
    chain = parse(Int,get(opts,"chain","1"))
    warmup = parse(Int,get(opts,"warmup",string(config["warmup"])))
    draws = parse(Int,get(opts,"draws",string(config["draws"])))
    thin = parse(Int,get(opts,"thin",string(config["thin"])))
    nspin = parse(Int,get(opts,"measurement-spins",string(config["measurement_spins"])))
    gamma, eta, spectrum = case["gamma"],case["eta"],case["c"]
    k = length(spectrum)
    @assert isapprox(sum(spectrum),k) && n >= k && n >= 2
    @assert isinteger(k*eta) "Exact exchange here requires integer K eta"
    b = Int(k*eta)
    c = gamma*eta
    caseidx = findfirst(x->x["name"]==name,cases)
    seed = parse(Int,get(opts,"seed",string(config["seed_base"]+100000caseidx+100n+chain)))
    rng, measurement_rng = Xoshiro(seed),Xoshiro(seed+1000000000)
    out = get(opts,"out",joinpath(@__DIR__,"..","results","production"))
    folder = joinpath(out,name,"N$(n)","chain$(chain)")
    mkpath(folder)
    isfile(joinpath(folder,"metadata.toml")) && error("Refusing to overwrite completed chain: $folder")
    reference = zeros(n)
    reference[1:k] .= spectrum ./ gamma
    reference .-= k/(n*gamma)
    a = initial_matrix(rng,n,c,spectrum,gamma,chain)
    eig = eigen(Symmetric(a))
    rho = parse(Float64,get(opts,"rho","0.15"))
    fixedrho = haskey(opts,"rho")
    accepted, proposed_aux, block_accept = 0,0,0
    start = time()
    traceio = open(joinpath(folder,"draws.csv"),"w")
    warmio = open(joinpath(folder,"warmup.csv"),"w")
    println(traceio,join(vcat("iteration",OBS),","))
    println(warmio,"iteration,lambda1,lambda2,gap12,u11,projector_c2,rho,acceptance")
    for it in 1:(warmup+draws)
        retention = sqrt(1-rho^2)
        proposal = retention*a + rho*centered_goe(rng,n,c)
        for j in 1:n
            proposal[j,j] += (1-retention)*reference[j]
        end
        peig = eigen(Symmetric(proposal))
        y, np = rand_bingham(rng,setup_bingham(peig.values),b)
        proposed_aux += np
        x = peig.vectors*y # unit sphere auxiliaries at proposed full matrix
        logalpha = (n/2)*sum(x .* ((a-proposal)*x))
        accept = log(rand(rng)) < min(0.0,logalpha)
        if accept
            a, eig = proposal,peig
            block_accept += 1
            it > warmup && (accepted += 1)
        end
        if it <= warmup && it % 100 == 0
            rate = block_accept/100
            println(warmio,join([it,eig.values[end],eig.values[end-1],eig.values[end]-eig.values[end-1],eig.vectors[1,end]^2,sum(abs2,eig.vectors[2,end-1:end]),rho,rate],","))
            # Diminishing adaptation, warmup only. Freeze production transition.
            if !fixedrho
                rho = clamp(exp(log(rho) + 0.8/(it/100)^0.6*(rate-0.3)),0.005,0.95)
            end
            block_accept = 0
        elseif it > warmup && (it-warmup)%thin == 0
            row = measure(measurement_rng,a,eig,n,c,nspin,accepted/(it-warmup),rho,b*it/proposed_aux)
            println(traceio,join(vcat(it-warmup,row),","))
        end
        if it % 10000 == 0
            flush(traceio); flush(warmio)
            @printf("%s N=%d chain=%d step=%d rho=%.4f elapsed=%.1fs\n",name,n,chain,it,rho,time()-start)
            flush(stdout)
        end
    end
    close(traceio); close(warmio)
    sourcehashes = Dict(basename(p)=>bytes2hex(sha256(read(p))) for p in [@__FILE__,joinpath(@__DIR__,"bingham.jl"),configpath])
    metadata = Dict("case"=>name,"N"=>n,"chain"=>chain,"gamma"=>gamma,"eta"=>eta,"c"=>spectrum,
      "warmup"=>warmup,"iterations"=>draws,"thin"=>thin,"measurement_spins"=>nspin,"seed"=>seed,
      "acceptance"=>accepted/draws,"rho"=>rho,"auxiliary_acceptance"=>b*(warmup+draws)/proposed_aux,
      "elapsed_seconds"=>time()-start,"julia_version"=>string(VERSION),"blas_threads"=>BLAS.get_num_threads(),
      "source_sha256"=>sourcehashes,"completed_utc"=>string(now(UTC)),
      "target"=>"exact finite-N Sym_N posterior; independent analytic trace; exchange with exact Bingham auxiliaries",
      "initialization"=>["prior","data_aligned","random_coalesced","broad_prior"][chain],
      "slurm_job_id"=>get(ENV,"SLURM_JOB_ID","local"))
    open(joinpath(folder,"metadata.toml"),"w") do io
        TOML.print(io,metadata)
    end
    @printf("DONE %s N%d chain%d accept=%.3f aux=%.3f seconds=%.1f\n",name,n,chain,metadata["acceptance"],metadata["auxiliary_acceptance"],metadata["elapsed_seconds"])
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
