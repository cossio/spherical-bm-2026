using TOML, SHA
# Byte-identical snapshot of USBMs2025.jl/src/rmt.jl used in this experiment.
const RMT_SOURCE = joinpath(@__DIR__,"..","provenance","reference_sources","rmt.jl")
@assert bytes2hex(sha256(read(RMT_SOURCE))) == "e18e09635b3a03306271837104c050074ad75e6b1a01e25b541f3d227627e40a"
include(RMT_SOURCE)
function predictions()
    root = normpath(joinpath(@__DIR__,".."))
    config = TOML.parsefile(joinpath(root,"config.toml"))
    output = Dict{String,Any}()
    for case in config["cases"]
        gamma, eta, spectrum = case["gamma"],case["eta"],case["c"]
        sol = RMT_Solution.solve(eta*spectrum,length(spectrum)*eta,gamma*eta)
        output[case["name"]] = Dict("gamma"=>gamma,"eta"=>eta,"c"=>spectrum,
            "lambda"=>sol.λ,"h_squared"=>sol.M,"u_squared"=>sol.u,
            "edge"=>2/sqrt(gamma*eta),"d"=>sol.d,"o"=>sol.o,"phase"=>string(sol.phase))
    end
    output["provenance"] = Dict("rmt_sha256"=>bytes2hex(sha256(read(RMT_SOURCE))))
    open(joinpath(root,"results","saddle_predictions.toml"),"w") do io
        TOML.print(io,output;sorted=true)
    end
    open(joinpath(root,"results","saddle_curve.csv"),"w") do io
        println(io,"gamma,lambda1,lambda2,h_squared")
        for gamma in sort(unique(vcat(collect(range(0.2,1.6;length=701)),[0.5625,1.25])))
            sol=RMT_Solution.solve([5.0,3.0],8.0,4gamma)
            println(io,join([gamma,sol.λ[1],sol.λ[2],sol.M],","))
        end
    end
end
predictions()
