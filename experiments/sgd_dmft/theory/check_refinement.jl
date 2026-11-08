function read_numeric_csv(path)
    lines = readlines(path)
    names = split(lines[1], ',')
    data = reduce(vcat, permutedims.(parse.(Float64, split(line, ',')) for line in lines[2:end]))
    return names, data
end

println("case,observable,max_abs_difference,rms_difference,coarse_final,fine_final")
coarse_tag, fine_tag = isempty(ARGS) ? ("0p05", "0p025") :
    length(ARGS) == 2 ? (ARGS[1], ARGS[2]) : error("Expected zero or two grid tags")
for tag in ("A", "B", "C", "D")
    names, coarse = read_numeric_csv(joinpath(@__DIR__, "$(tag)_dt$(coarse_tag)_optimized.csv"))
    fine_names, fine_all = read_numeric_csv(joinpath(@__DIR__, "$(tag)_dt$(fine_tag)_optimized.csv"))
    names == fine_names || error("Column mismatch")
    fine = fine_all[1:2:end, :]
    size(coarse) == size(fine) || error("Grid mismatch")
    maximum(abs, coarse[:, 1] - fine[:, 1]) < 1e-12 || error("Times differ")
    for j in 2:size(coarse, 2)
        delta = coarse[:, j] - fine[:, j]
        println(join((tag, names[j], maximum(abs, delta), sqrt(sum(abs2, delta) / length(delta)),
                      coarse[end, j], fine[end, j]), ','))
    end
end
