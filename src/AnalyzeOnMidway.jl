include( "SharedStructs.jl")
using .SharedStructs
include("MathFunctions.jl")
using .MathFunctions
include("TestCases.jl")
include("LatBoltz.jl")
include("BerisEdwards.jl")
include("Viscoelasticity.jl")
include("Misc.jl")
include("ReactAdvDiff.jl")
include("ImmBound.jl")
include("Analysis.jl")
#using JLD
using FileIO
using Statistics

dir = "DirsDP"
subDir = "Dirs_Ds"
objName = "Ds"
case = "sV"
v1 = "Ds"
v1Vec = collect(vcat(0.0, 10.0.^(range(-16, -11, length = 11))))
v2 = ""
v2Vec = vcat([0], (10).^(range(-5,stop=-3,length=3)))
func = Analysis.GetDistancePlot


if case == "sV"
    ## single variable loop
    pathBase = "/project/svaikunt/csfloyd/CatchHopfield/" * dir * "/" * subDir
    bigV1Dict = Dict([])
    for v1i in v1Vec
        try
            pathName = pathBase * "/" * v1 * "_" * string(v1i) * "/"
            d = load(pathName * "SavedData.jld2")
            ret = func(d)
            bigV1Dict[v1i] = ret
        catch
            println("Error reading " * string(v1i))
        end
    end
    global retDict = bigV1Dict
end

if case == "dV"
    ## double variable loop
    pathBase = "/project/svaikunt/csfloyd/CatchHopfield/" * dir * "/" * subDir
    bigV1Dict = Dict([])
    for v1i in v1Vec
        bigV2Dict = Dict([])
        for v2i in v2Vec
            try
                pathName = pathBase * "/" * v1 * "_" * string(v1i) * "/" * v2 * "_" * string(v2i) * "/"
                d = load(pathName * "SavedData.jld2")
                ret = func(d)
                bigV2Dict[v2i] = ret
            catch
                println("Error reading " * string(v1i) * "_" * string(v2i))
            end                
        end
        bigV1Dict[v1i] = bigV2Dict
    end
    global retDict = bigV1Dict
end


# save results
outputDir = "/project/svaikunt/csfloyd/CatchHopfield/" * dir * "/AnalyzedData/"
pathName = outputDir * objName * ".jld2"
save(pathName, "retDict", retDict)
