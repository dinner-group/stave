module Misc

    include("MathFunctions.jl")
    using .MathFunctions
########################################################################################
#
#                               I/O functions
#
########################################################################################

    function ParseInput(inputFile)
        f = open(inputFile)
        paramDict = Dict()
        try
            lines = readlines(f)
            for l in lines
                sl = split(l)
                intCases = ("Nx", "Ny", "Nz", "nSteps", "timeStride")
                strCases = ("obj", "veModel", "bcVE", "bcLB")
                if (sl[1] in intCases)
                    paramDict[string(sl[1])] = parse(Int64, string(sl[2]))
                elseif (sl[1] in strCases)
                    paramDict[string(sl[1])] = string(sl[2])
                else
                    paramDict[string(sl[1])] = parse(Float64, string(sl[2]))
                end
            end

        finally
            close(f)
        end
        return paramDict
    end



########################################################################################
#
#                               Miscellaneous helper functions
#
########################################################################################

    HeavisideStep(x) = (sign(x) + 1.0) / 2.0

    StepInterval(x, a, b) = HeavisideStep(x - a) - HeavisideStep(x - b)

    Sigmoid(t) = 1 / (1 + exp(-t))

    SmoothBump(t, a, Ta, Tb) = Sigmoid((t - Ta) / a) - Sigmoid((t - Tb) / a)

end
