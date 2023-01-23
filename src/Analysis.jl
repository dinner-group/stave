module Analysis

    include("MathFunctions.jl")
    using .MathFunctions

    include("SharedStructs.jl")
    using .SharedStructs

    include("LatBoltz.jl")
    include("ImmBound.jl")
    include("BerisEdwards.jl")
    include("TestCases.jl")


    using FFTW
    using Statistics
    using BinnedStatistics

    ########################################################################################
    #
    #                               Helper functions
    #
    ########################################################################################

    function thetaFromPVec(PVec)
        return atan(PVec[2], PVec[1])
    end

    function thetaFromPVecSoA(PVecX, PVecY)
        return [thetaFromPVec([PVecX[x,y],PVecY[x,y]]) for x in 1:size(PVecX)[1], y in 1:size(PVecX)[2]] # need to test
    end

    function thetaFromNem(PVec)
        return mod(thetaFromPVec(PVec), pi)
    end 

    function thetaFromNemSoA(PVecX, PVecY)
        return [thetaFromNem([PVecX[x,y],PVecY[x,y]]) for x in 1:size(PVecX)[1], y in 1:size(PVecX)[2]] # need to test
    end

    function TorqueDensitySoA(sigmaYX, sigmaXY)
        return sigmaYX .- sigmaXY
    end

    function DivV(grid, vSoA, bcx, bcy)
        return DivVectorOnSoA2D(grid, vSoA, BCDerivDict[bcx], BCDerivDict[bcy]).Values
    end

    function VorticitySoA(grid, velocitySoA, bcx, bcy)
        (OmegaSoA, PsiSoA) = OmegaPsi2DSoA(grid, velocitySoA, bcx, bcy)
        return 2 .* OmegaSoA.XYValues
    end

    function ViscousDissipation(grid, velocitySoA, bcx, bcy)
        bcDerivX = BCDerivDict[bcx]
        bcDerivY = BCDerivDict[bcy]
        gradVSoA = GradVecOnSoA2D(grid, velocitySoA, bcDerivX, bcDerivY)
        return MatrixDoubleDotMatrixOnSoA2D(grid, gradVSoA, gradVSoA) # returns SoA field
    end

    function TotalDissipationTrajectory(grid, velocityArray, bcx, bcy)
        dissList = []
        for t in 1:length(velocityArray)
            dissSoA = ViscousDissipation(grid, velocityArray[t], bcx, bcy)
            push!(dissList, sum(sum(dissSoA.Values)))
        end 
        return dissList 
    end

    function DivS(grid, tSoA, bcx, bcy)
        return DivTensorOnSoA2D(grid,  MatrixTransposeOnSoA2D(grid, tSoA), BCDerivDict[bcx], BCDerivDict[bcy])
    end

    function GetGrid(parameters)
        Nx = parameters["Nx"]
        Ny = parameters["Ny"]
        dx = parameters["dx"]
        return Grid2D(Nx, Ny, dx)
    end

    function NormalizeArray!(arr)
        arr .= abs.(arr)
        arr .= arr ./ mean(arr)
    end


    function ThetaSign(pAx, pAy, pBx, pBy)
        crit = pAx * pBy - pAy * pBx
        if crit > 0
            return 1
        else
            return -1
        end
    end
    
    function Dot2D(pA, pB)
        return pA[1]*pB[1] + pA[2]*pB[2]
    end
    
    function SafeACos(x)
        if x < -1.0
            x = -1.0
        elseif x > 1.0
            x = 1.0
        end
        return acos(x)
    end
    
    function RelativeAngle(pA2D0, pB2D0)
        dp0 = Dot2D(pA2D0, pB2D0) / sqrt((pA2D0[1]^2 + pA2D0[2]^2) * (pB2D0[1]^2 + pB2D0[2]^2))
        sign = ThetaSign(pA2D0[1], pA2D0[2], pB2D0[1], pB2D0[2])
        theta0 = sign * SafeACos(dp0)
        return theta0
    end
    
    function WindingNumbers(grid, dirSoA)
        Nx = grid.Nx
        Ny = grid.Ny
        windingNumberArray = zeros((Nx-1, Ny-1))
        for i = 1:Nx-1, j = 1:Ny-1
            p1 = [dirSoA.XValues[i,j], dirSoA.YValues[i,j]]
            p2 = [dirSoA.XValues[i+1,j], dirSoA.YValues[i+1,j]]
            p3 = [dirSoA.XValues[i+1,j+1], dirSoA.YValues[i+1,j+1]]
            p4 = [dirSoA.XValues[i,j+1], dirSoA.YValues[i,j+1]]
            delT1 = RelativeAngle(p1, p2)
            delT2 = RelativeAngle(p2, p3)
            delT3 = RelativeAngle(p3, p4)
            delT4 = RelativeAngle(p4, p1)
            windingNumberArray[i,j] = (delT1 + delT2 + delT3 + delT4) ./ (2*pi)
        end
        return windingNumberArray
    end

    function AngleRange(t)
        if t > pi/2
            return t - pi/2
        elseif t < - pi/2
            return t + pi/2
        else 
            return t
        end 
    end

    function WindingNumbersNematic(grid, dirSoA)
        Nx = grid.Nx
        Ny = grid.Ny
        windingNumberArray = zeros((Nx-1, Ny-1))
        for i = 1:Nx-1, j = 1:Ny-1
            t1 = Analysis.thetaFromNem([dirSoA.XValues[i,j], dirSoA.YValues[i,j]])
            t2 = Analysis.thetaFromNem([dirSoA.XValues[i+1,j], dirSoA.YValues[i+1,j]])
            t3 = Analysis.thetaFromNem([dirSoA.XValues[i+1,j+1], dirSoA.YValues[i+1,j+1]])
            t4 = Analysis.thetaFromNem([dirSoA.XValues[i,j+1], dirSoA.YValues[i,j+1]])
            delT1 = AngleRange(t2-t1)
            delT2 = AngleRange(t3-t2)
            delT3 = AngleRange(t4-t3)
            delT4 = AngleRange(t1-t4)

            windingNumberArray[i,j] = sum([delT1, delT2, delT3, delT4]) / (pi)
        end
        return windingNumberArray
    end
 
    function CreateScatterFromWNA(windingNumberArray, scale, cutoff = 0.4)
        indsp1 = findall(windingNumberArray .> cutoff)
        indsm1 = findall(windingNumberArray .< - cutoff)
        xsp = [indsp1[t][1] + 0.5 for t = 1:length(indsp1)] .* scale
        ysp = [indsp1[t][2] + 0.5 for t = 1:length(indsp1)] .* scale
        xsm = [indsm1[t][1] + 0.5 for t = 1:length(indsm1)] .* scale
        ysm = [indsm1[t][2] + 0.5 for t = 1:length(indsm1)] .* scale
        return (xsp, ysp, xsm, ysm)
    end

    ########################################################################################
    #
    #                               Analysis functions
    #
    ########################################################################################

    function GetTimeDomain(parameters, start = 1)
        nt = parameters["nSteps"] / parameters["timeStride"]
        return collect(start:nt) .* parameters["timeStride"];
    end

    function Distance(vec, init)
        return sqrt((vec[1] - init[1])^2 + (vec[2] - init[2])^2)
    end

    function DistancePointToLine(p0, p1, p2)
        return abs((p2[1] - p1[1]) * (p1[2] - p0[2]) - (p1[1] - p0[1]) * (p2[2] - p1[2])) / sqrt((p2[1] - p1[1])^2 + (p2[2] - p1[2])^2)
    end

    function TimeAverageFT(field, window, sep)
        field = field ./ maximum(field)
        arr = abs.(fftshift(fft(field))).^2
        ns = floor(size(arr)[1] / 2) # assume square
        retDict = Dict()
        for i = -window:window
            ix = Int(ns + i)
            r = abs(i)
            rInd = Int(floor(r / sep) + 1) # rm <= r < rp
            val = arr[ix]
            if !(rInd in keys(retDict))
                retDict[rInd] = [val]
            else
                push!(retDict[rInd], val)
            end
        end
        imax = abs(findmax(arr)[2] - ns)

        rads = sort(collect(keys(retDict)))
        vals = similar(rads)
        av = 0.0
        valT = 0.0
        for (r, rad) in enumerate(rads)
            vals[r] = mean(retDict[rad])
            av += rad * vals[r]
            valT += vals[r]
        end
        return (rads .* sep, vals, 2 * ns / (av * sep / valT) , 2 * ns / (imax))
    end

    function AngularAverageFT(field, window, sep)
        arr = abs.(fftshift(fft(field))).^2
        ns = floor(size(arr)[1] / 2) # assume square
        retDict = Dict()
        for j = -window:window, i = -window:window
            ix = Int(ns + j)
            iy = Int(ns + i)
            r = sqrt(j^2 + i^2)
            rInd = Int(floor(r / sep) + 1) # rm <= r < rp
            val = arr[ix, iy]
            if !(rInd in keys(retDict))
                retDict[rInd] = [val]
            else
                push!(retDict[rInd], val)
            end
        end
        rads = sort(collect(keys(retDict)))
        vals = similar(rads)
        av = 0.0
        valT = 0.0
        for (r, rad) in enumerate(rads)
            vals[r] = mean(retDict[rad])
            av += rad * vals[r]
            valT += vals[r]
        end
        # last element is the weighted average spatial wavelength, in units of pixels
        return (rads .* sep, vals, 2 * ns / (av * sep / valT) )
    end

    function CorrelationFunction(field)
        field = field ./ maximum(field)
        n = size(field)[1]
        sum = zeros(n)
        samples = zeros(n)
        @inbounds for i = 1:n, j = 1:n
            sep = abs(j-i) + 1
            sum[sep] += field[i,i]*field[i,j] + field[i,i]*field[j,i]
            samples[sep] += 2
        end
        return sum ./ samples
    end

    function NormOfVectorSoAList(vectorSoAList)
        normList = []
        for t = 1:length(vectorSoAList)
            vectorSoA = vectorSoAList[t]
            normArray = similar(vectorSoA.XValues)
            for y = 1:size(normArray)[2]
                for x = 1:size(normArray)[1]
                    normArray[x,y] = sqrt(vectorSoA.XValues[x,y]^2 + vectorSoA.YValues[x,y]^2)
                end 
            end
            push!(normList, normArray)
        end
        return normList 
    end

    function RootMeanSquared(fieldFrames)
        # assumed to be a list of 2D arrays
        nT = length(fieldFrames) 
        nX = size(fieldFrames[1])[1]
        nY = size(fieldFrames[1])[2]
        sum = 0.0
        for t = 1:nT
            for y = 1:nY
                for x = 1:nX 
                    sum += fieldFrames[t][x,y]^2
                end 
            end 
        end 
        sum /= (nT * nX * nY)
        return sqrt(sum)
    end

    

    function VelocityCorrelationFunction(u, v, dsFac, rMax)

        uTemp = u[1:dsFac:end, 1:dsFac:end]
        vTemp = v[1:dsFac:end, 1:dsFac:end]

        uShape = size(uTemp)
        cofr = zeros(rMax)
        nTot = uShape[1]*uShape[2]

        sz = uShape[2]
        xL = collect(0:2*sz-1)
        x = xL .* ones(2*sz, 2*sz)
        y = xL' .* ones(2*sz, 2*sz)
        
        distance = sqrt.((x .- sz).^(2) .+ (y .- sz).^(2))
        mag = sqrt.(uTemp.^2 .+ vTemp.^2)
        norm = 0.0

        for aa in 1:uShape[1]
	     	bb = aa
                r = distance[sz - aa + 2: 2 * sz - aa + 1, sz - bb + 2 : 2 * sz - bb + 1 ]
                corr = ((uTemp[aa,bb] .* uTemp) .+ (vTemp[aa,bb] .* vTemp)) ./ (sqrt(uTemp[aa,bb]^2 + vTemp[aa,bb]^2) .* mag)
                #corr = ((uTemp[aa,bb] .* uTemp) .+ (vTemp[aa,bb] .* vTemp)) 
                #norm += (uTemp[aa,bb]^2 + vTemp[aa,bb]^2) 
                edges, centers, result = BinnedStatistics.binnedStatistic(vcat(r...), vcat(corr...), statistic = :mean, nbins = rMax, binMin = 0, binMax = rMax)
                cofr .+= result
        end

        return cofr ./ nTot # / norm
    end

    function VorticityCorrelationFunction(w, dsFac, rMax)

        wTemp = w[1:dsFac:end, 1:dsFac:end]

        uShape = size(wTemp)
        cofw = zeros(rMax)
        nTot = uShape[1]*uShape[2]

        sz = uShape[2]
        xL = collect(0:2*sz-1)
        x = xL .* ones(2*sz, 2*sz)
        y = xL' .* ones(2*sz, 2*sz)
        
        distance = sqrt.((x .- sz).^(2) .+ (y .- sz).^(2))
        mag = sqrt.(wTemp.^2)

        norm = 0.0
        for aa in 1:uShape[1]
	    	bb = aa
                r = distance[sz - aa + 2: 2 * sz - aa + 1, sz - bb + 2 : 2 * sz - bb + 1 ]
                corr = (wTemp[aa,bb] .* wTemp)
                norm += wTemp[aa,bb]^2
                edges, centers, result = BinnedStatistics.binnedStatistic(vcat(r...), vcat(corr...), statistic = :mean, nbins = rMax, binMin = 0, binMax = rMax)
                cofw .+= result
        end

        return cofw ./ norm
    end

    function SumFieldOverCirlces(grid, field, sig, N)

        (Nx, Ny) = (grid.Nx, grid.Ny)

        Nsqr = sqrt(N)
        deltax = Nx / Nsqr
        deltay = Ny / Nsqr

        positions = []
        result = []

        for n = 1:N 

            xi = (mod1(n, Nsqr) - 0.5) * deltax
            yi = (floor((n-1) / Nsqr)  + 0.5) * deltay
            res = 0.0
            for x = 1:Nx, y = 1:Ny 
                rVec = [x - xi, y - yi]
                rNorm = VectorNorm2D(rVec) 
                if rNorm <= sig
                    res += field[x,y]
                end 
            end 

            push!(positions, (xi,yi))
            push!(result, res)
        end 

        return (positions, result)
    end

    function DivForceMagnitudeOverCircles(grid, tensorSoA, sig, N, parameters)
        
        viscoelasticForceSoA = DivTensorOnSoA2D(grid, MatrixTransposeOnSoA2D(grid, tensorSoA), BCDerivDict[parameters["bcVE_X"]], BCDerivDict[parameters["bcVE_Y"]])
        normSoA = VectorNormSoA(viscoelasticForceSoA)
        return SumFieldOverCirlces(grid, normSoA.Values, sig, N)
    end 

    ########################################################################################
    #
    #                               Midway analysis functions
    #
    ########################################################################################

    function GetCoMList(d)
        nodePositionArray = deepcopy(d["nodePositionArray"]);
        parameters = d["parameters"]
        timeDomain = GetTimeDomain(parameters, 0);
        comXList = []
        comYList = []
        for t in 1:length(nodePositionArray)
            push!(comXList,ImmBound.CenterOfMassPos(nodePositionArray[t])[1])
            push!(comYList,ImmBound.CenterOfMassPos(nodePositionArray[t])[2])
        end
        return [timeDomain, comXList, comYList]
    end

    function GetIBAngle(d)
        parameters = d["parameters"]
        (_, comXList, comYList) = GetCoMList(d)
        nodePositionArray = deepcopy(d["nodePositionArray"]);
        timeDomain = GetTimeDomain(parameters, 1);
        angleList = []
        for t in 1:length(comXList)
            n = nodePositionArray[t][1]
            v = [n[1] - comXList[t], n[2] - comYList[t]]
            v = v ./ VectorNorm2D(v)
            angle = atan(v[2], v[1])
            push!(angleList, angle)
        end 
        return [timeDomain, angleList]
    end

    function GetDistancePlot(d)
        parameters = d["parameters"]
        (timeDomain, comXList, comYList) = GetCoMList(d)
        distComList = []
        init = parameters["center"]
        push!(distComList, 0.0)
        for t in 1:length(comXList)
            push!(distComList, Distance([comXList[t], comYList[t]], init))
        end
        return [timeDomain, distComList]
    end

    function GetDistancePlotFinal(d)
        nodePositionArray = deepcopy(d["nodePositionArray"]);
        parameters = d["parameters"]
        timeDomain = GetTimeDomain(parameters, 0);
        comXListI = ImmBound.CenterOfMassPos(nodePositionArray[1])[1]
        comYListI = ImmBound.CenterOfMassPos(nodePositionArray[1])[2]
        comXListF = ImmBound.CenterOfMassPos(nodePositionArray[end])[1]
        comYListF = ImmBound.CenterOfMassPos(nodePositionArray[end])[2]
        init = parameters["center"]
        final = [comXListF, comYListF]
        return Distance(final, init)*parameters["Cx"]

    end

    function GetTransversePlot(d)
        nodePositionArray = deepcopy(d["nodePositionArray"]);
        parameters = d["parameters"]
        timeDomain = GetTimeDomain(parameters, 0);
        comXList = []
        comYList = []
        for t in 1:length(nodePositionArray)
            push!(comXList,ImmBound.CenterOfMassPos(nodePositionArray[t])[1])
            push!(comYList,ImmBound.CenterOfMassPos(nodePositionArray[t])[2])
        end
        init = parameters["center"]
        dest = parameters["rB"]
        distComList = []
        push!(distComList, 0.0)
        for t in 1:length(comXList)
            push!(distComList, DistancePointToLine([comXList[t], comYList[t]], init, dest))
        end
        return [timeDomain, distComList*parameters["Cx"]]
    end

    function GetAbsDiv(d)
        parameters = d["parameters"]
        Nx = params["Nx"]
        Ny = params["Ny"]
        dx = params["dx"]
        myGrid = Grid2D(Nx, Ny, dx)
        directorArray = d["directorArray"]
        timeDomain = GetTimeDomain(parameters, 1);
        absDiv = [sum(abs.(DivVecOnMesh2DBC(myGrid, directorArray[t].Values, parameters["bcLB_mX"][1:3], parameters["bcLB_mY"][1:3]))) for t in 1:(length(directorArray))]
        return [timeDomain, absDiv]
    end

    function DecimateData(d, timeFac)
        dD = Dict(
        "parameters" => d["parameters"],
        "densityArray" => d["densityArray"][1:timeFac:end],
        "velocityArray" => d["velocityArray"][1:timeFac:end],
        "sigmaVEArray" => d["sigmaVEArray"][1:timeFac:end],
        "directorArray" => d["directorArray"][1:timeFac:end],
        "kArray" => d["kArray"][1:timeFac:end],
        "rExtArray" => d["rExtArray"][1:timeFac:end],
        "nodePositionArray" => d["nodePositionArray"][1:timeFac:end]
        )
        dD["parameters"]["timeStride"]  = dD["parameters"]["timeStride"] * timeFac
        return dD
    end

    function SubsetData(d, arguments)
        dD = Dict([])
        for arg in arguments
            dD[arg] = d[arg]
        end
        return dD
    end

    function SpatialFrequency(d; window = 50, sep = 0.1)
        avList = []
        myGrid = GetGrid(d["parameters"])
        for t = 1:length(d["sigmaVEArray"])
            #field = TorqueDensitySoA(d["sigmaVEArray"][t].YXValues, d["sigmaVEArray"][t].XYValues)
            field = VorticitySoA(myGrid, d["velocityArray"][t], d["parameters"]["bcLB_mX"][1:3], d["parameters_Ym"]["bcLB_mY"][1:3])
            NormalizeArray!(field)
            (rads, vals, av) = AngularAverageFT(field, window, sep)
            push!(avList, av)
        end
        timeDomain = GetTimeDomain(d["parameters"], 1);
        return [timeDomain, avList]
    end

    function TimeFrequency(d, t1, t2; window = 20, sep = 0.1)
        avList = []
        myGrid = GetGrid(d["parameters"])
        ind1 = Int(floor(myGrid.Nx / 2))
        ind2 = Int(floor(myGrid.Ny / 2))
        #field = [sum(abs.(VorticitySoA(myGrid, d["velocityArray"][t], d["parameters"]["bcLB"]))) for t = t1:t2)]
        field = [VorticitySoA(myGrid, d["velocityArray"][t], d["parameters"]["bcLB_mX"][1:3], d["parameters"]["bcLB_mY"][1:3])[ind1,ind2] for t in t1:t2]
        return TimeAverageFT(field, window, sep)
    end

    function DetectInstability(d, divide)
        myGrid = GetGrid(d["parameters"])
        field = [sum(abs.(VorticitySoA(myGrid, d["velocityArray"][t], d["parameters"]["bcLB_mX"][1:3], d["parameters"]["bcLB_mY"][1:3]))) for t = 1:length(d["velocityArray"])]
        f1m = maximum(field[1:divide])
        f2m = maximum(field[divide:end])
    	inst = false
    	if f2m > f1m
    	   inst = true
    	end
    	density = d["densityArray"][end]
    	if !inst
    	   if isnan(density.Values[30,30])
    	      inst = true
    	   end
    	end

        return inst
    end

    function VorticityAtPoint(d, ind1, ind2)
        myGrid = GetGrid(d["parameters"])
        vor = [VorticitySoA(myGrid, d["velocityArray"][t], d["parameters"]["bcLB_mX"][1:3], d["parameters"]["bcLB_mY"][1:3])[ind1, ind2] for t = 1:length(d["velocityArray"])]
        timeDomain = GetTimeDomain(d["parameters"], 1)
        return [timeDomain, vor]
    end

    function VelocityAtPoint(d, ind1, ind2)
        vel = [[d["velocityArray"][t].XValues[ind1, ind2], d["velocityArray"][t].YValues[ind1, ind2]] for t = 1:length(d["velocityArray"])]
        return [d["parameters"], vel]
    end

    function SigmaVEAtPoint(d, ind1, ind2)
        sig = [[d["sigmaVEArray"][t].XXValues[ind1, ind2], d["sigmaVEArray"][t].XYValues[ind1, ind2], d["sigmaVEArray"][t].YXValues[ind1, ind2], d["sigmaVEArray"][t].YYValues[ind1, ind2]] for t = 1:length(d["velocityArray"])]
        return [d["parameters"], sig]
    end

    function TotalAbsVorticity(d)
        myGrid = GetGrid(d["parameters"])
        vor = [sum(abs.(VorticitySoA(myGrid, d["velocityArray"][t], d["parameters"]["bcLB_mX"][1:3], d["parameters"]["bcLB_mY"][1:3]))) for t = 1:length(d["velocityArray"])]
        timeDomain = GetTimeDomain(d["parameters"], 0)
        return [timeDomain, vor]
    end

    function MinimumCorrelationSeries(d, offset = 25)
        myGrid = GetGrid(d["parameters"])
        vor = [VorticitySoA(myGrid, d["velocityArray"][t], d["parameters"]["bcLB_mX"][1:3], d["parameters"]["bcLB_mY"][1:3]) for t = 1:length(d["velocityArray"])]
        minArray = []
        for t = 1:length(vor)
            corr = CorrelationFunction(vor[t])
            deriv = (circshift(corr, 1) - circshift(corr, -1))[offset:end-1] # neg deriv
            push!(minArray, (findfirst(deriv .< 0) + offset) * d["parameters"]["Cx"])
        end
        timeDomain = GetTimeDomain(d["parameters"], 1)
        return [timeDomain, minArray]
    end

    function DivSTimePoint(d, time)
        myGrid = GetGrid(d["parameters"])
        sigmaVESoA = deepcopy(d["sigmaVEArray"][time])
        divS = DivS(myGrid, sigmaVESoA, d["parameters"]["bcVE"])
        return divS
        
    end

    function AverageVelocityCorrelationFunction(d, dsFac; t1 = 1, skip = 1)

        velocityArray = d["velocityArray"]
        N = length(velocityArray)
        uTest = velocityArray[1].XValues
        shp = size(uTest)
        rMax = Int(shp[2] / dsFac)
        count = 0

        for t = t1:skip:N 
            cofrT = VelocityCorrelationFunction(velocityArray[t].XValues, velocityArray[t].YValues, dsFac, rMax)
            if t == t1
                global cofrAv = cofrT
            else 
                cofrAv .+= cofrT 
            end 
            count += 1
        end 
        
        return cofrAv ./ count
    end 

    function AverageVorticityCorrelationFunction(d, dsFac; t1 = 1, skip = 1)

        velocityArray = d["velocityArray"]
        myGrid = GetGrid(d["parameters"])
        N = length(velocityArray)
        uTest = velocityArray[1].XValues
        shp = size(uTest)
        rMax = Int(shp[2] / dsFac)
        count = 0

        for t = t1:skip:N 
            w = VorticitySoA(myGrid, velocityArray[t], d["parameters"]["bcLB_mX"], d["parameters"]["bcLB_mY"])
            cofwT = VorticityCorrelationFunction(w, dsFac, rMax)
            if t == t1
                global cofwAv = cofwT
            else 
                cofwAv .+= cofwT 
            end 
            count += 1
        end 
        
        return cofwAv ./ count
    end 

    function RMSVelocity(d; t1 = 1)

        velocityArray = d["velocityArray"][t1:end]
        normList = NormOfVectorSoAList(velocityArray)
        rms = RootMeanSquared(normList)

        return rms
    end

    function DefectDensityNematic(d; t1 = 1)

        nps = []
        nms = []
        nematicArray = d["nematicArray"][t1:end]
        myGrid = GetGrid(d["parameters"])

        for t = 1:length(nematicArray)
            dirSoA = BerisEdwards.GetDirectorFromTensorSoA(myGrid, nematicArray[t]) 
            wna = WindingNumbersNematic(myGrid, dirSoA)
            (xsp, ysp, xsm, ysm) =  Analysis.CreateScatterFromWNA(wna, 1)
            push!(nps, length(xsp))
            push!(nms, length(xsm))
        end 

        return (nps, nms)

    end 

    function EvaluateSpeedAtReadOuts(grid, velocitySoA, pointList)
        
        speedVals = sqrt.(DotProductOnSoA2D(grid, velocitySoA, velocitySoA).Values)
        speedList = []
        for point in pointList
            push!(speedList, speedVals[point[1], point[2]])
        end 
        return speedList 
    end

    function EvaluateSpeedAtReadOutVertStrips(grid, velocitySoA, pointList, a, b)
        
        speedVals = sqrt.(DotProductOnSoA2D(grid, velocitySoA, velocitySoA).Values)
        speedList = []
        for point in pointList
            p1 = point[1]
            p2 = point[2]
            speed = 0.0
            for y = p2-b : p2 + b
                speed += sum(speedVals[point[1]-a : point[1]+a, mod1(y, grid.Ny)])
            end
            push!(speedList, speed)
        end 
        return speedList 
    end

    function EvaluateSpeedAtReadOutHorStrips(grid, velocitySoA, pointList, a, b)
        
        speedVals = sqrt.(DotProductOnSoA2D(grid, velocitySoA, velocitySoA).Values)
        speedList = []
        for point in pointList
            p1 = point[1]
            p2 = point[2]
            speed = 0.0
            for x = p1-a : p1 + a
                speed += sum(speedVals[mod1(x, grid.Nx), p2-b : p2+b])
            end
            push!(speedList, speed)
        end 
        return speedList 
    end

end # module
