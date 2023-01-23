module Visualization

    include("MathFunctions.jl")
    using .MathFunctions

    include("SharedStructs.jl")
    using .SharedStructs

    include("Analysis.jl")

    include("LatBoltz.jl")
    include("ImmBound.jl")
    include("BerisEdwards.jl")

    using Interpolations
    using GLMakie
    using FFTW
    using Statistics
    using ColorTypes

    ########################################################################################
    #
    #                               Helper analysis functions
    #
    ########################################################################################

    function GetNeighborhood(grid, i, j)
    
        if i == 1 
            is = [grid.Nx, 1, 2]
        elseif i == grid.Nx
            is = [grid.Nx-1, grid.Nx, 1]
        else
            is = [i-1, i, i+1]
        end
        if j == 1 
            js = [grid.Ny, 1, 2]
        elseif j == grid.Ny
            js = [grid.Ny-1, grid.Ny, 1]
        else
            js = [j-1, j, j+1]
        end 
        return (is, js)
    end

    function OrientDirectorField!(grid, dirSoA)

        sz = size(dirSoA.XValues)
        for i = 1:sz[2]
            for j = 1:sz[1]
                (is, js) = GetNeighborhood(grid, i, j)
                xl = deepcopy(dirSoA.XValues[i,j])
                yl = deepcopy(dirSoA.YValues[i,j])
                
                x = dirSoA.XValues[is[1],js[1]] + dirSoA.XValues[is[2],js[1]] + dirSoA.XValues[is[3],js[1]] +
                    dirSoA.XValues[is[1],js[2]] + 0*dirSoA.XValues[is[2],js[2]] + dirSoA.XValues[is[3],js[2]] +
                    dirSoA.XValues[is[1],js[3]] + dirSoA.XValues[is[2],js[3]] + dirSoA.XValues[is[3],js[3]]
                y = dirSoA.YValues[is[1],js[1]] + dirSoA.YValues[is[2],js[1]] + dirSoA.YValues[is[3],js[1]] +
                    dirSoA.YValues[is[1],js[2]] + dirSoA.YValues[is[2],js[2]] + dirSoA.YValues[is[3],js[2]] +
                    dirSoA.YValues[is[1],js[3]] + dirSoA.YValues[is[2],js[3]] + dirSoA.YValues[is[3],js[3]]
                
                if x * xl + y * yl < 0
                    dirSoA.XValues[i,j] = -1*xl
                    dirSoA.YValues[i,j] = -1*yl
                end 
            end
        end
    end

    ########################################################################################
    #
    #                               Helper visualization functions
    #
    ########################################################################################


    function GetDens(t, grid, col, densityArray, directorArray, nematicArray, phibArray, phiuArray, sigmaVEArray, velocityArray, parameters)
        if col == "dens" # density
            dens = Observable(densityArray[t].Values)
        elseif col == "pol" # P orientation
            dens = Observable(Analysis.thetaFromPVecSoA(directorArray[t].XValues, directorArray[t].YValues))
        elseif col == "nemO" # Q magnitude
            dens = Observable((BerisEdwards.GetMagntiudeFromTensor2DSoA(grid, nematicArray[t])).Values)
        elseif col == "nemD" # Q orientation
            vecSoA = BerisEdwards.GetUnitDirectorFromTensor2DSoA(grid, nematicArray[t])
            dens = Observable(Analysis.thetaFromPVecSoA(vecSoA.XValues, vecSoA.YValues))
        elseif col == "phib" # phib
            dens = Observable(phibArray[t].Values)
        elseif col == "phiu" # phiu
            dens = Observable(phiuArray[t].Values)
        elseif col == "phitot" # phiu + phib
            dens = Observable(phiuArray[t].Values .+ phibArray[t].Values)
        elseif col == "tor" # torque
            dens = Observable(Analysis.TorqueDensitySoA(sigmaVEArray[t].YXValues, sigmaVEArray[t].XYValues))
        elseif col == "vor" # vorticity
            dens = Observable(Analysis.VorticitySoA(grid, velocityArray[t], parameters["bcLB_mX"][1:3], parameters["bcLB_mY"][1:3]))
        elseif col == "speed" # velocity magnitude
            dens = Observable(sqrt.(DotProductOnSoA2D(grid, velocityArray[t], velocityArray[t]).Values))
        elseif col == "diss" # viscous dissipation
            dens = Observable(Analysis.ViscousDissipation(grid,  velocityArray[t], parameters["bcLB_mX"][1:3], parameters["bcLB_mY"][1:3]).Values)
        elseif col == "sxy" # element of sigmaVE
            dens = Observable(sigmaVEArray[t].XYValues)
        elseif col == "tr" # trace of sigmaVE
            dens = Observable(sigmaVEArray[t].XXValues .+ 0 .* sigmaVEArray[t].YYValues)
        elseif col == "vef" # magnitude of forceVE
            viscoelasticForceSoA = DivTensorOnSoA2D(grid, MatrixTransposeOnSoA2D(grid, sigmaVEArray[t]), BCDerivDict[parameters["bcVE_X"]], BCDerivDict[parameters["bcVE_Y"]])
            dens = Observable(VectorNormSoA(viscoelasticForceSoA).Values)
        elseif col == "divV" # velocity divergence
            dens = Observable(DivV(grid, velocityArray[t], parameters["bcLB_mX"][1:3], parameters["bcLB_mY"][1:3]))
        else
            error("Color argument not recognized.")
        end
        return dens
    end

    function UpdateDens!(t, grid, dens, col, densityArray, directorArray, nematicArray, phibArray, phiuArray, sigmaVEArray, velocityArray, parameters)
        if col == "dens"
            dens[] = densityArray[t].Values
        elseif col == "pol"
            dens[] = Analysis.thetaFromPVecSoA(directorArray[t].XValues, directorArray[t].YValues)
        elseif col == "nemO"
            dens[] = (BerisEdwards.GetMagntiudeFromTensor2DSoA(grid, nematicArray[t])).Values
        elseif col == "nemD"
            vecSoA = BerisEdwards.GetUnitDirectorFromTensor2DSoA(grid, nematicArray[t])
            dens[] = Analysis.thetaFromPVecSoA(vecSoA.XValues, vecSoA.YValues)
        elseif col == "phib"
            dens[] = phibArray[t].Values
        elseif col == "phiu"
            dens[] = phiuArray[t].Values
        elseif col == "phitot"
            dens[] = phiuArray[t].Values .+ phibArray[t].Values
        elseif col == "tor"
            dens[] = Analysis.TorqueDensitySoA(sigmaVEArray[t].YXValues, sigmaVEArray[t].XYValues)
        elseif col == "vor"
            dens[] = Analysis.VorticitySoA(grid, velocityArray[t], parameters["bcLB_mX"][1:3], parameters["bcLB_mY"][1:3]) 
        elseif col == "divV"
            dens[] = DivV(grid, velocityArray[t], parameters["bcLB_mX"][1:3], parameters["bcLB_mY"][1:3])
        elseif col == "vef" # magnitude of forceVE
            viscoelasticForceSoA = DivTensorOnSoA2D(grid, MatrixTransposeOnSoA2D(grid, sigmaVEArray[t]), BCDerivDict[parameters["bcVE_X"]], BCDerivDict[parameters["bcVE_Y"]])
            dens[] = VectorNormSoA(viscoelasticForceSoA).Values
        end
    end


    function GetduVecs(t, grid, arrows, arrowFac, velocityArray, directorArray, nematicArray, sigmaVEArray, parameters)
        if arrows == "vel"
            duVecs = Observable(arrowFac * [Point2(velocityArray[t].XValues[i,j], velocityArray[t].YValues[i,j]) for i = 1:grid.Nx, j = 1:grid.Ny])
        elseif arrows == "pol"
            duVecs = Observable(arrowFac * [Point2(directorArray[t].XValues[i,j], directorArray[t].YValues[i,j]) for i = 1:grid.Nx, j = 1:grid.Ny])
        elseif arrows == "nem"
            vecSoA = BerisEdwards.GetUnitDirectorFromTensor2DSoA(grid, nematicArray[t])
            OrientDirectorField!(grid, vecSoA)
            us = []
            vs = []
            for i = 1:grid.Nx
                for j = 1:grid.Ny
                    push!(us, arrowFac * vecSoA.XValues[i,j])
                    push!(vs, arrowFac * vecSoA.YValues[i,j])
                end 
            end 
            #duVecs = Observable(arrowFac * [Point2(vecSoA.XValues[i,j], vecSoA.YValues[i,j]) for i = 1:grid.Nx, j = 1:grid.Ny])
            duVecs = [Observable(us), Observable(vs)]
        elseif arrows == "divS"
            divS = Analysis.DivS(grid, sigmaVEArray[t], parameters["bcVE_X"], parameters["bcVE_Y"])
            duVecs = Observable(arrowFac * [Point2(divS.XValues[i,j], divS.YValues[i,j]) for i = 1:grid.Nx, j = 1:grid.Ny])
        elseif arrows == "diva"
            if parameters["BEModel"] == "P"
                aSoA = BerisEdwards.ActiveStressTensorP2DSoA(directorArray[t], grid, BerisEdwards.ActiveParams(parameters["zeta"]))
            else 
                aSoA = BerisEdwards.ActiveStressTensorQ2DSoA(nematicArray[t], grid, BerisEdwards.ActiveParams(parameters["zeta"]))
            end
            diva = Analysis.DivS(grid, MatrixTransposeOnSoA2D(grid, aSoA), parameters["bcBE_X"], parameters["bcBE_Y"])
            duVecs = Observable(arrowFac * [Point2(diva.XValues[i,j], diva.YValues[i,j]) for i = 1:grid.Nx, j = 1:grid.Ny])
        elseif arrows == "divE"
            if parameters["BEModel"] == "P"
                beParams = BerisEdwards.BEPParams(parameters["xiBE"], parameters["alphaBE"], parameters["betaBE"], parameters["kappaBE"], parameters["GammaBE"], parameters["tauBE"])
                eSoA = BerisEdwards.EricksenStressTensorP2DSoA(grid, directorArray[t], beParams, parameters["bcBE_X"], parameters["bcBE_Y"])
            else 
                beParams = BerisEdwards.BEQParams(parameters["lambdaBE"], parameters["A0BE"], parameters["UBE"], parameters["LBE"], parameters["GammaBE"])
                eSoA = BerisEdwards.EricksenStressTensorQ2DSoA(grid, nematicArray[t], beParams, parameters["bcBE_X"], parameters["bcBE_Y"])
            end
            dive = Analysis.DivS(grid, MatrixTransposeOnSoA2D(grid, eSoA), parameters["bcBE_X"], parameters["bcBE_Y"])
            duVecs = Observable(arrowFac * [Point2(dive.XValues[i,j], dive.YValues[i,j]) for i = 1:grid.Nx, j = 1:grid.Ny])
        else
            error("Arrow argument not recognized.")
        end
        return duVecs
    end

    function UpdateDuVecs!(t, grid, duVecs, arrows, arrowFac, velocityArray, directorArray, nematicArray, sigmaVEArray, parameters)
        if arrows == "vel"
            duVecs[] = [arrowFac * Point2(velocityArray[t].XValues[i,j], velocityArray[t].YValues[i,j]) for i = 1:grid.Nx, j = 1:grid.Ny]
        elseif arrows == "pol"
            duVecs[] = [arrowFac * Point2(directorArray[t].XValues[i,j], directorArray[t].YValues[i,j]) for i = 1:grid.Nx, j = 1:grid.Ny]
        elseif arrows == "nem"
            vecSoA = BerisEdwards.GetUnitDirectorFromTensor2DSoA(grid, nematicArray[t])
            us = []
            vs = []
            for i = 1:grid.Nx
                for j = 1:grid.Ny
                    push!(us, arrowFac * vecSoA.XValues[i,j])
                    push!(vs, arrowFac * vecSoA.YValues[i,j])
                end 
            end 
            duVecs[1].val = us
            duVecs[2].val = vs
        elseif arrows == "divS"
            divS = Analysis.DivS(grid, sigmaVEArray[t], parameters["bcVE_X"], parameters["bcVE_Y"])
            duVecs[] = arrowFac * [Point2(divS.XValues[i,j], divS.YValues[i,j]) for i = 1:grid.Nx, j = 1:grid.Ny]
        elseif arrows == "diva"
            if parameters["BEModel"] == "P"
                aSoA = BerisEdwards.ActiveStressTensorP2DSoA(directorArray[t], grid, BerisEdwards.ActiveParams(parameters["zeta"]))
            else 
                aSoA = BerisEdwards.ActiveStressTensorQ2DSoA(nematicArray[t], grid, BerisEdwards.ActiveParams(parameters["zeta"]))
            end
            diva = Analysis.DivS(grid, MatrixTransposeOnSoA2D(grid, aSoA), parameters["bcBE_X"], parameters["bcBE_Y"])
            duVecs[] = arrowFac * [Point2(diva.XValues[i,j], diva.YValues[i,j]) for i = 1:grid.Nx, j = 1:grid.Ny]
        elseif arrows == "divE"
            if parameters["BEModel"] == "P"
                beParams = BerisEdwards.BEPParams(parameters["xiBE"], parameters["alphaBE"], parameters["betaBE"], parameters["kappaBE"], parameters["GammaBE"], parameters["tauBE"])
                eSoA = BerisEdwards.EricksenStressTensorP2DSoA(grid, directorArray[t], beParams, parameters["bcBE_X"], parameters["bcBE_Y"])
            else 
                beParams = BerisEdwards.BEQParams(parameters["lambdaBE"], parameters["A0BE"], parameters["UBE"], parameters["LBE"], parameters["GammaBE"])
                eSoA = BerisEdwards.EricksenStressTensorQ2DSoA(grid, nematicArray[t], beParams, parameters["bcBE_X"], parameters["bcBE_Y"])
            end
            dive = Analysis.DivS(grid, MatrixTransposeOnSoA2D(grid, eSoA), parameters["bcBE_X"], parameters["bcBE_Y"])
            duVecs[] = arrowFac * [Point2(dive.XValues[i,j], dive.YValues[i,j]) for i = 1:grid.Nx, j = 1:grid.Ny]
        end
    end

    function GetPts(grid)
        xs = map(x-> LatBoltz.GetGridPoint2D(x,1,grid)[1], 1:grid.Nx)
        ys = map(y-> LatBoltz.GetGridPoint2D(1,y,grid)[2], 1:grid.Ny)
        return (xs, ys, vec(Point2.(xs, ys')))
    end

    function DecimateArrows!(grid, pts, duVecs, skip, halfOff = false)
        if (!halfOff)
            nxs = map(x-> LatBoltz.GetGridPoint2D(x,1,grid)[1], 1:skip:grid.Nx)
            nys = map(y-> LatBoltz.GetGridPoint2D(1,y,grid)[2], 1:skip:grid.Ny)
            duVecs[] = duVecs[][1:skip:end, 1:skip:end]
            return vec(Point2.(nxs, nys'))
        else 
            nxs = []
            nys = []
            nus = []
            nvs = []
            for i = 1:skip:grid.Nx
                for j = 1:skip:grid.Ny
                    push!(nxs, LatBoltz.GetGridPoint2D(i,j,grid)[1] - 0.5 * duVecs[1][][(i-1) * grid.Nx + j])
                    push!(nys, LatBoltz.GetGridPoint2D(i,j,grid)[2] - 0.5 * duVecs[2][][(i-1) * grid.Nx + j])
                    push!(nus, duVecs[1][][(i-1) * grid.Nx + j])
                    push!(nvs, duVecs[2][][(i-1) * grid.Nx + j])
                end 
            end 
            duVecs[1].val = nus
            duVecs[2].val = nvs
            return [nxs, nys]
        end 
    end

    function DrawHeatMap!(scene, xs, ys, dens, col, colorrange)
        if col == "pol"
            #GLMakie.heatmap!(scene, xs, ys, dens, colorrange = (-pi/2, pi/2), colormap = range(LCHab(70,70,0), stop=LCHab(70,70,360), length=90))
            GLMakie.heatmap!(scene, xs, ys, dens, colorrange = (-pi, pi), colormap = range(LCHab(70,70,0), stop=LCHab(70,70,360), length=90))
        elseif col == "nemD"
            GLMakie.heatmap!(scene, xs, ys, dens, colorrange = (0, pi), colormap = range(LCHab(70,70,0), stop=LCHab(70,70,360), length=90))
        else
            GLMakie.heatmap!(scene, xs, ys, dens, colorrange=colorrange) # (-5e-4, 5e-4)
        end
    end

    function DrawHeatMapNS!(xs, ys, dens, col, colorrange)
        if col == "pol"
            hm = GLMakie.heatmap!(xs, ys, dens, colorrange = (pi/4 - 0.5 , pi/4 + 0.5), colormap = range(LCHab(70,70,0), stop=LCHab(70,70,360), length=90))
            #hm = GLMakie.heatmap!(xs, ys, dens, colorrange = (-pi, pi), colormap = range(LCHab(70,70,0), stop=LCHab(70,70,360), length=90))
        elseif col == "nemD"
            hm = GLMakie.heatmap!(xs, ys, dens, colorrange = (0, pi), colormap = range(LCHab(70,70,0), stop=LCHab(70,70,360), length=90))
        else
            hm = GLMakie.heatmap!(xs, ys, dens, colorrange=colorrange) # (-5e-4, 5e-4)
        end
        return hm
    end

    function SetLims!(scene, grid)
        #Makie.xlims!(scene, 0, grid.Nx+1)
        #Makie.ylims!(scene, 0, grid.Ny+1)
        Makie.xlims!(0, grid.Nx+1)
        Makie.ylims!(0, grid.Ny+1)
    end

    function SetLimsNS!(ax, grid)
        Makie.xlims!(ax, 0, grid.Nx+1)
        Makie.ylims!(ax, 0, grid.Ny+1)
    end

    function GetStreamFunction(grid, velocityArray)
        xObservables = 1:1:grid.Nx
        yObservables = 1:1:grid.Ny
        Observables = (xObservables, yObservables)
        interpolationXFuncList = [interpolate(Observables, velocityArray[t].XValues, Gridded(Linear())) for t in 1:length(velocityArray)]
        interpolationYFuncList = [interpolate(Observables, velocityArray[t].YValues, Gridded(Linear())) for t in 1:length(velocityArray)]
        streamFunction(x::Point2, t) = Point2(
            interpolationXFuncList[t](x[1], x[2]),
            interpolationYFuncList[t](x[1], x[2]))
        return streamFunction
    end

    function GetIBPositions(t, ObservablePositionArray, rExtArray)
        posX = Observable(vcat([n[1] for n in ObservablePositionArray[t]], ObservablePositionArray[t][1][1]))
        posY = Observable(vcat([n[2] for n in ObservablePositionArray[t]], ObservablePositionArray[t][1][2]))
        extX = Observable([rExtArray[t][1]])
        extY = Observable([rExtArray[t][2]])
        comX = Observable([ImmBound.CenterOfMassPos(ObservablePositionArray[t])[1]])
        comY = Observable([ImmBound.CenterOfMassPos(ObservablePositionArray[t])[2]])
        comXK = [ImmBound.CenterOfMassPos(ObservablePositionArray[1])[1]]
        comYK = [ImmBound.CenterOfMassPos(ObservablePositionArray[1])[2]]
        tComX = [ImmBound.CenterOfMassPos(ObservablePositionArray[i])[1] for i = 1:t]
        tComY = [ImmBound.CenterOfMassPos(ObservablePositionArray[i])[2] for i = 1:t]
        tComX_n = Observable(tComX)
        tComY_n = Observable(tComY)
        pos1X = Observable([ObservablePositionArray[t][1][1]])
        pos1Y = Observable([ObservablePositionArray[t][1][2]])

        IBPosDict = Dict(
        "posX" => posX,
        "posY" => posY,
        "extX" => extX,
        "extY" => extY,
        "comX" => comX,
        "comY" => comY,
        "comXK" => comXK,
        "comYK" => comYK,
        "tComX" => tComX,
        "tComY" => tComY,
        "tComX_n" => tComX_n,
        "tComY_n" => tComY_n,
        "pos1X" => pos1X,
        "pos1Y" => pos1Y,
        )

        return IBPosDict
    end

    function UpdateIBPositions!(t, IBPosDict, ObservablePositionArray, rExtArray)
        newIBPosDict = GetIBPositions(t, ObservablePositionArray, rExtArray)

        IBPosDict["posX"][] = newIBPosDict["posX"][]
        IBPosDict["posY"][] = newIBPosDict["posY"][]
        IBPosDict["extX"][] = newIBPosDict["extX"][]
        IBPosDict["extY"][] = newIBPosDict["extY"][]
        IBPosDict["comX"][] = newIBPosDict["comX"][]
        IBPosDict["comY"][] = newIBPosDict["comY"][]
        IBPosDict["comXK"] = newIBPosDict["comXK"]
        IBPosDict["comYK"] = newIBPosDict["comYK"]
        IBPosDict["tComX"] = newIBPosDict["tComX"]
        IBPosDict["tComY"] = newIBPosDict["tComY"]
        IBPosDict["tComX_n"].val = newIBPosDict["tComX"]
        IBPosDict["tComY_n"][] = newIBPosDict["tComY"]
        IBPosDict["pos1X"][] = newIBPosDict["pos1X"][]
        IBPosDict["pos1Y"][] = newIBPosDict["pos1Y"][]

    end

    function UpdateIBColors!(t, eColor, eColor1, eColor2, parameters, kArray)
        if kArray[t] > (parameters["kExt"] / 2)
            eColor[] = eColor1
        else
            eColor[] = eColor2
        end
    end

    function DrawIBPositions!(scene, IBPosDict, linewidth, markersize, tracking, p1Bool, iColor, eColor, cColor, oColor, tColor, p1Color)
        if tracking
            GLMakie.lines!(scene, IBPosDict["tComX_n"], IBPosDict["tComY_n"], color = tColor, linewidth = linewidth / 2)
        end
        GLMakie.lines!(scene, IBPosDict["posX"], IBPosDict["posY"], color = iColor, linewidth = linewidth)
        GLMakie.scatter!(scene, IBPosDict["extX"], IBPosDict["extY"], markersize = markersize, color = eColor)
        GLMakie.scatter!(scene, IBPosDict["comX"], IBPosDict["comY"], markersize = markersize, color = cColor)
        GLMakie.scatter!(scene, IBPosDict["comXK"], IBPosDict["comYK"], markersize = markersize, color = oColor)
        if p1Bool
            GLMakie.scatter!(scene, IBPosDict["pos1X"], IBPosDict["pos1Y"], markersize = markersize, color = p1Color)
        end
    end

    function DrawIBPositionsNS!(IBPosDict, linewidth, markersize, tracking, p1Bool, iColor, eColor, cColor, oColor, tColor, p1Color)
        if tracking
            GLMakie.lines!(IBPosDict["tComX_n"], IBPosDict["tComY_n"], color = tColor, linewidth = linewidth / 2)
        end
        GLMakie.lines!(IBPosDict["posX"], IBPosDict["posY"], color = iColor, linewidth = linewidth)
        GLMakie.scatter!(IBPosDict["extX"], IBPosDict["extY"], markersize = markersize, color = eColor)
        GLMakie.scatter!(IBPosDict["comX"], IBPosDict["comY"], markersize = markersize, color = cColor)
        #GLMakie.scatter!(IBPosDict["comXK"], IBPosDict["comYK"], markersize = markersize, color = oColor)
        if p1Bool
            GLMakie.scatter!(IBPosDict["pos1X"], IBPosDict["pos1Y"], markersize = markersize, color = p1Color)
        end
    end

    function DrawTextBox!(str ;textPosition = [5, 76], boff = 1.2, loff = 0.1, w = 20, h = 3, sw = 4, ts = 40)
        poly!(Rect(textPosition[1] - boff, textPosition[2] - loff, w, h), color = :white,
            strokecolor = :black, strokewidth = sw)
        GLMakie.text!(str,
            position = (textPosition[1], textPosition[2]), textsize = ts)
    end

    function NegLog(x)
        if x < 0
            return -log(-x)
        elseif x == 0
            return 0
        else 
            return log(x)
        end 
    end
    ########################################################################################
    #
    #                               Main visualization functions
    #
    ########################################################################################


    function AnimateTrajectoryArrows(parameters, densityArray, velocityArray, directorArray, nematicArray, phibArray, phiuArray, sigmaVEArray,
        ObservablePositionArray, rExtArray, kArray;
        arrows = "vel", col = "dens", defects = "none", arrowFac = 1e0, arrowSkip = 1, arrowWidth = 1, arrowHead = 1, arrowOp = 1,
        colorrange = (0.99, 1.001), recording = false, moviePath = "", sleepLength = 0.1, res = 2000, cScaleFac = 1,
        IBBool = false, tracking = false, p1Bool = false, IBlinewidth = 6.0, IBmarkersize = 15,
        iColor = :black, eColor1 = :red, eColor2 = :red, cColor = :black, oColor = :orange, tColor = :white, p1Color = :orange,
        fontsize = 35, ticks = [1], tickLabels = [""], xlabel = "", ylabel = "", clabel = "")

        grid = Analysis.GetGrid(parameters)
        (xs, ys, pts) = GetPts(grid)

        if IBBool
            IBPosDict = GetIBPositions(1, ObservablePositionArray, rExtArray)
        end

        f = Figure(resolution = (res, res), fontsize = fontsize, font = "Arial", dpi = 300)
        #tickLabels = [string((t - ticks[Int(ceil(length(ticks)/2))]) * parameters["Cx"] * fac ) for t in ticks]
        ax = Axis(f[1, 1],
            xlabel = xlabel,
            ylabel = ylabel,
            xticks = (ticks, tickLabels),
            yticks = (ticks, tickLabels)
        )

        dens = GetDens(1, grid, col, densityArray, directorArray, nematicArray, phibArray, phiuArray, sigmaVEArray, velocityArray, parameters)
        dens[] = dens[] .* cScaleFac
        hm = DrawHeatMapNS!(xs, ys, dens, col, colorrange)
        SetLimsNS!(ax, grid)
        GLMakie.Colorbar(f[1,2], hm, label = clabel)
        duVecs = GetduVecs(1, grid, arrows, arrowFac, velocityArray, directorArray, nematicArray, sigmaVEArray, parameters)
        if arrows == "nem"
            halfOff = true 
        else
            halfOff = false 
        end
        npts = DecimateArrows!(grid, pts, duVecs, arrowSkip, halfOff)
        if arrows != "nem"
            arrows!(npts, duVecs, linewidth = arrowWidth, arrowsize = arrowHead, color = (:black, arrowOp), aspect = 1)
        else 
            apts = [Observable(npts[1]), Observable(npts[2])]
            arrows!(apts[1], apts[2], duVecs[1], duVecs[2], linewidth = arrowWidth, arrowsize = arrowHead, color = (:black, arrowOp), aspect = 1)
        end 
        if IBBool
            eColor = Observable(eColor1)
            DrawIBPositionsNS!(IBPosDict, IBlinewidth, IBmarkersize, tracking, p1Bool, iColor, eColor, cColor, oColor, tColor, p1Color)
        end

        if defects != "none"
            if defects == "Q"
                dirSoA = BerisEdwards.GetUnitDirectorFromTensor2DSoA(grid, nematicArray[1]) 
                wna = Analysis.WindingNumbersNematic(grid, dirSoA)
                cut = 0.4
            else
                dirSoA = directorArray[1]
                wna = Analysis.WindingNumbers(grid, dirSoA)
                cut = 0.4
            end
            (xspV, yspV, xsmV, ysmV) = Analysis.CreateScatterFromWNA(wna, 1, cut)
            xsp = Observable(xspV)
            ysp = Observable(yspV)
            xsm = Observable(xsmV)
            ysm = Observable(ysmV)
            GLMakie.scatter!(xsp, ysp, markersize = 20, marker = :circle, color = :black)
            GLMakie.scatter!(xsm, ysm, markersize = 20, marker = :circle, color = :white)
        end

        if recording
            record(f, moviePath, 1:length(velocityArray)) do t
                UpdateDens!(t, grid, dens, col, densityArray, directorArray, nematicArray, phibArray, phiuArray, sigmaVEArray, velocityArray, parameters)
                dens[] = dens[] .* cScaleFac
                UpdateDuVecs!(t, grid, duVecs, arrows, arrowFac, velocityArray, directorArray, nematicArray, sigmaVEArray, parameters)
                nApts = DecimateArrows!(grid, pts, duVecs, arrowSkip, halfOff)
                if arrows == "nem"
                    apts[1][] = nApts[1]
                    apts[2][] = nApts[2]
                end
                if defects != "none"
                    if defects == "Q"
                        dirSoA = BerisEdwards.GetUnitDirectorFromTensor2DSoA(grid, nematicArray[t]) 
                        wna = Analysis.WindingNumbersNematic(grid, dirSoA)
                        cut = 0.4
                    else
                        dirSoA = directorArray[t]
                        wna = Analysis.WindingNumbers(grid, dirSoA)
                        cut = 0.4
                    end
                    (xspV, yspV, xsmV, ysmV) = Analysis.CreateScatterFromWNA(wna, 1, cut)
                    xsp.val = xspV
                    ysp[] = yspV 
                    xsm.val = xsmV 
                    ysm[] = ysmV
                end
                if IBBool
                    UpdateIBPositions!(t, IBPosDict, ObservablePositionArray, rExtArray)
                    UpdateIBColors!(t, eColor, eColor1, eColor2, parameters, kArray)
                end
                sleep(sleepLength)
            end
        else
            for t in 1:length(velocityArray)
                UpdateDens!(t, grid, dens, col, densityArray, directorArray, nematicArray, phibArray, phiuArray, sigmaVEArray, velocityArray, parameters)
                dens[] = dens[] .* cScaleFac
                UpdateDuVecs!(t, grid, duVecs, arrows, arrowFac, velocityArray, directorArray, nematicArray, sigmaVEArray, parameters)
                nApts = DecimateArrows!(grid, pts, duVecs, arrowSkip, halfOff)
                if arrows == "nem"
                    apts[1][] = nApts[1]
                    apts[2][] = nApts[2]
                end
                if defects != "none"
                    if defects == "Q"
                        dirSoA = BerisEdwards.GetUnitDirector2DSoA(grid, nematicArray[t]) 
                        wna = Analysis.WindingNumbersNematic(grid, dirSoA)
                        cut = 0.9
                    else
                        dirSoA = directorArray[t]
                        wna = Analysis.WindingNumbers(grid, dirSoA)
                        cut = 0.4
                    end
                    (xspV, yspV, xsmV, ysmV) =  Analysis.CreateScatterFromWNA(wna, 1, cut)
                    xsp.val = xspV
                    ysp[] = yspV 
                    xsm.val = xsmV 
                    ysm[] = ysmV
                end
                if IBBool
                    UpdateIBPositions!(t, IBPosDict, ObservablePositionArray, rExtArray)
                    UpdateIBColors!(t, eColor, eColor1, eColor2, parameters, kArray)
                end
                sleep(sleepLength)
                current_figure()
            end
        end
    end # AnimateTrajectoryArrows()

    function StaticArrows(t, parameters, densityArray, velocityArray, directorArray, nematicArray, phibArray, phiuArray, sigmaVEArray,
        ObservablePositionArray, rExtArray, kArray;
        arrows = "vel", col = "dens", defects = "none", 
        arrowFac = 1e0, arrowSkip = 1, arrowWidth = 1, arrowHead = 1, arrowOp = 1,
        colorrange = (0.99, 1.001), recording = false, imagePath = "", res = 2000, cScaleFac = 1,
        IBBool = false, tracking = false, p1Bool = false, IBlinewidth = 6.0, IBmarkersize = 15,
        iColor = :black, eColor1 = :red, eColor2 = :red, cColor = :black, oColor = :orange, tColor = :white, p1Color = :orange,
        fontsize = 35, ticks = [0], tickLabels = [""], fac = 10^6, xlabel = "", ylabel = "", clabel = "",
        display = true)

        grid = Analysis.GetGrid(parameters)
        (xs, ys, pts) = GetPts(grid)

        if IBBool
            IBPosDict = GetIBPositions(t, ObservablePositionArray, rExtArray)
        end

        f = Figure(resolution = (res, res), fontsize = fontsize, font = "Arial", dpi = 300)
        #tickLabels = [string((t - ticks[Int(ceil(length(ticks)/2))]) * parameters["Cx"] * fac ) for t in ticks]
        ax = Axis(f[1, 1],
            xlabel = xlabel,
            ylabel = ylabel,
            xticks = (ticks, tickLabels),
            yticks = (ticks, tickLabels)
        )

        dens = GetDens(t, grid, col, densityArray, directorArray, nematicArray, phibArray, phiuArray, sigmaVEArray, velocityArray, parameters)
        dens[] = dens[] .* cScaleFac
        hm = DrawHeatMapNS!(xs, ys, dens, col, colorrange)
        SetLimsNS!(ax, grid)
        GLMakie.Colorbar(f[1,2], hm, label = clabel)
        duVecs = GetduVecs(t, grid, arrows, arrowFac, velocityArray, directorArray, nematicArray, sigmaVEArray, parameters)
        if arrows == "nem"
            halfOff = true 
        else
            halfOff = false 
        end
        apts = DecimateArrows!(grid, pts, duVecs, arrowSkip, halfOff)
        if arrows != "nem"
            arrows!(apts, duVecs, linewidth = arrowWidth, arrowsize = arrowHead, color = (:black, arrowOp), aspect = 1)
        else 
            arrows!(apts[1], apts[2], duVecs[1], duVecs[2], linewidth = arrowWidth, arrowsize = arrowHead, color = (:black, arrowOp), aspect = 1)
        end 
        if IBBool
            eColor = Observable(eColor1)
            DrawIBPositionsNS!(IBPosDict, IBlinewidth, IBmarkersize, tracking, p1Bool, iColor, eColor, cColor, oColor, tColor, p1Color)
        end

        if defects != "none"
            if defects == "Q"
                dirSoA = BerisEdwards.GetUnitDirectorFromTensor2DSoA(grid, nematicArray[t]) 
                wna = Analysis.WindingNumbersNematic(grid, dirSoA)
                cut = 0.4
            else
                dirSoA = directorArray[t]
                wna = Analysis.WindingNumbers(grid, dirSoA)
                cut = 0.4
            end
            (xsp, ysp, xsm, ysm) =  Analysis.CreateScatterFromWNA(wna, 1, cut)
            GLMakie.scatter!(xsp, ysp, markersize = 20, marker = :square, color = :black)
            GLMakie.scatter!(xsm, ysm, markersize = 20, marker = :square, color = :white)
        end


        if recording
            save(imagePath, f, px_per_unit = 3)
        end

        if display
            GLMakie.current_figure()
        else
            return ax
        end

    end # StaticArrowsNS()


    function AnimateTrajectoryStreamlines(parameters, densityArray, velocityArray, directorArray, nematicArray, phibArray, phiuArray, sigmaVEArray,
        ObservablePositionArray, rExtArray, kArray;
        col = "dens", colorrange = (0.99, 1.001), recording = false, moviePath = "", sleepLength = 0.1, res = 2000, cScaleFac = 1,
        streamArrowSize = 0.1, streamLineWidth = 4.0, streamDensity = 3.0, streamStepSize = 0.01,
        IBBool = false, tracking = false, p1Bool = false, IBlinewidth = 6.0, IBmarkersize = 15,
        iColor = :black, eColor1 = :red, eColor2 = :red, cColor = :black, oColor = :orange, tColor = :white, p1Color = :orange,
        fontsize = 35, ticks = [1], tickLabels = [""], fac = 10^6, xlabel = "", ylabel = "", clabel = "",
        display = true)

        grid = Analysis.GetGrid(parameters)
        (xs, ys, pts) = GetPts(grid)

        if IBBool
            IBPosDict = GetIBPositions(1, ObservablePositionArray, rExtArray)
        end

        fx = grid.Nx / grid.Ny
        f = Figure(resolution = (res*fx, res), fontsize = fontsize, font = "Arial", dpi = 300)
        #tickLabels = [string((t - ticks[Int(ceil(length(ticks)/2))]) * parameters["Cx"] * fac ) for t in ticks]
        ax = Axis(f[1, 1],
            xlabel = xlabel,
            ylabel = ylabel,
            xticks = (ticks, tickLabels),
            yticks = (ticks, tickLabels)
        )

        dens = GetDens(1, grid, col, densityArray, directorArray, nematicArray, phibArray, phiuArray, sigmaVEArray, velocityArray, parameters)
        dens[] = dens[] .* cScaleFac
        #dens[] = (dens[] .- 1) .* cScaleFac
        hm = DrawHeatMapNS!(xs, ys, dens, col, colorrange)
        SetLimsNS!(ax, grid)
        GLMakie.Colorbar(f[1,2], hm, label = clabel)
        streamFunction = GetStreamFunction(grid, velocityArray)
        ff = Observable(Base.Fix2(streamFunction, 1))
        GLMakie.streamplot!(ff, 1:.1:grid.Nx, 1:.1:grid.Ny; arrow_size = streamArrowSize, linewidth = streamLineWidth,
            density = streamDensity, stepsize = streamStepSize, colormap = :grays)
        if IBBool
            eColor = Observable(eColor1)
            DrawIBPositionsNS!(scene, IBPosDict, IBlinewidth, IBmarkersize, tracking, p1Bool, iColor, eColor, cColor, oColor, tColor, p1Color)
        end

        if recording
            record(f, moviePath, 1:length(velocityArray)) do t
                UpdateDens!(t, grid, dens, col, densityArray, directorArray, nematicArray, phibArray, phiuArray, sigmaVEArray, velocityArray, parameters)
                dens[] = dens[] .* cScaleFac
                ff[] = Base.Fix2(streamFunction, t)
                if IBBool
                    UpdateIBPositions!(t, IBPosDict, ObservablePositionArray, rExtArray)
                    UpdateIBColors!(t, eColor, eColor1, eColor2, parameters, kArray)
                end
                sleep(sleepLength)
                current_figure()
            end
        else
            for t in 1:length(velocityArray)
                UpdateDens!(t, grid, dens, col, densityArray, directorArray, nematicArray, phibArray, phiuArray, sigmaVEArray, velocityArray, parameters)
                dens[] = dens[] .* cScaleFac
                ff[] = Base.Fix2(streamFunction, t)
                if IBBool
                    UpdateIBPositions!(t, IBPosDict, ObservablePositionArray, rExtArray)
                    UpdateIBColors!(t, eColor, eColor1, eColor2, parameters, kArray)
                end
                sleep(sleepLength)
                current_figure()
            end
        end

    end # AnimateTrajectoryStreamlines()

    function StaticStreamlines(t, parameters, # add defect tracking
        densityArray, velocityArray, directorArray, nematicArray, phibArray, phiuArray, sigmaVEArray,
        ObservablePositionArray, rExtArray, kArray;
        col = "dens", nem = false, colorrange = (0.99, 1.001), recording = false, imagePath = "", res = 2000, cScaleFac = 1,
        streamArrowSize = 0.1, streamLineWidth = 4.0, streamDensity = 3.0, streamStepSize = 0.01,
        IBBool = false, tracking = false, p1Bool = false, IBlinewidth = 6.0, IBmarkersize = 15,
        iColor = :black, eColor1 = :red, eColor2 = :red, cColor = :black, oColor = :orange, tColor = :white, p1Color = :orange,
        fontsize = 35, ticks = [1], tickLabels = [""], fac = 10^6, xlabel = "", ylabel = "", clabel = "",
        display = true)

        grid = Analysis.GetGrid(parameters)
        (xs, ys, pts) = GetPts(grid)

        if IBBool
            IBPosDict = GetIBPositions(t, ObservablePositionArray, rExtArray)
        end

        fx = grid.Nx / grid.Ny
        f = Figure(resolution = (res*fx, res), fontsize = fontsize, font = "Arial", dpi = 300)
        ax = Axis(f[1, 1],
            xlabel = xlabel,
            ylabel = ylabel,
            xticks = (ticks, tickLabels),
            yticks = (ticks, tickLabels)
        )

        dens = GetDens(t, grid, col, densityArray, directorArray, nematicArray, phibArray, phiuArray, sigmaVEArray, velocityArray, parameters)
        dens[] = dens[] .* cScaleFac
        #dens[] = (dens[] .- 1) .* cScaleFac
        hm = DrawHeatMapNS!(xs, ys, dens, col, colorrange)
        SetLimsNS!(ax, grid)
        GLMakie.Colorbar(f[1,2], hm, label = clabel)
        if nem
            velocityArray = [BerisEdwards.GetUnitDirectorFromTensor2DSoA(grid, nematicArray[i]) for i = 1:length(nematicArray)]
        end
        streamFunction = GetStreamFunction(grid, velocityArray)
        ff = Base.Fix2(streamFunction, t)
      
        GLMakie.streamplot!(ff, 1:.1:grid.Nx, 1:.1:grid.Ny; arrow_size = streamArrowSize, linewidth = streamLineWidth,
        density = streamDensity, stepsize = streamStepSize, colormap = :grays)
 
        if IBBool
            eColor = Observable(eColor1)
            DrawIBPositionsNS!(IBPosDict, IBlinewidth, IBmarkersize, tracking, p1Bool, iColor, eColor, cColor, oColor, tColor, p1Color)
        end

        if recording
            GLMakie.save(imagePath, f, px_per_unit = 3)
        end

        if display
            current_figure()
        else
            return ax
        end
    end # StaticStreamlinesNS


end # module
