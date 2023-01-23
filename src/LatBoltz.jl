module LatBoltz

    include("SharedStructs.jl")
    using .SharedStructs

    include("MathFunctions.jl")
    using .MathFunctions
    ########################################################################################
    #
    #                               Struct definitions
    #
    ########################################################################################

    abstract type Lattice end

    struct LatticeD2Q9 <: Lattice

        dx::Float64
        dt::Float64
        c::Float64
        LatticeMatrix::Matrix{Float64}
        WeightVec::Vector{Float64}
        LengthVec::Vector{Float64}
        nVecs::Int

        function LatticeD2Q9(dx::Float64, dt::Float64)

            lm = [ 0  0; # cAlpha's ordered differently from pg 88 in book
                   1  0;
                  -1  0;
                   0  1;
                   0 -1;
                   1  1;
                   1 -1;
                  -1  1;
                  -1 -1]';

            wv = [4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
                1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0];

            lv = [0.0, 1.0, 1.0, 1.0, 1.0,
                sqrt(2), sqrt(2), sqrt(2), sqrt(2)];

            return new(dx, dt, dx / dt, lm, wv, lv, length(wv))

        end

    end

    struct VelocityDistribution2D

        DistributionLattice::Lattice
        Values::Array{Real, 3}

        function VelocityDistribution2D(grid, lattice::LatticeD2Q9)
            vals = zeros(grid.Nx, grid.Ny, length(lattice.WeightVec))
            return new(lattice, vals)
        end
    end

    struct ExternalForceTimeSchedule
        Values::Array{Real}

        function ExternalForceTimeSchedule(timeDomain, func) # takes a generic function as second argument
            vals = map(x -> func(x), timeDomain)
            return new(vals)
        end

    end

    struct ExternalForce2D
        SpatialPattern
        TimeSchedule::ExternalForceTimeSchedule

        function ExternalForce2D(spatialPattern, timeSchedule::ExternalForceTimeSchedule)
            return new(spatialPattern, timeSchedule)
        end

    end

    ########################################################################################
    #
    #                               Struct of Array functions
    #
    ########################################################################################

    function UpdateDensity2DSoA!(densitySoA, velocityDistributionSoA)
        densitySoA.Values[:] .=  0.0
        @inbounds for r in 1:velocityDistributionSoA.nR, c in 1:velocityDistributionSoA.nC
                densitySoA.Values .+= velocityDistributionSoA.ValuesVector[LargeTensorListIndex(r, c, velocityDistributionSoA.nC)]
        end
    end

    function UpdateVelocity2DSoA!(lattice, velocitySoA, velocityDistributionSoA, densitySoA, externalForceSoA, dt)
        MultipyVectorSoA2D!(velocitySoA, 0.0)
        @inbounds for r in 1:velocityDistributionSoA.nR, c in 1:velocityDistributionSoA.nC
            ind = LargeTensorListIndex(r, c, velocityDistributionSoA.nC)
            velocitySoA.XValues .+= velocityDistributionSoA.ValuesVector[ind] .* lattice.LatticeMatrix[1,ind]
            velocitySoA.YValues .+= velocityDistributionSoA.ValuesVector[ind] .* lattice.LatticeMatrix[2,ind]
        end
        @fastmath velocitySoA.XValues .+= 0.5 .* dt .* externalForceSoA.XValues
        @fastmath velocitySoA.YValues .+= 0.5 .* dt .* externalForceSoA.YValues
        @fastmath velocitySoA.XValues ./= densitySoA.Values
        @fastmath velocitySoA.YValues ./= densitySoA.Values
        replace!(velocitySoA.XValues , NaN=>0.0)
        replace!(velocitySoA.YValues , NaN=>0.0)
    end

    function fEqAlphaBare2D(cAlpha, v)  # Book eq 3.54, assumes cAlpha and v are non-dimensional
        @fastmath return 1.0 + 3.0 * DotProduct2D(cAlpha, v) + (4.5) * DotProduct2D(cAlpha,v)^2 - (1.5) * DotProduct2D(v,v)
    end

    function ComputefEqForGrid2DSoA(grid, lattice, velocitySoA, velocityDistributionSoA, densitySoA)
        latticeMatrix = lattice.LatticeMatrix
        fEq = LargeTensorSoA2D(grid, velocityDistributionSoA.nR, velocityDistributionSoA.nC)
        @inbounds for r in 1:velocityDistributionSoA.nR, c in 1:velocityDistributionSoA.nC
            ind = LargeTensorListIndex(r, c, velocityDistributionSoA.nC)
            @fastmath dotProdCV = latticeMatrix[1,ind] .* velocitySoA.XValues .+ latticeMatrix[2,ind] .* velocitySoA.YValues
            @fastmath dotProdVV = velocitySoA.XValues .* velocitySoA.XValues .+ velocitySoA.YValues .* velocitySoA.YValues
            @fastmath fEq.ValuesVector[ind] .= densitySoA.Values .* lattice.WeightVec[ind] .+ (3.0 .* dotProdCV .+ 4.5 .* dotProdCV.^2 - 1.5 .* dotProdVV) .* densitySoA.Values .* lattice.WeightVec[ind]
        end
        return fEq
    end

    function ComputeFAlphaForGrid2DSoA(grid, lattice, velocitySoA, velocityDistributionSoA, externalForceSoA)
        latticeMatrix = lattice.LatticeMatrix
        FAlpha = LargeTensorSoA2D(grid, velocityDistributionSoA.nR, velocityDistributionSoA.nC)
        gAlpha = VectorSoA2D(grid)
        @inbounds for r in 1:velocityDistributionSoA.nR, c in 1:velocityDistributionSoA.nC
            ind = LargeTensorListIndex(r, c, velocityDistributionSoA.nC)
            @fastmath dotProd = latticeMatrix[1,ind] .* velocitySoA.XValues .+ latticeMatrix[2,ind] .* velocitySoA.YValues
            @fastmath gAlpha.XValues .= 3.0 .* (latticeMatrix[1,ind] .- velocitySoA.XValues) .+ 9.0 .* dotProd .* latticeMatrix[1,ind]
            @fastmath gAlpha.YValues .= 3.0 .* (latticeMatrix[2,ind] .- velocitySoA.YValues) .+ 9.0 .* dotProd .* latticeMatrix[2,ind]
            @fastmath FAlpha.ValuesVector[ind] .= (gAlpha.XValues .* externalForceSoA.XValues .+ gAlpha.YValues .* externalForceSoA.YValues) .* lattice.WeightVec[ind]
        end
        return FAlpha
    end

    function CollisionStep2DSoA!(grid, lattice, velocitySoA, velocityDistributionSoA, densitySoA, externalForceSoA, tRatio, dt)
        fEq = ComputefEqForGrid2DSoA(grid, lattice, velocitySoA, velocityDistributionSoA, densitySoA)
        FAlpha = ComputeFAlphaForGrid2DSoA(grid, lattice, velocitySoA, velocityDistributionSoA, externalForceSoA)
        @inbounds for r in 1:velocityDistributionSoA.nR, c in 1:velocityDistributionSoA.nC
            ind = LargeTensorListIndex(r, c, velocityDistributionSoA.nC)
            @fastmath velocityDistributionSoA.ValuesVector[ind] .= (1 - tRatio) .* velocityDistributionSoA.ValuesVector[ind] .+ tRatio .* fEq.ValuesVector[ind] .+ (dt .- 0.5 .* tRatio)  .* FAlpha.ValuesVector[ind]
        end
    end

    function StreamBulkSoA!(latticeMatrix, currentDistributionSoA, temporaryDistributionSoA) # assumes a rectangular geometry
        @inbounds for r in 1:currentDistributionSoA.nR, c in 1:currentDistributionSoA.nC
            aInd = LargeTensorListIndex(r, c, currentDistributionSoA.nC)
            dirX = Int(latticeMatrix[1,aInd])
            dirY = Int(latticeMatrix[2,aInd])
            @fastmath temporaryDistributionSoA.ValuesVector[aInd][2+dirX:end-1+dirX,2+dirY:end-1+dirY] .+= currentDistributionSoA.ValuesVector[aInd][2:end-1, 2:end-1]
        end
    end

    function GetBBAlphaInd2D(alpha, latticeMatrix, ind, bound)
        nDir = convert(Vector{Int}, latticeMatrix[:,alpha])
        hit = false
        if ((ind[1] == bound[1]) && (nDir[1] == 1))
            hit = true
        elseif ((ind[1] == 1) && (nDir[1] == -1))
            hit = true
        elseif ((ind[2] == bound[2]) && (nDir[2] == 1))
            hit = true
        elseif ((ind[2] == 1) && (nDir[2] == -1))
            hit = true
        end
        if hit
            newAlpha = -1 .* nDir
            newInd = ind
        else
            newAlpha = nDir
            newInd = ind .+ nDir
        end
        lmList = [latticeMatrix[:,i] for i in 1:size(latticeMatrix)[2]]
        newAlpha = Int(findall(x -> x == newAlpha, lmList)[1])
        return (newAlpha, newInd)
    end

    function GetIndInformation(bcType, latticeMatrix, bound, aInd, ind)
        ones = [1, 1]
        nDir = convert(Vector{Int}, latticeMatrix[:,aInd])
        if bcType == "pbc"
            newInd = ((bound .+ (ind .- ones) .+ nDir) .% bound) .+ ones
            newAInd = aInd 
        else # case bbc
            (newAInd, newInd) = GetBBAlphaInd2D(aInd, latticeMatrix, ind, bound)
        end
        return (newAInd, newInd)
    end

    function LidFunc(grid, ind)
        x = ind[1] / grid.Nx
        return x^2 * (1-x)^2
    end

    function GetVelFac(c, vel, aInd)
        wlX = [0.0, 1.0/9.0, -1.0/9.0, 0.0, 0.0, 1.0/36.0, 1.0/36.0, -1.0/36.0, -1.0/36.0];
        wlY = [0.0, 0.0, 0.0, 1.0/9.0, -1.0/9.0, 1.0/36.0, -1.0/36.0, 1.0/36.0, -1.0/36.0];
        if ((c == 1) || (c==2))
            wl = wlY[aInd]
        else
            wl = wlX[aInd]
        end
        hitFac = 0
        if c == 1 # mX
            if aInd in (3,8,9)
                hitFac = 1
            end
        elseif c == 2 # pX 
            if aInd in (2,6,7)
                hitFac = 1
            end
        elseif c == 3 # mY
            if aInd in (5,7,9)
                hitFac = 1
            end
        else # pY
            if aInd in (4,6,8)
                hitFac = 1
            end
        end 
        return 6.0 * vel * wl * hitFac
    end

    function MixedIndLookup(ind) # 1-xmym, 2-xmyp, 3-xpym, 4-xpyp
        if ind == 1 # xmym
            bulkAInds = [1,2,4,6]
            xmAInds = [3,8]
            ymAInds = [5,7]
            xpAInds = []
            ypAInds = []
            cInd = [9]
            xmAInds = vcat(xmAInds, bulkAInds) # arbitrary choice since these will be in the bulk
        elseif ind == 2 # xmyp
            bulkAInds = [1,2,5,7]
            xmAInds = [3,9]
            ymAInds = []
            xpAInds = []
            ypAInds = [4,6]
            cInd = [8]
            xmAInds = vcat(xmAInds, bulkAInds)
        elseif ind == 3 # xpym
            bulkAInds = [1,3,4,8]
            xmAInds = []
            ymAInds = []
            xpAInds = [2,6]
            ypAInds = [5,9]
            cInd = [7]
            xpAInds = vcat(xpAInds, bulkAInds)
        else # xpyp
            bulkAInds = [1,3,5,9]
            xmAInds = []
            ymAInds = []
            xpAInds = [2,7]
            ypAInds = [4,8]
            cInd = [6]
            xpAInds = vcat(xpAInds, bulkAInds)
        end
        return (xmAInds, xpAInds, ymAInds, ypAInds, cInd)
    end

    function FillBoundaryList(grid, lattice, indCollection, LBBCIDVec, wallFunctions)
        bound = [grid.Nx, grid.Ny]
        latticeMatrix = lattice.LatticeMatrix
        boundaryDict = Dict([])
        for bcI in 1:4
            boundaryDict[bcI] = []
            for ind in indCollection[2][bcI] 
                lbbcid = LBBCIDVec[bcI]
                if bcI == 1 || bcI == 2 # x-faces
                    wfInd = 2
                else
                    wfInd = 1
                end
                for aInd in 1:length(latticeMatrix[1,:])
                    (newAInd, newInd) = GetIndInformation(lbbcid.Type, latticeMatrix, bound, aInd, ind)
                    velFac = GetVelFac(bcI, lbbcid.Vel, aInd) * wallFunctions[bcI][ind[wfInd]]
                    push!(boundaryDict[bcI], [aInd, ind, newAInd, newInd, velFac])
                end
            end
        end

        # handle corner points 
        boundaryDict[5] = [] # 
        halvedInds = Dict([
            1 => (1,3), # xm ym
            2 => (1,4), # xm yp
            3 => (2,3), # xp ym
            4 => (2,4) # xp yp
        ])
        for c in 1:4
            ind = indCollection[2][5][c]
            (xmAInds, xpAInds, ymAInds, ypAInds, cInd) = MixedIndLookup(c)
            for aInd in xmAInds
                (newAInd, newInd) = GetIndInformation(LBBCIDVec[1].Type, latticeMatrix, bound, aInd, ind)
                velFac = GetVelFac(1, LBBCIDVec[1].Vel, aInd) * wallFunctions[1][ind[2]]
                push!(boundaryDict[1], [aInd, ind, newAInd, newInd, velFac])
            end 
            for aInd in xpAInds
                (newAInd, newInd) = GetIndInformation(LBBCIDVec[2].Type, latticeMatrix, bound, aInd, ind)
                velFac = GetVelFac(2, LBBCIDVec[2].Vel, aInd) * wallFunctions[2][ind[2]]
                push!(boundaryDict[2], [aInd, ind, newAInd, newInd, velFac])
            end 
            for aInd in ymAInds
                (newAInd, newInd) = GetIndInformation(LBBCIDVec[3].Type, latticeMatrix, bound, aInd, ind)
                velFac = GetVelFac(3, LBBCIDVec[3].Vel, aInd) * wallFunctions[3][ind[1]]
                push!(boundaryDict[3], [aInd, ind, newAInd, newInd, velFac])
            end 
            for aInd in ypAInds
                (newAInd, newInd) = GetIndInformation(LBBCIDVec[4].Type, latticeMatrix, bound, aInd, ind)
                velFac = GetVelFac(4, LBBCIDVec[4].Vel, aInd) * wallFunctions[4][ind[1]]
                push!(boundaryDict[4], [aInd, ind, newAInd, newInd, velFac])
            end 

            his = halvedInds[c] # corner arrow conditions
            aInd = cInd[1]
            if (LBBCIDVec[his[1]].Type == "pbc") && (LBBCIDVec[his[2]].Type == "bbc") # x is pbc, y is bbc
                c = his[2]
                (newAInd, newInd) = GetIndInformation(LBBCIDVec[c].Type, latticeMatrix, bound, aInd, ind)
                velFac = GetVelFac(c, LBBCIDVec[c].Vel, aInd) * wallFunctions[his[2]][1]
            elseif (LBBCIDVec[his[1]].Type == "bbc") && (LBBCIDVec[his[2]].Type == "pbc") # x is bbc, y is pbc
                c = his[1]
                (newAInd, newInd) = GetIndInformation(LBBCIDVec[c].Type, latticeMatrix, bound, aInd, ind)
                velFac = GetVelFac(c, LBBCIDVec[c].Vel, aInd) * wallFunctions[his[2]][2]
            elseif (LBBCIDVec[his[1]].Type == "pbc") && (LBBCIDVec[his[2]].Type == "pbc")
                c = his[1]
                (newAInd, newInd) = GetIndInformation(LBBCIDVec[c].Type, latticeMatrix, bound, aInd, ind)
                velFac = GetVelFac(c, LBBCIDVec[c].Vel, aInd) # returns 0
            else
                if abs(LBBCIDVec[his[1]].Vel) > abs(LBBCIDVec[his[1]].Vel)
                    c = his[1]
                    wfInd = 2
                else
                    c = his[2]
                    wfInd = 1
                end
                (newAInd, newInd) = GetIndInformation(LBBCIDVec[c].Type, latticeMatrix, bound, aInd, ind)
                velFac = GetVelFac(c, LBBCIDVec[c].Vel, aInd) * wallFunctions[c][wfInd]
            end
            push!(boundaryDict[5], [aInd, ind, newAInd, newInd, velFac])
        end    
        return boundaryDict
    end

    function StreamStep2DSoA!(grid, lattice, velocityDistributionSoA, densitySoA, indCollection, boundaryDict, LBBCIDVec, vCont = 1.0)
        temporaryDistributionSoA = LargeTensorSoA2D(grid, velocityDistributionSoA.nR, velocityDistributionSoA.nC)
        latticeMatrix = lattice.LatticeMatrix
        bound = Ind2D([grid.Nx, grid.Ny])
        StreamBulkSoA!(latticeMatrix, velocityDistributionSoA, temporaryDistributionSoA)
        for bcI in 1:5
            boundaryList = boundaryDict[bcI]
            for inds in boundaryList
                aInd = inds[1]
                ind = inds[2]
                newAInd = inds[3]
                newInd = inds[4]
                velFac = inds[5]
                temporaryDistributionSoA.ValuesVector[newAInd][newInd...] += velocityDistributionSoA.ValuesVector[aInd][ind...] - vCont * (densitySoA.Values[ind...] * velFac)
            end
        end
        SetLargeTensorFromSoA2D!(velocityDistributionSoA, temporaryDistributionSoA)
    end


    function LatticeBoltzmannStep2DSoA!(grid, lattice, densitySoA, velocityDistributionSoA, velocitySoA, nonViscousForceSoA, indCollection, dt, tRatio, timeStride, t, boundaryDict, LBBCIDVec)
        UpdateDensity2DSoA!(densitySoA, velocityDistributionSoA)
        UpdateVelocity2DSoA!(lattice, velocitySoA, velocityDistributionSoA, densitySoA, nonViscousForceSoA, dt)
        CollisionStep2DSoA!(grid, lattice, velocitySoA, velocityDistributionSoA, densitySoA, nonViscousForceSoA, tRatio, dt)
        StreamStep2DSoA!(grid, lattice, velocityDistributionSoA, densitySoA, indCollection, boundaryDict, LBBCIDVec)
    end

    function InitEquilibriumConverge2DSoA!(grid, lattice, velocitySoA, velocityDistributionSoA, densitySoA, nonViscousForceSoA, tRatio, dt, indCollection, boundaryDict, LBBCIDVec)
        fEq = ComputefEqForGrid2DSoA(grid, lattice, velocitySoA, velocityDistributionSoA, densitySoA)
        SetLargeTensorFromSoA2D!(velocityDistributionSoA, fEq)
        for i = 1:100
            UpdateDensity2DSoA!(densitySoA, velocityDistributionSoA)
            CollisionStep2DSoA!(grid, lattice, velocitySoA, velocityDistributionSoA, densitySoA, nonViscousForceSoA, tRatio, dt)
            StreamStep2DSoA!(grid, lattice, velocityDistributionSoA, densitySoA, indCollection, boundaryDict, LBBCIDVec, 0.0)
        end
    end





end # module
