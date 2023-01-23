module ReactAdvDiff

    include("SharedStructs.jl")
    using .SharedStructs

    include("MathFunctions.jl")
    using .MathFunctions

    struct RADParams
        kp::Real 
        km0::Real 
        Du::Real 
        theta::Real
        F0inv::Real
        phib0C::Real
        phib0a::Real
        function RADParams(kp, km0, Du, theta, F0inv, phib0C, phib0a)
            return new(kp, km0, Du, theta, F0inv, phib0C, phib0a)
        end
    end

    function GetScalingFactor(grid, phibSoA, fac) # fac should be 1/phib0C or 1/phib0a when called

        return MultipyScalarSoA2D(grid, phibSoA, fac)
    end

    function ComputeReactionRate(phibSoA, phiuSoA, densitySoA, viscoelasticForceSoA, radParams) 

        onTerm = radParams.kp .* phiuSoA.Values .* (radParams.theta .* densitySoA.Values .- phibSoA.Values)
        offTerm = radParams.km0 .* exp.(- VectorNormSoA(viscoelasticForceSoA).Values .* radParams.F0inv) .* phibSoA.Values
        
        return ScalarSoA2D(onTerm .- offTerm)
    end

    
    function ComputePhibRHS(grid, velocitySoA, phibSoA, reactionRateSoA, radParams, bcDerivX, bcDerivY)

        phibv = MultipyVectorByScalarSoA2D(grid, velocitySoA, phibSoA)
        rhs = DivVectorOnSoA2D(grid, phibv, bcDerivX, bcDerivY)
        MultipyScalarSoA2D!(rhs, -1.0)
        AddScalarSoA2D!(rhs, reactionRateSoA)

        return rhs
    end

    function ComputePhiuRHS(grid, velocitySoA, phiuSoA, reactionRateSoA, radParams, bcDerivX, bcDerivY)

        phiuv = MultipyVectorByScalarSoA2D(grid, velocitySoA, phiuSoA)
        rhs = DivVectorOnSoA2D(grid, phiuv, bcDerivX, bcDerivY)
        MultipyScalarSoA2D!(rhs, -1.0)
        diffTerm = ScalarSoA2D(radParams.Du .* ((bcDerivX(phiuSoA.Values, FiniteSecondDifferenceX) .+ bcDerivY(phiuSoA.Values, FiniteSecondDifferenceY)) ./ (grid.dx)^2))
        AddScalarSoA2D!(rhs, diffTerm)
        SubtractScalarSoA2D!(rhs, reactionRateSoA)

        return rhs
    end


    function PredictorCorrectorStepRAD2DSoA!(grid, densitySoA, velocitySoA, viscoelasticForceSoA, phibSoA, phiuSoA, dt, radParams, bcx, bcy)

        bcDerivX = BCDerivDict[bcx]
        bcDerivY = BCDerivDict[bcy]
        b = GetBoundariesFromConditions(grid, bcx, bcy)

        predPhibSoA = deepcopy(phibSoA)
        predPhiuSoA = deepcopy(phiuSoA)

        reactionRateSoA = ComputeReactionRate(predPhibSoA, predPhiuSoA, densitySoA, viscoelasticForceSoA, radParams) 
        rhs1b = ComputePhibRHS(grid, velocitySoA, predPhibSoA, reactionRateSoA, radParams, bcDerivX, bcDerivY)
        rhs1u = ComputePhiuRHS(grid, velocitySoA, predPhiuSoA, reactionRateSoA, radParams, bcDerivX, bcDerivY)
        predPhibSoA.Values[b[1]:b[2],b[3]:b[4]] .= predPhibSoA.Values[b[1]:b[2],b[3]:b[4]] .+ dt .* rhs1b.Values[b[1]:b[2],b[3]:b[4]]
        predPhiuSoA.Values[b[1]:b[2],b[3]:b[4]] .= predPhiuSoA.Values[b[1]:b[2],b[3]:b[4]] .+ dt .* rhs1u.Values[b[1]:b[2],b[3]:b[4]]

        reactionRateSoA = ComputeReactionRate(predPhibSoA, predPhiuSoA, densitySoA, viscoelasticForceSoA, radParams) 
        rhs2b = ComputePhibRHS(grid, velocitySoA, predPhibSoA, reactionRateSoA, radParams, bcDerivX, bcDerivY)
        rhs2u = ComputePhiuRHS(grid, velocitySoA, predPhiuSoA, reactionRateSoA, radParams, bcDerivX, bcDerivY)

        phibSoA.Values[b[1]:b[2],b[3]:b[4]] .= phibSoA.Values[b[1]:b[2],b[3]:b[4]] .+ (0.5 .* dt .* (rhs1b.Values[b[1]:b[2],b[3]:b[4]] .+ rhs2b.Values[b[1]:b[2],b[3]:b[4]]))
        phiuSoA.Values[b[1]:b[2],b[3]:b[4]] .= phiuSoA.Values[b[1]:b[2],b[3]:b[4]] .+ (0.5 .* dt .* (rhs1u.Values[b[1]:b[2],b[3]:b[4]] .+ rhs2u.Values[b[1]:b[2],b[3]:b[4]]))
        
    end

end
