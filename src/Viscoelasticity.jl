module Viscoelasticity

    include("SharedStructs.jl")
    using .SharedStructs

    include("MathFunctions.jl")
    using .MathFunctions

    using Bessels
    using LinearAlgebra

    struct VEParams
        eta::Real
        rhoF::Real
        function VEParams(eta, rhoF)
            return new(eta, rhoF)
        end
    end

    struct VECParams
        eta::Real
        C::Real
        Ds::Real
        function VECParams(eta, C, Ds)
            return new(eta, C, Ds)
        end
    end

    struct VETParams
        Ke1111::Real
        Ke1122::Real
        Ke1212::Real
        Ke2222::Real
        eta::Real
        Ds::Real
        function VETParams(Ke1111, Ke1122, Ke1212, Ke2222, eta, Ds)
            return new(Ke1111, Ke1122, Ke1212, Ke2222, eta, Ds)
        end
    end

    struct VETOddParams
        C11::Real
        C13::Real
        C31::Real
        C24::Real
        C33::Real
        C44::Real
        eta::Real
        Ds::Real
        function VETOddParams(C11, C13, C31, C24, C33, C44, eta, Ds)
            return new(C11, C13, C31, C24, C33, C44, eta, Ds)
        end
    end

    struct VETOddIsoParams
        AA::Real
        BB::Real
        mu::Real
        K0::Real
        eta::Real
        Ds::Real
        function VETOddIsoParams(AA, BB, mu, K0, eta, Ds)
            return new(AA, BB, mu, K0, eta, Ds)
        end
    end

    
    ########################################################################################
    #
    #                               Tensorial helper functions
    #
    ########################################################################################

    function kFromP(P)
        return 1.18724 * tan(1.54357 * P)
    end

    function thetaFromPVec(PVec)
        return (atan(PVec[2], PVec[1]) + 2 * pi) % (2 * pi) # return in	the range (0 , 2 pi)
        #return atan(PVec[2], PVec[1]) # return in	the range (0 , 2 pi)
    end

    function ParamsFromPVec(PVec)
        P = VectorNorm2D(PVec)
        if P > 1e-8
            Punit = PVec ./ P
            if P > 1.0 # cut off to keep tan(x) before periodic repeat
                P = 1.0
            end
            k = kFromP(P)
            thp = thetaFromPVec(PVec)
        else
            k = 0
            thp = 0
        end
        return [k, thp]
    end

    function CMat(k, thp, Ke1111, Ke1122, Ke1212, Ke2222)
        C = [0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0]
        b0 = besseli(0,k)
        b1 = besseli(1,k)
        b2 = besseli(2,k)
        c2 = cos(2*thp)
        c4 = cos(4*thp)
        s2 = sin(2*thp)
        s4 = sin(4*thp)
        C[1,1] = ((3*Ke1111 + 2*Ke1122 + 4*Ke1212 + 3*Ke2222)*b0 + 4*(Ke1111 - Ke2222)*b2*c2 +
                ((Ke1111 - 2*Ke1122 - 4*Ke1212 + Ke2222)*(k*(24 + k^2)*b0 -
                8*(6 + k^2)*b1)*c4)/k^3)/(8 *b0)
        C[1,2] = (Ke1111 + 6*Ke1122 - 4*Ke1212 + Ke2222 - ((Ke1111 - 2*Ke1122 -
                4*Ke1212 + Ke2222)*(k*(24 + k^2)*b0 - 8*(6 +
                k^2)*b1)*c4)/(k^3*b0))/8
        C[1,3] = (2*(Ke1111 - Ke2222)*pi*b2*s2 +
                ((Ke1111 - 2*Ke1122 - 4*Ke1212 + Ke2222)*pi*(k*(24 + k^2)*b0 -
                8*(6 + k^2)*b1)*s4)/k^3)/(4 *pi*b0)
        C[2,1] = C[1,2]
        C[2,2] = ((3*Ke1111 + 2*Ke1122 + 4*Ke1212 + 3*Ke2222)*b0 +
                4*(-Ke1111 + Ke2222)*b2*c2 + ((Ke1111 - 2*Ke1122 -
                4*Ke1212 + Ke2222)*(k*(24 + k^2)*b0 - 8*(6 +
                k^2)*b1)*c4)/k^3)/(8 *b0)
        C[2,3] = -0.25*(2*(-Ke1111 + Ke2222)*pi*b2*s2 + ((Ke1111 -
                2*Ke1122 - 4*Ke1212 + Ke2222)*pi*(k*(24 + k^2)*b0 - 8*(6 +
                k^2)*b1)*s4)/k^3)/(pi*b0)
        C[3,1] = C[1,3]/2
        C[3,2] = C[2,3]/2
        C[3,3] = (Ke1111 - 2*Ke1122 + 4*Ke1212 + Ke2222 - ((Ke1111 - 2*Ke1122 -
                4*Ke1212 + Ke2222)*(k*(24 + k^2)*b0 - 8*(6 +
                k^2)*b1)*c4)/(k^3*b0))/4

        return C
    end

    function CMat0(Ke1111, Ke1122, Ke1212, Ke2222)
        C = [0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0]
        C[1,1] = (3*Ke1111 + 2*Ke1122 + 4*Ke1212 + 3*Ke2222)/8
        C[1,2] = (Ke1111 + 6*Ke1122 - 4*Ke1212 + Ke2222)/8
        C[1,3] = 0
        C[2,1] = C[1,2]
        C[2,2] = (3*Ke1111 + 2*Ke1122 + 4*Ke1212 + 3*Ke2222)/8
        C[2,3] = 0
        C[3,1] = C[1,3]/2
        C[3,2] = C[2,3]/2
        C[3,3] = (Ke1111 - 2*Ke1122 + 4*Ke1212 + Ke2222)/4
        return C
    end


    function CMatOddIso(AA, BB, mu, K0)
        C = [0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]
        C[1,1] = BB
        C[1,2] = 0
        C[1,3] = 0
        C[1,4] = 0
        C[2,1] = AA
        C[2,2] = 0
        C[2,3] = 0
        C[2,4] = 0
        C[3,1] = 0
        C[3,2] = 0
        C[3,3] = mu
        C[3,4] = K0
        C[4,1] = 0
        C[4,2] = 0
        C[4,3] = -K0
        C[4,4] = mu
        return C
    end

    function CMatOdd(k, thp, C11, C13, C31, C24, C33, C44)
        C = [0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]
        b0 = besseli(0,k)
        b1 = besseli(1,k)
        b2 = besseli(2,k)
        c2 = cos(2*thp)
        c4 = cos(4*thp)
        s2 = sin(2*thp)
        s4 = sin(4*thp)
        C[1,1] = C11
        C[1,2] = 0
        C[1,3] = (C13*b2*c2)/b0
        C[1,4] = -((C13*b2*s2)/b0)
        C[2,1] = 0
        C[2,2] = 0
        C[2,3] = (C24*b2*s2)/b0
        C[2,4] = (C24*b2*c2)/b0
        C[3,1] = (C31*b2*c2)/b0
        C[3,2] = 0
        C[3,3] = (C33 + C44 + ((C33 - C44)*(k*(24 + k^2)*b0 - 8*(6 + k^2)*b1)*c4)/(k^3*b0))/2.
        C[3,4] = -0.5*((C33 - C44)*(k*(24 + k^2)*b0 - 8*(6 + k^2)*b1)*s4)/(k^3*b0)
        C[4,1] = -((C31*b2*s2)/b0)
        C[4,2] = 0
        C[4,3] = -0.5*((C33 - C44)*(k*(24 + k^2)*b0 - 8*(6 + k^2)*b1)*s4)/(k^3*b0)
        C[4,4] = (C33 + C44 + ((C33 - C44)*(-(k*(24 + k^2)) + (8*(6 + k^2)*b1)/b0)*c4)/k^3)/2.
        return C
    end

    function CMatOdd0(C11, C13, C31, C24, C33, C44)
        C = [0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]
        C[1,1] = C11
        C[1,2] = 0
        C[1,3] = 0
        C[1,4] = 0
        C[2,1] = 0
        C[2,2] = 0
        C[2,3] = 0
        C[2,4] = 0
        C[3,1] = 0
        C[3,2] = 0
        C[3,3] = 0.5 * (C33 + C44)
        C[3,4] = 0
        C[4,1] = 0
        C[4,2] = 0
        C[4,3] = 0
        C[4,4] = 0.5 * (C33 + C44)
        return C
    end



    ########################################################################################
    #
    #                               Struct of Array functions
    #
    ########################################################################################

    @inline function CMatFromPVecArraySoA(grid, directorSoA, veParams)
        CMatSoA = LargeTensorSoA2D(grid, 3, 3)
        for i = 1:grid.Nx, j = 1:grid.Ny
            params = ParamsFromPVec([directorSoA.XValues[i,j], directorSoA.YValues[i,j]])
            if params[1] > 100 # helps prevent overflow
                params[1] = 100
            end
            if params[1] == 0
                CMatV = CMat0(veParams.Ke1111, veParams.Ke1122, veParams.Ke1212, veParams.Ke2222)
            else
                CMatV = CMat(params[1], params[2], veParams.Ke1111, veParams.Ke1122, veParams.Ke1212, veParams.Ke2222)
            end
            for r = 1:CMatSoA.nR, c = 1:CMatSoA.nC
                CMatSoA.ValuesVector[LargeTensorListIndex(r,c,CMatSoA.nC)][i,j] = CMatV[r,c]
            end
        end
        # CMat is stored in its Mandel basis
        return CMatSoA
    end


    @inline function CMatOddFromPVecArraySoA(grid, directorSoA, veOddParams)
        CMatOddSoA = LargeTensorSoA2D(grid, 4, 4)
        for i = 1:grid.Nx, j = 1:grid.Ny
            params = ParamsFromPVec([directorSoA.XValues[i,j], directorSoA.YValues[i,j]])
            if params[1] > 100 # helps prevent overflow
                params[1] = 100
            end
            if params[1] == 0
                CMatOddV = CMatOdd0(veOddParams.C11, veOddParams.C13, veOddParams.C31, veOddParams.C24, veOddParams.C33, veOddParams.C44)
            else
                CMatOddV = CMatOdd(params[1], params[2], veOddParams.C11, veOddParams.C13, veOddParams.C31, veOddParams.C24, veOddParams.C33, veOddParams.C44)
            end
            for r = 1:CMatOddSoA.nR, c = 1:CMatOddSoA.nC
                CMatOddSoA.ValuesVector[LargeTensorListIndex(r,c,CMatOddSoA.nC)][i,j] = CMatOddV[r,c]
            end
        end
        # CMatOdd is stored in its tau basis
        return CMatOddSoA
    end

    @inline function CMatOddIsoOnGridSoA(grid, veOddIsoParams)
        CMatOddIsoSoA = LargeTensorSoA2D(grid, 4, 4)
        CMatOddIsoV = CMatOddIso(veOddIsoParams.AA, veOddIsoParams.BB, veOddIsoParams.mu, veOddIsoParams.K0)
        for i = 1:grid.Nx, j = 1:grid.Ny
            for r = 1:CMatOddIsoSoA.nR, c = 1:CMatOddIsoSoA.nC
                CMatOddIsoSoA.ValuesVector[LargeTensorListIndex(r,c,CMatOddIsoSoA.nC)][i,j] = CMatOddIsoV[r,c]
            end
        end
        # CMatOddIso is stored in its tau basis
        return CMatOddIsoSoA
    end

    @inline function CorotTermSoA(grid, sigmaVESoA, OmegaSoA, PsiSoA)
        OmegaDotSigma = MatrixDotMatrixOnSoA2D(grid, OmegaSoA, sigmaVESoA)
        SigmaDotOmega = MatrixDotMatrixOnSoA2D(grid, sigmaVESoA, OmegaSoA)
        XXValues = -(OmegaDotSigma.XXValues .- SigmaDotOmega.XXValues)
        XYValues = -(OmegaDotSigma.XYValues .- SigmaDotOmega.XYValues)
        YXValues = -(OmegaDotSigma.YXValues .- SigmaDotOmega.YXValues)
        YYValues = -(OmegaDotSigma.YYValues .- SigmaDotOmega.YYValues)
        return TensorSoA2D(XXValues, XYValues, YXValues, YYValues)
    end

    @inline function UpperConvTermSoA(grid, sigmaVESoA, OmegaSoA, PsiSoA)
        gradVSoA = AddTensorSoA2D(grid, OmegaSoA, PsiSoA)
        gradVTSoA = MatrixTransposeOnSoA2D(grid, gradVSoA)
        gradVTDotOmega = MatrixDotMatrixOnSoA2D(grid, gradVTSoA, sigmaVESoA)
        omegaDotGradV = MatrixDotMatrixOnSoA2D(grid, sigmaVESoA, gradVSoA)
        XXValues = gradVTDotOmega.XXValues .+ omegaDotGradV.XXValues
        XYValues = gradVTDotOmega.XYValues .+ omegaDotGradV.XYValues
        YXValues = gradVTDotOmega.YXValues .+ omegaDotGradV.YXValues
        YYValues = gradVTDotOmega.YYValues .+ omegaDotGradV.YYValues
        return TensorSoA2D(XXValues, XYValues, YXValues, YYValues)
    end

    @inline function AdvectiveTermSoA(grid, velocitySoA, sigmaVESoA, bcDerivX, bcDerivY)
        XXValuesX = velocitySoA.XValues .* bcDerivX(sigmaVESoA.XXValues, FiniteDifferenceX) ./ (2.0 * grid.dx)
        XYValuesX = velocitySoA.XValues .* bcDerivX(sigmaVESoA.XYValues, FiniteDifferenceX) ./ (2.0 * grid.dx)
        YXValuesX = velocitySoA.XValues .* bcDerivX(sigmaVESoA.YXValues, FiniteDifferenceX) ./ (2.0 * grid.dx)
        YYValuesX = velocitySoA.XValues .* bcDerivX(sigmaVESoA.YYValues, FiniteDifferenceX) ./ (2.0 * grid.dx)
        vxdxSoA = TensorSoA2D(XXValuesX, XYValuesX, YXValuesX, YYValuesX)
        XXValuesY = velocitySoA.YValues .* bcDerivY(sigmaVESoA.XXValues, FiniteDifferenceY) ./ (2.0 * grid.dx)
        XYValuesY = velocitySoA.YValues .* bcDerivY(sigmaVESoA.XYValues, FiniteDifferenceY) ./ (2.0 * grid.dx)
        YXValuesY = velocitySoA.YValues .* bcDerivY(sigmaVESoA.YXValues, FiniteDifferenceY) ./ (2.0 * grid.dx)
        YYValuesY = velocitySoA.YValues .* bcDerivY(sigmaVESoA.YYValues, FiniteDifferenceY) ./ (2.0 * grid.dx)
        vydySoA = TensorSoA2D(XXValuesY, XYValuesY, YXValuesY, YYValuesY)
        return (vxdxSoA, vydySoA)
    end

    @inline function DiffusionTermSoA(grid, sigmaVESoA, bcDerivX, bcDerivY)
        XXValues = (bcDerivX(sigmaVESoA.XXValues, FiniteSecondDifferenceX) .+ bcDerivY(sigmaVESoA.XXValues, FiniteSecondDifferenceY)) ./ (grid.dx)^2
        XYValues = (bcDerivX(sigmaVESoA.XYValues, FiniteSecondDifferenceX) .+ bcDerivY(sigmaVESoA.XYValues, FiniteSecondDifferenceY)) ./ (grid.dx)^2
        YXValues = (bcDerivX(sigmaVESoA.YXValues, FiniteSecondDifferenceX) .+ bcDerivY(sigmaVESoA.YXValues, FiniteSecondDifferenceY)) ./ (grid.dx)^2
        YYValues = (bcDerivX(sigmaVESoA.YYValues, FiniteSecondDifferenceX) .+ bcDerivY(sigmaVESoA.YYValues, FiniteSecondDifferenceY)) ./ (grid.dx)^2
        return TensorSoA2D(XXValues, XYValues, YXValues, YYValues)
    end

    @inline function MMat2DSoA(grid, sigmaVESoA, OmegaSoA, PsiSoA, velocitySoA, eta, C, Ds, bcDerivX, bcDerivY, matDeriv) # Maxwell
        (vxdxSoA, vydySoA) = AdvectiveTermSoA(grid, velocitySoA, sigmaVESoA, bcDerivX, bcDerivY)
        if matDeriv == "cor"
            matTermSoA = CorotTermSoA(grid, sigmaVESoA, OmegaSoA, PsiSoA)
        else # upper convected case
            matTermSoA = UpperConvTermSoA(grid, sigmaVESoA, OmegaSoA, PsiSoA)
        end
        diffTermSoA = DiffusionTermSoA(grid, sigmaVESoA, bcDerivX, bcDerivY)
        XXValues = 1 .* C .* PsiSoA.XXValues .- (C/eta) .* sigmaVESoA.XXValues .- (vxdxSoA.XXValues .+ vydySoA.XXValues) .+ matTermSoA.XXValues .+ Ds .* diffTermSoA.XXValues
        XYValues = 1 .* C .* PsiSoA.XYValues .- (C/eta) .* sigmaVESoA.XYValues .- (vxdxSoA.XYValues .+ vydySoA.XYValues) .+ matTermSoA.XYValues .+ Ds .* diffTermSoA.XYValues
        YXValues = 1 .* C .* PsiSoA.YXValues .- (C/eta) .* sigmaVESoA.YXValues .- (vxdxSoA.YXValues .+ vydySoA.YXValues) .+ matTermSoA.YXValues .+ Ds .* diffTermSoA.YXValues
        YYValues = 1 .* C .* PsiSoA.YYValues .- (C/eta) .* sigmaVESoA.YYValues .- (vxdxSoA.YYValues .+ vydySoA.YYValues) .+ matTermSoA.YYValues .+ Ds .* diffTermSoA.YYValues
        return TensorSoA2D(XXValues, XYValues, YXValues, YYValues)
    end

    function PredictorCorrectorStepMax2DSoA!(grid, OmegaSoA, PsiSoA, velocitySoA, sigmaVESoA, dt, veParams, bcx, bcy, matDeriv)
        bcDerivX = BCDerivDict[bcx]
        bcDerivY = BCDerivDict[bcy]
        b = GetBoundariesFromConditions(grid, bcx, bcy)
        m1 = MMat2DSoA(grid, sigmaVESoA, OmegaSoA, PsiSoA, velocitySoA, veParams.eta, veParams.C, veParams.Ds, bcDerivX, bcDerivY, matDeriv)
        predSigmaSoA = deepcopy(sigmaVESoA)
        predSigmaSoA.XXValues[b[1]:b[2],b[3]:b[4]] .= predSigmaSoA.XXValues[b[1]:b[2],b[3]:b[4]] .+ dt * m1.XXValues[b[1]:b[2],b[3]:b[4]]
        predSigmaSoA.XYValues[b[1]:b[2],b[3]:b[4]] .= predSigmaSoA.XYValues[b[1]:b[2],b[3]:b[4]] .+ dt * m1.XYValues[b[1]:b[2],b[3]:b[4]]
        predSigmaSoA.YXValues[b[1]:b[2],b[3]:b[4]] .= predSigmaSoA.YXValues[b[1]:b[2],b[3]:b[4]] .+ dt * m1.YXValues[b[1]:b[2],b[3]:b[4]]
        predSigmaSoA.YYValues[b[1]:b[2],b[3]:b[4]] .= predSigmaSoA.YYValues[b[1]:b[2],b[3]:b[4]] .+ dt * m1.YYValues[b[1]:b[2],b[3]:b[4]]
        m2 = MMat2DSoA(grid, predSigmaSoA, OmegaSoA, PsiSoA, velocitySoA, veParams.eta, veParams.C, veParams.Ds, bcDerivX, bcDerivY, matDeriv)
        sigmaVESoA.XXValues[b[1]:b[2],b[3]:b[4]] .= sigmaVESoA.XXValues[b[1]:b[2],b[3]:b[4]] .+ (0.5 .* dt .* (m1.XXValues[b[1]:b[2],b[3]:b[4]] .+ m2.XXValues[b[1]:b[2],b[3]:b[4]]))
        sigmaVESoA.XYValues[b[1]:b[2],b[3]:b[4]] .= sigmaVESoA.XYValues[b[1]:b[2],b[3]:b[4]] .+ (0.5 .* dt .* (m1.XYValues[b[1]:b[2],b[3]:b[4]] .+ m2.XYValues[b[1]:b[2],b[3]:b[4]]))
        sigmaVESoA.YXValues[b[1]:b[2],b[3]:b[4]] .= sigmaVESoA.YXValues[b[1]:b[2],b[3]:b[4]] .+ (0.5 .* dt .* (m1.YXValues[b[1]:b[2],b[3]:b[4]] .+ m2.YXValues[b[1]:b[2],b[3]:b[4]]))
        sigmaVESoA.YYValues[b[1]:b[2],b[3]:b[4]] .= sigmaVESoA.YYValues[b[1]:b[2],b[3]:b[4]] .+ (0.5 .* dt .* (m1.YYValues[b[1]:b[2],b[3]:b[4]] .+ m2.YYValues[b[1]:b[2],b[3]:b[4]]))
    end

    @inline function TSMat2DSoA(grid, sigmaVESoA, OmegaSoA, PsiSoA, CContrPsiSoA, CContrSigmaSoA, velocitySoA, eta, Ds, bcDerivX, bcDerivY, matDeriv)
        (vxdxSoA, vydySoA) = AdvectiveTermSoA(grid, velocitySoA, sigmaVESoA, bcDerivX, bcDerivY)
        if matDeriv == "cor"
            matTermSoA = CorotTermSoA(grid, sigmaVESoA, OmegaSoA, PsiSoA)
        else # upper convected case
            matTermSoA = UpperConvTermSoA(grid, sigmaVESoA, OmegaSoA, PsiSoA)
        end
        diffTermSoA = DiffusionTermSoA(grid, sigmaVESoA, bcDerivX, bcDerivY)
        XXValues = 1 .* CContrPsiSoA.XXValues .- (1/eta) .* CContrSigmaSoA.XXValues .- (vxdxSoA.XXValues .+ vydySoA.XXValues) .+ matTermSoA.XXValues .+ Ds .* diffTermSoA.XXValues
        XYValues = 1 .* CContrPsiSoA.XYValues .- (1/eta) .* CContrSigmaSoA.XYValues .- (vxdxSoA.XYValues .+ vydySoA.XYValues) .+ matTermSoA.XYValues .+ Ds .* diffTermSoA.XYValues
        YXValues = 1 .* CContrPsiSoA.YXValues .- (1/eta) .* CContrSigmaSoA.YXValues .- (vxdxSoA.YXValues .+ vydySoA.YXValues) .+ matTermSoA.YXValues .+ Ds .* diffTermSoA.YXValues
        YYValues = 1 .* CContrPsiSoA.YYValues .- (1/eta) .* CContrSigmaSoA.YYValues .- (vxdxSoA.YYValues .+ vydySoA.YYValues) .+ matTermSoA.YYValues .+ Ds .* diffTermSoA.YYValues
        return TensorSoA2D(XXValues, XYValues, YXValues, YYValues)
    end

    @inline function ContractWithCMatSoA(grid, CMatSoA, tSoA)
        XXValues = CMatSoA.ValuesVector[LargeTensorListIndex(1, 1, 3)] .* tSoA.XXValues .+ CMatSoA.ValuesVector[LargeTensorListIndex(1, 2, 3)] .* tSoA.YYValues .+ CMatSoA.ValuesVector[LargeTensorListIndex(1, 3, 3)] .* tSoA.XYValues
        YYValues = CMatSoA.ValuesVector[LargeTensorListIndex(2, 1, 3)] .* tSoA.XXValues .+ CMatSoA.ValuesVector[LargeTensorListIndex(2, 2, 3)] .* tSoA.YYValues .+ CMatSoA.ValuesVector[LargeTensorListIndex(2, 3, 3)] .* tSoA.XYValues
        XYValues = CMatSoA.ValuesVector[LargeTensorListIndex(3, 1, 3)] .* tSoA.XXValues .+ CMatSoA.ValuesVector[LargeTensorListIndex(3, 2, 3)] .* tSoA.YYValues .+ CMatSoA.ValuesVector[LargeTensorListIndex(3, 3, 3)] .* tSoA.XYValues
        YXValues = XYValues
        return TensorSoA2D(XXValues, XYValues, YXValues, YYValues)
    end

    function PredictorCorrectorStepTensS2DSoA!(grid, OmegaSoA, PsiSoA, velocitySoA, sigmaVESoA, CMatSoA, dt, veParams, bcx, bcy, matDeriv)
        bcDerivX = BCDerivDict[bcx]
        bcDerivY = BCDerivDict[bcy]
        b = GetBoundariesFromConditions(grid, bcx, bcy)
        gradVSoA = AddTensorSoA2D(grid, OmegaSoA, PsiSoA)
        CContrPsiSoA = ContractWithCMatSoA(grid, CMatSoA, PsiSoA)
        CContrSigmaSoA = ContractWithCMatSoA(grid, CMatSoA, sigmaVESoA)
        m1 = TSMat2DSoA(grid, sigmaVESoA, OmegaSoA, PsiSoA, CContrPsiSoA, CContrSigmaSoA, velocitySoA, veParams.eta, veParams.Ds, bcDerivX, bcDerivY, matDeriv)
        predSigmaSoA = deepcopy(sigmaVESoA)
        predSigmaSoA.XXValues[b[1]:b[2],b[3]:b[4]] .= predSigmaSoA.XXValues[b[1]:b[2],b[3]:b[4]] .+ dt * m1.XXValues[b[1]:b[2],b[3]:b[4]]
        predSigmaSoA.XYValues[b[1]:b[2],b[3]:b[4]] .= predSigmaSoA.XYValues[b[1]:b[2],b[3]:b[4]] .+ dt * m1.XYValues[b[1]:b[2],b[3]:b[4]]
        predSigmaSoA.YXValues[b[1]:b[2],b[3]:b[4]] .= predSigmaSoA.YXValues[b[1]:b[2],b[3]:b[4]] .+ dt * m1.YXValues[b[1]:b[2],b[3]:b[4]]
        predSigmaSoA.YYValues[b[1]:b[2],b[3]:b[4]] .= predSigmaSoA.YYValues[b[1]:b[2],b[3]:b[4]] .+ dt * m1.YYValues[b[1]:b[2],b[3]:b[4]]
        CContrSigmaSoA = ContractWithCMatSoA(grid, CMatSoA, sigmaVESoA)
        m2 = TSMat2DSoA(grid, predSigmaSoA, OmegaSoA, PsiSoA, CContrPsiSoA, CContrSigmaSoA, velocitySoA, veParams.eta, veParams.Ds, bcDerivX, bcDerivY, matDeriv)
        sigmaVESoA.XXValues[b[1]:b[2],b[3]:b[4]] .= sigmaVESoA.XXValues[b[1]:b[2],b[3]:b[4]] .+ (0.5 .* dt .* (m1.XXValues[b[1]:b[2],b[3]:b[4]] .+ m2.XXValues[b[1]:b[2],b[3]:b[4]]))
        sigmaVESoA.XYValues[b[1]:b[2],b[3]:b[4]] .= sigmaVESoA.XYValues[b[1]:b[2],b[3]:b[4]] .+ (0.5 .* dt .* (m1.XYValues[b[1]:b[2],b[3]:b[4]] .+ m2.XYValues[b[1]:b[2],b[3]:b[4]]))
        sigmaVESoA.YXValues[b[1]:b[2],b[3]:b[4]] .= sigmaVESoA.YXValues[b[1]:b[2],b[3]:b[4]] .+ (0.5 .* dt .* (m1.YXValues[b[1]:b[2],b[3]:b[4]] .+ m2.YXValues[b[1]:b[2],b[3]:b[4]]))
        sigmaVESoA.YYValues[b[1]:b[2],b[3]:b[4]] .= sigmaVESoA.YYValues[b[1]:b[2],b[3]:b[4]] .+ (0.5 .* dt .* (m1.YYValues[b[1]:b[2],b[3]:b[4]] .+ m2.YYValues[b[1]:b[2],b[3]:b[4]]))
    end

    function ContractWithCMatOddSoA(grid, CMatSoA, tSoA)
        comp1 = tSoA.XXValues .+ tSoA.YYValues
        comp2 = tSoA.YXValues .- tSoA.XYValues
        comp3 = tSoA.XXValues .- tSoA.YYValues
        comp4 = tSoA.YXValues .+ tSoA.XYValues
        term1 = 2 .* (CMatSoA.ValuesVector[LargeTensorListIndex(1, 1, 4)] .* comp1 .+ CMatSoA.ValuesVector[LargeTensorListIndex(1, 2, 4)] .* comp2 .+
                            CMatSoA.ValuesVector[LargeTensorListIndex(1, 3, 4)] .* comp3 .+ CMatSoA.ValuesVector[LargeTensorListIndex(1, 4, 4)] .* comp4)
        term2 = 2 .* (CMatSoA.ValuesVector[LargeTensorListIndex(2, 1, 4)] .* comp1 .+ CMatSoA.ValuesVector[LargeTensorListIndex(2, 2, 4)] .* comp2 .+
                            CMatSoA.ValuesVector[LargeTensorListIndex(2, 3, 4)] .* comp3 .+ CMatSoA.ValuesVector[LargeTensorListIndex(2, 4, 4)] .* comp4)
        term3 = 2 .* (CMatSoA.ValuesVector[LargeTensorListIndex(3, 1, 4)] .* comp1 .+ CMatSoA.ValuesVector[LargeTensorListIndex(3, 2, 4)] .* comp2 .+
                            CMatSoA.ValuesVector[LargeTensorListIndex(3, 3, 4)] .* comp3 .+ CMatSoA.ValuesVector[LargeTensorListIndex(3, 4, 4)] .* comp4)
        term4 = 2 .* (CMatSoA.ValuesVector[LargeTensorListIndex(4, 1, 4)] .* comp1 .+ CMatSoA.ValuesVector[LargeTensorListIndex(4, 2, 4)] .* comp2 .+
                            CMatSoA.ValuesVector[LargeTensorListIndex(4, 3, 4)] .* comp3 .+ CMatSoA.ValuesVector[LargeTensorListIndex(4, 4, 4)] .* comp4)
        XXValues = 0.5 .* (term1 .+ term3)
        XYValues = 0.5 .* (term4 .- term2)
        YXValues = 0.5 .* (term2 .+ term4)
        YYValues = 0.5 .* (term1 .- term3)
        return TensorSoA2D(XXValues, XYValues, YXValues, YYValues)
    end


    function PredictorCorrectorStepTensOdd2DSoA!(grid, OmegaSoA, PsiSoA, velocitySoA, sigmaVESoA, CMatSoA, dt, veParams, bcx, bcy, matDeriv)
        bcDerivX = BCDerivDict[bcx]
        bcDerivY = BCDerivDict[bcy]
        b = GetBoundariesFromConditions(grid, bcx, bcy)
        gradVSoA = AddTensorSoA2D(grid, OmegaSoA, PsiSoA)
        CContrPsiSoA = ContractWithCMatOddSoA(grid, CMatSoA, gradVSoA)
        CContrSigmaSoA = ContractWithCMatOddSoA(grid, CMatSoA, sigmaVESoA)
        m1 = TSMat2DSoA(grid, sigmaVESoA, OmegaSoA, PsiSoA, CContrPsiSoA, CContrSigmaSoA, velocitySoA, veParams.eta, veParams.Ds, bcDerivX, bcDerivY, matDeriv)
        predSigmaSoA = deepcopy(sigmaVESoA)
        predSigmaSoA.XXValues[b[1]:b[2],b[3]:b[4]] .= predSigmaSoA.XXValues[b[1]:b[2],b[3]:b[4]] .+ dt * m1.XXValues[b[1]:b[2],b[3]:b[4]]
        predSigmaSoA.XYValues[b[1]:b[2],b[3]:b[4]] .= predSigmaSoA.XYValues[b[1]:b[2],b[3]:b[4]] .+ dt * m1.XYValues[b[1]:b[2],b[3]:b[4]]
        predSigmaSoA.YXValues[b[1]:b[2],b[3]:b[4]] .= predSigmaSoA.YXValues[b[1]:b[2],b[3]:b[4]] .+ dt * m1.YXValues[b[1]:b[2],b[3]:b[4]]
        predSigmaSoA.YYValues[b[1]:b[2],b[3]:b[4]] .= predSigmaSoA.YYValues[b[1]:b[2],b[3]:b[4]] .+ dt * m1.YYValues[b[1]:b[2],b[3]:b[4]]
        SetTensorFromSoA2D!(CContrSigmaSoA, ContractWithCMatOddSoA(grid, CMatSoA, predSigmaSoA))
        m2 = TSMat2DSoA(grid, predSigmaSoA, OmegaSoA, PsiSoA, CContrPsiSoA, CContrSigmaSoA, velocitySoA, veParams.eta, veParams.Ds, bcDerivX, bcDerivY, matDeriv)
        sigmaVESoA.XXValues[b[1]:b[2],b[3]:b[4]] .= sigmaVESoA.XXValues[b[1]:b[2],b[3]:b[4]] .+ (0.5 .* dt .* (m1.XXValues[b[1]:b[2],b[3]:b[4]] .+ m2.XXValues[b[1]:b[2],b[3]:b[4]]))
        sigmaVESoA.XYValues[b[1]:b[2],b[3]:b[4]] .= sigmaVESoA.XYValues[b[1]:b[2],b[3]:b[4]] .+ (0.5 .* dt .* (m1.XYValues[b[1]:b[2],b[3]:b[4]] .+ m2.XYValues[b[1]:b[2],b[3]:b[4]]))
        sigmaVESoA.YXValues[b[1]:b[2],b[3]:b[4]] .= sigmaVESoA.YXValues[b[1]:b[2],b[3]:b[4]] .+ (0.5 .* dt .* (m1.YXValues[b[1]:b[2],b[3]:b[4]] .+ m2.YXValues[b[1]:b[2],b[3]:b[4]]))
        sigmaVESoA.YYValues[b[1]:b[2],b[3]:b[4]] .= sigmaVESoA.YYValues[b[1]:b[2],b[3]:b[4]] .+ (0.5 .* dt .* (m1.YYValues[b[1]:b[2],b[3]:b[4]] .+ m2.YYValues[b[1]:b[2],b[3]:b[4]]))
    end





end
