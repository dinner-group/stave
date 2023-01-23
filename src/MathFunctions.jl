module MathFunctions

    include("SharedStructs.jl")
    using .SharedStructs

    using SpecialFunctions

    export SafeACos,
    DotProduct3D, DotProduct2D,
    VectorNorm3D, VectorNorm2D,
    DotProductOnMesh3D, DotProductOnMesh2D,
    MatrixNorm3D, MatrixNorm2D,
    MatrixNormOnMesh3D, MatrixNormOnMesh2D,
    MatrixDoubleDot2D, MatrixDotVector4D,
    MatrixDotVector3D, MatrixDotVector2D,
    MatrixDotVectorOnMesh4D,
    MatrixDotVectorOnMesh3D, MatrixDotVectorOnMesh2D,
    MatrixDotMatrix3D, MatrixDotMatrix2D,
    MatrixDotMatrixOnMesh3D, MatrixDotMatrixOnMesh2D,
    MatrixTranspose2D, MatrixTranspose3D,
    MatrixInverse2D,
    DyadicProduct,
    vIComp2D,
    tIJComp2D,
    DotProductOnSoA2D, VectorNormSoA,
    DyadicProductOnSoA2D, MatrixDotMatrixOnSoA2D,
    MatrixDoubleDotMatrixOnSoA2D, IdentityMatrixSoA2D,
    MatrixDotVectorOnSoA2D, MatrixTransposeOnSoA2D,
    SubtractScalarSoA2D, SubtractScalarSoA2D!, 
    AddScalarSoA2D, AddScalarSoA2D!, 
    AddVectorSoA2D, AddVectorSoA2D!,
    AddTensorSoA2D, AddTensorSoA2D!,
    SubtractTensorSoA2D, SubtractTensorSoA2D!,
    MultipyScalarSoA2D, MultipyVectorSoA2D,
    MultipyVectorByScalarSoA2D, MultipyVectorByScalarSoA2D!,
    MultipyTensorSoA2D, MultipyLargeTensorSoA2D,
    MultipyTensorByScalarSoA2D,
    MultipyScalarSoA2D!, MultipyVectorSoA2D!,
    MultipyTensorSoA2D!, MultipyLargeTensorSoA2D!,
    MultipyLargeTensorByScalarSoA2D!,
    MultipyTensorByScalarSoA2D!,
    SetScalarFromSoA2D!, SetVectorFromSoA2D!,
    SetTensorFromSoA2D!, SetLargeTensorFromSoA2D!,
    ConvertVectorSoAToMesh2D, ConvertVectorMeshToSoA2D,
    ConvertTensorSoAToMesh2D, ConvertTensorMeshToSoA2D,
    FiniteDifferenceX, FiniteDifferenceY, FiniteDifferenceZ,
    FiniteSecondDifferenceX, FiniteSecondDifferenceY, FiniteSecondDifferenceZ,
    ExpandForGhostPoints2DSoA,
    PBCDeriv2DSoA, DirDeriv2DSoA, NeuDeriv2DSoA, BCDerivDict,
    GetBoundariesFromConditions,
    GradVecOnSoA2D,
    DivTensorOnSoA2D,
    DivVectorOnSoA2D,
    OmegaPsi2DSoA,
    PBCSmoothing



    ########################################################################################
    #
    #                               Linear Algebra
    #
    ########################################################################################

    function SafeACos(x)
        if x < -1.0
            x = -1.0
        elseif x > 1.0
            x = 1.0
        end
            ret = acos(x)
    end

    function DotProduct3D(a, b)
      return a[1] * b[1] +  a[2] * b[2] + a[3] * b[3]
    end

    function DotProduct2D(a, b)
      return a[1] * b[1] +  a[2] * b[2]
    end

    function VectorNorm3D(a)
      return sqrt(a[1] * a[1] +  a[2] * a[2] + a[3] * a[3])
    end

    function VectorNorm2D(a)
      return sqrt(a[1] * a[1] +  a[2] * a[2])
    end

    function DotProductOnMesh3D(vecMeshA, vecMeshB)
        return map(DotProduct3D, vecMeshA, vecMeshB)
    end

    function DotProductOnMesh2D(vecMeshA, vecMeshB)
        return map(DotProduct2D, vecMeshA, vecMeshB)
    end

    function MatrixNorm3D(matrix)
        return sqrt(matrix[1,1]^2 + matrix[1,2]^2 + matrix[1,3]^2 + matrix[2,1]^2 + matrix[2,2]^2 + matrix[2,3]^2 + matrix[3,1]^2 + matrix[3,2]^2 + matrix[3,3]^2)
    end

    function MatrixNorm2D(matrix)
        return sqrt(matrix[1,1]^2 + matrix[1,2]^2 + matrix[2,1]^2 + matrix[2,2]^2)
    end

    function MatrixNormOnMesh3D(matrixMesh)
        return map(MatrixNorm3D, matrixMesh)
    end

    function MatrixNormOnMesh2D(matrixMesh)
        return map(MatrixNorm2D, matrixMesh)
    end

    function MatrixDoubleDot2D(matrixA, matrixB) # 
        return matrixA[1,1] * matrixB[1,1] + matrixA[1,2] * matrixB[1,2] + matrixA[2,1] * matrixB[2,1] + matrixA[2,2] * matrixB[2,2]
    end

    function MatrixDotVector4D(matrix, vec)
      ret = [0.0 0.0 0.0 0.0];
      ret[1] = matrix[1,1] * vec[1] + matrix[1,2] * vec[2] + matrix[1,3] * vec[3] + matrix[1,4] * vec[4]
      ret[2] = matrix[2,1] * vec[1] + matrix[2,2] * vec[2] + matrix[2,3] * vec[3] + matrix[2,4] * vec[4]
      ret[3] = matrix[3,1] * vec[1] + matrix[3,2] * vec[2] + matrix[3,3] * vec[3] + matrix[3,4] * vec[4]
      ret[4] = matrix[4,1] * vec[1] + matrix[4,2] * vec[2] + matrix[4,3] * vec[3] + matrix[4,4] * vec[4]
      return ret
    end

    function MatrixDotVector3D(matrix, vec)
      ret = [0.0 0.0 0.0];
      ret[1] = matrix[1,1] * vec[1] + matrix[1,2] * vec[2] + matrix[1,3] * vec[3]
      ret[2] = matrix[2,1] * vec[1] + matrix[2,2] * vec[2] + matrix[2,3] * vec[3]
      ret[3] = matrix[3,1] * vec[1] + matrix[3,2] * vec[2] + matrix[3,3] * vec[3]
      return ret
    end

    function MatrixDotVector2D(matrix, vec)
      ret = [0.0 0.0];
      ret[1] = matrix[1,1] * vec[1] + matrix[1,2] * vec[2]
      ret[2] = matrix[2,1] * vec[1] + matrix[2,2] * vec[2]
      return ret
    end

    function MatrixDotVectorOnMesh4D(matrixMesh, vecMesh)
      return convert(Array{Vector{Real}},map(vec, map(MatrixDotVector4D, matrixMesh, vecMesh)))
    end

    function MatrixDotVectorOnMesh3D(matrixMesh, vecMesh)
      return convert(Array{Vector{Real}},map(vec, map(MatrixDotVector3D, matrixMesh, vecMesh)))
    end

    function MatrixDotVectorOnMesh2D(matrixMesh, vecMesh)
      return convert(Array{Vector{Real}},map(vec, map(MatrixDotVector2D, matrixMesh, vecMesh)))
    end

    function MatrixDotMatrix3D(mA, mB)
        ret = [0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0];
        ret[1, 1] = mA[1, 1] * mB[1, 1] + mA[1, 2] * mB[2, 1] + mA[1, 3] * mB[3, 1]
        ret[1, 2] = mA[1, 1] * mB[1, 2] + mA[1, 2] * mB[2, 2] + mA[1, 3] * mB[3, 2]
        ret[1, 3] = mA[1, 1] * mB[1, 3] + mA[1, 2] * mB[2, 3] + mA[1, 3] * mB[3, 3]
        ret[2, 1] = mA[2, 1] * mB[1, 1] + mA[2, 2] * mB[2, 1] + mA[2, 3] * mB[3, 1]
        ret[2, 2] = mA[2, 1] * mB[1, 2] + mA[2, 2] * mB[2, 2] + mA[2, 3] * mB[3, 2]
        ret[2, 3] = mA[2, 1] * mB[1, 3] + mA[2, 2] * mB[2, 3] + mA[2, 3] * mB[3, 3]
        ret[3, 1] = mA[3, 1] * mB[1, 1] + mA[3, 2] * mB[2, 1] + mA[3, 3] * mB[3, 1]
        ret[3, 2] = mA[3, 1] * mB[1, 2] + mA[3, 2] * mB[2, 2] + mA[3, 3] * mB[3, 2]
        ret[3, 3] = mA[3, 1] * mB[1, 3] + mA[3, 2] * mB[2, 3] + mA[3, 3] * mB[3, 3]
        return ret
    end

    function MatrixDotMatrix2D(mA, mB)
        ret = [0.0 0.0; 0.0 0.0];
        ret[1, 1] = mA[1, 1] * mB[1, 1] + mA[1, 2] * mB[2, 1]
        ret[1, 2] = mA[1, 1] * mB[1, 2] + mA[1, 2] * mB[2, 2]
        ret[2, 1] = mA[2, 1] * mB[1, 1] + mA[2, 2] * mB[2, 1]
        ret[2, 2] = mA[2, 1] * mB[1, 2] + mA[2, 2] * mB[2, 2]
        return ret
    end

    function MatrixDotMatrixOnMesh3D(matrixAMesh, matrixBMesh)
      return convert(Array{Matrix{Real}}, map(MatrixDotMatrix3D, matrixAMesh, matrixBMesh))
    end

    function MatrixDotMatrixOnMesh2D(matrixAMesh, matrixBMesh)
      return convert(Array{Matrix{Real}}, map(MatrixDotMatrix2D, matrixAMesh, matrixBMesh))
    end

    function MatrixTranspose3D(matrix)
      ret = similar(matrix)
      ret[1,1] = matrix[1,1]
      ret[1,2] = matrix[2,1]
      ret[1,3] = matrix[3,1]
      ret[2,1] = matrix[1,2]
      ret[2,2] = matrix[2,2]
      ret[2,3] = matrix[3,2]
      ret[3,1] = matrix[1,3]
      ret[3,2] = matrix[2,3]
      ret[3,3] = matrix[3,3]
      return ret
    end

    function MatrixTranspose2D(matrix)
      ret = similar(matrix)
      ret[1,1] = matrix[1,1]
      ret[1,2] = matrix[2,1]
      ret[2,1] = matrix[1,2]
      ret[2,2] = matrix[2,2]
      return ret
    end

    function MatrixInverse2D(matrix)
      ret = similar(matrix)
      det = matrix[1,1]*matrix[2,2] - matrix[1,2]*matrix[2,1]
      if det != 0.0
          ret[1,1] = matrix[2,2]
          ret[1,2] = -matrix[1,2]
          ret[2,1] = -matrix[2,1]
          ret[2,2] = matrix[1,1]
          ret  = ret ./ det
      else
          ret .= 0.0
      end
      return ret
    end

    function DyadicProduct(vecA, vecB) # works on any dimension
        n = size(vecA)[1]
        ret = zeros((n,n))
        for i=1:n, j=1:n
            ret[i,j] = vecA[i]*vecB[j]
        end
        return ret
    end

    function vIComp2D(grid, vMVals, I)
        return [vMVals[i,j][I] for i=1:grid.Nx, j=1:grid.Ny]
    end

    function tIJComp2D(grid, tMVals, I, J)
        return [tMVals[i,j][I,J] for i=1:grid.Nx, j=1:grid.Ny]
    end

    function DotProductOnSoA2D(grid, vSoAA, vSoAB)
        Values = (vSoAA.XValues .* vSoAA.XValues) .+ (vSoAA.YValues .* vSoAA.YValues)
        return ScalarSoA2D(Values)
    end


    function VectorNormSoA(vSoA)
        Values = sqrt.(vSoA.XValues.^ 2 .+ vSoA.YValues.^ 2)
        return ScalarSoA2D(Values)
    end

    function DyadicProductOnSoA2D(grid, vSoAA, vSoAB)
        XXValues = vSoAA.XValues .* vSoAB.XValues
        XYValues = vSoAA.XValues .* vSoAB.YValues
        YXValues = vSoAA.YValues .* vSoAB.XValues
        YYValues = vSoAA.YValues .* vSoAB.YValues
        return TensorSoA2D(XXValues, XYValues, YXValues, YYValues)
    end

    function MatrixDotMatrixOnSoA2D(grid, tSoAA, tSoAB) # computes A_{ik} B_{kj} 
        XXValues = tSoAA.XXValues .* tSoAB.XXValues .+ tSoAA.XYValues .* tSoAB.YXValues
        XYValues = tSoAA.XXValues .* tSoAB.XYValues .+ tSoAA.XYValues .* tSoAB.YYValues
        YXValues = tSoAA.YXValues .* tSoAB.XXValues .+ tSoAA.YYValues .* tSoAB.YXValues
        YYValues = tSoAA.YXValues .* tSoAB.XYValues .+ tSoAA.YYValues .* tSoAB.YYValues
        return TensorSoA2D(XXValues, XYValues, YXValues, YYValues)
    end

    function MatrixDoubleDotMatrixOnSoA2D(grid, tSoAA, tSoAB) # computes A_{ij} B_{ij} 
        values = (tSoAA.XXValues .* tSoAB.XXValues) .+ (tSoAA.XYValues .* tSoAB.XYValues) .+ (tSoAA.YXValues .* tSoAB.YXValues) .+ (tSoAA.YYValues .* tSoAB.YYValues)
        return ScalarSoA2D(values)
    end

    function IdentityMatrixSoA2D(grid, fac = 1.0)
        identity = TensorSoA2D(grid)
        identity.XXValues .= fac
        identity.YYValues .= fac
        return identity
    end

    function MatrixDotVectorOnSoA2D(grid, tSoA, vSoA)
        XValues = tSoA.XXValues .* vSoA.XValues .+ tSoA.XYValues .* vSoA.YValues
        YValues = tSoA.YXValues .* vSoA.XValues .+ tSoA.YYValues .* vSoA.YValues
        return VectorSoA2D(XValues, YValues)
    end

    function MatrixTransposeOnSoA2D(grid, tSoA)
        XXValues = tSoA.XXValues
        XYValues = tSoA.YXValues
        YXValues = tSoA.XYValues
        YYValues = tSoA.YYValues
        return TensorSoA2D(XXValues, XYValues, YXValues, YYValues)
    end

    function AddScalarSoA2D(grid, sSoAA, sSoAB)
        Values = sSoAA.Values .+ sSoAB.Values
        return ScalarSoA2D(Values)
    end

    function AddScalarSoA2D!(sSoAA, sSoAB)
        sSoAA.Values .= sSoAA.Values .+ sSoAB.Values
    end

    function SubtractScalarSoA2D(grid, sSoAA, sSoAB)
        Values = sSoAA.Values .- sSoAB.Values
        return ScalarSoA2D(Values)
    end

    function SubtractScalarSoA2D!(sSoAA, sSoAB)
        sSoAA.Values .= sSoAA.Values .- sSoAB.Values
    end

    function AddVectorSoA2D(grid, vSoAA, vSoAB)
        XValues = vSoAA.XValues .+ vSoAB.XValues
        YValues = vSoAA.YValues .+ vSoAB.YValues
        return VectorSoA2D(XValues, YValues)
    end

    function AddVectorSoA2D!(vSoA, vSoAO)
        vSoA.XValues .= vSoA.XValues .+ vSoAO.XValues
        vSoA.YValues .= vSoA.YValues .+ vSoAO.YValues
    end

    function AddTensorSoA2D(grid, tSoAA, tSoAB) # return A + B 
        XXValues = tSoAA.XXValues .+ tSoAB.XXValues
        XYValues = tSoAA.XYValues .+ tSoAB.XYValues
        YXValues = tSoAA.YXValues .+ tSoAB.YXValues
        YYValues = tSoAA.YYValues .+ tSoAB.YYValues
        return TensorSoA2D(XXValues, XYValues, YXValues, YYValues)
    end

    function AddTensorSoA2D!(tSoA, tSoAO) # A <- A + B 
        tSoA.XXValues .= tSoA.XXValues .+ tSoAO.XXValues
        tSoA.XYValues .= tSoA.XYValues .+ tSoAO.XYValues
        tSoA.YXValues .= tSoA.YXValues .+ tSoAO.YXValues
        tSoA.YYValues .= tSoA.YYValues .+ tSoAO.YYValues
    end

    function SubtractTensorSoA2D(grid, tSoAA, tSoAB) # return A - B 
        XXValues = tSoAA.XXValues .- tSoAB.XXValues
        XYValues = tSoAA.XYValues .- tSoAB.XYValues
        YXValues = tSoAA.YXValues .- tSoAB.YXValues
        YYValues = tSoAA.YYValues .- tSoAB.YYValues
        return TensorSoA2D(XXValues, XYValues, YXValues, YYValues)
    end

    function SubtractTensorSoA2D!(tSoA, tSoAO) # A <- A - B  
        tSoA.XXValues .= tSoA.XXValues .- tSoAO.XXValues
        tSoA.XYValues .= tSoA.XYValues .- tSoAO.XYValues
        tSoA.YXValues .= tSoA.YXValues .- tSoAO.YXValues
        tSoA.YYValues .= tSoA.YYValues .- tSoAO.YYValues
    end

    function MultipyScalarSoA2D(grid, sSoAO, a)
        Values = sSoAO.Values .* a
        return ScalarSoA2D(Values)
    end

    function MultipyVectorSoA2D(grid, vSoAO, a)
        XValues = vSoAO.XValues .* a
        YValues = vSoAO.YValues .* a
        return VectorSoA2D(XValues, YValues)
    end

    function MultipyVectorByScalarSoA2D(grid, vSoA, sSoA) # multiplies a vector field by a scalar field
        XValues = vSoA.XValues .* sSoA.Values
        YValues = vSoA.YValues .* sSoA.Values
        return VectorSoA2D(XValues, YValues)
    end

    function MultipyVectorByScalarSoA2D!(vSoA, sSoA) # multiplies a vector field by a scalar field
        vSoA.XValues .*= sSoA.Values
        vSoA.YValues .*= sSoA.Values
    end

    function MultipyTensorSoA2D(grid, tSoAO, a) # multiplies a tensor field by single scalar
        XXValues = tSoAO.XXValues .* a
        XYValues = tSoAO.XYValues .* a
        YXValues = tSoAO.YXValues .* a
        YYValues = tSoAO.YYValues .* a
        return TensorSoA2D(XXValues, XYValues, YXValues, YYValues)
    end

    function MultipyTensorByScalarSoA2D(grid, tSoAO, sSoA) # multiplies a tensor field by a scalar field
        XXValues = tSoAO.XXValues .* sSoA.Values
        XYValues = tSoAO.XYValues .* sSoA.Values
        YXValues = tSoAO.YXValues .* sSoA.Values
        YYValues = tSoAO.YYValues .* sSoA.Values
        return TensorSoA2D(XXValues, XYValues, YXValues, YYValues)
    end

    function MultipyLargeTensorSoA2D(grid, tSoAO, a)
        tSoA = LargeTensorSoA2D(grid, tSoAO.nR, tSoAO.nC)
        for r = 1:tSoA.nR, c = 1:tSoA.nC
            aInd = LargeTensorListIndex(r, c, tSoA.nC)
            tSoA.ValuesVector[aInd] .= tSoAO.ValuesVector[aInd] .* a
        end
        return tSoA
    end

    function MultipyScalarSoA2D!(sSoA, a)
        sSoA.Values .*= a
    end

    function MultipyVectorSoA2D!(vSoA, a)
        vSoA.XValues .*= a
        vSoA.YValues .*= a
    end

    function MultipyTensorSoA2D!(tSoA, a)
        tSoA.XXValues .*= a
        tSoA.XYValues .*= a
        tSoA.YXValues .*= a
        tSoA.YYValues .*= a
    end

    function MultipyTensorByScalarSoA2D!(tSoA, sSoA) # multiplies a tensor field by a scalar field
        tSoA.XXValues .*= sSoA.Values
        tSoA.XYValues .*= sSoA.Values
        tSoA.YXValues .*= sSoA.Values
        tSoA.YYValues .*= sSoA.Values
    end

    function MultipyLargeTensorSoA2D!(tSoA, a)
        for r = 1:tSoA.nR, c = 1:tSoA.nC
            aInd = LargeTensorListIndex(r, c, tSoA.nC)
            tSoA.ValuesVector[aInd] .*= a
        end
    end

    function MultipyLargeTensorByScalarSoA2D!(tSoA, sSoA)
        for r = 1:tSoA.nR, c = 1:tSoA.nC
            aInd = LargeTensorListIndex(r, c, tSoA.nC)
            tSoA.ValuesVector[aInd] .*= sSoA.Values
        end
    end

    function SetScalarFromSoA2D!(dSoA, sSoA) # must be same size
        dSoA.Values .= sSoA.Values
    end

    function SetVectorFromSoA2D!(dSoA, sSoA) # must be same size
        dSoA.XValues .= sSoA.XValues
        dSoA.YValues .= sSoA.YValues
    end

    function SetTensorFromSoA2D!(dSoA, sSoA) # must be same size
        dSoA.XXValues .= sSoA.XXValues
        dSoA.XYValues .= sSoA.XYValues
        dSoA.YXValues .= sSoA.YXValues
        dSoA.YYValues .= sSoA.YYValues
    end

    function SetLargeTensorFromSoA2D!(dSoA, sSoA) # must be same size
        for r in 1:dSoA.nR, c in 1:dSoA.nC
            aInd = LargeTensorListIndex(r, c, dSoA.nC)
            dSoA.ValuesVector[aInd] .= sSoA.ValuesVector[aInd]
        end
    end


    ########################################################################################
    #
    #             Discrete Derivatives using Struct of Array structures
    #
    ########################################################################################

    function ConvertVectorSoAToMesh2D(grid, vSoA)
        vM = VectorMesh2D(grid)
        for i = 1:grid.Nx, j = 1:grid.Ny
            vM.Values[i,j] = [vSoA.XValues[i,j], vSoA.YValues[i,j]]
        end
        return vM
    end

    function ConvertVectorMeshToSoA2D(grid, vM)
        vSoA = VectorSoA2D(grid)
        for i = 1:grid.Nx, j = 1:grid.Ny
            vSoA.XValues[i,j] = vM.Values[i,j][1]
            vSoA.YValues[i,j] = vM.Values[i,j][2]
        end
        return vSoA
    end

    function ConvertTensorSoAToMesh2D(grid, tSoA)
        tM = TensorMesh2D(grid)
        for i = 1:grid.Nx, j = 1:grid.Ny
            tM.Values[i,j] = [tSoA.XXValues[i,j] tSoA.XYValues[i,j]; tSoA.YXValues[i,j] tSoA.YYValues[i,j] ]
        end
        return tM
    end

    function ConvertTensorMeshToSoA2D(grid, tM)
        tSoA = TensorSoA2D(grid)
        for i = 1:grid.Nx, j = 1:grid.Ny
            tSoA.XXValues[i,j] = tM.Values[i,j][1,1]
            tSoA.XYValues[i,j] = tM.Values[i,j][1,2]
            tSoA.YXValues[i,j] = tM.Values[i,j][2,1]
            tSoA.YYValues[i,j] = tM.Values[i,j][2,2]
        end
        return tSoA
    end

    @inline function FiniteDifferenceX4(vals) # these don't include the discretization length
        # absorbs factor of 2 in common with second order FD
        ret = Array{Float64}(undef, size(vals))
        for j = 1:size(vals)[2], i = 3:size(vals)[1]-2
            ret[i,j] = - vals[i+2,j] / 6 + (4/3) * vals[i+1,j] - (4/3) * vals[i-1,j] + vals[i-2,j] / 6
        end
        for j = 1:size(vals)[2]
            ret[1,j] = - vals[3,j] / 6 + (4/3) * vals[2,j] - (4/3) * vals[end,j] + vals[end-1,j] / 6
            ret[2,j] = - vals[4,j] / 6 + (4/3) * vals[3,j] - (4/3) * vals[1,j] + vals[end,j] / 6
            ret[end-1,j] = - vals[1,j] / 6 + (4/3) * vals[end,j] - (4/3) * vals[end-2,j] + vals[end-3,j] / 6
            ret[end,j] = - vals[2,j] / 6 + (4/3) * vals[1,j] - (4/3) * vals[end-1,j] + vals[end-2,j] / 6
        end
        return ret
    end

    @inline function FiniteDifferenceY4(vals) # these don't include the discretization length
        # absorbs factor of 2 in common with second order FD
        ret = Array{Float64}(undef, size(vals))
        for j = 3:size(vals)[2]-2, i = 1:size(vals)[1]
            ret[i,j] = - vals[i,j+2] / 6 + (4/3) * vals[i,j+1] - (4/3) * vals[i,j-1] + vals[i,j-2] / 6
        end
        for i = 1:size(vals)[1]
            ret[i,1] = - vals[i,3] / 6 + (4/3) * vals[i,2] - (4/3) * vals[i,end] + vals[i,end-1] / 6
            ret[i,2] = - vals[i,4] / 6 + (4/3) * vals[i,3] - (4/3) * vals[i,1] + vals[i,end] / 6
            ret[i,end-1] = - vals[i,1] / 6 + (4/3) * vals[i,end] - (4/3) * vals[i,end-2] + vals[i,end-3] / 6
            ret[i,end] = - vals[i,2] / 6 + (4/3) * vals[i,1] - (4/3) * vals[i,end-1] + vals[i,end-2] / 6
        end
        return ret
    end

    @inline function FiniteSecondDifferenceX4(vals) # these don't include the discretization length
        # absorbs factor of 2 in common with second order FD
        ret = Array{Float64}(undef, size(vals))
        for j = 1:size(vals)[2], i = 3:size(vals)[1]-2
            ret[i,j] = - vals[i+2,j] / 6 + (8/3) * vals[i+1,j]  - 5 * vals[i,j] + (8/3) * vals[i-1,j] - vals[i-2,j] / 6
        end
        for j = 1:size(vals)[2]
            ret[1,j] = - vals[3,j] / 6 + (8/3) * vals[2,j] - 5 * vals[1,j] + (8/3) * vals[end,j] - vals[end-1,j] / 6
            ret[2,j] = - vals[4,j] / 6 + (8/3) * vals[3,j] - 5 * vals[2,j] + (8/3) * vals[1,j] - vals[end,j] / 6
            ret[end-1,j] = - vals[1,j] / 6 + (8/3) * vals[end,j] - 5 * vals[end-1,j] + (8/3) * vals[end-2,j] - vals[end-3,j] / 6
            ret[end,j] = - vals[2,j] / 6 + (8/3) * vals[1,j] - 5 * vals[end,j] + (8/3) * vals[end-1,j] - vals[end-2,j] / 6
        end
        return ret
    end

    @inline function FiniteSecondDifferenceY4(vals) # these don't include the discretization length
        # absorbs factor of 2 in common with second order FD
        ret = Array{Float64}(undef, size(vals))
        for j = 3:size(vals)[2]-2, i = 1:size(vals)[1]
            ret[i,j] = - vals[i,j+2] / 6 + (8/3) * vals[i,j+1] - 5 * vals[i,j] + (8/3) * vals[i,j-1] - vals[i,j-2] / 6
        end
        for i = 1:size(vals)[1]
            ret[i,1] = - vals[i,3] / 6 + (8/3) * vals[i,2] - 5 * vals[i,1] + (8/3) * vals[i,end] - vals[i,end-1] / 6
            ret[i,2] = - vals[i,4] / 6 + (8/3) * vals[i,3] - 5 * vals[i,2] + (8/3) * vals[i,1] - vals[i,end] / 6
            ret[i,end-1] = - vals[i,1] / 6 + (8/3) * vals[i,end] - 5 * vals[i,end-1] + (8/3) * vals[i,end-2] - vals[i,end-3] / 6
            ret[i,end] = - vals[i,2] / 6 + (8/3) * vals[i,1] - 5 * vals[i,end] + (8/3) * vals[i,end-1] - vals[i,end-2] / 6
        end
        return ret
    end

    @inline function ExpandForGhostPoints2DSoA4(vals)
        newVals = Array{Float64}(undef, (size(vals)[1] + 4, size(vals)[2] + 4))
        newVals[3:end-2,3:end-2] .= @view(vals[:,:])
        newVals[3:end-2,1] .= @view(vals[:,3])
        newVals[3:end-2,2] .= @view(vals[:,2])
        newVals[3:end-2,end] .= @view(vals[:,end-2])
        newVals[3:end-2,end-1] .= @view(vals[:,end-1])
        newVals[1,3:end-2] .= @view(vals[3,:])
        newVals[2,3:end-2] .= @view(vals[2,:])
        newVals[end,3:end-2] .= @view(vals[end-2,:])
        newVals[end-1,3:end-2] .= @view(vals[end-1,:])
        return newVals
    end

    @inline function FiniteDifferenceX(vals) # these don't include the discretization length
    # absorbs factor of 2 in common with second order FD
        ret = Array{Float64}(undef, size(vals))
        for j = 2:size(vals)[2]-1, i = 2:size(vals)[1]-1
            ret[i,j] = (vals[i+1,j+1] - vals[i-1,j+1]) / 6 + 2 * (vals[i+1,j] - vals[i-1,j]) / 3 + (vals[i+1,j-1] - vals[i-1,j-1]) / 6
        end
        for j = 2:size(vals)[2]-1
            ret[1,j] = (vals[2,j+1] - vals[end,j+1]) / 6 + 2 * (vals[2,j] - vals[end,j]) / 3 + (vals[2,j-1] - vals[end,j-1]) / 6
            ret[end,j] = (vals[1,j+1] - vals[end-1,j+1]) / 6 + 2 * (vals[1,j] - vals[end-1,j]) / 3 + (vals[1,j-1] - vals[end-1,j-1]) / 6
        end
        for i = 2:size(vals)[1]-1
            ret[i,1] = (vals[i+1,2] - vals[i-1,2]) / 6 + 2 * (vals[i+1,1] - vals[i-1,1]) / 3 + (vals[i+1,end] - vals[i-1,end]) / 6
            ret[i,end] = (vals[i+1,1] - vals[i-1,1]) / 6 + 2 * (vals[i+1,end] - vals[i-1,end]) / 3 + (vals[i+1,end-1] - vals[i-1,end-1]) / 6
        end
        ret[1,1] = (vals[2,2] - vals[end,2]) / 6 + 2 * (vals[2,1] - vals[end,1]) / 3 + (vals[2,end] - vals[end,end]) / 6
        ret[end,1] = (vals[1,2] - vals[end-1,2]) / 6 + 2 * (vals[1,1] - vals[end-1,1]) / 3 + (vals[1,end] - vals[end-1,end]) / 6
        ret[1,end] = (vals[2,1] - vals[end,1]) / 6 + 2 * (vals[2,end] - vals[end,end]) / 3 + (vals[2,end-1] - vals[end,end-1]) / 6
        ret[end,end] = (vals[1,1] - vals[end-1,1]) / 6 + 2 * (vals[1,end] - vals[end-1,end]) / 3 + (vals[1,end-1] - vals[end-1,end-1]) / 6
        return ret
    end

    @inline function FiniteDifferenceY(vals) # these don't include the discretization length
    # absorbs factor of 2 in common with second order FD
        return FiniteDifferenceX(vals')'
    end

    @inline function FiniteSecondDifferenceX(vals) # these don't include the discretization length
    # absorbs factor of 2 in common with second order FD
        ret = Array{Float64}(undef, size(vals))
        for j = 2:size(vals)[2]-1, i = 2:size(vals)[1]-1
            ret[i,j] = (vals[i+1,j+1] - 2 * vals[i,j+1] + vals[i-1,j+1]) / 6 + 5 * (vals[i+1,j] - 2 * vals[i,j] + vals[i-1,j]) / 3 + (vals[i+1,j-1] - 2 * vals[i,j-1] + vals[i-1,j-1]) / 6
        end
        for j = 2:size(vals)[2]-1
            ret[1,j] = (vals[2,j+1] - 2 * vals[1,j+1] + vals[end,j+1]) / 6 + 5 * (vals[2,j] - 2 * vals[1,j] + vals[end,j]) / 3 + (vals[2,j-1] - 2 * vals[1,j-1] + vals[end,j-1]) / 6
            ret[end,j] = (vals[1,j+1] - 2 * vals[end,j+1] + vals[end-1,j+1]) / 6 + 5 * (vals[1,j] - 2 * vals[end,j] + vals[end-1,j]) / 3 + (vals[1,j-1] - 2 * vals[end,j-1] + vals[end-1,j-1]) / 6
        end
        for i = 2:size(vals)[1]-1
            ret[i,1] = (vals[i+1,2] - 2 * vals[i,2] + vals[i-1,2]) / 6 + 5 * (vals[i+1,1] - 2 * vals[i,1] + vals[i-1,1]) / 3 + (vals[i+1,end] - 2 * vals[i,end] + vals[i-1,end]) / 6
            ret[i,end] = (vals[i+1,1] - 2 * vals[i,1] + vals[i-1,1]) / 6 + 5 * (vals[i+1,end] - 2 * vals[i,end] + vals[i-1,end]) / 3 + (vals[i+1,end-1] - 2 * vals[i,end-1] + vals[i-1,end-1]) / 6
        end
        ret[1,1] = (vals[2,2] - 2 * vals[1,2] + vals[end,2]) / 6 + 5 * (vals[2,1] - 2 * vals[1,1] + vals[end,1]) / 3 + (vals[2,end] - 2 * vals[1,end] + vals[end,end]) / 6
        ret[end,1] = (vals[1,2] - 2 * vals[end,2] + vals[end-1,2]) / 6 + 5 * (vals[1,1] - 2 * vals[end,1] + vals[end-1,1]) / 3 + (vals[1,end] - 2 * vals[end,end] + vals[end-1,end]) / 6
        ret[1,end] = (vals[2,1] - 2 * vals[1,1] + vals[end,1]) / 6 + 5 * (vals[2,end] - 2 * vals[1,end] + vals[end,end]) / 3 + (vals[2,end-1] - 2 * vals[1,end-1] + vals[end,end-1]) / 6
        ret[end,end] = (vals[1,1] - 2 * vals[end,1] + vals[end-1,1]) / 6 + 5 * (vals[1,end] - 2 * vals[end,end] + vals[end-1,end]) / 3 + (vals[1,end-1] - 2 * vals[end,end-1] + vals[end-1,end-1]) / 6
        return ret
    end

    @inline function FiniteSecondDifferenceY(vals) # these don't include the discretization length
    # absorbs factor of 2 in common with second order FD
        return FiniteSecondDifferenceX(vals')'
    end


    @inline function FiniteDifferenceX2(vals) # these don't include the discretization length
        ret = Array{Float64}(undef, size(vals))
        for j = 1:size(vals)[2], i = 2:size(vals)[1]-1
            @fastmath ret[i,j] = vals[i+1,j] - vals[i-1,j]
        end
        for j = 1:size(vals)[2]
            @fastmath ret[1,j] = vals[2,j] - vals[end,j]
            @fastmath ret[end,j] = vals[1,j] - vals[end-1,j]
        end
        return ret
    end

    @inline function FiniteDifferenceY2(vals) # these don't include the discretization length
        ret = Array{Float64}(undef, size(vals))
        for j = 2:size(vals)[2]-1, i = 1:size(vals)[1]
            @fastmath ret[i,j] = vals[i,j+1] - vals[i,j-1]
        end
        for i = 1:size(vals)[1]
            @fastmath ret[i,1] = vals[i,2] - vals[i,end]
            @fastmath ret[i,end] = vals[i,1] - vals[i,end-1]
        end
        return ret
    end

    @inline function FiniteSecondDifferenceX2(vals) # these don't include the discretization length
        ret = Array{Float64}(undef, size(vals))
        for j = 1:size(vals)[2], i = 2:size(vals)[1]-1
            @fastmath ret[i,j] = vals[i+1,j] + vals[i-1,j] - 2.0 * vals[i,j]
        end
        for j = 1:size(vals)[2]
            @fastmath ret[1,j] = vals[2,j] + vals[end,j] - 2.0 * vals[1,j]
            @fastmath ret[end,j] = vals[1,j] + vals[end-1,j] - 2.0 * vals[end,j]
        end
        return ret
    end

    @inline function FiniteSecondDifferenceY2(vals) # these don't include the discretization length
        ret = Array{Float64}(undef, size(vals))
        for j = 2:size(vals)[2]-1, i = 1:size(vals)[1]
            @fastmath ret[i,j] = vals[i,j+1] + vals[i,j-1] - 2.0 * vals[i,j]
        end
        for i = 1:size(vals)[1]
            @fastmath ret[i,1] = vals[i,2] + vals[i,end] - 2.0 * vals[i,1]
            @fastmath ret[i,end] = vals[i,1] + vals[i,end-1] - 2.0 * vals[i,end]
        end
        return ret
    end

    @inline function ExpandForGhostPoints2DSoA(vals)
        newVals = Array{Float64}(undef, (size(vals)[1] + 2, size(vals)[2] + 2))
        newVals[2:end-1,2:end-1] .= @view(vals[:,:])
        newVals[2:end-1,1] .= @view(vals[:,2])
        newVals[2:end-1,end] .= @view(vals[:,end-1])
        newVals[1,2:end-1] .= @view(vals[2,:])
        newVals[end,2:end-1] .= @view(vals[end-1,:])
        newVals[1,1] = vals[2,2]
        newVals[1,end] = vals[2, end-1]
        newVals[end,1] = vals[end-1, 2]
        newVals[end,end] = vals[end-1, end-1]
        return newVals
    end

    function PBCDeriv2DSoA(vals, derivFunc)
        deriv = derivFunc(vals)
        return deriv
    end

    function NeuDeriv2DSoA(vals, derivFunc)
        newVals = ExpandForGhostPoints2DSoA(vals)
        deriv = derivFunc(newVals)
        return deriv[2:end-1, 2:end-1]
    end

    function NeuDeriv2DSoA4(vals, derivFunc)
        newVals = ExpandForGhostPoints2DSoA(vals)
        deriv = derivFunc(newVals)
        return deriv[3:end-2, 3:end-2]
    end

    function DirDeriv2DSoA4(vals, derivFunc) # for fourth order
        deriv = derivFunc(vals)
        deriv[1,:] .= deriv[3,:]
        deriv[2,:] .= deriv[3,:]
        deriv[end,:] .= deriv[end-2,:]
        deriv[end-1,:] .= deriv[end-2,:]
        deriv[:,1] .= deriv[:,3]
        deriv[:,2] .= deriv[:,3]
        deriv[:,end] .= deriv[:,end-2]
        deriv[:,end-1] .= deriv[:,end-2]
        return deriv
    end

    function DirDeriv2DSoA(vals, derivFunc) # for fourth order
        deriv = derivFunc(vals)
        deriv[1,:] .= deriv[2,:]
        deriv[end,:] .= deriv[end-1,:]
        deriv[:,1] .= deriv[:,2]
        deriv[:,end] .= deriv[:,end-1]
        return deriv
    end

    global BCDerivDict = Dict([
            "pbc" => PBCDeriv2DSoA,
            "neu" => NeuDeriv2DSoA,
            "bbc" => DirDeriv2DSoA,
            "dir" => DirDeriv2DSoA
            ])

    function GetBoundariesFromConditions(grid, bcx, bcy)
        if (bcx != "dir") && (bcy != "dir")
            b = (1, grid.Nx, 1, grid.Ny)
        elseif (bcx == "dir") && (bcy != "dir")
            b = (2, grid.Nx-1, 1, grid.Ny)
        elseif (bcx != "dir") && (bcy == "dir")
            b = (1, grid.Nx, 2, grid.Ny-1)  
        else
            b = (2, grid.Nx-1, 2, grid.Ny-1)  
        end
        return b
    end

    function GradVecOnSoA2D(grid, vSoA, bcDerivX, bcDerivY) 

        gradV = TensorSoA2D(grid);
        gradV.XXValues .= bcDerivX(vSoA.XValues, FiniteDifferenceX) ./ (2.0 * grid.dx)
        gradV.XYValues .= bcDerivX(vSoA.YValues, FiniteDifferenceX) ./ (2.0 * grid.dx)
        gradV.YXValues .= bcDerivY(vSoA.XValues, FiniteDifferenceY) ./ (2.0 * grid.dx)
        gradV.YYValues .= bcDerivY(vSoA.YValues, FiniteDifferenceY) ./ (2.0 * grid.dx)

        return gradV
    end

    function DivTensorOnSoA2D(grid, tSoA, bcDerivX, bcDerivY)

        divT = VectorSoA2D(grid);
        divT.XValues .= (bcDerivX(tSoA.XXValues, FiniteDifferenceX) .+ bcDerivY(tSoA.YXValues, FiniteDifferenceY)) ./ (2.0 * grid.dx)
        divT.YValues .= (bcDerivX(tSoA.XYValues, FiniteDifferenceX) .+ bcDerivY(tSoA.YYValues, FiniteDifferenceY)) ./ (2.0 * grid.dx)

        return divT
    end

    function DivVectorOnSoA2D(grid, vSoA, bcDerivX, bcDerivY)

        Values = (bcDerivX(vSoA.XValues, FiniteDifferenceX) .+ bcDerivY(vSoA.YValues, FiniteDifferenceY)) ./ (2.0 * grid.dx)

        return ScalarSoA2D(Values)
    end

    function OmegaPsi2DSoA(grid, velocitySoA, bcx, bcy)

        bcDerivX = BCDerivDict[bcx]
        bcDerivY = BCDerivDict[bcy]
        gradV = GradVecOnSoA2D(grid, velocitySoA, bcDerivX, bcDerivY)

        Omega0Values = 0.0 .* gradV.XYValues
        OmegaXYValues = 0.5 .* (gradV.XYValues .- gradV.YXValues)
        OmegaYXValues = 0.5 .* (gradV.YXValues .- gradV.XYValues)
        OmegaSoA = TensorSoA2D(Omega0Values, OmegaXYValues, OmegaYXValues, Omega0Values)

        PsiXXValues = gradV.XXValues
        PsiXYValues = 0.5 .* (gradV.XYValues .+ gradV.YXValues)
        PsiYXValues = 0.5 .* (gradV.XYValues .+ gradV.YXValues)
        PsiYYValues = gradV.YYValues
        PsiSoA = TensorSoA2D(PsiXXValues, PsiXYValues, PsiYXValues, PsiYYValues)

        return (OmegaSoA, PsiSoA)
    end

    function PBCSmoothing(grid, vSoA)
        xValues = (circshift(vSoA.XValues, (0,1)) .+ circshift(vSoA.XValues, (0,-1)) .+ 
        circshift(vSoA.XValues, (1,0)) .+ circshift(vSoA.XValues, (-1,0)) .+ circshift(vSoA.XValues, (0,0))) ./ 5
        yValues = (circshift(vSoA.YValues, (0,1)) .+ circshift(vSoA.YValues, (0,-1)) .+ 
        circshift(vSoA.YValues, (1,0)) .+ circshift(vSoA.YValues, (-1,0)) .+ circshift(vSoA.YValues, (0,0))) ./ 5
        return VectorSoA2D(xValues, yValues)
    end


end
