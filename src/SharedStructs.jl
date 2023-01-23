module SharedStructs

    export Point,
    Grid2D,
    GetGridPoint2D,
    ScalarSoA2D,
    VectorSoA2D,
    TensorSoA2D,
    ScalarMesh2D,
    VectorMesh2D,
    TensorMesh2D,
    LargeTensorListIndex,
    LargeTensorArrayIndex,
    LargeTensorSoA2D,
    Ind2D,
    LBBCID,
    GetBulkBoundaryInds2D

    Point = Vector{AbstractFloat}

    abstract type Grid end

    struct Grid2D <: Grid
      Nx::Int
      Ny::Int
      dx::AbstractFloat
    end

    Ind2D = Tuple{Int, Int}

    struct ScalarSoA2D
        Values::Array{Float64}
        function ScalarSoA2D(grid, a::Float64 = 0.0)
            a = Float64(a)
            s = (grid.Nx, grid.Ny)
            Values = Array{Float64}(undef, s...)
            for j in 1:s[2], i in 1:s[1]
                Values[i,j] = a;
            end
            return new(Values)
        end
        function ScalarSoA2D(Values::Array{Float64})
            return new(Values)
        end
    end

    struct VectorSoA2D
        XValues::Array{Float64}
        YValues::Array{Float64}
        function VectorSoA2D(grid, a::Float64 = 0.0)
            a = Float64(a)
            s = (grid.Nx, grid.Ny)
            XValues = Array{Float64}(undef, s...)
            YValues = Array{Float64}(undef, s...)
            for j in 1:s[2], i in 1:s[1]
                XValues[i,j] = a;
                YValues[i,j] = a;
            end
            return new(XValues, YValues)
        end
        function VectorSoA2D(XValues::Array{Float64}, YValues::Array{Float64})
            return new(XValues, YValues)
        end
    end

    struct TensorSoA2D
        XXValues::Array{Float64}
        XYValues::Array{Float64}
        YXValues::Array{Float64}
        YYValues::Array{Float64}
        function TensorSoA2D(grid, a::Float64 = 0.0)
            a = Float64(a)
            s = (grid.Nx, grid.Ny)
            XXValues = Array{Float64}(undef, s...)
            XYValues = Array{Float64}(undef, s...)
            YXValues = Array{Float64}(undef, s...)
            YYValues = Array{Float64}(undef, s...)
            for j in 1:s[2], i in 1:s[1]
                XXValues[i,j] = a;
                XYValues[i,j] = a;
                YXValues[i,j] = a;
                YYValues[i,j] = a;
            end
            return new(XXValues, XYValues, YXValues, YYValues)
        end
        function TensorSoA2D(XXValues::Array{Float64}, XYValues::Array{Float64}, YXValues::Array{Float64}, YYValues::Array{Float64})
            return new(XXValues, XYValues, YXValues, YYValues)
        end
    end

    @inline function LargeTensorListIndex(r, c, nC)
        return (r-1)*nC + c
    end

    @inline function LargeTensorArrayIndex(i, nC)
        return (Int(floor((i-1)/nC))+1, (i-1)%nC + 1)
    end

    struct LargeTensorSoA2D
        ValuesVector::Vector{Array{Float64}} # linear index in column major order
        nR::Int
        nC::Int
        function LargeTensorSoA2D(grid, nR, nC, a::Float64 = 0.0)
            a = Float64(a)
            ValuesVector = [];
            s = (grid.Nx, grid.Ny)
            for c in 1:nC, r in 1:nR
                tempValues = Array{Float64}(undef, s...)
                for j in 1:s[2], i in 1:s[1]
                    tempValues[i,j] = a;
                    tempValues[i,j] = a;
                    tempValues[i,j] = a;
                    tempValues[i,j] = a;
                end
                push!(ValuesVector, tempValues)
            end
            return new(ValuesVector, nR, nC)
        end
    end

    struct ScalarMesh2D
        Values::Array{Real}
        function ScalarMesh2D(grid)
            s = (grid.Nx, grid.Ny)
            scalarMesh = Array{Real}(undef, s...)
            for j in 1:s[2], i in 1:s[1]
                scalarMesh[i,j] = 0.0;
            end
            return new(scalarMesh)
        end
    end

    struct VectorMesh2D
        Values::Array{Vector{Real}}

        function VectorMesh2D(grid)
            s = (grid.Nx, grid.Ny)
            vectorMesh = Array{Vector{Real}}(undef, s...)
            for j in 1:s[2], i in 1:s[1]
                vectorMesh[i,j] = [0.0, 0.0];
            end
            return new(vectorMesh)
        end
    end

    struct TensorMesh2D
        Values::Array{Matrix{Real}}
        function TensorMesh2D(grid)
            s = (grid.Nx, grid.Ny)
            tensorMesh = Array{Matrix{Real}}(undef, s...)
            for j in 1:s[2], i in 1:s[1]
                tensorMesh[i,j] = [0.0 0.0; 0.0 0.0];
            end
            return new(tensorMesh)
        end
    end

    struct LBBCID # lattice Boltzmann boundary condition ID
        Type::String
        Vel::Real
        function LBBCID(bcStr)
            if bcStr[1:3] == "pbc"
                t = "pbc"
                v = 0.0 
            else # case bb
                t = "bbc"
                v = parse(Float64, bcStr[5:end]) # expected format for bcStr is "bbc_XXX" where XXX is the wall speed in LU's
            end
            return new(t,v)
        end
    end

    #struct FDBCID # finite difference boundary condition ID


    function GetGridPoint2D(i, j, grid)
      return Point([i * grid.dx, j * grid.dx])
    end


    function GetBulkBoundaryInds2D(grid)
        (Nx, Ny) = (grid.Nx, grid.Ny)
        bulkInds = Vector{Ind2D}([])
        mXInds = Vector{Ind2D}([])
        pXInds = Vector{Ind2D}([])
        mYInds = Vector{Ind2D}([])
        pYInds = Vector{Ind2D}([])
        cornerInds = Vector{Ind2D}([])
        for i = 1:Nx, j = 1:Ny
            ind = Ind2D([i,j])
            if i == 1
                if (j != 1) && (j != Ny)
                    push!(mXInds, ind)
                end 
            elseif i == Nx
                if (j != 1) && (j != Ny)
                    push!(pXInds, ind)
                end 
            elseif j == 1
                push!(mYInds, ind)
            elseif j == Ny
                push!(pYInds, ind)
            else
                push!(bulkInds, ind)
            end
        end
        xmym = Ind2D([1,1])
        xmyp = Ind2D([1,Ny])
        xpym = Ind2D([Nx,1])
        xpyp = Ind2D([Nx,Ny])
        cornerInds = [xmym, xmyp, xpym, xpyp]
        return [bulkInds, [mXInds, pXInds, mYInds, pYInds, cornerInds]]
    end



end
