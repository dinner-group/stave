module TestCases

    include("MathFunctions.jl")
    using .MathFunctions

    include("SharedStructs.jl")
    using .SharedStructs
    using Distributions

########################################################################################
#
#                               Taylor Green
#
########################################################################################

    function TaylorGreenVortexXYT(x, y, t, grid, iParams)

      (Nx, Ny) = (grid.Nx, grid.Ny)
      (nu, uMax, rho0) = iParams

      kX = 2.0 * pi / Nx
      kY = 2.0 * pi / Ny
      td = 1.0 / (nu * (kX * kX + kY * kY))
      X = x + 0.5
      Y = y + 0.5
      ux = -uMax * sqrt(kY / kX) * cos(kX * X) * sin(kY * Y) * exp(-1.0 * t / td)
      uy = uMax * sqrt(kX / kY) * sin(kX * X) * cos(kY * Y) * exp(-1.0 * t / td)
      p = -0.25 * rho0 * uMax * uMax * ( (kY / kX) * cos(2.0 * kX * X) + (kX / kY) * cos(2.0 * kY * Y)) * exp(-2.0 * t / td)
      rho = rho0 + 3.0 * p

      return (rho, ux, uy)

    end


    function TaylorGreenVortexT2D!(t, density, velocityVectorMesh, grid, iParams)

      (Nx, Ny) = (grid.Nx, grid.Ny)
      (nu, uMax, rho0) = iParams

      for x = 1:Nx, y = 1:Ny
          (rho, ux, uy) = TaylorGreenVortexXYT(x, y, t, grid, iParams)
          density.Values[x,y] = rho
          velocityVectorMesh.Values[x,y] = [ux, uy]

      end
    end


    function TaylorGreenVortexT2D(t, grid, iParams)

      (Nx, Ny) = (grid.Nx, grid.Ny)
      retDensity = Array{Real}(undef, (Nx, Ny))
      retVVM = Array{Vector{Real}}(undef, (Nx, Ny))
      for x = 1:Nx, y = 1:Ny
          (rho, ux, uy) = TaylorGreenVortexXYT(x, y, t, grid, iParams)
          retDensity[x,y] = rho
          retVVM[x,y] = [ux, uy]
      end
      return (retDensity, retVVM)

    end

    function TaylorGreenVortexT2DStacked(t, grid, iParams)

      (Nx, Ny) = (grid.Nx, grid.Ny)
      retDensity = Array{Real}(undef, (Nx, Ny))
      retVVM = Array{Vector{Real}}(undef, (Nx, Ny))
      half = Ny/4

      for x = 1:Nx, y = 1:Ny
          (rho, ux, uy) = TaylorGreenVortexXYT(x, y - half, t, grid, iParams)
          retDensity[x,y] = rho
          if y > half
              retVVM[x,y] = [ux, uy]
          else
              retVVM[x,y] = [ux, uy]
          end
      end

      return (retDensity, retVVM)
    end

########################################################################################
#
#                               Custom
#
########################################################################################

    function TaylorGreenVortexSpace(grid, x, y)

      (Nx, Ny) = (grid.Nx, grid.Ny)
      kX = 2.0 * pi / Nx
      kY = 2.0 * pi / Ny
      ux = -sqrt(kY / kX) * cos(kX * x) * sin(kY * y)
      uy = sqrt(kX / kY) * sin(kX * x) * cos(kY * y)
      return (ux, uy)

    end

    function TaylorGreenVortexT2DTiled(grid, NVX, NVY; ccw = true)

      (Nx, Ny) = (grid.Nx, grid.Ny)
      retVVM = Array{Vector{Real}}(undef, (Nx, Ny))
      for x = 1:Nx, y = 1:Ny
          (ux, uy) = TaylorGreenVortexSpace(grid, x * NVX, y * NVY)
          retVVM[x,y] = [ux, uy]
      end
      if !ccw
          retVVM .= retVVM .* (-1)
      end
      return retVVM ./ VectorNorm2D(maximum(retVVM))
    end

    function SingleVortex(grid; decay = 3.0, ccw = true)
        (Nx, Ny) = (grid.Nx, grid.Ny)
        retVVM = Array{Vector{Real}}(undef, (Nx, Ny))
        rT = (Nx * decay)

        for x = 1:Nx, y = 1:Ny
            r = [x - Nx/2.0, y - Ny/2.0]
            rMag = sqrt(r[1]^2 + r[2]^2)
            ux = sin(y * 2.0 * pi / Ny) * exp(- rMag / rT)
            uy = -sin(x * 2.0 * pi / Nx) * exp(- rMag / rT)
            retVVM[x,y] = [ux, uy]
        end
        if !ccw
            retVVM .= retVVM .* (-1)
        end
        return retVVM ./ VectorNorm2D(maximum(retVVM))
    end

    function ShearFlow(grid)
        (Nx, Ny) = (grid.Nx, grid.Ny)
        retVVM = Array{Vector{Real}}(undef, (Nx, Ny))
        rT = (Ny/2)
        for x = 1:Nx, y = 1:Ny
            ux = rT^2 - (y - rT)^2
            uy = 0
            retVVM[x,y] = [ux, uy]
        end
        return retVVM ./ VectorNorm2D(maximum(retVVM))
    end

    function Poiseuille(grid)
        (Nx, Ny) = (grid.Nx, grid.Ny)
        retVVM = Array{Vector{Real}}(undef, (Nx, Ny))
        for x = 1:Nx, y = 1:Ny
            ux = 1
            uy = 0
            retVVM[x,y] = [ux, uy]
        end
        return retVVM ./ VectorNorm2D(maximum(retVVM))
    end


    function FourRoller(grid)
        (Nx, Ny) = (grid.Nx, grid.Ny)
        retVVM = Array{Vector{Real}}(undef, (Nx, Ny))
      
        for x = 1:Nx, y = 1:Ny
            ux = 2 * sin(x * 2 * pi / grid.Nx) * cos(y * 2 * pi / grid.Ny)
            uy = - 2 * cos(x * 2 * pi / grid.Nx) * sin(y * 2 * pi / grid.Ny)
            retVVM[x,y] = [ux, uy]
        end
        return retVVM 
    end

    function SinY(grid, n)
        (Nx, Ny) = (grid.Nx, grid.Ny)
        retVVM = Array{Vector{Real}}(undef, (Nx, Ny))
      
        for x = 1:Nx, y = 1:Ny
            ux = sin(n * y * 2 * pi / grid.Ny) 
            uy = 0
            retVVM[x,y] = [ux, uy]
        end
        return retVVM ./ VectorNorm2D(maximum(retVVM))
    end

    function ShearFlowLin(grid)
        (Nx, Ny) = (grid.Nx, grid.Ny)
        retVVM = Array{Vector{Real}}(undef, (Nx, Ny))
        rT = (Ny/2)
        for x = 1:Nx, y = 1:Ny
            ux = rT - abs(y - rT)
            uy = 0
            retVVM[x,y] = [ux, uy]
        end
        return retVVM ./ VectorNorm2D(maximum(retVVM))
    end

    function DivOut(grid)
        (Nx, Ny) = (grid.Nx, grid.Ny)
        retVVM = Array{Vector{Real}}(undef, (Nx, Ny))
        rTy = (Ny/2)
        rTx = (Nx/2)
        for x = 1:Nx, y = 1:Ny
            ux = x - rTx
            uy = y - rTy
            norm = sqrt(ux*ux + uy*uy) + 0.001
            retVVM[x,y] = [ux / norm, uy / norm]
        end
        return retVVM ./ VectorNorm2D(maximum(retVVM))
    end

    function Randomized(grid)
        (Nx, Ny) = (grid.Nx, grid.Ny)
        retVVM = Array{Vector{Real}}(undef, (Nx, Ny))
        for x = 1:Nx, y = 1:Ny
            the = rand(Float64)*2*pi
            ux = cos(the)
            uy = sin(the)
            retVVM[x,y] = [ux, uy]
        end
        return retVVM ./ VectorNorm2D(maximum(retVVM))
    end

    function Kraichnan(grid, k0, N = 300)
        (Nx, Ny) = (grid.Nx, grid.Ny)
        dx = grid.dx
        retVVM = Array{Vector{Real}}(undef, (Nx, Ny))
        knList = []
        pVecList = []
        ZnumList = []
        dist = Normal(0.0, k0 / sqrt(3))
        distN = Normal(0.0, 1)
        for n = 1:N
            knxo = rand(dist)
            knyo = rand(dist)
            knx = (1/dx) * sin(knxo * dx)
            kny = (1/dx) * sin(knyo * dx)
            push!(knList, [knxo, knyo])
            ksq = knx^2 + kny^2
            pVec = [1, 0] .- [knx * knx / ksq, kny * knx / ksq]
            push!(pVecList, pVec)
            Zn1 = rand(distN)
            Zn2 = rand(distN)
            push!(ZnumList, [Zn1, Zn2])
        end
        for x = 1:Nx, y = 1:Ny
            ux = 0.0
            uy = 0.0
            for n = 1:N
                fac = ZnumList[n][1] * cos(knList[n][1]*x + knList[n][2]*y) + ZnumList[n][2] * sin(knList[n][1]*x + knList[n][2]*y)
                ux += fac * pVecList[n][1]
                uy += fac * pVecList[n][2]
            end
            retVVM[x,y] = [ux, uy]
        end
        return retVVM ./ VectorNorm2D(maximum(retVVM))
    end

    function RandomSign()
        return sign(rand(1)[1] - 0.5)
    end

    function NewRandom(grid, k0, N = 300)
        (Nx, Ny) = (grid.Nx, grid.Ny)
        dx = grid.dx

        xs = zeros(Nx, Ny)
        ys = zeros(Nx, Ny)
        xF = 2 * pi / (Nx)
        yF = 2 * pi / (Ny)

        fDist = Normal(0.0, 1)
        gDist = Distributions.DiscreteUniform(-k0,k0)
        for n = 1:N
            xfncx = rand(fDist)
            xfnsx = rand(fDist)
            xfncy = rand(fDist)
            xfnsy = rand(fDist)
            xgn1 = rand(gDist)
            xgn2 = rand(gDist)
            xgn3 = rand(gDist)
            xgn4 = rand(gDist)

            yfncx = rand(fDist)
            yfnsx = rand(fDist)
            yfncy = rand(fDist)
            yfnsy = rand(fDist)
            ygn1 = rand(gDist)
            ygn2 = rand(gDist)
            ygn3 = rand(gDist)
            ygn4 = rand(gDist)

            for x = 1:Nx, y = 1:Ny
                xs[x,y] += xfncx * cos(xgn1 * x * xF + ygn1 * y * yF) + xfnsx * sin(xgn2 * x * xF + ygn2 * y * yF)
                ys[x,y] += yfncx * cos(xgn3 * x * xF + ygn3 * y * yF) + yfnsx * sin(xgn4 * x * xF + ygn4 * y * yF)

            end
        end
        retVVM = Array{Vector{Real}}(undef, (Nx, Ny))
        for x = 1:Nx, y = 1:Ny
            retVVM[x,y] = [xs[x,y], ys[x,y]]
        end
        return retVVM ./ VectorNorm2D(maximum(retVVM))
    end

    ########################################################################################
    #
    #                               Gaussian curl functions
    #
    ########################################################################################

    function GaussianCurl(center, point, R, sig, fac)

        xi, yi = center[1], center[2]
        x, y = point[1], point[2]
        rVec = [x - xi, y - yi]
        uVec = [R[1,1] * rVec[1] + R[1,2] * rVec[2], R[2,1] * rVec[1] + R[2,2] * rVec[2]]
        rNorm = VectorNorm2D(rVec) 
        if x == xi && y == yi
            rNorm = 1.0
        end
        fr = exp(- rNorm^2 / (2 * sig^2)) / sqrt(sig^2 * 2 * pi)
        xVal = fac * fr * uVec[1] / rNorm
        yVal = fac * fr * uVec[2] / rNorm
        return [xVal, yVal]
    end

    function UpdateFieldWithGaussians!(xs, ys, xi, yi, Nx, Ny, R, sig, fac)

        for x = 1:Nx, y = 1:Ny # main

            vals = GaussianCurl([xi, yi], [x, y], R, sig, fac)
            xs[x,y] += vals[1]
            ys[x,y] += vals[2]
        end 

        for x = (Nx+1):(2*Nx), y = 1:Ny # x periodicities
            vals = GaussianCurl([xi, yi], [x, y], R, sig, fac)
            xs[x - Nx,y] += vals[1]
            ys[x - Nx,y] += vals[2]

        end 

        for x = (1-Nx):0, y = 1:Ny
            vals = GaussianCurl([xi, yi], [x, y], R, sig, fac)
            xs[x + Nx,y] += vals[1]
            ys[x + Nx,y] += vals[2]

        end 

        for x = 1:Nx, y = (Ny+1):(2*Ny) # y periodicites
            vals = GaussianCurl([xi, yi], [x, y], R, sig, fac)
            xs[x,y - Ny] += vals[1]
            ys[x,y - Ny] += vals[2]

        end

        for x = 1:Nx, y = (1-Ny):0
            vals = GaussianCurl([xi, yi], [x, y], R, sig, fac)
            xs[x,y + Ny] += vals[1]
            ys[x,y + Ny] += vals[2]
        end
    end

    function ArrayIndexToPoint(nx, ny, Nsqrt)

        return Nsqrt*(ny-1) + nx
    end

    function LinearToArrayIndex(n, Nsqrt)

        nx = mod1(n, Nsqrt) 
        ny = floor((n-1) / Nsqrt)
        return (Int(nx), Int(ny+1))
        
    end

    function LinearToArrayPoint(n, Nsqrt, deltax, deltay)

        (nx, ny) = LinearToArrayIndex(n, Nsqrt)
        xi = (nx - 0.5) * deltax
        yi = (ny - 0.5) * deltay
        return (xi, yi)
    end

    function CreatePattern(N, plusInds, flip = false)

        pattern = -1.0 .* ones(N)
        for i in plusInds
            pattern[i] = 1.0
        end 
        if flip
            pattern .= -1.0 .* pattern 
        end 
        return pattern
    end

    function MakePatternArray(pattern)

        N = length(pattern)
        Nsqrt = Int(sqrt(N))
        return reshape(pattern, (Nsqrt, Nsqrt)) 
    end

    function MakeArrayPattern(array)

        N = size(array)[1]^2
        return reshape(array, N)
    end

    function LinearToLeftNeighbor(n, Nsqrt, deltax, deltay)

        (nx, ny) = LinearToArrayIndex(n, Nsqrt)
        xi = (nx - 1.0) * deltax
        yi = (ny - 0.5) * deltay
        return (xi, yi)
    end

    function LinearToBottomNeighbor(n, Nsqrt, deltax, deltay)

        (nx, ny) = LinearToArrayIndex(n, Nsqrt)
        xi = (nx - 0.5) * deltax
        yi = (ny - 1.0) * deltay
        return (xi, yi)
    end

    function CenterPoints(grid, N) 
        (Nx, Ny) = (grid.Nx, grid.Ny)
        Nsqrt = sqrt(N)
        deltax = Nx / Nsqrt
        deltay = Ny / Nsqrt

        pointList = []
        for n = 1:N 
            push!(pointList, LinearToArrayPoint(n, Nsqrt, deltax, deltay))
        end 
        return pointList 
    end

    function LeftNeighborPoints(grid, N)
        (Nx, Ny) = (grid.Nx, grid.Ny)
        Nsqrt = sqrt(N)
        deltax = Nx / Nsqrt
        deltay = Ny / Nsqrt

        pointList = []
        for n = 1:N 
            push!(pointList, Int.(floor.(LinearToLeftNeighbor(n, Nsqrt, deltax, deltay))) .+ 1)
        end 
        return pointList 
    end

    function BottomNeighborPoints(grid, N)
        (Nx, Ny) = (grid.Nx, grid.Ny)
        Nsqrt = sqrt(N)
        deltax = Nx / Nsqrt
        deltay = Ny / Nsqrt

        pointList = []
        for n = 1:N 
            push!(pointList, Int.(floor.(LinearToBottomNeighbor(n, Nsqrt, deltax, deltay))) .+ 1)
        end 
        return pointList 
    end


    function GetPredictions(pattern)

        array = MakePatternArray(pattern)
        boolH = (array .!= circshift(array, (1,0)))
        boolV = (array .!= circshift(array, (0,1)))
        return (MakeArrayPattern(Float64.(boolH)), MakeArrayPattern(Float64.(boolV)))
    end


    function ManyGaussians(grid, sig, o, N, rand = true, alt = false, onInds = true, sign = 1.0)

        (Nx, Ny) = (grid.Nx, grid.Ny)
        dx = grid.dx

        xs = zeros(Nx, Ny)
        ys = zeros(Nx, Ny)

        Nsqrt = sqrt(N)
        deltax = Nx / Nsqrt
        deltay = Ny / Nsqrt

        R = [cos(o) -sin(o); sin(o) cos(o)]

        if !(onInds == true) # will be true if onInds is an array, false if onInds = true
            mask = zeros(N)
            for i in onInds
                mask[i] = 1.0
            end 
        else 
            mask = ones(N)
        end 

        for n = 1:N 

            if rand
                xi = rand(1:Nx)
                yi = rand(1:Ny)
            else 
                (xi, yi) = LinearToArrayPoint(n, Nsqrt, deltax, deltay)
            end

            if alt && (n % Nsqrt != 1.0)
                sign = - sign
            end

            m = mask[n]
            fac = m * sign

            UpdateFieldWithGaussians!(xs, ys, xi, yi, Nx, Ny, R, sig, fac)
        end


        retVVM = Array{Vector{Real}}(undef, (Nx, Ny))
        for x = 1:Nx, y = 1:Ny
            retVVM[x,y] = [xs[x,y], ys[x,y]]
        end
        return retVVM ./ VectorNorm2D(maximum(retVVM))


    end

    function PatternedGaussians(grid, sig, o, N, pattern)

        (Nx, Ny) = (grid.Nx, grid.Ny)
        dx = grid.dx

        xs = zeros(Nx, Ny)
        ys = zeros(Nx, Ny)

        Nsqrt = sqrt(N)
        deltax = Nx / Nsqrt
        deltay = Ny / Nsqrt

        R = [cos(o) -sin(o); sin(o) cos(o)]

        for n = 1:N 

            (xi, yi) = LinearToArrayPoint(n, Nsqrt, deltax, deltay)

            fac = pattern[n]
            UpdateFieldWithGaussians!(xs, ys, xi, yi, Nx, Ny, R, sig, fac)
        end

        retVVM = Array{Vector{Real}}(undef, (Nx, Ny))
        for x = 1:Nx, y = 1:Ny
            retVVM[x,y] = [xs[x,y], ys[x,y]]
        end
        return retVVM #./ VectorNorm2D(maximum(retVVM))



    end




end
