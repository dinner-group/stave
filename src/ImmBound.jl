module ImmBound

    include("SharedStructs.jl")
    using .SharedStructs

    include("MathFunctions.jl")
    using .MathFunctions

    include("Misc.jl")
    using .Misc

    struct IBNode2D
        Label::String
        Position::Vector{Real}
        function IBNode2D(label, position)
            return new(label, position)
        end
    end

    struct ICircularBoundary2D
        NodeList::Vector{IBNode2D} # ordered to indicate neighbors
        ReferencePositions::Vector{Vector{Real}}
        d::Real
        delTheta::Real
        function ICircularBoundary2D(nodeArg, center, radius, distBool) # initializes a circular object
            if distBool
                circum = 2*pi*radius
                numNodes = floor(circum / nodeArg)
            else
                numNodes = nodeArg
            end
            nodeList = []
            referencePositions = []
            delTheta = 2 * pi / numNodes
            d = radius * sqrt(2 - 2 * cos(delTheta))
            for i in 1:numNodes
                position = center .+ radius * [cos((i-1) * delTheta), sin((i-1) * delTheta)]
                label = "IB" * string(i)
                push!(nodeList, IBNode2D(label, position))
                push!(referencePositions, position)
            end
            return new(nodeList, referencePositions, d, delTheta)
        end
    end

    struct PullingProtocol
        kLegList::Vector{Any}
        rExtLegList::Vector{Any}
        function PullingProtocol()
            return new([], [])
        end
    end

    struct kLeg
        k::Real
        tA::Real
        tB::Real
        a::Real
        function kLeg(k, tA, tB, a)
            return new(k, tA, tB, a)
        end
    end

    function AddkLeg(pullingProtocol, kLeg)
        push!(pullingProtocol.kLegList, kLeg)
    end

    function evalkLeg(t, kLeg)
        return kLeg.k * Misc.SmoothBump(t, kLeg.a, kLeg.tA, kLeg.tB)
    end

    function evalk(t, pullingProtocol)
        ret = evalkLeg(t, pullingProtocol.kLegList[1])
        if length(pullingProtocol.kLegList) > 1
            for kLeg in pullingProtocol.kLegList[2:end]
                ret = ret + evalkLeg(t, kLeg)
            end
        end
        return ret
    end

    struct rExtLeg
        rA::Vector{Real}
        rB::Vector{Real}
        tA::Real
        tB::Real
        a::Real
        function rExtLeg(rA, rB, tA, tB, a)
            return new(rA, rB, tA, tB, a)
        end
    end

    function AddrExtLeg(pullingProtocol, rExtLeg)
        push!(pullingProtocol.rExtLegList, rExtLeg)
    end

    function evalrExtLeg(t, rExtLeg)
        return (rExtLeg.rA .+ (rExtLeg.rB .- rExtLeg.rA) .* ((t - rExtLeg.tA) / (rExtLeg.tB - rExtLeg.tA))) .* Misc.SmoothBump(t, rExtLeg.a, rExtLeg.tA, rExtLeg.tB)
    end

    function evalrExt(t, pullingProtocol)
        ret = evalrExtLeg(t, pullingProtocol.rExtLegList[1])
        if length(pullingProtocol.rExtLegList) > 1
            for rExtLeg in pullingProtocol.rExtLegList[2:end]
                ret .= ret .+ evalrExtLeg(t, rExtLeg)
            end
        end
        return ret
    end

    function NodePositions(nodeList)
        return [node.Position for node in nodeList]
    end

    function RigidForces(nodeList, referencePositions, kappa, dx, d)
        posList = NodePositions(nodeList)
        fac = kappa * d / dx
        forceList = - fac * (posList .- referencePositions)
        return forceList
    end

    function CosStencil(x, dx)
        if abs(x) <= 2*dx
            return 0.25 * (1 + cos(pi * x / 2))
        else
            return 0.0
        end
    end

    function GetGridNeighborsOfPos(rx, ry, grid) # does not account for periodic boundaries!!
        ir = Int(floor(rx / grid.dx))
        jr = Int(floor(ry / grid.dx))
        ip = min(grid.Nx, ir + 2)
        im = max(1, ir - 2)
        jp = min(grid.Ny, jr + 2)
        jm = max(1, jr - 2)
        return [im, ip, jm, jp]
    end

    function GetGridNeighborsOfNodePosList2D(nodePositions, grid) # does not account for periodic boundaries!!
        return map(x -> GetGridNeighborsOfPos(x[1], x[2], grid), nodePositions)
    end

    function GetGridNeighborsOfNode2D(node, grid) # does not account for periodic boundaries!!
        return GetGridNeighborsOfPos(node.Position[1], node.Position[2], grid)
    end

    function GetGridNeighborsOfNodeList2D(nodeList, grid) # does not account for periodic boundaries!!
        return map(x -> GetGridNeighborsOfNode2D(x, grid), nodeList)
    end

    function InterpolateVectorAtIB(grid, nodePositions, neighborList, vSoA)

        vectorList = similar(nodePositions)
        ImmBound.ZeroList!(vectorList)
        for (k, pos) in enumerate(nodePositions)
            indRange = neighborList[k]
            for j in indRange[3]:indRange[4], i in indRange[1]:indRange[2]
                delX = pos[1] - i*grid.dx
                delY = pos[2] - j*grid.dx
                cs = CosStencil(delX, grid.dx) * CosStencil(delY, grid.dx)
                vectorList[k][1] += vSoA.XValues[i,j] * cs
                vectorList[k][2] += vSoA.YValues[i,j] * cs
            end
        end

        return vectorList 
    end



    function BoundaryForceOnFluid2DSoA(grid, nodeList, forceList, neighborList)
        boundaryForceSoA = VectorSoA2D(grid)
        for (k, node) in enumerate(nodeList)
            indRange = neighborList[k]
            for j in indRange[3]:indRange[4], i in indRange[1]:indRange[2]
                delX = node.Position[1] - i*grid.dx
                delY = node.Position[2] - j*grid.dx
                cs = CosStencil(delX, grid.dx) * CosStencil(delY, grid.dx)
                boundaryForceSoA.XValues[i,j] += forceList[k][1] * cs
                boundaryForceSoA.YValues[i,j] += forceList[k][2] * cs
            end
        end
        boundaryForceSoA.XValues .= boundaryForceSoA.XValues .* (1 / grid.dx)^2
        boundaryForceSoA.YValues .= boundaryForceSoA.YValues .* (1 / grid.dx)^2
        return boundaryForceSoA
    end

    function ZeroList!(list)
        for i = 1:length(list)
            list[i] = [0.0, 0.0]
        end
    end

    function NodeVelocities2DSoA!(nodeList, nodeVelocityList, velocitySoA, neighborList, grid)
        ZeroList!(nodeVelocityList)
        for (k, node) in enumerate(nodeList)
            indRange = neighborList[k]
            for j in indRange[3]:indRange[4], i in indRange[1]:indRange[2]
                delX = node.Position[1] - i*grid.dx
                delY = node.Position[2] - j*grid.dx
                cs = CosStencil(delX, grid.dx) * CosStencil(delY, grid.dx)
                nodeVelocityList[k][1] += velocitySoA.XValues[i,j] * cs
                nodeVelocityList[k][2] += velocitySoA.YValues[i,j] * cs
            end
        end
        nodeVelocityList .= nodeVelocityList .* (grid.dx)^2
    end

    function UpdateNodePositions!(iBoundary, nodeVelocityList, dt)
        for (k, node) in enumerate(iBoundary.NodeList)
            node.Position .= node.Position .+ nodeVelocityList[k] * dt
        end
    end

    function SpringForces2D(nodeList, k, l0)
        posList = NodePositions(nodeList)
        forceList = similar(posList)
        ZeroList!(forceList)
        nS = length(posList)
        for i = 1:nS - 1
            ra = posList[i]
            rb = posList[i+1]
            (dra, drb) = DerivativeSpringLength2D(ra, rb, k, l0)
            forceList[i]  .=  forceList[i] .+ dra
            forceList[i+1] .=  forceList[i+1] .+ drb
        end
        ra = posList[nS]
        rb = posList[1]
        (dra, drb) = DerivativeSpringLength2D(ra, rb, k, l0)
        forceList[nS]  .=  forceList[nS] .+ dra
        forceList[1] .=  forceList[1] .+ drb
        return - forceList

    end

    function FrictionForces2D(nodeList, nodeVelocityList, gamma)
        posList = NodePositions(nodeList)
        forceList = similar(posList)
        ZeroList!(forceList)
        nS = length(posList)
        for i = 1:nS - 1
            forceList[i]  .=  - gamma .* nodeVelocityList[i]
        end
        return forceList

    end

    function DerivativeSpringLength2D(ra, rb, k, l0)
        rab = rb .- ra
        dist = sqrt(DotProduct2D(rab, rab))
        fac = k * (dist - l0)
        drabdra = - rab ./ dist
        drabdrb = rab ./dist
        return (fac * drabdra, fac * drabdrb)
    end

    function AngleForces2D(nodeList, eps, theta0)
        posList = NodePositions(nodeList)
        forceList = similar(posList)
        ZeroList!(forceList)
        nS = length(posList)
        for i = 1:nS - 2
            ra = posList[i]
            rb = posList[i+1]
            rc = posList[i+2]
            (dra, drb, drc) = DerivativeAngle2D(ra, rb, rc, eps, theta0)
            forceList[i]  .=  forceList[i] .+ dra
            forceList[i+1] .=  forceList[i+1] .+ drb
            forceList[i+2] .=  forceList[i+2] .+ drc
        end
        ra = posList[nS-2]
        rb = posList[nS-1]
        rc = posList[1]
        (dra, drb, drc) = DerivativeAngle2D(ra, rb, rc, eps, theta0)
        forceList[nS-2]  .=  forceList[nS-2] .+ dra
        forceList[nS-1] .=  forceList[nS-1] .+ drb
        forceList[1] .=  forceList[1] .+ drc
        ra = posList[nS-1]
        rb = posList[1]
        rc = posList[2]
        (dra, drb, drc) = DerivativeAngle2D(ra, rb, rc, eps, theta0)
        forceList[nS-1]  .=  forceList[nS-1] .+ dra
        forceList[1] .=  forceList[1] .+ drb
        forceList[2] .=  forceList[2] .+ drc
        return - forceList

    end

    function DerivativeAngle2D(ra, rb, rc, eps, theta0)
        rab = rb .- ra
        rbc = rc .- rb
        distAB = sqrt(DotProduct2D(rab, rab))
        distBC = sqrt(DotProduct2D(rbc, rbc))
        si = DotProduct2D(rab, rbc) / (distAB * distBC)
        theta = SafeACos(si)
        aCosInvDervTerm = - 1 / sqrt(1 - si^2)
        sinTerm = eps * sin(theta - theta0)
        comDsiTerm = 1 / (distAB * distBC)
        fac = aCosInvDervTerm * sinTerm * comDsiTerm

        rax = ra[1]
        ray = ra[2]
        rbx = rb[1]
        rby = rb[2]
        rcx = rc[1]
        rcy = rc[2]

        rat1 = ((ray - rby)*(ray*(rbx - rcx) + rby*rcx - rbx*rcy + rax*(-rby + rcy)))
        rat2 = ((rax - rbx)*(-(rby*rcx) + ray*(-rbx + rcx) + rax*(rby - rcy) + rbx*rcy))
        racomt = (rax^2 + ray^2 - 2*rax*rbx + rbx^2 - 2*ray*rby + rby^2)
        dsidra = [rat1, rat2] ./ racomt

        rbt1 = rax - 2*rbx + rcx - ((-rax + rbx)*((rax - rbx)*(rbx - rcx) + (ray - rby)*(rby - rcy)))/((rax - rbx)^2 + (ray - rby)^2) + ((-rbx + rcx)*((rax - rbx)*(rbx - rcx) + (ray - rby)*(rby - rcy)))/((rbx - rcx)^2 + (rby - rcy)^2)
        rbt2 = ray - 2*rby - ((-ray + rby)*((rax - rbx)*(rbx - rcx) + (ray - rby)*(rby - rcy)))/((rax - rbx)^2 + (ray - rby)^2) + rcy + (((rax - rbx)*(rbx - rcx) + (ray - rby)*(rby - rcy))*(-rby + rcy))/((rbx - rcx)^2 + (rby - rcy)^2)
        dsidrb = [rbt1, rbt2]

        rct1 = ((rby - rcy)*(ray*(rbx - rcx) + rby*rcx - rbx*rcy + rax*(-rby + rcy)))
        rct2 = ((rbx - rcx)*(-(rby*rcx) + ray*(-rbx + rcx) + rax*(rby - rcy) + rbx*rcy))
        rccomt = (rbx^2 + rby^2 - 2*rbx*rcx + rcx^2 - 2*rby*rcy + rcy^2)
        dsidrc = [rct1, rct2] ./ rccomt

        return (fac * dsidra, fac * dsidrb, fac * dsidrc)

    end

    function CenterOfMass(nodeList)
        posList = NodePositions(nodeList)
        nS = length(posList)
        return sum(posList) / nS
    end

    function CenterOfMassPos(nodePosList)
        nS = length(nodePosList)
        return sum(nodePosList) / nS
    end

    function ExternalForces2D(nodeList, rCOMExt, kExt)
        posList = NodePositions(nodeList)
        forceList = similar(posList)
        nS = length(posList)
        com = CenterOfMass(nodeList)
        for i = 1:nS
            forceList[i] = kExt * (com - rCOMExt) / nS
        end
        return - forceList
    end





end #module
