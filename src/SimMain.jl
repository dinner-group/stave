module SimMain

    using Random
    using FileIO

    include("MathFunctions.jl")
    using .MathFunctions

    include("SharedStructs.jl")
    using .SharedStructs

    include("TestCases.jl")
    include("LatBoltz.jl")
    include("BerisEdwards.jl")
    include("Viscoelasticity.jl")
    include("ReactAdvDiff.jl")
    include("Misc.jl")
    include("ReactAdvDiff.jl")
    include("ImmBound.jl")

    function InitializeAndRun(parameters)
                
        ########################################################################################
        #
        #                    Load the parameters into global variables
        #
        ########################################################################################

        println("Initializing...")

        # General params
        global Nx = parameters["Nx"] 
        global Ny = parameters["Ny"]
        global dx = parameters["dx"] # set to 1
        global dt = parameters["dt"] # set to 1
        global tau = parameters["tau"] # collision time
        global nSteps = parameters["nSteps"] # number of steps to run
        global timeStride = parameters["timeStride"] # save every timeStride frames
        global seed = parameters["seed"] # random seed - will use this if this it is not zero
        global tRatio = dt / tau # defined for convenience

        ### Boundary conditions
        # LB conditions
        global bcLB_pX = parameters["bcLB_pX"] # LB condition on +x wall
        global bcLB_mX = parameters["bcLB_mX"] # LB condition on -x wall
        global bcLB_pY = parameters["bcLB_pY"] # LB condition on +y wall
        global bcLB_mY = parameters["bcLB_mY"] # LB condition on -y wall
        global lf = parameters["lf"] # bool to turn on spatial function for applied velocity
        # VE conditions
        global bcVE_X = parameters["bcVE_X"] # VE condition in x direction
        global bcVE_Y = parameters["bcVE_Y"] # VE condition in y direction
        # BE conditions
        global bcBE_X = parameters["bcBE_X"] # BE condition in x direction
        global bcBE_Y = parameters["bcBE_Y"] # BE condition in x direction
        # RAD conditions
        global bcRAD_X = parameters["bcRAD_X"] # RAD condition in x direction
        global bcRAD_Y = parameters["bcRAD_Y"] # RAD condition in y direction

        ### External force
        global fBool = parameters["fBool"] # true or false
        global FM = parameters["FM"] # magnitude of force in LU
        global forceCase = parameters["forceCase"] 
        global forceArg = parameters["forceArg"]
        global forceTime = parameters["forceTime"]
        global aF = parameters["aF"]

        ### Friction force
        global friction = parameters["friction"] # LU

        ### Viscoelasticity params
        global VEBool = parameters["VEBool"]
        global VEModel = parameters["VEModel"]
        global matDeriv = parameters["matDeriv"]
        global KFac = parameters["KFac"]
        # S model
        global Ke1111 = KFac * parameters["Ke1111"] 
        global Ke1122 = KFac * parameters["Ke1122"] 
        global Ke1212 = KFac * parameters["Ke1212"] 
        global Ke2222 = KFac * parameters["Ke2222"] 
        # O model
        global C11 = KFac * parameters["C11"] 
        global C13 = KFac * parameters["C13"] 
        global C31 = KFac * parameters["C31"] 
        global C24 = KFac * parameters["C24"] 
        global C33 = KFac * parameters["C33"] 
        global C44 = KFac * parameters["C44"] 
        # OI model
        global AA = KFac * parameters["AA"] 
        global BB = KFac * parameters["BB"] 
        global mu = KFac * parameters["mu"] 
        global K0 = KFac * parameters["K0"] 
        # M model
        global C = KFac * parameters["C"] 
        global eta = KFac * parameters["eta"] 
        # shared
        global Ds = parameters["Ds"] 

        ### Beris-Edwards params
        global BEBool = parameters["BEBool"] # true or false
        global BEModel = parameters["BEModel"] # P or Q
        # P model 
        global xiBE = parameters["xiBE"]
        global alphaBE = parameters["alphaBE"]
        global betaBE = parameters["betaBE"]
        global kappaBE = parameters["kappaBE"]
        global tauBE = parameters["tauBE"]
        # Q model 
        global lambdaBE = parameters["lambdaBE"] # flow-alignment parameter
        global A0BE = parameters["A0BE"] # from Landau de-Gennes energy
        global UBE = parameters["UBE"] # sets equilibrium nematic ordering
        global LBE = parameters["LBE"] # one-constant stiffness
        # Shared 
        global thetaInit = parameters["thetaInit"] # initial orientation
        global BERand = parameters["BERand"] # true or false to randomize initial orientations
        global GammaBE = parameters["GammaBE"] # rotation diffusion constant
        global PInit = parameters["PInit"] # interpret as q_initial for Q model, P0 for P model

        ### Active params
        global zeta = parameters["zeta"] # magnitude of activity
        global activityTimeOn = parameters["activityTimeOn"] # when should activity turn on
        global activityPattern = parameters["activityPattern"] # true/false + bounds of on region if true

        ### RAD params
        global RADBool = parameters["RADBool"] # turn on RAD dynamics
        global kp = parameters["kp"] # on rate
        global km0 = parameters["km0"] # on rate
        global Du = parameters["Du"] # diffusion constant
        global theta = parameters["theta"] # binding site fraction
        global F0inv = parameters["F0inv"] # inverse charactertic binding force
        global phiuInit = parameters["phiuInit"] # initial phiu
        global phibInit = parameters["phibInit"] # initial phiu
        global phib0C = parameters["phib0C"] # characteristic bound concentration for C_{ijkl}
        global phib0a = parameters["phib0a"] # characteristic bound concentration for active stress

        ### IB params
        global IBBool = parameters["IBBool"]
        global nodeDist = parameters["nodeDist"]
        global center = parameters["center"]
        global radius = parameters["radius"]
        global rB = parameters["rB"]
        global rtA = parameters["rtA"]
        global rtB = parameters["rtB"]
        global aIB = parameters["aIB"]
        global nLegs = parameters["nLegs"]
        global zetaIB = parameters["zetaIB"]
        global springK = zetaIB * parameters["springK"] 
        global eps = zetaIB * parameters["eps"] 
        global kExt = parameters["kExt"] 

        # Restart file
        global varParam = parameters["varParam"] # name of variable whose value will go in the path name 
        global restartLoadOn = parameters["restartLoadOn"] # true of false to load initial fields from pathToLoadRestartFile
        global pathToLoadRestartFile = parameters["pathToLoadRestartFile"] # partial path name
        global restartSaveOn = parameters["restartSaveOn"] # true of false to save final fields to pathToSaveRestartFile
        global pathToSaveRestartFile = parameters["pathToSaveRestartFile"] # partial path name

        ########################################################################################
        #
        #               Initialize the various fields using the global variables
        #
        ########################################################################################

        ### Create the grid and the boundary inds
        global grid = Grid2D(Nx, Ny, dx)
        global indCollection = GetBulkBoundaryInds2D(grid)
        global LBBCIDVec = [LBBCID(bcLB_mX), LBBCID(bcLB_pX), LBBCID(bcLB_mY), LBBCID(bcLB_pY)]

        ### boundary velocity function
        if lf 
            xv = LinRange(0, 1, grid.Ny) 
            yv = LinRange(0, 1, grid.Nx) 
            ff(x) = 16 * (x^2) * (1-x)^2
            fx = ff.(xv)
            fy = ff.(yv)
        else
            fx = ones(grid.Ny)
            fy = ones(grid.Nx)
        end

        ### Lattice Boltzmann init
        global lattice = LatBoltz.LatticeD2Q9(dx, dt)
        global boundaryDict = LatBoltz.FillBoundaryList(grid, lattice, indCollection, LBBCIDVec, [fx, fx, fy, fy])
        global velocityDistributionSoA = LatBoltz.LargeTensorSoA2D(grid, 9, 1);
        global velocitySoA = VectorSoA2D(grid, 0.0);
        global densitySoA = ScalarSoA2D(grid, 1.0);
        global nonViscousForceSoA = VectorSoA2D(grid, 0.0);

        ### Set the random seed if using it
        if seed != 0
            Random.seed!(seed)
        end

        ### External force init
        if fBool
            if forceCase == "tg"
                retVVM = TestCases.TaylorGreenVortexT2DTiled(grid, forceArg...)
            elseif forceCase == "sv"
                retVVM = TestCases.SingleVortex(grid, forceArg...)
            elseif forceCase == "sf"
                retVVM = TestCases.ShearFlowLin(grid)
            elseif forceCase == "fr"
                retVVM = TestCases.FourRoller(grid)
            elseif forceCase == "pois"
                retVVM = TestCases.Poiseuille(grid)
            elseif forceCase == "siny"
                retVVM = TestCases.SinY(grid, forceArg...)
            elseif forceCase == "r"
                retVVM = TestCases.Randomized(grid)
            elseif forceCase == "k"
                retVVM = TestCases.Kraichnan(grid, forceArg...)
            elseif forceCase == "nr"
                retVVM = TestCases.NewRandom(grid, forceArg...)
            elseif forceCase == "do"
                retVVM = TestCases.Out(grid);
            elseif forceCase == "mg"
                retVVM = TestCases.ManyGaussians(grid, forceArg...)
            end
            vM = VectorMesh2D(grid)
            vM.Values .= retVVM
            global TGForcePatternSoA = ConvertVectorMeshToSoA2D(grid, vM)
        else
            global TGForcePatternSoA = VectorSoA2D(grid, 0.0)
        end
        # Set external force timing
        global t0 = 10
        global tg = t0 + forceTime
        global kLeg = ImmBound.kLeg(FM, t0, tg, aF)
        forceTiming(t) = ImmBound.evalkLeg(t, kLeg)
        global timeDomain = collect(range(0, stop = nSteps - 1, length = nSteps))
        global timeSchedule = LatBoltz.ExternalForceTimeSchedule(timeDomain, forceTiming)
        global externalForcePattern = LatBoltz.ExternalForce2D(TGForcePatternSoA, timeSchedule); # combine timing and spatial

        ### Beris-Edwards init
        if BEModel == "P"
            global beParams = BerisEdwards.BEPParams(xiBE, alphaBE, betaBE, kappaBE, GammaBE, tauBE)
            global directorSoA = VectorSoA2D(grid);
            for i=1:Nx, j=1:Ny
                if BERand                   
                    the = rand(Float64)*2*pi
                    directorSoA.XValues[i,j] = PInit * cos(the)
                    directorSoA.YValues[i,j] = PInit * sin(the)
                else
                    directorSoA.XValues[i,j] = PInit * cos(thetaInit)
                    directorSoA.YValues[i,j] = PInit * sin(thetaInit)
                end
            end
            global nematicSoA = TensorSoA2D(grid);
        else # Q model case
            global beParams = BerisEdwards.BEQParams(lambdaBE, A0BE, UBE, LBE, GammaBE)
            global nematicSoA = TensorSoA2D(grid);
            for i=1:Nx, j=1:Ny
                if BERand
                    the = rand(Float64)*2*pi
                    nVec = [cos(the), sin(the)]
                else
                    nVec = [cos(thetaInit), sin(thetaInit)]
                end
                    QTens = BerisEdwards.GetTensorFromDirector(PInit, nVec)
                    nematicSoA.XXValues[i,j] = QTens[1,1] 
                    nematicSoA.XYValues[i,j] = QTens[1,2] 
                    nematicSoA.YXValues[i,j] = QTens[2,1] 
                    nematicSoA.YYValues[i,j] = QTens[2,2] 
            end
            global directorSoA = VectorSoA2D(grid);
        end

        ### Activity init
        global activeParams = BerisEdwards.ActiveParams(zeta, activityTimeOn)
        global activityField = ScalarSoA2D(grid)
        if activityPattern[1]
            activityField.Values[activityPattern[2][1]:activityPattern[2][2], activityPattern[2][3]:activityPattern[2][4]] .= 1
        else 
            activityField.Values .= 1.0
        end

        ### Viscoelasticity init
        global sigmaVESoA = TensorSoA2D(grid)
        if VEModel == "O"
            global veParams = Viscoelasticity.VETOddParams(C11, C13, C31, C24, C33, C44, eta, Ds)
            global CMatSoA = Viscoelasticity.CMatOddFromPVecArraySoA(grid, directorSoA, veParams)
        elseif VEModel == "OI"
            global veParams = Viscoelasticity.VETOddIsoParams(AA, BB, mu, K0, eta, Ds)
            global CMatSoA = Viscoelasticity.CMatOddIsoOnGridSoA(grid, veParams)
        elseif VEModel == "S"
            global veParams = Viscoelasticity.VETParams(Ke1111, Ke1122, Ke1212, Ke2222, eta, Ds)
            global CMatSoA = Viscoelasticity.CMatFromPVecArraySoA(grid, directorSoA, veParams)
        elseif VEModel == "M"
            global veParams = Viscoelasticity.VECParams(eta, C, Ds)
        end

        ### RAD init
        global phibSoA = ScalarSoA2D(grid, phibInit)
        global phiuSoA = ScalarSoA2D(grid, phiuInit)
        global CScaleSoA = ScalarSoA2D(grid, 1.0)
        global aScaleSoA = ScalarSoA2D(grid, 1.0)
        global radParams = ReactAdvDiff.RADParams(kp, km0, Du, theta, F0inv, phib0C, phib0a)

        ### Create the immersed boundary object
        global iBoundary2D = ImmBound.ICircularBoundary2D(nodeDist, center, radius, true)
        global nodePosition = ImmBound.NodePositions(iBoundary2D.NodeList)
        global forceList = similar(nodePosition)
        ImmBound.ZeroList!(forceList)
        global nodeVelocityList = similar(nodePosition)
        ImmBound.ZeroList!(nodeVelocityList)
        global boundaryForceSoA = VectorSoA2D(grid, 0.01)
        # Create the IB pulling protocol
        global timeDomain = collect(range(0, stop = nSteps - 1, length = nSteps))
        global pullingProtocol = ImmBound.PullingProtocol()
        global r0 = center
        global rA = center
        global r1 = rB
        global rt0 = -nSteps
        global rt1 = 2*nSteps
        global delt = rtB - rtA
        ImmBound.AddrExtLeg(pullingProtocol, ImmBound.rExtLeg(r0, rA, rt0, rtA, aIB))
        ImmBound.AddrExtLeg(pullingProtocol, ImmBound.rExtLeg(rA, rB, rtA, rtB, aIB)) # this is the first leg

        if false # nLegs == 1 # add final leg
            ImmBound.AddrExtLeg(pullingProtocol, ImmBound.rExtLeg(rB, r1, rtB, rt1, aIB))
            global kt0 = rt0
            global ktA = rtB + (nLegs - 1) * delt

        elseif false # cycle through until nLegs = 1, Hardcoded switch between cases
            thisPoint = rB
            thatPoint = rA
            thisTime = rtB
            thatTime = rtB + delt
            x = deepcopy(nLegs)
            while x > 1
                ImmBound.AddrExtLeg(pullingProtocol, ImmBound.rExtLeg(thisPoint, thatPoint, thisTime, thatTime, aIB))
                thisTime += delt
                thatTime += delt
                thisPoint, thatPoint = thatPoint, thisPoint
                x -= 1
            end
            ImmBound.AddrExtLeg(pullingProtocol, ImmBound.rExtLeg(thisPoint, thisPoint, thisTime, rt1, aIB)) # add final leg
            global kt0 = rt0
            global ktA = rtB + (nLegs[2] - 1) * delt

        else # triangle case, make a reflection over vertical line through rB

            fac = nLegs
            v = rB .- rA
            v[1] = - v[1]
            rC = rB .+ (fac .* v)
            rtC = rtB + fac*delt
            ImmBound.AddrExtLeg(pullingProtocol, ImmBound.rExtLeg(rB, rC, rtB, rtC, aIB))
            ImmBound.AddrExtLeg(pullingProtocol, ImmBound.rExtLeg(rC, rC, rtC, rt1, aIB))
            global kt0 = rt0
            global ktA = rtC
        end 

        ImmBound.AddkLeg(pullingProtocol, ImmBound.kLeg(kExt, kt0, ktA, aIB))
        global rExt = ImmBound.evalrExt(1, pullingProtocol)
        global k = ImmBound.evalk(1, pullingProtocol)


        ### Re-initialize fields if restartLoadOn is true
        pathToLoadRestartFile = pathToLoadRestartFile * parameters["varParam"] * "_" * string(parameters[varParam]) * ".jld2"
        pathToSaveRestartFile = pathToSaveRestartFile * parameters["varParam"] * "_" * string(parameters[varParam]) * ".jld2"
        if restartLoadOn 
            # This will only re-initialize these fields - all other parameters etc are assumed to be consistent.
            # Note that some fields (nonViscousForceSoA, CMatSoA) will just be re-computed during the first loop iteration.
            d = load(pathToLoadRestartFile)
            # LB fields
            velocityDistributionSoA = d["velocityDistributionSoA"]
            velocitySoA = d["velocitySoA"]
            densitySoA = d["densitySoA"]
            # BE fields
            directorSoA = d["directorSoA"]
            nematicSoA = d["nematicSoA"]
            # VE fields
            sigmaVESoA = d["sigmaVESoA"]
            # RAD fields
            phibSoA = d["phibSoA"]
            phiuSoA = d["phiuSoA"]
            # IB field
            iBoundary2D = d["iBoundary2D"]
        end

        println("Done initializing.")

        ########################################################################################
        #
        #                            Run the main simulation loop
        #
        ########################################################################################

        # Create arrays to store the data
        global densityArray = [] 
        global velocityArray = [] 
        global sigmaVEArray = [] 
        global directorArray = [] 
        global nematicArray = [] 
        global phibArray = [] 
        global phiuArray = [] 
        global nodePositionArray = [] 
        global kArray = [] 
        global rExtArray = []

        println("Beginning simulation...")

        # Equilibrate before simulation
        LatBoltz.InitEquilibriumConverge2DSoA!(grid, lattice, velocitySoA, velocityDistributionSoA,
            densitySoA, nonViscousForceSoA, tRatio, dt, indCollection, boundaryDict, LBBCIDVec)
            
        # Beginning of loop
        @time for t in 1:nSteps

            # Copy down data to use for the updates of various fields, so that the order of updates doesn't matter
            velocitySoAN = deepcopy(velocitySoA)
            densitySoAN = deepcopy(densitySoA)
            directorSoAN = deepcopy(directorSoA)
            nematicSoAN = deepcopy(nematicSoA)
            sigmaVESoAN = deepcopy(sigmaVESoA)
            phibSoAN = deepcopy(phibSoA)
            phiuSoAN = deepcopy(phiuSoA)
            nodeList = deepcopy(iBoundary2D.NodeList)

            # Push to the saved data if at multiple of timeStride
            if (t == 1) || (t % timeStride == 0)
                push!(velocityArray, velocitySoAN)
                push!(densityArray, densitySoAN)
                VEBool && push!(sigmaVEArray, sigmaVESoAN)
                if BEBool
                    if BEModel == "P"
                        push!(directorArray, directorSoAN)
                    else # Q model case
                        push!(nematicArray, nematicSoAN)
                    end
                end
                if RADBool
                    push!(phibArray, phibSoAN)
                    push!(phiuArray, phiuSoAN)
                end
                if IBBool
                    push!(nodePositionArray, ImmBound.NodePositions(nodeList))
                    push!(rExtArray, rExt)
                    push!(kArray, k)
                end
            end

            # Get the velocity gradient tensors
            (OmegaSoA, PsiSoA) = OmegaPsi2DSoA(grid, velocitySoA, bcLB_mX[1:3], bcLB_mY[1:3])

            # Get the RAD scaling factors
            if RADBool
                CScaleSoA = ReactAdvDiff.GetScalingFactor(grid, phibSoA, 1 / radParams.phib0C)
                aScaleSoA = ReactAdvDiff.GetScalingFactor(grid, phibSoA, 1 / radParams.phib0a)
            end

            # Get polymer field forces
            if BEBool
                # turn on activity if it's time
                if t > activeParams.activityTimeOn
                    aFac = 1.0
                else 
                    aFac = 0.0
                end
                if BEModel == "P"
                    eSoA = BerisEdwards.EricksenStressTensorP2DSoA(grid, directorSoA, beParams, bcBE_X, bcBE_Y)
                    aSoA = BerisEdwards.ActiveStressTensorP2DSoA(directorSoA, grid, activeParams, activityField,  aFac)              
                else # Q model case
                    eSoA = BerisEdwards.EricksenStressTensorQ2DSoA(grid, nematicSoA, beParams, bcBE_X, bcBE_Y)
                    aSoA = BerisEdwards.ActiveStressTensorQ2DSoA(nematicSoA, grid, activeParams, activityField, aFac)
                end
                MultipyTensorByScalarSoA2D!(aSoA, aScaleSoA)
                polymerForceSoA = DivTensorOnSoA2D(grid, MatrixTransposeOnSoA2D(grid, eSoA), BCDerivDict[bcBE_X], BCDerivDict[bcBE_Y])
                activeForceSoA = DivTensorOnSoA2D(grid, MatrixTransposeOnSoA2D(grid, aSoA), BCDerivDict[bcBE_X], BCDerivDict[bcBE_Y])
                AddVectorSoA2D!(polymerForceSoA, activeForceSoA)
            else # BEBool
                polymerForceSoA = VectorSoA2D(grid)
            end

            # Get viscoelastic forces
            if VEBool
                if VEModel == "OI"                
                    CMatSoA = Viscoelasticity.CMatOddIsoOnGridSoA(grid, veParams)
                elseif VEModel == "O"
                    if BEModel == "Q"
                        directorSoA = BerisEdwards.GetDirectorFromTensor2DSoA(grid, nematicSoA)
                    end
                    CMatSoA = Viscoelasticity.CMatOddFromPVecArraySoA(grid, directorSoA, veParams)
                elseif VEModel == "S" 
                    if BEModel == "Q"
                        directorSoA = BerisEdwards.GetDirectorFromTensor2DSoA(grid, nematicSoA)
                    end
                    CMatSoA = Viscoelasticity.CMatFromPVecArraySoA(grid, directorSoA, veParams)
                end
                if VEModel != "M"
                    MultipyLargeTensorByScalarSoA2D!(CMatSoA, CScaleSoA)
                end
                viscoelasticForceSoA = DivTensorOnSoA2D(grid, MatrixTransposeOnSoA2D(grid, sigmaVESoA), BCDerivDict[bcVE_X], BCDerivDict[bcVE_Y])

            else # VEBool
                viscoelasticForceSoA = VectorSoA2D(grid)
            end

            # Get immersed boundary forces
            if IBBool
                # Update pulling protocol
                rExt = ImmBound.evalrExt(t, pullingProtocol)
                k = ImmBound.evalk(t, pullingProtocol)
                neighborList = ImmBound.GetGridNeighborsOfNodeList2D(nodeList, grid)
                forceListSpring = ImmBound.SpringForces2D(nodeList, springK, iBoundary2D.d)
                forceListAngle = ImmBound.AngleForces2D(nodeList, eps, iBoundary2D.delTheta)
                forceListExt = ImmBound.ExternalForces2D(nodeList, rExt, k)
                forceList = forceListSpring .+ forceListAngle .+ forceListExt
                boundaryForceSoA = ImmBound.BoundaryForceOnFluid2DSoA(grid, nodeList, forceList, neighborList);
            else
                boundaryForceSoA = VectorSoA2D(grid)
            end

            # Get the friction force
            frictionForceSoA = MultipyVectorSoA2D(grid, PBCSmoothing(grid, velocitySoAN), - friction)

            # Group all the non solvent viscosity forces together (solvent viscosity is handled implicitly by the LB collision step)
            global externalForceSoA = MultipyVectorSoA2D(grid, externalForcePattern.SpatialPattern, externalForcePattern.TimeSchedule.Values[t])
            nonViscousForceSoA = VectorSoA2D(grid)
            AddVectorSoA2D!(nonViscousForceSoA, polymerForceSoA)
            AddVectorSoA2D!(nonViscousForceSoA, viscoelasticForceSoA)
            AddVectorSoA2D!(nonViscousForceSoA, externalForceSoA)
            AddVectorSoA2D!(nonViscousForceSoA, boundaryForceSoA)
            AddVectorSoA2D!(nonViscousForceSoA, frictionForceSoA)

            # Update polymer field
            if BEBool
                if BEModel == "P"
                    BerisEdwards.PredictorCorrectorStepP2DSoA!(grid, OmegaSoA, PsiSoA, velocitySoA, directorSoAN, dt, beParams, bcBE_X, bcBE_Y)
                else # model Q case
                    BerisEdwards.PredictorCorrectorStepQ2DSoA!(grid, OmegaSoA, PsiSoA, velocitySoA, nematicSoAN, dt, beParams, bcBE_X, bcBE_Y)
                end
            end

            # Update viscoelasticity tensors
            if VEBool
                if VEModel == "M"
                    Viscoelasticity.PredictorCorrectorStepMax2DSoA!(grid, OmegaSoA, PsiSoA, velocitySoA, sigmaVESoAN, dt, veParams, bcVE_X, bcVE_Y, matDeriv)
                elseif VEModel == "S"
                    Viscoelasticity.PredictorCorrectorStepTensS2DSoA!(grid, OmegaSoA, PsiSoA, velocitySoA, sigmaVESoAN, CMatSoA, dt, veParams, bcVE_X, bcVE_Y, matDeriv)
                else  # case O or OI
                    Viscoelasticity.PredictorCorrectorStepTensOdd2DSoA!(grid, OmegaSoA, PsiSoA, velocitySoA, sigmaVESoAN, CMatSoA, dt, veParams, bcVE_X, bcVE_Y, matDeriv)
                end
            end

            # Update concentrations
            if RADBool
                ReactAdvDiff.PredictorCorrectorStepRAD2DSoA!(grid, densitySoA, velocitySoA, viscoelasticForceSoA, phibSoAN, phiuSoAN, dt, radParams, bcRAD_X, bcRAD_Y)
            end

            # Find node velocities
            if IBBool
                ImmBound.NodeVelocities2DSoA!(nodeList, nodeVelocityList, velocitySoA, neighborList, grid)
            end

            # Updates densitySoAN, velocitySoAN, and velocityDistributionSoA using nonViscousForceSoA
            LatBoltz.LatticeBoltzmannStep2DSoA!(grid, lattice, densitySoAN, velocityDistributionSoA, velocitySoAN,
                nonViscousForceSoA, indCollection, dt, tRatio, timeStride, t, boundaryDict, LBBCIDVec)

            # Store the new values
            SetVectorFromSoA2D!(velocitySoA, velocitySoAN) # velocitySoA <- velocitySoAN
            SetScalarFromSoA2D!(densitySoA, densitySoAN)
            if BEBool
                if BEModel == "P"
                    SetVectorFromSoA2D!(directorSoA, directorSoAN)
                else # model Q case
                    SetTensorFromSoA2D!(nematicSoA, nematicSoAN)
                end
            end
            VEBool && SetTensorFromSoA2D!(sigmaVESoA, sigmaVESoAN)
            if RADBool
                SetScalarFromSoA2D!(phibSoA, phibSoAN)
                SetScalarFromSoA2D!(phiuSoA, phiuSoAN)
            end
            IBBool && ImmBound.UpdateNodePositions!(iBoundary2D, nodeVelocityList, dt)
            

        end
        println("Done with simulation.")

        ### Save final fields if restartSaveOn is true
        if restartSaveOn
            save(pathToSaveRestartFile,
            "velocityDistributionSoA", velocityDistributionSoA,
            "velocitySoA", velocitySoA,
            "densitySoA", densitySoA,
            "directorSoA", directorSoA,
            "nematicSoA", nematicSoA,
            "phibSoA", phibSoA,
            "phiuSoA", phiuSoA,
            "sigmaVESoA", sigmaVESoA,
            "iBoundary2D", iBoundary2D
            )
        end

        ### Return the saved trajectory data, to be selectivey exported from the calling scope
        return (
        densityArray,
        velocityArray,
        sigmaVEArray,
        directorArray,
        nematicArray,
        phibArray,
        phiuArray,
        nodePositionArray,
        kArray,
        rExtArray)

    end # InitializeAndRun


end # Main
