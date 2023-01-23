srcPath = "/project/svaikunt/csfloyd/CatchHopfield/JuliaCode/"
include(srcPath * "SimMain.jl")
include(srcPath * "Misc.jl")
using FileIO

args = Base.ARGS;

# set input/output
inputFile = string(args[1])
outputDir = string(args[2])

# set default parameters
defaultParams = Dict(
# grid params
"Nx" => 50,
"Ny" => 100,
"dx" => 1.0,
"dt" => 1.0,
"tau" => 1,
"nSteps" => 10000,#8000,
"timeStride" => 100,
"startCollecting" => 20000,
"seed" => 0,
    
# boundary conditions
"bcLB_mX" => "pbc",
"bcLB_pX" => "pbc",
"bcLB_mY" => "bbc_0",
"bcLB_pY" => "bbc_5e-4",
"lf" => false,
    
"bcVE_X" => "pbc",
"bcVE_Y" => "neu",

"bcBE_X" => "pbc",
"bcBE_Y" => "pbc",

"bcRAD_X" => "pbc",
"bcRAD_Y" => "pbc",
    
# restart file
"varParam" => "Ds",
"restartLoadOn" => false,
"pathToLoadRestartFile" => "/project/svaikunt/csfloyd/CatchHopfield/SaveStates/DirsFA/Dirs_Ds/",
"restartSaveOn" => false,
"pathToSaveRestartFile" => "/project/svaikunt/csfloyd/CatchHopfield/SaveStates/DirsFA/Dirs_Ds/",
    
# external force   
"fBool" => false,
"FM" => 1e-6,
"forceCase" => "pois",
"forceArg" => (20),
"forceTime" => 150000,
"aF" => 30,
    
# friction force
"friction" => 0.00, # LU
    
# viscoelasticity
"VEBool" => false,
"VEModel" => "S",
"matDeriv" => "cor",
"KFac" => 1.0,
# S model
"Ke1111" => 2e-5, 
"Ke1122" => 1e-5, 
"Ke1212" => 1e-5, 
"Ke2222" => 1e-5, 
# OI model
"C11" => 0.1, 
"C13" => 0.00, 
"C31" => 0.00, 
"C24" => 0.001, 
"C33" => 0.005, 
"C44" => 0.005, 
# O model
"AA" => 10.0, 
"BB" => 2.0, 
"mu" => 2.0,
"K0" => 5.0, 
# M model
"C" => 2e-5, 
"eta" => 25, 
"Ds" => 5e-4, 
    
# Beris-Edwards
"BEBool" => false, 
"BEModel" => "P",
# P model params, all LU
"xiBE" => 1.1,
"alphaBE" => -0.9, 
"betaBE" => 1.0, 
"kappaBE" => 1e-3,
"tauBE" => 0,
# Q model params, all LU
"lambdaBE" => 0.7,
"A0BE" => 0.1,
"UBE" => 3.5,
"LBE" => 0.1,
# shared params
"thetaInit" => pi/4,
"GammaBE" => 0.13,
"BERand" => false,
"PInit" => 0.62,
    
# active params
"zeta" => 0.00,
"activityTimeOn" => 0,

# Reaction dynamics
"RADBool" => false, 
"kp" => 0.005,
"km0" => 0.008,
"Du" => 1e-4,
"theta" => 1.0,
"F0inv" => 2.5e4,
"phiuInit" => 1.0,
"phibInit" => 0.0,
"phib0C" => 0.1,
"phib0a" => 1.0,

# immersed boundary
"IBBool" => true, 
"nodeDist" => 0.5, # LU
"center" => [12.5, 25], # LU
"radius" => 5.0, # LU
"rB" => [45, 50],
"rtA" => 1000000,
"rtB" => 1000000,
"aIB" => 10,
"nLegs" => 1,
"zetaIB" => 1,
"springK" => 1.0e-7, # LU
"eps" => 1e-22, # LU
"kExt" => 1e-6 # LU
    

);
# set parameters
inputParams = Misc.ParseInput(inputFile);
parameters = deepcopy(defaultParams)
for p in keys(inputParams)
    parameters[p] = inputParams[p]
end


(densityArray,
velocityArray,
sigmaVEArray,
directorArray,
nematicArray,
phibArray,
phiuArray,
nodePositionArray,
kArray,
rExtArray) = SimMain.InitializeAndRun(parameters);

# save results
objName = "SavedData"
pathName = outputDir * objName * ".jld2"

@time save(pathName,
"parameters", parameters,
"densityArray", densityArray,
"velocityArray", velocityArray,
"sigmaVEArray", sigmaVEArray,
"directorArray", directorArray,
"nematicArray", nematicArray,
"phibArray", phibArray,
"phiuArray", phiuArray,
"kArray", kArray,
"rExtArray", rExtArray,
"nodePositionArray", nodePositionArray
)

println("Done saving results.")
