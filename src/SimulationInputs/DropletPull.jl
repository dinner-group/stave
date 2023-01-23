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
"Nx" => 250,
"Ny" => 250,
"dx" => 1.0,
"dt" => 1.0,
"tau" => 1.25,
"nSteps" => 187500,
"timeStride" => 625,
"startCollecting" => 0,
"seed" => 0,
"trial" => 1,

# boundary conditions
"bcLB_mX" => "bbc_0",
"bcLB_pX" => "bbc_0",
"bcLB_mY" => "bbc_0",
"bcLB_pY" => "bbc_0",
"lf" => false,
    
"bcVE_X" => "neu",
"bcVE_Y" => "neu",

"bcBE_X" => "neu",
"bcBE_Y" => "neu",

"bcRAD_X" => "neu",
"bcRAD_Y" => "neu",

# restart file
"varParam" => "zeta",
"restartLoadOn" => false,
"pathToLoadRestartFile" => "/Users/csfloyd/Dropbox/Projects/CatchHopfield/SaveStates/BigL0p1.jld2",
"restartSaveOn" => false,
"pathToSaveRestartFile" => "/Users/csfloyd/Dropbox/Projects/CatchHopfield/SaveStates/BigL0p1.jld2",

 # external force
"fBool" => false,
"FM" => 1e-7,
"forceCase" => "nr",
"forceArg" => (20),
"forceTime" => 500,
"aF" => 30,

# friction force
"friction" => 0.00, # LU

# viscoelasticity
"VEBool" => true,
"VEModel" => "O",
"matDeriv" => "cor",
"KFac" => 1.0,
# S model
"Ke1111" => 2e-5, 
"Ke1122" => 1e-5, 
"Ke1212" => 1e-5, 
"Ke2222" => 1e-5, 
# OI model
"C11" => 2e-5, 
"C13" => 0.00, 
"C31" => 0.00, 
"C24" => 1e-5, 
"C33" => 1e-5, 
"C44" => 1e-5, 
# O model
"AA" => 1e-4, 
"BB" => 2e-4, 
"mu" => 2e-4,
"K0" => 1e-4, 
# M model
"C" => 2e-5, 
"eta" => 25, 
"Ds" => 5e-4,  

# Beris-Edwards
"BEBool" => true, 
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
"GammaBE" => 1.0,
"BERand" => false,
"PInit" => sqrt(0.9),

# active params
"zeta" => 0.00,
"activityTimeOn" => 0,
"activityPattern" => (false),

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
"center" => [75, 125], # LU
"radius" => 12.5, # LU
"rB" => [175, 125],
"rtA" => 50,
"rtB" => 25050,
"aIB" => 25,
"nLegs" => 1,
"zetaIB" => 1,
"springK" => 0.5, 
"eps" => 0.3125, 
"kExt" => 0.05 
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
