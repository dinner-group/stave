CCytVec = vcat([0], (10).^(range(4,stop=9,length=6)))
etaCytVec = vcat([0], (10).^(range(0,stop=5,length=6)))

k0Vec = [0.2, 0.4, 0.6, 0.8, 1.0, 2.5, 5.0]
etaVec = [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 1.0]

trialParamDict = Dict(
"AA" => AAVec
#"K0" => K0Vec
)

global baseDir = "/project/svaikunt/csfloyd/CatchHopfield/"
global Dirs = baseDir * "Dirs/Dirs_OI_AA_pbc_TG/"

function ReplaceBatchRunDelete(batchFile, inputFile, outputDir)
    f = open(batchFile)
    (tmppath, tmpio) = mktemp()
    try
        lines = readlines(f)
        for l in lines
            sl = split(l)
            if (length(sl) > 0) && (sl[1] == "julia")
                ns = replace(l, "inputFile" => inputFile)
                ns = replace(ns, "outputDir" => outputDir)
                write(tmpio, ns)
                write(tmpio, "\n")

            else
                write(tmpio, l)
                write(tmpio, "\n")

            end
        end
    finally
        close(f)
        close(tmpio)
    end
    newBatch = outputDir * "slatboltz.sh"
    mv(tmppath, newBatch, force = true)
    oldDir = pwd()
    cd(outputDir)
    try
        run(`sbatch slatboltz.sh`)
    catch
        println("Failed to submit the batch job.")
    end
    cd(oldDir)
    rm(newBatch, force = true)
end

function AddLineToFile!(inputFile, newLine)
    f = open(inputFile, "a")
    write(f, newLine)
    write(f, "\n")
    close(f)
end


function MakeDirectoriesAndRun(currParamDict, currDir, currInputFile)

    global baseDir

    currParam = collect(keys(currParamDict))[1]

    if length(collect(keys(currParamDict))) == 1 # reached the bottom
        for val in currParamDict[currParam]
            # create the new directory
            newDir = currDir * "$currParam"*"_"*"$val/"
            mkdir(newDir)
            # update the input file
            newInputFile = newDir * "inputFile.txt"
            newLine = string(currParam) * "     " * string(val)
            cp(currInputFile, newInputFile)
            AddLineToFile!(newInputFile, newLine)
            cp(baseDir * "latboltz.sh", newDir * "latboltz.sh", force = true)
            ReplaceBatchRunDelete(newDir * "latboltz.sh", newInputFile, newDir)
            rm(newDir * "latboltz.sh")
        end
        return
    end

    newParamDict = deepcopy(currParamDict)
    delete!(newParamDict, currParam)

    for val in currParamDict[currParam]
        # create the new directory
        newDir = currDir * "$currParam"*"_"*"$val/"
        mkdir(newDir)
        # update the input file
        newInputFile = newDir * "tempInputFile.txt"
        newLine = string(currParam) * "     " * string(val)
        cp(currInputFile, newInputFile)
        AddLineToFile!(newInputFile, newLine)
        # do the recursive call
        MakeDirectoriesAndRun(newParamDict, newDir, newInputFile)
        # delete the temporary input file
        rm(newDir * "tempInputFile.txt")
    end
end


# make sure the directory is empty
try
    rm(Dirs, recursive = true)
catch
end
mkdir(Dirs)

# do it all
MakeDirectoriesAndRun(trialParamDict, Dirs, baseDir * "/baseInput.txt")
