import os
import sys
import torch

from models.unets import BaseVessels, BaseVeins, BaseArteries, BaseOD, liver3D, NewKernel, BaseVessels_drop
# from models.dvaefl import DVAE
from models.imported import Resnet50_Last, Resnet50_Full, CORN, ROAD, WALL, VESSELS, \
                            FROMVESSELS_Full_K1, FROMVESSELS_Full_K2, FROMVESSELS_Full_K3, \
                            FROMVESSELS_Decoder_K1, FROMVESSELS_Decoder_K2, FROMVESSELS_Decoder_K3, \
                            FROMVESSELS_Final_K1, FROMVESSELS_Final_K2, FROMVESSELS_Final_K3,\
                            FROMVESSELS_toAlienor, FROMARTERIES_toAlienor, FROMVEINS_toAlienor
# from models.imported import SAM
# from models.transformers import Transformer


from utils.buildImage import buildImage

def usage():
    print("An error as occured, you should provide these options and arguments:")
    print()
    print(" Options         : Arguments")
    print()
    print(" --input         : Path to input data folder")
    print(" --roi           : Path to roi data folder")
    print(" --output        : Path to save the newly generated segmented files")
    print(" --kernelWidth   : Patches width")
    print(" --kernelHeight  : Patches height") 
    print(" --colorMode     : Color mode") 
    print(" --daMode        : Data Augmentation mode") 
    print(" --modelType     : Model definition file name")
    print(" --modelLocation : Path to the saved model's weights file, including file name")
    print(" --gpu           : GPU index")
    print()

def checkInput(optionsList, argumentsList):
    inputOpt = "--input"
    assert optionsList.count(inputOpt), usage()
    idx = optionsList.index(inputOpt)
    assert os.path.isdir(argumentsList[idx]), usage()
    return argumentsList[idx]

def checkRoi(optionsList, argumentsList):
    roiOpt = "--roi"
    assert optionsList.count(roiOpt), usage()
    idx = optionsList.index(roiOpt)
    assert os.path.isdir(argumentsList[idx]), usage()
    return argumentsList[idx]

def checkOutput(optionsList, argumentsList):
    outputOpt = "--output"
    assert optionsList.count(outputOpt), usage()
    idx = optionsList.index(outputOpt)
    assert os.path.isdir(argumentsList[idx]), usage()
    return argumentsList[idx]

def checkKernelWidth(optionsList, argumentsList):
    kernelWidthOpt = "--kernelWidth"
    assert optionsList.count(kernelWidthOpt), usage()
    idx = optionsList.index(kernelWidthOpt)
    assert argumentsList[idx].isdigit and int(argumentsList[idx]) >= 0, usage()
    return argumentsList[idx]

def checkKernelHeight(optionsList, argumentsList):
    kernelHeightOpt = "--kernelHeight"
    assert optionsList.count(kernelHeightOpt), usage()
    idx = optionsList.index(kernelHeightOpt)
    assert argumentsList[idx].isdigit and int(argumentsList[idx]) >= 0, usage()
    return argumentsList[idx]

def checkColorMode(optionsList, argumentsList):
    colorModeOpt = "--colorMode"
    assert optionsList.count(colorModeOpt), usage()
    idx = optionsList.index(colorModeOpt)
    colorModeList = ["RGB","Green","Grayscale","AVO"]
    assert colorModeList.count(argumentsList[idx]), usage()
    return argumentsList[idx]

def checkDaMode(optionsList, argumentsList):
    daModeOpt = "--daMode"
    assert optionsList.count(daModeOpt), usage()
    idx = optionsList.index(daModeOpt)
    daModeList = ["Flip","Rota","Shear","Elastic","Brightness","Contrast","Cutout","Blur", "noDA"]
    argumentsList[idx] = argumentsList[idx].split(",")
    assert set(argumentsList[idx]).issubset(daModeList), usage()
    return argumentsList[idx]

def checkModelType(optionsList, argumentsList):
    modelTypeOpt = "--modelType"
    assert optionsList.count(modelTypeOpt), usage()
    idx = optionsList.index(modelTypeOpt)
    modelTypeList = ["base","baseBN1","baseBN2","baseIN1","baseIN2", "baseDrop","baseGCT", \
                     "baseD2","baseD3","baseD5","baseD6","baseD7", \
                     "vesselsSum","veinsSum","arteriesSum","odSum", \
                     "vesselsStride","veinsStride","arteriesStride","odStride", \
                     "vesselsNearest","veinsNearest","arteriesNearest","odNearest", \
                     "vesselsBilinear","veinsBilinear","arteriesBilinear","odBilinear", \
                     "vesselsBicubic","veinsBicubic","arteriesBicubic","odBicubic", \
                     "baseVessels", "baseVeins", "baseArteries", "baseOD", \
                     "resnet50_no","resnet50_last","resnet50_full", \
                     "corn","road","wall","vessels", \
                     "fromvessels_full_k1","fromvessels_full_k2","fromvessels_full_k3", \
                     "fromvessels_decoder_k1","fromvessels_decoder_k2","fromvessels_decoder_k3", \
                    "fromvessels_final_k1","fromvessels_final_k2","fromvessels_final_k3", \
                     "fromvessels_toAlienor","fromarteries_toAlienor","fromveins_toAlienor",\
                     "unetAVO","dvae","transformer","sam","liver3D","NewKernel","BaseVessels_drop"]
    assert modelTypeList.count(argumentsList[idx]), usage()
    return argumentsList[idx]

def checkModelLocation(optionsList, argumentsList):
    modelLocationOpt = "--modelLocation"
    assert optionsList.count(modelLocationOpt), usage()
    idx = optionsList.index(modelLocationOpt)
    assert os.path.isfile(argumentsList[idx]), usage()
    return argumentsList[idx]

def checkGpu(optionsList, argumentsList):
    gpuOpt = "--gpu"
    assert optionsList.count(gpuOpt), usage()
    idx = optionsList.index(gpuOpt)
    assert argumentsList[idx].isdigit and int(argumentsList[idx]) >= 0, usage()
    return argumentsList[idx]

def main(argv):

    optionsList = [opt.replace(" ", "") for opt in argv[1::2]]
    argumentsList = [arg.replace(" ", "") for arg in argv[2::2]]

    inputArg = checkInput(optionsList, argumentsList)
    # roiArg = checkRoi(optionsList, argumentsList)
    outputArg = checkOutput(optionsList, argumentsList)
    kernelWidthArg = checkKernelWidth(optionsList, argumentsList)
    kernelHeightArg = checkKernelHeight(optionsList, argumentsList)
    daModeArg = checkDaMode(optionsList, argumentsList)
    colorModeArg = checkColorMode(optionsList, argumentsList)
    modelTypeArg = checkModelType(optionsList, argumentsList)  
    modelLocationArg = checkModelLocation(optionsList, argumentsList)
    gpuArg = checkGpu(optionsList, argumentsList)

    device = torch.device("cuda:0")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuArg

    for mName,mClass in [("baseVessels",BaseVessels), ("baseVeins",BaseVeins), ("baseArteries",BaseArteries), ("baseOD",BaseOD), \
                         ("liver3D",liver3D), ("NewKernel",NewKernel), ("BaseVessels_drop",BaseVessels_drop), \
                         ("resnet50_last",Resnet50_Last),("resnet50_full",Resnet50_Full), \
                         ("corn",CORN), ("road",ROAD), ("wall",WALL), ("vessels",VESSELS), \
                         ("fromvessels_full_k1",FROMVESSELS_Full_K1), ("fromvessels_full_k2",FROMVESSELS_Full_K2), ("fromvessels_full_k3",FROMVESSELS_Full_K3), \
                         ("fromvessels_decoder_k1",FROMVESSELS_Decoder_K1), ("fromvessels_decoder_k2",FROMVESSELS_Decoder_K2), ("fromvessels_decoder_k3",FROMVESSELS_Decoder_K3), \
                         ("fromvessels_final_k1",FROMVESSELS_Final_K1), ("fromvessels_final_k2",FROMVESSELS_Final_K2), ("fromvessels_final_k3",FROMVESSELS_Final_K3),\
                         ("fromvessels_toAlienor",FROMVESSELS_toAlienor), ("fromarteries_toAlienor",FROMARTERIES_toAlienor), ("fromveins_toAlienor",FROMVEINS_toAlienor)
                         ]:
        if modelTypeArg == mName:
            print("mClass:",mClass)
            assert(colorModeArg == "RGB" or colorModeArg == "Green" or colorModeArg == "Grayscale")        
            if colorModeArg == "RGB":
                model = mClass(3,1)
            else:
                model = mClass(1,1)
            model.load_state_dict(torch.load(modelLocationArg))
            model.to(device)
            buildImage(model, inputArg, outputArg, kernelWidthArg, kernelHeightArg, colorModeArg, daModeArg)

    if modelTypeArg == "unetAVO":
        assert(colorModeArg == "AVO")       
        model = UnetAVO(3,4)
        model.load_state_dict(torch.load(modelLocationArg))
        model.to(device)
        buildImage(model, inputArg, outputArg, kernelWidthArg, kernelHeightArg, colorModeArg, daModeArg)

if __name__ == '__main__':
    main(sys.argv)
