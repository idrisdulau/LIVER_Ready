import os
import subprocess

################################## --- toSelf --- ################################### 
### TRAIN

# GPU = 0
# MD = "baseVessels"
# DSmode = "toSelf"
# daMode = "Flip"
# for DB in ["CA_029","CMC_030","GM_035","LM_036","MA_033","MC_031","MF_037","MP_034","SJP_028","SN_032"]:
#     savePath = os.path.join("train","toSelf")
#     os.makedirs(savePath, exist_ok=True)
#     subprocess.run([
#         "python3", os.path.join("_run","train.py"),\
#             "--input        ",   os.path.join("_datasets",DSmode,DB,"source"),\
#             "--target       ",   os.path.join("_datasets",DSmode,DB,"target"),\
#             "--modelType    ",   MD,\
#             "--lossType    ",    "dice",\
#             "--save         ",   savePath,\
#             "--saveName     ",   DB,\
#             "--kernelWidth  ",   "256",\
#             "--strideWidth  ",   "1",\
#             "--kernelHeight ",   "192",\
#             "--strideHeight ",   "1",\
#             "--colorMode    ",   "Grayscale",\
#             "--daMode       ",   daMode,\
#             "--epochs       ",   "20",\
#             "--gpu          ",   str(GPU)
#         ])

### PRED

# GPU = 0
# MD = "baseVessels"
# DSmode = "toSelf"
# for DB in ["CA_029","CMC_030","GM_035","LM_036","MA_033","MC_031","MF_037","MP_034","SJP_028","SN_032"]:
    # modelLocation = os.path.join("train",DSmode,DB+"_20.pt")
    # savePath = os.path.join("pred",DSmode,DB)
    # print("savePath:", savePath)
    # os.makedirs(savePath, exist_ok=True)
    # subprocess.run([
    #     "python3", os.path.join("_run","predict.py"),\
    #     "--input        ",   os.path.join("_datasets",DSmode,DB,"source"),\
    #     "--output       ",   savePath,\
    #     "--kernelWidth  ",   "256",\
    #     "--kernelHeight ",   "192",\
    #     "--colorMode    ",   "Grayscale",\
    #     "--daMode       ",   "noDA",\
    #     "--modelType    ",   MD,\
    #     "--modelLocation",   modelLocation,\
    #     "--gpu          ",   str(GPU)
    # ])

### COMPARE

# GPU = 0
# MD = "baseVessels"
# DSmode = "toSelf"
# for DB in ["CA_029","CMC_030","GM_035","LM_036","MA_033","MC_031","MF_037","MP_034","SJP_028","SN_032"]:
#     gtPath = os.path.join("_datasets",DSmode,DB,"target","test")
#     predPath = os.path.join("pred",DSmode,DB)
#     subprocess.run([
#         "python3", os.path.join("_run","compare.py"),\
#             gtPath,\
#             predPath,\
#             os.path.join("stats"),\
#             "average",\
#             "CSV",\
#             DSmode,\
#             predPath.split("/")[-2]+" "+predPath.split("/")[-1]
#     ])


################################## --- toOthers --- ################################### 
### PRED

# GPU = 0
# MD = "baseVessels"
# DSmode = "toOthers"
# DBList = ["CA_029","CMC_030","GM_035","LM_036","MA_033","MC_031","MF_037","MP_034","SJP_028","SN_032"]
# for DB in DBList:
#     otherDBList = [x for x in DBList if x != DB]
#     modelLocation = os.path.join("train","toSelf",DB+"_20.pt")
#     for otherDB in otherDBList:
#         savePath = os.path.join("pred",DSmode,DB,otherDB)
#         print("savePath:", savePath)
#         os.makedirs(savePath, exist_ok=True)
#         subprocess.run([
#             "python3", os.path.join("_run","predict.py"),\
#             "--input        ",   os.path.join("_datasets","toSelf",otherDB,"source"),\
#             "--output       ",   savePath,\
#             "--kernelWidth  ",   "256",\
#             "--kernelHeight ",   "192",\
#             "--colorMode    ",   "Grayscale",\
#             "--daMode       ",   "noDA",\
#             "--modelType    ",   MD,\
#             "--modelLocation",   modelLocation,\
#             "--gpu          ",   str(GPU)
#         ])


### COMPARE

# GPU = 0
# MD = "baseVessels"
# DSmode = "toOthers"
# DBList = ["CA_029","CMC_030","GM_035","LM_036","MA_033","MC_031","MF_037","MP_034","SJP_028","SN_032"]
# for DB in DBList:
#     otherDBList = [x for x in DBList if x != DB]
#     for otherDB in otherDBList:
#         gtPath = os.path.join("_datasets","toSelf",otherDB,"target","test")
#         predPath = os.path.join("pred",DSmode,DB,otherDB)
#         subprocess.run([
#             "python3", os.path.join("_run","compare.py"),\
#                 gtPath,\
#                 predPath,\
#                 os.path.join("stats"),\
#                 "average",\
#                 "CSV",\
#                 DSmode+"_"+DB,\
#                 predPath.split("/")[-2]+" "+predPath.split("/")[-1]
#         ])



################################## --- butSelf --- ################################### 
### TRAIN

# GPU = 0
# MD = "baseVessels"
# DSmode = "butSelf"
# daMode = "Flip"
# for DB in ["CA_029","CMC_030","GM_035","LM_036","MA_033","MC_031","MF_037","MP_034","SJP_028","SN_032"]:
#     savePath = os.path.join("train",DSmode)
#     os.makedirs(savePath, exist_ok=True)
#     subprocess.run([
#         "python3", os.path.join("_run","train.py"),\
#             "--input        ",   os.path.join("_datasets",DSmode,DB,"source"),\
#             "--target       ",   os.path.join("_datasets",DSmode,DB,"target"),\
#             "--modelType    ",   MD,\
#             "--lossType    ",    "dice",\
#             "--save         ",   savePath,\
#             "--saveName     ",   DB,\
#             "--kernelWidth  ",   "256",\
#             "--strideWidth  ",   "1",\
#             "--kernelHeight ",   "192",\
#             "--strideHeight ",   "1",\
#             "--colorMode    ",   "Grayscale",\
#             "--daMode       ",   daMode,\
#             "--epochs       ",   "50",\
#             "--gpu          ",   str(GPU)
#         ])

### PRED

# GPU = 0
# MD = "baseVessels"
# DSmode = "butSelf"
# for DB in ["CA_029","CMC_030","GM_035","LM_036","MA_033","MC_031","MF_037","MP_034","SJP_028","SN_032"]:
#     modelLocation = os.path.join("train",DSmode,DB+"_50.pt")
#     savePath = os.path.join("pred",DSmode,DB)
#     print("savePath:", savePath)
#     os.makedirs(savePath, exist_ok=True)
#     subprocess.run([
#         "python3", os.path.join("_run","predict.py"),\
#         "--input        ",   os.path.join("_datasets",DSmode,DB,"source"),\
#         "--output       ",   savePath,\
#         "--kernelWidth  ",   "256",\
#         "--kernelHeight ",   "192",\
#         "--colorMode    ",   "Grayscale",\
#         "--daMode       ",   "noDA",\
#         "--modelType    ",   MD,\
#         "--modelLocation",   modelLocation,\
#         "--gpu          ",   str(GPU)
#     ])

### COMPARE

# GPU = 0
# MD = "baseVessels"
# DSmode = "butSelf"
# for DB in ["CA_029","CMC_030","GM_035","LM_036","MA_033","MC_031","MF_037","MP_034","SJP_028","SN_032"]:
#     gtPath = os.path.join("_datasets",DSmode,DB,"target","test")
#     predPath = os.path.join("pred",DSmode,DB)
#     subprocess.run([
#         "python3", os.path.join("_run","compare.py"),\
#             gtPath,\
#             predPath,\
#             os.path.join("stats"),\
#             "average",\
#             "CSV",\
#             DSmode,\
#             predPath.split("/")[-2]+" "+predPath.split("/")[-1]
#     ])