import os
import subprocess

################################## --- butSelf --- ################################### 
### TRAIN

# GPU = 0
# MD = "BaseVessels_drop" #baseVessels", "NewKernel"
# # MD = "resnet50_last"
# DSmode = "new17ButSelf"
# daMode = "Flip"
# # for DB in ["AD_068","BB_038","BG_052","BL_047","CA_029"]:
# # for DB in ["CMC_030","DM_050","FP_046","GM_035","KK_062","LM_039"]:
# # for DB in ["MC_031","MF_037","MP_034","OM_065","SF_053","SJP_028"]:


# for DB in ["AD_068","BB_038","BG_052","BL_047","CA_029","CMC_030","DM_050","FP_046","GM_035","KK_062", \
#     "LM_039","MC_031","MF_037","MP_034","OM_065","SF_053","SJP_028"]:

#     savePath = os.path.join("train",DSmode,MD)
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
# MD = "resnet50_last" #"baseOD" "BaseVessels_drop" "resnet50_last" "NewKernel"
# DSmode = "new17ButSelf"
# new17List = ["AD_068","BB_038","BG_052","BL_047","CA_029","CMC_030","DM_050","FP_046","GM_035","KK_062",\
#            "LM_039","MC_031","MF_037","MP_034","OM_065","SF_053","SJP_028"]
# fullList = ["AD_068","BB_038","BG_052","BJM_045","BJ_043","BL_047","CA_029","CB_044","CC_059",\
#             "CMC_030","DM_050","DM_061","DN_072","FP_046","GM_035","KK_062","LM_036","LM_039","MA_033",\
#             "MB_058","MC_031","MC_067","MD_040","MF_037","ML_049","MP_034","MR_070","MY_041","OM_064",\
#             "OM_065","PJC_071","RF_063","RG_042","RM_069","SF_053","SF_066","SJP_028","SN_032","TA_051","TD_056"]
# otherList = [e for e in fullList if e not in new17List]
# for DB in otherList:
#     modelLocation = os.path.join("train",DSmode,MD,"CA_029_50.pt")
#     savePath = os.path.join("pred",DSmode+"_others",MD,DB)    
# # for DB in new17List:
# #     modelLocation = os.path.join("train",DSmode,MD,DB+"_50.pt")
# #     savePath = os.path.join("pred",DSmode,MD,DB)
#     print("savePath:", savePath)
#     os.makedirs(savePath, exist_ok=True)
#     subprocess.run([
#         "python3", os.path.join("_run","predict.py"),\
#         "--input        ",   os.path.join("_datasets","extendedButSelf",DB,"source"),\
#         "--output       ",   savePath,\
#         "--kernelWidth  ",   "256",\
#         "--kernelHeight ",   "192",\
#         "--colorMode    ",   "Grayscale",\
#         "--daMode       ",   "noDA",\
#         "--modelType    ",   MD,\
#         "--modelLocation",   modelLocation,\
#         "--gpu          ",   str(GPU)
#     ])

# ### COMPARE

# GPU = 0
# MD = "resnet50_last" #"baseOD" "BaseVessels_drop" "resnet50_last" "NewKernel"
# DSmode = "new17ButSelf"
# new17List = ["AD_068","BB_038","BG_052","BL_047","CA_029","CMC_030","DM_050","FP_046","GM_035","KK_062",\
#            "LM_039","MC_031","MF_037","MP_034","OM_065","SF_053","SJP_028"]
# fullList = ["AD_068","BB_038","BG_052","BJM_045","BJ_043","BL_047","CA_029","CB_044","CC_059",\
#             "CMC_030","DM_050","DM_061","DN_072","FP_046","GM_035","KK_062","LM_036","LM_039","MA_033",\
#             "MB_058","MC_031","MC_067","MD_040","MF_037","ML_049","MP_034","MR_070","MY_041","OM_064",\
#             "OM_065","PJC_071","RF_063","RG_042","RM_069","SF_053","SF_066","SJP_028","SN_032","TA_051","TD_056"]
# # otherList = [e for e in fullList if e not in new17List]
# # print(len(otherList))
# # for DB in otherList:
# #     predPath = os.path.join("pred","new17ButSelf_others",MD,DB)
# for DB in new17List :
#     predPath = os.path.join("pred",DSmode,MD,DB)

#     gtPath = os.path.join("_datasets","extendedButSelf",DB,"target","test")
#     subprocess.run([
#         "python3", os.path.join("_run","compare.py"),\
#             gtPath,\
#             predPath,\
#             os.path.join("stats"),\
#             "average",\
#             "CSV",\
#             DSmode+"_resnet50",\
#             predPath.split("/")[-2]+" "+predPath.split("/")[-1]
#     ])


#RERUN to compare

# GPU = 0
# MD = "baseVessels" #"baseOD"
# DSmode = "fromOldCA_029"
# for DB in ["AD_068","BB_038","BG_052","BJM_045","BJ_043","BL_047","CA_029","CB_044","CC_059",\
#     "CMC_030","DM_050","DM_061","DN_072","FP_046","GM_035","KK_062","LM_036","LM_039","MA_033",\
#     "MB_058","MC_031","MC_067","MD_040","MF_037","ML_049","MP_034","MR_070","MY_041","OM_064",\
#     "OM_065","PJC_071","RF_063","RG_042","RM_069","SF_053","SF_066","SJP_028","SN_032","TA_051","TD_056"]:
#     modelLocation = os.path.join("train","butSelf","CA_029"+"_50.pt")
#     savePath = os.path.join("pred",DSmode,DB)
#     print("savePath:", savePath)
#     os.makedirs(savePath, exist_ok=True)
#     subprocess.run([
#         "python3", os.path.join("_run","predict.py"),\
#         "--input        ",   os.path.join("_datasets","extendedButSelf",DB,"source"),\
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
# MD = "baseVessels" #"baseOD"
# DSmode = "fromOldCA_029"
# for DB in ["AD_068","BB_038","BG_052","BJM_045","BJ_043","BL_047","CA_029","CB_044","CC_059",\
#     "CMC_030","DM_050","DM_061","DN_072","FP_046","GM_035","KK_062","LM_036","LM_039","MA_033",\
#     "MB_058","MC_031","MC_067","MD_040","MF_037","ML_049","MP_034","MR_070","MY_041","OM_064",\
#     "OM_065","PJC_071","RF_063","RG_042","RM_069","SF_053","SF_066","SJP_028","SN_032","TA_051","TD_056"]:
#     gtPath = os.path.join("_datasets","extendedButSelf",DB,"target","test")
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



################################## --- butSelf --- ################################### 
### TRAIN

# GPU = 0
# MD = "liver3D" #"baseVessels" #"liver3D"
# DSmode = "new17ButSelf" #"toSelf"
# daMode = "noDA" #"Flip"

# for DB in ["CMC_030"]: 
#     savePath = os.path.join("train","3D",DSmode,MD)
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
#             "--epochs       ",   "10",\
#             "--gpu          ",   str(GPU)
#         ])

### PRED

# for DB in ["CMC_030"]: 
#     modelLocation = os.path.join("train","3D",DSmode,MD,"CMC_030_10.pt")
#     savePath = os.path.join("pred","3D",DSmode,MD,DB)    
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


############################################ TIF STACKS

GPU = 0
MD = "liver3D"
DSmode = "stacks"
daMode = "noDA"

### TRAIN (Over the 17 best stacks, and pred over the 20 bad)

savePath = os.path.join("train","3D",DSmode,MD)
os.makedirs(savePath, exist_ok=True)
subprocess.run([
    "python3", os.path.join("_run","train.py"),\
        "--input        ",   os.path.join("_datasets",DSmode,"source"),\
        "--target       ",   os.path.join("_datasets",DSmode,"target"),\
        "--modelType    ",   MD,\
        "--lossType    ",    "dice",\
        "--save         ",   savePath,\
        "--saveName     ",   "allBN_DA_Rota_X4",\
        "--kernelWidth  ",   "256",\
        "--strideWidth  ",   "1",\
        "--kernelHeight ",   "192",\
        "--strideHeight ",   "1",\
        "--colorMode    ",   "Grayscale",\
        "--daMode       ",   daMode,\
        "--epochs       ",   "50",\
        "--gpu          ",   str(GPU)
    ])

### PRED

# modelName = "allBN_DA_Base_50.pt"
# modelName = "allBN_DA_Dupl_50.pt"

# modelLocation = os.path.join("train","3D",DSmode,MD,modelName)
# savePath = os.path.join("pred","3D",DSmode,modelName.split(".")[0])    
# print("savePath:", savePath)
# os.makedirs(savePath, exist_ok=True)
# subprocess.run([
#     "python3", os.path.join("_run","predict.py"),\
#     "--input        ",   os.path.join("_datasets",DSmode,"source"),\
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
# MD = "baseVessels" #"baseOD"
# for DB in [str(e) for e in range(18,38)]:
#     gtPath = os.path.join("_datasets","stacks","targetSplitted","test",DB)
#     predPath = os.path.join("pred","3D","stacks",modelName.split(".")[0],DB)
#     subprocess.run([
#         "python3", os.path.join("_run","compare.py"),\
#             gtPath,\
#             predPath,\
#             os.path.join("stats"),\
#             "average",\
#             "CSV",\
#             modelName.split(".")[0],\
#             predPath.split("/")[-2]+" "+predPath.split("/")[-1]
#     ])


