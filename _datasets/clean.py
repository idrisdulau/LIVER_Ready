import os
import shutil

# sourcePath = "toSelf"

# # Rename source/target 
# for folder in sorted([f for f in os.listdir(sourcePath) if os.path.isdir(os.path.join(sourcePath,f))]): 
#     print(folder)
#     for subF in os.listdir(os.path.join(sourcePath,folder)):
#         print(">",subF)
#         if subF == "Display_map":
#             os.rename(os.path.join(sourcePath,folder,subF),os.path.join(sourcePath,folder,"source"))
#         elif subF == "Liver_mask":
#             os.rename(os.path.join(sourcePath,folder,subF),os.path.join(sourcePath,folder,"target"))

# # Normalize names from 1 to x
# ext = ".dcm"
# for folder in sorted([f for f in os.listdir(sourcePath) if os.path.isdir(os.path.join(sourcePath,f))]): 
#     for mode in ["source","target"]:
#         path = os.path.join(sourcePath,folder,mode)
#         filesList = [f for f in os.listdir(path)]
#         sortedList = sorted(filesList, key=lambda x: int(x.split('_')[-1].split('.')[0]))
#         for idx,file in enumerate(sortedList):
#             os.rename(os.path.join(path,file), os.path.join(path,str(idx+1)+ext))

# #Move to test
# for folder in sorted([f for f in os.listdir(sourcePath) if os.path.isdir(os.path.join(sourcePath,f))]): 
#     print(folder)
#     for mode in ["source","target"]:
#         path = os.path.join(sourcePath,folder,mode)
#         filesList = [f for f in os.listdir(path)]
#         sortedList = sorted(filesList, key=lambda x: int(x.split('_')[-1].split('.')[0]))

#         testPath = os.path.join(sourcePath,folder,mode,"test")
#         trainPath = os.path.join(sourcePath,folder,mode,"train")
#         os.makedirs(testPath, exist_ok=True)
#         os.makedirs(trainPath, exist_ok=True)  

#         for idx,file in enumerate(sortedList):
#             # print(idx,file)
#             if idx <3:
#                 shutil.move(os.path.join(path,file), os.path.join(testPath,file))
#             else:
#                 shutil.move(os.path.join(path,file), os.path.join(trainPath,file))

########################################################################################################

# sourcePath = "base"

# # Rename source/target 
# for folder in sorted([f for f in os.listdir(sourcePath) if os.path.isdir(os.path.join(sourcePath,f))]): 
#     print(folder)
#     for subF in os.listdir(os.path.join(sourcePath,folder)):
#         print(">",subF)
#         if subF == "Display_map":
#             os.rename(os.path.join(sourcePath,folder,subF),os.path.join(sourcePath,folder,"source"))
#         elif subF == "Liver_mask":
#             os.rename(os.path.join(sourcePath,folder,subF),os.path.join(sourcePath,folder,"target"))

# # Normalize names from 1 to x
# ext = ".dcm"
# for folder in sorted([f for f in os.listdir(sourcePath) if os.path.isdir(os.path.join(sourcePath,f))]): 
#     for mode in ["source","target"]:
#         path = os.path.join(sourcePath,folder,mode)
#         filesList = [f for f in os.listdir(path)]
#         sortedList = sorted(filesList, key=lambda x: int(x.split('_')[-1].split('.')[0]))
#         for idx,file in enumerate(sortedList):
#             os.rename(os.path.join(path,file), os.path.join(path,str(idx+1)+ext))

# # cpt rename
# cpt = 1
# for folder in sorted([f for f in os.listdir(sourcePath) if os.path.isdir(os.path.join(sourcePath,f))]):
#     filesList = [f for f in os.listdir(os.path.join(sourcePath,folder,"source"))]
#     # print("len(filesList)",len(filesList))
#     sortedList = sorted(filesList, key=lambda x: int(x.split('_')[-1].split('.')[0]))
#     # print("sortedList",sortedList)
#     # print("len(sortedList)",len(sortedList))
#     for file in sortedList:
#         print(folder, file, str(cpt)+".dcm")
#         # print(cpt)
#         # print(os.path.join(sourcePath,folder,"source",file), os.path.join(sourcePath,folder,"source",str(cpt)+".dcm"))
#         # print(os.path.join(sourcePath,folder,"target",file), os.path.join(sourcePath,folder,"target",str(cpt)+".dcm"))
#         destpath = os.path.join("renamed",folder,"source")
#         os.makedirs(destpath, exist_ok=True)
#         shutil.copy2(os.path.join(sourcePath,folder,"source",file), os.path.join(destpath,str(cpt)+".dcm"))
#         destpath = os.path.join("renamed",folder,"target")
#         os.makedirs(destpath, exist_ok=True)
#         shutil.copy2(os.path.join(sourcePath,folder,"target",file), os.path.join(destpath,str(cpt)+".dcm"))
#         cpt += 1
#     print("----------")


#Combine to form butSelf

# sourcePath = "butSelf"
# DBList = ["CA_029","CMC_030","GM_035","LM_036","MA_033","MC_031","MF_037","MP_034","SJP_028","SN_032"]
# for DB in DBList:
#     otherDBList = [x for x in DBList if x != DB]
#     for otherDB in otherDBList:
#         for mode in ["source","target"]:

#             inTrainPath = os.path.join("renamed",otherDB,mode)
#             destTrainPath = os.path.join(sourcePath,DB,mode,"train")
#             os.makedirs(os.path.join(destTrainPath), exist_ok=True)
#             for file in os.listdir(inTrainPath):
#                 shutil.copy2(os.path.join(inTrainPath,file), os.path.join(destTrainPath,file))
            
#             inTestPath = os.path.join("renamed",DB,mode)
#             destTestPath = os.path.join(sourcePath,DB,mode,"test")
#             os.makedirs(os.path.join(destTestPath), exist_ok=True)
#             for file in os.listdir(inTestPath):
#                 shutil.copy2(os.path.join(inTestPath,file), os.path.join(destTestPath,file))

