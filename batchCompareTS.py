import os
import numpy
import nibabel
import subprocess
from PIL import Image

# for DB in [str(e) for e in range(18,38)]:
#     niiFile = os.path.join("totalSegmentorNii",DB+".nii")
#     outputDir = os.path.join("totalSegmentorTif",DB)

#     os.makedirs(outputDir, exist_ok=True)

#     nii = nibabel.load(niiFile)
#     data = nii.get_fdata()

#     for i in range(data.shape[2]): #Axial
#         sliceData = data[:, :, i]

#         sliceNorm = (sliceData > 0).astype(numpy.uint8) * 255

#         sliceOriented = numpy.rot90(sliceNorm)

#         im = Image.fromarray(sliceOriented)
#         im.save(os.path.join(outputDir, str(i+1)+".tif"))

#     print("Done")

# SAVENAME = "GT_VS_SOTA"
# for DB in [str(e) for e in range(18,38)]:
#     gtPath = os.path.join("_datasets","stacks","targetSplitted","test",DB)
#     predPath = os.path.join("totalSegmentorTif",DB) #SOTA
#     subprocess.run([
#         "python3", os.path.join("_run","compare.py"),\
#             gtPath,\
#             predPath,\
#             os.path.join("statsSOTA"),\
#             "details",\
#             "CSV",\
#             SAVENAME,\
#             DB
#     ])


# SAVENAME = "GT_VS_PRED"
# for DB in ["20","24","25","31","34","37"]:
#     gtPath = os.path.join("_datasets","stacks","targetSplitted","test",DB)
#     predPath = os.path.join("pred","3D","stacks","liver3D",DB) #No post
#     subprocess.run([
#         "python3", os.path.join("_run","compare.py"),\
#             gtPath,\
#             predPath,\
#             os.path.join("statsSOTA"),\
#             "details",\
#             "CSV",\
#             SAVENAME,\
#             DB
#     ])


# SAVENAME = "GT_VS_POST"
# for DB in ["20","24","25","31","34","37"]:
#     gtPath = os.path.join("_datasets","stacks","targetSplitted","test",DB)
#     predPath = os.path.join("post",DB+"_pred_one") #Post
#     subprocess.run([
#         "python3", os.path.join("_run","compare.py"),\
#             gtPath,\
#             predPath,\
#             os.path.join("statsSOTA"),\
#             "details",\
#             "CSV",\
#             SAVENAME,\
#             DB
#     ])