# import os
# import cv2
# import numpy
# import tifffile

# from PIL import Image

# # Iterate over all .tif files in the "train" folder
# # folderPath = os.path.join("stacks","targetSplitted","test")
# folderPath = os.path.join("..","post")
# for filename in os.listdir(folderPath):
    

#     if filename.endswith(".tif"):
#         # print(filename)

#         newFolderpath = os.path.join(folderPath,filename.split(".")[0])
#         os.makedirs(newFolderpath, exist_ok=True)
#         # print("test",os.path.join(folderPath,filename))


#         # Open the .tif stack and split into individual images
#         tiffStack = os.path.join(folderPath,filename)
#         print(tiffStack)
#         with tifffile.TiffFile(tiffStack) as tif:
#             images = tif.asarray()
#             print("aaaaaaaaa --", images.shape)
#             if len(images.shape) == 3:  # Ensure the stack contains multiple images
#                 for i, image in enumerate(images):
#                     image_path = os.path.join(newFolderpath, f"{i + 1}.tif")
#                     # print(image_path)
#                     img = Image.fromarray(image)
#                     img.save(image_path)

#         # print("-------------------")

#         for filename in sorted(os.listdir(newFolderpath), key=str):
#             img = cv2.imread(os.path.join(newFolderpath,filename), cv2.IMREAD_UNCHANGED)
#             # print("numpy.max(img)",numpy.max(img))
#             img = numpy.uint8(img)
#             cv2.imwrite(os.path.join(newFolderpath,filename), img)
