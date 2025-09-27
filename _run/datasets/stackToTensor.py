import os
import cv2
import torch 
import numpy
import pydicom
from utils.dataAugmentation import *

class stackToTensor(torch.utils.data.Dataset):
    
    def __init__(self, input, target, colorMode, daMode):
        super(stackToTensor, self).__init__()
        self.sourceDir, self.targetDir, self.colorMode, self.daMode = input, target, colorMode, daMode
        self.sourceValidItems, self.targetValidItems = stackToTensor.validCouples(input, target)
        self.length = len(self.sourceValidItems)
        self.length = min(16,len(self.sourceValidItems))
        # print("colorMode :", colorMode, "| daMode :", daMode, "| stackToTensor length:",self.length)

    def __getitem__(self, index):

        sourceItem = self.sourceValidItems[index]

        sourcePath = self.sourceDir.replace(" ", "")
    
        # Initialize a list to hold each 2D slice
        sourceSlices = []
        # Loop through all files in the directory and load them
        for item in self.sourceValidItems:
            sourceFile = os.path.join(sourcePath, item)
            # Read DICOM file and get the pixel array
            sourceImage = pydicom.dcmread(sourceFile).pixel_array.astype(numpy.uint8)*1
            sourceSlices.append(sourceImage)
        

        # print("sourceSlices:",len(sourceSlices))

        # Stack all slices along a new axis to create a 3D volume
        sourceStack = numpy.stack(sourceSlices, axis=0)  # (depth, height, width)
        sourceStack = sourceStack[numpy.newaxis, :, :, :]
        # print("sourceStack.shape:",sourceStack.shape)

        targetPath = self.targetDir.replace(" ", "")
        targetSlices = []
        for item in self.targetValidItems:
            targetFile = os.path.join(targetPath, item)
            targetImage = pydicom.dcmread(targetFile).pixel_array.astype(numpy.uint8)*1
            targetSlices.append(targetImage)
        # print("targetSlices:",len(targetSlices))

        targetStack = numpy.stack(targetSlices, axis=0)  # (depth, height, width)
        targetStack = targetStack[numpy.newaxis, :, :, :]
        # print("targetStack.shape:",targetStack.shape)

        #No DA for the test
        # sourceItem = self.sourceValidItems[index]
        # sourcePath = self.sourceDir.replace(" ", "")
        # sourceFile = os.path.join(sourcePath, sourceItem)
        # sourceImage = pydicom.dcmread(sourceFile).pixel_array.astype(numpy.uint8)*255
        # # sourceImage = cv2.imread(sourceFile, cv2.IMREAD_UNCHANGED)
        
        # targetItem = self.targetValidItems[index]
        # targetPath = self.targetDir.replace(" ", "")
        # targetFile = os.path.join(targetPath, targetItem)
        # targetImage = pydicom.dcmread(targetFile).pixel_array.astype(numpy.uint8)*255
        # # targetImage = cv2.imread(targetFile, cv2.IMREAD_UNCHANGED)    

        # print("target ",targetImage.shape)

        # sourceTensor, targetTensor = callTransforms(sourceImage, targetImage, self.daMode)

        # sourceTensor = torchvision.transforms.ToTensor()(sourceStack).permute(0,2,1)
        # targetTensor = torchvision.transforms.ToTensor()(targetStack).permute(0,2,1)
        
        # print("sourceTensor ",sourceTensor.shape)
        # print("targetTensor ",targetTensor.shape)

        # print("end")
        # exit()


        # To display 
        # sourceImg = toImage(sourceTensor.permute(1,2,0))
        # targetImg = toImage(targetTensor.permute(1,2,0))
        # combinedImg = cv2.hconcat([sourceImg, cv2.merge((targetImg,targetImg,targetImg))])
        # cv2.imshow("source <---> target", combinedImg)
        # cv2.waitKey(0)
       
        if self.colorMode == "AVO":
            black = torch.zeros((1,1024,1024))
            targetTensor = torch.cat((black, targetTensor), dim=0)

        return [sourceTensor, targetTensor, sourceItem]
        
    def __len__(self):
        return self.length

    def validCouples(input, target):
        _, extension = os.path.splitext(os.listdir(input)[0])

        sourceList = os.listdir(input)
        sourceIdxList = []
        for i in range (len(sourceList)):
            idx, _ = os.path.splitext(sourceList[i])
            sourceIdxList.append(int(idx))
        sourceIdxList.sort() 

        targetList = os.listdir(target)
        targetIdxList = []
        for i in range (len(targetList)):
            idx, _ = os.path.splitext(targetList[i])
            targetIdxList.append(int(idx))
        targetIdxList.sort() 

        notBothIdxList = []
        for i in range (len(sourceIdxList)-1):
            if sourceIdxList[i] not in targetIdxList:
                notBothIdxList.append(i)
        for i in range (len(targetIdxList)-1):
            if targetIdxList[i] not in sourceIdxList:
                notBothIdxList.append(i)

        itemToExclude = [str(item)+extension for item in notBothIdxList]
        sourceList.sort()
        sourceValidItems = [item for item in sourceList if item not in itemToExclude]
        targetList.sort()
        targetValidItems = [item for item in targetList if item not in itemToExclude]
        return sourceValidItems, targetValidItems
