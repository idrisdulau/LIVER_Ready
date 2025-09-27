import os
import cv2
import torch 
import numpy
import torchio
import pydicom
import skimage.io
from utils.dataAugmentation import *

class imageToTensor(torch.utils.data.Dataset):
    
    def __init__(self, input, target, colorMode, daMode):
        super(imageToTensor, self).__init__()
        self.sourceDir, self.targetDir, self.colorMode, self.daMode = input, target, colorMode, daMode
        self.sourceValidItems, self.targetValidItems = imageToTensor.validCouples(input, target)
        self.length = len(self.sourceValidItems)
        # self.length = min(5,len(self.sourceValidItems))
        # print("colorMode :", colorMode, "| daMode :", daMode, "| imageToTensor length:",self.length)

    def __getitem__(self, index):
        
        sourceItem = self.sourceValidItems[index]
        sourcePath = self.sourceDir.replace(" ", "")
        sourceFile = os.path.join(sourcePath, sourceItem)
        # sourceImage = pydicom.dcmread(sourceFile).pixel_array.astype(numpy.uint8)*255
        sourceImage = skimage.io.imread(sourceFile)
        # sourceImage = sourceImage.transpose(1,2,0)
        sourceImage = sourceImage.reshape(1, sourceImage.shape[0], sourceImage.shape[1], sourceImage.shape[2])
        sourceTensor = torch.tensor(sourceImage).float()
        # sourceImage = cv2.imread(sourceFile, cv2.IMREAD_UNCHANGED)

        
        targetItem = self.targetValidItems[index]
        targetPath = self.targetDir.replace(" ", "")
        targetFile = os.path.join(targetPath, targetItem)
        # targetImage = pydicom.dcmread(targetFile).pixel_array.astype(numpy.uint8)*255
        targetImage = skimage.io.imread(targetFile)
        # targetImage = targetImage.transpose(2,1,0)
        targetImage = targetImage.reshape(1, targetImage.shape[0], targetImage.shape[1], targetImage.shape[2])
        targetTensor = torch.tensor(targetImage).float()
        # targetImage = cv2.imread(targetFile, cv2.IMREAD_UNCHANGED)    

        # sourceTensor, targetTensor = callTransforms(sourceImage, targetImage, self.daMode)

        # To display 
        # sourceImg = toImage(sourceTensor.permute(1,2,0))
        # targetImg = toImage(targetTensor.permute(1,2,0))
        # combinedImg = cv2.hconcat([sourceImg, cv2.merge((targetImg,targetImg,targetImg))])
        # cv2.imshow("source <---> target", combinedImg)
        # cv2.waitKey(0)

        # print("sourceTensor:",sourceTensor.shape)
        # print("targetTensor:",targetTensor.shape)

        #TODO : don't do in pred mode
        # transform = torchio.transforms.RandomFlip(flip_probability=0.75)
        # sourceTensor = transform(sourceTensor)
        # targetTensor = transform(targetTensor)

        #TODO : FLips
        # subMode = random.choice(["none", "hFlip", "vFlip", "hvFlip"])  # Choose mode

        # if subMode == "none":
        #     pass 
        # else: 
        #     if subMode == "hFlip":
        #         transform = torchio.RandomFlip(axes=(2,))  # Flip along X (left-right)
        #     elif subMode == "vFlip":
        #         transform = torchio.RandomFlip(axes=(1,))  # Flip along Y (top-bottom)
        #     elif subMode == "hvFlip":
        #         transform = torchio.RandomFlip(axes=(1, 2))  # Flip both X and Y (diagonal)

        #     sourceTensor = transform(sourceTensor)
        #     targetTensor = transform(targetTensor)    

        #TODO : Rota
        transform = torchio.RandomFlip(axes=(2,))  # Flip along X (left-right)        
        sourceTensor = transform(sourceTensor)
        targetTensor = transform(targetTensor)    

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
