import os
import cv2
import tqdm
import numpy
import torch
import PIL

from datasets.imageToTensor import imageToTensor
from datasets.tensorToPatch import tensorToPatch
from datasets.patchToTensor import patchToTensor

from datasets.stackToTensor import stackToTensor
from datasets.tensorToPatch3D import tensorToPatch3D
from datasets.patchToTensor3D import patchToTensor3D


from utils.trainSimple import colorMapping

def buildImage(model, inputArg, outputArg, kernelWidthArg, kernelHeightArg, colorModeArg, daModeArg):

    tensorDataset = imageToTensor(os.path.join(inputArg,"test"), os.path.join(inputArg,"test"), colorModeArg, daModeArg)
    # tensorDataset = stackToTensor(os.path.join(inputArg,"test"), os.path.join(inputArg,"test"), colorModeArg, daModeArg)

    # For the sake of swap
    # tensorDataset = imageToTensor(os.path.join(inputArg,"train"), os.path.join(inputArg,"train"), colorModeArg, daModeArg)

    tensorDataloader = torch.utils.data.DataLoader(dataset=tensorDataset, batch_size=1, shuffle=False)

    # print("len(tensorDataloader)",len(tensorDataloader))
    
    for k,batch in tqdm.tqdm(enumerate(tensorDataloader)):
        strItem = batch[2][0]
        strItem = strItem.split(".")[0]+".png"

        print("k",k)
        newPath = os.path.join(outputArg,str(k+18))
        os.makedirs(newPath)
        
        # patchesDataset = tensorToPatch(batch[0], batch[1], kernelWidthArg, kernelWidthArg, kernelHeightArg, kernelHeightArg)
        patchesDataset = tensorToPatch3D(batch[0], batch[1], kernelWidthArg, kernelWidthArg, kernelHeightArg, kernelHeightArg)

        patchesDataloader = torch.utils.data.DataLoader(dataset=patchesDataset, batch_size=1, shuffle=False)
        unfoldShape = patchesDataset[0][2]

        batchPatch = len(patchesDataset)
        # channelPatch = patchesDataset[0][1].size(patchesDataset[0][1].dim()-3)
        # heightPatch = patchesDataset[0][1].size(patchesDataset[0][1].dim()-2)
        # widthPatch = patchesDataset[0][1].size(patchesDataset[0][1].dim()-1)
        # zeroSubImage = torch.zeros(batchPatch, channelPatch, heightPatch, widthPatch)

        channelPatch = patchesDataset[0][1].size(patchesDataset[0][1].dim()-4)
        depthPatch = patchesDataset[0][1].size(patchesDataset[0][1].dim()-3)
        heightPatch = patchesDataset[0][1].size(patchesDataset[0][1].dim()-2)
        widthPatch = patchesDataset[0][1].size(patchesDataset[0][1].dim()-1)
        zeroSubImage = torch.zeros(batchPatch, channelPatch, depthPatch, heightPatch, widthPatch)

        for i, patch in enumerate(patchesDataloader):
            with torch.no_grad(): 
                model.train() # To ensure the use of batch statistics at prediction time instead of learned statistics
                zeroSubImage[i] = model(patch[0].float().cuda())  #3 Channel output even for binary output ...#ChannelPatch => can go at 1 for binary
                # zeroSubImage[i] = model(patch[0].float().cuda(), "_")  #dvaefl

        # outputDataset = patchToTensor(zeroSubImage, unfoldShape)
        outputDataset = patchToTensor3D(zeroSubImage, unfoldShape)
        outputDataloader = torch.utils.data.DataLoader(dataset=outputDataset, batch_size=1, shuffle=False)

        for tensor in outputDataloader:
            output = tensor[0][0].detach().numpy()
            output = numpy.where(output > 0.5, 255, 0)
            output = output.astype(numpy.uint8)  

            # print(len(tensor))
            # print(output)
            # print(output.shape)
            # print(type(output))

            for i in range (len(output[0])): 
                cv2.imwrite(os.path.join(newPath,str(i+1)+".tif"), output[0][i])
