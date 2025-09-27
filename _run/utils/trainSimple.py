import os
import cv2
import csv
import tqdm
import numpy
import torch
import matplotlib.pyplot

from losses.diceLoss import DiceLoss
from losses.clDiceLoss import softAlphaCenterlineDice
from losses.acLoss import active_contour_loss
from losses.aceLoss import ACELoss
from losses.elasticLoss import EnergyLoss
from losses.gcLoss import GC_2D_Original, GC_2D
from losses.topoLoss2 import getTopoLoss

from losses.dccLoss import BccLoss, AlphaDBccLoss
from losses.avoLoss import avoLoss

from datasets.imageToTensor import imageToTensor
from datasets.tensorToPatch import tensorToPatch

from datasets.stackToTensor import stackToTensor
from datasets.tensorToPatch3D import tensorToPatch3D

def trainModel(input, target, model, loss, save, saveName, kernelWidth, strideWidth, kernelHeight, strideHeight, colorMode, daMode, epochs, device):

    model = model.to(device)
    trainMeanLoss,valMeanLoss = [],[]

    trainTensorDataset = imageToTensor(os.path.join(input,"train"), os.path.join(target,"train"), colorMode=colorMode, daMode=daMode)
    # trainTensorDataset = stackToTensor(os.path.join(input,"train"), os.path.join(target,"train"), colorMode=colorMode, daMode=daMode)
    # For the sake of swap
    # trainTensorDataset = imageToTensor(os.path.join(input,"test"), os.path.join(target,"test"), colorMode=colorMode, daMode=daMode)

    datasetReplicationFactor = 4 #4
    trainTensorDatasetList = [trainTensorDataset for _ in range(datasetReplicationFactor)]
    trainTensorDataset = torch.utils.data.ConcatDataset(trainTensorDatasetList)
    print("concatLen",len(trainTensorDataset))    
    trainTensorDataloader = torch.utils.data.DataLoader(dataset=trainTensorDataset, batch_size=1, shuffle=True)

    valTensorDataset = imageToTensor(os.path.join(input,"test"), os.path.join(target,"test"), colorMode=colorMode, daMode=["noDA"])
    # valTensorDataset = stackToTensor(os.path.join(input,"test"), os.path.join(target,"test"), colorMode=colorMode, daMode=["noDA"])

    # For the sake of swap
    # valTensorDataset = imageToTensor(os.path.join(input,"train"), os.path.join(target,"train"), colorMode=colorMode, daMode=["noDA"])

    valTensorDataloader = torch.utils.data.DataLoader(dataset=valTensorDataset, batch_size=1, shuffle=False)

    for epoch in tqdm.tqdm(range(1, epochs+1)):
        run("Train", model, loss, device, epoch, trainTensorDataloader, trainMeanLoss, kernelWidth, strideWidth, kernelHeight, strideHeight, save, saveName)
        valCheckpointsRange = 5
        if epoch % valCheckpointsRange == 0:
            run("Val", model, loss, device, epoch, valTensorDataloader, valMeanLoss, kernelWidth, strideWidth, kernelHeight, strideHeight, save, saveName)   
            finalPlot(trainMeanLoss, valMeanLoss, valCheckpointsRange, save, saveName)
        # if epoch % 50 == 0:
        #     torch.save(model.state_dict(), os.path.join(save, saveName+"_"+str(epoch)+".pt"))   
    return model

def run(mode, model, loss, device, epoch, tensorDataloader, meanLossList, kernelWidth, strideWidth, kernelHeight, strideHeight, save, saveName):
    assert mode=="Train" or mode=="Val"
    model.train()
    metricsDict = {}
    epochLoss = 0.0
    
    lossFunction = getLoss(loss, epoch)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)

    if mode == "Train":
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.InstanceNorm2d) or isinstance(m, torch.nn.BatchNorm3d):
                m.track_running_stats = True 
    else:
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.InstanceNorm2d) or isinstance(m, torch.nn.BatchNorm3d):
                m.track_running_stats = False # Avoid learning statistics of the validation set
                
    for batch in tensorDataloader: 

        # print("batch[0].shape:",batch[0].shape) #Source
        # print("batch[1].shape:",batch[1].shape) #Target
        # exit()
        # patchDataset = tensorToPatch(batch[0], batch[1], kernelWidth, strideWidth, kernelHeight, strideHeight)
        patchDataset = tensorToPatch3D(batch[0], batch[1], kernelWidth, strideWidth, kernelHeight, strideHeight)
        patchDataloader = torch.utils.data.DataLoader(dataset=patchDataset, batch_size=1)

        for patch in patchDataloader:

            # print("batch[2][0]",batch[2][0])
            # exit()

            optimizer.zero_grad()
            xTrue = patch[0].to(device)
            yTrue = patch[1].to(device) 

            if mode == "Train":
                yPred = model(xTrue) # yPred = model(xTrue, "training") #dvaefl
                loss = lossFunction(yPred, yTrue)
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    # model.eval() #Shows the effect of using learned runnings statistics to predict segmentations in few-shot configuration : disappointing
                    yPred = model(xTrue) # yPred = model(xTrue, "_") #dvaefl
                loss = lossFunction(yPred, yTrue)
                metricsDict[batch[2][0]] = computesMetrics(batch[2][0], yPred, yTrue)

            # print("loss.item()",loss.item())
            epochLoss += loss.item()

    epochLoss = epochLoss / (len(patchDataloader)*len(tensorDataloader))  
    meanLossList.append(epochLoss)
    print("%s loss: %6.4f" % (mode, epochLoss))
    if mode == "Val":
        toCSV(save, saveName, epoch, metricsDict)

def getLoss(loss, epoch):
    # lossFunction = active_contour_loss
    # lossFunction = ACELoss
    # lossFunction = EnergyLoss(alpha=0.35)
    # lossFunction = GC_2D_Original(3,0.5)
    # lossFunction = GC_2D(3)
    # lossFunction = getTopoLoss
    
    if loss == "avo":
        lossFunction = avoLoss()
        # lossFunction = torch.nn.CrossEntropyLoss()
    if loss == "dice":
        lossFunction = DiceLoss()
    elif loss == "clDice":
        lossFunction = softAlphaCenterlineDice() #Need patches or more than 8Vram
    elif loss == "alphaDBCC25":
        lossFunction = AlphaDBccLoss(0.25)
    elif loss == "alphaDBCC50":
        lossFunction = AlphaDBccLoss(0.50) 
    elif loss == "alphaDBCC75":
        lossFunction = AlphaDBccLoss(0.75) 
    elif loss == "Bcc":
        lossFunction = BccLoss()
    elif loss == "EvoluteDBCC":
        alpha = max(0, epoch/50-0.2)
        print("alpha:",alpha)
        lossFunction = AlphaDBccLoss(alpha)  

    return lossFunction

def colorMapping(y):

    output = y.detach().cpu().numpy()
    classMax = numpy.argmax(output, axis=1)

    colorMap = {
        0: [0, 0, 0],    # Black
        1: [255, 0, 0],  # Pure Red
        2: [0, 255, 0],  # Pure Green
        3: [0, 0, 255],  # Pure Blue
    }
    
    rgbImg = numpy.zeros((3, 1024, 1024), dtype=numpy.uint8)
    for index, color in colorMap.items():
        mask = (classMax == index)
        for i in range(3):
            rgbImg[i] = numpy.where(mask, color[i], rgbImg[i])

    return numpy.transpose(rgbImg, (1, 2, 0))

def displayRGBfromOneHot(yPred, yTrue):

    classTrue = colorMapping(yTrue)
    classPred = colorMapping(yPred)
    combined = cv2.hconcat([classTrue, classPred])
    cv2.imshow("True < ==== > Pred", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def computesMetrics(imageName, yPred, yTrue): 
    intyPred = torch.where(yPred > 0.5, torch.ones(yPred.size()).cuda(), torch.zeros(yPred.size()).cuda())
    conf = intyPred/yTrue
    TP = torch.sum(conf == 1).item()
    FP = torch.sum(conf == float('inf')).item()
    TN = torch.sum(torch.isnan(conf)).item()
    FN = torch.sum(conf == 0).item()
    
    if TP+FP == 0:
        PRE = -1
    else:
        PRE = round(TP/(TP+FP),3)

    if TP+FN == 0:
        REC = -1
    else:
        REC = round(TP/(TP+FN),3)
    
    if TP+FP+FN == 0:
        DICE = -1
    else:
        DICE = round(2*TP/(2*TP+FP+FN),3)

    print(TP+TN+FP+FN, "|", imageName, "| TP", TP, "| TN", TN, "| FP", FP, "| FN", FN, "| PRE", PRE, "| REC", REC, "| DICE", DICE)
    return {"TP":TP, "TN":TN, "FP":FP, "FN":FN, "PRE":PRE, "REC":REC, "DICE":DICE}

def toCSV(save, saveName, epoch, metricsList):
    diceList = []
    metricsDict = metricsList
    csvFilePath = os.path.join(str(save),str(saveName)+"_metrics.csv")
    with open(csvFilePath, mode='a+', newline='') as file:
        fieldnames = ['epoch:'+str(epoch),'TP','TN','FP','FN','PRE','REC','DICE']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for image, metric in metricsDict.items():
            writer.writerow({
                'epoch:'+str(epoch): image,
                'TP': metric['TP'],
                'TN': metric['TN'],
                'FP': metric['FP'],
                'FN': metric['FN'],
                'PRE': metric['PRE'],
                'REC': metric['REC'],
                'DICE': metric['DICE'],
            })     
            diceList.append(metric['DICE'])
        writer.writerow({})
        meanDice = round(sum(diceList) / len(diceList), 3)
        print("MEAN DICE:", meanDice)
        print()

def finalPlot(trainMeanLoss, valMeanLoss, valCheckpointsRange, save, saveName):
    epochList = range(1, (len(trainMeanLoss)+1))
    fig, ax = matplotlib.pyplot.subplots()
    ax.plot(epochList, trainMeanLoss, label='Train', color='blue')
    ax.plot(epochList[valCheckpointsRange-1::valCheckpointsRange], valMeanLoss, label='Test', color = 'red')
    matplotlib.pyplot.xlabel('epochs')
    matplotlib.pyplot.ylabel('losses')
    matplotlib.pyplot.title('Losses over epochs')
    ax.legend(loc='upper right')
    matplotlib.pyplot.savefig(save+"/"+saveName+"_Plot.png")
