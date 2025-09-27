import os 
import sys
import csv
import cv2
import tqdm
import numpy
import torch
import pydicom
import torchvision

def metrics(gtPath, predPath, mode):

    assert(len(os.listdir(gtPath)) == len(os.listdir(predPath)))   

    metricsList = [("imgName","TP","FP","TN","FN","DICE","CC")]
    for imgName in tqdm.tqdm(sorted(os.listdir(predPath), key=lambda x: int(x.split(".")[0]))):
        name, ext = imgName.split(".")
        pred = cv2.imread(os.path.join(predPath,imgName), cv2.IMREAD_UNCHANGED)
        gt = cv2.imread(os.path.join(gtPath,imgName), cv2.IMREAD_UNCHANGED)
        # gt = pydicom.dcmread(os.path.join(gtPath,name+".dcm")).pixel_array.astype(numpy.uint8)*255

        predTensor = torchvision.transforms.ToTensor()(pred)
        gtTensor = torchvision.transforms.ToTensor()(gt)

        conf = predTensor/gtTensor

        TP = torch.sum(conf == 1).item()
        FP = torch.sum(conf == float('inf')).item()
        TN = torch.sum(torch.isnan(conf)).item()
        FN = torch.sum(conf == 0).item()
        DICE = 2*TP/(2*TP+FP+FN)
        assert predTensor.shape[1]*predTensor.shape[2] == gtTensor.shape[1]*gtTensor.shape[2] == TP+TN+FP+FN

        NUMLABELS, _, _, _ = cv2.connectedComponentsWithStats(pred.astype(numpy.uint8), connectivity=8)

        metricsList += [(imgName,TP,FP,TN,FN,round(DICE,3),NUMLABELS-1)]

    if mode == "details":
        return metricsList
    elif mode == "average":
        meanTP = sum([e[1] for e in metricsList[1:]])/len(metricsList[1:])
        meanFP = sum([e[2] for e in metricsList[1:]])/len(metricsList[1:])
        meanTN = sum([e[3] for e in metricsList[1:]])/len(metricsList[1:])
        meanFN = sum([e[4] for e in metricsList[1:]])/len(metricsList[1:])
        meanDice = sum([e[5] for e in metricsList[1:]])/len(metricsList[1:])
        meanCC = sum([e[6] for e in metricsList[1:]])/len(metricsList[1:])
        return [("TP","FP","TN","FN","DICE","CC"),\
                (round(meanTP),round(meanFP),round(meanTN),round(meanFN),round(meanDice,3),round(meanCC,1))]
    else:
        print("Invalid choice. Please use 'details' or 'average' but not '",mode,"'.")
        return

def main(argv):
    gtPath, predPath,savePath, mode, display, fileName, DB = argv[1:11]
    savePath = os.path.join(savePath,fileName+"_compare_"+mode+".csv")
    assert(mode=="details" or mode=="average")
    assert(display=="TXT" or display=="CSV")
    print(gtPath,"==>",predPath)
    
    assert os.path.isdir(gtPath)
    assert os.path.isdir(predPath)

    metricsList = metrics(gtPath, predPath, mode)

    if display == "TXT":
        print((DB))
        for elem in metricsList:
            print(elem)
        print()
    
    if display == "CSV":
        with open(savePath, 'a+', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow((DB,))
            for elem in metricsList:
                csvwriter.writerow(elem)
            csvwriter.writerow([])

if __name__ == '__main__':
    main(sys.argv)
