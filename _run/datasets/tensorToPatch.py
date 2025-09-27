import torch

class tensorToPatch(torch.utils.data.Dataset):
    
    def __init__(self, sourceTensor, targetTensor, kernelWidth, strideWidth, kernelHeight, strideHeight):
        super(tensorToPatch, self).__init__()
        self.sourceTensor = sourceTensor[0]
        self.targetTensor = targetTensor[0]
        self.kernelWidth,self.strideWidth = int(kernelWidth),int(strideWidth)
        self.kernelHeight,self.strideHeight = int(kernelHeight),int(strideHeight)

        self.imageWidth = self.sourceTensor.shape[self.sourceTensor[0].dim()]

        assert self.imageWidth%self.kernelWidth == 0, "Patches width doesn't match image width"
        self.numberOfPatchesInWidth = (self.imageWidth-self.kernelWidth)//self.strideWidth + 1

        self.imageHeight = self.sourceTensor.shape[self.sourceTensor[0].dim()-1]
        assert self.imageHeight%self.kernelHeight == 0, "Patches height doesn't match image height"
        self.numberOfPatchesInHeight = (self.imageHeight-self.kernelHeight)//self.strideHeight + 1

        self.length = self.numberOfPatchesInHeight*self.numberOfPatchesInWidth

    def __getitem__(self, index):
        print("sourceTensor in patch ",self.sourceTensor.shape)
        
        sourcePatches = self.sourceTensor.unfold(1, self.kernelHeight, self.strideHeight).unfold(2, self.kernelWidth, self.strideWidth)

        print("sourcePatches 1:",sourcePatches.shape)
        
        sourcePatches = sourcePatches.contiguous().view(-1, sourcePatches.size(0), self.kernelHeight, self.kernelWidth)
        
        print("sourcePatches 2:",sourcePatches.shape)
        # exit()

        targetPatches = self.targetTensor.unfold(1, self.kernelHeight, self.strideHeight).unfold(2, self.kernelWidth, self.strideWidth)
        unfoldShape = targetPatches.size()
        targetPatches = targetPatches.contiguous().view(-1, targetPatches.size(0), self.kernelHeight, self.kernelWidth)

        return [sourcePatches[index], targetPatches[index], unfoldShape]

    def __len__(self):
        return self.length
