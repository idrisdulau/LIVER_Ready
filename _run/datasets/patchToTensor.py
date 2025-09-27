import torch

class patchToTensor(torch.utils.data.Dataset):
    
    def __init__(self, patches, unfoldShape):
        super(patchToTensor, self).__init__()
        self.patches = patches
        self.unfoldShape = unfoldShape
        self.length = 1

    def __getitem__(self, index):      
        sourcePatches = self.patches
        sourceMergedPatches = sourcePatches.view(self.unfoldShape)
        batch = self.unfoldShape[0]
        outputH = self.unfoldShape[1] * self.unfoldShape[3]
        outputW = self.unfoldShape[2] * self.unfoldShape[4]
        sourceMergedPatches = sourceMergedPatches.permute(0, 1, 3, 2, 4).contiguous()
        source = sourceMergedPatches.view(batch, outputH, outputW)
        return [source]

    def __len__(self):
        return self.length
