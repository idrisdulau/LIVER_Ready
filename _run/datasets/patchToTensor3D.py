import torch

class patchToTensor3D(torch.utils.data.Dataset):
    
    def __init__(self, patches, unfoldShape):
        super(patchToTensor3D, self).__init__()
        self.patches = patches
        self.unfoldShape = unfoldShape

        # print("self.unfoldShape",self.unfoldShape)
        
        self.length = 1

    def __getitem__(self, index):      
        sourcePatches = self.patches

        # print("sourcePatches.shape",sourcePatches.shape)

        sourceMergedPatches = sourcePatches.view(self.unfoldShape)

        # print("sourceMergedPatches.shape",sourceMergedPatches.shape)

        batch = self.unfoldShape[0]       
        outputC = self.unfoldShape[1] * self.unfoldShape[4]
        outputH = self.unfoldShape[2] * self.unfoldShape[5]
        outputW = self.unfoldShape[3] * self.unfoldShape[6]
        sourceMergedPatches = sourceMergedPatches.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        source = sourceMergedPatches.view(batch, outputC, outputH, outputW)

        # print("source.shape",source.shape)

        return [source]

    def __len__(self):
        return self.length
