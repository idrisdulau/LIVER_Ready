import torch
import torchvision

class Resnet50_Last(torch.nn.Module):
    def __init__(self, inChan, outChan):
        super(Resnet50_Last, self).__init__()

        self.inChan = inChan
        self.outChan = outChan

        resnetWeights = torchvision.models.segmentation.FCN_ResNet50_Weights.DEFAULT
        self.model = torchvision.models.segmentation.fcn_resnet50(weights=resnetWeights)

        # Train Last
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.backbone.conv1 = torch.nn.Conv2d(self.inChan, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    
        self.model.classifier[4] = torch.nn.Sequential(
            torch.nn.Conv2d(512, self.outChan, kernel_size=1),
            torch.nn.Sigmoid(),
            torch.nn.Threshold(0.5, 0)
        )

    def forward(self, x):
        x = self.model(x)
        return x["out"]
    