import torch
   
class BaseVessels(torch.nn.Module):
    def __init__(self, inChan, outChan):
        super(BaseVessels, self).__init__()
        self.inChan = inChan
        self.outChan = outChan

        self.enco3 = BaseVessels.doubleBlock(self.inChan, 64)
        self.enco2 = BaseVessels.doubleBlock(64, 128)
        self.enco1 = BaseVessels.doubleBlock(128, 256)
        self.enco0 = BaseVessels.doubleBlock(256, 512)

        self.bottleneck = BaseVessels.doubleBlock(512, 1024)

        self.upconv0 = BaseVessels.upconvBlock(1024, 512)
        self.upconv1 = BaseVessels.upconvBlock(512, 256)
        self.upconv2 = BaseVessels.upconvBlock(256, 128)
        self.upconv3 = BaseVessels.upconvBlock(128, 64)

        self.deco0 = BaseVessels.doubleBlock(1024, 512)
        self.deco1 = BaseVessels.doubleBlock(512, 256)
        self.deco2 = BaseVessels.doubleBlock(256, 128)
        self.deco3 = BaseVessels.doubleBlock(128, 64)

        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.threshold = torch.nn.Threshold(0.5, 0)
        self.output = torch.nn.Conv2d(64, self.outChan, kernel_size=1)

    def forward(self, patch):
        enco3 = self.enco3(patch)
        enco2 = self.enco2(self.pool(enco3))
        enco1 = self.enco1(self.pool(enco2))
        enco0 = self.enco0(self.pool(enco1))

        bottleneck = self.bottleneck(self.pool(enco0))

        tmp = self.upconv0(bottleneck)
        deco0 = self.deco0(torch.cat((tmp, enco0), dim=1))
        tmp = self.upconv1(deco0)
        deco1 = self.deco1(torch.cat((tmp, enco1), dim=1))
        tmp = self.upconv2(deco1)
        deco2 = self.deco2(torch.cat((tmp, enco2), dim=1))
        tmp = self.upconv3(deco2)
        deco3 = self.deco3(torch.cat((tmp, enco3), dim=1))

        return self.threshold(torch.sigmoid(self.output(deco3)))

    def doubleBlock(inChan, outChan):
        return torch.nn.Sequential(
        torch.nn.Conv2d(inChan, outChan, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(outChan),
        torch.nn.ReLU(),
        torch.nn.Conv2d(outChan, outChan, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(outChan),
        torch.nn.ReLU())

    def upconvBlock(inChan, outChan):
        return torch.nn.Sequential(
        torch.nn.ConvTranspose2d(inChan, outChan, kernel_size=2, stride=2))  

class NewKernel(torch.nn.Module):
    def __init__(self, inChan, outChan):
        super(NewKernel, self).__init__()
        self.inChan = inChan
        self.outChan = outChan

        self.enco3 = NewKernel.doubleBlock(self.inChan, 64)
        self.enco2 = NewKernel.doubleBlock(64, 128)
        self.enco1 = NewKernel.doubleBlock(128, 256)
        # self.enco0 = NewKernel.doubleBlock(256, 512)

        self.bottleneck = NewKernel.doubleBlock(256, 512)

        # self.upconv0 = NewKernel.upconvBlock(1024, 512)
        self.upconv1 = NewKernel.upconvBlock(512, 256)
        self.upconv2 = NewKernel.upconvBlock(256, 128)
        self.upconv3 = NewKernel.upconvBlock(128, 64)

        # self.deco0 = NewKernel.doubleBlock(1024, 512)
        self.deco1 = NewKernel.doubleBlock(512, 256)
        self.deco2 = NewKernel.doubleBlock(256, 128)
        self.deco3 = NewKernel.doubleBlock(128, 64)

        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.threshold = torch.nn.Threshold(0.5, 0)
        self.output = torch.nn.Conv2d(64, self.outChan, kernel_size=1)

    def forward(self, patch):
        enco3 = self.enco3(patch)
        enco2 = self.enco2(self.pool(enco3))
        enco1 = self.enco1(self.pool(enco2))
        # enco0 = self.enco0(self.pool(enco1))

        bottleneck = self.bottleneck(self.pool(enco1))

        # tmp = self.upconv0(bottleneck)
        # deco0 = self.deco0(torch.cat((tmp, enco0), dim=1))
        tmp = self.upconv1(bottleneck)
        deco1 = self.deco1(torch.cat((tmp, enco1), dim=1))
        tmp = self.upconv2(deco1)
        deco2 = self.deco2(torch.cat((tmp, enco2), dim=1))
        tmp = self.upconv3(deco2)
        deco3 = self.deco3(torch.cat((tmp, enco3), dim=1))

        return self.threshold(torch.sigmoid(self.output(deco3)))

    def doubleBlock(inChan, outChan):
        return torch.nn.Sequential(
        torch.nn.Conv2d(inChan, outChan, kernel_size=5, padding=2),
        torch.nn.BatchNorm2d(outChan),
        torch.nn.ReLU(),
        torch.nn.Conv2d(outChan, outChan, kernel_size=7, padding=3),
        torch.nn.BatchNorm2d(outChan),
        torch.nn.ReLU())

    def upconvBlock(inChan, outChan):
        return torch.nn.Sequential(
        torch.nn.ConvTranspose2d(inChan, outChan, kernel_size=2, stride=2))    

class BaseVessels_drop(torch.nn.Module):
    def __init__(self, inChan, outChan):
        super(BaseVessels_drop, self).__init__()
        self.inChan = inChan
        self.outChan = outChan

        self.enco3 = BaseVessels_drop.doubleBlock(self.inChan, 64)
        self.enco2 = BaseVessels_drop.doubleBlock(64, 128)
        self.enco1 = BaseVessels_drop.doubleBlock(128, 256)
        self.enco0 = BaseVessels_drop.doubleBlock(256, 512)

        self.bottleneck = BaseVessels_drop.doubleBlock(512, 1024)

        self.upconv0 = BaseVessels_drop.upconvBlock(1024, 512)
        self.upconv1 = BaseVessels_drop.upconvBlock(512, 256)
        self.upconv2 = BaseVessels_drop.upconvBlock(256, 128)
        self.upconv3 = BaseVessels_drop.upconvBlock(128, 64)

        self.deco0 = BaseVessels_drop.doubleBlock(1024, 512)
        self.deco1 = BaseVessels_drop.doubleBlock(512, 256)
        self.deco2 = BaseVessels_drop.doubleBlock(256, 128)
        self.deco3 = BaseVessels_drop.doubleBlock(128, 64)

        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.threshold = torch.nn.Threshold(0.5, 0)
        self.drop = torch.nn.Dropout(p=0.2)
        self.output = torch.nn.Conv2d(64, self.outChan, kernel_size=1)

        
    def forward(self, patch):
        enco3 = self.enco3(patch)
        enco2 = self.enco2(self.drop(self.pool(enco3)))
        enco1 = self.enco1(self.drop(self.pool(enco2)))
        enco0 = self.enco0(self.drop(self.pool(enco1)))

        bottleneck = self.bottleneck(self.drop(self.pool(enco0)))

        tmp = self.upconv0(bottleneck)
        deco0 = self.deco0(torch.cat((tmp, enco0), dim=1))
        tmp = self.upconv1(deco0)
        deco1 = self.deco1(torch.cat((tmp, enco1), dim=1))
        tmp = self.upconv2(deco1)
        deco2 = self.deco2(torch.cat((tmp, enco2), dim=1))
        tmp = self.upconv3(deco2)
        deco3 = self.deco3(torch.cat((tmp, enco3), dim=1))

        return self.threshold(torch.sigmoid(self.output(deco3)))

    def doubleBlock(inChan, outChan):
        return torch.nn.Sequential(
        torch.nn.Conv2d(inChan, outChan, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(outChan),
        torch.nn.ReLU(),
        torch.nn.Conv2d(outChan, outChan, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(outChan),
        torch.nn.ReLU())

    def upconvBlock(inChan, outChan):
        return torch.nn.Sequential(
        torch.nn.ConvTranspose2d(inChan, outChan, kernel_size=2, stride=2))  

class liver3D(torch.nn.Module):
    def __init__(self, inChan, outChan):
        super(liver3D, self).__init__()
        self.inChan = inChan
        self.outChan = outChan

        self.enco3 = liver3D.doubleBlock(self.inChan, 64)
        self.enco2 = liver3D.doubleBlock(64, 128)
        self.enco1 = liver3D.doubleBlock(128, 256)
        self.enco0 = liver3D.doubleBlock(256, 512)

        self.bottleneck = liver3D.doubleBlock(512, 1024)

        self.upconv0 = liver3D.upconvBlock(1024, 512)
        self.upconv1 = liver3D.upconvBlock(512, 256)
        self.upconv2 = liver3D.upconvBlock(256, 128)
        self.upconv3 = liver3D.upconvBlock(128, 64)

        self.deco0 = liver3D.doubleBlock(1024, 512)
        self.deco1 = liver3D.doubleBlock(512, 256)
        self.deco2 = liver3D.doubleBlock(256, 128)
        self.deco3 = liver3D.doubleBlock(128, 64)

        self.pool = torch.nn.MaxPool3d(kernel_size=2, stride=2)
        self.threshold = torch.nn.Threshold(0.5, 0)
        self.output = torch.nn.Conv3d(64, self.outChan, kernel_size=1)

    def forward(self, patch):
        # print("patch.shape:",patch.shape)
        enco3 = self.enco3(patch)
        # print("enco3.shape:",enco3.shape)
        enco2 = self.enco2(self.pool(enco3))
        # print("enco2.shape:",enco2.shape)
        enco1 = self.enco1(self.pool(enco2))
        # print("enco1.shape:",enco1.shape)
        enco0 = self.enco0(self.pool(enco1))
        # print("enco0.shape:",enco0.shape)

        bottleneck = self.bottleneck(self.pool(enco0))
        # print("bottleneck.shape:",bottleneck.shape)

        tmp = self.upconv0(bottleneck)
        deco0 = self.deco0(torch.cat((tmp, enco0), dim=1))
        tmp = self.upconv1(deco0)
        deco1 = self.deco1(torch.cat((tmp, enco1), dim=1))
        tmp = self.upconv2(deco1)
        deco2 = self.deco2(torch.cat((tmp, enco2), dim=1))
        tmp = self.upconv3(deco2)
        deco3 = self.deco3(torch.cat((tmp, enco3), dim=1))

        return self.threshold(torch.sigmoid(self.output(deco3)))

    def doubleBlock(inChan, outChan):
        return torch.nn.Sequential(
        torch.nn.Conv3d(inChan, outChan, kernel_size=3, padding=1),
        torch.nn.BatchNorm3d(outChan),
        torch.nn.ReLU(),
        torch.nn.Conv3d(outChan, outChan, kernel_size=3, padding=1),
        torch.nn.BatchNorm3d(outChan),
        torch.nn.ReLU())

    def upconvBlock(inChan, outChan):
        return torch.nn.Sequential(
        torch.nn.ConvTranspose3d(inChan, outChan, kernel_size=2, stride=2))  


