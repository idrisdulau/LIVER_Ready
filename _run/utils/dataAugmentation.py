import cv2
import numpy
import torch
import random
import torchvision
import albumentations

#StrPath >>> Image
def colorTransform(sourceFile, mode):
    assert(mode == "Grayscale" or mode == "Green" or mode == "RGB" or mode == "AVO")
    if mode == "Grayscale":
        return cv2.imread(sourceFile, cv2.IMREAD_GRAYSCALE)
    if mode == "Green":
        return cv2.split(cv2.imread(sourceFile, cv2.IMREAD_UNCHANGED))[1]
    if mode == "RGB" or mode == "AVO":
        return cv2.imread(sourceFile, cv2.IMREAD_UNCHANGED)

#Image/Tensor >>> Tensor
def geometryTransform(source, target, mode):
    assert(isImage(source) or isTensor(source))
    assert(isImage(target) or isTensor(target))
    if isImage(source):
        sourceTensor = toTensor(source)
    else: 
        sourceTensor = source
    if isImage(target):
        targetTensor = toTensor(target)
    else: 
        targetTensor = target
    assert(mode == "Flip" or mode == "Rota")
    if mode == "Flip":
        subMode = random.choice(["hFlip","vFlip","dFlip"])
        if subMode == "hFlip":
            sourceTensor = torchvision.transforms.functional.hflip(sourceTensor)
            targetTensor = torchvision.transforms.functional.hflip(targetTensor)
        elif subMode == "vFlip":
            sourceTensor = torchvision.transforms.functional.vflip(sourceTensor)
            targetTensor = torchvision.transforms.functional.vflip(targetTensor)
        else:
            sourceTensor = torchvision.transforms.functional.hflip(sourceTensor)
            sourceTensor = torchvision.transforms.functional.vflip(sourceTensor)
            targetTensor = torchvision.transforms.functional.hflip(targetTensor)
            targetTensor = torchvision.transforms.functional.vflip(targetTensor)
    else:
        angle = random.randrange(0, 360, 1)
        sourceTensor = torchvision.transforms.RandomRotation((angle,angle))(sourceTensor)
        targetTensor = torchvision.transforms.RandomRotation((angle,angle))(targetTensor)
    return sourceTensor, targetTensor      

#Image/Tensor >>> Tensor
def distorsionTransform(source, target, mode):
    assert(isImage(source) or isTensor(source))
    assert(isImage(target) or isTensor(target))
    if isImage(source):
        sourceTensor = toTensor(source)
    else: 
        sourceTensor = source
    if isImage(target):
        targetTensor = toTensor(target)
    else: 
        targetTensor = target
    assert(mode == "Shear" or mode == "Elastic")
    if mode == "Shear":
        alpha = random.randrange(10, 30, 1)
        sourceTensor = torchvision.transforms.RandomAffine(degrees=0, shear=(alpha,alpha))(sourceTensor)
        targetTensor = torchvision.transforms.RandomAffine(degrees=0, shear=(alpha,alpha))(targetTensor)
    else:
        sourceTensor = torchvision.transforms.ElasticTransform(alpha=20.0, sigma=3.0)(sourceTensor)
        targetTensor = torchvision.transforms.ElasticTransform(alpha=20.0, sigma=3.0)(targetTensor)
        targetTensor = torch.where(targetTensor==0, torch.zeros(targetTensor.shape), torch.ones(targetTensor.shape))
    return sourceTensor, targetTensor      

#Image/Tensor >>> Tensor (target just travelling)
def intensityTransform(source, target, mode):
    assert(isImage(source) or isTensor(source))
    assert(isImage(target) or isTensor(target))
    if isImage(target):
        targetImage = target
    else:
        targetImage = toImage(target.permute(1,2,0))
    if isImage(source):
        sourceImage = source
    else:
        sourceImage = toImage(source.permute(1,2,0))
    assert(mode == "Brightness" or mode == "Contrast")
    if mode == "Brightness":
        sourceImage = albumentations.RandomGamma(always_apply=False, p=1.0, gamma_limit=(25, 50), eps=1e-7)(image=sourceImage)['image']
    else:
        sourceImage = albumentations.CLAHE(always_apply=False, p=1.0, clip_limit=(10, 30), tile_grid_size=(8, 8))(image=sourceImage)['image']
    return toTensor(sourceImage), toTensor(targetImage)

#Image/Tensor >>> Tensor (target just travelling)
def channelTransform(source, target, mode):
    assert(isImage(source) or isTensor(source))
    assert(isImage(target) or isTensor(target))
    if isImage(target):
        targetImage = target
    else:
        targetImage = toImage(target.permute(1,2,0))
    if isImage(source):
        sourceImage = source
    else:
        sourceImage = toImage(source.permute(1,2,0))
    assert(mode == "Shuffle" or mode == "Hue")
    if mode == "Shuffle":
        sourceImage = albumentations.ChannelShuffle(always_apply=False, p=1.0)(image=sourceImage)['image']
    else:
        sourceImage = albumentations.ColorJitter(always_apply=False, p=1.0, brightness=(1.0, 1.0), contrast=(1.0, 1.0), saturation=(1.0, 1.0), hue=(-0.5, 0.5))(image=sourceImage)['image']
    return toTensor(sourceImage), toTensor(targetImage)

#Image/Tensor >>> Tensor (target just travelling)
def overlayTransform(source, target, mode):
    assert(isImage(source) or isTensor(source))
    assert(isImage(target) or isTensor(target))
    if isImage(target):
        targetImage = target
    else:
        targetImage = toImage(target.permute(1,2,0))
    if isImage(source):
        sourceImage = source
    else:
        sourceImage = toImage(source.permute(1,2,0))
    assert(mode == "Cutbig" or mode == "Cutpix")
    if mode == "Cutbig":
        sourceImage = albumentations.CoarseDropout(always_apply=False, p=1.0, max_holes=100, max_height=10, max_width=10, min_holes=100, min_height=5, min_width=5, fill_value=(0, 0, 0), mask_fill_value=None)(image=sourceImage)['image']
    else:
        sourceImage = albumentations.PixelDropout(always_apply=False, p=1.0, dropout_prob=0.05, per_channel=0, drop_value=(0, 0, 0), mask_drop_value=None)(image=sourceImage)['image']
    return toTensor(sourceImage), toTensor(targetImage)

def callTransforms(sourceImage, targetImage, cumModeList):
    assert len(cumModeList) > 0
    assert set(cumModeList).issubset(["Flip","Rota","Shear","Elastic","Brightness","Contrast","Shuffle","Hue","Cutbig","Cutpix","noDA"])
    # print("cumModeList",cumModeList)
    for mode in cumModeList:
        if mode == "Flip" or mode == "Rota":
            sourceImage,targetImage = geometryTransform(sourceImage, targetImage, mode)
        if mode == "Shear" or mode == "Elastic":
            sourceImage,targetImage = distorsionTransform(sourceImage, targetImage, mode)
        if mode == "Brightness" or mode == "Contrast":
            sourceImage,targetImage = intensityTransform(sourceImage, targetImage, mode)
        if mode == "Shuffle" or mode == "Hue":
            sourceImage,targetImage = channelTransform(sourceImage, targetImage, mode) 
        if mode == "Cutbig" or mode == "Cutpix":
            sourceImage,targetImage = overlayTransform(sourceImage, targetImage, mode)     
        if mode == "noDA":
            if isImage(sourceImage):
                sourceImage = toTensor(sourceImage) 
            if isImage(targetImage):
                targetImage = toTensor(targetImage)  
    return sourceImage,targetImage

def toImage(tensor):
    image = tensor.numpy().squeeze()
    image = image * (255/image.max())
    image = image.astype(numpy.uint8)
    assert(isImage(image))
    return image

def toTensor(image):
    tensor = torchvision.transforms.ToTensor()(image)
    assert(isTensor(tensor))
    return tensor

def isImage(source):
    return isinstance(source, numpy.ndarray) 

def isTensor(source):
    return isinstance(source, torch.Tensor)
