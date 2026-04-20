from .ResNet import *
from .ResNets import *
from .Swin import swin_tiny, swin_small, swin_base
from .VGG import *
from .VGG_LTH import *

model_dict = {
    "resnet18": resnet18,
    "resnet50": resnet50,
    "resnet20s": resnet20s,
    "resnet44s": resnet44s,
    "resnet56s": resnet56s,
    "swin_t": swin_tiny,
    "swin_s": swin_small,
    "swin_b": swin_base,
    "vgg16_bn": vgg16_bn,
    "vgg16_bn_lth": vgg16_bn_lth,
}
