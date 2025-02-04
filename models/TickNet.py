import re
import types

import torch.nn
import torch.nn.init

from .common import conv1x1_block, Classifier,conv3x3_dw_blockAll,conv3x3_block
from .SE_Attention import *
#proposal ->1/2pw->dw->channelpw(out)(for modified ticknet)
class FR_PDP_block(torch.nn.Module):
    """
    DSC_modified block internally used in TickNet.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 use_bottleneck=False):
        super().__init__()
        self.use_bottleneck = use_bottleneck
        self.Pw1 = conv1x1_block(in_channels=in_channels,
                                out_channels=in_channels,                                
                                use_bn=False,
                                activation=None)
        self.Dw = conv3x3_dw_blockAll(channels=in_channels, stride=stride)         
        self.Pw2 = conv1x1_block(in_channels=in_channels,
                                             out_channels=out_channels,                                             
                                             groups=1)
        self.PwR = conv1x1_block(in_channels=in_channels,
                                out_channels=out_channels,
                                stride=stride)
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.SE = SE(out_channels, 16)
        if use_bottleneck:
            # bottleneck_channels = 64  
            self.bottleneck = Bottleneck(in_channels=512, 
                                        bottleneck_channels=64,  
                                        out_channels=128) 
    def forward(self, x):
        residual = x
        x = self.Pw1(x)        
        x = self.Dw(x)        
        x = self.Pw2(x)
        x = self.SE(x)
        if self.stride == 1 and self.in_channels == self.out_channels:
            #print('vao1: ' + str(residual.size()))
            #print('vao2: ' + str(x.size()))
            x = x + residual
        else:
            if self.use_bottleneck and self.in_channels > self.out_channels:
                residual = self.bottleneck(residual)
            else:
                residual = self.PwR(residual)
            x = x + residual
        return x
        
class Bottleneck(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.relu(out)
        out = self.bn2(self.conv2(out))
        return out
        
class TickNet(torch.nn.Module):
    """
    Class for constructing TickNet.
    
    """
    def __init__(self,
                 num_classes,
                 init_conv_channels,
                 init_conv_stride,
                 channels,
                 strides,
                 in_channels=3,
                 in_size=(224, 224),
                 use_data_batchnorm=True):
        super().__init__()
        self.use_data_batchnorm = use_data_batchnorm
        self.in_size = in_size

        self.backbone = torch.nn.Sequential()

        # data batchnorm
        if self.use_data_batchnorm:
            self.backbone.add_module("data_bn", torch.nn.BatchNorm2d(num_features=in_channels))

        # init conv
        self.backbone.add_module("init_conv", conv3x3_block(in_channels=in_channels, out_channels=init_conv_channels, stride=init_conv_stride))

        # stages
        in_channels = init_conv_channels
        for stage_id, stage_channels in enumerate(channels):
            stage = torch.nn.Sequential()
            for unit_id, unit_channels in enumerate(stage_channels):
                stride = strides[stage_id] if unit_id == 0 else 1                
                stage.add_module("unit{}".format(unit_id + 1), FR_PDP_block(in_channels=in_channels, out_channels=unit_channels, stride=stride, use_bottleneck=use_bottleneck))
                in_channels = unit_channels
            self.backbone.add_module("stage{}".format(stage_id + 1), stage)
        self.final_conv_channels = 1024        
        self.backbone.add_module("final_conv", conv1x1_block(in_channels=in_channels, out_channels=self.final_conv_channels, activation="relu"))
        self.backbone.add_module("global_pool", torch.nn.AdaptiveAvgPool2d(output_size=1))
        in_channels = self.final_conv_channels
        # classifier
        self.classifier = Classifier(in_channels=in_channels, num_classes=num_classes)

        self.init_params()

    def init_params(self):
        # backbone
        for name, module in self.backbone.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)

        # classifier
        self.classifier.init_params()

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

###
#%% model definitions
###

def build_TickNet(num_classes, typesize='large', cifar=False):
    init_conv_channels = 32
    if typesize=='basic':
        channels = [[128],[64],[128],[256],[512]]#TickBlock(basic)
    if typesize=='small':
        #channels = [[128],[64,128,256,512],[128],[64,128,256],[512]]#2TickBlock00(new)
        #test flops
        channels = [[128],[64,128],[256,512,128],[64,128,256],[512]]#FLOPs se giam dang ke    
    if typesize=='large_new':
        #channels = [[128],[64,128,256,512],[128,64,128,256,512],[128,64,128,256],[512]]#large_new (small + one more block)
        #test flops
        channels = [[128],[64,128],[256,512,128,64,128,256],[512,128,64,128,256],[512]]#large_new (small + one more block)    
        #channels = [[128],[64,128],[256,512,128,64,128],[256,512,128,64,128,256],[512]]#large_new (small + one more block)    
    if typesize=='large_new1':
        #channels = [[128],[64,128,256,512],[128,64,128,256,512],[128,64,128,256],[512]]#large_new (small + one more block)
        #test flops
        channels = [[128],[64,128],[256,512,128],[64,128,256,512,128],[64,128,256,512]]#large_new (small + one more block)    
        #channels = [[128],[64,128],[256,512,128,64,128],[256,512,128,64,128,256],[512]]#large_new (small + one more block)    
    if typesize=='large_new2':
        #channels = [[128],[64,128,256,512],[128,64,128,256,512],[128,64,128,256],[512]]#large_new (small + one more block)
        #test flops
        channels = [[128],[64,128],[256,512,128],[64,128,256,512,128,64,128],[256,512]]#large_new (small + one more block)    
        #channels = [[128],[64,128],[256,512,128,64,128],[256,512,128,64,128,256],[512]]#large_new (small + one more block)   
    if typesize=='large_new3':
        #channels = [[128],[64,128,256,512],[128,64,128,256,512],[128,64,128,256],[512]]#large_new (small + one more block)
        #test flops
        channels = [[128], [64,128,256],[512,128,64,128,256],[512,128,64,128,256],[512]]#large_new (small + one more block)    
        #channels = [[128],[64,128],[256,512,128,64,128],[256,512,128,64,128,256],[512]]#large_new (small + one more block)   
    if typesize=='testsmall':
        channels = [[128],[64],[128,256,512,128],[64,128,256],[512]]#FLOPs se giam dang ke    
    if typesize=='testlarge_new':
        #channels = [[128],[64,128,256,512],[128,64,128,256,512],[128,64,128,256],[512]]#large_new (small + one more block)
        #test flops
        channels = [[128],[64],[128,256,512,128,64,128,256],[512,128,64,128,256],[512]]#large_new (small + one more block)    
    if typesize=='large_new4':#strides = [2, 2, 2, 1, 2] channels of large_new
        #channels = [[128],[64,128,256,512],[128,64,128,256,512],[128,64,128,256],[512]]#large_new (small + one more block)
        #test flops
        channels = [[128],[64,128],[256,512,128,64,128,256],[512,128,64,128,256],[512]]#large_new (small + one more block)    
        #channels = [[128],[64,128],[256,512,128,64,128],[256,512,128,64,128,256],[512]]#large_new (small + one more block)    
    if typesize=='large_new5':#strides = [2, 2, 2, 2, 1] channels of large_new 
        #channels = [[128],[64,128,256,512],[128,64,128,256,512],[128,64,128,256],[512]]#large_new (small + one more block)
        #test flops
        channels = [[128],[64,128],[256,512,128,64,128,256],[512,128,64,128,256],[512]]#large_new (small + one more block)    
        #channels = [[128],[64,128],[256,512,128,64,128],[256,512,128,64,128,256],[512]]#large_new (small + one more block)    
    if typesize=='large_new6':#strides = [2, 2, 1, 2, 2] channels of large_new 
        #channels = [[128],[64,128,256,512],[128,64,128,256,512],[128,64,128,256],[512]]#large_new (small + one more block)
        #test flops
        channels = [[128],[64,128],[256,512,128,64,128,256],[512,128,64,128,256],[512]]#large_new (small + one more block)    
        #channels = [[128],[64,128],[256,512,128,64,128],[256,512,128,64,128,256],[512]]#large_new (small + one more block)    
    if typesize=='large_new7':#strides = [2, 1, 2, 2, 2] channels of large_new 
        #channels = [[128],[64,128,256,512],[128,64,128,256,512],[128,64,128,256],[512]]#large_new (small + one more block)
        #test flops
        channels = [[128],[64,128],[256,512,128,64,128,256],[512,128,64,128,256],[512]]#large_new (small + one more block)    
        #channels = [[128],[64,128],[256,512,128,64,128],[256,512,128,64,128,256],[512]]#large_new (small + one more block)    
    if typesize=='small7':#strides = [2, 1, 2, 2, 2]
        #channels = [[128],[64,128,256,512],[128],[64,128,256],[512]]#2TickBlock00(new)
        #test flops
        channels = [[128],[64,128],[256,512,128],[64,128,256],[512]]#FLOPs se giam dang ke    
    if cifar:
        in_size = (32, 32)
        init_conv_stride = 1
        strides = [1, 1, 2, 2, 2]        
    else:
        in_size = (224, 224)
        init_conv_stride = 2
        #strides = [1, 2, 2, 2, 2]
        #strides = [2, 2, 2, 1, 2]
        if typesize=='small7' or typesize=='large_new7': 
            print('vao model 7')
            strides = [2, 1, 2, 2, 2]
        else:
            strides = [1, 2, 2, 2, 2]

    return  TickNet(num_classes=num_classes,
                       init_conv_channels=init_conv_channels,
                       init_conv_stride=init_conv_stride,
                       channels=channels,
                       strides=strides,
                       in_size=in_size)
