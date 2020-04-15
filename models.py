# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 16:05:38 2020

@author: Ulrich
"""

import torch
import torch.nn as nn
import numpy as np

from collections import defaultdict

from parse_config import parse_model_config


def generate_modules(layers):
    """
    Generate a module list from the layers with configurations.
    
    Input:
        layers - A list of layers with a dictionary for configurations.
    Return:
        hyperparams
        module_list
    """
    hyperparams = layers.pop(0)
    output_filters = [int(hyperparams['channels'])]
    
    module_list = nn.ModuleList()
    for i, layer in enumerate(layers):
        modules = nn.Sequential()
        
        if layers['type'] == 'convolutional':
            bn = int(layers['batch_normalize'])
            filters = int(layers['filters'])
            kernel_size = int(layers['size'])
            pad = (kernel_size-1) // 2 if int(layers['pad']) else 0
            modules.add_module(
                "conv_%d" % i,
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(layers['stride']),
                    padding=pad,
                    bias=not bn,
                ),
            )
            if bn:
                modules.add_module("batch_norm_%d" % i, nn.BatchNorm2d(filters))
            if layers['activation'] == "leaky":
                modules.add_module("leaky_%d" % i, nn.LeakyReLU(0.1))
            
        elif layers['type'] == 'maxpool':
            kernel_size = int(layers['size'])
            stride = int(layers['stride'])
            if kernel_size == 2 and stride == 1:
                padding = nn.ZeroPad2d((0, 1, 0, 1))
                modules.add_module("_debug_padding_%d" % i, padding)
            maxpool = nn.MaxPool2d(
                kernel_size=kernel_size,
                stride=stride,
                padding=int((kernel_size - 1) // 2),
            )
            modules.add_module("maxpool_%d" % i, maxpool)
            
        elif layers['type'] == 'upsample':
            stride = int(layers['stride'])
            upsample = nn.Upsample(scale_factor=stride, mode='nearest')
            modules.add_module("upsample_%d" % i, upsample)
            
        elif layers['type'] == 'route':
            layers = [int(x) for x in layers['layers'].split(',')]
            filters = sum([output_filters[layer_idx] for layer_idx in layers])
            modules.add_module("route_%d" % i, EmptyLayer())
            
        elif layers['type'] == 'shortcut':
            filters = output_filters[int(layers['from'])]
            modules.add_module("shortcut_%d" % i, EmptyLayer())
            
        elif layers['type'] == 'yolo':
            num_classes = int(layers['classes'])
            img_height = int(hyperparams['height'])
            
            # Extract anchors.
            anchor_idx = [int(x) for x in layers['mask'].split(',')]
            anchors = [int(x) for x in layers['anchors'].split(',')]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idx]
            
            # YOLO detection layer.
            yolo_layer = YOLOLayer(anchors, num_classes, img_height)
            modules.add_module("yolo_%d" % i, yolo_layer)
        
        # Add the module seqnence into module list.
        module_list.append(modules)
        output_filters.append(filters)
    
    return hyperparams, module_list

class Darknet(nn.Module):
    """
    YOLOv3 object detection model.
    """
    
    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()
        self.layers = parse_model_config(config_path)
        self.hyperparams, self.module_list = generate_modules(self.layers)
        
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0])
        self.loss_names = ["x", "y", "w", "h", "conf", "cls", "recall", "precision"]
        
    def forward(self, x, targets=None):
        is_training = targets is not None
        
        self.losses = defaultdict(float)
        output = []
        layer_outputs = []
        
        # Customize route, shortcut and yolo layer.
        for i, (layer, module) in enumerate(zip(self.layers, self.module_list)):
            if layer['type'] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
                
            elif layer['type'] == 'route':
                layer_idx = [int(x) for x in layer['layers'].split(',')]
                x = torch.cat([layer_outputs[i] for i in layer_idx], 1)
                
            elif layer['type'] == 'shortcut':  
                layer_idx = int(layer["from"])
                x = layer_outputs[-1] + layer_outputs[layer_idx]
                
            elif layer['type'] == 'yolo':
                # Training.
                if is_training:
                    x, *losses = module[0](x, targets)
                    for name, loss in zip(self.loss_names, losses):
                        self.losses[name] += loss
                # Testing.
                else:
                    x = module(x)
                output.append(x)
                
            # Add to the list whatever it is.    
            layer_outputs.append(x)
        
        self.losses['recall'] /= 3
        self.losses['precision'] /= 3
        return sum(output) if is_training else torch.cat(output, 1)
        
    
    def load_weights(self, weights_path):
        """
        Load pre-trained weights.
        """
        
        # Load weights from the file.
        fp = open(weights_path, 'rb')
        header = np.fromfile(fp, dtype=np.int32, count=5)   # First five are header values.
        self.header_info = header   # Save it for writing.
        self.seen = header[3]
        weights = np.fromfile(fp, dtype=np.float32)
        fp.close()
        
        # Parsing the weights from layers.
        ptr = 0
        for i, (layers, module) in enumerate(zip(self.layers, self.module_list)):
            if layers['type'] == 'convolutional':
                conv_layer = module[0]
                if layers['batch_normalize']:       
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                            
                    # Bias
                    num_b = bn_layer.bias.numel()  # Number of biases
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load convolutional layer bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                    
                # Load convolutional layer weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w
        
class EmptyLayer(nn.Module):
    """
    Placeholder for 'route' and 'shortcut' layers.
    """
    def __init__(self):
        super(EmptyLayer, self).__init__()
        
        
class YOLOLayer(nn.Module):
    """
    YOLO detection layer.
    """      
    def __init__(self, anchors, num_classes, img_dim):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.image_dim = img_dim
        
        self.bbox_attrs = 5 + num_classes
        self.ignore_thres = 0.5
        self.lambda_coord = 1

        self.mse_loss = nn.MSELoss(size_average=True)   # Coordinate loss
        self.bce_loss = nn.BCELoss(size_average=True)   # Confidence loss
        self.ce_loss = nn.CrossEntropyLoss()            # Class loss
    
    def forward(self, x, targets=None):
        
        
        pass
        
        
        
        